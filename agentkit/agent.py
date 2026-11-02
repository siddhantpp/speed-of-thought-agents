"""An Agent is a model + an Agent Card (its one job) + the few tools it may use."""
import base64
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .llm import TIER_MODELS, chat

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
FROM_CARD = object()   # sentinel: use the card's own output schema


@dataclass
class AgentCard:
    name: str
    tier: str                      # "worker" | "group" | "master"
    job: str                       # one verb phrase: what this agent is for
    instructions: str              # short, specific, with definitions and examples
    tools: list = field(default_factory=list)       # tool names from the Toolbox
    output_schema: dict | None = None                # JSON Schema for the final answer
    reasoning: str | None = "none"                   # thinking budget
    temperature: float = 0.2
    max_tool_turns: int = 4
    max_tokens: int = 4096                           # thinking tokens count against this
    model: str | None = None                         # override the tier's default model
    success_check: str = ""                          # how we know it worked (documentation)

    @property
    def resolved_model(self):
        return self.model or TIER_MODELS[self.tier]

    @classmethod
    def load(cls, name):
        return cls(**json.loads((AGENTS_DIR / f"{name}.json").read_text()))


def _loads(text):
    """json.loads, forgiving two small-model slips: raw control characters and a broken \\u escape."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return json.loads(re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", cleaned))


class Agent:
    def __init__(self, card: AgentCard, toolbox=None, tracer=None):
        self.card, self.toolbox, self.tracer = card, toolbox, tracer
        self._local, self._last_tool_log = threading.local(), []

    @property
    def last_tool_log(self):
        """Tool calls of this agent's last run, in order: [{"tool", "args", "result"}, ...].

        One Agent can run in several threads at once (Group.fan_out), so each thread sees its own
        last run; a thread that never ran this agent sees the run that started most recently.
        """
        return getattr(self._local, "tool_log", self._last_tool_log)

    def system_prompt(self):
        c = self.card
        return f"You are {c.name}, a {c.tier} agent.\nYour one job: {c.job}\n\n{c.instructions}"

    def _ask(self, messages, label, **kw):
        c = self.card
        kw.setdefault("reasoning", self._reasoning)
        kw.setdefault("max_tokens", self._max_tokens)
        reply = chat(c.resolved_model, messages, temperature=c.temperature, **kw)
        if self.tracer:
            self.tracer.llm(agent=c.name, tier=c.tier, model=c.resolved_model, label=label, reply=reply)
        return reply

    def run(self, task: str, images: list[bytes] | None = None, label: str = "", schema=FROM_CARD,
            reasoning=FROM_CARD, max_tokens=None):
        """Run the agent on one task. Returns parsed JSON if there is a schema, else text.

        `schema` and `reasoning` override the card for one call (schema=None = free text),
        which lets one master agent plan, judge and write with different outputs and
        thinking budgets.
        """
        c = self.card
        schema = c.output_schema if schema is FROM_CARD else schema
        # Per-tier override for experiments, e.g. MASTER_REASONING=high (explicit per-call values win).
        tier_default = os.environ.get(f"{c.tier.upper()}_REASONING") or c.reasoning
        self._reasoning = tier_default if reasoning is FROM_CARD else reasoning
        self._max_tokens = max_tokens or c.max_tokens
        user = task
        if images:
            user = [{"type": "text", "text": task}] + [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
                for b in images]
        messages = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": user}]
        specs = self.toolbox.specs(c.tools) if (self.toolbox and c.tools) else None
        # Constrained decoding forces JSON from the first token, which leaves no room to think.
        # So an agent that should think answers in free text first, then formats (step 2).
        thinks = bool(schema) and self._reasoning not in (None, "none")

        # 1) Tool loop: let the model call tools until it stops asking for them.
        reply, tool_log = None, []
        self._local.tool_log = self._last_tool_log = tool_log   # fills in as the tools run
        for _ in range(c.max_tool_turns if specs else 1):
            # When a JSON formatting turn follows, prose here is wasted tokens: cap it.
            reply = self._ask(messages, label, tools=specs, schema=None if (specs or thinks) else schema,
                              **({"max_tokens": 400} if specs and schema and not thinks else {}))
            if not reply.tool_calls:
                break
            messages.append(reply.raw_message.model_dump(exclude_none=True))
            for call in reply.tool_calls:
                result = self.toolbox.call(call.function.name, call.function.arguments,
                                           tracer=self.tracer, agent=c.name, label=label)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                tool_log.append({"tool": call.function.name, "args": call.function.arguments, "result": result})
        if not schema:
            return reply.content

        # 2) Structured answer. Constrained decoding doesn't mix with tool calls or with
        #    thinking, so those agents get one extra "format your answer" turn, thinking off.
        if specs or thinks:
            messages.append({"role": "assistant", "content": reply.content or "(tools finished)"})
            ask = "Now return your final answer as JSON matching the schema."
            if tool_log:  # small models often fail to carry tool results into the answer, so restate them
                ask += " Use these exact tool results wherever a field asks for them:\n" + "\n".join(
                    f"- {t['tool']}({t['args']}) returned {json.dumps(t['result'])}" for t in tool_log)
            messages.append({"role": "user", "content": ask})
            reply = self._ask(messages, label + ":format", schema=schema, reasoning="none")
        for attempt in range(2):
            try:
                return _loads(reply.content)
            except json.JSONDecodeError as err:
                if attempt:
                    raise
                ask = f"That was not valid JSON ({err}). Return only the JSON object."
                if reply.finish_reason == "length":   # cut off by max_tokens, not malformed
                    ask = (f"That was not valid JSON ({err}): you hit the length limit and the answer was cut off. "
                           "Answer more briefly: return only the JSON object, with shorter text in every field.")
                messages += [{"role": "assistant", "content": reply.content},
                             {"role": "user", "content": ask}]
                reply = self._ask(messages, label + ":retry", schema=schema, reasoning="none")
