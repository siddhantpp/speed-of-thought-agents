"""Cloud backends for `chat()`: the same decomposed pipelines on a small cloud model instead of LM Studio.

Pick one with environment variables (llm.py reads them; LM Studio stays the default):

    AGENTKIT_PROVIDER=openai     SMALL_MODEL=<model id>   OpenAI Chat Completions, https://api.openai.com/v1
    AGENTKIT_PROVIDER=anthropic  SMALL_MODEL=<model id>   Anthropic Messages API, the official `anthropic` SDK

SMALL_MODEL puts the same model in every role (worker, group lead, master). Embeddings always stay local:
`embed()` keeps using the nomic embedding model in LM Studio, a 137M-parameter helper, not a reasoning model.

Keys come only from SOT_OPENAI_API_KEY / SOT_ANTHROPIC_API_KEY, or from KEY=VALUE lines in
~/.config/speed-of-thought/keys.env. They are passed straight to the SDK client, never printed, logged or written,
and the generic OPENAI_API_KEY / ANTHROPIC_API_KEY are never read or set.

Both backends take agentkit's OpenAI-style messages (what Agent.run builds and appends to) and return the same
`Reply` as LM Studio, plus `usage`: the exact billed token counts (uncached input, cache reads, cache writes,
output, of which reasoning) and `meta`: what was sent and what came back, so a run's cost can be computed from
the provider's prices.

How agentkit's inputs map (each choice is recorded per call in Reply.meta["sent"]):

  reasoning "none"/"low"/"medium"/"high"
      OpenAI     reasoning_effort with the same value; None sends nothing (the model's default).
      Anthropic  "none" -> thinking {"type": "disabled"}; the others -> thinking {"type": "adaptive"} with
                 output_config.effort set to that level; None sends neither (the model's default).
  temperature
      OpenAI     sent only with reasoning "none": reasoning models reject sampling parameters while reasoning.
                 If the API still rejects it, the call is re-sent without it once and the model is remembered.
      Anthropic  never sent: Claude Haiku 5.5 rejects non-default sampling values, and API version 2026-10-21
                 (anthropic SDK 2.x) rejects temperature, top_p and top_k on every model.
  max_tokens     OpenAI max_completion_tokens; Anthropic max_tokens. Reasoning counts against it on both, as in LM Studio.
  schema         OpenAI response_format json_schema, strict (made strict-compatible: every property required,
                 optional ones nullable and their nulls removed again; unsupported keywords moved into the
                 description). Anthropic output_config.format json_schema through the SDK's transform_schema.
  tools          OpenAI as they are (GPT-6 models take function calls in Chat Completions only with
                 reasoning_effort "none", which is what every tool-using Agent Card asks for; anything else is
                 refused here rather than sent). Anthropic {name, description, input_schema}; tool results go back as
                 tool_result blocks in one user message. A request that carries earlier tool calls but no tools
                 (Agent.run's "format" turn) re-sends the known tool definitions with tool choice "none".
  images         data: URLs in image_url parts. OpenAI as they are; Anthropic base64 image blocks, same order.
  caching        OpenAI: the API's own (implicit) prompt caching. Anthropic: one cache breakpoint on the system
                 prompt (cards' prompts are shared by every call of that agent); AGENTKIT_PROMPT_CACHE=off drops it.

AGENTKIT_USAGE_LOG=<file> also appends every call's billed usage to that JSON-lines file as it happens, so the
spend of a run that crashes before saving its trace is still known.
"""
import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import Reply

KEY_FILE = Path.home() / ".config" / "speed-of-thought" / "keys.env"
KEY_VARS = {"openai": "SOT_OPENAI_API_KEY", "anthropic": "SOT_ANTHROPIC_API_KEY"}
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
MAX_RETRIES = 5                       # the SDKs retry 408/409/429/5xx and connection errors with backoff
_clients, _keys, _lock = {}, {}, threading.Lock()
_KNOWN_TOOLS = {}                     # tool name -> OpenAI-style spec, from every request that passed tools
_NO_TEMPERATURE = set()               # OpenAI models that rejected temperature once
# OpenAI: on these models Chat Completions accepts function calling only with reasoning_effort "none"
# (developers.openai.com/api/docs/models/gpt-6-luna); reasoning with tools needs the Responses API.
TOOLS_ONLY_WITHOUT_REASONING = ("gpt-6",)


# ---------------------------------------------------------------- keys and clients
def _read_key_file(path=None):
    """KEY=VALUE lines (optional `export`, quotes and # comments). Values are returned, never printed."""
    path = Path(path or KEY_FILE)
    out = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[name] = value
    return out


def api_key(provider):
    """The provider's key from SOT_<PROVIDER>_API_KEY, else from the key file. Raises (without the value) if neither."""
    var = KEY_VARS[provider]
    with _lock:
        if provider not in _keys:
            value = os.environ.get(var) or _read_key_file().get(var)
            if not value:
                raise RuntimeError(f"no {provider} key: set {var} in the environment or add a {var}=... line to "
                                   f"{KEY_FILE}")
            _keys[provider] = value
        return _keys[provider]


def secret_values():
    """The keys loaded so far (for scrubbing anything about to be written). Never print these."""
    return [v for v in _keys.values() if v]


def get_client(provider):
    with _lock:
        if provider in _clients:
            return _clients[provider]
    key = api_key(provider)
    if provider == "openai":
        from openai import OpenAI
        c = OpenAI(api_key=key, base_url=OPENAI_BASE_URL, max_retries=MAX_RETRIES)
    elif provider == "anthropic":
        import anthropic
        # An explicit api_key means the SDK consults no environment variable or profile for credentials.
        c = anthropic.Anthropic(api_key=key, base_url=ANTHROPIC_BASE_URL, max_retries=MAX_RETRIES)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    with _lock:
        return _clients.setdefault(provider, c)


# ---------------------------------------------------------------- shared helpers
@dataclass
class _Function:
    name: str
    arguments: str                     # JSON text, as the OpenAI SDK gives it


@dataclass
class ToolCall:
    """The shape Agent.run reads from a tool call: .id, .function.name, .function.arguments."""
    id: str
    function: _Function
    type: str = "function"


@dataclass
class RawMessage:
    """What Agent.run appends to the conversation: model_dump() gives an OpenAI-style assistant message.
    `blocks` keeps the provider's own content (thinking blocks, signatures) so the next turn can send it back
    unchanged."""
    content: str
    tool_calls: list = field(default_factory=list)
    blocks: list | None = None

    def model_dump(self, exclude_none=False, **_):
        d = {"role": "assistant", "content": self.content or None,
             "tool_calls": [{"id": t.id, "type": "function",
                             "function": {"name": t.function.name, "arguments": t.function.arguments}}
                            for t in self.tool_calls] or None,
             "_anthropic_content": self.blocks}
        return {k: v for k, v in d.items() if v is not None} if exclude_none else d


def _history_tool_names(messages):
    return {tc["function"]["name"] for m in messages if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])}


def _remember_tools(tools):
    for t in tools or []:
        _KNOWN_TOOLS[t["function"]["name"]] = t


def _tools_for_history(messages):
    """Definitions for the tools earlier turns called (both APIs expect them when a conversation has tool calls)."""
    names = _history_tool_names(messages)
    return [_KNOWN_TOOLS[n] for n in sorted(names) if n in _KNOWN_TOOLS]


_ASIS_STRICT = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf",
                "$ref", "$defs", "description", "pattern", "format", "multipleOf", "maximum", "exclusiveMaximum",
                "minimum", "exclusiveMinimum", "minItems", "maxItems", "title"}
_STRICT_FORMATS = {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}


def openai_strict_schema(schema):
    """A copy of `schema` that OpenAI's strict json_schema mode accepts.

    Strict mode needs every property in `required` and additionalProperties false on every object; an optional
    property becomes required but nullable (strip_added_nulls removes those nulls from the answer again).
    Keywords strict mode does not support (minLength, maxLength, ...) move into the description, as the
    anthropic SDK's transform_schema does, so the model still reads them.
    """
    s = copy.deepcopy(schema)

    def walk(node, nullable=False):
        if not isinstance(node, dict):
            return node
        out, extra = {}, {}
        for k, v in node.items():
            if k == "format" and v not in _STRICT_FORMATS:
                extra[k] = v
            elif k in _ASIS_STRICT:
                out[k] = v
            else:
                extra[k] = v
        if "properties" in out:
            req = set(out.get("required") or [])
            out["properties"] = {k: walk(v, nullable=k not in req) for k, v in out["properties"].items()}
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        if "items" in out:
            out["items"] = walk(out["items"])
        if "anyOf" in out:
            out["anyOf"] = [walk(v) for v in out["anyOf"]]
        if "$defs" in out:
            out["$defs"] = {k: walk(v) for k, v in out["$defs"].items()}
        if extra:
            note = "{" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + "}"
            out["description"] = (out["description"] + "\n\n" if out.get("description") else "") + note
        if nullable:
            t = out.get("type")
            if isinstance(t, str):
                out["type"] = [t, "null"]
                if "enum" in out:
                    out["enum"] = list(out["enum"]) + [None]
            elif isinstance(t, list) and "null" not in t:
                out["type"] = t + ["null"]
            elif "anyOf" in out:
                out["anyOf"] = out["anyOf"] + [{"type": "null"}]
        return out

    return walk(s)


def strip_added_nulls(value, schema):
    """Remove the nulls openai_strict_schema allowed for properties the original schema did not require."""
    if isinstance(value, dict) and isinstance(schema, dict) and "properties" in schema:
        req = set(schema.get("required") or [])
        return {k: strip_added_nulls(v, schema["properties"].get(k, {})) for k, v in value.items()
                if not (v is None and k not in req)}
    if isinstance(value, list) and isinstance(schema, dict) and isinstance(schema.get("items"), dict):
        return [strip_added_nulls(v, schema["items"]) for v in value]
    return value


# ---------------------------------------------------------------- OpenAI (Chat Completions)
def _openai_message(m):
    """Only the fields Chat Completions accepts (Agent.run appends whole SDK message dumps)."""
    role = m.get("role")
    if role == "assistant":
        out = {"role": "assistant", "content": m.get("content")}
        if m.get("tool_calls"):
            out["tool_calls"] = [{"id": tc["id"], "type": "function",
                                  "function": {"name": tc["function"]["name"],
                                               "arguments": tc["function"]["arguments"]}}
                                 for tc in m["tool_calls"]]
        return out
    if role == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content", "")}
    return {"role": role, "content": m.get("content")}


def openai_request(model, messages, *, tools=None, schema=None, reasoning="none", max_tokens=4096,
                   temperature=0.2):
    """The Chat Completions kwargs for one agentkit call, and a summary of what was sent."""
    msgs = [_openai_message(m) for m in messages]
    kwargs = dict(model=model, messages=msgs, max_completion_tokens=max_tokens)
    sent = {"max_completion_tokens": max_tokens}
    if reasoning:
        kwargs["reasoning_effort"] = reasoning
    sent["reasoning_effort"] = reasoning or "(model default)"
    if temperature is not None and reasoning == "none" and model not in _NO_TEMPERATURE:
        kwargs["temperature"] = temperature
        sent["temperature"] = temperature
    else:
        sent["temperature"] = "not sent" + (" (reasoning on)" if reasoning != "none" else "")
    if tools:
        _remember_tools(tools)
        kwargs["tools"] = tools
    elif _history_tool_names(msgs):
        known = _tools_for_history(msgs)
        if known:
            kwargs["tools"], kwargs["tool_choice"] = known, "none"
            sent["tools"] = "earlier tools re-sent, tool_choice none"
    if kwargs.get("tools") and reasoning != "none" and model.startswith(TOOLS_ONLY_WITHOUT_REASONING):
        raise ValueError(f"{model}: Chat Completions takes function calls only with reasoning 'none' "
                         f"(this call asked for {reasoning!r} with tools); not sent")
    if schema:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "output", "strict": True, "schema": openai_strict_schema(schema)}}
        sent["schema"] = "json_schema strict"
    return kwargs, sent


def openai_chat(model, messages, *, tools=None, schema=None, reasoning="none", max_tokens=4096,
                temperature=0.2, client=None) -> Reply:
    import openai
    c = client or get_client("openai")
    kwargs, sent = openai_request(model, messages, tools=tools, schema=schema, reasoning=reasoning,
                                  max_tokens=max_tokens, temperature=temperature)
    start = time.time()
    try:
        resp = c.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        if "temperature" not in kwargs or "temperature" not in str(e).lower():
            raise
        _NO_TEMPERATURE.add(model)          # the model rejects sampling parameters: send none from now on
        kwargs.pop("temperature")
        sent["temperature"] = "rejected by the API, re-sent without it"
        resp = c.chat.completions.create(**kwargs)
    seconds = time.time() - start
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    if schema and content.strip():
        try:
            content = json.dumps(strip_added_nulls(json.loads(content), schema), ensure_ascii=False)
        except json.JSONDecodeError:
            pass                            # Agent.run reports it and asks again
    u = resp.usage
    pt = getattr(u, "prompt_tokens", 0) or 0
    ptd = getattr(u, "prompt_tokens_details", None)
    ctd = getattr(u, "completion_tokens_details", None)
    cached = getattr(ptd, "cached_tokens", 0) or 0          # billed cache reads (excludes cache writes)
    written = getattr(ptd, "cache_write_tokens", 0) or 0     # tokens written to the cache
    usage = {"prompt_total": pt, "input_uncached": max(pt - cached - written, 0), "cache_read": cached,
             "cache_write": written, "cache_write_1h": 0, "output": getattr(u, "completion_tokens", 0) or 0,
             "reasoning": getattr(ctd, "reasoning_tokens", 0) or 0, "images": getattr(ptd, "image_tokens", 0) or 0}
    finish = choice.finish_reason or ""
    if getattr(msg, "refusal", None):
        finish = "refusal"
    meta = {"provider": "openai", "response_model": resp.model, "service_tier": getattr(resp, "service_tier", None),
            "sent": sent, "stop_reason": choice.finish_reason}
    if getattr(msg, "refusal", None):
        meta["refusal"] = str(msg.refusal)[:300]
    return Reply(content=content, tool_calls=list(msg.tool_calls or []), reasoning="",
                 prompt_tokens=pt, completion_tokens=usage["output"], seconds=seconds, raw_message=msg,
                 finish_reason=finish, provider="openai", usage=usage, meta=meta)


# ---------------------------------------------------------------- Anthropic (Messages API)
_STOP = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length",
         "model_context_window_exceeded": "length", "refusal": "refusal", "pause_turn": "pause_turn"}
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _anthropic_part(part):
    """One OpenAI-style content part -> an Anthropic content block."""
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if part.get("type") == "text":
        return {"type": "text", "text": part["text"]}
    if part.get("type") == "image_url":
        url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
        m = re.match(r"data:(?P<mt>[\w/+.-]+);base64,(?P<data>.+)$", url, re.S)
        if m:
            return {"type": "image", "source": {"type": "base64", "media_type": m.group("mt"),
                                                "data": m.group("data")}}
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise ValueError(f"unsupported content part {part.get('type')!r}")


def _blocks(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [_anthropic_part(p) for p in content]


def _anthropic_messages(messages, keep_thinking):
    """OpenAI-style messages -> (system text, Anthropic messages). Consecutive same-role turns are merged, so
    every tool result of one assistant turn lands in a single user message."""
    system, out = [], []

    def add(role, blocks):
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role")
        if role == "system":
            system.append(m["content"] if isinstance(m["content"], str)
                          else "\n".join(p.get("text", "") for p in m["content"]))
        elif role == "user":
            add("user", _blocks(m.get("content")))
        elif role == "tool":
            content = m.get("content", "")
            add("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                          "content": content if isinstance(content, str) else json.dumps(content)}])
        elif role == "assistant":
            if m.get("_anthropic_content") is not None:
                blocks = [dict(b) for b in m["_anthropic_content"]]
            else:
                blocks = _blocks(m.get("content"))
                for tc in m.get("tool_calls") or []:
                    args = tc["function"]["arguments"]
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": json.loads(args) if isinstance(args, str) else args})
            if not keep_thinking:   # allowed with thinking off; with it on, blocks go back unchanged
                blocks = [b for b in blocks if b.get("type") not in ("thinking", "redacted_thinking")]
            blocks = [b for b in blocks if not (b.get("type") == "text" and not b.get("text"))]
            add("assistant", blocks or [{"type": "text", "text": "(empty answer)"}])
        else:
            raise ValueError(f"unsupported role {role!r}")
    return "\n\n".join(s for s in system if s), out


def _anthropic_tool(t):
    f = t["function"]
    return {"name": f["name"], "description": f.get("description", ""),
            "input_schema": f.get("parameters") or {"type": "object", "properties": {}}}


def anthropic_request(model, messages, *, tools=None, schema=None, reasoning="none", max_tokens=4096,
                      temperature=0.2):
    """The Messages API kwargs for one agentkit call, and a summary of what was sent."""
    import anthropic
    thinking_on = reasoning not in (None, "none")
    system, msgs = _anthropic_messages(messages, keep_thinking=thinking_on or reasoning is None)
    kwargs = dict(model=model, max_tokens=max_tokens, messages=msgs)
    sent = {"max_tokens": max_tokens, "temperature": "not sent (rejected by Claude Haiku 5.5 and API 2026-10-21)"}
    cache = os.environ.get("AGENTKIT_PROMPT_CACHE", "system").lower() not in ("off", "0", "none", "")
    if system:
        block = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        kwargs["system"] = [block]
    sent["prompt_cache"] = "system prompt (5-minute)" if (system and cache) else "off"
    output_config = {}
    if reasoning is None:
        sent["thinking"] = "(model default)"
    elif reasoning == "none":
        kwargs["thinking"] = {"type": "disabled"}
        sent["thinking"] = "disabled"
    else:
        effort = reasoning if reasoning in _EFFORTS else "low"     # e.g. OpenAI's "minimal"
        kwargs["thinking"] = {"type": "adaptive"}
        output_config["effort"] = effort
        sent["thinking"], sent["effort"] = "adaptive", effort
    if tools:
        _remember_tools(tools)
        kwargs["tools"] = [_anthropic_tool(t) for t in tools]
    elif any(b.get("type") == "tool_use" for m in msgs if m["role"] == "assistant" for b in m["content"]):
        known = _tools_for_history(messages)
        if known:
            kwargs["tools"] = [_anthropic_tool(t) for t in known]
            kwargs["tool_choice"] = {"type": "none"}
            sent["tools"] = "earlier tools re-sent, tool_choice none"
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        sent["schema"] = "output_config.format json_schema"
    if output_config:
        kwargs["output_config"] = output_config
    return kwargs, sent


def anthropic_chat(model, messages, *, tools=None, schema=None, reasoning="none", max_tokens=4096,
                   temperature=0.2, client=None) -> Reply:
    c = client or get_client("anthropic")
    kwargs, sent = anthropic_request(model, messages, tools=tools, schema=schema, reasoning=reasoning,
                                     max_tokens=max_tokens, temperature=temperature)
    start = time.time()
    resp = c.messages.create(**kwargs)
    seconds = time.time() - start
    texts, thinking, calls, blocks = [], [], [], []
    for b in resp.content:
        blocks.append(b.model_dump(mode="json", exclude_none=True))
        if b.type == "text":
            texts.append(b.text)
        elif b.type == "thinking":
            thinking.append(b.thinking or "")
        elif b.type == "tool_use":
            calls.append(ToolCall(id=b.id, function=_Function(name=b.name, arguments=json.dumps(b.input))))
    u = resp.usage
    cc = getattr(u, "cache_creation", None)
    w1h = getattr(cc, "ephemeral_1h_input_tokens", 0) or 0
    written = getattr(u, "cache_creation_input_tokens", 0) or 0
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    otd = getattr(u, "output_tokens_details", None)
    usage = {"input_uncached": u.input_tokens or 0, "cache_read": read, "cache_write": written - w1h,
             "cache_write_1h": w1h, "output": u.output_tokens or 0,
             "reasoning": getattr(otd, "thinking_tokens", 0) or 0}
    usage["prompt_total"] = usage["input_uncached"] + read + written
    content = "".join(texts)
    meta = {"provider": "anthropic", "response_model": resp.model, "service_tier": getattr(u, "service_tier", None),
            "sent": sent, "stop_reason": resp.stop_reason}
    details = getattr(resp, "stop_details", None)
    if resp.stop_reason == "refusal" and details is not None:
        meta["refusal_category"] = getattr(details, "category", None)
    return Reply(content=content, tool_calls=calls, reasoning="\n".join(t for t in thinking if t),
                 prompt_tokens=usage["prompt_total"], completion_tokens=usage["output"], seconds=seconds,
                 raw_message=RawMessage(content=content, tool_calls=calls, blocks=blocks),
                 finish_reason=_STOP.get(resp.stop_reason or "", resp.stop_reason or ""),
                 provider="anthropic", usage=usage, meta=meta)


BACKENDS = {"openai": openai_chat, "anthropic": anthropic_chat}


def chat(provider, model, messages, **kw) -> Reply:
    reply = BACKENDS[provider](model, messages, **kw)
    log = os.environ.get("AGENTKIT_USAGE_LOG")
    if log:   # token counts and model names only
        line = json.dumps({"t": round(time.time(), 3), "provider": provider, "model": model,
                           "response_model": reply.meta.get("response_model"), "usage": reply.usage})
        with _lock, open(log, "a") as fh:
            fh.write(line + "\n")
    return reply

