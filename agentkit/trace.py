"""Records every model call and tool call so a run can be replayed, costed and charted."""
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

COLORS = {"worker": "\033[38;5;79m", "group": "\033[38;5;141m", "master": "\033[38;5;99m",
          "tool": "\033[38;5;245m", "note": "\033[38;5;214m"}
RESET = "\033[0m"


class Tracer:
    def __init__(self, workflow: str, quiet: bool = False):
        self.workflow, self.quiet = workflow, quiet
        self.t0 = time.time()
        self.events, self._lock = [], threading.Lock()

    def _add(self, event, line, color):
        event["t"] = round(time.time() - self.t0, 3)
        with self._lock:
            self.events.append(event)
            if not self.quiet:
                print(f"{COLORS[color]}[{event['t']:7.1f}s] {line}{RESET}", flush=True)

    def llm(self, agent, tier, model, label, reply):
        event = {"kind": "llm", "agent": agent, "tier": tier, "model": model, "label": label,
                 "seconds": round(reply.seconds, 3), "prompt_tokens": reply.prompt_tokens,
                 "completion_tokens": reply.completion_tokens, "tool_calls": len(reply.tool_calls),
                 "reasoning_chars": len(reply.reasoning)}
        provider = getattr(reply, "provider", "lmstudio")
        if provider != "lmstudio":   # cloud: the billed tokens by kind, so cost = usage x the provider's prices
            meta = getattr(reply, "meta", {}) or {}
            event.update(provider=provider, response_model=meta.get("response_model"),
                         usage=getattr(reply, "usage", {}), finish_reason=reply.finish_reason,
                         service_tier=meta.get("service_tier"), sent=meta.get("sent"),
                         **{k: meta[k] for k in ("refusal", "refusal_category") if k in meta})
        self._add(event,
                  f"{tier:<6} {agent:<22} {label:<18} {reply.prompt_tokens:>6} in {reply.completion_tokens:>5} out "
                  f"{reply.seconds:5.1f}s  {model}", tier)

    def tool(self, agent, name, label, seconds, ok):
        self._add({"kind": "tool", "agent": agent, "tool": name, "label": label,
                   "seconds": round(seconds, 4), "ok": ok},
                  f"tool   {name:<22} {label:<18} called by {agent}" + ("" if ok else "  (error)"), "tool")

    def note(self, text):
        self._add({"kind": "note", "text": text}, text, "note")

    def summary(self):
        by_tier = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                       "model_seconds": 0.0, "max_prompt_tokens": 0})
        for e in self.events:
            if e["kind"] != "llm":
                continue
            s = by_tier[e["tier"]]
            s["calls"] += 1
            s["prompt_tokens"] += e["prompt_tokens"]
            s["completion_tokens"] += e["completion_tokens"]
            s["model_seconds"] = round(s["model_seconds"] + e["seconds"], 2)
            s["max_prompt_tokens"] = max(s["max_prompt_tokens"], e["prompt_tokens"])
            s["model"] = e["model"]
        llm = [e for e in self.events if e["kind"] == "llm"]
        out = {"workflow": self.workflow, "wall_seconds": round(time.time() - self.t0, 2),
               "llm_calls": len(llm), "tool_calls": sum(e["kind"] == "tool" for e in self.events),
               "prompt_tokens": sum(e["prompt_tokens"] for e in llm),
               "completion_tokens": sum(e["completion_tokens"] for e in llm),
               "by_tier": dict(by_tier)}
        billed = [e["usage"] for e in llm if e.get("usage")]
        if billed:   # cloud runs: totals of the billed token kinds (cost needs each call's own, see the events)
            out["usage"] = {k: sum(u.get(k, 0) or 0 for u in billed) for k in sorted({k for u in billed for k in u})}
        return out

    def save(self, directory: Path, extra: dict | None = None):
        """<workflow>-<side>-[<tag>-]<stamp>.json; side is "local" for LM Studio, else the run label (llm.run_label)."""
        from . import llm
        directory.mkdir(parents=True, exist_ok=True)
        tag = os.environ.get("RUN_TAG", "")   # e.g. think, to keep experiment runs apart
        path = directory / f"{self.workflow}-{llm.run_label()}-{tag + '-' if tag else ''}{time.strftime('%Y%m%d-%H%M%S')}.json"
        text = json.dumps({"summary": self.summary(), "setup": llm.setup(), **(extra or {}), "events": self.events},
                          indent=2)
        if llm.PROVIDER != "lmstudio":   # an API key must never reach a file
            from .providers import secret_values
            for secret in secret_values():
                text = text.replace(secret, "[redacted]")
        path.write_text(text)
        return path
