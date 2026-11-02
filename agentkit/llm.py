"""One function, `chat`, over LM Studio's OpenAI-compatible server.

Because it speaks the OpenAI wire format, the same code runs against any local
server (LM Studio, Ollama, vLLM, llama.cpp) by changing LMSTUDIO_BASE_URL.

The same pipelines can also run on a small cloud model (providers.py):
AGENTKIT_PROVIDER=openai|anthropic SMALL_MODEL=<model id> puts that model in every role.
Embeddings (`embed`) stay on the local model in LM Studio either way.
"""
import os
import re
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("LMSTUDIO_API_KEY", "lmstudio"))

PROVIDER = os.environ.get("AGENTKIT_PROVIDER", "lmstudio").strip().lower() or "lmstudio"
if PROVIDER not in ("lmstudio", "openai", "anthropic"):
    raise ValueError(f"AGENTKIT_PROVIDER={PROVIDER!r}: use lmstudio, openai or anthropic")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "").strip()

# Which model plays which role (~23 GB of weights, peak 32–34 GB in our runs of all seven workflows, fits a 64 GB laptop).
# Override with env vars, e.g. MASTER_MODEL=google/gemma-4-12b on a smaller machine.
TIER_MODELS = {
    "worker": os.environ.get("WORKER_MODEL", "google/gemma-4-12b"),
    "group": os.environ.get("GROUP_MODEL", "google/gemma-4-12b"),
    "master": os.environ.get("MASTER_MODEL", "qwen/qwen3.8-27b"),
}
if SMALL_MODEL:          # one model in every role
    TIER_MODELS = {tier: SMALL_MODEL for tier in TIER_MODELS}
elif PROVIDER != "lmstudio" and not all(os.environ.get(f"{t.upper()}_MODEL") for t in TIER_MODELS):
    raise ValueError(f"AGENTKIT_PROVIDER={PROVIDER} needs SMALL_MODEL=<model id> (or WORKER_, GROUP_ and MASTER_MODEL)")


def run_label():
    """The side's name in trace file names: "local" for LM Studio, else AGENTKIT_RUN_LABEL or the model id."""
    if os.environ.get("AGENTKIT_RUN_LABEL"):
        return re.sub(r"[^A-Za-z0-9_.]+", "_", os.environ["AGENTKIT_RUN_LABEL"])
    if PROVIDER == "lmstudio":
        return "local"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", SMALL_MODEL or TIER_MODELS["worker"])


@dataclass
class Reply:
    content: str
    tool_calls: list = field(default_factory=list)
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    raw_message: object = None
    finish_reason: str = ""      # "stop", "tool_calls", or "length" when max_tokens cut the answer off
    provider: str = "lmstudio"
    usage: dict = field(default_factory=dict)   # cloud: billed tokens by kind (providers.py)
    meta: dict = field(default_factory=dict)    # cloud: what was sent, response model, stop reason


def chat(model, messages, *, tools=None, schema=None, reasoning="none",
         max_tokens=4096, temperature=0.2) -> Reply:
    """Send one request. `schema` turns on constrained JSON output."""
    if PROVIDER != "lmstudio":
        from . import providers
        return providers.chat(PROVIDER, model, messages, tools=tools, schema=schema, reasoning=reasoning,
                              max_tokens=max_tokens, temperature=temperature)
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if reasoning:
        kwargs["reasoning_effort"] = reasoning      # "none" | "low" | "medium" | "high"
    if tools:
        kwargs["tools"] = tools
    if schema:
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "output", "strict": True, "schema": schema}}
    start = time.time()
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    thinking = (msg.model_extra or {}).get("reasoning_content") or ""
    content = msg.content or ""
    # LM Studio 0.4.28: with thinking on, schema-constrained JSON lands in
    # reasoning_content and content comes back empty. Recover it.
    if schema and not content.strip() and thinking.strip().startswith(("{", "[")):
        content, thinking = thinking, ""
    usage = resp.usage
    return Reply(content=content, tool_calls=list(msg.tool_calls or []), reasoning=thinking,
                 prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                 completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                 seconds=time.time() - start, raw_message=msg,
                 finish_reason=resp.choices[0].finish_reason or "")


EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
EMBED_STATS = {"requests": 0, "texts": 0}     # every setup embeds locally; the trace records how much
_embed_lock = threading.Lock()


def embed(texts: list[str]) -> list[list[float]]:
    """Embedding vectors from the local embedding model (used for de-duplication)."""
    out = []
    for i in range(0, len(texts), 64):   # batches keep each request under the model's 2,048-token context
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 64])
        out += [d.embedding for d in resp.data]
        with _embed_lock:
            EMBED_STATS["requests"] += 1
            EMBED_STATS["texts"] += len(texts[i:i + 64])
    return out


def setup():
    """Which provider and models a run used, for its trace."""
    return {"provider": PROVIDER, "tier_models": dict(TIER_MODELS),
            "base_url": BASE_URL if PROVIDER == "lmstudio" else
            {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com"}[PROVIDER],
            "embeddings": {"model": EMBED_MODEL, "where": f"local, LM Studio ({BASE_URL})",
                           "note": "every setup embeds locally: a 137M-parameter helper for de-duplication, "
                                   "not a reasoning model", **EMBED_STATS}}
