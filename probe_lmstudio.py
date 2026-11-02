"""Probe what the local LM Studio server can do, and record it.

Run:  .venv/bin/python probe_lmstudio.py
Writes results/lmstudio-capabilities.json
"""
import base64, concurrent.futures as cf, io, json, pathlib, time

from openai import OpenAI
from PIL import Image, ImageDraw

BASE = "http://localhost:1234/v1"
client = OpenAI(base_url=BASE, api_key="lmstudio")
OUT = pathlib.Path(__file__).resolve().parent / "results" / "lmstudio-capabilities.json"
MODELS = ["google/gemma-4-12b", "qwen/qwen3.8-27b"]
results = {"server": BASE, "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checks": {}}


def record(name, ok, **detail):
    results["checks"][name] = {"ok": ok, **detail}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(detail)[:300])


def try_check(name, fn):
    try:
        fn()
    except Exception as e:  # keep probing even if one feature is missing
        record(name, False, error=f"{type(e).__name__}: {e}"[:400])


def tools_check(model):
    def run():
        tools = [{"type": "function", "function": {
            "name": "resolve_date",
            "description": "Convert a relative date phrase to an ISO date.",
            "parameters": {"type": "object", "properties": {
                "phrase": {"type": "string"}, "reference": {"type": "string"}},
                "required": ["phrase", "reference"]}}}]
        r = client.chat.completions.create(model=model, tools=tools, messages=[
            {"role": "user", "content": "The email was received 2026-10-30. It says the essay is due 'next Friday'. Call the tool to resolve the date."}],
            reasoning_effort="none" if "qwen" in model else None)
        calls = r.choices[0].message.tool_calls or []
        record(f"tool_call:{model}", bool(calls),
               call=calls[0].function.model_dump() if calls else r.choices[0].message.content)
    return run


def schema_check(model):
    def run():
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"category": {"type": "string", "enum": ["deadline", "event", "admin", "social", "promo"]},
                                 "action_required": {"type": "boolean"}},
                  "required": ["category", "action_required"]}
        r = client.chat.completions.create(model=model, messages=[
            {"role": "user", "content": "Classify: 'Reminder: Problem Set 5 is due Friday at 5pm.'"}],
            response_format={"type": "json_schema", "json_schema": {"name": "label", "strict": True, "schema": schema}})
        data = json.loads(r.choices[0].message.content)
        record(f"json_schema:{model}", set(data) == {"category", "action_required"}, output=data)
    return run


def reasoning_check():
    model = "qwen/qwen3.8-27b"
    out = {}
    for effort in ["none", "low", "high"]:
        t = time.time()
        r = client.chat.completions.create(model=model, reasoning_effort=effort, messages=[
            {"role": "user", "content": "What is 17 * 23? Answer with just the number."}])
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content") or ""
        out[effort] = {"answer": msg.content.strip(), "reasoning_chars": len(reasoning),
                       "completion_tokens": r.usage.completion_tokens, "seconds": round(time.time() - t, 2)}
    record("reasoning_effort:qwen", out["none"]["reasoning_chars"] == 0 and out["high"]["reasoning_chars"] > 0, efforts=out)


def vision_check(model):
    def run():
        img = Image.new("RGB", (360, 120), "white")
        ImageDraw.Draw(img).text((20, 45), "TOTAL  $42.17", fill="black", font_size=36)
        buf = io.BytesIO(); img.save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
            {"type": "text", "text": "What total amount is printed? Reply with just the number."},
            {"type": "image_url", "image_url": {"url": uri}}]}],
            reasoning_effort=None)
        ans = r.choices[0].message.content.strip()
        record(f"vision:{model}", "42.17" in ans, answer=ans)
    return run


def parallel_check(model="google/gemma-4-12b", n=4):
    def one(i):
        t = time.time()
        r = client.chat.completions.create(model=model, reasoning_effort="none", max_tokens=200, messages=[
            {"role": "user", "content": f"Write 120 words about hash tables (variant {i})."}])
        return time.time() - t, r.usage.completion_tokens
    t0 = time.time(); one(0); single = time.time() - t0
    t0 = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    toks = sum(r[1] for r in res)
    record("parallel_requests", wall < single * n * 0.8, model=model, n=n,
           single_request_s=round(single, 2), parallel_wall_s=round(wall, 2),
           aggregate_tokens_per_s=round(toks / wall, 1))


def embeddings_check():
    r = client.embeddings.create(model="text-embedding-nomic-embed-text-v1.5", input=["load factor", "hash collision"])
    record("embeddings", len(r.data) == 2, dims=len(r.data[0].embedding))


for m in MODELS:
    try_check(f"tool_call:{m}", tools_check(m))
    try_check(f"json_schema:{m}", schema_check(m))
    try_check(f"vision:{m}", vision_check(m))
try_check("reasoning_effort:qwen", reasoning_check)
try_check("parallel_requests", parallel_check)
try_check("embeddings", embeddings_check)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print("wrote", OUT)
