"""Measure speed of each local model on this machine (streaming, so we see time-to-first-token).

Run:  .venv/bin/python bench.py      ->  results/local-bench.json
"""
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lmstudio")
OUT = Path(__file__).resolve().parent / "results" / "local-bench.json"
PROMPT = ("You are helping a student. Explain, in about 200 words, why a hash table gives average O(1) "
          "lookups and what happens to performance as the load factor grows. Use plain language.")
CASES = [("google/gemma-4-12b", "none"), ("qwen/qwen3.8-27b", "none"), ("qwen/qwen3.8-27b", "high")]
RUNS = 3


def one(model, effort, max_tokens=400):
    start, first, chunks = time.time(), None, 0
    usage = None
    stream = client.chat.completions.create(model=model, reasoning_effort=effort, max_tokens=max_tokens,
                                            temperature=0.3, stream=True, stream_options={"include_usage": True},
                                            messages=[{"role": "user", "content": PROMPT}])
    for ev in stream:
        if ev.usage:
            usage = ev.usage
        if ev.choices:
            d = ev.choices[0].delta
            if (d.content or (d.model_extra or {}).get("reasoning_content")) and first is None:
                first = time.time()
    end = time.time()
    out = usage.completion_tokens if usage else 0
    return {"ttft_s": round(first - start, 3), "total_s": round(end - start, 2), "completion_tokens": out,
            "decode_tok_per_s": round((out - 1) / (end - first), 1) if out > 1 else None}


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


results = {"machine": {"chip": chip(), "memory_gb": 128, "os": platform.platform()},
           "engine": "LM Studio 0.4.28, MLX, 4-bit weights", "prompt": PROMPT, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "models": []}
for model, effort in CASES:
    one(model, effort, max_tokens=20)  # warm-up
    runs = [one(model, effort) for _ in range(RUNS)]
    med = sorted(runs, key=lambda r: r["decode_tok_per_s"] or 0)[RUNS // 2]
    results["models"].append({"model": model, "reasoning": effort, "runs": runs, "median": med})
    print(model, effort, med)

# Throughput when a group orchestrator fans out to 4 workers at once
def timed(i):
    return one("google/gemma-4-12b", "none", max_tokens=300)
t = time.time()
with ThreadPoolExecutor(4) as pool:
    par = list(pool.map(timed, range(4)))
wall = time.time() - t
single = one("google/gemma-4-12b", "none", max_tokens=300)   # the same request alone, for comparison
results["parallel_12b_x4"] = {"wall_s": round(wall, 2), "total_tokens": sum(r["completion_tokens"] for r in par),
                              "aggregate_tok_per_s": round(sum(r["completion_tokens"] for r in par) / wall, 1),
                              "single_request_s": round(single["total_s"], 2) if "total_s" in single else None,
                              "single_tok_per_s": single.get("decode_tok_per_s")}
print("parallel", results["parallel_12b_x4"])

ps = subprocess.run([str(Path.home() / ".lmstudio/bin/lms"), "ps"], capture_output=True, text=True).stdout
results["lms_ps"] = ps
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print("wrote", OUT)
