"""Sample LM Studio's model processes' physical memory (macOS `footprint`) until a stop file appears.

Run:  .venv/bin/python experiments/mem_sampler.py <label> <stop_file>   ->  results/memory-<label>.json
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

label, stop = sys.argv[1], Path(sys.argv[2])
OUT = Path(__file__).resolve().parent.parent / "results" / f"memory-{label}.json"


def model_pids():
    ps = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    return [int(line.split()[0]) for line in ps.splitlines() if ".lmstudio/.internal/utils/node" in line]


def footprint_gb(pid):
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint:\s*([\d.]+)\s*(MB|GB)", out)
    if not m:
        return 0.0
    return float(m.group(1)) / (1024 if m.group(2) == "MB" else 1)


samples, peak = [], 0.0
start = time.time()
while not stop.exists():
    per = {pid: round(footprint_gb(pid), 2) for pid in model_pids()}
    total = round(sum(per.values()), 2)
    peak = max(peak, total)
    samples.append({"t": round(time.time() - start, 1), "total_gb": total})
    time.sleep(2)
lms = subprocess.run([str(Path.home() / ".lmstudio/bin/lms"), "ps"], capture_output=True, text=True).stdout
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"label": label, "peak_gb": peak, "samples": samples, "lms_ps_at_end": lms,
                           "method": "sum of macOS `footprint` of LM Studio model processes, every 2 s"}, indent=2))
print(f"{label}: peak {peak} GB over {len(samples)} samples")
