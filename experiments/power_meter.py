"""Meter the laptop's own power draw, without sudo, and compute mean W and Wh between two times.

Read-only sources, sampled every second:

  SMC key PSTR (W): system total power, what the whole Mac draws (SoC, memory, SSD, built-in display, fans, ports),
      whether it comes from the adapter or the battery. This is the headline figure.
  SMC key PDTR (W): DC input power from the adapter. Both are read through IOKit's AppleSMC user client with the
      read-only key calls (kSMCGetKeyInfo, kSMCReadKey) that menu-bar monitors use; nothing is written.
  ioreg -rw0 -c AppleSmartBattery -> PowerTelemetryData (macOS's battery manager): SystemPowerIn (adapter input,
      mW = SystemVoltageIn mV x SystemCurrentIn mA), SystemLoad (system total), BatteryPower (negative while the
      battery discharges) and their accumulators (running sum and count of the gauge's own samples). macOS
      refreshes this block about once a minute (UpdateTime), not every second, so a single reading is a
      snapshot; the accumulators give the exact mean over each refresh interval and cross-check the SMC readings.

Charging and discharging: input power includes any charging power, and under a heavy load the battery can add to
the input (BatteryPower < 0 with the battery full and the adapter connected). So the headline is the system total
(PSTR), the DC input (PDTR) and the battery's share are reported next to it, and every window lists the battery
state seen (ExternalConnected, IsCharging, FullyCharged, CurrentCapacity, BatteryPower). Nothing is subtracted.
All figures are DC-side: the adapter's own conversion loss and an external monitor on its own supply are not in them.

Usage (run from the folder that holds agentkit/; DATA is where your results go, e.g. results/):
  .venv/bin/python experiments/power_meter.py record OUT.json [--stop FILE] [--seconds N] [--label L]
        sample every second until FILE exists or N seconds pass (or Ctrl-C); OUT.json is rewritten every 30 s
  .venv/bin/python experiments/power_meter.py accumulators OUT.jsonl --stop FILE
        keep each ioreg refresh (polled every 10 s), for a trace recorded without the load and battery accumulators
  .venv/bin/python experiments/power_meter.py window TRACE.json T0 T1      mean W and Wh between two epoch times
  .venv/bin/python experiments/power_meter.py check TRACE.json [ACC.jsonl] SMC against ioreg, every refresh interval
  .venv/bin/python experiments/power_meter.py state                         battery, charger and display, now
  .venv/bin/python experiments/power_meter.py runs TRACE.json DATA/run-all-local.json DATA/power [--idle-w W]
        one DATA/power/<trace-stem>.json per run listed in the timing file
  .venv/bin/python experiments/power_meter.py summary --idle IDLE.json --local TRACE.json --acc ACC.jsonl \
        --run-all DATA/run-all-local.json --waiting WAIT.json --out DATA/power-summary.json
        the per-run files plus DATA/power-summary.json (idle baseline, every run, each pass, the waiting draw)

A metered run_all.sh (what produced DATA/power-summary.json):
  .venv/bin/python experiments/power_meter.py record DATA/power/run-all-local-<stamp>.trace.json --stop STOP &
  PASSES=3 bash run_all.sh; touch STOP
"""
import ctypes
import ctypes.util
import json
import os
import re
import signal
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

IOREG_KEYS = ("SystemPowerIn", "SystemVoltageIn", "SystemCurrentIn", "AccumulatedSystemPowerIn",
              "SystemPowerInAccumulatorCount", "BatteryPower", "SystemLoad", "AdapterEfficiencyLoss",
              "AccumulatedSystemLoad", "SystemLoadAccumulatorCount", "AccumulatedBatteryPower",
              "BatteryPowerAccumulatorCount")
SIGNED = {"BatteryPower", "InstantAmperage"}   # printed as unsigned 64-bit: -4095 mW shows as 18446744073709547521
FLAG_KEYS = ("ExternalConnected", "IsCharging", "FullyCharged")
NUM_KEYS = ("CurrentCapacity", "MaxCapacity", "InstantAmperage", "UpdateTime", "Voltage")


# ---------------------------------------------------------------- sources
def signed(key, v):
    return v - (1 << 64) if key in SIGNED and v >= 1 << 63 else v


def ioreg_battery():
    """The battery manager's power telemetry and charging state (read only)."""
    s = subprocess.run(["ioreg", "-rw0", "-c", "AppleSmartBattery"], capture_output=True, text=True, timeout=10).stdout
    out = {}
    m = re.search(r'"PowerTelemetryData" = \{([^}]*)\}', s)
    if m:
        tel = dict(re.findall(r'"(\w+)"=(-?\d+)', m.group(1)))
        out.update({k: signed(k, int(tel[k])) for k in IOREG_KEYS if k in tel})
    for k in FLAG_KEYS:
        f = re.search(rf'"{k}" = (Yes|No)', s)
        if f:
            out[k] = f.group(1) == "Yes"
    for k in NUM_KEYS:
        f = re.search(rf'^\s*"{k}" = (-?\d+)$', s, re.M)
        if f:
            out[k] = signed(k, int(f.group(1)))
    a = re.search(r'"AdapterDetails" = \{[^}]*?"Watts"=(\d+)[^}]*?"Name"="([^"]*)"', s)
    if a:
        out["adapter"] = f"{a.group(2)} ({a.group(1)} W)"
    return out


class SMC:
    """Read-only access to SMC keys through IOKit (kSMCGetKeyInfo, kSMCReadKey)."""

    class _Vers(ctypes.Structure):
        _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8), ("build", ctypes.c_uint8),
                    ("reserved", ctypes.c_uint8), ("release", ctypes.c_uint16)]

    class _PLimit(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16), ("cpu", ctypes.c_uint32),
                    ("gpu", ctypes.c_uint32), ("mem", ctypes.c_uint32)]

    class _KeyInfo(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("type", ctypes.c_uint32), ("attributes", ctypes.c_uint8)]

    class _Data(ctypes.Structure):
        pass

    _Data._fields_ = [("key", ctypes.c_uint32), ("vers", _Vers), ("plimit", _PLimit), ("info", _KeyInfo),
                      ("result", ctypes.c_uint8), ("status", ctypes.c_uint8), ("data8", ctypes.c_uint8),
                      ("data32", ctypes.c_uint32), ("bytes", ctypes.c_uint8 * 32)]

    def __init__(self):
        io = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
        libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("c"))
        io.IOServiceMatching.restype = ctypes.c_void_p
        io.IOServiceMatching.argtypes = [ctypes.c_char_p]
        io.IOServiceGetMatchingService.restype = ctypes.c_uint32
        io.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        io.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        io.IOConnectCallStructMethod.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.io = io
        assert ctypes.sizeof(self._Data) == 80, "unexpected SMC struct layout"
        svc = io.IOServiceGetMatchingService(0, io.IOServiceMatching(b"AppleSMC"))
        self.conn = ctypes.c_uint32()
        if not svc or io.IOServiceOpen(svc, ctypes.c_uint32.in_dll(libc, "mach_task_self_").value, 0,
                                       ctypes.byref(self.conn)):
            raise OSError("cannot open AppleSMC")
        self.info = {}

    def _call(self, inp):
        out, size = self._Data(), ctypes.c_size_t(ctypes.sizeof(self._Data))
        kr = self.io.IOConnectCallStructMethod(self.conn.value, 2, ctypes.byref(inp), ctypes.sizeof(self._Data),
                                               ctypes.byref(out), ctypes.byref(size))
        return out if kr == 0 and out.result == 0 else None

    def read_float(self, key):
        k = struct.unpack(">I", key.encode())[0]
        if key not in self.info:
            inp = self._Data(key=k, data8=9)
            out = self._call(inp)
            self.info[key] = (out.info.size, out.info.type) if out else None
        if not self.info[key]:
            return None
        size, typ = self.info[key]
        inp = self._Data(key=k, data8=5)
        inp.info.size = size
        out = self._call(inp)
        if not out or struct.pack(">I", typ) != b"flt ":
            return None
        return round(struct.unpack("<f", bytes(out.bytes[:4]))[0], 3)


def display_state():
    """Built-in display brightness and the displays that are on (read only; nothing is changed)."""
    out = {}
    s = subprocess.run(["ioreg", "-rw0", "-c", "AppleARMBacklight"], capture_output=True, text=True).stdout
    b = re.search(r'"brightness"=\{"min"=(\d+),"max"=(\d+),"value"=(\d+)\}', s)
    if b:
        lo, hi, v = map(int, b.groups())
        out["builtin_brightness_fraction"] = round((v - lo) / (hi - lo), 3)
    n = re.search(r'"BrightnessMilliNits"=\{"min"=\d+,"value"=(\d+)', s)
    if n:
        out["builtin_brightness_nits"] = round(int(n.group(1)) / 1000, 1)
    sp = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=60).stdout
    disp = []
    for block in re.split(r"\n(?=\s{8}\S[^\n]*:\n)", sp):
        if "Resolution:" in block:
            res = re.search(r"Resolution: ([^\n]+)", block).group(1).strip()
            typ = re.search(r"Display Type: ([^\n]+)", block)
            disp.append({"display": typ.group(1).strip() if typ else "external display", "resolution": res,
                         "main": bool(re.search(r"Main Display: Yes", block))})
    out["displays"] = disp
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    out["pmset"] = {k: v for k, v in re.findall(r"^\s*(displaysleep|sleep|powermode|lowpowermode)\s+(\S+)", pm, re.M)}
    return out


def state():
    bat = ioreg_battery()
    return {"epoch": round(time.time(), 1), "battery": {k: bat.get(k) for k in FLAG_KEYS + NUM_KEYS + ("adapter",)},
            "system_power_in_w": round(bat.get("SystemPowerIn", 0) / 1000, 3), "display": display_state()}


# ---------------------------------------------------------------- recording
def record(out_path, stop=None, seconds=None, label="", interval=1.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    smc = SMC()
    trace = {"label": label, "started_epoch": round(time.time(), 3), "interval_s": interval,
             "sources": {"pdtr_w": "SMC key PDTR (DC input power, W), every second",
                         "pstr_w": "SMC key PSTR (system total power, W), every second",
                         "ioreg": "ioreg AppleSmartBattery PowerTelemetryData, kept at each refresh (UpdateTime)"},
             "state_at_start": state(), "samples": [], "ioreg_updates": []}
    done = {"stop": False}
    signal.signal(signal.SIGTERM, lambda *_: done.update(stop=True))
    signal.signal(signal.SIGINT, lambda *_: done.update(stop=True))
    last_update, last_write, t_next = None, time.time(), time.time()

    def save(final=False):
        trace["ended_epoch"] = round(time.time(), 3)
        if final:
            trace["state_at_end"] = state()
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(trace, separators=(",", ":")))
        os.replace(tmp, out_path)

    while not done["stop"]:
        t = time.time()
        s = {"t": round(t, 3), "pdtr_w": smc.read_float("PDTR"), "pstr_w": smc.read_float("PSTR")}
        try:
            bat = ioreg_battery()
        except (subprocess.SubprocessError, OSError):
            bat = {}
        s["in_w"] = round(bat["SystemPowerIn"] / 1000, 3) if "SystemPowerIn" in bat else None
        trace["samples"].append(s)
        if bat.get("UpdateTime") and bat.get("UpdateTime") != last_update:
            trace["ioreg_updates"].append({"t_seen": round(t, 3), **bat})
            last_update = bat["UpdateTime"]
        if t - last_write >= 30:
            save()
            last_write = t
        if (stop and Path(stop).exists()) or (seconds and t - trace["started_epoch"] >= seconds):
            break
        t_next += interval
        time.sleep(max(0.0, t_next - time.time()))
    save(final=True)
    return trace


# ---------------------------------------------------------------- analysis
def load(path):
    return json.loads(Path(path).read_text())


def updates(trace, acc_path=None):
    """ioreg refreshes, from the trace and (if given) an `accumulators` file, one per UpdateTime, in order."""
    ups = {u["UpdateTime"]: u for u in trace.get("ioreg_updates", []) if u.get("UpdateTime")}
    if acc_path and Path(acc_path).exists():
        for line in Path(acc_path).read_text().splitlines():
            if line.strip():
                u = json.loads(line)
                ups[u["UpdateTime"]] = {**ups.get(u["UpdateTime"], {}), **u}
    for u in ups.values():
        for k in SIGNED:
            if isinstance(u.get(k), int):
                u[k] = signed(k, u[k])
    return [ups[k] for k in sorted(ups)]


def ioreg_intervals(ups, acc="AccumulatedSystemPowerIn", count="SystemPowerInAccumulatorCount"):
    """[(t0, t1, mean_w, refresh)] from consecutive refreshes: mean = d(accumulated) / d(count)."""
    ups = [u for u in ups if acc in u and count in u]
    out = []
    for a, b in zip(ups, ups[1:]):
        dn = b[count] - a[count]
        if dn > 0:
            out.append((a["UpdateTime"], b["UpdateTime"], (b[acc] - a[acc]) / dn / 1000, b))
    return out


def _mean(xs):
    return round(statistics.fmean(xs), 2) if xs else None


def window(trace, t0, t1, acc_path=None):
    """Mean W and Wh over [t0, t1] (epoch seconds): system total (PSTR) at 1 Hz, the adapter input (PDTR), the
    battery's share, the ioreg cross-check and the battery state."""
    secs = t1 - t0
    xs = [s for s in trace["samples"] if t0 <= s["t"] <= t1 and s.get("pstr_w") is not None]
    if not xs or secs <= 0:
        return None
    tot = [s["pstr_w"] for s in xs]
    inp = [s["pdtr_w"] for s in xs if s.get("pdtr_w") is not None]
    mean_w, in_w = statistics.fmean(tot), (statistics.fmean(inp) if inp else None)
    res = {"start_epoch": t0, "end_epoch": t1, "seconds": round(secs, 1), "samples": len(xs),
           "coverage": round(len(xs) / max(1.0, secs), 3),
           "mean_w": round(mean_w, 2), "wh": round(mean_w * secs / 3600, 4),
           "median_w": round(statistics.median(tot), 2), "min_w": round(min(tot), 2), "max_w": round(max(tot), 2),
           "input_mean_w": round(in_w, 2) if in_w is not None else None,
           "input_wh": round(in_w * secs / 3600, 4) if in_w is not None else None,
           "battery_wh": round((mean_w - in_w) * secs / 3600, 4) if in_w is not None else None,
           "measure": "system total power (SMC PSTR) at 1 Hz; input = adapter DC input (SMC PDTR); battery = the "
                      "difference (positive while the battery adds to the input)"}
    ups = updates(trace, acc_path)
    for name, acc, count in (("ioreg_input", "AccumulatedSystemPowerIn", "SystemPowerInAccumulatorCount"),
                             ("ioreg_system", "AccumulatedSystemLoad", "SystemLoadAccumulatorCount")):
        covered, wh = 0.0, 0.0
        for a, b, w, _ in ioreg_intervals(ups, acc, count):
            ov = max(0.0, min(b, t1) - max(a, t0))
            covered += ov
            wh += w * ov / 3600
        if covered:
            res[name + "_mean_w"] = round(wh * 3600 / covered, 2)
    near = [u for u in ups if t0 - 90 <= u["UpdateTime"] <= t1 + 90]
    if near:
        res["battery"] = {"external_connected": all(u.get("ExternalConnected") for u in near),
                          "charging_seen": any(u.get("IsCharging") for u in near),
                          "fully_charged_all": all(u.get("FullyCharged") for u in near),
                          "capacity_pct": sorted({u.get("CurrentCapacity") for u in near}),
                          "battery_power_mw_min": min(u.get("BatteryPower", 0) for u in near),
                          "battery_power_mw_max": max(u.get("BatteryPower", 0) for u in near),
                          "refreshes": len(near)}
    return res


def check(trace, acc_path=None):
    """SMC against ioreg over every whole refresh interval: PDTR vs the input accumulator, PSTR vs the system-load
    accumulator. Returns the energy ratio and the per-interval ratio range for each."""
    ups = updates(trace, acc_path)
    out = {}
    for name, key, acc, count in (("input", "pdtr_w", "AccumulatedSystemPowerIn", "SystemPowerInAccumulatorCount"),
                                  ("system", "pstr_w", "AccumulatedSystemLoad", "SystemLoadAccumulatorCount")):
        rows = []
        for a, b, w, _ in ioreg_intervals(ups, acc, count):
            xs = [s[key] for s in trace["samples"] if a < s["t"] <= b and s.get(key) is not None]
            if len(xs) >= 0.8 * (b - a) and w:
                rows.append((b - a, w, statistics.fmean(xs)))
        if rows:
            e_io, e_smc = sum(d * w for d, w, _ in rows), sum(d * x for d, _, x in rows)
            ratios = [x / w for _, w, x in rows]
            out[name] = {"intervals": len(rows), "seconds": sum(d for d, _, _ in rows),
                         "smc_over_ioreg_energy": round(e_smc / e_io, 4),
                         "interval_ratio_median": round(statistics.median(ratios), 4),
                         "interval_ratio_range": [round(min(ratios), 4), round(max(ratios), 4)]}
    return out


def accumulators(out_path, stop, every=10.0):
    """Poll ioreg every few seconds and keep each refresh (JSON lines): a light companion to `record` whose trace
    began before the system-load and battery accumulators were kept."""
    last = None
    with open(out_path, "a") as fh:
        while not Path(stop).exists():
            try:
                bat = ioreg_battery()
            except (subprocess.SubprocessError, OSError):
                bat = {}
            if bat.get("UpdateTime") and bat["UpdateTime"] != last:
                fh.write(json.dumps({"t_seen": round(time.time(), 3), **bat}) + "\n")
                fh.flush()
                last = bat["UpdateTime"]
            time.sleep(every)


def runs(trace_path, run_all_path, out_dir, idle_w=None, acc_path=None):
    """One data/power/<trace-stem>.json per run of a run_all.sh timing file, from the trace that covered it."""
    trace, rows = load(trace_path), load(run_all_path)["workflows"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for r in rows:
        if not r.get("trace"):
            continue
        w = window(trace, r["start_epoch"], r["end_epoch"], acc_path)
        if not w:
            continue
        rec = {"workflow": r["workflow"], "pass": r["pass"], "trace": r["trace"], "score": r.get("score"),
               "exit_status": r["exit_status"], "power_trace": os.path.relpath(trace_path, out_dir.parent), **w,
               "window_from": os.path.relpath(run_all_path, out_dir.parent) + " start_epoch/end_epoch (date +%s)"}
        if idle_w is not None:
            rec["idle_w"] = idle_w
            rec["above_idle_w"] = round(w["mean_w"] - idle_w, 2)
            rec["above_idle_wh"] = round((w["mean_w"] - idle_w) * w["seconds"] / 3600, 4)
        p = out_dir / (Path(r["trace"]).stem + ".json")
        p.write_text(json.dumps(rec, indent=1) + "\n")
        made.append(rec)
    return made


def spread(xs):
    return {"min": min(xs), "median": round(statistics.median(xs), 4), "max": max(xs)} if xs else None


def summary(idle_trace, local_trace, run_all, waiting, out, acc=None, power_dir=None, notes=None, idle_extra=None):
    """data/power-summary.json: the idle baseline, every metered local run, each pass, and the waiting draw."""
    root = Path(out).resolve().parent.parent
    rel = lambda p: os.path.relpath(Path(p).resolve(), root)
    it = load(idle_trace)
    idle = window(it, it["samples"][0]["t"], it["samples"][-1]["t"])
    idle_w = idle["mean_w"]
    made = runs(local_trace, run_all, power_dir or Path(out).parent / "power", idle_w, acc)
    pdir = Path(power_dir or Path(out).parent / "power")
    for m in made:
        m["power_file"] = rel(pdir / (Path(m["trace"]).stem + ".json"))
    lt, rows = load(local_trace), load(run_all)["workflows"]
    passes = []
    for p in sorted({r["pass"] for r in rows}):
        rs = [r for r in rows if r["pass"] == p]
        w = window(lt, min(r["start_epoch"] for r in rs), max(r["end_epoch"] for r in rs), acc)
        passes.append({"pass": p, "workflows": len(rs), "seconds_in_runs": sum(r["seconds"] for r in rs),
                       "wh_in_runs": round(sum(m["wh"] for m in made if m["pass"] == p), 3),
                       "window_incl_scoring": {k: w[k] for k in ("seconds", "mean_w", "wh", "input_wh", "battery_wh")}})
    by_wf = {}
    for m in made:
        by_wf.setdefault(m["workflow"], []).append(m)
    wait = load(waiting)
    res = {
        "generated_by": "demos/experiments/power_meter.py summary",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "what": "The measurement laptop's own power draw, metered every second without sudo: SMC PSTR (system total, "
                "the headline), SMC PDTR (adapter DC input) and macOS's battery telemetry (ioreg AppleSmartBattery "
                "PowerTelemetryData) as a cross-check. DC side: the adapter's conversion loss and the external "
                "monitor (on its own supply) are not included.",
        "headline": "Wh = mean system total power over the run x its wall time (run_all.sh start/end, date +%s). "
                    "above_idle_wh = (mean W - the idle baseline) x wall time, a note: the share the workload added.",
        "conditions": {"state_at_start": it.get("state_at_start"), "local_state_at_end": lt.get("state_at_end"),
                       "notes": notes or []},
        "idle_baseline": {"trace": rel(idle_trace), "label": it.get("label"), **idle,
                          "check_against_ioreg": check(it)},
        "other_idle_baselines": [
            {"trace": rel(x), "label": load(x).get("label"),
             **window(load(x), load(x)["samples"][0]["t"], load(x)["samples"][-1]["t"]),
             "check_against_ioreg": check(load(x))} for x in (idle_extra or [])],
        "local_trace": rel(local_trace), "local_accumulators": rel(acc) if acc else None,
        "local_check_against_ioreg": check(lt, acc),
        "local_runs": made,
        "local_by_workflow": {wf: {"passes": len(ms), "wh": spread([m["wh"] for m in ms]),
                                   "mean_w": spread([m["mean_w"] for m in ms]),
                                   "above_idle_wh": spread([m["above_idle_wh"] for m in ms]),
                                   "seconds": spread([m["seconds"] for m in ms])} for wf, ms in by_wf.items()},
        "local_passes": passes,
        "claude_waiting": wait,
    }
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    return res


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "record":
        opts = {"--stop": None, "--seconds": None, "--label": ""}
        out = rest[0]
        for i, a in enumerate(rest[1:], 1):
            if a in opts:
                opts[a] = rest[i + 1]
        tr = record(out, stop=opts["--stop"], seconds=float(opts["--seconds"]) if opts["--seconds"] else None,
                    label=opts["--label"])
        w = window(tr, tr["samples"][0]["t"], tr["samples"][-1]["t"]) if tr["samples"] else None
        print(json.dumps({"trace": out, "samples": len(tr["samples"]), "window": w}, indent=1))
    elif cmd == "window":
        print(json.dumps(window(load(rest[0]), float(rest[1]), float(rest[2]), rest[3] if len(rest) > 3 else None),
                         indent=1))
    elif cmd == "check":
        print(json.dumps(check(load(rest[0]), rest[1] if len(rest) > 1 else None), indent=1))
    elif cmd == "state":
        print(json.dumps(state(), indent=1))
    elif cmd == "summary":
        opt = lambda k: rest[rest.index(k) + 1] if k in rest else None
        res = summary(opt("--idle"), opt("--local"), opt("--run-all"), opt("--waiting"), opt("--out"), opt("--acc"),
                      opt("--power-dir"), json.loads(opt("--notes")) if opt("--notes") else None,
                      [rest[i + 1] for i, a in enumerate(rest) if a == "--idle-extra"])
        print(json.dumps({"idle_w": res["idle_baseline"]["mean_w"], "waiting_w": res["claude_waiting"].get("mean_w"),
                          "runs": len(res["local_runs"]), "passes": res["local_passes"]}, indent=1))
    elif cmd == "accumulators":
        accumulators(rest[0], rest[rest.index("--stop") + 1])
    elif cmd == "runs":
        idle = float(rest[rest.index("--idle-w") + 1]) if "--idle-w" in rest else None
        acc = rest[rest.index("--acc") + 1] if "--acc" in rest else None
        made = runs(rest[0], rest[1], rest[2], idle, acc)
        for m in made:
            print(f"  p{m['pass']} {m['workflow']:<12} {m['seconds']:>6} s  {m['mean_w']:>6} W  {m['wh']:>7} Wh")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
