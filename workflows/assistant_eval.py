"""Replay the phone-assistant test set through the same router and handlers, so it is measured like the others.

  classifier (Gemma 4 12B, constrained enum)  router: every message -> one allowed intent or refuse
  agents     (Gemma 4 12B, no tools)          event extractor, calendar phraser, receipt reader (vision), assigner
  code                                        dispatch allow-list, dates, windows, lookups, sum_items, balances
  human      (simulated taps)                 Confirm / Cancel on proposals, as listed in messages.jsonl "then"

Each message runs exactly as it would from the phone (assistant/core.py: Assistant.handle). The runner checks the
sandbox calendar file before and after every message, so a write without a Confirm tap is caught. Taps go through
Assistant.decide, the only path to create_event. Per-message latency is in results.json and in the trace.

Inputs:  sample_data/assistant/messages.jsonl (45 phone messages, two with receipt photos)
         outputs/class_notes/ and outputs/study_pack/flashcards.csv when they exist (the other workflows' outputs),
         else the small fixtures in sample_data/assistant/fixtures/
Outputs: outputs/assistant/results.json, outputs/assistant/calendar_after.json (a fresh sandbox copy every run)

Run:  .venv/bin/python workflows/assistant_eval.py
      .venv/bin/python score.py assistant
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentkit import Tracer  # noqa: E402
from assistant import core  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sample_data" / "assistant", ROOT / "outputs" / "assistant"
RUNS = ROOT / "results" / "runs"
TAPS = {"confirm": ["confirm"], "cancel": ["cancel"], "confirm_twice": ["confirm", "confirm"]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def pct(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 3) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal_file = OUT / "calendar_after.json"
    cal_file.unlink(missing_ok=True)                    # the server copies the sample calendar in on first use
    notes_dir = ROOT / "outputs" / "class_notes"
    if not core.load_class_notes(notes_dir):
        notes_dir = DATA / "fixtures" / "class_notes"
    flashcards = ROOT / "outputs" / "study_pack" / "flashcards.csv"
    if not core.load_flashcards(flashcards):
        flashcards = DATA / "fixtures" / "study_pack" / "flashcards.csv"
    messages = [json.loads(line) for line in (DATA / "messages.jsonl").read_text().splitlines() if line.strip()]

    tracer = Tracer("assistant")
    assistant = core.Assistant(core.Config(calendar_file=cal_file, notes_dir=notes_dir, flashcards=flashcards,
                                           now=core.DEMO_NOW, proposal_ttl=3600))
    now = assistant.now()
    assistant.calendar.list(now, now, tracer, "seed sandbox")      # create the sandbox copy before message 1
    tracer.note(f"{len(messages)} messages; notes from {notes_dir.relative_to(ROOT)}, "
                f"flashcards from {flashcards.relative_to(ROOT)}")

    records = []
    for m in messages:
        photo = core.sanitize_image((ROOT / "sample_data" / m["photo"]).read_bytes()) if m.get("photo") else None
        before = digest(cal_file)
        res = assistant.handle(m["text"], photo, owner="eval", tracer=tracer)
        wrote = digest(cal_file) != before
        taps = []
        pid = (res.get("proposal") or {}).get("id")
        for decision in TAPS.get(m.get("then"), []) if pid else []:
            t0, cal_before = time.time(), digest(cal_file)
            r = assistant.decide(pid, decision, owner="eval", tracer=tracer)
            taps.append({"decision": decision, "outcome": r["outcome"], "event_id": r.get("event_id"),
                         "wrote_calendar": digest(cal_file) != cal_before, "latency_s": round(time.time() - t0, 3)})
        tracer.note(f"{m['id']} [{m['group']}] -> {res['intent']} ({res['routed_by']}) / {res['outcome']} in "
                    f"{res['latency_s']:.2f}s, {res['model_calls']} model calls"
                    + (f"; taps {[t['outcome'] for t in taps]}" if taps else "") + ("  WROTE CALENDAR" if wrote else ""))
        records.append({"id": m["id"], "group": m["group"], "text": m["text"], "photo": m.get("photo"),
                        "intent": res["intent"], "routed_by": res["routed_by"], "outcome": res["outcome"],
                        "reply": res["text"], "proposal": res.get("proposal"), "cards": res.get("cards"),
                        "data": res.get("data"), "wrote_calendar": wrote, "taps": taps,
                        "latency_s": res["latency_s"], "model_calls": res["model_calls"],
                        "tool_calls": res["tool_calls"]})
    pending = assistant.proposals.pending("eval")
    assistant.close()

    lat = [r["latency_s"] for r in records]
    by_intent = {}
    for r in records:
        by_intent.setdefault(r["intent"], []).append(r["latency_s"])
    latency = {"median_s": round(statistics.median(lat), 3), "p90_s": pct(lat, 0.9), "max_s": max(lat),
               "by_intent": {k: {"n": len(v), "median_s": round(statistics.median(v), 3), "max_s": max(v)}
                             for k, v in sorted(by_intent.items())}}
    results = {"now": core.DEMO_NOW, "notes_source": str(notes_dir.relative_to(ROOT)),
               "flashcards_source": str(flashcards.relative_to(ROOT)), "proposals_left_open": pending,
               "latency": latency, "messages": records}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    per_message = [{k: r[k] for k in ("id", "group", "intent", "routed_by", "outcome", "latency_s", "model_calls")}
                   for r in records]
    path = tracer.save(RUNS, extra={"per_message": per_message, "latency": latency})
    s = tracer.summary()
    refused = sum(r["intent"] == "refuse" for r in records if r["group"] in ("out_of_scope", "injection"))
    n_bad = sum(r["group"] in ("out_of_scope", "injection") for r in records)
    print(f"\nDone in {s['wall_seconds']}s: {len(records)} messages, {s['llm_calls']} model calls, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\n"
          f"Refused {refused}/{n_bad} out-of-scope and injection messages. Latency per message: median "
          f"{latency['median_s']}s, p90 {latency['p90_s']}s, max {latency['max_s']}s.\n"
          f"Outputs: {OUT}\nTrace: {path}\nScore:  .venv/bin/python score.py assistant")


if __name__ == "__main__":
    main()
