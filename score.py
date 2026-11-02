"""Score a workflow's outputs against the answer keys.

Usage:  .venv/bin/python score.py inbox      [outputs_dir]  [--tag local]
        .venv/bin/python score.py study_pack [outputs_dir]  [--tag local]
        .venv/bin/python score.py receipts   [outputs_dir]  [--tag local]
        .venv/bin/python score.py <workflow> ...   for workflows with a scorer in scoring/<workflow>.py
Writes results/scores/<workflow>-<tag>.json
"""
import csv
import importlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = ROOT / "sample_data"
SCORES = ROOT / "results" / "scores"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def same_minute(a, b):
    try:
        return abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()) < 60
    except (TypeError, ValueError):
        return False


def match(pred, gold, key):
    """Greedy one-to-one match on exact time; returns (true_pos, false_pos list, missed list)."""
    left, tp, fp = list(gold), 0, []
    for p in pred:
        hit = next((g for g in left if same_minute(p.get(key), g[key])), None)
        if hit:
            left.remove(hit)
            tp += 1
        else:
            fp.append(p)
    return tp, fp, left


def _items(path):
    """Events from a calendar file: Google's events.list shape {"items": [...]} or a bare list."""
    cal = json.loads(path.read_text())
    return cal.get("items", []) if isinstance(cal, dict) else cal


def _t(e, k):   # start/end as an ISO string, from an event resource ({"dateTime": ...}) or a flat plan entry
    v = e.get(k)
    return v.get("dateTime") if isinstance(v, dict) else v


def _links(e):
    raw = (e.get("extendedProperties") or {}).get("private", {}).get("source_message_ids", "")
    return {x.strip() for x in str(raw).split(",") if x.strip()}


def _live(e):
    return e.get("status", "confirmed") != "cancelled"


def _hit(e, keywords):
    return any(norm(k) in norm(e.get("summary", "")) for k in keywords)


def _changed(a, b):
    """Fields that differ between two versions of an event (times compared to the minute)."""
    diff = [k for k in ("start", "end") if not same_minute(_t(a, k), _t(b, k))]
    diff += [k for k in ("summary", "description", "location", "recurrence") if a.get(k) != b.get(k)]
    return diff + ([] if _live(b) else ["status: cancelled"])


def score_calendar(out, ck):
    """Grade calendar_plan.json and calendar_after.json against the answer key's expected changes."""
    missing = [f for f in ("calendar_plan.json", "calendar_after.json") if not (out / f).exists()]
    if missing:
        return {"ok": False, "error": f"missing {missing}"}
    exp = ck["expected_changes"]
    before = {e["id"]: e for e in _items(KEYS / "inbox" / "calendar_before.json")}
    after_list = _items(out / "calendar_after.json")
    after = {e["id"]: e for e in after_list}
    live = [e for e in after_list if _live(e)]
    new = [e for e in live if e["id"] not in before]

    # Creates: each expected new event exactly once (start and end to the minute, title keyword), linked to its message.
    pool, created, unlinked = list(new), [], []
    for c in exp["create"]:
        hit = next((e for e in pool if same_minute(_t(e, "start"), c["start"]) and same_minute(_t(e, "end"), c["end"])
                    and _hit(e, c["keywords"])), None)
        if hit:
            pool.remove(hit)
            created.append(c)
            if not set(c["source_message_ids"]) <= _links(hit):
                unlinked.append(f"{hit.get('summary')} (want link {c['source_message_ids']})")
    # Updates: the SAME event moved (not a new one), now linked to the update message too.
    updated = []
    for u in exp["update"]:
        e = after.get(u["event_id"])
        if e and _live(e) and same_minute(_t(e, "start"), u["start"]) and same_minute(_t(e, "end"), u["end"]):
            updated.append(u)
            if not set(u["link_must_include"]) <= _links(e):
                unlinked.append(f"{e.get('summary')} (want link {u['link_must_include']})")
    deleted = [d for d in exp["delete"] if d["event_id"] not in after or not _live(after[d["event_id"]])]
    # Collateral: every other event that existed before must be exactly as it was.
    collateral = [f"{i} {before[i].get('summary')!r}: " + ", ".join(_changed(before[i], after[i])) if i in after
                  else f"{i} {before[i].get('summary')!r}: removed"
                  for i in ck["untouched_event_ids"] if i not in after or _changed(before[i], after[i])]
    # Duplicates: two live events for one item (same time with the same title words, or linked to the same message).
    duplicates = []
    for x in exp["create"] + exp["update"] + exp["no_change"]:
        same = [e for e in live if _hit(e, x["keywords"]) and
                (same_minute(_t(e, "start"), x["start"]) or set(x["source_message_ids"]) & _links(e) or e["id"] == x.get("event_id"))]
        if len(same) > 1:
            duplicates.append(f"{len(same)} events for {x['keywords'][0]!r}: {[e.get('summary') for e in same]}")

    def in_final(f):   # an expected final event: a new one we matched above, or an old one at its expected time
        if f["event_id"] is None:
            return any(same_minute(c["start"], f["start"]) for c in created)
        e = after.get(f["event_id"])
        return bool(e) and _live(e) and same_minute(_t(e, "start"), f["start"]) and same_minute(_t(e, "end"), f["end"])
    final_ok = sum(map(in_final, ck["final_calendar"]))

    # The plan: what the human approved. Every expected write must be in it, deletions listed explicitly.
    plan = json.loads((out / "calendar_plan.json").read_text())
    changes = plan.get("changes", [])
    by_action = {a: [c for c in changes if c.get("action") == a] for a in ("create", "update", "delete")}
    plan_creates = sum(1 for c in exp["create"] if any(same_minute(_t(p, "start"), c["start"]) for p in by_action["create"]))
    plan_updates = sum(1 for u in exp["update"] if any(p.get("event_id") == u["event_id"] and same_minute(_t(p, "start"), u["start"])
                                                       for p in by_action["update"]))
    want_del = {d["event_id"] for d in exp["delete"]}
    planned_del = {p.get("event_id") for p in by_action["delete"]}
    listed_del = {d.get("event_id") for d in plan.get("deletions", [])}
    unexpected = [f"{p['action']} {p.get('event_id')} {p.get('summary')!r}" for p in by_action["update"] + by_action["delete"]
                  if p.get("event_id") not in {u["event_id"] for u in exp["update"]} | want_del]
    result = {
        "creates": f"{len(created)}/{len(exp['create'])}",
        "missing_creates": [f"{c['title']} @ {c['start']}" for c in exp["create"] if c not in created],
        "updates": f"{len(updated)}/{len(exp['update'])}",
        "deletes": f"{len(deleted)}/{len(exp['delete'])}",
        "no_change_respected": all(x["event_id"] in after and not _changed(before[x["event_id"]], after[x["event_id"]])
                                   for x in exp["no_change"]),
        "collateral_changes": collateral,
        "duplicates": duplicates,
        "extra_events": [f"{e.get('summary')} @ {_t(e, 'start')}" for e in pool],
        "unlinked": unlinked,
        "final_state": f"{final_ok}/{len(ck['final_calendar'])}",
        "final_state_exact": final_ok == len(ck["final_calendar"]) and not pool and len(deleted) == len(exp["delete"]),
        "plan": {"creates": f"{plan_creates}/{len(exp['create'])}", "updates": f"{plan_updates}/{len(exp['update'])}",
                 "deletes_exact": planned_del == want_del, "deletions_listed": listed_del == planned_del,
                 "unexpected_changes": unexpected, "applied": bool(plan.get("applied"))}}
    result["ok"] = (len(created) == len(exp["create"]) and len(updated) == len(exp["update"])
                    and len(deleted) == len(exp["delete"]) and result["no_change_respected"]
                    and not (collateral or duplicates or pool or unlinked) and result["final_state_exact"]
                    and plan_creates == len(exp["create"]) and plan_updates == len(exp["update"])
                    and planned_del == want_del == listed_del and not unexpected and result["plan"]["applied"])
    return result


def score_inbox(out):
    key = json.loads((KEYS / "inbox" / "answer_key.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    sched = json.loads((out / "schedule.json").read_text())
    correct = [l for l in labels if l.get("category") == key["messages"][l["id"]]["category"]]
    tp, fp, missed = match(sched["deadlines"], key["deadlines"], "due")
    etp, efp, emissed = match(sched["events"], key["events"], "start")
    result = {"classification_accuracy": round(len(correct) / len(labels), 3),
              "misclassified": [{"id": l["id"], "got": l.get("category"), "want": key["messages"][l["id"]]["category"]}
                                for l in labels if l not in correct],
              "deadlines": {"precision": round(tp / max(len(sched["deadlines"]), 1), 3), "recall": round(tp / len(key["deadlines"]), 3),
                            "correct": tp, "expected": len(key["deadlines"]),
                            "extra": [f"{d.get('task')} @ {d.get('due')}" for d in fp],
                            "missed": [f"{d['task']} @ {d['due']}" for d in missed]},
              "events": {"correct": etp, "expected": len(key["events"]),
                         "extra": [f"{e.get('title')} @ {e.get('start')}" for e in efp],
                         "missed": [f"{e['title']} @ {e['start']}" for e in emissed]},
              "calendar": score_calendar(out, key["calendar"])}
    result["full_score"] = (result["classification_accuracy"] == 1 and tp == len(key["deadlines"]) and not fp
                            and etp == len(key["events"]) and not efp and result["calendar"]["ok"])
    return result


def score_study(out):
    key = json.loads((KEYS / "lecture" / "answer_key.json").read_text())
    transcript = norm((KEYS / "lecture" / "transcript.txt").read_text())
    summary = norm((out / "summary.md").read_text())
    with open(out / "flashcards.csv") as f:
        cards = [row for row in csv.reader(f) if row]
    deck = norm(" ".join(" ".join(r[:2]) for r in cards))
    everything = summary + " " + deck
    concepts = [c["term"] for c in key["key_concepts"]]
    def variants(term):  # "rehashing (resizing)" matches either word
        return [norm(v) for v in re.split(r"[()/]", term) if norm(v)]
    covered = [t for t in concepts if any(v in everything for v in variants(t))]
    terms = key.get("must_include_terms", [])
    quiz = (out / "quiz.md").read_text()
    n_q = len(re.findall(r"^\*\*\d+\.", quiz, flags=re.M))
    quotes = []
    cj = out / "cards.json"
    if cj.exists():
        quotes = [c.get("source_quote", "") for c in json.loads(cj.read_text()).get("kept", [])]
    verbatim = sum(1 for q in quotes if q and norm(q) in transcript)
    return {"key_concepts_covered": f"{len(covered)}/{len(concepts)}",
            "key_concept_coverage": round(len(covered) / len(concepts), 3),
            "missing_concepts": [t for t in concepts if t not in covered],
            "must_include_terms_covered": f"{sum(norm(t) in everything for t in terms)}/{len(terms)}",
            "flashcards": len(cards), "quiz_questions": n_q,
            "flashcard_quotes_verbatim": f"{verbatim}/{len(quotes)}" if quotes else "n/a",
            "summary_words": len(summary.split()),
            "false_claims_to_review_by_hand": [c if isinstance(c, str) else c.get("claim") for c in key.get("false_claims", [])]}


def score_receipts(out):
    key = json.loads((KEYS / "receipts" / "answer_key.json").read_text())
    got = json.loads((out / "receipts.json").read_text())
    gold = key["receipts"]
    gold_by_id = {g.get("id", g.get("receipt", f"r{i + 1}")): g for i, g in enumerate(gold)} if isinstance(gold, list) else gold
    totals_ok, items_ok, items_n, assign_ok = 0, 0, 0, 0
    for r in got["receipts"]:
        g = gold_by_id.get(r["id"])
        if not g:
            continue
        totals_ok += abs(float(r["total"]) - float(g["total"])) < 0.005
        gl = list(g["items"])
        for it in r["items"]:
            hit = next((x for x in gl if abs(float(x["line_total"]) - float(it["line_total"])) < 0.005), None)
            if hit:
                gl.remove(hit)
                items_ok += 1
                assign_ok += sorted(hit["assigned_to"]) == sorted(it["assigned_to"])
        items_n += len(g["items"])
    want = {(s["from"], s["to"]): round(float(s["amount"]), 2) for s in key["settlement"]}
    have = {(s["from"], s["to"]): round(float(s["amount"]), 2) for s in got["balances"]["settlement"]}
    return {"receipts_total_exact": f"{totals_ok}/{len(gold_by_id)}",
            "line_items_read_exact": f"{items_ok}/{items_n}",
            "items_assigned_correctly": f"{assign_ok}/{items_n}",
            "settlement_exact": want == have, "settlement_expected": key["settlement"],
            "settlement_got": got["balances"]["settlement"]}


if __name__ == "__main__":
    wf = sys.argv[1]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "local"
    rest = [a for a in sys.argv[2:] if a not in ("--tag", tag)]
    out = Path(rest[0]) if rest else ROOT / "outputs" / wf
    builtin = {"inbox": score_inbox, "study_pack": score_study, "receipts": score_receipts}
    # Newer workflows keep their scorer in scoring/<workflow>.py, exposing score(outputs_dir) -> dict.
    result = builtin[wf](out) if wf in builtin else importlib.import_module(f"scoring.{wf}").score(out)
    result.update(workflow=wf, tag=tag, outputs=str(out))
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / f"{wf}-{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
