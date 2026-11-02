"""Inbox -> Google Calendar update, through an MCP tool server.

  master  (Qwen3.8 27B, thinks only when writing)  plans, settles escalations, writes the digest
  groups  (Gemma 4 12B)            intake lead reviews; scheduler drops superseded/duplicate items;
                                   cal_reconciler decides create / update / delete / no_change for ONE item
  workers (Gemma 4 12B)            classify each message; extract dated items with resolve_date
  code                             dates, end times, matching items to calendar events, checking every
                                   decision, overlaps (mcp_servers/gcal_reconcile.py), applying the approved plan
  MCP     mcp_servers/gcal_server.py   list/get/create/update/delete events (a sandbox file, or real Google)

Human gate: a plain run only writes a plan (calendar_plan.json + calendar_plan.md). The calendar changes
only with --apply, and only by exactly what the plan says.

Run:  .venv/bin/python workflows/inbox_calendar.py                  # plan only (dry run)
      .venv/bin/python workflows/inbox_calendar.py --apply          # plan, then apply it
      .venv/bin/python workflows/inbox_calendar.py --apply --keep   # same calendar again: no changes
Sandbox: each run copies sample_data/inbox/calendar_before.json to outputs/inbox/calendar_after.json and
changes that copy (--keep continues from the last calendar_after.json instead).
GCAL_BACKEND=google GCAL_TOKEN_FILE=... uses your real calendar instead (see mcp_servers/gcal_server.py).
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentkit import Agent, AgentCard, Group, Tracer  # noqa: E402
from agentkit.mcp import MCPServer  # noqa: E402
from agentkit.tools import default_toolbox, resolve_date  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_servers"))
from gcal_reconcile import (ACTIONS, candidates, check_decision, course_in, end_time, links,  # noqa: E402
                            overlaps, plan_markdown, same_minute, when, words)

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sample_data" / "inbox", ROOT / "outputs" / "inbox"
RUNS = ROOT / "results" / "runs"
NOW = "2026-11-02T08:00:00-05:00"   # the demo's "current time" (matches the answer key)
WINDOW_DAYS = 30                    # how far ahead the calendar is read

PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "horizon_days": {"type": "integer", "minimum": 1, "maximum": 14},
    "intake_instruction": {"type": "string"}, "scheduling_instruction": {"type": "string"},
    "digest_focus": {"type": "string"}},
    "required": ["horizon_days", "intake_instruction", "scheduling_instruction", "digest_focus"]}


def fmt(m):
    return (f"From: {m['from_name']} <{m['from']}>\nReceived: {m['received']}\nSource: {m['source']}\n"
            f"Subject: {m['subject']}\n\n{m['body']}")


def check_items(job, result):
    """Code-level critic: every item needs a real ISO time from resolve_date, and a dated message needs an item."""
    if not result.get("items"):
        return ("this message was labelled a deadline or event, but you returned no items. List every dated item, "
                "including one the message cancels (status 'cancelled', original time) or moves (status 'update', new time)")
    for it in result.get("items", []):
        try:
            datetime.fromisoformat(it["due"])
        except (ValueError, KeyError):
            return f"item {it.get('task')!r} has due={it.get('due')!r}; call resolve_date and use its 'iso' value"
    return None


# ---------------------------------------------------------------- the workflow
def main():
    apply, keep = "--apply" in sys.argv, "--keep" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deadlines.ics").unlink(missing_ok=True)   # replaced by the calendar update
    tracer, tools = Tracer("inbox"), default_toolbox(OUT)
    google = os.environ.get("GCAL_BACKEND") == "google"
    cal_file = OUT / "calendar_after.json"
    if not google and not (keep and cal_file.exists()):
        shutil.copyfile(DATA / "calendar_before.json", cal_file)   # the sandbox copy this run may change
    # The same MCP server LM Studio and Claude Code use; its five tools join the toolbox like local ones.
    gcal = tools.add_mcp(MCPServer([sys.executable, str(ROOT / "mcp_servers" / "gcal_server.py")],
                                   env={**os.environ, "GCAL_BACKEND": "google" if google else "sandbox",
                                        "GCAL_SANDBOX_FILE": str(cal_file)}))

    def agent(name):
        return Agent(AgentCard.load(name), tools, tracer)

    master = agent("inbox_master")
    intake = Group("intake", agent("intake_lead"),
                   {"classify": agent("inbox_classifier"), "extract": agent("deadline_extractor")}, tracer)
    scheduling = Group("scheduling", agent("scheduler"),
                       {"merge": agent("scheduler"), "reconcile": agent("cal_reconciler")}, tracer)
    messages = [json.loads(line) for line in (DATA / "messages.jsonl").read_text().splitlines() if line.strip()]
    by_id = {m["id"]: m for m in messages}

    # 0. The master plans from the headers only: a few hundred tokens, not the whole inbox.
    headers = "\n".join(f"{m['id']} | {m['received']} | {m['from_name']} | {m['subject']}" for m in messages)
    plan = master.run(f"PLAN. Current time: {NOW}. Goal: tomorrow-morning-ready digest and an up-to-date calendar "
                      f"for the student.\nInbox headers:\n{headers}", label="plan", schema=PLAN_SCHEMA)
    tracer.note(f"plan: horizon {plan['horizon_days']} days")

    # 1. Intake workers: classify every message in parallel.
    note = f"Note from the lead: {plan['intake_instruction']}\n\n"
    labels = intake.fan_out("classify", [{"label": m["id"], "task": note + fmt(m)} for m in messages])
    for m, lab in zip(messages, labels):
        lab["id"] = m["id"]

    # 2. Intake workers: extract dated items from deadline/event messages, with a code check + retry.
    dated = [m for m, lab in zip(messages, labels) if lab.get("category") in ("deadline", "event")]
    extracted = intake.fan_out("extract", [{"label": m["id"], "task": f"Current time: {NOW}\n\n{fmt(m)}"} for m in dated],
                               check=check_items)
    items = []
    for m, res in zip(dated, extracted):
        for k, it in enumerate(res.get("items", [])):
            items.append({"item_id": f"{m['id']}-{k}", "message_id": m["id"], "subject": m["subject"],
                          "received": m["received"], **it})

    # 3. Intake lead reviews compact results, not raw messages, and escalates doubts.
    compact = [{k: it[k] for k in ("item_id", "subject", "kind", "course", "task", "due", "status", "confidence", "evidence")}
               for it in items]
    review = intake.review(f"Current time: {NOW}\nItems:\n{json.dumps(compact, indent=1)}")

    # 4. The master settles escalations: it reads only the escalated messages.
    item_schema = AgentCard.load("deadline_extractor").output_schema["properties"]["items"]["items"]
    for esc in review["escalate"]:
        it = next((x for x in items if x["item_id"] == esc["item_id"]), None)
        if not it:
            continue
        hint = resolve_date(it["phrase"], it["received"])
        fixed = master.run(f"SETTLE. Current time: {NOW}\nReviewer's concern: {esc['reason']}\n"
                           f"Item: {json.dumps({k: it[k] for k in item_schema['required']})}\n"
                           f"resolve_date({it['phrase']!r}, {it['received']!r}) = {json.dumps(hint)}\n\n"
                           f"Original message:\n{fmt(by_id[it['message_id']])}",
                           label=f"settle:{it['item_id']}", schema=item_schema)
        it.update(fixed, settled_by_master=True)

    # 5. Code drops past items and sets cancelled ones aside; the scheduling lead judges superseded/duplicates.
    live = [it for it in items if it["status"] not in ("past", "cancelled")]
    sched_in = [{k: it[k] for k in ("item_id", "received", "kind", "course", "task", "due", "status", "evidence")}
                for it in live]

    def decided_all(job, res):
        missing = {it["item_id"] for it in live} - {d["item_id"] for d in res.get("decisions", [])}
        return f"give a decision for every item; missing {sorted(missing)}" if missing else None

    decisions = scheduling.fan_out("merge", [{"label": "merge", "task":
        f"Current time: {NOW}\nNote from the master: {plan['scheduling_instruction']}\n"
        f"Items:\n{json.dumps(sched_in, indent=1)}"}], check=decided_all)[0]
    drop = {d["item_id"] for d in decisions.get("decisions", []) if d["action"] == "drop"}
    kept = sorted((it for it in live if it["item_id"] not in drop), key=lambda it: it["due"])

    # 6. Code turns items into calendar items: course codes, titles, end times, and every source message.
    codes = sorted({c for m in messages for f in ("subject", "body", "from_name")
                    for c in re.findall(r"\b[A-Z]{2,5} \d{2,3}\b", m[f])})
    sender_courses = {}
    for m in messages:   # a professor who names the course once: their other messages are about it too
        sender_courses.setdefault(m["from"], set()).add(course_in([m["subject"], m["from_name"]], codes))

    def course_of(it):
        m = by_id[it["message_id"]]
        found = course_in([it["course"], it["task"], m["subject"], m["from_name"]], codes)
        known = sender_courses[m["from"]] - {""}
        return found or (known.pop() if len(known) == 1 else "")

    def calendar_item(it):
        course = course_of(it)
        title = it["task"] if (not course or course.lower() in it["task"].lower()) else f"{course}: {it['task']}"
        return {"item_id": it["item_id"], "kind": it["kind"], "course": course, "task": it["task"],
                "status": it["status"], "summary": title + (" due" if it["kind"] == "deadline" else ""),
                "start": it["due"], "end": end_time(it), "location": it.get("where", ""),
                "source_message_ids": [it["message_id"]], "subject": it["subject"], "evidence": it["evidence"]}

    cal_items = [calendar_item(it) for it in kept]
    for it in items:   # an update keeps the ids of the messages it supersedes (m01 -> m14), so the old event is found
        if it["item_id"] in drop:
            src = calendar_item(it)
            for ci in cal_items:
                if src["course"] in ("", ci["course"]) and len(words(src["task"], codes) & words(ci["task"], codes)) >= 2:
                    ci["source_message_ids"] = sorted(set(ci["source_message_ids"]) | {it["message_id"]},
                                                      key=lambda i: by_id[i]["received"])
                    ci["course"] = ci["course"] or src["course"]
    cal_items += [calendar_item(it) for it in items if it["status"] == "cancelled"]

    schedule = {
        "deadlines": [{"course": ci["course"] or None, "task": ci["task"], "due": ci["start"],
                       "source_message_ids": ci["source_message_ids"], "status": ci["status"], "when": when(ci["start"])}
                      for ci in cal_items if ci["kind"] == "deadline" and ci["status"] != "cancelled"],
        "events": [{"title": ci["task"], "course": ci["course"] or None, "start": ci["start"], "end": ci["end"],
                    "location": ci["location"], "source_message_ids": ci["source_message_ids"], "when": when(ci["start"])}
                   for ci in cal_items if ci["kind"] == "event" and ci["status"] != "cancelled"],
        "dropped": [{"item_id": d["item_id"], "reason": d["reason"]} for d in decisions.get("decisions", []) if d["action"] == "drop"]}
    dl = schedule["deadlines"]
    schedule["conflicts"] = [f"{a['task']} and {b['task']} are due within 3 hours" for a, b in zip(dl, dl[1:])
                             if (datetime.fromisoformat(b["due"]) - datetime.fromisoformat(a["due"])).total_seconds() < 3 * 3600]

    # 7. Read the calendar through the MCP server (a plain tool call by code), then match items to events in code.
    window = {"time_min": NOW, "time_max": (datetime.fromisoformat(NOW) + timedelta(days=WINDOW_DAYS)).isoformat()}
    events = tools.call("list_events", window, tracer, "code", "read calendar")["items"]
    events = [e for e in events if "dateTime" in e.get("start", {})]   # all-day events (birthdays) are not items
    tracer.note(f"calendar: {len(events)} events in the next {WINDOW_DAYS} days")
    jobs = []
    for ci in cal_items:
        cands = candidates(ci, events, codes)
        shown = "\n".join(f"{n}. event_id {c['event_id']} | \"{c['summary']}\" | {when(c['start'])} | "
                          f"linked to the same message: {'yes' if c['linked'] else 'no'} | "
                          f"same time as the item: {'yes' if c['same_time'] else 'no'}" for n, c in enumerate(cands, 1))
        brief = {k: ci[k] for k in ("kind", "course", "task", "status", "source_message_ids", "evidence")}
        jobs.append({"label": ci["item_id"], "item": ci, "cands": cands, "task":
                     f"Current time: {when(NOW)}.\nInbox item: {json.dumps(brief)}\nItem time: {when(ci['start'])}\n"
                     f"Calendar events code found for this item:\n{shown or 'none'}"})

    # 8. The reconciler decides per item, in parallel; code checks every decision and sends problems back.
    verdicts = scheduling.fan_out("reconcile", jobs, check=check_decision)

    # 9. Code writes the plan: exact times, ids and links. Deletions are listed on their own.
    by_event = {ev["id"]: ev for ev in events}
    changes = []
    for job, v in zip(jobs, verdicts):
        ci = job["item"]
        base = {"item_id": ci["item_id"], "kind": ci["kind"], "source_message_ids": ci["source_message_ids"]}
        if "_failed" in v:   # the model never passed the check: a human decides, nothing is written
            changes.append({**base, "action": "needs_human", "event_id": None, "summary": ci["summary"],
                            "start": ci["start"], "end": ci["end"], "reason": v["_failed"]})
            continue
        ev = by_event.get(v.get("event_id"))
        entry = {**base, "action": v["action"], "event_id": ev["id"] if ev else None, "reason": v["reason"]}
        if v["action"] == "create":
            entry.update(event_id=None, summary=ci["summary"], start=ci["start"], end=ci["end"], location=ci["location"],
                         description=f"From inbox message {', '.join(ci['source_message_ids'])}: {ci['subject']}\n"
                                     f"\"{ci['evidence']}\"")
        elif v["action"] == "update":
            entry.update(summary=ev["summary"], start=ci["start"], end=ci["end"],
                         was={"start": ev["start"]["dateTime"], "end": ev["end"]["dateTime"]},
                         source_message_ids=sorted(set(links(ev)) | set(ci["source_message_ids"]),
                                                   key=lambda i: by_id.get(i, {}).get("received", i)))
        elif ev:   # delete or no_change on an event that exists
            entry.update(summary=ev["summary"], start=ev["start"]["dateTime"], end=ev["end"]["dateTime"])
        else:      # no_change and nothing on the calendar (a cancelled item that was never added)
            entry.update(summary=ci["summary"], start=ci["start"], end=ci["end"])
            weekly = [e for e in events if e.get("recurringEventId") and same_minute(e["start"]["dateTime"], ci["start"])]
            if ci["status"] == "cancelled" and weekly:   # e.g. this week's office hours are off: a person's job
                entry["by_hand"] = (f"'{weekly[0]['summary']}' on {when(ci['start'])} is one occurrence of a weekly "
                                    "event; this workflow never edits those, so cancel that occurrence yourself")
        changes.append(entry)
    counts = Counter(c["action"] for c in changes)
    cal_plan = {"calendar": "Google Calendar (primary)" if google else f"sandbox file {cal_file.name}",
                "now": NOW, "window": window, "applied": False,
                "counts": {a: counts.get(a, 0) for a in ACTIONS + ("needs_human",) if counts.get(a) or a in ACTIONS},
                "deletions": [{k: c[k] for k in ("event_id", "summary", "start", "reason")}
                              for c in changes if c["action"] == "delete"],
                "changes": changes, "conflicts": overlaps(changes, events)}

    # 10. Human gate: write the plan; touch the calendar only with --apply, through the same MCP tools.
    if apply:
        for c in changes:
            ids = ",".join(c["source_message_ids"])
            if c["action"] == "create":
                # Idempotent: if an event from the same message is already at this time, do not add another.
                there = tools.call("list_events", {"time_min": c["start"], "time_max": c["end"]}, tracer, "code", c["item_id"])
                if any(set(c["source_message_ids"]) & set(links(e)) for e in there.get("items", [])):
                    c["result"] = "already there"
                    continue
                res = tools.call("create_event", {"summary": c["summary"], "start": c["start"], "end": c["end"],
                                                  "description": c["description"], "location": c["location"],
                                                  "source_message_ids": ids}, tracer, "code", c["item_id"])
                c["event_id"] = res.get("id", c["event_id"])
            elif c["action"] == "update":
                res = tools.call("update_event", {"event_id": c["event_id"], "start": c["start"], "end": c["end"],
                                                  "source_message_ids": ids}, tracer, "code", c["item_id"])
            elif c["action"] == "delete":
                res = tools.call("delete_event", {"event_id": c["event_id"]}, tracer, "code", c["item_id"])
            else:
                continue
            c["result"] = res.get("error", "done")
        cal_plan["applied"] = True
        if google:   # the sandbox file is already the "after" state; for Google, save a snapshot of the window
            cal_file.write_text(json.dumps(tools.call("list_events", window, tracer, "code", "snapshot"), indent=2) + "\n")
    gcal.close()
    (OUT / "calendar_plan.json").write_text(json.dumps(cal_plan, indent=2) + "\n")
    (OUT / "calendar_plan.md").write_text(plan_markdown(cal_plan))

    # 11. The master writes the digest from the clean, merged facts, including what changed on the calendar.
    cats = Counter(lab.get("category", "unknown") for lab in labels)
    cal_facts = [{"action": c["action"], "summary": c["summary"], "when": when(c["start"]),
                  **({"was": when(c["was"]["start"])} if c["action"] == "update" else {})}
                 for c in changes if c["action"] != "no_change"]
    cal_facts += [{"action": "do by hand", "note": c["by_hand"]} for c in changes if c.get("by_hand")]
    digest = master.run(f"WRITE the digest. Current time: {when(NOW)} (today is Monday Nov 2). Use the 'when' labels as given. Horizon: {plan['horizon_days']} days.\n"
                        f"Focus: {plan['digest_focus']}\nDeadlines:\n{json.dumps(schedule['deadlines'], indent=1)}\n"
                        f"Events:\n{json.dumps(schedule['events'], indent=1)}\nDeadline conflicts: {schedule['conflicts']}\n"
                        f"Calendar changes ({'APPLIED' if apply else 'PROPOSED, not applied yet: say so'}):\n{json.dumps(cal_facts, indent=1)}\n"
                        f"Calendar overlaps found by code: {cal_plan['conflicts']}\n"
                        f"Message counts by category: {dict(cats)}", label="digest", schema=None,
                        reasoning="low")   # writing from clean facts needs little thinking

    (OUT / "labels.json").write_text(json.dumps(labels, indent=2))
    (OUT / "items.json").write_text(json.dumps(items, indent=2))
    (OUT / "schedule.json").write_text(json.dumps(schedule, indent=2))
    (OUT / "digest.md").write_text(digest.strip() + "\n")
    path = tracer.save(RUNS, extra={"plan": plan, "review": review, "calendar_plan": cal_plan})
    s = tracer.summary()
    print(f"\nDone in {s['wall_seconds']}s: {s['llm_calls']} model calls, {s['tool_calls']} tool calls, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\n"
          f"Calendar: {dict(counts)} ({'applied' if apply else 'plan only; run with --apply to make these changes'})\n"
          f"Outputs: {OUT}\nTrace: {path}")


if __name__ == "__main__":
    main()
