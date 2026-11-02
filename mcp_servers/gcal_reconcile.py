"""Calendar reconciliation in plain code, for workflows/inbox_calendar.py (this is not an MCP server).

The workflow's rule: models read and judge, code does times and matching. Everything here is deterministic
and unit-tested (tests/test_gcal.py):
  end_time        the end of an event from the words a model copied ("until 9:30")
  candidates      which calendar events could be a given inbox item (same message link, or same course + shared word)
  check_decision  the code critic for the cal_reconciler's create / update / delete / no_change answer
  overlaps        new or moved events that collide with something already on the calendar
  plan_markdown   the human-readable change plan
Events are in Google Calendar API v3 format, as returned by gcal_server.py's list_events.
"""
import re
from datetime import datetime, timedelta

DEADLINE_MINUTES = 30               # a deadline is a 30-minute marker starting at the due time
ACTIONS = ("create", "update", "delete", "no_change")


def when(iso):  # code formats dates, so no model ever does calendar arithmetic
    return datetime.fromisoformat(iso).strftime("%a %b %-d, %-I:%M %p").replace(":00 ", " ")


def end_time(it):
    """An item's end. Models copy the words ('until 9:30', '4:00 to 5:15 pm'); code turns them into a time."""
    start = datetime.fromisoformat(it["due"])
    if it["kind"] == "deadline":
        return (start + timedelta(minutes=DEADLINE_MINUTES)).isoformat()
    ends = it.get("ends", "").strip()
    try:   # a model sometimes writes the end as an ISO time instead of the words: fine if it is after the start
        end = datetime.fromisoformat(ends)
        if end.tzinfo and start < end < start + timedelta(hours=12):
            return end.astimezone(start.tzinfo).isoformat()
    except ValueError:
        pass
    words_ = re.sub(r"\d{4}-\d{2}-\d{2}\S*", " ", f"{it.get('phrase', '')} | {ends}".lower())   # drop ISO dates
    t = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"
    m = (re.search(r"\d\s*(?:am|pm)?\s*(?:\bto\b|-|–)\s*" + t, words_)     # a range: "2:00 to 4:00 pm" -> 4:00 pm
         or re.search(r"(?:\buntil\b|\btill\b|\bto\b)\s*" + t, words_)    # "until 9:30"
         or re.match(r"\s*" + t, ends.lower()))                             # just "5:15 pm"
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        hours = [hour + 12] if (m.group(3) or "").startswith("p") else [hour] if m.group(3) else [hour, hour + 12]
        for h in hours:   # "until 9:30" after a 7 pm start means 9:30 pm: the first such time after the start
            end = start.replace(hour=h, minute=minute)
            if end > start:
                return end.isoformat()
    return (start + timedelta(hours=1)).isoformat()   # no end given: one hour


STOP = set("a an and the of for to in on at by with from due is are be this that your new moved only week "
           "course".split())
SYNONYMS = {"hw": ["homework"], "ps": ["problem", "set"], "pset": ["problem", "set"], "oh": ["office", "hours"]}


def words(text, codes=()):
    """Comparable words: 'PS5' -> problem set 5, plurals folded, filler words and course codes dropped."""
    text = text.lower()
    for code in codes:
        text = re.sub(code_pattern(code), " ", text)
    out = set()
    for w in re.findall(r"[a-z]+|\d+", text):
        for v in SYNONYMS.get(w, [w]):
            v = v[:-1] if len(v) > 3 and v.endswith("s") and not v.endswith("ss") else v
            if v not in STOP:
                out.add(v)
    return out


def code_pattern(code):   # "CS 152" matches "CS 152", "cs152" and "#cs152-study-group"
    letters, digits = code.split()
    return rf"(?<![a-z]){letters.lower()}\s?{digits}(?!\d)"


def course_in(texts, codes):
    for t in texts:
        for code in codes:
            if re.search(code_pattern(code), (t or "").lower()):
                return code
    return ""


def links(ev):
    raw = ev.get("extendedProperties", {}).get("private", {}).get("source_message_ids", "")
    return [x for x in raw.split(",") if x]


def is_deadline(ev):
    return ev.get("summary", "").rstrip().lower().endswith(" due")


def same_minute(a, b):
    return abs((datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()) < 60


def candidates(ci, events, codes):
    """Calendar events that could be this item: linked to the same message, or same course, AND a shared word.

    Recurring events (class meetings, weekly office hours) are never candidates. A cancelled item only matches
    an event at its own (original) time, so a cancellation can never remove the event that replaced it."""
    out = []
    for ev in events:
        if ev.get("recurringEventId") or ev.get("recurrence") or "dateTime" not in ev.get("start", {}):
            continue   # recurring and all-day events are never an inbox item
        shared = sorted(w for w in words(ci["task"], codes) & words(ev.get("summary", ""), codes) if w.isalpha())
        linked = bool(set(ci["source_message_ids"]) & set(links(ev)))
        same_course = course_in([ev.get("summary", "")], codes) == ci["course"]
        same_time = same_minute(ev["start"]["dateTime"], ci["start"])
        if shared and (linked or same_course) and (same_time or ci["status"] != "cancelled"):
            out.append({"event_id": ev["id"], "summary": ev.get("summary", ""), "start": ev["start"]["dateTime"],
                        "end": ev["end"]["dateTime"], "linked": linked, "shared_words": shared, "same_time": same_time})
    out.sort(key=lambda c: (not c["linked"], -len(c["shared_words"]), not c["same_time"]))
    return out[:3]


def check_decision(job, d):
    """Code-level critic for one calendar decision. Returns a problem for the model to fix, or None."""
    ci, cands = job["item"], {c["event_id"]: c for c in job["cands"]}
    action, eid = d.get("action"), d.get("event_id", "")
    cancelled = ci["status"] == "cancelled"
    linked = [c for c in cands.values() if c["linked"]]
    if action not in ACTIONS:
        return f"action must be one of {ACTIONS}"
    if action == "delete" and not cancelled:
        return "only a cancelled item may delete an event, and this item is not cancelled"
    if cancelled and action in ("create", "update"):
        return "the item is cancelled: delete its event if one is listed, otherwise no_change"
    if action == "create":
        if linked:
            return f"{linked[0]['event_id']} came from the same message: update it or choose no_change, never a duplicate"
        if any(c["same_time"] for c in cands.values()):
            return "a listed event with a similar title is already at that time: choose no_change"
        return None
    if not cands:
        return None if action == "no_change" else \
            f"nothing is listed, so there is no event_id to use: {'no_change' if cancelled else 'create'}"
    if action == "no_change" and not eid and cancelled:
        return None if not any(c["same_time"] for c in cands.values()) else \
            "the cancelled item is on the calendar at that time: delete it"
    if eid not in cands:
        return f"event_id must be one of {sorted(cands)}"
    if linked and not cands[eid]["linked"]:
        return f"use the event linked to the same message: {linked[0]['event_id']}"
    c = cands[eid]
    if action == "delete" and not c["same_time"]:
        return f"{eid} is at {when(c['start'])}, not at the cancelled item's time {when(ci['start'])}: choose no_change"
    if action == "no_change" and cancelled and c["same_time"]:
        return f"the cancelled item is on the calendar as {eid}: delete it"
    if action == "no_change" and not cancelled and not c["same_time"]:
        return f"{eid} is at {when(c['start'])} but the item is at {when(ci['start'])}: update it"
    if action == "update" and c["same_time"]:
        return f"{eid} is already at {when(c['start'])}: choose no_change"
    return None


def overlaps(changes, events):
    """Conflicts are arithmetic, so code finds them: new or moved events that overlap something else."""
    busy = [(ev.get("summary", ""), ev["start"]["dateTime"], ev["end"]["dateTime"], ev["id"]) for ev in events
            if not is_deadline(ev) and "dateTime" in ev.get("start", {})]
    busy += [(c["summary"], c["start"], c["end"], c.get("event_id") or c["item_id"]) for c in changes
             if c["action"] == "create" and c["kind"] == "event"]
    out = []
    for c in changes:
        if c["action"] not in ("create", "update") or c["kind"] != "event":
            continue
        for name, s, e, ref in busy:
            if ref in (c.get("event_id"), c["item_id"]):
                continue
            if datetime.fromisoformat(s) < datetime.fromisoformat(c["end"]) and \
                    datetime.fromisoformat(c["start"]) < datetime.fromisoformat(e):
                out.append(f"{c['summary']} ({when(c['start'])}) overlaps {name} ({when(s)} to {when(e)})")
    return sorted(set(out))


def plan_markdown(plan):
    verbs = {"create": "Add", "update": "Move", "delete": "DELETE", "no_change": "Leave"}
    md = [f"# Calendar plan ({'APPLIED' if plan['applied'] else 'not applied yet: run with --apply'})\n",
          f"Calendar: {plan['calendar']}  \nRead {plan['window']['time_min']} to {plan['window']['time_max']}.  \n"
          + ", ".join(f"{n} {a}" for a, n in plan["counts"].items()) + "\n",
          "## Deletions (please check these)\n"]
    md += [f"- **{d['summary']}** ({when(d['start'])}), event `{d['event_id']}`: {d['reason']}" for d in plan["deletions"]] \
        or ["- none"]
    md += ["\n## Changes\n", "| Action | Event | When | From messages | Why |", "|---|---|---|---|---|"]
    for c in plan["changes"]:
        at = when(c["start"]) if c.get("start") else ""
        if c["action"] == "update":
            at = f"{when(c['was']['start'])} -> {when(c['start'])}"
        md.append(f"| {verbs.get(c['action'], c['action'])} | {c['summary']} | {at} | "
                  f"{', '.join(c['source_message_ids'])} | {c['reason']} |")
    md += ["\n## Overlaps found by code\n"] + ([f"- {x}" for x in plan["conflicts"]] or ["- none"])
    md += ["\n## For you to do by hand\n"] + ([f"- {c['by_hand']}" for c in plan["changes"] if c.get("by_hand")] or ["- nothing"])
    md += ["\nRecurring events (classes, weekly office hours) are never changed by this workflow."]
    return "\n".join(md) + "\n"
