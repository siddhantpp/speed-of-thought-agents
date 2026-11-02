"""Slide decks + lecture transcripts -> notes for every class today, plus a one-page "today".

  master  (Qwen3.8 27B)            plans from slide titles (no thinking); writes today.md (thinks a little)
  groups  (Gemma 4 12B)            one class lead per class: slide-vs-speech conflicts, announcements, review
  workers (Gemma 4 12B, vision)    slide reader (one image per call); note writer (one slide + only its speech);
                                   a reply that fails its check twice escalates to the 27B
  code                             WebVTT parsing, embedding + dynamic-programming alignment of speech to slides,
                                   verbatim-quote and wrong-value checks, date resolution, Markdown and JSON

Run:  .venv/bin/python workflows/class_notes.py
"""
import dataclasses
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentkit import TIER_MODELS, Agent, AgentCard, Group, Tracer  # noqa: E402
from agentkit.llm import embed  # noqa: E402
from agentkit.tools import resolve_date  # noqa: E402
from workflows.class_notes_tools import (align, find_quote_time, norm, parse_vtt, quote_in, quote_mismatch,  # noqa: E402
                                         runs, short_ts, similarity, slide_files, spoken_form, time_ranges, tokens,
                                         windows)

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sample_data" / "classes", ROOT / "outputs" / "class_notes"
RUNS = ROOT / "results" / "runs"
# Escalate to the first tier above the worker that runs a different (bigger) model.
ESCALATE_MODEL = next((m for m in (TIER_MODELS["group"], TIER_MODELS["master"]) if m != TIER_MODELS["worker"]),
                      TIER_MODELS["master"])

STR = {"type": "string"}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "classes": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "id": STR, "focus": {"type": "array", "items": STR}, "note_instruction": STR},
        "required": ["id", "focus", "note_instruction"]}}}, "required": ["classes"]}


# ---------------------------------------------------------------- code-level critics
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")
BARE_NUMBER = re.compile(r"^\s*[-+]?\d[\d.,%]*\s*$")


def strings(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    return [s for v in obj for s in strings(v)] if isinstance(obj, list) else []


def garbled(r):
    """Constrained JSON decoding sometimes turns symbols like β into control characters."""
    return ("some text contains control characters (a garbled symbol): write Greek letters and math symbols "
            "as plain words or ASCII, e.g. theta0 + theta2*z1, alpha") if any(CONTROL.search(x) for x in strings(r)) else None


def check_reading(job, r):
    """A slide reading must have a title, and every chart/plot/table must be described, with its numbers."""
    v = r.get("visual") or {}
    problems = [garbled(r)] if garbled(r) else []
    if not r.get("title", "").strip():
        problems.append("title is empty: copy the slide's title exactly")
    if not r.get("text_lines") and v.get("kind", "none") == "none":
        problems.append("text_lines is empty and visual.kind is none: copy every line of text on the slide")
    if v.get("kind") in ("chart", "plot", "table", "diagram") and len(v.get("description", "").split()) < 8:
        problems.append(f"visual.description is too short: say what the {v['kind']} shows and what you can see in it")
    if v.get("kind") in ("chart", "table") and not v.get("values"):
        problems.append("visual.values is empty: list every labeled number with its label")
    bare = [x for x in v.get("values", []) if BARE_NUMBER.match(x)]
    if v.get("kind") == "chart" and len(bare) > len(v.get("values", [])) / 2:
        problems.append(f"visual.values {bare[:3]} have no labels: write what each number is, e.g. "
                        "'Quicksort at n=1000: 13.7'")
    if len(r.get("text_lines", [])) > 40 or any(len(x) > 300 for x in r.get("text_lines", [])):
        problems.append("text_lines is too long: copy each printed line once")
    return "; ".join(problems) or None


def check_notes(job, r):
    """Every claim about the speech must be backed by words copied exactly from THIS slide's transcript lines."""
    hay, problems = job["haystack"], [garbled(r)] if garbled(r) else []
    bad = [x["quote"] for x in r.get("lecturer_added", []) if not quote_in(x["quote"], hay)]
    printed = job["slide_text"]   # an announcement may quote the slide instead of the speech
    bad += [a["quote"] for a in r.get("announcements", []) if a.get("quote") and not quote_in(a["quote"], hay)
            and not quote_in(a["quote"], printed)]
    if bad:   # say exactly where each quote stops matching, so a small model can fix it
        problems.append("quotes must be 5-30 words copied exactly from the transcript lines: "
                        + "; ".join(quote_mismatch(q, hay) for q in bad[:3]) + ". Fix each quote or drop the item")
    fix = r.get("correction") or {}
    if fix.get("has_correction"):
        if not quote_in(fix.get("quote", ""), hay):
            problems.append("correction.quote: " + quote_mismatch(fix.get("quote", ""), hay))
        wrong = norm(fix.get("slide_said", ""))
        if not wrong or not fix.get("lecturer_said", "").strip():
            problems.append("a correction needs both slide_said and lecturer_said")
        elif any(wrong in norm(p) for p in r.get("points", [])):
            problems.append(f"points repeat the slide's wrong value {fix['slide_said']!r} as a fact; "
                            f"use the corrected value ({fix['lecturer_said']!r})")
    if r.get("skipped") and not quote_in(r.get("skip_quote", ""), job["own"]):   # not a context line
        problems.append("skipped is true but skip_quote is not the lecturer's exact words, said while THIS slide "
                        "was up, saying they skip it (context lines belong to other slides)")
    if not r.get("points") and not r.get("skipped"):
        problems.append("points is empty")
    return "; ".join(problems) or None


def salvage_notes(r, hay, printed, own):
    """Last resort after escalation: keep only what code can verify."""
    r = dict(r)
    r["lecturer_added"] = [x for x in r.get("lecturer_added", []) if quote_in(x["quote"], hay)]
    r["announcements"] = [a for a in r.get("announcements", []) if not a.get("quote") or quote_in(a["quote"], hay)
                          or quote_in(a["quote"], printed)]
    if not quote_in((r.get("correction") or {}).get("quote", ""), hay):
        r["correction"] = {"has_correction": False, "slide_said": "", "lecturer_said": "", "quote": ""}
    if r.get("skipped") and not quote_in(r.get("skip_quote", ""), own):
        r["skipped"] = False
    return r


def check_lead(job, r):
    n, ids = job["n_slides"], job["announcement_ids"]
    problems = [f"conflict slide {c['slide']} is not a slide number 1..{n}" for c in r.get("conflicts", [])
                if not 1 <= c["slide"] <= n]
    unknown = [i for i in r.get("keep_announcements", []) if i not in ids]
    if unknown:
        problems.append(f"keep_announcements has ids {unknown} that are not in the list {sorted(ids)}")
    if not r.get("summary", "").strip() or not r.get("review"):
        problems.append("summary and review must not be empty")
    return "; ".join(problems) or None


# ---------------------------------------------------------------- small formatting helpers
def slide_text(reading):
    """What the aligner compares speech against: title, printed lines, and what the chart shows."""
    v = reading.get("visual") or {}
    return " ".join([reading.get("title", ""), *reading.get("text_lines", []), v.get("description", ""),
                     *v.get("values", [])])


def lecture_title(reading, course):
    """The title slide's own title: the first printed line that is not the course-name line."""
    code = norm(course.split(":")[0])
    lines = [reading.get("title", ""), *reading.get("text_lines", [])]
    return next((x for x in lines if x.strip() and not norm(x).startswith(code)), lines[0])


def mentions_date(text, label):
    """Does `text` name the day in `label` ('Mon Nov 9, 11:59 pm'), as 'Nov 9', 'November 9th' or '11/9'?"""
    d = datetime.strptime(label.split(",")[0][4:] + " 2000", "%b %d %Y")   # any leap year: only month/day matter
    return bool(re.search(rf"\b{d:%b}[a-z]*\.? {d.day}(st|nd|rd|th)?\b|\b{d.month}/{d.day}\b", text, re.I))


DATE_WORDS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "november", "december"}


def mentions_item(text, a):
    """today.md names this announcement: a line with its date also has one of its own longer words."""
    words = {w for w in tokens(a["text"]) if len(w) >= 5 and w not in DATE_WORDS and not w[0].isdigit()}
    return any(mentions_date(line, a["date"]) and (not words or words & set(tokens(line)))
               for line in text.split("\n"))


def when_label(iso, timed=True):
    d = datetime.fromisoformat(iso)
    label = f"{d:%a} {d:%b} {d.day}"
    return label + (f", {d.hour % 12 or 12}:{d:%M} {'am' if d.hour < 12 else 'pm'}" if timed else "")


def transcript_block(cues, labels, slide):
    """Only this slide's speech (every visit), with one line of context on each side, the text the
    quote check searches, and the slide's own speech alone. Timestamps are minutes:seconds into the recording."""
    lines, hay, own = [], [], []
    for k, (lab, i, j) in enumerate(r for r in runs(labels) if r[0] == slide):
        if k:
            lines.append("(later, the lecturer came back to this slide)")
        lo, hi = max(0, i - 1), min(len(cues) - 1, j + 1)
        for t in range(lo, hi + 1):
            c = cues[t]
            tag = "(context) " if t < i or t > j else ""
            lines.append(f"{tag}[{short_ts(c['start'])}] {c['speaker']}: {c['text']}")
            hay.append(c["text"])
            if not tag:
                own.append(c["text"])
    return ("\n".join(lines) or "(No speech was matched to this slide.)"), " ".join(hay), " ".join(own)


def render_class(c):
    """Per-class notes in Markdown, built by code from checked JSON (no model writes this file)."""
    when = datetime.fromisoformat(c["start"])
    md = [f"# {c['course']}: {c['session']}", f"## {c['title']}", "",
          f"{when:%a %b} {when.day}, {when.hour % 12 or 12}:{when:%M} {'am' if when.hour < 12 else 'pm'} · "
          f"{c['instructor']} · {len(c['slides'])} slides", "", "## Summary", "", c["summary"], ""]
    if c["corrections"]:
        md += ["## Corrections (the slide was wrong; use the corrected fact)", ""]
        for x in c["corrections"]:
            said = f' The lecturer: "{x["quote"]}"' if x.get("quote") else " (flagged by the class lead)"
            md.append(f"- **Slide {x['slide']}** said: {x['slide_said']}. **Corrected:** {x['lecturer_said']}.{said}")
        md.append("")
    if c["announcements"]:
        md += ["## Announcements", ""] + [f"- **{a['when']}**: {a['text']}" if a["when"] else f"- {a['text']}"
                                          for a in c["announcements"]] + [""]
    if c["exam_hints"]:
        md += ["## Exam and homework hints (said aloud)", ""] + [f"- {h}" for h in c["exam_hints"]] + [""]
    md += ["## Slide by slide", ""]
    for s in c["slides"]:
        span = ", ".join(f"{short_ts(a)}-{short_ts(b)}" for a, b in s["_seconds"]) or "no speech"
        md.append(f"### Slide {s['slide']}: {s['title']} ({span})")
        if s["skipped"]:
            md += ["", f'*Skipped in class* ("{s["skip_quote"]}").', ""]
            continue
        md += [""] + [f"- {p}" for p in s["points"]]
        for x in s["lecturer_added"]:
            at = f" [{short_ts(x['_t'])}]" if x.get("_t") is not None else ""
            md.append(f'- *Lecturer added ({x["kind"].replace("_", " ")}):* {x["point"]} "{x["quote"]}"{at}')
        md.append("")
    if c["skipped_slides"]:
        md += ["## Skipped slides", "", "Skipped in class: " + ", ".join(f"slide {n}" for n in c["skipped_slides"]), ""]
    md += ["## Review", ""] + [f"- {r}" for r in c["review"]]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- the workflow
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracer = Tracer("class_notes")

    def agent(card_name, **override):
        return Agent(dataclasses.replace(AgentCard.load(card_name), **override), None, tracer)

    master = agent("notes_master")
    team = Group("notes", agent("notes_class_lead"), {
        "read": agent("notes_slide_reader"),
        "plot": agent("notes_plot_reader"),
        "write": agent("notes_slide_writer"),
        "lead": agent("notes_class_lead"),
        # Escalation: the same card on the next bigger model up the hierarchy.
        "read_escalated": agent("notes_slide_reader", name="notes_slide_reader_escalated", model=ESCALATE_MODEL),
        "plot_escalated": agent("notes_plot_reader", name="notes_plot_reader_escalated", model=ESCALATE_MODEL),
        "write_escalated": agent("notes_slide_writer", name="notes_slide_writer_escalated", model=ESCALATE_MODEL),
        "lead_escalated": agent("notes_class_lead", name="notes_class_lead_escalated", model=ESCALATE_MODEL)},
        tracer)

    def checked(worker, jobs, check):
        """Fan out with a code check and one retry; whatever still fails goes to the bigger model."""
        results = team.fan_out(worker, jobs, retries=1, check=check)
        failed = [i for i, r in enumerate(results) if "_failed" in r]
        for i in failed:
            tracer.note(f"escalating {jobs[i]['label']} to {ESCALATE_MODEL}: {results[i]['_failed']}")
        for i, r in zip(failed, team.fan_out(worker + "_escalated", [jobs[i] for i in failed], retries=1, check=check)):
            results[i] = r
        return results

    schedule = json.loads((DATA / "schedule.json").read_text())
    start = time.time()
    classes = [{**c, "files": slide_files(DATA / c["slides"]), "cues": parse_vtt((DATA / c["transcript"]).read_text())}
               for c in schedule["classes"]]
    tracer.tool("code", "parse_vtt", f"{sum(len(c['cues']) for c in classes)} cues", time.time() - start, True)

    # 1. Vision workers read every slide of every class, one image per call; code checks each reading.
    jobs = [{"label": f"{c['id']} read s{n}", "images": [p.read_bytes()],
             "task": f"{c['course']}, {c['session']}. Slide {n} of {len(c['files'])}. Read it."}
            for c in classes for n, p in enumerate(c["files"], 1)]
    readings = iter(checked("read", jobs, check_reading))
    blank = {"title": "", "text_lines": [], "visual": {"kind": "none", "description": "", "values": []}}
    for c in classes:   # a reading that failed even after escalation keeps its last attempt
        c["readings"] = [r if "_failed" not in r else blank | (r.get("_last") or {}) for r in
                         (next(readings) for _ in c["files"])]

    # 1b. A plot gets one narrow follow-up question; code writes what it shows (open descriptions miss the shape).
    plots = [(c, n, r) for c in classes for n, r in enumerate(c["readings"], 1) if (r.get("visual") or {}).get("kind") == "plot"]
    answers = checked("plot", [{"label": f"{c['id']} plot s{n}", "images": [c["files"][n - 1].read_bytes()],
                                "task": "How does the vertical spread of the points change from left to right?"}
                               for c, n, r in plots], lambda job, a: garbled(a))
    for (c, n, r), a in zip(plots, answers):
        if "_failed" not in a:
            r["visual"]["description"] = (f"Scatter plot: the points form a pattern described as '{a['pattern']}'; the "
                                          f"vertical spread is {a['spread_left']} on the left and {a['spread_right']} "
                                          f"on the right.")

    # 2. The master plans from slide titles only (no transcripts, no images).
    outline = "\n".join(f"- {c['id']}: {c['course']}, {c['session']} ({c['instructor']}). Slide titles: "
                        + " | ".join(r.get("title", "") for r in c["readings"]) for c in classes)
    plan = master.run(f"PLAN. Today is {schedule['date']}. Classes:\n{outline}", label="plan", schema=PLAN_SCHEMA)
    guidance = {p["id"]: p for p in plan["classes"]}
    tracer.note("plan: " + "; ".join(f"{k}: {v['note_instruction']}" for k, v in guidance.items()))

    # 3. Code aligns the speech to the slides: local embeddings + a mostly-monotonic DP (class_notes_tools.align).
    for c in classes:
        start = time.time()
        win, texts = windows(c["cues"]), [spoken_form(slide_text(r)) for r in c["readings"]]
        inputs = ["search_query: " + w for w in win] + ["search_document: " + t for t in texts]
        vecs = [v for i in range(0, len(inputs), 64) for v in embed(inputs[i:i + 64])]
        scores = similarity(vecs[:len(win)], vecs[len(win):], win, texts)
        c["labels"] = [s + 1 for s in align(scores, [q["text"] for q in c["cues"]])]   # 1-based slide numbers
        c["ranges"] = time_ranges(c["labels"], c["cues"])
        tracer.tool("code", "align", f"{c['id']} {len(win)} cues", time.time() - start, True)

    # 4. One note writer per slide, seeing that slide's reading and ONLY that slide's speech.
    jobs = []
    for c in classes:
        hint = guidance.get(c["id"], {}).get("note_instruction", "")
        for n, r in enumerate(c["readings"], 1):
            block, hay, own = transcript_block(c["cues"], c["labels"], n)
            jobs.append({"label": f"{c['id']} write s{n}", "haystack": hay, "own": own, "slide_text": slide_text(r),
                         "cls": c["id"], "slide": n,
                         "task": f"Class: {c['course']}, {c['session']} ({schedule['date']}). Slide {n} of "
                                 f"{len(c['readings'])}.\nNote from the master: {hint}\n\nSLIDE (read by a vision "
                                 f"model):\n{json.dumps(r, ensure_ascii=False)}\n\nTRANSCRIPT while this slide was on "
                                 f"screen (minutes:seconds into the recording):\n{block}"})
    notes = checked("write", jobs, check_notes)
    by_class = {c["id"]: [] for c in classes}
    for job, r in zip(jobs, notes):
        if "_failed" in r:
            tracer.note(f"{job['label']}: keeping only the verified parts of the last attempt")
            r = salvage_notes(r.get("_last") or {"points": []}, job["haystack"], job["slide_text"], job["own"])
        by_class[job["cls"]].append(r)

    # 5. Code resolves every announced date against the class's own start time (not "today"), drops remarks
    #    with no date at all, and merges same-day items that share most of their words.
    for c in classes:
        c["notes"], c["announced"] = by_class[c["id"]], []
        for n, r in enumerate(c["notes"], 1):
            for a in r.get("announcements", []):
                got = resolve_date(a.get("when") or a["text"], c["start"])
                if "iso" not in got:
                    got = resolve_date(a["text"], c["start"])
                if "iso" not in got:
                    tracer.note(f"{c['id']} s{n}: dropped undated announcement {a['text'][:60]!r}")
                    continue
                item = {"slide": n, "text": a["text"], "due": got["iso"],
                        "when": when_label(got["iso"], "no time given" not in got["interpretation"])}
                words = set(tokens(a["text"]))
                same = next((b for b in c["announced"] if b["due"][:10] == item["due"][:10] and
                             len(words & set(tokens(b["text"]))) >= 0.3 * len(words | set(tokens(b["text"])))), None)
                if same:   # keep the longer wording and the more precise time
                    if len(item["text"]) > len(same["text"]):
                        same["text"] = item["text"]
                    if same["due"].endswith("23:59:00-05:00") and not item["due"].endswith("23:59:00-05:00"):
                        same["due"], same["when"] = item["due"], item["when"]
                    continue
                c["announced"].append({"id": len(c["announced"]) + 1, **item})

    # 6. Each class lead reviews compact JSON: never the images, never the whole transcript.
    def compact(c):
        return [{"slide": n, "title": r.get("title", ""), "printed": [x[:120] for x in r.get("text_lines", [])[:6]],
                 "points": w.get("points", []), "lecturer_added": [x["point"] for x in w.get("lecturer_added", [])],
                 "correction": w["correction"] if (w.get("correction") or {}).get("has_correction") else None,
                 "skipped": bool(w.get("skipped"))} for n, (r, w) in enumerate(zip(c["readings"], c["notes"]), 1)]
    ljobs = [{"label": f"{c['id']} lead", "n_slides": len(c["readings"]), "announcement_ids": [a["id"] for a in c["announced"]],
              "task": f"Class: {c['course']}, {c['session']} ({c['instructor']}), {len(c['readings'])} slides.\n"
                      f"Slides:\n{json.dumps(compact(c), indent=1, ensure_ascii=False)}\n\nAnnouncements (id. text [date]):\n"
                      + ("\n".join(f"{a['id']}. {a['text']} [{a['when'] or 'no date'}]" for a in c["announced"]) or "(none)")}
             for c in classes]
    reviews = checked("lead", ljobs, check_lead)

    # 7. Code assembles each class's notes (Markdown + JSON) from the checked pieces.
    result = {"date": schedule["date"], "student": schedule["student"], "classes": []}
    for c, rv in zip(classes, reviews):
        if "_failed" in rv:
            rv = {"summary": "", "conflicts": [], "keep_announcements": [a["id"] for a in c["announced"]],
                  "review": [], "exam_hints": []} | (rv.get("_last") or {})
        corrections = [{"slide": n, "slide_said": w["correction"]["slide_said"], "lecturer_said": w["correction"]["lecturer_said"],
                        "quote": w["correction"]["quote"], "source": "lecturer"}
                       for n, w in enumerate(c["notes"], 1) if (w.get("correction") or {}).get("has_correction")]
        corrections += [{**x, "quote": "", "source": "class lead"} for x in rv.get("conflicts", [])
                        if 1 <= x["slide"] <= len(c["notes"]) and x["slide"] not in {y["slide"] for y in corrections}]
        keep = set(rv.get("keep_announcements") or [a["id"] for a in c["announced"]])
        announced = sorted((a for a in c["announced"] if a["id"] in keep), key=lambda a: a["due"] or "9999")
        slides = []
        for n, (r, w) in enumerate(zip(c["readings"], c["notes"]), 1):
            secs = [(c["cues"][i]["start"], c["cues"][j]["end"]) for lab, i, j in runs(c["labels"]) if lab == n]
            slides.append({"slide": n, "title": r.get("title", ""), "skipped": bool(w.get("skipped")),
                           "skip_quote": w.get("skip_quote", ""), "time_ranges": c["ranges"].get(n, []), "_seconds": secs,
                           "points": w.get("points", []),
                           "lecturer_added": [{**x, "_t": find_quote_time(x["quote"], c["cues"])}
                                              for x in w.get("lecturer_added", [])]})
        entry = {"id": c["id"], "course": c["course"], "session": c["session"], "instructor": c["instructor"],
                 "start": c["start"], "title": lecture_title(c["readings"][0], c["course"]),
                 "summary": rv.get("summary", ""),
                 "slides": slides, "corrections": corrections,
                 "announcements": [{"text": a["text"], "when": a["when"], "due": a["due"], "slide": a["slide"]} for a in announced],
                 "skipped_slides": [s["slide"] for s in slides if s["skipped"]],
                 "exam_hints": rv.get("exam_hints", []), "review": rv.get("review", [])}
        (OUT / f"{c['id']}.md").write_text(render_class(entry))
        result["classes"].append(entry)

    # 8. The master writes the one page the student reads first, from the leads' results only.
    day = [{"time": when_label(e["start"]).split(", ")[1], "course": e["course"], "session": e["session"],
            "title": e["title"], "summary": e["summary"],
            "corrections": [{"slide": x["slide"], "slide_said": x["slide_said"], "corrected": x["lecturer_said"]}
                            for x in e["corrections"]],
            "review": e["review"], "exam_hints": e["exam_hints"], "skipped_slides": e["skipped_slides"]}
           for e in result["classes"]]
    dated = sorted(({"course": e["course"].split(":")[0], "text": a["text"], "date": a["when"], "_due": a["due"]}
                    for e in result["classes"] for a in e["announcements"]), key=lambda a: a["_due"] or "9999")
    task = (f"WRITE today.md. Today is {datetime.fromisoformat(classes[0]['start']):%A, %B} "
            f"{datetime.fromisoformat(classes[0]['start']).day}, {schedule['date'][:4]}; the student is "
            f"{schedule['student']}.\nClasses:\n{json.dumps(day, indent=1, ensure_ascii=False)}\n\n"
            "Announcements, already sorted by date (code resolved every date; copy them exactly):\n"
            + "\n".join(f"- [{a['date'] or 'no date'}] {a['course']}: {a['text']}" for a in dated))
    today, error = "", None
    for attempt in range(2):
        ask = task + (f"\n\nYour previous draft had a problem: {error}\nFix it." if error else "")
        today = master.run(ask, label="write today" if not attempt else "write today:retry", schema=None,
                           reasoning="low")   # writing from clean, checked JSON needs little thinking
        missing = [f"[{a['date']}] {a['course']}: {a['text']}" for a in dated if a["date"] and not mentions_item(today, a)]
        missing += [e["course"].split(":")[0] for e in result["classes"] if norm(e["course"].split(":")[0]) not in norm(today)]
        error = ("these announcements or courses are missing (each announcement goes on its own line with its date): "
                 + "; ".join(missing)) if missing else None
        if not error:
            break
        tracer.note(f"today.md check: {error}")

    (OUT / "today.md").write_text(today.strip() + "\n")
    (OUT / "readings.json").write_text(json.dumps({c["id"]: c["readings"] for c in classes}, indent=1, ensure_ascii=False))
    for e in result["classes"]:   # strip the private helper fields before writing the contract file
        for s in e["slides"]:
            s.pop("_seconds", None)
            for x in s["lecturer_added"]:
                t = x.pop("_t", None)
                x["time"] = short_ts(t) if t is not None else ""
    (OUT / "notes.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    path = tracer.save(RUNS, extra={"plan": plan, "alignment": {c["id"]: c["ranges"] for c in classes}})
    s = tracer.summary()
    print(f"\nDone in {s['wall_seconds']}s: {s['llm_calls']} model calls for {len(jobs)} slides in {len(classes)} classes, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\nOutputs: {OUT}\nTrace: {path}")


if __name__ == "__main__":
    main()
