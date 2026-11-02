"""Brief + rough notes + a small data table -> a slide deck (HTML + inline SVG), exported to PDF.

  master  (Qwen3.8 27B)            plans the storyline, one message per slide (no thinking);
                                   writes the presenter's Q&A prep (thinks a little)
  group   (Gemma 4 12B)            deck lead: reviews flow and consistency from compact JSON, sends back <= 2 slides
  workers (Gemma 4 12B)            one writer per slide: title, bullets, speaker notes, chart or diagram choice;
                                   a slide that fails its checks twice escalates to the 27B
  code                             layout, charts and diagrams, word/bullet/notes limits, number tracing,
                                   overflow (rendered in headless Chrome), PDF export and page count (slide_tools.py)
  human                            rehearses, presents and submits: nothing is uploaded or sent

Run:     .venv/bin/python workflows/slide_deck.py
Export:  .venv/bin/python workflows/slide_tools.py export outputs/slides/deck.html outputs/slides/deck.pdf
"""
import dataclasses
import json
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentkit import TIER_MODELS, Agent, AgentCard, Group, Tracer  # noqa: E402
from workflows import slide_tools as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sample_data" / "slides", ROOT / "outputs" / "slides"
RUNS = ROOT / "results" / "runs"
# Escalate to the first tier above the worker that runs a different (bigger) model.
ESCALATE_MODEL = next((m for m in (TIER_MODELS["group"], TIER_MODELS["master"]) if m != TIER_MODELS["worker"]),
                      TIER_MODELS["master"])
TITLE_WORDS, BULLET_WORDS = 7, 12        # what we ask for; the checks allow two words of slack
LIMITS = ("minutes", "min_slides", "max_slides", "max_bullets", "max_words")

STR, INT = {"type": "string"}, {"type": "integer"}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "deck_title": STR, "audience": STR, **{k: INT for k in LIMITS},
    "slides": {"type": "array", "minItems": 3, "maxItems": 12, "items": {
        "type": "object", "additionalProperties": False, "properties": {
            "message": STR, "visual": {"type": "string", "enum": ["none", "chart", "diagram"]},
            "covers": {"type": "array", "items": INT}},
        "required": ["message", "visual", "covers"]}}},
    "required": ["deck_title", "audience", *LIMITS, "slides"]}


# ---------------------------------------------------------------- reading the inputs (code)
def key_messages(brief):
    """The brief's numbered 'Key messages' list (a parse, not a judgment)."""
    parts = re.split(r"(?im)^#+\s*key messages.*$", brief)
    return [m.group(1).strip() for m in re.finditer(r"(?m)^\s*\d+\.\s+(.+)$", parts[-1])]


def clean_notes(text):
    """Struck-out text (~~35 of 40~~) is a miscount the student crossed out: workers never see it."""
    return re.sub(r"~~.*?~~", "", text)


def check_plan(plan, brief, n_keys):
    """Code checks the master's plan: limits copied from the brief, slide count in range, every key message covered."""
    nums = set(st.numbers_in(brief))
    problems = [f"{k}={plan[k]} is not a number in the brief" for k in LIMITS if str(plan[k]) not in nums]
    n = len(plan["slides"])
    if not plan["min_slides"] <= n <= plan["max_slides"]:
        problems.append(f"the plan has {n} slides; the brief allows {plan['min_slides']} to {plan['max_slides']}")
    if plan["slides"] and plan["slides"][0]["visual"] != "none":
        problems.append("slide 1 is the title slide: visual must be 'none'")
    covered = {c for s in plan["slides"] for c in s["covers"]}
    missing = [k for k in range(1, n_keys + 1) if k not in covered]
    if missing:
        problems.append(f"key messages {missing} are not covered by any slide")
    for kind in ("chart", "diagram"):
        if kind in brief.lower() and not any(s["visual"] == kind for s in plan["slides"]):
            problems.append(f"the brief asks for a {kind}: give one slide visual '{kind}'")
    return "; ".join(problems) or None


# ---------------------------------------------------------------- one slide: schema, assembly, checks (code)
def slide_schema(base, visual, table, max_bullets, notes_words):
    """Per-slide JSON schema. The chart's columns and highlight are enums of the REAL columns and rows,
    so the model cannot name a column that does not exist. Every string has a maxLength, so a model stuck
    in a repetition loop stops after a few hundred characters instead of burning its whole token budget."""
    props = {**base["properties"], "bullets": {**base["properties"]["bullets"], "maxItems": max_bullets},
             "notes": {"type": "string", "maxLength": notes_words * 8}}
    req = list(base["required"])
    if visual == "chart":
        props["chart"] = {"type": "object", "additionalProperties": False, "properties": {
            "type": {"type": "string", "enum": ["bar", "hbar", "grouped_bar"]},
            "x": {"type": "string", "enum": table["labels"]},
            "y": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "string", "enum": table["numeric"]}},
            "highlight": {"type": "string", "enum": [""] + [r[table["labels"][0]] for r in table["rows"]]},
            "title": {"type": "string", "maxLength": 60}}, "required": ["type", "x", "y", "highlight", "title"]}
    elif visual == "diagram":
        props["diagram"] = {"type": "object", "additionalProperties": False, "properties": {
            "nodes": {"type": "array", "minItems": 2, "maxItems": 6, "items": {
                "type": "object", "additionalProperties": False, "properties": {
                    "id": {"type": "string", "maxLength": 12}, "label": {"type": "string", "maxLength": 40}},
                "required": ["id", "label"]}},
            "edges": {"type": "array", "maxItems": 8, "items": {
                "type": "object", "additionalProperties": False, "properties": {
                    "from": STR, "to": STR, "label": {"type": "string", "maxLength": 16}},
                "required": ["from", "to", "label"]}}}, "required": ["nodes", "edges"]}
    req += [k for k in ("chart", "diagram") if k in props]
    return {**base, "properties": props, "required": req}


def assemble(job, r, table):
    """Worker JSON -> slide spec. Code validates the visual choice against the data."""
    # Small models sometimes append a stage direction like "[Pause]" to the notes. Deleting bracketed text is
    # a cleanup with one right answer, so code does it instead of spending a retry.
    notes = re.sub(r"[ \t]*\[[^\]]*\][ \t]*", " ", str(r.get("notes", ""))).strip()
    s = {"n": job["n"], "message": job["message"], "title": str(r.get("title", "")).strip(),
         "bullets": [b.strip() for b in r.get("bullets", []) if str(b).strip()], "notes": notes,
         "visual": {"kind": "none"}}
    problems = []
    if job["visual"] == "chart":
        chart, problems = st.normalize_chart(r.get("chart") or {}, table)
        s["visual"] = {"kind": "chart", "chart": chart}
    elif job["visual"] == "diagram":
        diagram, problems = st.normalize_diagram(r.get("diagram") or {})
        s["visual"] = {"kind": "diagram", "diagram": diagram}
    return s, problems


def on_slide(s):
    """The words the audience reads: title, bullets, chart title (chart and diagram labels don't count)."""
    return " ".join([s["title"], *s["bullets"], (s["visual"].get("chart") or {}).get("title", "")])


def all_text(s):
    d = s["visual"].get("diagram") or {}
    return " ".join([on_slide(s), s["notes"], *[n["label"] for n in d.get("nodes", [])],
                     *[e.get("label", "") for e in d.get("edges", [])]])


def limit_problems(s, lim):
    p = []
    if st.count_words(s["title"]) > TITLE_WORDS + 2:
        p.append(f"the title has {st.count_words(s['title'])} words; use at most {TITLE_WORDS}")
    long = [b for b in s["bullets"] if st.count_words(b) > BULLET_WORDS + 2]
    if long:
        p.append(f"these bullets are longer than {BULLET_WORDS} words: {long[:2]}")
    if st.count_words(on_slide(s)) > lim["max_words"]:
        p.append(f"the slide has {st.count_words(on_slide(s))} words (title + bullets + chart title); the limit is "
                 f"{lim['max_words']}: cut or shorten bullets")
    nw = st.count_words(s["notes"])
    if nw > lim["notes_max"]:
        p.append(f"the speaker notes have {nw} words, over the limit of {lim['notes_max']}: cut them to "
                 f"{lim['sentences'] - 1} short sentences")
    elif nw < lim["notes_min"]:
        p.append(f"the speaker notes have {nw} words; write {lim['sentences']} full sentences")
    if re.search(r"\*\*|`|^#", on_slide(s) + "\n" + s["notes"], flags=re.M):
        p.append("plain text only: no Markdown")
    if s["n"] > 1 and not s["bullets"] and s["visual"]["kind"] == "none":
        p.append("a slide without a chart or diagram needs 1 to 4 bullets")
    echo = [b for b in s["bullets"] if repeats(b, s["title"])]
    if echo:
        p.append(f"these bullets repeat the title: {echo[:2]}; say something new or drop them")
    if LEAK.search(s["notes"]):
        p.append(f"the speaker notes contain prompt text ({LEAK.search(s['notes']).group(0)!r}): write only what the "
                 "presenter says aloud")
    d = s["visual"].get("diagram") or {}
    coded = [x for x in [n["label"] for n in d.get("nodes", [])] + [e.get("label", "") for e in d.get("edges", [])]
             if "_" in x or re.search(r"[a-z][A-Z]", x)]
    if coded:
        p.append(f"diagram labels must be plain words for the audience, not code: {coded[:3]}")
    if s["n"] == 1 and any(re.match(r"^\w+:\s", b) for b in s["bullets"]):
        p.append("write the subtitle lines without labels (Riley Park, not Presenter: Riley Park)")
    return p


LEAK = re.compile(r"\[[^\]]*\]|next slide'?s message|storyline|^\s*message:", re.I | re.M)


STOP = set("the and for are was were with that this from into than their our your its has have".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if (len(w) > 2 or w.isdigit()) and w not in STOP}


def repeats(bullet, title):
    """A bullet that only restates the title wastes the audience's reading time."""
    b, t = content_words(bullet), content_words(title)
    return bool(b) and len(b & t) >= 0.7 * len(b)


def untraceable(s, allowed):
    return sorted(set(st.numbers_in(all_text(s))) - allowed, key=float)


def overflow_of(s, table):
    """Render this one slide alone in headless Chrome and measure what does not fit."""
    rep = st.overflow_of_html(st.render_deck({"title": "check", "slides": [s]}, table))
    return [{**o, "slide": s["n"]} for o in rep["overflow"]]


def make_check(table, allowed, tracer, counts):
    lock = threading.Lock()

    def bump(kind):
        with lock:
            counts[kind] = counts.get(kind, 0) + 1

    def check(job, r):
        """Code-level critic for one slide. Returns the specific error the writer gets on its retry."""
        s, problems = assemble(job, r, table)
        if problems:
            bump("visual")
        lim = limit_problems(s, job["limits"])
        if lim:
            bump("limits")
        bad = untraceable(s, allowed)
        if bad:
            bump("numbers")
            lim.append(f"these numbers are not in the brief, notes or data: {bad}. Use only numbers written there "
                       "(e.g. '12 of 50', '64%'), exactly as written; never compute new ones")
        problems += lim
        if problems:       # cheap checks first; only a clean slide is worth rendering
            return "; ".join(problems)
        start = time.time()
        over = overflow_of(s, table)
        tracer.tool("code", "overflow_check", f"slide {s['n']}", time.time() - start, not over)
        if over:
            bump("overflow")
            return st.overflow_message(over[0], len(s["bullets"]))
        return None
    return check


def salvage(job, r, table, allowed):
    """Last resort after escalation: code makes the slide pass by removing things, never by adding them."""
    s, _ = assemble(job, r or {}, table)
    lim, fixed = job["limits"], []
    if untraceable({**s, "notes": ""}, allowed):
        keep = [b for b in s["bullets"] if not set(st.numbers_in(b)) - allowed]
        fixed.append(f"dropped {len(s['bullets']) - len(keep)} bullet(s) with untraceable numbers")
        s["bullets"] = keep
    sentences = re.split(r"(?<=[.!?])\s+", s["notes"])
    sentences = [x for x in sentences if not set(st.numbers_in(x)) - allowed]
    while sentences and st.count_words(" ".join(sentences)) > lim["notes_max"]:
        sentences.pop()
    if " ".join(sentences) != s["notes"]:
        fixed.append("trimmed the speaker notes")
        s["notes"] = " ".join(sentences)
    if st.count_words(s["title"]) > TITLE_WORDS + 2 or not s["title"]:   # no title at all: the planned message
        s["title"] = " ".join((s["title"] or job["message"]).split()[:TITLE_WORDS])
        fixed.append("shortened the title")
    if s["visual"]["kind"] == "chart":
        c = s["visual"]["chart"]
        c, bad = st.normalize_chart({**c, "y": c["y"][:1], "title": " ".join(c.get("title", "").split()[:6])}, table)
        if bad:
            s["visual"] = {"kind": "none"}
            fixed.append("dropped an invalid chart")
        else:
            s["visual"]["chart"] = c
    if s["visual"]["kind"] == "diagram" and st.normalize_diagram(s["visual"]["diagram"])[1]:
        s["visual"], fixed = {"kind": "none"}, fixed + ["dropped an invalid diagram"]
    while s["bullets"] and (st.count_words(on_slide(s)) > lim["max_words"] or overflow_of(s, table)):
        s["bullets"].pop()
        fixed.append("dropped the last bullet to fit")
    s["fixed_by_code"] = fixed
    return s


# ---------------------------------------------------------------- the workflow
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracer = Tracer("slides")

    def agent(card_name, **override):
        return Agent(dataclasses.replace(AgentCard.load(card_name), **override), None, tracer)

    master = agent("slides_master")
    team = Group("deck", agent("slides_lead"), {
        "write": agent("slides_writer"),
        "lead": agent("slides_lead"),
        # Escalation: the same card on the next bigger model up the hierarchy.
        "write_escalated": agent("slides_writer", name="slides_writer_escalated", model=ESCALATE_MODEL)}, tracer)

    brief, notes_raw = (DATA / "brief.md").read_text(), (DATA / "notes.md").read_text()
    notes, csv_text = clean_notes(notes_raw), (DATA / "data.csv").read_text()
    table = st.read_table(DATA / "data.csv")
    sources = [DATA / "brief.md", DATA / "notes.md", DATA / "data.csv"]
    allowed = st.source_numbers(sources)          # every number a slide may use
    keys = key_messages(brief)
    tracer.note(f"inputs: {len(keys)} key messages, data.csv {len(table['rows'])} rows x {len(table['columns'])} "
                f"columns, {len(allowed)} traceable numbers")

    # 1. The master plans the storyline from the brief and the table's shape (never the raw notes).
    task = f"PLAN.\n\nbrief.md:\n{brief}\n\ndata.csv:\n{csv_text}"
    plan, error = None, None
    for attempt in range(2):
        plan = master.run(task + (f"\n\nYour previous plan had a problem: {error}\nFix it." if error else ""),
                          label="plan" if not attempt else "plan:retry", schema=PLAN_SCHEMA)
        error = check_plan(plan, brief, len(keys))
        if not error:
            break
        tracer.note(f"plan check: {error}")
    if error:
        sys.exit(f"The plan still fails its checks after a retry: {error}")
    n = len(plan["slides"])
    budget = plan["minutes"] * st.WPM // n        # speaker-notes words per slide that keep the talk on time
    tracer.note(f"plan: {n} slides, {plan['minutes']} min, notes <= {budget} words/slide; "
                + " | ".join(f"{i}:{s['visual']}" for i, s in enumerate(plan["slides"], 1)))

    # 2. One writer per slide, in parallel. Code checks limits, numbers and the visual, then renders the slide in
    #    Chrome to measure overflow; a failure goes back with the specific error, then to the 27B.
    base = AgentCard.load("slides_writer").output_schema
    storyline = "\n".join(f"{i}. [{s['visual']}] {s['message']}" for i, s in enumerate(plan["slides"], 1))
    jobs = []
    # Notes: a small model cannot count words, but it can count sentences. So we ask for sentences
    # (about 20 words each) and code enforces the word budget.
    for i, p in enumerate(plan["slides"], 1):
        max_b = {"none": plan["max_bullets"], "chart": 3, "diagram": 2}[p["visual"]] if i > 1 else 3   # room for a visual
        lim = {"max_words": plan["max_words"], "notes_min": 15 if i == 1 else 30, "notes_max": budget,
               "sentences": 3 if i == 1 else max(2, budget // 20)}
        nxt = plan["slides"][i]["message"] if i < n else "(this is the last slide)"
        what = {"chart": "a chart (fill in `chart`)", "diagram": "a diagram (fill in `diagram`)",
                "none": "no visual (bullets only)" if i > 1 else "the title slide"}[p["visual"]]
        jobs.append({"label": f"slide {i}", "n": i, "message": p["message"], "visual": p["visual"], "limits": lim,
                     "schema": slide_schema(base, p["visual"], table, max_b, budget),
                     "task": f"Deck: \"{plan['deck_title']}\" for {plan['audience']}. {n} slides, {plan['minutes']} "
                             f"minutes.\nStoryline:\n{storyline}\n\nYOUR SLIDE: {i} of {n}, with {what}.\n"
                             f"Message: {p['message']}\nThe next slide will say: {nxt}\n"
                             f"Limits: title at most {TITLE_WORDS} words; at most {max_b} bullets, each at most "
                             f"{BULLET_WORDS} words; at most {plan['max_words']} words on the slide in total (title, "
                             f"bullets and chart title); speaker notes {lim['sentences']} short sentences of about 15 "
                             f"words each (never more than {budget} words).\n\n"
                             f"brief.md:\n{brief}\n\nnotes.md (the presenter's notes):\n{notes}\n\n"
                             f"data.csv:\n{csv_text}"})
    counts = {}
    check = make_check(table, allowed, tracer, counts)
    results = team.fan_out("write", jobs, retries=1, check=check)
    failed = [k for k, r in enumerate(results) if "_failed" in r]
    for k in failed:
        tracer.note(f"escalating {jobs[k]['label']} to {ESCALATE_MODEL}: {results[k]['_failed']}")
    for k, r in zip(failed, team.fan_out("write_escalated", [jobs[k] for k in failed], retries=1, check=check)):
        results[k] = r
    slides = []
    for job, r in zip(jobs, results):
        if "_failed" in r:
            s = salvage(job, r.get("_last"), table, allowed)
            tracer.note(f"{job['label']}: fixed by code after escalation: {s['fixed_by_code']}")
        else:
            s = assemble(job, r, table)[0]
        slides.append(s)

    # 3. The deck lead reviews compact JSON (never the notes, the data or the rendered slides).
    def describe(s):
        v = s["visual"]
        if v["kind"] == "chart":
            c = v["chart"]
            return (f"{c['type']} chart of {', '.join(c['y'])} by {c['x']}" + (f", highlighting {c['highlight']}"
                                                                              if c.get("highlight") else ""))
        if v["kind"] == "diagram":
            label = {x["id"]: x["label"] for x in v["diagram"]["nodes"]}
            return "diagram: " + "; ".join(f"{label[e['from']]} -> {label[e['to']]}" for e in v["diagram"]["edges"])
        return "none"
    compact = [{"slide": s["n"], "message": s["message"], "title": s["title"], "bullets": s["bullets"],
                "visual": describe(s), "notes_start": re.split(r"(?<=[.!?])\s", s["notes"])[0][:160],
                "notes_words": st.count_words(s["notes"])} for s in slides]

    def check_lead(job, rv):
        bad = [x["slide"] for x in rv.get("revisions", []) if not 1 <= x["slide"] <= n]
        bad += [k for k in rv.get("missing", []) if not 1 <= k <= len(keys)]
        return f"slide or key-message numbers out of range: {bad}" if bad else None
    review = team.fan_out("lead", [{"label": "review deck", "task": "Key messages from the brief:\n" + "\n".join(
        f"{k}. {m}" for k, m in enumerate(keys, 1)) + "\n\nDeck:\n" + json.dumps(compact, indent=1, ensure_ascii=False)}],
        retries=1, check=check_lead)[0]
    if "_failed" in review:
        review = {"missing": [], "revisions": [], "flow": f"(review failed: {review['_failed']})"}
    tracer.note(f"review: missing {review['missing']}, revisions {[x['slide'] for x in review['revisions']]}")

    # 4. At most 3 slides go back to their writers with the lead's instruction. A revision must pass the same
    #    checks; if it does not, the slide that already passed stays.
    asks = {}
    for x in review["revisions"]:
        asks.setdefault(x["slide"], []).append(x["instruction"])
    for k in review["missing"]:
        home = next((i for i, p in enumerate(plan["slides"], 1) if k in p["covers"]), None)
        if home:
            asks.setdefault(home, []).append(f"Make the title or a bullet state this key message: {keys[k - 1]}")
    rjobs = [{**jobs[i - 1], "label": f"revise slide {i}", "task": jobs[i - 1]["task"] + "\n\nREVISION. The reviewer "
              f"read the whole deck and asks: {' '.join(a)}\nYour current slide: " + json.dumps(
                  {k: slides[i - 1][k] for k in ("title", "bullets", "notes")}, ensure_ascii=False)
              + "\nReturn the whole slide again with that change; keep what already works."}
             for i, a in sorted(asks.items())[:3]]
    revised = []
    for job, r in zip(rjobs, team.fan_out("write", rjobs, retries=1, check=check)):
        if "_failed" not in r:
            slides[job["n"] - 1] = assemble(job, r, table)[0]
            revised.append(job["n"])
    tracer.note(f"revised slides {revised} of {[j['n'] for j in rjobs]} sent back")

    # 5. Code renders the deck, re-measures it as a whole, exports the PDF and counts its pages.
    for old in OUT.glob("slide*-*.svg"):
        old.unlink()
    spec = {"title": plan["deck_title"], "audience": plan["audience"], "minutes": plan["minutes"],
            "sources": [p.name for p in sources], "slides": slides}
    start = time.time()
    (OUT / "deck.html").write_text(st.render_deck(spec, table))
    for s in slides:
        if s["visual"]["kind"] != "none":
            (OUT / f"slide{s['n']}-{s['visual']['kind']}.svg").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n' + st.visual_svg(s, table))
    pages = st.export_pdf(OUT / "deck.html", OUT / "deck.pdf")
    tracer.tool("code", "render_and_export", f"{n} slides, {pages} pages", time.time() - start, pages == n)
    report = st.check_deck(OUT / "deck.html", sources, DATA / "data.csv", OUT / "deck.pdf", plan["max_words"],
                           plan["max_bullets"], plan["minutes"] * st.WPM)
    (OUT / "deck.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    (OUT / "checks.json").write_text(json.dumps({**report, "check_failures": counts}, indent=2) + "\n")
    if report["problems"]:
        tracer.note(f"final deck check: {report['problems']}")

    # 6. The master writes the one page for the person giving the talk: likely questions and honest answers.
    slide_text = "\n".join(f"{s['n']}. {s['title']}: " + "; ".join(s["bullets"]) for s in slides)
    task = (f"WRITE the Q&A prep. Audience: {plan['audience']}.\nThe slides:\n{slide_text}\n\n"
            f"notes.md (the presenter's notes):\n{notes}\n\ndata.csv:\n{csv_text}")
    qa, error = "", None
    for attempt in range(2):
        qa = master.run(task + (f"\n\nYour previous draft had a problem: {error}\nFix it." if error else ""),
                        label="write Q&A" if not attempt else "write Q&A:retry", schema=None,
                        reasoning="low")   # writing from clean inputs needs little thinking
        bad = sorted(set(st.numbers_in(qa)) - allowed, key=float)
        q = len(re.findall(r"^\s*\*\*Q:", qa, flags=re.M))
        error = "; ".join(x for x in [f"numbers not in the notes or data: {bad}" if bad else "",
                                      f"write exactly 3 questions as '**Q: ...**' lines (found {q})" if q != 3 else "",
                                      "at most 220 words" if st.count_words(qa) > 240 else ""] if x) or None
        if not error:
            break
        tracer.note(f"Q&A check: {error}")
    (OUT / "qa_prep.md").write_text(qa.strip() + "\n")

    path = tracer.save(RUNS, extra={"plan": plan, "review": review, "revised": revised, "check_failures": counts,
                                    "escalated": [jobs[k]["label"] for k in failed], "pdf_pages": pages,
                                    "final_problems": report["problems"]})
    s = tracer.summary()
    print(f"\nDone in {s['wall_seconds']}s: {s['llm_calls']} model calls for {n} slides, {pages}-page PDF, "
          f"check failures {counts or 'none'}, escalated {len(failed)}, revised {revised}, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\n"
          f"Final checks: {report['problems'] or 'all pass'}\nOutputs: {OUT}\nTrace: {path}\n"
          "Nothing was submitted: read deck.pdf, rehearse with the notes (press n in deck.html), then upload it yourself.")


if __name__ == "__main__":
    main()
