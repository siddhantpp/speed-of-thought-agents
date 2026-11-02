"""Score a slide deck (local workflow or Claude Code) against sample_data/slides/answer_key.json.

Output contract (same for every harness), in one folder:
  deck.html   one <section class="slide"> per slide, in order, each 1280x720 px (16:9);
              speaker notes inside each section in <aside class="notes"> (hidden on the slide and in the PDF);
              every chart an inline <svg class="chart"> whose bars/points carry data-label (the row's value in
              data.csv's label column), data-series (the data.csv column) and data-value (the number drawn);
              every diagram an inline <svg class="diagram">; slide numbers, if any, in an element with class
              "slide-number"
  deck.pdf    the deck exported with `python3 slide_tools.py export deck.html deck.pdf` (one page per slide)
  deck.json   optional; the local workflow's spec (chart values are read from it only if the SVG has no marks)

What is graded (each pass/fail; `overall` is their mean, with required_messages counting its hit rate):
  slide_count            within the answer key's range
  required_messages      every key message lands on a slide (regex parts, see the key's message_matching)
  numbers_traceable      every number in the slide text, diagram labels and speaker notes is in brief.md,
                         notes.md (struck-out text excluded) or data.csv; chart SVGs are graded by their marks
  chart_values_exact     at least one chart, and every mark equals its data.csv cell
  diagram_present        at least one <svg class="diagram">
  no_overflow            nothing clipped or off the slide when rendered in headless Chrome
  pdf_one_page_per_slide deck.pdf exists and has exactly one page per slide
  speaker_notes          notes on every slide (>= the key's minimum) and the talk fits the time limit
  word_limits            visible words and bullets per slide within the brief's limits
"""
import json
import re
import unicodedata
from pathlib import Path

from workflows import slide_tools as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "sample_data" / "slides"


def plain(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    for a, b in {"’": "'", "‘": "'", "–": "-", "—": "-", "−": "-", " ": " "}.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def part_on(slide_text, groups):
    return all(any(re.search(alt, slide_text) for alt in g) for g in groups)


def marks_from_spec(out):
    """Fallback: chart values recorded in deck.json (the local workflow writes them next to each chart spec)."""
    spec = out / "deck.json"
    if not spec.exists():
        return []
    slides = json.loads(spec.read_text()).get("slides", [])
    return [m for s in slides for m in ((s.get("visual") or {}).get("chart") or {}).get("data", [])]


def score(out):
    out = Path(out)
    key = json.loads((DATA / "answer_key.json").read_text())
    if not (out / "deck.html").exists():
        return {"ok": False, "full_score": False, "overall": 0.0, "error": f"missing {out / 'deck.html'}"}
    slides = st.parse_deck((out / "deck.html").read_text())
    n, lim = len(slides), key["limits"]
    on_slide = [plain(" ".join([s["visible"], s["svg_text"], s["chart_text"]])) for s in slides]
    checks = {}

    lo, hi = key["slide_count"]["min"], key["slide_count"]["max"]
    checks["slide_count"] = {"pass": lo <= n <= hi, "got": n, "want": f"{lo}-{hi}"}

    found, missing = {}, []
    for m in key["required_messages"]:
        where = [next((i for i, t in enumerate(on_slide, 1) if part_on(t, part)), None) for part in m["parts"]]
        if all(where):
            found[m["id"]] = where
        else:
            missing.append({"id": m["id"], "text": m["text"], "parts_missing": [k for k, w in enumerate(where) if not w]})
    checks["required_messages"] = {"pass": not missing, "rate": round(len(found) / len(key["required_messages"]), 3),
                                   "found_on_slides": found, "missing": missing}

    allowed = {x["value"] for x in key["numbers"]}
    stale = {x["value"] for x in key["stale_numbers"]}
    bad, stale_used = {}, []
    for i, s in enumerate(slides, 1):
        nums = set(st.numbers_in(" ".join([s["visible"], s["svg_text"], s["notes"]])))
        if nums - allowed:
            bad[i] = sorted(nums - allowed, key=float)
        stale_used += [f"slide {i}: {v}" for v in sorted(nums & stale)]
    checks["numbers_traceable"] = {"pass": not bad, "untraceable_by_slide": bad, "stale_numbers_used": stale_used}

    table = st.read_table(DATA / "data.csv")
    marks, source = [m for s in slides for m in s["marks"]], "svg"
    if not marks:
        marks, source = marks_from_spec(out), "deck.json"
    mismatches = st.chart_mismatches(marks, table)
    charts = sum(s["charts"] for s in slides)
    checks["chart_values_exact"] = {"pass": charts >= 1 and len(marks) >= 2 and not mismatches, "charts": charts,
                                    "marks_checked": len(marks), "marks_from": source, "mismatches": mismatches}
    checks["diagram_present"] = {"pass": sum(s["diagrams"] for s in slides) >= 1,
                                 "diagrams": sum(s["diagrams"] for s in slides)}

    try:
        over = st.overflow_report(out / "deck.html")["overflow"]
        checks["no_overflow"] = {"pass": not over, "overflow": over}
    except st.ChromeMissing as e:   # graded as unknown, not as a failure
        checks["no_overflow"] = {"pass": None, "note": str(e)}

    pages = st.pdf_pages(out / "deck.pdf") if (out / "deck.pdf").exists() else 0
    checks["pdf_one_page_per_slide"] = {"pass": pages == n and n > 0, "pdf_pages": pages, "slides": n}

    notes = [st.count_words(s["notes"]) for s in slides]
    thin = [i for i, w in enumerate(notes, 1) if w < lim["notes_min_words_per_slide"]]
    checks["speaker_notes"] = {"pass": bool(notes) and not thin and sum(notes) <= lim["notes_max_total_words"],
                               "words_per_slide": notes, "total_words": sum(notes),
                               "max_total_words": lim["notes_max_total_words"], "slides_without_notes": thin}

    words = [st.count_words(s["visible"]) for s in slides]
    over_words = [f"slide {i}: {w} words" for i, w in enumerate(words, 1) if w > lim["max_words_per_slide"]]
    over_words += [f"slide {i}: {s['bullets']} bullets" for i, s in enumerate(slides, 1)
                   if s["bullets"] > lim["max_bullets_per_slide"]]
    checks["word_limits"] = {"pass": not over_words, "words_per_slide": words,
                             "bullets_per_slide": [s["bullets"] for s in slides], "over": over_words}

    graded = {k: c for k, c in checks.items() if c["pass"] is not None}
    parts = [c["rate"] if k == "required_messages" else float(c["pass"]) for k, c in graded.items()]
    return {"ok": True, "overall": round(sum(parts) / len(parts), 3),
            "full_score": all(c["pass"] for c in graded.values()),
            "passed": f"{sum(bool(c['pass']) for c in graded.values())}/{len(graded)}",
            "failed": [k for k, c in graded.items() if not c["pass"]], **checks}
