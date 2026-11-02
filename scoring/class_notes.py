"""Score class notes (local workflow or Claude Code) against sample_data/classes/answer_key.json.

Output contract (same for every harness), in one folder:
  <class id>.md   one Markdown notes file per class: econ140.md, stat121.md, cs152.md
  today.md        the one-page summary: announcements with dates, corrections, what to review
  notes.json      {"date", "classes": [{"id", "slides": [{"slide", "title", "skipped", "time_ranges":
                  [["HH:MM:SS.mmm", "HH:MM:SS.mmm"], ...], "points": [...], "lecturer_added": [{"point", "quote"}]}],
                  "corrections": [{"slide", "slide_said", "lecturer_said", "quote"}], "announcements": [{"text", "due"}],
                  "skipped_slides": [...], "review": [...]}]}

What is graded (each a 0-1 rate; `overall` is their mean):
  key_points      every key point's regex groups match the class's notes (its .md plus its notes.json entry)
  verbal_only     the subset said only aloud (examples, hints, answers to questions)
  vision_only     the subset that is only in a chart or plot image
  corrections     the corrected fact is present AND the slide's wrong value is never stated as fact
                  (a line with the wrong value passes only if it also says it was wrong: "slide said", "typo",
                  "moved", "not", ...; quoted transcript words and slide_said fields are ignored)
  skipped         skipped slides flagged in notes.json (false flags are listed)
  announcements   every announcement, with its date, appears in today.md
  alignment       share of speaking time whose cue midpoint falls in the right slide's time_ranges
  quotes          share of quotes (lecturer_added, corrections) copied verbatim from that class's transcript
"""
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "sample_data" / "classes"
MARKER = re.compile(r"\b(slide|typo|wrong|incorrect|correct(ed|ion|s)?|actually|not|instead|old|outdated|original(ly)?|"
                    r"previous(ly)?|was|misprint|mistake|mixed up|error|said|says|moved|moving|no longer|from|"
                    r"rather than|ignore|fix(ed)?|usual(ly)?|normally|regular)\b|~~")
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")


def plain(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    for a, b in {"β": "beta ", "’": "'", "‘": "'", "–": "-", "—": "-", "−": "-", " ": " "}.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def compact(text):
    """Formula form: 'β₁ + β₃·x₂', 'beta_1 + beta_3 * x_2' and 'b1 + b3 times x2' all become 'b1+b3x2'."""
    t = plain(text).replace("\\cdot", " ").replace("\\beta", "b")
    t = re.sub(r"\bbeta\s*", "b", t)
    t = re.sub(r"\bplus\b", "+", re.sub(r"\btimes\b", " ", t))
    return re.sub(r"[\s*·×⋅_{}()\\$]", "", t)


def hit(text, alternative):
    if alternative.startswith("c:"):   # 'c:a|c:b' -> compact-form regex 'a|b'
        return re.search(alternative[2:].replace("|c:", "|"), compact(text)) is not None
    return re.search(alternative, plain(text)) is not None


def matches(text, groups):
    return all(any(hit(text, a) for a in g) for g in groups)


def strings(obj, skip=()):
    """Every string inside a JSON value, leaving out the fields named in `skip`."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() if k not in skip for s in strings(v, skip)]
    if isinstance(obj, list):
        return [s for v in obj for s in strings(v, skip)]
    return []


def stated_as_fact(lines, groups, allow=None):
    """Lines that state the wrong value without saying it was wrong (quoted transcript words ignored).
    `allow` is a regex for lines where the value is legitimately true of something else (Java's 0.75)."""
    out = []
    for line in lines:
        bare = QUOTED.sub(" ", line)
        if matches(bare, groups) and not MARKER.search(plain(bare)) and not (allow and re.search(allow, plain(bare))):
            out.append(line.strip()[:160])
    return out


def to_seconds(x):
    if isinstance(x, (int, float)):
        return float(x)
    secs = 0.0
    for p in str(x).strip().replace(",", ".").split(":"):
        secs = secs * 60 + float(p)
    return secs


def vtt(path, speakers):
    """Cue timings and the spoken text (speaker labels and tags removed) of a WebVTT file."""
    cues, words = [], []
    for block in re.split(r"\n\s*\n", path.read_text()):
        lines = block.strip().split("\n")
        i = next((k for k, ln in enumerate(lines) if "-->" in ln), None)
        if i is None:
            continue
        a, b = re.findall(r"[\d:.,]+", lines[i])[:2]
        payload = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:]))
        for sp in speakers:
            payload = re.sub(rf"^{re.escape(sp)}:\s*", "", payload)
        cues.append((to_seconds(a), to_seconds(b)))
        words.append(payload)
    return cues, " ".join(words)


def norm_words(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower().replace("’", "'").replace("'", ""))).strip()


def rate(got, total):
    return round(got / total, 3) if total else None


def score(outputs_dir):
    out = Path(outputs_dir)
    key = json.loads((DATA / "answer_key.json").read_text())
    notes_path, today_path = out / "notes.json", out / "today.md"
    try:
        notes = json.loads(notes_path.read_text())
    except (OSError, json.JSONDecodeError):
        notes = {"classes": []}
    today = today_path.read_text() if today_path.exists() else ""
    by_id = {c.get("id"): c for c in notes.get("classes", []) if isinstance(c, dict)}
    totals = {k: [0, 0] for k in ("key_points", "verbal_only", "vision_only", "corrections", "skipped",
                                  "announcements", "quotes")}
    align_right = align_total = 0.0
    align_seen = False
    per_class, files = {}, {"today.md": today_path.exists(), "notes.json": notes_path.exists()}
    for k in key["classes"]:
        md_path = out / f"{k['id']}.md"
        files[md_path.name] = md_path.exists()
        md = md_path.read_text() if md_path.exists() else ""
        entry = by_id.get(k["id"], {})
        text = md + "\n" + "\n".join(strings(entry))
        claim_lines = md.split("\n") + strings(entry, skip=("quote", "slide_said", "skip_quote"))
        report = {}

        points = [dict(p, slide=s["slide"]) for s in k["slides"] for p in s["key_points"]]
        missed = [f"s{p['slide']}: {p['text']}" for p in points if not matches(text, p["match"])]
        report["key_points"] = f"{len(points) - len(missed)}/{len(points)}"
        report["missed_key_points"] = missed
        for name, source in (("verbal_only", "speech"), ("vision_only", "image")):
            sub = [p for p in points if p["source"] == source]
            got = sum(matches(text, p["match"]) for p in sub)
            report[name] = f"{got}/{len(sub)}"
            totals[name][0] += got
            totals[name][1] += len(sub)
        totals["key_points"][0] += len(points) - len(missed)
        totals["key_points"][1] += len(points)

        fixes = []
        for c in k["corrections"]:
            right = matches(text, c["right"])
            wrong = stated_as_fact(claim_lines + today.split("\n"), c["wrong"], c.get("allow"))
            fixes.append({"slide": c["slide"], "corrected_value_present": right, "wrong_value_stated_as_fact": wrong,
                          "ok": right and not wrong})
        report["corrections"] = fixes
        totals["corrections"][0] += sum(f["ok"] for f in fixes)
        totals["corrections"][1] += len(fixes)

        flagged = {int(n) for n in entry.get("skipped_slides", []) if str(n).isdigit()}
        flagged |= {int(s["slide"]) for s in entry.get("slides", []) if s.get("skipped") and str(s.get("slide")).isdigit()}
        want = set(k["skipped_slides"])
        report["skipped"] = {"flagged": sorted(flagged & want), "missed": sorted(want - flagged),
                             "false_flags": sorted(flagged - want)}
        totals["skipped"][0] += len(flagged & want)
        totals["skipped"][1] += len(want)

        anns = []
        for a in k["announcements"]:
            in_today, in_notes = matches(today, a["match"]), matches(text, a["match"])
            wrong = stated_as_fact(today.split("\n") + claim_lines, a["wrong"]) if a.get("wrong") else []
            anns.append({"text": a["text"], "in_today": in_today, "in_class_notes": in_notes,
                         "old_value_stated_as_fact": wrong, "ok": in_today and not wrong})
        report["announcements"] = anns
        totals["announcements"][0] += sum(x["ok"] for x in anns)
        totals["announcements"][1] += len(anns)

        cues, spoken = vtt(DATA / k["transcript"], k["speakers"])
        spoken = f" {norm_words(spoken)} "
        quotes = [x.get("quote", "") for s in entry.get("slides", []) for x in s.get("lecturer_added", [])
                  if isinstance(x, dict)] + [x.get("quote", "") for x in entry.get("corrections", [])
                                             if isinstance(x, dict) and x.get("quote")]
        ok_quotes = [q for q in quotes if len(norm_words(q).split()) >= 4 and f" {norm_words(q)} " in spoken]
        report["quotes_verbatim"] = f"{len(ok_quotes)}/{len(quotes)}"
        report["bad_quotes"] = [q for q in quotes if q not in ok_quotes][:5]
        totals["quotes"][0] += len(ok_quotes)
        totals["quotes"][1] += len(quotes)

        ranges = {}
        for s in entry.get("slides", []):
            for r in s.get("time_ranges", []) or []:
                try:
                    ranges.setdefault(int(s["slide"]), []).append((to_seconds(r[0]), to_seconds(r[1])))
                except (KeyError, ValueError, TypeError, IndexError):
                    pass
        if ranges and len(cues) == len(k["cue_slides"]):
            align_seen = True
            right = total = 0.0
            for (a, b), gold in zip(cues, k["cue_slides"]):
                mid, dur = (a + b) / 2, b - a
                pred = next((n for n, rs in sorted(ranges.items()) if any(x <= mid <= y for x, y in rs)), None)
                total += dur
                right += dur * (pred == gold)
            report["alignment"] = round(right / total, 3)
            align_right += right
            align_total += total
        else:
            report["alignment"] = None
        per_class[k["id"]] = report

    components = {name: rate(*v) for name, v in totals.items()}
    components["alignment"] = round(align_right / align_total, 3) if align_seen else None
    graded = [v for v in components.values() if v is not None]
    return {"overall": round(sum(graded) / len(graded), 3) if graded else 0.0,
            "components": components,
            "counts": {name: f"{a}/{b}" for name, (a, b) in totals.items()},
            "files_present": files,
            "today_words": len(today.split()),
            "classes": per_class}
