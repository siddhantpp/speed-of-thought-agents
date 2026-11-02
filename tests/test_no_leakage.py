"""Test-data leakage check: no prompt may carry examples, values or phrases taken from the test set.

Every workflow is scored on the files in sample_data/ (the test inputs) against their answer keys. A prompt that
already holds one of those inputs' values, phrases or answers (an Agent Card example that is really the trap in a
test message, say) lets a model score on data it has seen. This test reads what a model is given before it sees
the input, and fails on:

  1. shared word runs: 6 or more words in a row that also occur in a test input or an answer key;
  2. distinctive values that also occur in the test data: percentages, money, times, dates, decimals, "N of M"
     counts, numbers with units, message / work / card ids, numbered tasks (Problem Set 5, HW7, Quiz 4), a course
     code paired with a task, formulas (beta1 + beta3*x2), relative dates ("next Friday") and proper names;
  3. a quoted example of 3 or more words that occurs word for word in a test input or an answer key;
  4. near-copies of the short phone messages in sample_data/assistant/messages.jsonl: a quoted example most of
     whose words are one test message's words.

What counts as a prompt (text a model reads that does not come from the input files):
  agents/*.json                  job, instructions and the output schema's descriptions (success_check is
                                 documentation and is never sent)
  workflows/*.py, assistant/*.py every string literal except docstrings: task texts and the feedback that code
                                 checks send back to a model
  workflows/slide_tools.py       the whole file, comments included: Claude Code gets it as an input of the slides run
  agentkit/tools.py, mcp_servers/  tool descriptions (both sides see them in their tool lists) and the calendar
                                 reconciler's feedback
  claude_compare/prompts/*.md    the Claude Code prompts

The test set: every input and answer key under sample_data/ (messages, calendar, transcripts, slide and receipt
generators, which hold the text printed on the images, house rules, brief, notes, data, question, the recorded
research responses when present, the phone messages and fixtures) and sample_data/README.md, which describes the
traps. ALLOWED lists what a prompt may legitimately share with it, each with its reason: the persona, the course
list, the sample data's "now", the inputs' own limits restated, tool and file names.

    .venv/bin/python -m unittest tests.test_no_leakage      # from the repo root
    .venv/bin/python tests/test_no_leakage.py --report      # every finding, allowed ones included
"""
import ast
import io
import json
import re
import sys
import tokenize
import unicodedata
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works in the exported repo (tests/ next to agentkit/) and in the source tree (tools/repo_extras/tests).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "agentkit").is_dir())
DATA = ROOT / "sample_data"
RUN = 6   # words in a row

# (value, reason[, where]). A finding is allowed when its text equals an entry (word runs: lie inside one); an entry
# with a `where` applies only to prompt texts whose location starts with it. Allowed values are blanked out of a
# prompt before its values and names are read, so "CS 152" allows the 152 in it.
ALLOWED = [
    # The persona and the people the inputs name as the task's context (sample_data/README.md "Persona").
    ("riley park", "the student's name (persona)"),
    ("riley", "the student's name (persona)"),
    ("alex", "a roommate named in house_rules.md, the receipts input"),
    ("sam", "a roommate named in house_rules.md, the receipts input"),
    ("jordan", "a roommate named in house_rules.md, the receipts input"),
    # The course list (persona) and the class ids of schedule.json, which name the per-class output files.
    ("econ 140", "the course list"), ("stat 121", "the course list"), ("cs 152", "the course list"),
    ("hist 97", "the course list"), ("econ140", "a class id in schedule.json (an output file name)"),
    ("stat121", "a class id in schedule.json"), ("cs152", "a class id in schedule.json"),
    # The sample data's "now" and time zone (sample_data/README.md "Reference now").
    ("monday 2026-11-02 08:00 america/new_york", "the sample data's now"),
    ("2026-11-02t08:00:00-05:00", "the sample data's now"),
    ("2026-11-02", "today in the sample data"), ("nov 2", "today in the sample data"),
    ("8:00am", "the sample data's now"), ("08:00", "the sample data's now"),
    ("america/new_york", "the sample data's time zone"),
    # Inputs described, and their own limits restated, by the prompt that gets those inputs.
    ("800 to 1,200 words", "question.md's own limit"), ("800", "question.md: 800 to 1,200 words"),
    ("1,200", "question.md: 800 to 1,200 words"),
    ("how strong the evidence is, including whether well-known findings held up when others repeated them",
     "question.md's own requirement, restated"),
    ("peer-reviewed journal article or review", "question.md asks for peer-reviewed journal articles or reviews"),
    ("not a peer-reviewed journal article", "the credibility rule's exclusion reason (question.md's requirement)",
     "workflows/research_report.py"),
    ("openalex, crossref, eric or europe pmc", "the research tools' data sources"),
    ("700", "brief.md: at most 700 words of speaker notes"),
    ("key messages that must land. notes.md: riley's", "describes brief.md and notes.md, inputs"),
    ("day's classes (id, course, session, instructor, start", "describes the fields of schedule.json, an input"),
    ("3,600", "the transcript's length: describes the input"),
    ("r1", "a receipt photo's file name (input)"), ("r5", "a receipt photo's file name (input)"),
    ("0.0", "a zero placeholder in a JSON format example"),
    ("18 messages", "messages.jsonl holds 18 messages"),
    # Output contracts the scorers check, which the prompt has to state.
    ("starting at the due time and lasting 30 minutes", "the deadline-event contract (gcal_server.py)"),
    ("is already on the calendar at the right time", "the no_change rule (the answer key explains one case in the "
                                                     "same words)"),
    # Tool and font names.
    ("chrome", "the browser slide_tools.py renders with"), ("arial", "a font name in the slide CSS"),
]

STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its me my no not of on
or our she so than that the their them then there these they this those to was we were what when where which who
will with you your""".split())
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
MONTHS = {m: m[:3] for m in "january february march april may june july august september october november december".split()}
MONTHS["sept"] = "sep"
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "θ": "theta", "λ": "lambda",
         "μ": "mu", "σ": "sigma", "π": "pi", "Σ": "sum", "∑": "sum"}


# ---------------------------------------------------------------- normalizing
def norm(text):
    """Lowercase ASCII-ish text: Greek letters as words, curly quotes and dashes plain, dot and times signs as '*'."""
    text = unicodedata.normalize("NFKC", text)
    for g, word in GREEK.items():
        text = text.replace(g, word)
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
                                         "·": "*", "×": "*", "∙": "*", " ": " "}))
    return text.lower()


def words(text):
    return re.findall(r"[a-z0-9]+(?:[.:,][0-9]+)*", norm(text))


def canon(text):
    """The form values are compared in: month names short, no ordinals, times as h:mm am/pm, single spaces."""
    t = norm(text)
    t = re.sub(r"\b(" + "|".join(MONTHS) + r")\b\.?", lambda m: MONTHS[m.group(1)], t)
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?(?![a-z])",
               lambda m: f"{int(m.group(1))}:{m.group(2) or '00'}{m.group(3)}m", t)
    t = re.sub(r"\b(" + "|".join(WEEKDAYS) + r"),? at (?=\d)", r"\1 ", t)
    return re.sub(r"\s+", " ", t)


def phrase(text):
    return " ".join(words(text))


# ---------------------------------------------------------------- what the models read (prompt units)
def strings_in(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings_in(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings_in(v)]
    return [repr(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def schema_descriptions(schema):
    if isinstance(schema, dict):
        own = [schema["description"]] if isinstance(schema.get("description"), str) else []
        return own + [d for k, v in schema.items() if k != "description" for d in schema_descriptions(v)]
    if isinstance(schema, list):
        return [d for v in schema for d in schema_descriptions(v)]
    return []


def literal_strings(path, docstrings=False, tool_docs_only=False, inside=None):
    """[(line, text)] of a Python file's string literals of 2+ words; an f-string gives its literal parts.

    docstrings: keep docstrings too. tool_docs_only: only the docstrings of functions decorated as tools (an MCP
    server's tool descriptions). inside: only strings within the functions or classes with these names."""
    tree = ast.parse(path.read_text())
    docs, tool_docs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
                if any("tool" in ast.unparse(d) for d in getattr(node, "decorator_list", [])):
                    tool_docs.add(id(first.value))
    roots = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in inside] \
        if inside else [tree]
    out, seen = [], set()
    for root in roots:
        nodes = list(ast.walk(root))
        parts_of_f = {id(v) for n in nodes if isinstance(n, ast.JoinedStr) for v in n.values}
        for node in nodes:
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, ast.JoinedStr) and not tool_docs_only:
                out.append((node.lineno, "".join(v.value if isinstance(v, ast.Constant) else " ... " for v in node.values)))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in parts_of_f:
                if tool_docs_only and id(node) not in tool_docs:
                    continue
                if id(node) in docs and not (docstrings or id(node) in tool_docs):
                    continue
                out.append((node.lineno, node.value))
    return [(line, s) for line, s in out if len(s.split()) >= 2]


def comments(path):
    return [(t.start[0], t.string.lstrip("# ")) for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
            if t.type == tokenize.COMMENT]


# Which code text reaches a model. Files not listed in SCOPE are read whole (every string but docstrings).
SCOPE = {
    # The phone front ends only pass the student's text to assistant/core.py and show its replies.
    "assistant/server.py": None, "assistant/telegram_adapter.py": None,
    # core.py: the model calls (Models) and the code checks whose feedback goes back to a model. Its replies to the
    # student (help, refusals, confirmations) and its keyword lists are never sent to a model.
    "assistant/core.py": {"Models", "receipt_problems", "receipt_hints", "assignment_problems", "_extraction_problem"},
}


def prompt_units():
    """[(where, text)] for every piece of text a model reads that is written in the code, cards or prompts."""
    units = []
    for p in sorted((ROOT / "agents").glob("*.json")):
        card = json.loads(p.read_text())
        units += [(f"agents/{p.name} job", card.get("job", "")), (f"agents/{p.name} instructions", card.get("instructions", ""))]
        units += [(f"agents/{p.name} schema", d) for d in schema_descriptions(card.get("output_schema") or {})]
    for folder in ("workflows", "assistant"):
        for p in sorted((ROOT / folder).glob("*.py")):
            rel = f"{folder}/{p.name}"
            if p.name == "slide_tools.py":   # Claude Code reads this file: its docstrings and comments count
                units += [(f"{rel}:{line} (a Claude Code input)", s)
                          for line, s in literal_strings(p, docstrings=True) + comments(p)
                          if not re.search(r"[{}]|<[a-z/!]", s)]   # CSS, JavaScript and SVG markup are not prose
            elif rel in SCOPE:
                if SCOPE[rel]:
                    units += [(f"{rel}:{line}", s) for line, s in literal_strings(p, inside=SCOPE[rel])]
            else:
                units += [(f"{rel}:{line}", s) for line, s in literal_strings(p)]
    units += [(f"agentkit/tools.py:{line}", s) for line, s in literal_strings(ROOT / "agentkit" / "tools.py")]
    for name in ("gcal_server.py", "research_server.py"):
        units += [(f"mcp_servers/{name}:{line} (tool description)", s)
                  for line, s in literal_strings(ROOT / "mcp_servers" / name, tool_docs_only=True)]
    units += [(f"mcp_servers/gcal_reconcile.py:{line}", s)    # the reconciler's retry feedback
              for line, s in literal_strings(ROOT / "mcp_servers" / "gcal_reconcile.py", inside={"check_decision"})]
    for p in sorted((ROOT / "claude_compare" / "prompts").glob("*.md")):
        units.append((f"claude_compare/prompts/{p.name}", p.read_text()))
    return [(w, t) for w, t in units if t.strip()]


# ---------------------------------------------------------------- the test set
def inverted_abstract(index):
    return " ".join(w for _, w in sorted((i, w) for w, places in index.items() for i in places))


def json_texts(value):
    out = []

    def walk(v):
        if isinstance(v, dict):
            if isinstance(v.get("abstract_inverted_index"), dict):
                out.append(inverted_abstract(v["abstract_inverted_index"]))
            for k, x in v.items():
                if k != "abstract_inverted_index":
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            out.extend(strings_in(v))
    walk(value)
    return out


def vtt_text(text):
    keep = [re.sub(r"</?v[^>]*>", " ", line) for line in text.splitlines()
            if line.strip() and "-->" not in line and not line.strip().isdigit() and not line.startswith(("WEBVTT", "NOTE"))]
    return " ".join(keep)


def test_set():
    """[(file, [texts])] of every test input and answer key under sample_data/."""
    files = []
    for p in sorted(DATA.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(DATA).as_posix()
        if p.suffix == ".jsonl":
            texts = [t for line in p.read_text().splitlines() if line.strip() for t in json_texts(json.loads(line))]
        elif p.suffix == ".json":
            texts = json_texts(json.loads(p.read_text()))
        elif p.suffix == ".vtt":
            texts = [vtt_text(p.read_text())]
        elif p.suffix == ".py" and p.name.startswith("make_"):   # the generators hold the text on the images
            texts = [s for _, s in literal_strings(p, docstrings=True)]
        elif p.suffix in (".md", ".txt", ".csv"):
            texts = [p.read_text()]
        else:
            continue
        files.append((rel, texts))
    return files


# ---------------------------------------------------------------- the checks
UNITS = (r"probes|meals|first-years|people|students|keys|buckets|points|times|submissions|hires|receipts|"
         r"roommates|cents|classmates|participants|studies|sessions")
VALUE_RES = [
    ("percent", r"\d+(?:\.\d+)?\s?%"),
    ("money", r"\$\s?\d[\d,]*(?:\.\d+)?"),
    ("time", r"\b\d{1,2}:\d{2}[ap]m\b"),
    ("date", r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) \d{1,2}\b"),
    ("date", r"(?<![\d/])\d{1,2}/\d{1,2}(?![\d/])"),
    ("date", r"\b\d{4}-\d{2}-\d{2}(?:t\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|z)?)?"),
    ("date", r"\b(?:" + "|".join(WEEKDAYS) + r") \d{1,2}:\d{2}[ap]m\b"),
    ("relative date", r"\b(?:next|this|last) (?:" + "|".join(WEEKDAYS) + r")\b"),
    ("decimal", r"(?<![\d.])\d+\.\d+(?![\d.])"),
    ("count", r"\b\d+ (?:of|out of) (?:the )?\d+\b"),
    ("number", r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\d.,$-])(?!(?:19|20)\d\d(?!\d))\d{3,}(?![\d.,-])"),
    ("number with unit", r"\b\d[\d,]*(?:\.\d+)? (?:" + UNITS + r")\b"),
    ("id", r"\bm\d{2}\b|\bw\d{7,}\b|\br\d\b|\*{2,}\s?\d{4}|\b10\.\d{4,}/[^\s\"']+"),
    ("task", r"\b(?:problem set|pset|ps|homework|hw|essay|quiz|exam|midterm|milestone|lab|assignment|project|"
             r"lecture|section)\s?#?\d+\b"),
    ("course + task", r"\b[a-z]{2,5} ?\d{2,3}\b:?[^\n.;\"']{0,40}?\b(?:problem set|homework|essay|quiz|exam|milestone|"
                      r"project|proposal|study group|office hours|draft|report|autograder|regrade)\b"),
    ("formula", r"\b[a-z]+\d+ ?[+*-] ?[a-z]+\d+(?: ?[+*-] ?[a-z]+\d+)*"),
]
# Bare numbers, and names, are not matched against research/cache: most of it is off-topic search hits, and the
# sources that matter are in research/answer_key.json.
BARE = {"decimal", "number", "number with unit"}
NAME = re.compile(r"(?<![\w'-])[A-Z][a-z]+(?:[-'][A-Z][a-z]+)*(?![\w-])")
MID_SENTENCE = re.compile(r"(?:[a-z0-9,]|\b(?:Dr|Prof|Mr|Ms|Mrs)\.) ([A-Z][a-z]+(?:[-'][A-Z][a-z]+)*)(?![\w-])")
QUOTED = re.compile(r"'([^'\n]{3,80})'|\"([^\"\n]{3,80})\"|“([^”\n]{3,80})”")


def compact(kind, text):
    """Formulas, money and percentages compared without the spaces around their signs ('beta1 + beta3*x2')."""
    if kind not in ("formula", "money", "percent"):
        return text
    return re.sub(r"\$\s+", "$", re.sub(r"\s*([+*%-])\s*", r"\1", text))


def build_index():
    runs, texts_by_file, capital, lower = {}, {}, {}, {}
    for rel, texts in test_set():
        for t in texts:
            ws = words(t)
            for i in range(len(ws) - RUN + 1):
                runs.setdefault(tuple(ws[i:i + RUN]), rel)
            raw = unicodedata.normalize("NFKC", t)
            if not rel.startswith("research/cache/"):
                # lower-case words of prose only: a one-word string is a key or a pattern ("mueller" in a grading rule)
                for w in re.findall(r"(?<![A-Za-z])[a-z][a-z'-]+", raw) if len(raw.split()) > 1 else []:
                    lower[w] = lower.get(w, 0) + 1
                mid = {m.group(1).lower() for m in MID_SENTENCE.finditer(raw)}
                for m in NAME.finditer(raw):
                    n = m.group(0).lower()
                    rel0, k, is_mid = capital.get(n, (rel, 0, False))
                    capital[n] = (rel0, k + 1, is_mid or n in mid)
        texts_by_file[rel] = "\n".join(canon(t) for t in texts)
    # A proper name: written capitalized in mid-sentence in the test data, and seldom in lower case there.
    names = {n: rel for n, (rel, k, mid) in capital.items() if mid and lower.get(n, 0) * 3 <= k
             and n not in WEEKDAYS and n[:3] not in {d[:3] for d in WEEKDAYS} and n not in MONTHS
             and n[:3] not in MONTHS.values() and len(n) >= 3}
    messages = [m["text"] for m in (json.loads(line) for line in (DATA / "assistant" / "messages.jsonl").read_text()
                                    .splitlines() if line.strip())]
    return runs, texts_by_file, names, messages


def allowed_entries(where):
    return [a for a in ALLOWED if len(a) < 3 or where.startswith(a[2])]


def blank_allowed(text, where):
    t = canon(text)
    for a in sorted(allowed_entries(where), key=lambda a: -len(a[0])):
        t = re.sub(r"(?<![\w.])" + re.escape(canon(a[0])) + r"(?:'s)?(?![\w])", " _ ", t)
    return t


def findings(units=None, index=None):
    """[(where, kind, value, test file)] for every prompt text shared with the test set, before the allow-list."""
    units = prompt_units() if units is None else units
    runs, texts_by_file, names, messages = index or build_index()
    hay = {k: {rel: compact(k, t) for rel, t in texts_by_file.items()
               if not (k in BARE and rel.startswith("research/cache/"))} for k, _ in VALUE_RES}
    flats = {rel: f" {phrase(t)} " for rel, t in texts_by_file.items() if not rel.startswith("research/cache/")}
    found = []
    for where, text in units:
        # 1. word runs, overlapping 6-word hits merged into one span
        ws, spans = words(text), []
        for i in (i for i in range(len(ws) - RUN + 1) if tuple(ws[i:i + RUN]) in runs):
            if spans and i <= spans[-1][1]:
                spans[-1][1] = i + RUN
            else:
                spans.append([i, i + RUN, runs[tuple(ws[i:i + RUN])]])
        found += [(where, "word run", " ".join(ws[a:b]), rel) for a, b, rel in spans]
        # 2. distinctive values and names (what the allow-list permits is blanked out first)
        ct = blank_allowed(text, where)
        for kind, rx in VALUE_RES:
            for m in re.finditer(rx, ct):
                value = compact(kind, re.sub(r"\s+", " ", m.group(0).strip()))
                pattern = re.compile(r"(?<![\w.])" + re.escape(value) + r"(?![\w])")
                rel = next((r for r, t in hay[kind].items() if pattern.search(t)), None)
                if rel:
                    found.append((where, kind, value, rel))
        raw = unicodedata.normalize("NFKC", text)
        ok_words = {w for a in allowed_entries(where) for w in words(a[0])}
        for m in NAME.finditer(raw):
            n = re.sub(r"'s$", "", m.group(0).lower())
            if n in names and n not in ok_words:
                found.append((where, "name", n, names[n]))
        # 3. a quoted example of 3+ words that occurs word for word in the test data (research/cache aside)
        for q in QUOTED.finditer(text):
            quoted = next(g for g in q.groups() if g)
            ph = phrase(quoted)
            if len(ph.split()) >= 3 and not re.fullmatch(r"[\w.-]+", quoted.strip()):   # not an identifier
                rel = next((r for r, flat in flats.items() if f" {ph} " in flat), None)
                if rel:
                    found.append((where, "quoted example", ph, rel))
        # 4. a quoted example that is mostly one test phone message's words
        for q in QUOTED.finditer(text):
            quoted = next(g for g in q.groups() if g)
            ex = set(words(quoted))
            if len(ex) < 2 or re.fullmatch(r"[\w.-]+", quoted.strip()):   # one identifier, such as "event_id"
                continue
            for msg in messages:
                shared = ex & set(words(msg))
                if len(shared) >= 0.6 * len(ex) and len(shared - STOP) >= 2:
                    found.append((where, "phone message", " ".join(words(quoted)), f"assistant/messages.jsonl: {msg!r}"))
    return sorted(set(found))


def allowed(finding):
    where, kind, value, _ = finding
    v = phrase(value)
    for a in allowed_entries(where):
        pa = phrase(a[0])
        if v == pa or value.replace(" ", "") == canon(a[0]).replace(" ", ""):
            return True
        if kind == "word run" and f" {v} " in f" {pa} ":
            return True
    return False


def leaks():
    return [f for f in findings() if not allowed(f)]


class NoLeakageTest(unittest.TestCase):
    def test_prompts_share_nothing_with_the_test_set(self):
        bad = leaks()
        self.assertEqual(bad, [], "prompt text shared with the test set (allow it in ALLOWED only if the task needs it):\n"
                         + "\n".join(f"  {w}: {k} {v!r} (also in {rel})" for w, k, v, rel in bad))

    def test_checker_catches_planted_leaks(self):
        """The checker itself: plant one leak of each kind in a fake prompt and expect each to be found."""
        planted = [("planted", "e.g. 'Problem Set 5 is due next Friday at 5:00 pm'"),
                   ("planted", "the slide said beta1 + beta3*x1 but the lecturer said beta 1 plus beta 3 times x2"),
                   ("planted", "taxable_base: 'TAX 6.25% ON 63.23' gives 63.23"),
                   ("planted", "Typos are normal: 'whats due tmrw' is calendar"),
                   ("planted", "e.g. 'Put 31 of 40 in the title'"),
                   ("planted", "search for the authors' surnames, e.g. Mueller"),
                   ("planted", "keep its hedges ('did not consistently differ')")]
        kinds = {f[1] for f in findings(planted) if not allowed(f)}
        for kind in ("task", "relative date", "formula", "word run", "percent", "decimal", "phone message", "count",
                     "name", "quoted example"):
            self.assertIn(kind, kinds)

    def test_allowed_entries_occur_in_the_test_set(self):
        """An allow-list entry that occurs nowhere in the test data allows nothing: remove it."""
        runs, texts_by_file, _, _ = build_index()
        text = "\n".join(texts_by_file.values())
        flat = phrase(text)
        missing = [a[0] for a in ALLOWED if canon(a[0]) not in text and phrase(a[0]) not in flat]
        self.assertEqual(missing, [])


def report():
    n_allowed = 0
    for f in findings():
        ok = allowed(f)
        n_allowed += ok
        print(f"{'allowed' if ok else 'LEAK   '}  {f[0]}: {f[1]} {f[2]!r}  (also in {f[3]})")
    bad = leaks()
    print(f"\n{len(bad)} leak(s), {n_allowed} allowed finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--report" in sys.argv:
        sys.exit(report())
    unittest.main()
