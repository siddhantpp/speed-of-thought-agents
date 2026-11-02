"""The phone assistant's brain: one router, a fixed table of handlers, and calendar writes that need a tap.

Every step goes to the cheapest executor that does it reliably:

  classifier  (Gemma 4 12B, constrained enum)   router: one phone message -> exactly one allowed intent, or refuse
  agents      (Gemma 4 12B, no tools at all)     event extractor (title + the date words, copied)
                                                 phraser (one opening sentence for a calendar answer)
                                                 receipt reader (vision) and receipt assigner (who shares what)
  code                                           the dispatch allow-list, dates (resolve_date) and their checks,
                                                 calendar windows and filters, class-notes and flashcard lookup,
                                                 sum_items, compute_balances, every reply's facts
  MCP         mcp_servers/gcal_server.py         sandbox calendar; only list_events and create_event are registered
  human       you, on your phone                 Confirm or Cancel on every calendar write; sending any money request

No model ever gets a tool. Code holds the two calendar tools, and only Assistant.decide(), which runs when you
tap Confirm, can call create_event. Nothing is ever sent, paid, moved or deleted.

The web server (server.py), the Telegram adapter (telegram_adapter.py) and the batch eval
(workflows/assistant_eval.py) all call the same Assistant.handle() and Assistant.decide().
"""
import copy
import csv
import dataclasses
import difflib
import hashlib
import io
import json
import os
import random
import re
import secrets
import sys
import threading
import time
import unicodedata
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent          # demos/ (the repo root in the exported repo)
sys.path.insert(0, str(ROOT))
from agentkit import Agent, AgentCard, Tracer  # noqa: E402
from agentkit.mcp import MCPServer  # noqa: E402
from agentkit.tools import Toolbox, compute_balances, resolve_date, sum_items  # noqa: E402

TZ = ZoneInfo("America/New_York")
DEMO_NOW = "2026-11-02T08:00:00-05:00"     # the sample data's "now" (Monday morning), as in the other workflows
GCAL_SERVER = ROOT / "mcp_servers" / "gcal_server.py"

INTENTS = ("calendar", "add_event", "class_notes", "quiz", "split_receipt")   # the allow-list
REFUSE = "refuse"
CAPABILITIES = [
    ("calendar", "What's due this week? / What's on today?"),
    ("add_event", "Add to my calendar: study group Thursday 7pm (you confirm first)"),
    ("class_notes", "Notes from ECON 140"),
    ("quiz", "Quiz me on load factor"),
    ("split_receipt", "Split this receipt (attach a photo; I draft, you send)"),
]
MAX_TEXT = 1000                  # characters per message
MAX_PHOTO_BYTES = 6 * 1024 * 1024
MAX_PIXELS = 40_000_000          # decompression-bomb guard
MAX_SIDE = 2048                  # photos are scaled down to this before a model sees them
PROPOSAL_TTL = 600               # seconds a Confirm button stays valid
CONFIRM_WORDS = re.compile(r"^\s*(yes|yep|yeah|ok|okay|sure|confirm(ed)?|do it|go ahead|approve[d]?)\b", re.I)


class UploadError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- small helpers
def clean_text(text):
    """Printable text only, at most a few spaces in a row; the router's quote markers cannot be forged."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = text.replace("<<<", "‹‹‹").replace(">>>", "›››")
    return re.sub(r"[ \t]{3,}", "  ", text).strip()


def fmt_clock(t):
    hour = t.hour % 12 or 12
    return f"{hour}{'' if t.minute == 0 else f':{t.minute:02d}'} {'am' if t.hour < 12 else 'pm'}"


def fmt_when(iso_or_dt):
    t = iso_or_dt if isinstance(iso_or_dt, datetime) else datetime.fromisoformat(iso_or_dt)
    t = t.astimezone(TZ)
    return f"{t:%a %b} {t.day}, {fmt_clock(t)}"


def norm_words(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def last4(text):
    return "".join(ch for ch in str(text) if ch.isdigit())[-4:]


def money(x):
    return f"${x:,.2f}"


# ---------------------------------------------------------------- calendar windows (code, not a model)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_ALIASES = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3, "thurs": 3,
                   "fri": 4, "sat": 5, "sun": 6, **{d: i for i, d in enumerate(WEEKDAYS)}}
TOMORROW = re.compile(r"\b(tomorrow|tomorow|tommorow|tommorrow|tomorrw|tmrw|tmr|tmrow|tmrrw|2morrow|2moro|2mrw)\b")
TODAY = re.compile(r"\b(today|todays|tonight|tonite|tday|2day|this (morning|afternoon|evening))\b")
DUE_WORDS = re.compile(r"\b(due|deadlines?|submit|submissions?|assignments?|homework|hw|psets?)\b")
MONTH_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|\b\d{1,2}/\d{1,2}\b")


def _day_start(d):
    return datetime(d.year, d.month, d.day, tzinfo=TZ)


def find_weekday(text):
    """(weekday index, said 'next') for the first weekday in the text, typos included ('wendsday')."""
    words = re.findall(r"[a-z]+", text)
    for k, w in enumerate(words):
        i = WEEKDAY_ALIASES.get(w)
        if i is None and len(w) >= 6:
            close = difflib.get_close_matches(w, WEEKDAYS, n=1, cutoff=0.8)
            i = WEEKDAYS.index(close[0]) if close else None
        if i is not None:
            return i, k > 0 and words[k - 1] == "next"
    return None


def calendar_window(text, now):
    """Which stretch of time a 'what's due / what's on' message asks about: (start, end, label, spoken label)."""
    t = text.lower()
    today = now.date()
    next_monday = _day_start(today + timedelta(days=7 - now.weekday()))
    if m := re.search(r"\b(?:next|coming)\s+(\d{1,2})\s+days?\b", t):
        n = max(1, min(31, int(m.group(1))))
        return now, now + timedelta(days=n), f"in the next {n} days", "the next few days"
    if re.search(r"\bnext\s+(week|wk)\b", t):
        return next_monday, next_monday + timedelta(days=7), "next week", "next week"
    if re.search(r"\b(this|the|rest of the)\s+(week|wk)\b", t):
        return now, next_monday, "this week", "this week"
    if re.search(r"\bweekend\b", t):
        sat = _day_start(today + timedelta(days=(5 - now.weekday()) % 7)) if now.weekday() < 5 else now
        return max(sat, now), next_monday, "this weekend", "this weekend"
    tomorrow = _day_start(today + timedelta(days=1))
    if TOMORROW.search(t) and TODAY.search(t):
        return now, tomorrow + timedelta(days=1), "today and tomorrow", "today and tomorrow"
    if TOMORROW.search(t):
        return tomorrow, tomorrow + timedelta(days=1), f"tomorrow ({tomorrow:%a %b} {tomorrow.day})", "tomorrow"
    if TODAY.search(t):
        return now, tomorrow, "today", "today"
    phrase = None
    if m := MONTH_DATE.search(t):
        phrase = m.group(0)
    elif wd := find_weekday(t):
        phrase = ("next " if wd[1] else "") + WEEKDAYS[wd[0]]
    if phrase:
        r = resolve_date(phrase, now.isoformat())     # the same date rules as everywhere else
        if "iso" in r:
            day = datetime.fromisoformat(r["iso"]).date()
            lo = _day_start(day)
            return max(lo, now), lo + timedelta(days=1), f"on {lo:%A} ({lo:%b} {lo.day})", lo.strftime("%A")
    return now, now + timedelta(days=7), "in the next 7 days", "the coming week"


def is_deadline(event):
    return bool(re.search(r"\b(due|deadline)\b", event.get("summary", ""), re.I))


def event_start(event):
    """An event's start as a datetime in New York time (all-day events start at midnight); None if it has none."""
    s = event.get("start", {})
    if s.get("dateTime"):
        return datetime.fromisoformat(s["dateTime"]).astimezone(TZ)
    if s.get("date"):
        return _day_start(datetime.fromisoformat(s["date"]).date())
    return None


# ---------------------------------------------------------------- the calendar, through MCP
class CalendarError(RuntimeError):
    pass


class Calendar:
    """The calendar MCP server with only two of its five tools registered: no update_event, no delete_event."""

    TOOLS = ("list_events", "create_event")

    def __init__(self, sandbox_file, seed, log=None):
        self.file = Path(sandbox_file)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        backend = os.environ.get("ASSISTANT_GCAL_BACKEND", "sandbox")   # "google" = your real calendar (unverified)
        env = {"GCAL_BACKEND": backend, "GCAL_SANDBOX_FILE": str(self.file), "GCAL_SANDBOX_SEED": str(seed)}
        if log:                                  # private like the other state files
            os.close(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
        self.toolbox = Toolbox()
        self.server = self.toolbox.add_mcp(MCPServer([sys.executable, str(GCAL_SERVER)], env=env, log=log),
                                           only=self.TOOLS)

    def list(self, lo, hi, tracer=None, label="read"):
        res = self.toolbox.call("list_events", {"time_min": lo.isoformat(), "time_max": hi.isoformat()},
                                tracer, "code", label)
        if not isinstance(res, dict) or "error" in res:
            raise CalendarError(str(res.get("error") if isinstance(res, dict) else res))
        return res.get("items", [])

    def create(self, args, tracer=None, label="confirm"):
        """Called only by Assistant.decide(), after a human tapped Confirm."""
        res = self.toolbox.call("create_event", args, tracer, "code", label)
        if not isinstance(res, dict) or "error" in res:
            raise CalendarError(str(res.get("error") if isinstance(res, dict) else res))
        return res

    def close(self):
        self.server.close()


# ---------------------------------------------------------------- proposals: a write waits for a tap
@dataclass
class Proposal:
    id: str
    owner: str
    kind: str                    # only "create_event" exists
    args: dict
    summary: str
    expires: float


class Proposals:
    def __init__(self, ttl=PROPOSAL_TTL, max_open=20):
        self.ttl, self.max_open = ttl, max_open
        self._open: dict[str, Proposal] = {}
        self._lock = threading.Lock()

    def add(self, owner, kind, args, summary):
        with self._lock:
            now = time.monotonic()
            self._open = {k: p for k, p in self._open.items() if p.expires > now}
            mine = sorted((p for p in self._open.values() if p.owner == owner), key=lambda p: p.expires)
            for p in mine[:max(0, len(mine) - self.max_open + 1)]:
                del self._open[p.id]
            p = Proposal(secrets.token_urlsafe(12), owner, kind, args, summary, now + self.ttl)
            self._open[p.id] = p
            return p

    def take(self, pid, owner):
        """Remove and return the proposal if it is open, unexpired and this owner's; else None. One use only."""
        with self._lock:
            p = self._open.get(str(pid))
            if p is None or not secrets.compare_digest(p.owner, owner):
                return None
            del self._open[p.id]
            return p if p.expires > time.monotonic() else None

    def pending(self, owner):
        now = time.monotonic()
        with self._lock:
            return sum(1 for p in self._open.values() if p.owner == owner and p.expires > now)


# ---------------------------------------------------------------- audit log and rate limits (shared by both front ends)
class AuditLog:
    """One JSON object per line. Never holds the token; message text is cut to 200 characters."""

    def __init__(self, path, max_bytes=5 * 1024 * 1024):
        self.path, self.max_bytes = Path(path), max_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **event):
        event = {"t": datetime.now(TZ).isoformat(timespec="seconds"), **event}
        if isinstance(event.get("text"), str):
            event["text"] = event["text"][:200]
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(line + "\n")


class RateLimiter:
    """At most `limit` hits per key in any `window` seconds (sliding window)."""

    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._hits) > 10_000:      # forget idle clients
                self._hits = {k: v for k, v in self._hits.items() if v}
            return True

    def count(self, key):
        now = time.monotonic()
        with self._lock:
            return sum(1 for t in self._hits.get(key, ()) if t > now - self.window)


# ---------------------------------------------------------------- photos
def sanitize_image(data: bytes) -> bytes:
    """Check an uploaded photo and re-encode it as a PNG: no EXIF (no GPS), at most MAX_SIDE pixels a side."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    if len(data) > MAX_PHOTO_BYTES:
        raise UploadError(f"photo is larger than {MAX_PHOTO_BYTES // (1024 * 1024)} MB", 413)
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            im = Image.open(io.BytesIO(data))
            if im.format not in ("JPEG", "PNG", "WEBP"):
                raise UploadError(f"photo must be JPEG, PNG or WebP, not {im.format}", 415)
            if im.width * im.height > MAX_PIXELS:
                raise UploadError("photo has too many pixels", 413)
            im.load()
        except UploadError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
            raise UploadError(f"photo rejected: {e}", 413)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise UploadError("that file is not a readable image", 400)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


# ---------------------------------------------------------------- class notes (outputs of workflows/class_notes.py)
DEPT_SYNONYMS = {"econ": ["econ", "economics", "behavioral econ"], "stat": ["stat", "stats", "statistics", "regression"],
                 "cs": ["cs", "comp sci", "compsci", "computer science"], "hist": ["hist", "history"]}


def _md_section(md, heading):
    m = re.search(rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", md, re.M | re.S)
    return m.group(1).strip() if m else ""


def load_class_notes(notes_dir):
    """{class id: {course, session, start, summary, announcements, corrections, source}}; {} if nothing is there."""
    notes_dir = Path(notes_dir)
    out = {}
    try:
        data = json.loads((notes_dir / "notes.json").read_text())
        for c in data.get("classes", []):
            if isinstance(c, dict) and c.get("id") and str(c.get("summary", "")).strip():
                out[str(c["id"])] = {"course": c.get("course", c["id"]), "session": c.get("session", ""),
                                     "start": c.get("start", ""), "summary": str(c["summary"]).strip(),
                                     "announcements": [a for a in c.get("announcements", []) if isinstance(a, dict)],
                                     "corrections": [x for x in c.get("corrections", []) if isinstance(x, dict)],
                                     "source": "notes.json"}
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    if out:
        return out
    for md_file in sorted(notes_dir.glob("*.md")) if notes_dir.is_dir() else []:
        if md_file.stem == "today":
            continue
        md = md_file.read_text(errors="replace")
        summary = _md_section(md, "Summary")
        if not summary:
            continue
        title = re.search(r"^# (.+)$", md, re.M)
        course, _, session = (title.group(1) if title else md_file.stem).rpartition(": ")
        anns = [{"text": re.sub(r"\*\*", "", line[2:]).strip()} for line in _md_section(md, "Announcements").splitlines()
                if line.startswith("- ")]
        out[md_file.stem] = {"course": course or md_file.stem, "session": session, "start": "", "summary": summary,
                             "announcements": anns, "corrections": [], "source": md_file.name}
    return out


def class_aliases(cid, info, schedule):
    """Words that name a class: its id, code, department nicknames, course name, session and instructor."""
    s = next((c for c in schedule if c.get("id") == cid), {})
    course = s.get("course") or info.get("course", cid)
    aliases = {norm_words(cid)}
    if m := re.match(r"\s*([A-Za-z]{2,5})\s*(\d{2,3})", course):
        dept, num = m.group(1).lower(), m.group(2)
        aliases |= {f"{dept} {num}", f"{dept}{num}", *DEPT_SYNONYMS.get(dept, [dept])}
    name = course.split(":", 1)[1] if ":" in course else ""
    if name.strip():
        aliases.add(norm_words(name))
    for sess in (s.get("session"), info.get("session")):
        if sess:
            aliases.add(norm_words(sess) + f" {norm_words(cid)}")   # "lecture 17" alone is too vague
    instructor = re.sub(r"\(.*?\)|\b(prof|dr|tf)\b\.?", "", s.get("instructor", ""), flags=re.I).split()
    if instructor:
        aliases.add(norm_words(instructor[-1]))
    return {a for a in aliases if a}


def match_class(text, notes, schedule):
    """The class a message names: (id, None), (None, 'none') or (None, 'ambiguous')."""
    words = norm_words(text)
    padded = f" {words} "
    tokens = words.split()
    hits = {}
    for cid, info in notes.items():
        for a in class_aliases(cid, info, schedule):
            if f" {a} " in padded:
                hits[cid] = hits.get(cid, 0) + 1 + a.count(" ")
            elif " " not in a and len(a) >= 4 and difflib.get_close_matches(a, [t for t in tokens if len(t) >= 4],
                                                                             n=1, cutoff=0.74):
                hits[cid] = hits.get(cid, 0) + 0.5      # a typo, e.g. "ecno" or "statistcs"
    if not hits:
        return None, "none"
    best = max(hits.values())
    top = [c for c, h in hits.items() if h == best]
    return (top[0], None) if len(top) == 1 else (None, "ambiguous")


# ---------------------------------------------------------------- flashcards (outputs of workflows/study_pack.py)
QUIZ_STOP = set("""quiz quizz quizes quizzes quizme test tests testing me on about some a an the please pls plz can
could you u give few couple of my for flashcard flashcards cards card practice question questions with from hey hi and
or let lets do want wanna to i im would like study review is are what quick three four five one two me up go over
everything anything stuff something all random any topic topics again more new another this that lecture class
cs 152 cs152 hash table tables hashtables course material deck time""".split())


def stem(w):
    w = w.lower()
    for suf in ("ing", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def load_flashcards(path):
    try:
        with open(path, newline="") as f:
            return [{"q": r[0].strip(), "a": r[1].strip(), "tag": (r[2].strip() if len(r) > 2 else "")}
                    for r in csv.reader(f) if len(r) >= 2 and r[0].strip() and r[1].strip()]
    except OSError:
        return []


def topic_terms(text):
    words = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    return [stem(w) for w in words if w not in QUIZ_STOP and not w.isdigit() and len(w) >= 3]


def _term_in(term, tokens):
    return any(tok.startswith(term) or (len(tok) >= 4 and term.startswith(tok)) or
               (len(term) >= 5 and difflib.SequenceMatcher(None, term, tok).ratio() >= 0.84) for tok in tokens)


def pick_cards(cards, text, seed):
    """(terms, chosen indexes). Topic words must appear in a card; no topic words = a random few."""
    n = 3
    if m := re.search(r"\b([1-5])\s*(?:cards?|questions?|qs)\b", text.lower()):
        n = int(m.group(1))
    terms = list(dict.fromkeys(topic_terms(text)))
    if not terms:
        rng = random.Random(seed)
        return terms, sorted(rng.sample(range(len(cards)), min(n, len(cards))))
    scored = []
    for i, c in enumerate(cards):
        tokens = {stem(w) for w in re.findall(r"[a-z0-9]+", f"{c['q']} {c['a']} {c['tag']}".lower().replace("-", " "))}
        k = sum(_term_in(t, tokens) for t in terms)
        if k:
            scored.append((-k, i))
    if not scored:
        return terms, []
    best = -min(scored)[0]
    keep = [i for k, i in sorted(scored) if -k == best]
    return terms, keep[:n]


# ---------------------------------------------------------------- receipts and house rules
def house_rules(path):
    text = Path(path).read_text()
    line = re.search(r"Roommates:(.*)", text)
    roommates = re.findall(r"\*\*([A-Z][A-Za-z]+)\*\*", line.group(1)) if line else []
    cards = {last4(card): person for card, person in
             re.findall(r"\|\s*([A-Z]+ \*{2,}\d{4})\s*\|\s*([A-Z][A-Za-z]+)\s*\|", text)}
    return {"text": text, "roommates": roommates, "cards": cards}


def receipt_problems(r, cards, now=None):
    """Code-level critic for one receipt read: the arithmetic adds up, the card is a house card, the date is sane."""
    try:
        res = sum_items(r["items"], r["subtotal"], r["tax"], r["total"], r.get("tip", 0), r.get("taxable_base", 0))
    except Exception as e:  # missing fields, non-numbers
        return [f"missing or malformed field ({type(e).__name__}: {e})"]
    problems = list(res["problems"])
    if not r["items"]:
        problems.append("no items were read")
    if last4(r.get("card_last4")) not in cards:
        problems.append(f"card_last4 {r.get('card_last4')!r} is not a house card {sorted(cards)}; "
                        "read the payment line (e.g. 'CARD ****1234') again")
    if now is not None:
        try:
            day = datetime.strptime(str(r.get("date", "")), "%Y-%m-%d").date()
            if not now.date() - timedelta(days=366) <= day <= now.date() + timedelta(days=1):
                problems.append(f"date {r.get('date')} is not within the last year (today is {now.date()}); read the "
                                "date line again, year included")
        except ValueError:
            problems.append(f"date {r.get('date')!r} is not YYYY-MM-DD; read the date line again")
    return problems


def receipt_hints(r):
    """Code can often say WHICH number was misread; a precise hint fixes more retries than 'it does not add up'."""
    hints = []
    try:
        lines = round(sum(float(it["line_total"]) for it in r["items"]), 2)
        sub, base = float(r["subtotal"]), float(r.get("taxable_base") or 0)
        if base and abs(sub - base) < 0.005 and abs(lines - sub) >= 0.005:
            hints.append(f"your subtotal {sub:.2f} is the number after ON in the tax line (that is taxable_base); "
                         "read the number on the SUBTOTAL line instead")
        if abs(lines - sub) >= 0.005 and abs(lines + float(r["tax"]) + float(r.get("tip") or 0) - float(r["total"])) < 0.005:
            hints.append(f"the items add up to {lines:.2f} and that plus tax and tip matches the total, so re-read the "
                         "SUBTOTAL line")
    except (KeyError, TypeError, ValueError):
        pass
    return hints


def assignment_problems(a, n, roommates):
    got = a.get("assignments", []) if isinstance(a, dict) else []
    idx = [x.get("item_index") for x in got]
    if sorted(idx) != list(range(n)):
        return [f"assign every item index 0..{n - 1} exactly once"]
    bad = [x for x in got if not x.get("assigned_to") or any(p not in roommates for p in x["assigned_to"])
           or len(set(x["assigned_to"])) != len(x["assigned_to"])]
    return [f"unknown, repeated or empty names in {bad[:2]}"] if bad else []


# ---------------------------------------------------------------- the model calls (tests swap in a stub)
class Models:
    """Every model call the assistant makes. None of these agents has a tool."""

    CARDS = ("assistant_router", "assistant_event_extractor", "assistant_phraser", "assistant_receipt_reader",
             "assistant_receipt_assigner")

    def __init__(self):
        self.cards = {name: AgentCard.load(name) for name in self.CARDS}

    def _run(self, name, task, tracer, label, images=None, schema=None, temperature=None):
        card = self.cards[name] if temperature is None else dataclasses.replace(self.cards[name], temperature=temperature)
        agent = Agent(card, None, tracer)     # a fresh Agent per call: safe across server threads
        return agent.run(task, images=images, label=label, **({"schema": schema} if schema else {}))

    def route(self, text, has_photo, tracer):
        task = f"Photo attached: {'yes' if has_photo else 'no'}\nPhone message:\n<<<\n{text}\n>>>"
        return self._run("assistant_router", task, tracer, "route").get("intent", REFUSE)

    def extract_event(self, text, now, feedback, tracer):
        task = f"Message received {fmt_when(now)} ({now:%A}).\nMessage:\n<<<\n{text}\n>>>"
        if feedback:
            task += f"\n\nYour previous answer had a problem: {feedback}\nFix it and answer again."
        return self._run("assistant_event_extractor", task, tracer, "extract")

    def phrase(self, window, facts, tracer):
        task = f"Window asked about: {window}\nFacts:\n" + "\n".join(f"- {f['weekday']}: {f['title']}" for f in facts)
        return self._run("assistant_phraser", task, tracer, "phrase").get("lead", "")

    def read_receipt(self, png, feedback, tracer):
        task = "Read this receipt."
        if feedback:
            task += f"\n\nYour previous reading did not add up: {feedback}\nLook at the photo again and fix it."
        # A retry at temperature 0 repeats the same reading word for word, so the retry samples a little.
        return self._run("assistant_receipt_reader", task, tracer, "read receipt" + (" retry" if feedback else ""),
                         images=[png], temperature=0.4 if feedback else None)

    def assign_items(self, rules, receipt, payer, feedback, tracer):
        n = len(receipt["items"])
        schema = copy.deepcopy(self.cards["assistant_receipt_assigner"].output_schema)
        arr = schema["properties"]["assignments"]
        arr["minItems"] = arr["maxItems"] = n
        arr["items"]["properties"]["assigned_to"] = {"type": "array", "minItems": 1,
                                                     "items": {"type": "string", "enum": rules["roommates"]}}
        task = (f"house_rules.md:\n{rules['text']}\n\nReceipt from {receipt['store']} on {receipt['date']}, paid by "
                f"{payer}. Items:\n" + "\n".join(f"{k}. {it['name']} (${it['line_total']})"
                                                for k, it in enumerate(receipt["items"])))
        if feedback:
            task += f"\n\nYour previous answer had a problem: {feedback}"
        return self._run("assistant_receipt_assigner", task, tracer, "assign items", schema=schema)


# ---------------------------------------------------------------- the assistant
@dataclass
class Config:
    calendar_file: Path                                   # the sandbox copy this assistant reads and writes
    calendar_seed: Path = ROOT / "sample_data" / "inbox" / "calendar_before.json"
    notes_dir: Path = ROOT / "outputs" / "class_notes"
    schedule_file: Path = ROOT / "sample_data" / "classes" / "schedule.json"
    flashcards: Path = ROOT / "outputs" / "study_pack" / "flashcards.csv"
    house_rules: Path = ROOT / "sample_data" / "receipts" / "house_rules.md"
    now: str | None = DEMO_NOW                            # None = the real clock
    proposal_ttl: int = PROPOSAL_TTL
    mcp_log: Path | None = None
    extra: dict = field(default_factory=dict)


class Assistant:
    def __init__(self, config: Config, models=None):
        self.cfg = config
        self.models = models or Models()
        self.calendar = Calendar(config.calendar_file, config.calendar_seed, log=config.mcp_log)
        self.proposals = Proposals(config.proposal_ttl)
        # The dispatch allow-list: a plain dict lookup, never getattr on a string a model produced.
        self.handlers = {"calendar": self._calendar, "add_event": self._add_event, "class_notes": self._class_notes,
                         "quiz": self._quiz, "split_receipt": self._split_receipt}
        assert tuple(self.handlers) == INTENTS

    def now(self):
        return datetime.fromisoformat(self.cfg.now).astimezone(TZ) if self.cfg.now else datetime.now(TZ)

    def close(self):
        self.calendar.close()

    # ------------------------------------------------------------ one message in, one reply out
    def handle(self, text, photo=None, owner="local", tracer=None):
        """photo: PNG bytes from sanitize_image(), or None. Returns a reply dict; never writes the calendar."""
        start = time.time()
        tracer = tracer or Tracer("assistant", quiet=True)
        n_events = len(tracer.events)
        text = clean_text(text)
        routed_by, intent = "code", REFUSE
        try:
            if len(text) > MAX_TEXT:
                reply = self._refuse(text, owner, f"That message is longer than {MAX_TEXT} characters.")
            elif not text and photo is None:
                reply = self._refuse(text, owner, "Hi! Here is what I can do.")
            elif not text:
                intent = "split_receipt"           # a bare photo: the only intent that takes one
                reply = self.handlers[intent](text, photo, owner, tracer)
            else:
                routed_by = "router"
                intent = self.models.route(text, photo is not None, tracer)
                label = intent
                handler = self.handlers.get(label) if isinstance(label, str) else None
                if handler is None:                # refuse, or anything that is not on the allow-list
                    intent = REFUSE
                    reply = self._refuse(text, owner)
                    if label != REFUSE:
                        reply["data"] = {"router_label_not_allowed": str(label)[:60]}
                else:
                    reply = handler(text, photo, owner, tracer)
        except CalendarError as e:
            reply = {"outcome": "error", "text": f"I couldn't reach the calendar ({e}). Nothing was changed."}
        except Exception as e:  # model server down, malformed output after retries, ...
            hint = "Is LM Studio running on this Mac?" if "connect" in str(e).lower() else ""
            reply = {"outcome": "error", "text": f"Something went wrong on the laptop ({type(e).__name__}). "
                                                 f"Nothing was changed. {hint}".strip()}
        reply.setdefault("proposal", None)
        reply.setdefault("cards", [])
        reply.setdefault("suggestions", [])
        reply.setdefault("data", {})
        new = tracer.events[n_events:]
        reply.update(intent=intent, routed_by=routed_by, latency_s=round(time.time() - start, 3),
                     model_calls=sum(e["kind"] == "llm" for e in new), tool_calls=sum(e["kind"] == "tool" for e in new))
        return reply

    def decide(self, proposal_id, decision, owner="local", tracer=None):
        """The only path to a calendar write: a human tapped Confirm on a proposal this assistant made."""
        if decision not in ("confirm", "cancel"):
            return {"outcome": "rejected", "text": "Unknown decision. Nothing changed."}
        p = self.proposals.take(proposal_id, owner)
        if p is None:
            return {"outcome": "rejected", "text": "That proposal is no longer open (already decided, expired, or not "
                                                   "yours). Nothing changed."}
        if decision == "cancel":
            return {"outcome": "cancelled", "proposal_id": p.id, "text": f"Cancelled. {p.summary} was not added."}
        if p.kind != "create_event":
            return {"outcome": "rejected", "text": "Nothing changed."}
        event = self.calendar.create(p.args, tracer, f"confirm {p.id[:6]}")
        return {"outcome": "confirmed", "proposal_id": p.id, "event_id": event.get("id"),
                "text": f"Added to your calendar: {p.args['summary']}, {fmt_when(p.args['start'])}."}

    # ------------------------------------------------------------ refuse (code, no model)
    def _refuse(self, text, owner, opening="Sorry, I can't do that one."):
        lines = [opening, "From your phone I can only:"] + [f"• {c}" for _, c in CAPABILITIES]
        lines.append("I never send, pay, delete or run anything.")
        if CONFIRM_WORDS.match(text or "") and self.proposals.pending(owner):
            lines.insert(1, "To confirm a proposal, tap its Confirm button; I never take a confirmation by text.")
        return {"outcome": "refused", "text": "\n".join(lines),
                "suggestions": ["What's due this week?", "Notes from ECON 140", "Quiz me on load factor"]}

    # ------------------------------------------------------------ 1. what's due / what's on (read only)
    def _calendar(self, text, photo, owner, tracer):
        now = self.now()
        lo, hi, label, spoken = calendar_window(text, now)
        due_only = bool(DUE_WORDS.search(text.lower()))
        # The API returns every event that overlaps the window; a day's answer lists only what STARTS that day
        # (a Thursday 11:59 pm deadline marker runs to 12:29 am, but it is not "on Friday").
        floor = _day_start(lo.date())
        events = [e for e in self.calendar.list(lo, hi, tracer) if (event_start(e) or lo) >= floor and event_start(e)]
        if due_only:
            events = [e for e in events if is_deadline(e)]
        events = events[:12]
        data = {"window": [lo.isoformat(), hi.isoformat()], "label": label, "due_only": due_only,
                "event_ids": [e["id"] for e in events]}
        if not events:
            return {"outcome": "answer", "data": data,
                    "text": f"Nothing {'due' if due_only else 'on your calendar'} {label}."}
        facts = []
        for e in events:
            s = e["start"]
            when = fmt_when(s["dateTime"]) if "dateTime" in s else f"{datetime.fromisoformat(s['date']):%a %b %-d}, all day"
            day = datetime.fromisoformat(s.get("dateTime") or s["date"]).strftime("%A")
            facts.append({"when": when, "weekday": day, "title": e.get("summary", "(no title)")})
        lead = ""
        try:
            lead = self.models.phrase(spoken, facts, tracer).strip()
        except Exception:
            pass
        if not self._lead_ok(lead, facts):
            data["phraser_rejected"] = lead
            n = len(facts)
            lead = f"{n} {'deadline' if due_only else 'item'}{'s' if n != 1 else ''} {label}:"
        body = "\n".join(f"• {f['when']} · {f['title']}" for f in facts)
        return {"outcome": "answer", "text": f"{lead}\n{body}", "data": data}

    @staticmethod
    def _lead_ok(lead, facts):
        """Code checks the phraser: no times or dates, only numbers and weekdays that are in the facts."""
        if not lead or len(lead.split()) > 25 or "\n" in lead:
            return False
        low = lead.lower()
        if re.search(r"\d\s*(am|pm)\b|\d:\d\d|\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d", low):
            return False
        allowed_numbers = {n for f in facts for n in re.findall(r"\d+", f["title"])}
        if any(n not in allowed_numbers for n in re.findall(r"\d+", low)):
            return False
        days = {f["weekday"].lower() for f in facts}
        return not any(d in low and d not in days for d in WEEKDAYS)

    # ------------------------------------------------------------ 2. add to my calendar (a proposal, never a write)
    def _add_event(self, text, photo, owner, tracer):
        now = self.now()
        ext, feedback, problem = {}, "", None
        for _ in range(2):                       # one retry with the code check's feedback
            ext = self.models.extract_event(text, now, feedback, tracer)
            problem = self._extraction_problem(ext, text)
            if not problem:
                break
            feedback = problem
        title = re.sub(r"\s+", " ", str(ext.get("title", ""))).strip(" \"'“”‘’.,:;-")[:80]
        phrase = str(ext.get("date_phrase", "")).strip()
        ask = {"outcome": "ask", "data": {"title": title, "date_phrase": phrase},
               "suggestions": ["Add to my calendar: study group Thursday 7pm"]}
        if not title:
            return {**ask, "text": "What should I add? For example: “Add to my calendar: study group Thursday 7pm”."}
        if not phrase or (problem and "date_phrase" in problem):
            return {**ask, "text": f"When is “{title}”? Try again with a day and time, e.g. “Thursday 7pm”."}
        r = resolve_date(phrase, now.isoformat())
        if "iso" not in r:
            return {**ask, "text": f"I couldn't work out a date from “{phrase}”. Try e.g. “Thursday 7pm” or “Nov 13 5pm”."}
        start = datetime.fromisoformat(r["iso"])
        if start < now - timedelta(minutes=1):
            return {**ask, "text": f"“{phrase}” is {fmt_when(start)}, which has already passed. Nothing was added."}
        if start > now + timedelta(days=366):
            return {**ask, "text": f"“{phrase}” is more than a year away ({fmt_when(start)}). Nothing was added."}
        minutes = ext.get("duration_minutes") or 0
        minutes = int(minutes) if isinstance(minutes, (int, float)) and 0 < minutes <= 720 else 30
        end = start + timedelta(minutes=minutes)
        nearby = self.calendar.list(start - timedelta(minutes=1), start + timedelta(minutes=1), tracer, "duplicate check")
        dup = next((e for e in nearby if difflib.SequenceMatcher(None, norm_words(e.get("summary", "")),
                                                                 norm_words(title)).ratio() > 0.8), None)
        data = {"title": title, "date_phrase": phrase, "start": start.isoformat(), "end": end.isoformat(),
                "interpretation": r["interpretation"], "location": str(ext.get("location", ""))[:80]}
        if dup:
            return {"outcome": "duplicate", "data": {**data, "event_id": dup["id"]},
                    "text": f"“{dup.get('summary')}” is already on your calendar at {fmt_when(start)}. Nothing to add."}
        args = {"summary": title, "start": data["start"], "end": data["end"], "location": data["location"],
                "description": "Added by the local phone assistant after a tap on Confirm."}
        span = f"{fmt_when(start)} to {fmt_clock(end)}" if minutes != 30 else fmt_when(start)
        p = self.proposals.add(owner, "create_event", args, f"“{title}” ({span})")
        note = " No time was given, so I used 11:59 pm." if "no time given" in r["interpretation"] else ""
        return {"outcome": "proposal", "data": {**data, "proposal_id": p.id},
                "proposal": {"id": p.id, "summary": title, "when": span, "start": data["start"], "end": data["end"],
                             "expires_in": self.proposals.ttl},
                "text": f"Add “{title}” on {span}?{note}\nNothing changes until you tap Confirm."}

    @staticmethod
    def _extraction_problem(ext, text):
        if not isinstance(ext, dict) or not str(ext.get("title", "")).strip():
            return "title is empty; copy what the event is from the message"
        phrase = norm_words(ext.get("date_phrase", ""))
        if phrase and phrase not in norm_words(text):
            return (f"date_phrase {ext.get('date_phrase')!r} is not in the message; copy the date and time words "
                    "exactly as written")
        return None

    # ------------------------------------------------------------ 3. notes from <class> (code)
    def _class_notes(self, text, photo, owner, tracer):
        notes = load_class_notes(self.cfg.notes_dir)
        if not notes:
            return {"outcome": "unavailable", "data": {"notes_dir": str(self.cfg.notes_dir)},
                    "text": "Class notes aren't ready yet. Run workflows/class_notes.py on the laptop first."}
        try:
            schedule = json.loads(Path(self.cfg.schedule_file).read_text()).get("classes", [])
        except (OSError, ValueError):
            schedule = []
        cid, why = match_class(text, notes, schedule)
        names = {k: re.split(r":", v["course"])[0].strip() for k, v in notes.items()}
        if cid is None:
            which = ", ".join(names.values())
            return {"outcome": "ask", "data": {"class_id": None, "reason": why},
                    "suggestions": [f"Notes from {n}" for n in names.values()],
                    "text": f"Which class? I have notes for {which}."}
        c = notes[cid]
        when = ""
        if c.get("start"):
            try:
                when = f" ({datetime.fromisoformat(c['start']):%a %b %-d})"
            except ValueError:
                pass
        lines = [f"{c['course']}{' · ' + c['session'] if c.get('session') else ''}{when}", c["summary"]]
        anns = c["announcements"][:4]
        if anns:
            lines.append("Dates said in class:")
            lines += [f"• {a.get('when') + ': ' if a.get('when') else ''}{a.get('text', '')}" for a in anns]
        if c["corrections"]:
            k = len(c["corrections"])
            lines.append(f"{k} slide correction{'s' if k != 1 else ''} in the full notes (the slide was wrong; the "
                         "lecturer corrected it).")
        return {"outcome": "answer", "text": "\n".join(lines),
                "data": {"class_id": cid, "source": c["source"], "summary": c["summary"]}}

    # ------------------------------------------------------------ 4. quiz me on <topic> (code)
    def _quiz(self, text, photo, owner, tracer):
        cards = load_flashcards(self.cfg.flashcards)
        if not cards:
            return {"outcome": "unavailable", "data": {"flashcards": str(self.cfg.flashcards)},
                    "text": "No flashcards yet. Run workflows/study_pack.py on the laptop first."}
        seed = int(hashlib.sha256(f"{text}|{self.now():%Y%m%d%H%M}".encode()).hexdigest()[:8], 16)
        terms, chosen = pick_cards(cards, text, seed)
        tags = sorted({c["tag"] for c in cards if c["tag"]})
        deck = f"{len(cards)} cards" + (f" on {', '.join(t.replace('-', ' ') for t in tags)}" if tags else "")
        if not chosen:
            return {"outcome": "no_cards", "data": {"terms": terms, "card_indexes": []},
                    "suggestions": ["Quiz me on load factor", "Quiz me on collisions"],
                    "text": f"I don't have flashcards on that. My deck has {deck} (CS 152 Lecture 9). "
                            "Try “quiz me on load factor”."}
        picked = [{"q": cards[i]["q"], "a": cards[i]["a"]} for i in chosen]
        about = f" on {' '.join(terms)}" if terms else ""
        return {"outcome": "answer", "cards": picked,
                "data": {"terms": terms, "card_indexes": chosen, "cards": picked},
                "text": f"{len(picked)} card{'s' if len(picked) != 1 else ''}{about}. Answer in your head, then "
                        "tap a card to check."}

    # ------------------------------------------------------------ 5. split this receipt (vision agent + code)
    def _split_receipt(self, text, photo, owner, tracer):
        if photo is None:
            return {"outcome": "need_photo", "text": "Attach a photo of the receipt and send “split this receipt”."}
        rules = house_rules(self.cfg.house_rules)
        r, feedback, problems = None, "", ["not read"]
        for _ in range(2):                       # code checks the arithmetic; one retry with its feedback
            try:
                r = self.models.read_receipt(photo, feedback, tracer)
                problems = receipt_problems(r, rules["cards"], self.now())
            except Exception as e:  # malformed JSON after the agent's own retry
                problems = [f"{type(e).__name__}: {e}"]
            if not problems:
                break
            feedback = "; ".join(problems + (receipt_hints(r) if isinstance(r, dict) else []))
        if problems:
            return {"outcome": "unreadable", "data": {"problems": problems},
                    "text": "I couldn't make this receipt add up, so I won't guess:\n" +
                            "\n".join(f"• {p}" for p in problems[:3]) + "\nTry a flatter, sharper photo."}
        payer = rules["cards"][last4(r["card_last4"])]
        a, feedback, problems = {}, "", ["not assigned"]
        for _ in range(2):
            a = self.models.assign_items(rules, r, payer, feedback, tracer)
            problems = assignment_problems(a, len(r["items"]), rules["roommates"])
            if not problems:
                break
            feedback = "; ".join(problems)
        if problems:
            return {"outcome": "needs_human", "data": {"problems": problems},
                    "text": "I read the receipt but couldn't apply the house rules cleanly. Split this one by hand."}
        who = {x["item_index"]: x["assigned_to"] for x in a["assignments"]}
        receipt = {"id": "photo", "payer": payer, "subtotal": r["subtotal"], "tax": r["tax"], "tip": r.get("tip", 0),
                   "total": r["total"], "items": [{**it, "assigned_to": who[k]} for k, it in enumerate(r["items"])]}
        bal = compute_balances([receipt])
        shares = {p: round(bal["share"].get(p, 0.0), 2) for p in rules["roommates"]}
        owe = [(p, v) for p, v in shares.items() if p != payer and v > 0]
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        day = f"{d:%a %b} {d.day}"
        store = r["store"].title() if r["store"].isupper() else r["store"]
        drafts = [f"Hi {p}, your share of {store} ({day}) is {money(v)}. Please pay {payer} back when you can."
                  for p, v in owe]
        lines = [f"{store}, {day}: {money(r['total'])}, paid by {payer} (card ending {last4(r['card_last4'])}).",
                 "Shares under house_rules.md: " + " · ".join(f"{p} {money(v)}" for p, v in shares.items()),
                 "Draft messages (NOT sent):"] + [f"“{d}”" for d in drafts] + [
                 "Nothing has been sent or paid. Copy a draft if you want to send it yourself."]
        return {"outcome": "draft", "text": "\n".join(lines),
                "data": {"store": r["store"], "date": r["date"], "total": r["total"], "payer": payer, "shares": shares,
                         "settlement": bal["settlement"], "drafts": drafts, "sent": False,
                         "items": [{"name": it["name"], "line_total": it["line_total"], "assigned_to": who[k]}
                                   for k, it in enumerate(r["items"])]}}
