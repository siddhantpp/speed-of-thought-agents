"""Deterministic helpers for the class-notes workflow: no model calls in this file.

  parse_vtt        WebVTT (Zoom "Name: text", Teams <v Name> tags, NOTE blocks, cue settings) -> cues
  slide_files      PowerPoint/Google Slides exports in natural order (Slide2 before Slide10)
  quote_in         the verbatim-quote check every "lecturer added" point must pass
  quote_mismatch   ...and the retry message when it fails: where the quote stops matching
  spoken_form      slide text as it is said aloud ('β1 + β3·x2' -> 'beta 1 plus beta 3 times x2')
  similarity       slide-vs-speech scores: embedding cosine + IDF-weighted word overlap, each standardized per slide
  align            mostly-monotonic dynamic programming: which slide was on screen for each cue
  time_ranges      cue labels -> [[start, end], ...] per slide

Everything here is unit-tested in tests/test_class_notes.py without a model server.
"""
import html
import math
import re
from pathlib import Path

# ---------------------------------------------------------------- timestamps
def parse_ts(text):
    """'01:02:03.500', '02:03.500' or SRT-style '00:02:03,500' -> seconds."""
    parts = text.strip().replace(",", ".").split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def fmt_ts(sec):
    """seconds -> 'HH:MM:SS.mmm' (the WebVTT form used in notes.json)."""
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def short_ts(sec):
    """seconds -> 'M:SS' (or 'H:MM:SS') for people reading the notes."""
    s = int(sec)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------- WebVTT
TIMING = re.compile(r"^\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})")
VOICE = re.compile(r"<v(?:\.[\w.-]+)?\s+([^>]+)>")
# "Name: text" (Zoom). A name is 1-4 capitalized words, so "Note: ..." style sentences are rare false hits.
NAME_PREFIX = re.compile(r"^([A-Z][\w.'-]*(?: [A-Z][\w.'()-]*){0,3}):\s+")


def parse_vtt(text):
    """Parse a WebVTT transcript into cues [{start, end, speaker, text}] in time order.

    Handles the flavours lecture-capture tools write: Zoom (numbered cues, 'Name: text', long cues
    wrapped over two lines), Teams/Stream (<v Name>text</v> voice tags), header metadata, NOTE/STYLE
    blocks, cue settings after the timing, inline tags (<c>, <i>, <00:00:01.000>) and HTML entities.
    A cue without a speaker inherits the previous speaker (captions often omit repeats).
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues, speaker = [], ""
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        m = TIMING.match(lines[idx]) if idx is not None else None
        if not m:
            continue
        payload = " ".join(ln.strip() for ln in lines[idx + 1:])
        voice = VOICE.search(payload)
        if voice:
            speaker = voice.group(1).strip()
        payload = html.unescape(re.sub(r"<[^>]+>", "", payload)).strip()
        named = NAME_PREFIX.match(payload)
        if named and not voice:
            speaker, payload = named.group(1), payload[named.end():]
        if payload:
            cues.append({"start": parse_ts(m.group(1)), "end": parse_ts(m.group(2)),
                         "speaker": speaker, "text": re.sub(r"\s+", " ", payload)})
    return sorted(cues, key=lambda c: c["start"])


# ---------------------------------------------------------------- slide exports
def natural_key(name):
    """'Slide10.png' sorts after 'Slide9.png' (plain sorting puts it after Slide1)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def slide_files(folder):
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                  key=lambda p: natural_key(p.name))


# ---------------------------------------------------------------- quotes
def norm(text):
    """Lowercase, punctuation to spaces, collapse whitespace (curly quotes and dashes included)."""
    text = str(text).lower().replace("’", "'").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


def quote_in(quote, text, min_words=4, max_words=40):
    """True if `quote` is copied verbatim (ignoring case and punctuation) from `text`."""
    q = norm(quote)
    return min_words <= len(q.split()) <= max_words and f" {q} " in f" {norm(text)} "


def quote_mismatch(quote, text):
    """Why a quote failed, in words a small model can act on: where it stops matching the transcript."""
    q, words = norm(quote).split(), norm(text).split()
    if len(q) < 4:
        return f"{quote!r} is too short (use 5-30 words)"
    if len(q) > 40:
        return f"{quote[:40]!r}... is too long (use 5-30 words)"
    best = 0
    for i in range(len(words)):
        k = 0
        while k < len(q) and i + k < len(words) and words[i + k] == q[k]:
            k += 1
        best = max(best, k)
    if best == 0:
        return f"{quote!r} is not in the transcript lines"
    return (f"{quote!r} matches the transcript only up to {' '.join(q[max(0, best - 4):best])!r}; "
            f"then it says {' '.join(q[best:best + 3])!r}, which the transcript does not")


def find_quote_time(quote, cues):
    """Start time of the cue where a verbatim quote begins (for timestamps in the notes)."""
    q = norm(quote).split()
    if not q:
        return None
    words, owner = [], []
    for c in cues:
        for w in norm(c["text"]).split():
            words.append(w)
            owner.append(c["start"])
    for i in range(len(words) - len(q) + 1):
        if words[i:i + len(q)] == q:
            return owner[i]
    return None


# ---------------------------------------------------------------- alignment
STOP = set("""a an and are as at be because but by can do does for from had has have he her here him his how i
if in into is it its it's just like me more my no not now of on one or our out so some than that that's the their
them then there these they this those to too up us very was we were what when where which who why will with
would you your yeah okay um uh right well really going get got let let's see say said thing things""".split())


def tokens(text):
    """Content words for overlap scoring; numbers like 0.75 stay whole, plurals fold to singular."""
    out = []
    for w in re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", str(text).lower().replace("’", "'")):
        if w in STOP or len(w) < 2:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


SPOKEN = {"β": " beta ", "α": " alpha ", "δ": " delta ", "ε": " epsilon ", "σ": " sigma ", "μ": " mu ",
          "λ": " lambda ", "Σ": " sum ", "×": " times ", "·": " times ", "+": " plus ", "−": " minus ",
          "≈": " about ", "≥": " at least ", "≤": " at most ", "%": " percent ", "²": " squared ", "=": " equals "}


def spoken_form(text):
    """Write slide text the way it is said aloud, so 'β1 + β3·x2' can match 'beta 1 plus beta 3 times x2'."""
    for sym, word in SPOKEN.items():
        text = text.replace(sym, word)
    text = re.sub(r"\b(beta|alpha|delta|sigma|mu)(\d)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def lexical_scores(cue_texts, slide_texts):
    """IDF-weighted word overlap: words that appear on only one slide count the most."""
    slide_sets = [set(tokens(s)) for s in slide_texts]
    df = {}
    for ws in slide_sets:
        for w in ws:
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + len(slide_sets) / d) for w, d in df.items()}
    return [[sum(idf[w] for w in set(tokens(c)) & ws) / math.sqrt(1 + len(ws)) for ws in slide_sets]
            for c in cue_texts]


def zscore_columns(matrix):
    """Standardize each column (one slide) over all rows (cues). A busy slide that resembles every
    sentence of the lecture stops winning everywhere; what counts is a cue matching a slide unusually well."""
    cols = []
    for col in zip(*matrix):
        mean = sum(col) / len(col)
        sd = math.sqrt(sum((x - mean) ** 2 for x in col) / len(col)) or 1.0
        cols.append([(x - mean) / sd for x in col])
    return [list(row) for row in zip(*cols)]


def windows(cues, context=1):
    """Each cue's text with its neighbours, so one short caption still carries enough meaning."""
    texts = [c["text"] for c in cues]
    return [" ".join(texts[max(0, i - context):i + context + 1]) for i in range(len(texts))]


def similarity(cue_vecs, slide_vecs, cue_texts, slide_texts, lexical_weight=0.5):
    """Score of every (cue, slide) pair: standardized embedding cosine + standardized word overlap."""
    emb = zscore_columns([[cosine(c, s) for s in slide_vecs] for c in cue_vecs])
    lex = zscore_columns(lexical_scores(cue_texts, slide_texts))
    return [[e + lexical_weight * l for e, l in zip(er, lr)] for er, lr in zip(emb, lex)]


# Words lecturers use when they move around the deck. They only make a move cheaper, never force it.
BACK_CUE = re.compile(r"\b(go(ing)? back|flip(ping)? back|jump(ing)? back|back to (the|that|this|our) "
                      r"(earlier |previous |first |original )?(slide|chart|graph|plot|figure|formula|table|diagram|"
                      r"picture|example)|previous slide|earlier slide)")
NEXT_CUE = re.compile(r"\b(next slide|the next (slide|one)|moving on|move on|forward again|back to where we were|"
                      r"last slide|last thing|next\.|okay,? next|on this slide|here's the|this is the)\b")


def align(scores, cue_texts=None, advance=1.0, skip=4.0, back=15.0, back_cued=2.0, advance_cued=0.3, linger=0.7):
    """Which slide was on screen for each cue: a Viterbi path over a mostly-monotonic deck.

    scores[t][s] is how well cue t matches slide s (higher is better). The path starts on slide 1 and pays:
      `advance`  to move forward one slide (`advance_cued` after "next slide", "moving on", ...),
      `skip`     for every never-shown slide it jumps over (lecturers rarely skip a slide silently),
      `back`     to return to an earlier slide: expensive unless the lecturer just said so
                 ("let me go back to the chart"), because unprompted returns are rare,
      `linger`   for every cue spent on an earlier slide after a return, because returns are brief.
    The state remembers the furthest slide shown, so resuming there after a return ("okay, forward
    again") is an ordinary move; hopping to some other slide already shown costs as much as going back. Cue phrases count for the cue itself and the one after it, since
    people say "let me go back" a moment before they click. Returns one 0-based slide index per cue.
    """
    T, S = len(scores), len(scores[0])
    cue_texts = cue_texts or [""] * T
    NEG = float("-inf")
    # dp[s][f]: best total score with slide s on screen and slide f the furthest shown so far
    dp = [[NEG] * S for _ in range(S)]
    for s in range(S):
        dp[s][s] = scores[0][s] - skip * s
    pointers = []
    for t in range(1, T):
        text = (cue_texts[t - 1] + " " + cue_texts[t]).lower()
        c_back = back_cued if BACK_CUE.search(text) else back
        c_adv = advance_cued if NEXT_CUE.search(text) else advance
        new = [[NEG] * S for _ in range(S)]
        ptr = [[None] * S for _ in range(S)]
        for sp in range(S):
            for fp in range(sp, S):
                prev = dp[sp][fp]
                if prev == NEG:
                    continue
                for s in range(S):
                    if s == sp:                       # stay
                        f, cost = fp, 0.0
                    elif s == fp or (s == sp + 1 and sp == fp):   # next slide, or resume after a return
                        f, cost = max(fp, s), c_adv
                    elif s > fp:                      # jump ahead, skipping slides never shown
                        f, cost = s, c_adv + skip * (s - fp - 1)
                    else:                             # back, or hopping among slides already shown
                        f, cost = fp, c_back
                    val = prev - cost - (linger if s < f else 0.0) + scores[t][s]
                    if val > new[s][f]:
                        new[s][f], ptr[s][f] = val, (sp, fp)
        dp = new
        pointers.append(ptr)
    # finish: slides never shown at the end are also skips
    best, state = NEG, None
    for s in range(S):
        for f in range(s, S):
            val = dp[s][f] - skip * (S - 1 - f)
            if val > best:
                best, state = val, (s, f)
    path = [state[0]]
    for ptr in reversed(pointers):
        state = ptr[state[0]][state[1]]
        path.append(state[0])
    return path[::-1]


def runs(labels):
    """[2, 2, 3, 3, 2] -> [(2, 0, 1), (3, 2, 3), (2, 4, 4)]: (label, first index, last index)."""
    out = []
    for i, lab in enumerate(labels):
        if out and out[-1][0] == lab:
            out[-1][2] = i
        else:
            out.append([lab, i, i])
    return [tuple(r) for r in out]


def time_ranges(labels, cues):
    """Cue labels -> {label: [[start, end], ...]} in 'HH:MM:SS.mmm' (one range per visit)."""
    out = {}
    for lab, i, j in runs(labels):
        out.setdefault(lab, []).append([fmt_ts(cues[i]["start"]), fmt_ts(cues[j]["end"])])
    return out


def accuracy(pred, gold, cues):
    """Share of speaking time assigned to the right slide."""
    total = sum(c["end"] - c["start"] for c in cues)
    right = sum(c["end"] - c["start"] for p, g, c in zip(pred, gold, cues) if p == g)
    return right / total if total else 0.0
