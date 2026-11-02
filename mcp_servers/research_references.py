"""References and evidence checks, in code: APA 7 from metadata, verbatim quotes, [n] citations.

A model never types a reference. It cites [n]; code maps n to a fetched record and formats that record.
"""
import re

SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor", "of", "on", "or",
         "over", "per", "the", "to", "up", "via", "vs", "with", "versus", "than", "is", "are", "be"}
# A citation is [3], [2, 5] or [2-4]; the en dash is accepted too.
CITE = re.compile(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]")
QUOTE = re.compile(r"[\"“]([^\"“”]+)[\"”]")


def norm(text):
    """Compare text the way a careful reader would: case, punctuation and spacing don't matter."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower().replace("’", "'").replace("'", ""))).strip()


def is_verbatim(quote, text, min_words=1):
    """True if the quote is copied word for word. An ellipsis splits it: each piece must be verbatim."""
    pieces = [norm(p) for p in re.split(r"\.\.\.|\u2026", quote) if norm(p)]
    hay = f" {norm(text)} "
    return sum(len(p.split()) for p in pieces) >= min_words and all(f" {p} " in hay for p in pieces)


def citation_numbers(text):
    """All cited numbers in order of appearance; [2-4] expands to 2, 3, 4."""
    out = []
    for m in CITE.finditer(text):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            if re.search(r"[–-]", part):
                a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part))
                out += list(range(a, b + 1)) if 0 < b - a < 20 else [a, b]
            else:
                out.append(int(part))
    return out


def renumber(text, mapping):
    """Rewrite every citation group with new numbers (old -> new), sorted and de-duplicated."""
    def sub(m):
        nums = sorted({mapping[n] for n in citation_numbers(m.group(0)) if n in mapping})
        return "[" + ", ".join(map(str, nums)) + "]" if nums else m.group(0)
    return CITE.sub(sub, text)


def quoted_spans(text, min_words=5):
    """Text inside double quotes that is long enough to be a quotation (not a scare quote)."""
    return [q.strip() for q in QUOTE.findall(text) if len(q.split()) >= min_words]


def sentence_case(title, abstract=""):
    """APA wants sentence case. Titles already in sentence case are kept as published; in Title Case
    ones, a capitalised word stays capitalised only if it is an acronym, has inner capitals or digits,
    or the abstract capitalises it mid-sentence (so 'Mueller' stays, 'Keyboard' goes)."""
    tokens = title.split()
    capped = [t for t in tokens[1:] if t[:1].isalpha() and t.lower().strip(",:;") not in SMALL]
    if not capped or sum(t[0].isupper() for t in capped) / len(capped) < 0.6:
        return re.sub(r": ([a-z])", lambda m: ": " + m.group(1).upper(), title)   # APA: subtitle starts upper
    proper = set(re.findall(r"(?<![.!?]\s)(?<=[\s(])([A-Z][a-z’']+)", abstract or ""))
    def lower(part):   # one piece of a hyphenated word: keep acronyms, inner capitals, digits, proper nouns
        core = part.strip("()[],:;?!\"'")
        if (core.isupper() and len(core) > 1) or any(c.isdigit() for c in core) or any(c.isupper() for c in core[1:]) \
                or core in proper or core.split("\u2019")[0] in proper:
            return part
        return part[:1].lower() + part[1:]
    out, start = [], True
    for t in tokens:
        out.append(t if start else "-".join(lower(p) for p in t.split("-")))
        start = t.endswith((":", "?", "!"))      # not '.': in titles that is an abbreviation like 'vs.'
    return " ".join(out)


def initials(given):
    """'Pam A.' -> 'P. A.', 'Jean-Paul' -> 'J.-P.'"""
    parts = []
    for name in given.replace(".", ". ").split():
        sub = [s for s in name.split("-") if s]
        if sub and sub[0][:1].isalpha():
            parts.append("-".join(s[0].upper() + "." for s in sub))
    return " ".join(parts)


def author_list(authors):
    names = [f"{a['family']}, {initials(a.get('given', ''))}".rstrip(", ") for a in authors if a.get("family")]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", . . . " + names[-1]          # APA 7: first 19, ellipsis, last


def apa(w):
    """APA 7 journal-article reference in Markdown (italic journal and volume)."""
    authors = author_list(w.get("authors") or [])
    year = w.get("year") or "n.d."
    title = sentence_case((w.get("title") or "Untitled").rstrip("."), w.get("abstract", ""))
    title += "" if title.endswith(("?", "!")) else "."
    venue = w.get("venue")
    where = ""
    if venue:
        where = f" *{venue}*"
        if w.get("volume"):
            where += f", *{w['volume']}*" + (f"({w['issue']})" if w.get("issue") else "")
        if w.get("pages"):
            where += f", {w['pages']}"
        elif w.get("article_number"):
            where += f", Article {w['article_number']}"
        where += "."
    doi = f" https://doi.org/{w['doi']}" if w.get("doi") else ""
    head = f"{authors} ({year})." if authors else f"{title} ({year})."
    return f"{head} {title if authors else ''}{where}{doi}".replace("  ", " ").strip()
