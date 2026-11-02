"""Deterministic slide tools: deck spec (JSON) -> HTML with inline SVG -> PDF, plus the checks.

Models write words and small structured specs; this file does everything with one right answer:
  layout          one 1280x720 (16:9) <section class="slide"> per slide, a clean neutral theme
  charts          drawn from data.csv by code (the model picks the chart type and the columns),
                  so every value is exact; each bar carries data-label / data-series / data-value
  diagrams        laid out by code from a model-written node/edge list (layered, left to right)
  checks          words and bullets per slide, every number traceable to the sources, chart values
                  against data.csv, text overflow (rendered in headless Chrome), PDF pages = slides
  export          headless Chrome --print-to-pdf, one slide per page

Standard library only (plus a local Chrome), so the same file runs in any folder with plain python3:
  python3 slide_tools.py export deck.html deck.pdf
  python3 slide_tools.py check  deck.html [--pdf deck.pdf]     # sources: brief.md, notes.md, data.csv beside it
  python3 slide_tools.py render deck.json deck.html [--data data.csv]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

W, H = 1280, 720                      # one slide, CSS pixels (13.33 x 7.5 in, 16:9)
WPM = 140                             # speaking rate used to turn minutes into a notes budget
INK, INK2, MUTED, GRID, SURFACE = "#1d1d1b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (fixed order, never cycled)
OTHER = "#b9b7af"                     # non-highlighted bars
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- data, numbers, words
def read_table(path):
    """data.csv -> {"columns", "rows" (raw strings), "numeric", "labels"} (labels = text columns)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys()) if rows else []
    numeric = [c for c in cols if all(_is_number(r[c]) for r in rows)]
    return {"columns": cols, "rows": rows, "numeric": numeric, "labels": [c for c in cols if c not in numeric]}


def _is_number(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


# A number is a run of digits, optionally with thousands commas and a decimal part. The lookbehind skips
# digits glued to letters ("Q3", "L4", "y1"), which are names, not quantities.
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def canon(s):
    """'1,250' -> '1250', '4.50' -> '4.5', '07' -> '7' (so '$1,250' on a slide matches '1250' in the CSV)."""
    v = float(str(s).replace(",", ""))
    return str(int(v)) if v == int(v) else f"{v:.10g}"


def numbers_in(text):
    return [canon(a + (b or "")) for a, b in NUMBER.findall(text or "")]


def source_numbers(paths):
    """Every number the deck may use. Struck-out text (~~like this~~) in the notes does not count."""
    found = set()
    for p in paths:
        text = re.sub(r"~~.*?~~", " ", Path(p).read_text(), flags=re.S)
        found |= set(numbers_in(text))
    return found


def count_words(text):
    return len(re.findall(r"[\w$%][\w$%'’./-]*", text or ""))


def pretty(col):
    """Column name -> axis/legend label: 'avg_trip_minutes' -> 'Avg trip minutes', 'cost_usd' -> 'Cost (USD)'."""
    words = col.replace("_", " ").strip()
    words = re.sub(r"\busd$", "(USD)", words)
    return words[:1].upper() + words[1:]


def fmt_value(raw, col):
    s = str(raw).strip()
    if col.endswith("_usd"):
        return "$" + f"{float(s):,.0f}" if float(s) == int(float(s)) else "$" + f"{float(s):,.2f}"
    return s


def wrap(text, max_chars):
    """Greedy word wrap for SVG text (SVG has no automatic line breaking)."""
    lines, cur = [], ""
    for word in str(text).split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


def esc(s):
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- charts (code draws; values are exact)
def normalize_chart(chart, table):
    """Validate a model's chart choice against the real table. Returns (chart with resolved data, problems)."""
    c = dict(chart)
    problems = []
    if c.get("x") not in table["labels"]:
        problems.append(f"chart x must be a text column, one of {table['labels']}")
    ys = [y for y in dict.fromkeys(c.get("y") or [])]
    bad = [y for y in ys if y not in table["numeric"]]
    if bad or not ys:
        problems.append(f"chart y must list 1-3 numeric columns from {table['numeric']}")
    ys = [y for y in ys if y in table["numeric"]][:3]
    # Fixing the type is a lookup, not a judgment: one series is a bar chart, several are grouped.
    kind = c.get("type", "bar")
    if len(ys) > 1:
        kind = "grouped_bar"
    elif kind == "grouped_bar":
        kind = "bar"
    if len(ys) > 1:   # one axis only: series on very different scales would need two
        tops = {y: max(float(r[y]) for r in table["rows"]) for y in ys}
        if max(tops.values()) > 5 * max(min(tops.values()), 1e-9):
            problems.append(f"chart columns {ys} are on very different scales ({tops}); chart only one of them "
                            "or columns with the same unit")
    if count_words(c.get("title", "")) > 8:
        problems.append("chart title is longer than 8 words")
    labels = [r[c["x"]] for r in table["rows"]] if c.get("x") in table["columns"] else []
    if c.get("highlight") and c["highlight"] not in labels:
        problems.append(f"chart highlight must be empty or one of {labels}")
        c["highlight"] = ""
    c.update(type=kind, y=ys)
    c["data"] = [{"label": r[c["x"]], "series": y, "value": r[y]} for y in ys for r in table["rows"]] if labels else []
    return c, problems


def nice_max(v):
    if v <= 0:
        return 1
    exp = 10 ** math.floor(math.log10(v))
    return next(m * exp for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if v <= m * exp + 1e-9)


def _bar_path(x, y, w, h, horizontal=False, r=4):
    """A bar with a 4px rounded data end and a square baseline end."""
    r = min(r, w / 2, h / 2) if h > 0 and w > 0 else 0
    if horizontal:   # baseline on the left
        return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.1f}"
                f"a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z")
    return (f"M{x:.1f},{y + h:.1f}v-{h - r:.1f}a{r},{r} 0 0 1 {r},-{r}h{w - 2 * r:.1f}"
            f"a{r},{r} 0 0 1 {r},{r}v{h - r:.1f}z")


def chart_svg(chart, table, width=640, height=420):
    """Draw a bar, horizontal bar or grouped bar chart as SVG from the table. Every mark carries its value."""
    x, ys, kind, hl = chart["x"], chart["y"], chart["type"], chart.get("highlight", "")
    rows = table["rows"]
    top = nice_max(max(float(r[y]) for r in rows for y in ys))
    ticks = [top * i / 4 for i in range(5)]
    out = [f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{esc(chart.get("title", ""))}" font-family="{FONT}">']

    def color(row, si):
        if len(ys) > 1:
            return SERIES[si]
        return SERIES[0] if not hl or row[x] == hl else OTHER

    def tick_label(t):
        return fmt_value(canon(t), ys[0]) if len(ys) == 1 else canon(t)

    if len(ys) > 1:   # legend: a series needs a name next to its color (identity is never color alone)
        lx = 8
        for si, y in enumerate(ys):
            out.append(f'<rect x="{lx}" y="6" width="16" height="16" rx="3" fill="{SERIES[si]}"/>'
                       f'<text x="{lx + 24}" y="20" font-size="18" fill="{INK2}">{esc(pretty(y))}</text>')
            lx += 44 + len(pretty(y)) * 10
    if kind == "hbar":
        left, right, top_m, bottom = 210, 84, 16, 40
        pw, ph = width - left - right, height - top_m - bottom
        band = ph / len(rows)
        bh = min(band * 0.6, 64)
        for t in ticks:
            gx = left + pw * t / top
            out.append(f'<line x1="{gx:.1f}" y1="{top_m}" x2="{gx:.1f}" y2="{top_m + ph}" stroke="{GRID}"/>'
                       f'<text x="{gx:.1f}" y="{height - 12}" font-size="16" fill="{MUTED}" text-anchor="middle">'
                       f'{esc(tick_label(t))}</text>')
        for i, r in enumerate(rows):
            v = float(r[ys[0]])
            cy, bw = top_m + band * i + band / 2, pw * v / top
            out.append(f'<path class="bar" d="{_bar_path(left, cy - bh / 2, bw, bh, horizontal=True)}" '
                       f'fill="{color(r, 0)}" data-label="{esc(r[x])}" data-series="{esc(ys[0])}" '
                       f'data-value="{esc(r[ys[0]])}"/>')
            lines, bold = wrap(r[x], 16)[:2], ' font-weight="700"' if r[x] == hl else ""
            for k, line in enumerate(lines):
                out.append(f'<text x="{left - 12}" y="{cy + 7 + (k - (len(lines) - 1) / 2) * 23:.1f}" font-size="20" '
                           f'fill="{INK}" text-anchor="end"{bold}>{esc(line)}</text>')
            out.append(f'<text x="{left + bw + 8:.1f}" y="{cy + 8:.1f}" font-size="22" font-weight="700" '
                       f'fill="{INK}">{esc(fmt_value(r[ys[0]], ys[0]))}</text>')
    else:
        left, right, top_m, bottom = 64, 12, 48 if len(ys) > 1 else 32, 70
        pw, ph = width - left - right, height - top_m - bottom
        band = pw / len(rows)
        group_w = min(band * 0.62, 150 if len(ys) == 1 else 220)
        bw = (group_w - 2 * (len(ys) - 1)) / len(ys)            # 2px surface gap between touching bars
        for t in ticks:
            gy = top_m + ph - ph * t / top
            out.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width - right}" y2="{gy:.1f}" stroke="{GRID}"/>'
                       f'<text x="{left - 10}" y="{gy + 5:.1f}" font-size="16" fill="{MUTED}" text-anchor="end">'
                       f'{esc(tick_label(t))}</text>')
        for i, r in enumerate(rows):
            gx = left + band * i + (band - group_w) / 2
            for si, y in enumerate(ys):
                v = float(r[y])
                bx, bh = gx + si * (bw + 2), ph * v / top
                out.append(f'<path class="bar" d="{_bar_path(bx, top_m + ph - bh, bw, bh)}" fill="{color(r, si)}" '
                           f'data-label="{esc(r[x])}" data-series="{esc(y)}" data-value="{esc(r[y])}"/>')
                out.append(f'<text x="{bx + bw / 2:.1f}" y="{top_m + ph - bh - 8:.1f}" font-size="{22 if len(ys) == 1 else 18}" '
                           f'font-weight="700" fill="{INK}" text-anchor="middle">{esc(fmt_value(r[y], y))}</text>')
            bold = ' font-weight="700"' if r[x] == hl else ""
            for k, line in enumerate(wrap(r[x], 14)[:2]):
                out.append(f'<text x="{gx + group_w / 2:.1f}" y="{top_m + ph + 26 + k * 23}" font-size="20" fill="{INK}" '
                           f'text-anchor="middle"{bold}>{esc(line)}</text>')
        out.append(f'<line x1="{left}" y1="{top_m + ph}" x2="{width - right}" y2="{top_m + ph}" stroke="{MUTED}"/>')
    return "\n".join(out + ["</svg>"])


# ---------------------------------------------------------------- diagrams (model writes nodes/edges, code lays out)
def normalize_diagram(d, max_nodes=6, max_label=40):
    nodes, edges = list(d.get("nodes") or []), list(d.get("edges") or [])
    ids = [n.get("id") for n in nodes]
    problems = []
    if not 2 <= len(nodes) <= max_nodes:
        problems.append(f"a diagram needs 2 to {max_nodes} nodes (got {len(nodes)})")
    if len(set(ids)) != len(ids):
        problems.append("diagram node ids must be unique")
    long = [n["label"] for n in nodes if len(n.get("label", "")) > max_label or not n.get("label", "").strip()]
    if long:
        problems.append(f"diagram labels must be 1-{max_label} characters: {long[:2]}")
    bad = [e for e in edges if e.get("from") not in ids or e.get("to") not in ids or e.get("from") == e.get("to")]
    if bad:
        problems.append(f"diagram edges must join two different node ids from {ids}: {bad[:2]}")
    if any(count_words(e.get("label", "")) > 3 or len(e.get("label", "")) > 16 for e in edges):
        problems.append("diagram edge labels must be at most 3 words and 16 characters")
    if nodes and edges and len(edges) > 8:
        problems.append("a diagram has at most 8 edges")
    return {"nodes": nodes, "edges": [e for e in edges if e not in bad]}, problems


def layer(nodes, edges):
    """Rank = longest path from a source (a layered left-to-right layout). Edges that would close a cycle
    are kept for drawing but ignored for ranking."""
    ids = [n["id"] for n in nodes]
    succ = {i: [] for i in ids}

    def reaches(a, b, seen=()):
        return a == b or any(reaches(c, b, seen + (a,)) for c in succ[a] if c not in seen)
    for e in edges:
        if not reaches(e["to"], e["from"]):
            succ[e["from"]].append(e["to"])
    rank = {i: 0 for i in ids}
    for _ in ids:   # relax |V| times: fine for 6 nodes
        for a in ids:
            for b in succ[a]:
                rank[b] = max(rank[b], rank[a] + 1)
    return rank


def diagram_svg(diagram, width=760, height=460, uid="d"):
    nodes, edges = diagram["nodes"], diagram["edges"]
    rank = layer(nodes, edges)
    cols = max(rank.values()) + 1
    col_w = width / cols
    # Box size comes from the labels: leave each gap wide enough for its arrow label, then fit the longest word.
    label_w = max((len(e["label"]) * 9.6 + 14 for e in edges if e.get("label", "").strip()), default=0)
    gap = max(76, label_w + 16) if cols > 1 else 40
    box_w = min(col_w - gap, 300)
    longest = max(len(w) for n in nodes for w in n["label"].split())
    font = max(15.0, min(26.0 if width >= 1000 else 22.0, (box_w - 24) / (0.56 * longest)))
    chars, line_h = max(longest, int((box_w - 24) / (font * 0.56))), font * 1.25
    boxes = {}
    for c in range(cols):
        members = [n for n in nodes if rank[n["id"]] == c]
        heights = [len(wrap(n["label"], chars)) * line_h + 26 for n in members]
        gap_y = 28
        y = (height - sum(heights) - gap_y * (len(members) - 1)) / 2
        for n, h in zip(members, heights):
            boxes[n["id"]] = (col_w * c + (col_w - box_w) / 2, y, box_w, h)
            y += h + gap_y
    sinks = {n["id"] for n in nodes} - {e["from"] for e in edges}
    out = [f'<svg class="diagram" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
           f'font-family="{FONT}"><defs><marker id="arrow-{uid}" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="{INK2}"/>'
           f'</marker></defs>']
    labels = []
    for e in edges:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes[e["from"]], boxes[e["to"]]
        if rank[e["to"]] > rank[e["from"]]:      # forward: right edge -> left edge, a gentle S-curve
            ax, ay, bx, by = x1 + w1, y1 + h1 / 2, x2 - 2, y2 + h2 / 2
            mx = (ax + bx) / 2
            d = f"M{ax:.1f},{ay:.1f}C{mx:.1f},{ay:.1f} {mx:.1f},{by:.1f} {bx:.1f},{by:.1f}"
        else:                                     # same column or a back edge: bottom -> top, or a loop below
            ax, ay = x1 + w1 / 2, y1 + h1
            bx, by = x2 + w2 / 2, (y2 - 2 if y2 > y1 else y2 + h2 + 2)
            d = (f"M{ax:.1f},{ay:.1f}L{bx:.1f},{by:.1f}" if y2 > y1 else
                 f"M{ax:.1f},{ay:.1f}C{ax:.1f},{height - 4} {bx:.1f},{height - 4} {bx:.1f},{by:.1f}")
        out.append(f'<path class="edge" d="{d}" fill="none" stroke="{INK2}" stroke-width="2" '
                   f'marker-end="url(#arrow-{uid})" data-from="{esc(e["from"])}" data-to="{esc(e["to"])}"/>')
        if e.get("label", "").strip():
            lx, ly = (ax + bx) / 2, (ay + by) / 2
            tw = len(e["label"]) * 9.6 + 14
            labels.append(f'<rect x="{lx - tw / 2:.1f}" y="{ly - 13:.1f}" width="{tw:.1f}" height="24" rx="4" '
                          f'fill="{SURFACE}"/><text x="{lx:.1f}" y="{ly + 6:.1f}" font-size="17" fill="{INK2}" '
                          f'text-anchor="middle">{esc(e["label"])}</text>')
    for n in nodes:
        x, y, w, h = boxes[n["id"]]
        stroke = SERIES[0] if n["id"] in sinks else MUTED
        out.append(f'<g class="node" data-id="{esc(n["id"])}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                   f'height="{h:.1f}" rx="10" fill="#ffffff" stroke="{stroke}" stroke-width="2"/>')
        for k, line in enumerate(wrap(n["label"], chars)):
            out.append(f'<text x="{x + w / 2:.1f}" y="{y + 13 + font * 0.95 + k * line_h:.1f}" font-size="{font:.1f}" '
                       f'fill="{INK}" text-anchor="middle">{esc(line)}</text>')
        out.append("</g>")
    return "\n".join(out + labels + ["</svg>"])


# ---------------------------------------------------------------- deck HTML
# Sizes follow the usual projection minimums: body text >= 24 pt (32 px on a 1280 px slide), titles 36 pt.
CSS = f"""
@page {{ size: {W}px {H}px; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: #e9e8e4; }}
body {{ font-family: {FONT}; color: {INK}; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.slide {{ position: relative; width: {W}px; height: {H}px; overflow: hidden; margin: 24px auto; background: {SURFACE};
  padding: 56px 72px 64px; display: flex; flex-direction: column; break-after: page; page-break-after: always;
  box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
.slide:last-of-type {{ break-after: auto; page-break-after: auto; }}
.slide::before {{ content: ""; position: absolute; left: 72px; top: 36px; width: 56px; height: 6px;
  border-radius: 3px; background: {SERIES[0]}; }}
h1, h2 {{ margin: 0; font-weight: 700; letter-spacing: -0.01em; }}
h2.title {{ font-size: 48px; line-height: 1.15; max-height: 2.3em; overflow: hidden; margin: 0 0 30px; flex: none; }}
.text {{ flex: 1 1 auto; min-height: 0; overflow: hidden; font-size: 36px; line-height: 1.3; }}
.text ul {{ margin: 0; padding-left: 1.05em; }}
.text li {{ margin: 0 0 18px; }}
.text li::marker {{ color: {SERIES[0]}; }}
.cols {{ flex: 1 1 auto; min-height: 0; display: flex; gap: 44px; }}
.cols .text {{ flex: 0 0 440px; font-size: 32px; }}
.stack {{ flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; gap: 6px; }}
.stack .text {{ flex: none; max-height: 196px; font-size: 32px; }}
.stack .text li {{ margin-bottom: 10px; }}
figure.visual {{ flex: 1 1 auto; min-width: 0; min-height: 0; margin: 0; display: flex; flex-direction: column; }}
figure.visual figcaption {{ flex: none; font-size: 24px; font-weight: 600; color: {INK2}; margin: 0 0 10px; }}
figure.visual svg {{ flex: 1 1 auto; min-height: 0; width: 100%; }}
.layout-title {{ justify-content: center; padding-left: 96px; }}
.layout-title::before {{ display: none; }}
.layout-title h1::before {{ content: ""; display: block; width: 96px; height: 6px; border-radius: 3px;
  background: {SERIES[0]}; margin: 0 0 30px; }}
.layout-title h1 {{ font-size: 64px; line-height: 1.12; max-height: 3.9em; overflow: hidden; margin: 0 0 28px; }}
.layout-title .text {{ flex: none; font-size: 32px; color: {INK2}; }}
.layout-title .text ul {{ list-style: none; padding: 0; }}
.layout-title .text li {{ margin: 0 0 8px; }}
.slide-number {{ position: absolute; right: 44px; bottom: 26px; font-size: 18px; color: {MUTED}; }}
.notes {{ display: none; }}
body.show-notes .slide {{ overflow: visible; margin-bottom: 300px; }}
body.show-notes .notes {{ display: block; position: absolute; top: {H + 12}px; left: 0; right: 0; background: #fff;
  border-left: 4px solid {SERIES[0]}; padding: 12px 18px; font-size: 19px; line-height: 1.45; color: {INK2}; }}
@media print {{ html, body {{ background: none; }} .slide {{ margin: 0; box-shadow: none; }} }}
"""
# Press "n" to show the speaker notes under each slide (screen only; the PDF never shows them).
NOTES_TOGGLE = ("<script>document.addEventListener('keydown',function(e){if(e.key==='n')"
                "document.body.classList.toggle('show-notes')});</script>")


def layout_of(slide):
    if slide.get("layout"):
        return slide["layout"]
    if slide.get("n") == 1:
        return "title"
    return "visual" if (slide.get("visual") or {}).get("kind") in ("chart", "diagram") else "bullets"


def visual_layout(slide):
    """Where the visual goes: 'side' (bullets left, chart right), 'stack' (bullets above a full-width diagram:
    a left-to-right flow needs the width) or 'wide' (no bullets)."""
    if not slide.get("bullets"):
        return "wide"
    return "stack" if (slide.get("visual") or {}).get("kind") == "diagram" else "side"


def visual_svg(slide, table, where=None):
    v, where = slide.get("visual") or {}, where or visual_layout(slide)
    c = v.get("chart") or {}
    if v.get("kind") == "chart" and table and c.get("y") and c.get("x") in table["columns"]:
        return chart_svg(c, table, 640 if where == "side" else 1100, 420)
    if v.get("kind") == "diagram" and (v.get("diagram") or {}).get("nodes"):
        return diagram_svg(v["diagram"], 760 if where == "side" else 1100, 330 if where == "stack" else 460,
                           uid=f"s{slide.get('n', 0)}")
    return ""


def slide_html(slide, table=None):
    n, lay = slide.get("n", 0), layout_of(slide)
    bullets = "".join(f"<li>{esc(b)}</li>" for b in slide.get("bullets", []))
    notes = "".join(f"<p>{esc(p.strip())}</p>" for p in str(slide.get("notes", "")).split("\n\n") if p.strip())
    parts = [f'<section class="slide layout-{lay}" id="slide-{n}" data-n="{n}">']
    if lay == "title":
        parts.append(f'<h1 class="title">{esc(slide.get("title", ""))}</h1>')
        parts.append(f'<div class="text"><ul>{bullets}</ul></div>' if bullets else "")
    else:
        parts.append(f'<h2 class="title">{esc(slide.get("title", ""))}</h2>')
        if lay == "visual":
            v = slide.get("visual") or {}
            cap = (v.get("chart") or {}).get("title", "") if v.get("kind") == "chart" else v.get("caption", "")
            where = visual_layout(slide)
            fig = (f'<figure class="visual">{f"<figcaption>{esc(cap)}</figcaption>" if cap else ""}'
                   f'{visual_svg(slide, table, where)}</figure>')
            text = f'<div class="text"><ul>{bullets}</ul></div>' if bullets else ""
            parts.append(f'<div class="{"stack" if where == "stack" else "cols"}">{text}{fig}</div>')
        else:
            parts.append(f'<div class="text"><ul>{bullets}</ul></div>')
    parts.append(f'<aside class="notes">{notes}</aside>')
    if lay != "title":
        parts.append(f'<div class="slide-number">{n}</div>')
    return "".join(parts) + "</section>"


def render_deck(spec, table=None):
    """Deck spec -> one HTML file. spec = {"title", "slides": [{"n", "title", "bullets", "notes", "visual"}]}."""
    body = "\n".join(slide_html(s, table) for s in spec["slides"])
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>{esc(spec.get("title", "Slides"))}'
            f'</title>\n<meta name="viewport" content="width={W}"><style>{CSS}</style></head>\n<body>\n{body}\n'
            f'{NOTES_TOGGLE}\n</body></html>\n')


# ---------------------------------------------------------------- reading any deck HTML back (for checks and scoring)
class _DeckParser(HTMLParser):
    """Splits a deck into slides: visible text, speaker notes, bullets, chart marks. Works on any deck that
    uses <section class="slide"> (or bare <section>s), <aside class="notes"> and <svg class="chart|diagram">."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    HIDDEN = {"script", "style", "template", "head", "title", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.slides, self.stack = [], []

    def _flags(self, tag, attrs):
        cls = (attrs.get("class") or "").split()
        top = self.stack[-1][1] if self.stack else {}
        f = dict(top)
        if tag in self.HIDDEN:
            f["hidden"] = True
        if "notes" in cls:
            f["notes"] = True
        if "slide-number" in cls:
            f["number"] = True
        if tag == "svg" and "svg" not in f:
            f["svg"] = "chart" if "chart" in cls else "diagram" if "diagram" in cls else "other"
            if f.get("slide") is not None and not f.get("notes"):
                self.slides[f["slide"]][f["svg"] + "s"] += 1
        if tag == "section" and f.get("slide") is None:
            self.slides.append({"is_slide_class": "slide" in cls, "visible": [], "notes": [], "svg_text": [],
                                "chart_text": [], "bullets": 0, "marks": [], "charts": 0, "diagrams": 0, "others": 0})
            f["slide"] = len(self.slides) - 1
        return f

    def _attrs(self, f, tag, attrs):
        s = f.get("slide")
        if s is None or f.get("notes") or f.get("hidden"):
            return
        if tag == "li" and "svg" not in f:
            self.slides[s]["bullets"] += 1
        if f.get("svg") == "chart" and attrs.get("data-value") is not None:
            self.slides[s]["marks"].append({"label": attrs.get("data-label", ""), "series": attrs.get("data-series", ""),
                                            "value": attrs.get("data-value")})

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        f = self._flags(tag, attrs)
        self._attrs(f, tag, attrs)
        self.handle_data(" ")
        if tag not in self.VOID:
            self.stack.append((tag, f))

    def handle_startendtag(self, tag, attrs):
        attrs = dict(attrs)
        self._attrs(self._flags(tag, attrs), tag, attrs)

    def handle_endtag(self, tag):
        if any(t == tag for t, _ in self.stack):
            while self.stack and self.stack.pop()[0] != tag:
                pass
        self.handle_data(" ")

    def handle_data(self, data):
        f = self.stack[-1][1] if self.stack else {}
        s = f.get("slide")
        if s is None or f.get("hidden"):
            return
        sl = self.slides[s]
        if f.get("notes"):
            sl["notes"].append(data)
        elif f.get("svg") == "chart":
            sl["chart_text"].append(data)
        elif f.get("svg"):
            sl["svg_text"].append(data)
        elif not f.get("number"):
            sl["visible"].append(data)


def parse_deck(text):
    """deck.html -> [{"visible", "notes", "svg_text", "chart_text", "bullets", "marks", "charts", "diagrams"}]."""
    p = _DeckParser()
    p.feed(text)
    slides = p.slides
    if any(s["is_slide_class"] for s in slides):
        slides = [s for s in slides if s["is_slide_class"]]
    for s in slides:
        for k in ("visible", "notes", "svg_text", "chart_text"):
            s[k] = re.sub(r"\s+", " ", "".join(s[k])).strip()
    return slides


# ---------------------------------------------------------------- headless Chrome: overflow and PDF
class ChromeMissing(RuntimeError):
    pass


def find_chrome():
    for c in (os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium", shutil.which("google-chrome"),
              shutil.which("google-chrome-stable"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if c and Path(c).exists():
            return c
    raise ChromeMissing("Google Chrome or Chromium not found (set CHROME=/path/to/chrome)")


def _chrome(args, timeout=90):
    with tempfile.TemporaryDirectory() as profile:   # a throwaway profile: never touches the user's Chrome
        cmd = [find_chrome(), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               "--disable-extensions", "--use-mock-keychain", "--hide-scrollbars", f"--window-size={W},{H}",
               f"--user-data-dir={profile}", *args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# Injected into a copy of the deck: for every slide, find text that is clipped by its box or that runs off the
# slide, and write the worst case per slide into a body attribute that --dump-dom hands back.
OVERFLOW_JS = r"""<script>(function () {
  var all = [].slice.call(document.querySelectorAll('section.slide'));
  if (!all.length) all = [].slice.call(document.querySelectorAll('section'));
  var slides = all.filter(function (s) { return !s.parentElement.closest('section'); });
  var report = [];
  slides.forEach(function (s, i) {
    var sr = s.getBoundingClientRect(), worst = null;
    function note(el, px, axis) {
      var cs = getComputedStyle(el), lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.2;
      if (px <= Math.max(4, lh * 0.35)) return;   // glyph overhang and rounding, not a hidden line
      var cls = (typeof el.className === 'string' && el.className.trim()) ? '.' + el.className.trim().split(/\s+/).join('.') : '';
      if (!worst || px > worst.px) worst = {slide: i + 1, px: Math.round(px), axis: axis, line_px: Math.round(lh),
        lines: Math.ceil((px - 4) / lh), where: el.tagName.toLowerCase() + cls,
        scroll_px: el.scrollHeight, client_px: el.clientHeight};
    }
    if (sr.height > sr.width * 9 / 16 + 2) note(s, sr.height - sr.width * 9 / 16, 'slide is taller than 16:9');
    [s].concat([].slice.call(s.querySelectorAll('*'))).forEach(function (el) {
      if (el !== s && (el.closest('.notes') || (el.closest('svg') && el.tagName.toLowerCase() !== 'svg'))) return;
      var cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden') return;
      if (cs.overflowY !== 'visible') note(el, el.scrollHeight - el.clientHeight, 'clipped vertically');
      if (cs.overflowX !== 'visible') note(el, el.scrollWidth - el.clientWidth, 'clipped horizontally');
      var own = [].some.call(el.childNodes, function (c) { return c.nodeType === 3 && c.textContent.trim(); });
      if (el !== s && (own || el.tagName.toLowerCase() === 'svg' || el.tagName.toLowerCase() === 'img')) {
        var r = el.getBoundingClientRect();
        if (r.width && r.height) note(el, Math.max(r.bottom - sr.bottom, r.right - sr.right), 'runs off the slide');
      }
    });
    if (worst) report.push(worst);
  });
  document.body.setAttribute('data-overflow-report', JSON.stringify({slides: slides.length, overflow: report}));
})();</script>"""


def _inject(text, snippet):
    m = re.search(r"</body\s*>", text, flags=re.I)
    return text[:m.start()] + snippet + text[m.start():] if m else text + snippet


def overflow_report(html_path):
    """Render the deck in headless Chrome; return {"slides": n, "overflow": [{"slide", "px", "lines", "where", ...}]}."""
    src = Path(html_path).resolve()
    tmp = src.with_name(f".overflow-check-{os.getpid()}-{src.name}")   # beside the deck, so relative links work
    tmp.write_text(_inject(src.read_text(), OVERFLOW_JS))
    try:
        res = _chrome(["--dump-dom", tmp.as_uri()])
    finally:
        tmp.unlink(missing_ok=True)
    m = re.search(r'data-overflow-report="([^"]*)"', res.stdout)
    if not m:
        raise RuntimeError(f"Chrome returned no overflow report (exit {res.returncode}): {res.stderr[-300:]}")
    return json.loads(html.unescape(m.group(1)))


def overflow_of_html(text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "slide.html"
        p.write_text(text)
        return overflow_report(p)


def export_pdf(html_path, pdf_path):
    """Headless Chrome prints the deck: one 1280x720 page per slide (adds @page if the deck has none)."""
    src, out = Path(html_path).resolve(), Path(pdf_path).resolve()
    text, tmp = src.read_text(), None
    if "@page" not in text:
        tmp = src.with_name(f".print-{os.getpid()}-{src.name}")
        tmp.write_text(_inject(text, f"<style>@page {{ size: {W}px {H}px; margin: 0; }}</style>"))
    try:
        res = _chrome(["--no-pdf-header-footer", "--print-to-pdf-no-header", f"--print-to-pdf={out}",
                       (tmp or src).as_uri()], timeout=120)
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)
    if not out.exists():
        raise RuntimeError(f"Chrome did not write {out} (exit {res.returncode}): {res.stderr[-300:]}")
    return pdf_pages(out)


def pdf_pages(pdf_path):
    """Page count from the PDF's page objects (Chrome writes them uncompressed)."""
    data = Path(pdf_path).read_bytes()
    n = len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])", data))
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", data, flags=re.S)]
    return n or (max(counts) if counts else 0)


# ---------------------------------------------------------------- checks
def overflow_message(o, n_bullets):
    """Turn a measured overflow into an instruction a small model can follow."""
    where = "the title" if "title" in o["where"] or o["where"].startswith(("h1", "h2")) else "the text"
    if where == "the title":
        return f"slide {o['slide']}: the title is too long and wraps past its box ({o['lines']} extra line(s)); use at most 6 words"
    total = max(1, round(o["scroll_px"] / o["line_px"])) if o.get("scroll_px") else o["lines"] + 1
    cut = math.ceil(o["lines"] / max(total / max(n_bullets, 1), 1))
    keep = max(1, n_bullets - cut)
    return (f"slide {o['slide']} overflows by {o['lines']} line(s) of text; cut it to {keep} bullet(s)"
            + (" or shorten every bullet to about 8 words" if keep < n_bullets else ", each about 8 words"))


def check_deck(html_path, sources=(), data_csv=None, pdf=None, max_words=45, max_bullets=4,
               max_notes_words=700, min_notes_words=15, overflow=True):
    """Check any deck.html against the contract. Returns a report with a flat `problems` list."""
    slides = parse_deck(Path(html_path).read_text())
    allowed = source_numbers(sources) if sources else None
    table = read_table(data_csv) if data_csv else None
    problems, per = [], []
    for i, s in enumerate(slides, 1):
        words, notes_words = count_words(s["visible"]), count_words(s["notes"])
        untraced = sorted(set(numbers_in(s["visible"] + " " + s["svg_text"] + " " + s["notes"])) - allowed,
                          key=float) if allowed is not None else []
        per.append({"slide": i, "words": words, "bullets": s["bullets"], "notes_words": notes_words,
                    "charts": s["charts"], "diagrams": s["diagrams"], "untraceable_numbers": untraced})
        if words > max_words:
            problems.append(f"slide {i} has {words} words on the slide (max {max_words})")
        if s["bullets"] > max_bullets:
            problems.append(f"slide {i} has {s['bullets']} bullets (max {max_bullets})")
        if notes_words < min_notes_words:
            problems.append(f"slide {i} has {notes_words} words of speaker notes (min {min_notes_words})")
        if untraced:
            problems.append(f"slide {i} uses numbers not found in the sources: {untraced}")
    notes_total = sum(p["notes_words"] for p in per)
    if notes_total > max_notes_words:
        problems.append(f"speaker notes total {notes_total} words (max {max_notes_words})")
    marks = [m for s in slides for m in s["marks"]]
    mismatches = chart_mismatches(marks, table) if table else []
    problems += [f"chart value {m}" for m in mismatches]
    report = {"slides": len(slides), "per_slide": per, "notes_words_total": notes_total,
              "charts": sum(s["charts"] for s in slides), "diagrams": sum(s["diagrams"] for s in slides),
              "chart_marks": len(marks), "chart_mismatches": mismatches, "overflow": None, "pdf_pages": None}
    if overflow:
        try:
            report["overflow"] = overflow_report(html_path)["overflow"]
            problems += [f"slide {o['slide']} overflows by {o['px']}px (~{o['lines']} line(s)) at {o['where']}"
                         for o in report["overflow"]]
        except ChromeMissing as e:
            report["overflow_error"] = str(e)
    if pdf:
        report["pdf_pages"] = pdf_pages(pdf) if Path(pdf).exists() else 0
        if report["pdf_pages"] != len(slides):
            problems.append(f"{pdf} has {report['pdf_pages']} pages for {len(slides)} slides")
    report["problems"] = problems
    return report


def chart_mismatches(marks, table):
    """Every drawn mark must equal the data.csv cell for its row label and column."""
    out = []
    for m in marks:
        row = next((r for r in table["rows"] if any(_same(r[c], m["label"]) for c in table["labels"])), None)
        if row is None or m["series"] not in table["numeric"]:
            out.append(f"{m['label']!r}/{m['series']!r}={m['value']}: no such row/column in data.csv")
        elif not _is_number(m["value"]) or abs(float(str(m["value"]).replace(",", "")) - float(row[m["series"]])) > 1e-9:
            out.append(f"{m['label']!r}/{m['series']!r} drawn as {m['value']}, data.csv has {row[m['series']]}")
    return out


def _same(a, b):
    return re.sub(r"\s+", " ", str(a)).strip().lower() == re.sub(r"\s+", " ", str(b)).strip().lower()


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="deck.html -> deck.pdf with headless Chrome (one page per slide)")
    e.add_argument("html")
    e.add_argument("pdf")
    c = sub.add_parser("check", help="check a deck.html: words, bullets, notes, numbers, chart values, overflow")
    c.add_argument("html")
    c.add_argument("--pdf")
    c.add_argument("--data", help="data.csv (default: beside the deck)")
    c.add_argument("--sources", nargs="*", help="files numbers may come from (default: brief.md notes.md data.csv)")
    c.add_argument("--max-words", type=int, default=45)
    c.add_argument("--max-bullets", type=int, default=4)
    c.add_argument("--max-notes-words", type=int, default=700)
    r = sub.add_parser("render", help="deck.json -> deck.html")
    r.add_argument("json")
    r.add_argument("html")
    r.add_argument("--data")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        n = export_pdf(a.html, a.pdf)
        print(f"wrote {a.pdf}: {n} pages")
    elif a.cmd == "check":
        here = Path(a.html).resolve().parent
        data = a.data or (str(here / "data.csv") if (here / "data.csv").exists() else None)
        sources = a.sources if a.sources is not None else [str(here / f) for f in ("brief.md", "notes.md", "data.csv")
                                                             if (here / f).exists()]
        rep = check_deck(a.html, sources, data, a.pdf, a.max_words, a.max_bullets, a.max_notes_words)
        print(json.dumps(rep, indent=1))
        return 1 if rep["problems"] else 0
    else:
        spec = json.loads(Path(a.json).read_text())
        Path(a.html).write_text(render_deck(spec, read_table(a.data) if a.data else None))
        print(f"wrote {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
