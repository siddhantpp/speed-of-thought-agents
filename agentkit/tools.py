"""Deterministic tools. Rule of thumb: math, dates and money are code, not model output."""
import json
import re
import time
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

HOME_TZ = ZoneInfo("America/New_York")
TZ_ABBR = {"pt": "America/Los_Angeles", "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
           "mt": "America/Denver", "ct": "America/Chicago", "cst": "America/Chicago",
           "et": "America/New_York", "est": "America/New_York", "edt": "America/New_York"}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


class Toolbox:
    """Registry of tools a model may call, with their JSON-schema specs."""

    def __init__(self, **context):
        self.context, self._fns, self._specs = context, {}, {}

    def register(self, fn, description, properties, required):
        self._fns[fn.__name__] = fn
        self._specs[fn.__name__] = {"type": "function", "function": {
            "name": fn.__name__, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}
        return fn

    def add_mcp(self, server, only=None):
        """Register an MCP server's tools (all of them, or the names in `only`) like local ones."""
        for t in server.tools:
            if only and t["name"] not in only:
                continue
            self._fns[t["name"]] = lambda _name=t["name"], **args: server.call(_name, args)
            self._specs[t["name"]] = {"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}}
        return server

    def specs(self, names):
        return [self._specs[n] for n in names]

    def call(self, name, arguments, tracer=None, agent="", label=""):
        start, ok = time.time(), True
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            result = self._fns[name](**args, **({"_ctx": self.context} if name in CONTEXT_TOOLS else {}))
        except Exception as e:  # report the error to the model instead of crashing the run
            result, ok = {"error": f"{type(e).__name__}: {e}"}, False
        if tracer:
            tracer.tool(agent, name, label, time.time() - start, ok)
        return result


# ---------------------------------------------------------------- dates
def _parse_time(text):
    text = text.lower()
    if "noon" in text:
        return 12, 0
    if "midnight" in text or "end of day" in text or "eod" in text:
        return 23, 59
    # A range like "3:30 to 4:45 pm" or "2-4pm" means it starts at 3:30 pm / 2 pm.
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-|–|until)\s*\d{1,2}(?::\d{2})?\s*(am|pm)", text)
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        return hour + (12 if (m.group(3) or m.group(4)).startswith("p") else 0), minute
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", text)
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        return hour + (12 if m.group(3).startswith("p") else 0), minute
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def resolve_date(phrase: str, reference: str):
    """Turn a date phrase ("next Tuesday at 10am", "Dec 2, 11:59 pm PT") into an exact time.

    `reference` is when the message was received (ISO 8601). Conventions:
      - "Friday" = the next Friday on or after the reference day.
      - "next Friday" said Fri-Sun = the coming Friday; said Mon-Thu = Friday of next week.
      - No time given = 23:59 (end of day). A time range ("4 to 5:15 pm") gives its start.
      - Result is converted to America/New_York.
    """
    ref = datetime.fromisoformat(reference).astimezone(HOME_TZ)
    text = phrase.lower().strip()
    tz = HOME_TZ
    for abbr, zone in TZ_ABBR.items():
        if re.search(rf"\b{abbr}\b", text):
            tz = ZoneInfo(zone)
    ref_local = ref.astimezone(tz)
    day, why = None, ""
    if "today" in text or "tonight" in text:
        day, why = ref_local.date(), "today"
    elif "tomorrow" in text:
        day, why = ref_local.date() + timedelta(days=1), "tomorrow"
    elif m := re.search(r"in (\d+) days?", text):
        day, why = ref_local.date() + timedelta(days=int(m.group(1))), f"in {m.group(1)} days"
    elif m := re.search(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2})", text):
        month, dom = MONTHS.index(m.group(1)) + 1, int(m.group(2))
        year = ref_local.year + (1 if month < ref_local.month - 6 else 0)
        day, why = datetime(year, month, dom).date(), "explicit date"
    elif m := re.search(r"\b(\d{1,2})/(\d{1,2})\b", text):
        day, why = datetime(ref_local.year, int(m.group(1)), int(m.group(2))).date(), "explicit date"
    else:
        for i, name in enumerate(WEEKDAYS):
            if re.search(rf"\b{name[:3]}(?:{name[3:]})?\b", text):
                ahead = (i - ref_local.weekday()) % 7
                if re.search(rf"\bnext\s+{name[:3]}", text):
                    ahead = ahead or 7
                    if ref_local.weekday() <= 3 and i > ref_local.weekday():
                        ahead += 7
                    why = f"next {name.title()} (said on a {WEEKDAYS[ref_local.weekday()].title()})"
                else:
                    why = f"{name.title()} on/after the reference day"
                day = ref_local.date() + timedelta(days=ahead)
                break
    if day is None:
        return {"error": f"could not find a date in {phrase!r}"}
    hour, minute = _parse_time(text) or (23, 59)
    when = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).astimezone(HOME_TZ)
    return {"iso": when.isoformat(), "weekday": WEEKDAYS[when.weekday()].title(),
            "interpretation": why + ("" if _parse_time(text) else ", no time given so 23:59")}


# ---------------------------------------------------------------- calendar
def write_ics(events: list, filename: str = "deadlines.ics", _ctx=None):
    """Write events [{title, start, end?, description?}] (ISO 8601) to an .ics calendar file."""
    def stamp(iso):
        return datetime.fromisoformat(iso).astimezone(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//speed-of-thought//agentkit//EN"]
    for i, ev in enumerate(events):
        end = ev.get("end") or (datetime.fromisoformat(ev["start"]) + timedelta(minutes=30)).isoformat()
        desc = ev.get("description", "").replace("\n", "\\n").replace(",", "\\,")
        lines += ["BEGIN:VEVENT", f"UID:{i}-{stamp(ev['start'])}@speed-of-thought", f"DTSTAMP:{stamp(ev['start'])}",
                  f"DTSTART:{stamp(ev['start'])}", f"DTEND:{stamp(end)}",
                  f"SUMMARY:{ev['title'].replace(',', chr(92) + ',')}", f"DESCRIPTION:{desc}", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    path = Path(_ctx["output_dir"]) / Path(filename).name
    path.write_text("\r\n".join(lines) + "\r\n")
    return {"written": str(path.name), "events": len(events)}


# ---------------------------------------------------------------- money
def _d(x):
    return Decimal(str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def sum_items(items: list, subtotal: float, tax: float, total: float, tip: float = 0, taxable_base: float = 0):
    """Check a receipt's arithmetic: each line, the subtotal, subtotal + tax + tip = total,
    and (if the receipt prints it) that the taxed items add up to the taxable base."""
    problems = []
    for it in items:
        expect = _d(Decimal(str(it.get("qty", 1))) * Decimal(str(it["unit_price"])))
        if abs(expect - _d(it["line_total"])) > Decimal("0.01"):
            problems.append(f"{it['name']}: {it.get('qty', 1)} x {it['unit_price']} = {expect}, not {it['line_total']}")
    lines = sum((_d(it["line_total"]) for it in items), Decimal("0"))
    if lines != _d(subtotal):
        problems.append(f"line totals add to {lines}, but subtotal says {_d(subtotal)}")
    if _d(subtotal) + _d(tax) + _d(tip) != _d(total):
        problems.append(f"subtotal + tax + tip = {_d(subtotal) + _d(tax) + _d(tip)}, but total says {_d(total)}")
    taxed = sum((_d(it["line_total"]) for it in items if it.get("tax_code")), Decimal("0"))
    if taxable_base and taxed != _d(taxable_base):
        problems.append(f"items marked T/M add to {taxed}, but the tax line says it applies to {_d(taxable_base)}; "
                        "re-check which items have a T or M after the price")
    return {"ok": not problems, "problems": problems, "sum_of_lines": float(lines)}


def compute_balances(receipts: list):
    """Split receipts and settle up, following the house rules exactly.

    receipts: [{payer, subtotal, tax, tip, total, items:[{line_total, tax_code, assigned_to:[names]}]}]
    - Each item is split equally among assigned_to.
    - Tax is shared in proportion to each person's part of the TAXED items (tax_code "T" or "M").
    - Tip is shared in proportion to each person's part of the pre-tax subtotal.
    - Non-payers' shares are rounded half-up to the cent; the payer absorbs any leftover cent.
    """
    shares, paid = {}, {}
    for r in receipts:
        items_share, taxed_share = {}, {}
        for it in r["items"]:
            people = it["assigned_to"]
            part = Decimal(str(it["line_total"])) / len(people)
            for p in people:
                items_share[p] = items_share.get(p, Decimal("0")) + part
                if it.get("tax_code"):
                    taxed_share[p] = taxed_share.get(p, Decimal("0")) + part
        tax, tip = Decimal(str(r.get("tax", 0))), Decimal(str(r.get("tip", 0)))
        items_total = sum(items_share.values(), Decimal("0"))
        taxed_total = sum(taxed_share.values(), Decimal("0"))
        if tax and not taxed_total:          # no tax codes read: fall back to all items
            taxed_share, taxed_total = items_share, items_total
        exact = {p: v + (tax * taxed_share.get(p, 0) / taxed_total if taxed_total else 0)
                 + (tip * v / items_total if items_total else 0) for p, v in items_share.items()}
        total = _d(r["total"]) if "total" in r else _d(items_total + tax + tip)
        rounded = {p: _d(v) for p, v in exact.items() if p != r["payer"]}
        rounded[r["payer"]] = total - sum(rounded.values(), Decimal("0"))
        for p, v in rounded.items():
            shares[p] = shares.get(p, Decimal("0")) + v
        paid[r["payer"]] = paid.get(r["payer"], Decimal("0")) + total
    people = sorted(set(shares) | set(paid))
    net = {p: paid.get(p, Decimal("0")) - shares.get(p, Decimal("0")) for p in people}
    debtors = [[p, -v] for p, v in sorted(net.items(), key=lambda x: x[1]) if v < 0]
    creditors = [[p, v] for p, v in sorted(net.items(), key=lambda x: -x[1]) if v > 0]
    settlement, i, j = [], 0, 0
    while i < len(debtors) and j < len(creditors):
        amount = min(debtors[i][1], creditors[j][1])
        settlement.append({"from": debtors[i][0], "to": creditors[j][0], "amount": float(amount)})
        debtors[i][1] -= amount
        creditors[j][1] -= amount
        i += debtors[i][1] == 0
        j += creditors[j][1] == 0
    return {"share": {p: float(v) for p, v in shares.items()}, "paid": {p: float(v) for p, v in paid.items()},
            "net": {p: float(v) for p, v in net.items()}, "settlement": settlement}


CONTEXT_TOOLS = {"write_ics"}


def default_toolbox(output_dir):
    tb = Toolbox(output_dir=str(output_dir))
    tb.register(resolve_date, "Convert a date phrase from a message into an exact ISO 8601 time in America/New_York. "
                "Always use this instead of working out dates yourself.",
                {"phrase": {"type": "string", "description": "the date words exactly as written, e.g. 'the Tuesday after next at 10:50am'"},
                 "reference": {"type": "string", "description": "the message's received time, ISO 8601"}},
                ["phrase", "reference"])
    tb.register(write_ics, "Write calendar events to an .ics file the student can import.",
                {"events": {"type": "array", "items": {"type": "object", "properties": {
                    "title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"},
                    "description": {"type": "string"}}, "required": ["title", "start"]}},
                 "filename": {"type": "string"}}, ["events"])
    tb.register(sum_items, "Check a receipt's arithmetic. Call it after reading a receipt.",
                {"items": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "qty": {"type": "number"}, "unit_price": {"type": "number"},
                    "line_total": {"type": "number"}, "tax_code": {"type": "string"}},
                    "required": ["name", "unit_price", "line_total"]}},
                 "subtotal": {"type": "number"}, "tax": {"type": "number"}, "tip": {"type": "number"},
                 "total": {"type": "number"},
                 "taxable_base": {"type": "number", "description": "the amount the tax line says it is ON, if printed"}},
                ["items", "subtotal", "tax", "total"])
    return tb
