#!/usr/bin/env python3
"""Render r1.png..r5.png and answer_key.json from ONE data dict (RECEIPTS below).

Re-runnable and deterministic:  ../../.venv/bin/python make_receipts.py
Everything printed on a receipt and everything in answer_key.json is derived from
RECEIPTS + HOUSE, so the numbers cannot drift apart. All arithmetic uses Decimal /
Fraction, and the script asserts the invariants before writing anything.
"""
from __future__ import annotations

import json
import random
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
MENLO = "/System/Library/Fonts/Menlo.ttc"          # index 0 = Regular, 1 = Bold
COURIER = "/System/Library/Fonts/Supplemental/Courier New.ttf"

# --------------------------------------------------------------------------------------
# Single source of truth
# --------------------------------------------------------------------------------------
PEOPLE = ["Alex", "Sam", "Jordan"]
ALL3 = ["Alex", "Sam", "Jordan"]
SJ = ["Sam", "Jordan"]

CARDS = {"Alex": "VISA ****4821", "Sam": "MC ****3307", "Jordan": "DEBIT ****6650"}

# MA-style rates. Package-store alcohol carries no sales tax in MA (excise is built into
# the shelf price), so it is 0% like groceries.
TAX_RATES = {
    "grocery": Decimal("0"),
    "general": Decimal("0.0625"),
    "alcohol": Decimal("0"),
    "prepared_meal": Decimal("0.07"),
}
FLAG = {"general": "T", "prepared_meal": "M"}  # printed after the line total

# item tuple: (printed name, qty, unit price, category, assigned_to)
RECEIPTS = [
    dict(
        id="r1", kind="grocery run (shared food)",
        store=["KETTLE & CRATE MARKET"], address="212 Alder Row, Brookhollow, MA", phone="(617) 555-0142",
        date="2026-10-24", time="11:42", payer="Alex", register="REG 03", trans="4471", cashier="MARISOL",
        auth="08213A", footer=["THANK YOU FOR SHOPPING LOCAL!", "RETURNS W/ RECEIPT WITHIN 30 DAYS"],
        items=[
            ("EGGS LARGE DOZEN", 1, "4.29", "grocery", ALL3),
            ("WHOLE MILK GAL", 2, "3.89", "grocery", ALL3),
            ("SOURDOUGH LOAF", 1, "5.49", "grocery", ALL3),
            ("CHICKEN THIGHS", 2, "7.45", "grocery", ALL3),
            ("PENNE PASTA 1LB", 3, "1.79", "grocery", ALL3),
            ("MARINARA SAUCE", 2, "3.99", "grocery", ALL3),
            ("BABY SPINACH", 1, "3.49", "grocery", ALL3),
            ("JASMINE RICE 5LB", 1, "8.99", "grocery", ALL3),
            ("ALUMINUM FOIL", 1, "4.79", "general", ALL3),
        ],
        tip_pct=None,
        style=dict(angle=1.2, ink=(28, 27, 30), paper=(247, 244, 236), bg=(182, 176, 166), noise=6.0,
                   blur=0.55, seed=11, hard=False),
    ),
    dict(
        id="r2", kind="pharmacy (personal + shared items)",
        store=["LINDENWICK PHARMACY"], address="48 Wren Street, Brookhollow, MA", phone="(617) 555-0187",
        date="2026-10-26", time="18:15", payer="Sam", register="REG 01", trans="20918", cashier="DEV",
        auth="55K102", footer=["FEEL BETTER SOON!", "PHARMACY HOURS 9A-9P DAILY"],
        items=[
            ("PAPER TOWELS 6RL", 1, "9.99", "general", ALL3),
            ("COLD/FLU CAPLETS", 2, "7.49", "general", ALL3),
            ("FACIAL TISSUE 3PK", 1, "5.49", "general", ALL3),
            ("SHAMPOO 12OZ", 1, "6.79", "general", ["Alex"]),
            ("CONTACT LENS SOLN", 1, "10.99", "general", ["Jordan"]),
            ("RAZOR CARTRIDGE 4", 1, "14.99", "general", ["Sam"]),
            ("GRANOLA BARS 6CT", 2, "3.29", "grocery", ["Sam"]),
        ],
        tip_pct=None,
        style=dict(angle=-0.8, ink=(30, 30, 34), paper=(245, 245, 240), bg=(170, 172, 170), noise=6.0,
                   blur=0.55, seed=22, hard=False),
    ),
    dict(
        id="r3", kind="wine & spirits shop",
        store=["TALLOWAY", "WINE & SPIRITS"], address="7 Cooper Lane, Brookhollow, MA", phone="(617) 555-0119",
        date="2026-10-30", time="17:38", payer="Jordan", register="REG 02", trans="3365", cashier="BEA",
        auth="91F4C7", footer=["MUST BE 21+. PLEASE ENJOY RESPONSIBLY.", "NO TAX ON PACKAGE ALCOHOL (MA)"],
        items=[
            ("CABERNET SAUV 750", 2, "13.99", "alcohol", SJ),
            ("IPA 6PK 12OZ", 1, "11.49", "alcohol", SJ),
            ("TEQUILA BLANCO 750", 1, "24.99", "alcohol", SJ),
            ("LIME SELTZER 12PK", 1, "5.99", "grocery", ALL3),
            ("TORTILLA CHIPS", 2, "3.49", "grocery", ALL3),
            ("PARTY CUPS 50CT", 1, "4.29", "general", ALL3),
        ],
        tip_pct=None,
        style=dict(angle=0.6, ink=(26, 24, 28), paper=(246, 242, 232), bg=(150, 140, 128), noise=6.5,
                   blur=0.55, seed=33, hard=False),
    ),
    dict(
        id="r4", kind="pizza night takeout (prepared food, 7% meals tax, tip)",
        store=["LANTERNFLY PIZZA CO."], address="330 Mill Street, Brookhollow, MA", phone="(617) 555-0173",
        date="2026-10-29", time="20:05", payer="Sam", register="TAKEOUT", trans="1182", cashier="NICO",
        auth="3HH820", footer=["THANKS FOR ORDERING!", "ASK ABOUT OUR TUESDAY SLICE DEAL"],
        items=[
            ("LG MARGHERITA", 1, "18.50", "prepared_meal", SJ),
            ("LG PEPPERONI", 1, "19.75", "prepared_meal", SJ),
            ("GARLIC KNOTS 12", 1, "7.25", "prepared_meal", SJ),
            ("CAESAR SALAD", 1, "9.50", "prepared_meal", SJ),
            ("SODA 2 LITER", 1, "3.29", "prepared_meal", SJ),
        ],
        tip_pct=18,
        # deliberately harder: lower contrast, faded right edge, crease shadow
        style=dict(angle=-1.4, ink=(92, 88, 84), paper=(232, 227, 214), bg=(160, 156, 150), noise=8.0,
                   blur=0.7, seed=44, hard=True),
    ),
    dict(
        id="r5", kind="hardware / household supplies",
        store=["IRONBARK", "HARDWARE & HOME"], address="1450 Bridge Road, Brookhollow, MA", phone="(617) 555-0156",
        date="2026-11-01", time="13:10", payer="Alex", register="REG 04", trans="77120", cashier="TOMAS",
        auth="60C3D9", footer=["HELPFUL ADVICE SINCE 1987", "KEEP RECEIPT FOR RETURNS"],
        items=[
            ("LED BULBS 4PK", 1, "8.97", "general", ALL3),
            ("DISH SOAP 24OZ", 2, "3.29", "general", ALL3),
            ("TRASH BAGS 40CT", 1, "12.49", "general", ALL3),
            ("SPONGES 6PK", 1, "4.99", "general", ALL3),
            ("AA BATTERIES 8PK", 1, "9.49", "general", ALL3),
            ("DOOR DRAFT STOPPER", 1, "14.99", "general", ALL3),
        ],
        tip_pct=None,
        style=dict(angle=1.0, ink=(30, 28, 26), paper=(248, 246, 240), bg=(140, 146, 152), noise=6.0,
                   blur=0.55, seed=55, hard=False),
    ),
]

ROUNDING_RULE = (
    "For each receipt, compute every person's exact share (items split equally among the people assigned; "
    "each tax line split in proportion to each person's share of the items taxed on that line; tip split in "
    "proportion to each person's share of the pre-tax subtotal). Round each NON-payer's share to the nearest "
    "cent, half-up. The payer's share is the receipt total minus the other people's rounded shares, so the "
    "payer absorbs any leftover cent. Every payer in this house is part of the split of their own receipt."
)

# --------------------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------------------
CENT = Decimal("0.01")


def r2(x) -> Decimal:
    if isinstance(x, Fraction):
        x = Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def F(d: Decimal) -> Fraction:
    return Fraction(str(d))


def compute(rc):
    items = []
    for name, qty, unit, cat, who in rc["items"]:
        unit = Decimal(unit)
        items.append(dict(name=name, qty=qty, unit_price=unit, line_total=(unit * qty).quantize(CENT),
                          tax_rate=TAX_RATES[cat], category=cat, assigned_to=list(who)))
    subtotal = sum((it["line_total"] for it in items), Decimal("0"))
    tax_lines = []
    for rate in sorted({it["tax_rate"] for it in items if it["tax_rate"] > 0}):
        base = sum((it["line_total"] for it in items if it["tax_rate"] == rate), Decimal("0"))
        tax_lines.append(dict(rate=rate, taxable_base=base, amount=r2(base * rate)))
    tax = sum((t["amount"] for t in tax_lines), Decimal("0"))
    tip = r2(subtotal * Decimal(rc["tip_pct"]) / 100) if rc["tip_pct"] else Decimal("0.00")
    total = subtotal + tax + tip

    # exact shares (method A: proportional allocation of the PRINTED tax line and tip)
    item_share = {p: Fraction(0) for p in PEOPLE}
    taxable_share = {(p, t["rate"]): Fraction(0) for p in PEOPLE for t in tax_lines}
    for it in items:
        part = F(it["line_total"]) / len(it["assigned_to"])
        for p in it["assigned_to"]:
            item_share[p] += part
            if it["tax_rate"] > 0:
                taxable_share[(p, it["tax_rate"])] += part
    exact = {}
    for p in PEOPLE:
        s = item_share[p]
        for t in tax_lines:
            s += F(t["amount"]) * taxable_share[(p, t["rate"])] / F(t["taxable_base"])
        if tip:
            s += F(tip) * item_share[p] / F(subtotal)
        exact[p] = s

    # method B (a sloppier but reasonable approach): per-item tax at the nominal rate, unrounded
    exact_b = {p: Fraction(0) for p in PEOPLE}
    for it in items:
        part = F(it["line_total"]) * (1 + F(it["tax_rate"])) / len(it["assigned_to"])
        for p in it["assigned_to"]:
            exact_b[p] += part
    if tip:
        for p in PEOPLE:
            exact_b[p] += F(tip) * item_share[p] / F(subtotal)

    payer = rc["payer"]
    share = {}
    for p in PEOPLE:
        if p != payer:
            share[p] = r2(exact[p])
    share[payer] = total - sum(share.values(), Decimal("0"))

    return dict(items=items, subtotal=subtotal, tax_lines=tax_lines, tax=tax, tip=tip, total=total,
                exact=exact, exact_b=exact_b, share={p: share[p] for p in PEOPLE})


def settle(net):
    """Greedy largest-debtor -> largest-creditor; optimal (n-1 transfers max) for 3 people."""
    cred = sorted([[p, v] for p, v in net.items() if v > 0], key=lambda x: -x[1])
    debt = sorted([[p, -v] for p, v in net.items() if v < 0], key=lambda x: -x[1])
    out = []
    while cred and debt:
        c, d = cred[0], debt[0]
        amt = min(c[1], d[1])
        out.append(dict(**{"from": d[0], "to": c[0]}, amount=amt))
        c[1] -= amt
        d[1] -= amt
        if c[1] == 0:
            cred.pop(0)
        if d[1] == 0:
            debt.pop(0)
        cred.sort(key=lambda x: -x[1])
        debt.sort(key=lambda x: -x[1])
    return out


def verify(results):
    for rc, res in results:
        assert 3 <= len(rc["items"]) <= 9, rc["id"]
        assert "2026-10-24" <= rc["date"] <= "2026-11-01", rc["id"]
        assert res["subtotal"] == sum(it["line_total"] for it in res["items"])
        for it in res["items"]:
            assert it["line_total"] == (it["unit_price"] * it["qty"]).quantize(CENT)
        assert res["total"] == res["subtotal"] + res["tax"] + res["tip"]
        assert sum(res["share"].values()) == res["total"], rc["id"]
        assert sum(res["exact"].values()) == F(res["total"]), rc["id"]
        for p in PEOPLE:
            if p == rc["payer"]:
                continue
            # both allocation methods must land on the same cent, well away from a half-cent boundary
            assert r2(res["exact"][p]) == r2(res["exact_b"][p]), (rc["id"], p)
            frac_cents = (res["exact"][p] * 100) % 1
            assert abs(frac_cents - Fraction(1, 2)) > Fraction(1, 20), (rc["id"], p, float(frac_cents))
        # the payer only ever absorbs at most one cent
        assert abs(F(res["share"][rc["payer"]]) - res["exact"][rc["payer"]]) < Fraction(2, 100), rc["id"]


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
COLS = 41


def money(d: Decimal) -> str:
    return f"{d:.2f}"


def item_line(it) -> str:
    flag = FLAG.get(it["category"], " ")
    return f"{it['name']:<19}{it['qty']:>2} x{money(it['unit_price']):>7}{money(it['line_total']):>9} {flag}"


def lr(left: str, right: str, cols: int = COLS) -> str:
    gap = cols - len(left) - len(right)
    assert gap >= 1, (left, right)
    return left + " " * gap + right


def receipt_lines(rc, res):
    """List of (kind, payload). kinds: title, center, text, rule, gap, total, barcode."""
    y, m, d = rc["date"].split("-")
    hh, mm = map(int, rc["time"].split(":"))
    ampm = "AM" if hh < 12 else "PM"
    clock = f"{(hh - 1) % 12 + 1}:{mm:02d} {ampm}"
    L = []
    for s in rc["store"]:
        L.append(("title", s))
    L += [("center", rc["address"]), ("center", rc["phone"]), ("gap", 10),
          ("text", lr(f"{m}/{d}/{y}", clock)),
          ("text", lr(f"{rc['register']}  #{rc['trans']}", f"CASHIER: {rc['cashier']}")),
          ("rule", "-")]
    for it in res["items"]:
        L.append(("text", item_line(it)))
    L.append(("rule", "-"))
    flags = sorted({it["category"] for it in res["items"] if it["category"] in FLAG})
    L.append(("text", lr("SUBTOTAL", money(res["subtotal"]))))
    for t in res["tax_lines"]:
        label = "MEALS TAX" if t["rate"] == TAX_RATES["prepared_meal"] else "MA SALES TAX"
        pct = f"{(t['rate'] * 100).normalize():f}%"
        L.append(("text", lr(f"{label} {pct} ON {money(t['taxable_base'])}", money(t["amount"]))))
    if res["tip"]:
        L.append(("text", lr(f"TIP {rc['tip_pct']}%", money(res["tip"]))))
    L.append(("rule", "="))
    L.append(("total", lr("TOTAL", "$" + money(res["total"]), cols=27)))
    L.append(("gap", 12))
    L.append(("text", lr(CARDS[rc["payer"]], money(res["total"]))))
    L.append(("text", lr(f"AUTH {rc['auth']}", "APPROVED")))
    for cat in flags:
        legend = {"general": "T = TAXABLE ITEM (6.25%)", "prepared_meal": "M = MEALS TAX ITEM (7%)"}[cat]
        L.append(("center", legend))
    L.append(("gap", 14))
    for f in rc["footer"]:
        L.append(("center", f))
    L.append(("gap", 10))
    L.append(("barcode", rc["id"]))
    L.append(("center", f"{rc['id'].upper()}-{y}{m}{d}-{rc['trans']}"))
    return L


def zigzag_mask(w, h, rng, tooth=9, depth=7):
    mask = Image.new("L", (w, h), 0)
    top = [(0, depth)]
    x = 0
    while x < w:
        top.append((x, rng.randint(0, depth)))
        x += tooth
    top.append((w, depth))
    bottom = []
    x = w
    while x > 0:
        bottom.append((x, h - 1 - rng.randint(0, depth)))
        x -= tooth
    bottom.append((0, h - 1 - depth))
    ImageDraw.Draw(mask).polygon(top + bottom, fill=255)
    return mask


def render(rc, res, out_path: Path):
    st = rc["style"]
    rng = random.Random(st["seed"])
    nrng = np.random.default_rng(st["seed"])
    size = 24
    font = ImageFont.truetype(MENLO, size, index=0)
    bold = ImageFont.truetype(MENLO, size + 10, index=1)
    totf = ImageFont.truetype(MENLO, size + 8, index=1)
    cw = font.getlength("M")
    margin = 34
    pw = int(round(COLS * cw + 2 * margin))
    line_h = int(size * 1.42)

    lines = receipt_lines(rc, res)
    # layout pass
    h = 46
    for kind, payload in lines:
        h += {"title": int((size + 10) * 1.35), "gap": payload if kind == "gap" else 0, "rule": line_h,
              "total": int((size + 8) * 1.5), "barcode": 70}.get(kind, line_h)
    ph = h + 40

    paper = Image.new("RGB", (pw, ph), st["paper"])
    dr = ImageDraw.Draw(paper)
    ink = st["ink"]
    y = 46
    for kind, payload in lines:
        if kind == "title":
            w = dr.textlength(payload, font=bold)
            dr.text(((pw - w) / 2, y), payload, font=bold, fill=ink)
            y += int((size + 10) * 1.35)
        elif kind == "center":
            w = dr.textlength(payload, font=font)
            dr.text(((pw - w) / 2, y), payload, font=font, fill=ink)
            y += line_h
        elif kind == "text":
            dr.text((margin, y), payload, font=font, fill=ink)
            y += line_h
        elif kind == "rule":
            dr.text((margin, y), payload * COLS, font=font, fill=ink)
            y += line_h
        elif kind == "total":
            w = dr.textlength(payload, font=totf)
            dr.text(((pw - w) / 2, y), payload, font=totf, fill=ink)
            y += int((size + 8) * 1.5)
        elif kind == "gap":
            y += payload
        elif kind == "barcode":
            bw = int(COLS * cw * 0.8)
            x0 = (pw - bw) // 2
            x = x0
            brng = random.Random(payload)
            while x < x0 + bw:
                t = brng.choice([2, 2, 3, 4, 6])
                dr.rectangle([x, y, x + t - 1, y + 54], fill=ink)
                x += t + brng.choice([2, 3, 4])
            y += 70

    arr = np.asarray(paper).astype(np.float32)
    H, W = arr.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # thermal paper: faint horizontal streaks + gentle overall shading
    streak = nrng.normal(0, 1.2, size=(H, 1, 1)).astype(np.float32)
    arr += streak
    arr *= (1.0 - 0.035 * ((xx / W - 0.5) ** 2) * 4)[..., None]
    if st["hard"]:
        # faded right side (uneven thermal print) and a crease shadow across the paper
        ink_arr = np.array(ink, np.float32)
        paper_arr = np.array(st["paper"], np.float32)
        fade = np.clip((xx / W - 0.55) / 0.45, 0, 1) * 0.22
        darkness = np.clip(((paper_arr - arr) / np.maximum(paper_arr - ink_arr, 1)).mean(axis=2), 0, 1)
        arr += (darkness * fade)[..., None] * (paper_arr - ink_arr)[None, None, :]
        cy = 0.47 * H
        slope = 0.035
        dist = (yy - (cy + slope * (xx - W / 2))) / np.sqrt(1 + slope ** 2)
        shadow = 46 * np.exp(-((dist - 7) ** 2) / (2 * 9.0 ** 2))       # soft shadow below the fold
        line = 36 * np.exp(-(dist ** 2) / (2 * 1.3 ** 2))                 # sharp fold line
        highlight = 10 * np.exp(-((dist + 6) ** 2) / (2 * 4.0 ** 2))      # light catch above it
        arr += (highlight - shadow - line)[..., None]
        # second, fainter crease near the bottom third
        dist2 = (yy - (0.78 * H - 0.02 * (xx - W / 2)))
        arr -= (20 * np.exp(-(dist2 ** 2) / (2 * 5.0 ** 2)))[..., None]
    arr += nrng.normal(0, st["noise"], size=arr.shape).astype(np.float32)
    paper = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    rgba = paper.convert("RGBA")
    rgba.putalpha(zigzag_mask(pw, ph, rng))
    rot = rgba.rotate(st["angle"], resample=Image.BICUBIC, expand=True)

    canvas_w = 800
    canvas_h = rot.height + 90
    bg = np.full((canvas_h, canvas_w, 3), st["bg"], np.float32)
    gy, gx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    vign = 1.0 - 0.18 * (((gx / canvas_w - 0.5) ** 2 + (gy / canvas_h - 0.5) ** 2) * 2)
    bg *= vign[..., None]
    bg += nrng.normal(0, 4.0, size=bg.shape).astype(np.float32)
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")

    ox = (canvas_w - rot.width) // 2
    oy = 45
    shadow = Image.new("RGBA", rot.size, (0, 0, 0, 0))
    shadow.putalpha(rot.split()[3].point(lambda a: int(a * 0.45)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    canvas.alpha_composite(shadow, (ox + 7, oy + 10))
    canvas.alpha_composite(rot, (ox, oy))

    final = canvas.convert("RGB").filter(ImageFilter.GaussianBlur(st["blur"]))
    fa = np.asarray(final).astype(np.float32)
    fa += nrng.normal(0, 2.0, size=fa.shape).astype(np.float32)
    Image.fromarray(np.clip(fa, 0, 255).astype(np.uint8)).save(out_path, optimize=True)
    return final.size


# --------------------------------------------------------------------------------------
# Answer key
# --------------------------------------------------------------------------------------
def num(d: Decimal) -> float:
    return float(d)


def build_key(results):
    receipts = []
    paid = {p: Decimal("0") for p in PEOPLE}
    share_tot = {p: Decimal("0") for p in PEOPLE}
    by_receipt = {}
    for rc, res in results:
        paid[rc["payer"]] += res["total"]
        for p in PEOPLE:
            share_tot[p] += res["share"][p]
        by_receipt[rc["id"]] = {p: num(res["share"][p]) for p in PEOPLE}
        receipts.append(dict(
            id=rc["id"], file=f"{rc['id']}.png", kind=rc["kind"], store=" ".join(rc["store"]).title(),
            store_as_printed=" ".join(rc["store"]), address=rc["address"], date=rc["date"], time=rc["time"],
            payer=rc["payer"], payment=CARDS[rc["payer"]],
            items=[dict(name=it["name"], qty=it["qty"], unit_price=num(it["unit_price"]),
                        line_total=num(it["line_total"]), tax_rate=num(it["tax_rate"]), category=it["category"],
                        assigned_to=it["assigned_to"]) for it in res["items"]],
            subtotal=num(res["subtotal"]),
            tax_lines=[dict(rate=num(t["rate"]), taxable_base=num(t["taxable_base"]), amount=num(t["amount"]))
                       for t in res["tax_lines"]],
            tax=num(res["tax"]), tip=num(res["tip"]), total=num(res["total"]),
            split={p: num(res["share"][p]) for p in PEOPLE},
            split_exact_unrounded={p: round(float(res["exact"][p]), 6) for p in PEOPLE},
        ))
    net = {p: paid[p] - share_tot[p] for p in PEOPLE}
    assert sum(net.values()) == 0
    assert sum(share_tot.values()) == sum(paid.values())
    transfers = settle(net)
    # applying the settlement must zero every balance
    after = dict(net)
    for t in transfers:
        after[t["from"]] += t["amount"]
        after[t["to"]] -= t["amount"]
    assert all(v == 0 for v in after.values()), after
    assert len(transfers) <= len(PEOPLE) - 1

    return dict(
        currency="USD",
        people=PEOPLE,
        cards=CARDS,
        tax_rates={k: num(v) for k, v in TAX_RATES.items()},
        tax_note="MA-style: groceries/food 0%, general merchandise 6.25%, prepared meals 7% (meals tax). "
                 "Package-store alcohol is 0% in MA. Each receipt prints its tax line(s) explicitly, and tax is "
                 "rounded half-up to the cent per tax line.",
        rounding_rule=ROUNDING_RULE,
        receipts=receipts,
        per_person_share=dict(by_receipt=by_receipt, total={p: num(share_tot[p]) for p in PEOPLE}),
        paid={p: num(paid[p]) for p in PEOPLE},
        net_balances={p: num(net[p]) for p in PEOPLE},
        net_balances_note="net = total paid - total share; positive means the person is owed money.",
        settlement=[dict(**{"from": t["from"], "to": t["to"]}, amount=num(t["amount"])) for t in transfers],
        grand_total=num(sum(paid.values())),
    )


def main():
    for p in (MENLO,):
        assert Path(p).exists(), f"missing font {p}"
    results = [(rc, compute(rc)) for rc in RECEIPTS]
    verify(results)
    key = build_key(results)
    for rc, res in results:
        wh = render(rc, res, HERE / f"{rc['id']}.png")
        print(f"{rc['id']}.png {wh[0]}x{wh[1]}  subtotal {res['subtotal']}  tax {res['tax']}  tip {res['tip']}  "
              f"total {res['total']}  payer {rc['payer']}  split {dict((p, str(v)) for p, v in res['share'].items())}")
    with open(HERE / "answer_key.json", "w") as fh:
        json.dump(key, fh, indent=2)
        fh.write("\n")
    print("paid", key["paid"])
    print("shares", key["per_person_share"]["total"])
    print("net", key["net_balances"])
    print("settlement", key["settlement"])


if __name__ == "__main__":
    main()
