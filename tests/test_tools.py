"""Tests for the deterministic tools in agentkit/tools.py, checked against the sample_data answer keys.

No model or LM Studio server is needed. From the repo root:

    .venv/bin/python -m unittest            # discovers this file
    .venv/bin/python tests/test_tools.py    # or run it directly
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works in the exported repo (tests/ next to agentkit/) and in the source tree (tools/repo_extras/tests).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "agentkit").is_dir())
sys.path.insert(0, str(ROOT))
from agentkit.tools import compute_balances, resolve_date, sum_items  # noqa: E402

DATA = ROOT / "sample_data"
INBOX_KEY = json.loads((DATA / "inbox" / "answer_key.json").read_text())
RECEIPTS_KEY = json.loads((DATA / "receipts" / "answer_key.json").read_text())
MESSAGES = {m["id"]: m for m in (json.loads(line) for line in
                                 (DATA / "inbox" / "messages.jsonl").read_text().splitlines() if line.strip())}

# The answer key stores each item's tax rate; receipts print a code after the price instead
# (house_rules.md): T = general merchandise, M = prepared meals, nothing = untaxed.
TAX_CODE = {RECEIPTS_KEY["tax_rates"]["general"]: "T", RECEIPTS_KEY["tax_rates"]["prepared_meal"]: "M"}


def key_time(section, field, message_id):
    """The answer key's time for the deadline or event that came from `message_id`."""
    return next(x[field] for x in INBOX_KEY[section] if message_id in x["source_message_ids"])


def resolve(message_id, phrase):
    """Resolve `phrase` against the message's received time, after checking the phrase is really in it."""
    message = MESSAGES[message_id]
    assert phrase in message["body"], f"{phrase!r} is not in {message_id}"
    return resolve_date(phrase, message["received"])


def receipt_items(receipt):
    return [{"name": it["name"], "qty": it["qty"], "unit_price": it["unit_price"], "line_total": it["line_total"],
             "tax_code": TAX_CODE.get(it["tax_rate"], ""), "assigned_to": it["assigned_to"]}
            for it in receipt["items"]]


class ResolveDateTest(unittest.TestCase):
    def test_next_friday_said_on_a_friday(self):
        # m05 arrives Fri Oct 30: "next Friday" is the coming Friday, Nov 6, not Nov 13.
        got = resolve("m05", "next Friday at 5:00 pm")
        self.assertEqual(got["iso"], "2026-11-06T17:00:00-05:00")
        self.assertEqual(got["iso"], key_time("deadlines", "due", "m05"))

    def test_pacific_time_rolls_over_to_the_next_day(self):
        # 11:59 pm PST on Wed Nov 4 is 2:59 am EST on Thu Nov 5.
        got = resolve("m03", "Wednesday, November 4 at 11:59 pm PT")
        self.assertEqual(got["iso"], "2026-11-05T02:59:00-05:00")
        self.assertEqual(got["iso"], key_time("deadlines", "due", "m03"))

    def test_time_range_gives_its_start(self):
        cases = [("m06", "Thursday, November 5, 4:00 to 5:15 pm", "2026-11-05T16:00:00-05:00"),
                 ("m15", "Wednesday, November 4 from 2:00 to 4:00 pm", "2026-11-04T14:00:00-05:00")]
        for message_id, phrase, want in cases:
            with self.subTest(message_id):
                got = resolve(message_id, phrase)
                self.assertEqual(got["iso"], want)
                self.assertEqual(got["iso"], key_time("events", "start", message_id))


class SumItemsTest(unittest.TestCase):
    def test_every_receipt_in_the_key_adds_up(self):
        for receipt in RECEIPTS_KEY["receipts"]:
            with self.subTest(receipt["id"]):
                taxable_base = round(sum(line["taxable_base"] for line in receipt["tax_lines"]), 2)
                got = sum_items(receipt_items(receipt), receipt["subtotal"], receipt["tax"], receipt["total"],
                                receipt["tip"], taxable_base)
                self.assertEqual(got["problems"], [])
                self.assertTrue(got["ok"])
                self.assertAlmostEqual(got["sum_of_lines"], receipt["subtotal"], places=2)

    def test_a_misread_line_is_reported(self):
        receipt = RECEIPTS_KEY["receipts"][0]
        items = receipt_items(receipt)
        items[1]["line_total"] = round(items[1]["unit_price"], 2)   # qty 2 read as a single unit
        got = sum_items(items, receipt["subtotal"], receipt["tax"], receipt["total"], receipt["tip"])
        self.assertFalse(got["ok"])
        self.assertTrue(any(items[1]["name"] in p for p in got["problems"]))


class ComputeBalancesTest(unittest.TestCase):
    def test_reproduces_the_key_settlement_exactly(self):
        receipts = [{"id": r["id"], "payer": r["payer"], "subtotal": r["subtotal"], "tax": r["tax"], "tip": r["tip"],
                     "total": r["total"], "items": receipt_items(r)} for r in RECEIPTS_KEY["receipts"]]
        got = compute_balances(receipts)
        self.assertEqual(got["settlement"], RECEIPTS_KEY["settlement"])
        self.assertEqual(got["settlement"], [{"from": "Jordan", "to": "Alex", "amount": 56.48},
                                             {"from": "Sam", "to": "Alex", "amount": 2.68}])
        self.assertEqual(got["net"], RECEIPTS_KEY["net_balances"])
        self.assertEqual(got["share"], RECEIPTS_KEY["per_person_share"]["total"])
        self.assertEqual(got["paid"], RECEIPTS_KEY["paid"])


if __name__ == "__main__":
    unittest.main()
