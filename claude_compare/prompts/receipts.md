Three roommates need to split shared receipts. Inputs in this folder: five receipt photos `r1.png` to `r5.png` and `house_rules.md` (who owns which card, who shares what, and exactly how to do the math, tax, tip and rounding).

Read every receipt exactly (every line item, quantities, the T/M tax markers, subtotal, tax, tip, total, and the card used), apply the house rules exactly, compute each person's share, net balances, and the fewest transfers to settle up. Then draft (do not send) a short, friendly payment request for each transfer.

Write these files into this folder (exact formats; they are machine-graded):
- `receipts.json`: `{"receipts": [{"id": "r1", "store": "...", "date": "YYYY-MM-DD", "payer": "...", "subtotal": 0.0, "tax": 0.0, "tip": 0.0, "total": 0.0, "items": [{"name": "...", "qty": 1, "unit_price": 0.0, "line_total": 0.0, "tax_code": "T|M|", "assigned_to": ["Alex", "Sam"]}]}], "balances": {"share": {"Alex": 0.0}, "paid": {"Alex": 0.0}, "net": {"Alex": 0.0}, "settlement": [{"from": "...", "to": "...", "amount": 0.0}]}}`
- `expenses.csv` (one row per item: receipt, store, date, payer, item, qty, unit_price, line_total, shared_by)
- `balances.md` (table of paid / share / net, then the settle-up transfers)
- `messages.txt` (the draft payment requests, clearly marked as drafts)
Do not ask questions; finish the task.
