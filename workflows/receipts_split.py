"""Receipt photos -> who owes whom (drafts only; nothing is sent or paid).

  master  (Qwen3.8 27B, no thinking)  turns house_rules.md into structured rules; reviews; drafts requests
  group   (Gemma 4 12B)            ledger lead assigns every item to people
  workers (Gemma 4 12B, vision)    read each receipt photo, checking themselves with sum_items;
                                   a failed read escalates to the next bigger model (the 27B)
  code                             sum_items, compute_balances

Run:  .venv/bin/python workflows/receipts_split.py
"""
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentkit import TIER_MODELS, Agent, AgentCard, Group, Tracer  # noqa: E402
from agentkit.tools import compute_balances, default_toolbox, sum_items  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sample_data" / "receipts", ROOT / "outputs" / "receipts"
RUNS = ROOT / "results" / "runs"
# Escalate to the first tier above the worker that runs a different (bigger) model.
ESCALATE_MODEL = next((m for m in (TIER_MODELS["group"], TIER_MODELS["master"]) if m != TIER_MODELS["worker"]),
                      TIER_MODELS["master"])

STR = {"type": "string"}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "roommates": {"type": "array", "items": STR},
    "cards": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                         "properties": {"last4": STR, "person": STR}, "required": ["last4", "person"]}},
    "rules": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                         "properties": {"applies_to": STR, "shared_by": {"type": "array", "items": STR},
                                                        "note": STR}, "required": ["applies_to", "shared_by", "note"]}}},
    "required": ["roommates", "cards", "rules"]}
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "anomalies": {"type": "array", "items": STR},
    "drafts": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "from": STR, "to": STR, "amount": {"type": "number"}, "message": STR},
        "required": ["from", "to", "amount", "message"]}}},
    "required": ["anomalies", "drafts"]}


def receipt_check(known_cards):
    """Code-level critic: the receipt's numbers must add up and its card must be one we know."""
    def check(job, r):
        try:
            res = sum_items(r["items"], r["subtotal"], r["tax"], r["total"], r.get("tip", 0), r.get("taxable_base", 0))
        except (KeyError, TypeError) as e:
            return f"missing field {e}"
        problems = res["problems"]
        if last4(r.get("card_last4")) not in known_cards:
            problems = problems + [f"card_last4 {r.get('card_last4')!r} is not one of the house cards {sorted(known_cards)}; "
                                   "read the payment line (e.g. 'CARD ****1234') again"]
        return "; ".join(problems) or None
    return check


def last4(text):
    return "".join(ch for ch in str(text) if ch.isdigit())[-4:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracer, tools = Tracer("receipts"), default_toolbox(OUT)

    def agent(card_name, **override):
        return Agent(dataclasses.replace(AgentCard.load(card_name), **override), tools, tracer)

    master = agent("split_master")
    ledger = Group("ledger", agent("ledger_lead"), {
        "read": agent("receipt_reader"),
        # Escalation: the same card on the next bigger model up the hierarchy.
        "read_escalated": agent("receipt_reader", name="receipt_reader_escalated", model=ESCALATE_MODEL),
        "assign": agent("ledger_lead")}, tracer)
    rules_text = (DATA / "house_rules.md").read_text()
    photos = sorted(DATA.glob("r*.png"))
    ids = [p.stem for p in photos]

    # 0. Master turns natural-language house rules into structured rules.
    plan = master.run(f"PLAN. Receipts: {ids}\n\nhouse_rules.md:\n{rules_text}", label="plan", schema=PLAN_SCHEMA)
    card_owner = {c["last4"]: c["person"] for c in plan["cards"]}
    tracer.note(f"plan: {len(plan['rules'])} rules, cards {card_owner}")

    # 1. Vision workers read every photo in parallel; code checks the arithmetic; retry once with feedback.
    jobs = [{"label": p.stem, "task": f"Receipt {p.stem}. Read it.", "images": [p.read_bytes()]} for p in photos]
    check = receipt_check(set(card_owner))
    read = ledger.fan_out("read", jobs, retries=1, check=check)

    # 2. Anything still failing escalates up the hierarchy (same card, bigger model).
    for i, r in enumerate(read):
        if "_failed" in r:
            tracer.note(f"escalating {ids[i]} to the group-tier model: {r['_failed']}")
            read[i] = ledger.fan_out("read_escalated", [jobs[i]], retries=1, check=check)[0]

    # 3. Ledger lead assigns every item to people, one receipt at a time.
    def assign_check(job, res):
        n = job["n_items"]
        seen = {a["item_index"] for a in res.get("assignments", [])}
        bad = [a for a in res.get("assignments", []) if not a["assigned_to"] or
               any(x not in plan["roommates"] for x in a["assigned_to"])]
        if seen != set(range(n)):
            return f"assign every item index 0..{n - 1} exactly once"
        return f"unknown or empty names in {bad[:2]}" if bad else None

    ok = [(rid, r) for rid, r in zip(ids, read) if "_failed" not in r]
    # The payer is whoever owns the card printed on the receipt: a lookup, so code does it.
    payer = {rid: card_owner.get(last4(r.get("card_last4"))) for rid, r in ok}
    lead_schema = AgentCard.load("ledger_lead").output_schema

    def sized(n):  # constrained decoding: exactly one assignment per item (no lazy empty lists)
        return {**lead_schema, "properties": {"assignments": {**lead_schema["properties"]["assignments"],
                                                              "minItems": n, "maxItems": n}}}
    ajobs = [{"label": rid, "n_items": len(r["items"]), "schema": sized(len(r["items"])),
              "task": f"Roommates: {plan['roommates']}\nStructured rules: {json.dumps(plan['rules'], indent=1)}\n\n"
                      f"Original house_rules.md:\n{rules_text}\n\n"
                      f"Receipt {rid} from {r['store']} on {r['date']}, paid by {payer[rid]}. Items:\n"
                      + "\n".join(f"{k}. {it['name']} (${it['line_total']})" for k, it in enumerate(r["items"]))}
             for rid, r in ok]
    assignments = ledger.fan_out("assign", ajobs, retries=1, check=assign_check)

    # 4. Code computes the money. Models never do arithmetic here.
    receipts = []
    for (rid, r), a in zip(ok, assignments):
        amap = {x["item_index"]: x for x in a.get("assignments", [])}
        receipts.append({"id": rid, "store": r["store"], "date": r["date"], "payer": payer.get(rid),
                         "tax": r["tax"], "tip": r.get("tip", 0), "total": r["total"], "subtotal": r["subtotal"],
                         "items": [{**it, "assigned_to": amap.get(k, {}).get("assigned_to", plan["roommates"]),
                                    "rule": amap.get(k, {}).get("rule", "default: shared by all")}
                                   for k, it in enumerate(r["items"])]})
    start = time.time()
    balances = compute_balances(receipts)
    tracer.tool("ledger", "compute_balances", "all receipts", time.time() - start, True)

    # 5. Master reviews and drafts payment requests. A human approves; nothing is sent.
    review = master.run(f"REVIEW.\nRules: {json.dumps(plan['rules'])}\nAssignments:\n" + json.dumps(
        [{"receipt": x["id"], "store": x["store"], "payer": x["payer"],
          "items": [{"name": it["name"], "line_total": it["line_total"], "assigned_to": it["assigned_to"]} for it in x["items"]]}
         for x in receipts], indent=1) + f"\nComputed balances (by code): {json.dumps(balances)}",
        label="review", schema=REVIEW_SCHEMA)

    with open(OUT / "expenses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receipt", "store", "date", "payer", "item", "qty", "unit_price", "line_total", "shared_by", "rule"])
        for x in receipts:
            for it in x["items"]:
                w.writerow([x["id"], x["store"], x["date"], x["payer"], it["name"], it.get("qty", 1), it["unit_price"],
                            it["line_total"], " + ".join(it["assigned_to"]), it["rule"]])
    md = ["# Roommate balances\n", "| Person | Paid | Share | Net |", "|---|---:|---:|---:|"]
    md += [f"| {p} | ${balances['paid'].get(p, 0):.2f} | ${balances['share'].get(p, 0):.2f} | ${balances['net'][p]:+.2f} |"
           for p in sorted(balances["net"])]
    md += ["\n## Settle up\n"] + [f"- **{s['from']}** pays **{s['to']}** ${s['amount']:.2f}" for s in balances["settlement"]]
    md += ["\n## Flagged for a human\n"] + [f"- {a}" for a in review["anomalies"] or ["Nothing unusual."]]
    (OUT / "balances.md").write_text("\n".join(md) + "\n")
    (OUT / "messages.txt").write_text("DRAFTS ONLY: review before sending. Nothing has been sent or paid.\n\n" + "\n\n".join(
        f"To {d['from']} (pay {d['to']} ${d['amount']:.2f}):\n{d['message']}" for d in review["drafts"]) + "\n")
    (OUT / "receipts.json").write_text(json.dumps({"plan": plan, "receipts": receipts, "balances": balances}, indent=2))
    path = tracer.save(RUNS, extra={"plan": plan, "balances": balances, "review": review})
    s = tracer.summary()
    print(f"\nDone in {s['wall_seconds']}s: {s['llm_calls']} model calls, {len(receipts)}/{len(photos)} receipts read, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\nSettlement: {balances['settlement']}\n"
          f"Outputs: {OUT}\nTrace: {path}")


if __name__ == "__main__":
    main()
