"""Lecture transcript -> study pack (summary, Anki flashcards, practice quiz).

  master  (Qwen3.8 27B, thinks only when writing)  plans, reviews for gaps, writes the summary + study plan
  groups  (Gemma 4 12B)            outline merger; fact-checker
  workers (Gemma 4 12B)            chunk summarizer; flashcard writer; quiz writer
  code                             chunking, verbatim-quote check, embedding de-dup

Run:  .venv/bin/python workflows/study_pack.py
"""
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentkit import Agent, AgentCard, Group, Tracer  # noqa: E402
from agentkit.llm import embed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "sample_data" / "lecture", ROOT / "outputs" / "study_pack"
RUNS = ROOT / "results" / "runs"

PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "flashcards_per_section": {"type": "integer", "minimum": 2, "maximum": 5},
    "quiz_per_section": {"type": "integer", "minimum": 1, "maximum": 2},
    "focus_topics": {"type": "array", "items": {"type": "string"}},
    "comprehension_instruction": {"type": "string"}, "practice_instruction": {"type": "string"}},
    "required": ["flashcards_per_section", "quiz_per_section", "focus_topics",
                 "comprehension_instruction", "practice_instruction"]}
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": {"missing_topics": {"type": "array", "items": {"type": "string"}}, "notes": {"type": "string"}},
                 "required": ["missing_topics", "notes"]}


def chunk(text, target_words=550):
    """Split on paragraph boundaries into chunks of roughly target_words."""
    chunks, cur = [], []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        cur.append(para)
        if sum(len(p.split()) for p in cur) >= target_words:
            chunks.append("\n".join(cur))
            cur = []
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def quote_check(excerpt):
    """Code-level critic: every source_quote must be copied verbatim from the excerpt."""
    ex = norm(excerpt)

    def check(job, result):
        bad = [c["source_quote"] for c in result.get("cards", result.get("questions", []))
               if not c.get("source_quote") or norm(c["source_quote"]) not in ex]
        return (f"these source_quote values are not exact copies from the excerpt: {bad[:3]}. "
                "Copy consecutive words exactly.") if bad else None
    return check


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracer = Tracer("study_pack")

    def agent(name):
        return Agent(AgentCard.load(name), None, tracer)

    master = agent("study_master")
    comprehension = Group("comprehension", agent("outline_merger"), {"summarize": agent("chunk_summarizer")}, tracer)
    practice = Group("practice", agent("fact_checker"),
                     {"cards": agent("flashcard_writer"), "quiz": agent("quiz_writer")}, tracer)
    transcript = (DATA / "transcript.txt").read_text()
    chunks = chunk(transcript)
    title = transcript.strip().splitlines()[0]

    # 0. Master plans from the title and opening only.
    plan = master.run(f"PLAN. Lecture: {title}\nLength: {len(transcript.split())} words in {len(chunks)} chunks.\n"
                      f"Opening:\n{' '.join(transcript.split()[:250])}", label="plan", schema=PLAN_SCHEMA)
    tracer.note(f"plan: {plan['flashcards_per_section']} cards/section, focus {plan['focus_topics']}")

    # 1. Comprehension workers summarize each chunk in parallel; the lead merges an outline.
    note = f"Note from the lead: {plan['comprehension_instruction']}\n\n"
    notes = comprehension.fan_out("summarize", [{"label": f"chunk {i + 1}", "task": f"{note}Chunk {i + 1} of {len(chunks)}:\n{c}"}
                                                for i, c in enumerate(chunks)])
    outline = comprehension.review("Notes in lecture order:\n" + json.dumps(
        [{"chunk": i + 1, **n} for i, n in enumerate(notes)], indent=1), label="merge outline")

    # 2. Practice workers write cards and questions per section, each checked for verbatim quotes.
    def excerpt(sec):
        ids = [i for i in sec.get("chunks", []) if 1 <= i <= len(chunks)] or list(range(1, len(chunks) + 1))
        return "\n".join(chunks[i - 1] for i in ids)

    note = f"Note from the lead: {plan['practice_instruction']}\n\n"
    jobs = [{"label": s["title"][:18], "excerpt": excerpt(s),
             "task": f"{note}Write {plan['flashcards_per_section']} flashcards for the section '{s['title']}'.\n"
                     f"Section points: {json.dumps(s['points'])}\n\nTranscript excerpt:\n{excerpt(s)}"}
            for s in outline["sections"]]
    card_sets = _parallel_with_checks(practice, "cards", jobs)
    qjobs = [{**j, "task": j["task"].replace(f"Write {plan['flashcards_per_section']} flashcards",
                                             f"Write {plan['quiz_per_section']} multiple-choice questions")} for j in jobs]
    quiz_sets = _parallel_with_checks(practice, "quiz", qjobs)

    cards = [{"id": f"s{si + 1}c{ci + 1}", "section": si, **c}
             for si, cs in enumerate(card_sets) for ci, c in enumerate(cs.get("cards", []))]

    # 3. Code de-duplicates near-identical cards with the local embedding model.
    start = time.time()
    vecs = embed([f"{c['front']} {c['back']}" for c in cards]) if cards else []
    kept, kept_vecs = [], []
    for c, v in zip(cards, vecs):
        if all(cosine(v, kv) < 0.92 for kv in kept_vecs):
            kept.append(c)
            kept_vecs.append(v)
    tracer.tool("dedupe", "embed_dedupe", f"{len(cards)}->{len(kept)} cards", time.time() - start, True)

    # 4. The fact-checker lead reviews each section's cards against that section's excerpt.
    final = []
    for si, j in enumerate(jobs):
        batch = [c for c in kept if c["section"] == si]
        if not batch:
            continue
        verdicts = practice.review(f"Excerpt:\n{j['excerpt']}\n\nCards:\n" + json.dumps(
            [{k: c[k] for k in ("id", "front", "back", "source_quote")} for c in batch], indent=1),
            label=f"check s{si + 1}")["verdicts"]
        vmap = {v["id"]: v for v in verdicts}
        for c in batch:
            v = vmap.get(c["id"], {"verdict": "keep"})
            if v["verdict"] == "drop":
                continue
            if v["verdict"] == "fix" and v.get("fixed_back"):
                c = {**c, "back": v["fixed_back"], "fixed": True}
            final.append(c)

    # 5. Master reviews for gaps; one revision round if a focus topic has no card.
    review = master.run(f"REVIEW. Focus topics: {plan['focus_topics']}\nOutline sections: "
                        f"{[s['title'] for s in outline['sections']]}\nFlashcard fronts:\n"
                        + "\n".join(f"- {c['front']}" for c in final), label="review", schema=REVIEW_SCHEMA)
    for topic in review["missing_topics"][:3]:
        extra = _parallel_with_checks(practice, "cards", [{
            "label": f"gap {topic[:14]}", "excerpt": transcript,
            "task": f"Write 2 flashcards about '{topic}'.\n\nTranscript excerpt:\n{transcript}"}])[0]
        final += [{"id": f"gap{i}", "section": -1, **c} for i, c in enumerate(extra.get("cards", []))]

    # 6. Master writes the study summary from the outline (not the raw transcript).
    summary = master.run(f"WRITE the study summary. Lecture: {title}\nFocus topics: {plan['focus_topics']}\n"
                         f"Outline:\n{json.dumps(outline, indent=1)}", label="write summary", schema=None,
                         reasoning="low")   # writing from a clean outline needs little thinking

    (OUT / "summary.md").write_text(summary.strip() + "\n")
    (OUT / "outline.json").write_text(json.dumps(outline, indent=2))
    with open(OUT / "flashcards.csv", "w", newline="") as f:
        w = csv.writer(f)
        for c in final:
            w.writerow([c["front"], c["back"], "hash-tables"])
    questions = [q for qs in quiz_sets for q in qs.get("questions", [])]
    lines, key = [f"# Practice quiz: {title}\n"], ["\n## Answer key\n"]
    for n, q in enumerate(questions, 1):
        lines.append(f"**{n}. {q['question']}**\n" + "\n".join(f"- {'ABCD'[i]}. {ch}" for i, ch in enumerate(q["choices"][:4])) + "\n")
        key.append(f"{n}. {'ABCD'[q['answer_index']] if 0 <= q['answer_index'] < 4 else '?'}: {q['explanation']}")
    (OUT / "quiz.md").write_text("\n".join(lines + key) + "\n")
    (OUT / "cards.json").write_text(json.dumps({"kept": final, "dropped_or_merged": len(cards) - len(final)}, indent=2))
    path = tracer.save(RUNS, extra={"plan": plan, "review": review, "chunks": len(chunks)})
    s = tracer.summary()
    print(f"\nDone in {s['wall_seconds']}s: {s['llm_calls']} model calls, {len(final)} flashcards, {len(questions)} quiz questions, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\nOutputs: {OUT}\nTrace: {path}")


def _parallel_with_checks(group, worker, jobs):
    """Fan out with a per-job verbatim-quote check (each job has its own excerpt)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(group.max_parallel) as pool:
        return list(pool.map(lambda j: group.fan_out(worker, [j], check=quote_check(j["excerpt"]))[0], jobs))


if __name__ == "__main__":
    main()
