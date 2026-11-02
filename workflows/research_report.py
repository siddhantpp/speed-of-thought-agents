"""A student's research question -> a short evidence report built only on credible, checkable sources.

  master  (Qwen3.8 27B)   plans sub-questions and search queries (no thinking); writes the report (thinks a little)
  group   (Gemma 4 12B)   evidence lead groups findings per sub-question and flags conflicts
  workers (Gemma 4 12B)   screeners call search_works over MCP and judge each hit's relevance;
                          extractors read ONE abstract each and return findings with verbatim quotes
                          (a job that still fails its check escalates to the 27B)
  code                    the research MCP server (OpenAlex + Crossref search; ERIC and Europe PMC for missing
                          abstracts; every response cached), plan and id checks, de-dup, the credibility rule,
                          citation snowballing, source selection, quote and citation checks, APA references

Run:  .venv/bin/python workflows/research_report.py
      RESEARCH_OFFLINE=1 .venv/bin/python workflows/research_report.py   # no network: replay the cache
"""
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "mcp_servers")]
from agentkit import TIER_MODELS, Agent, AgentCard, Group, Toolbox, Tracer  # noqa: E402
from agentkit.mcp import MCPServer  # noqa: E402
from research_apis import Fetcher, searched_hits  # noqa: E402
from research_references import citation_numbers, is_verbatim, norm, quoted_spans, renumber  # noqa: E402

DATA, OUT = ROOT / "sample_data" / "research", ROOT / "outputs" / "research"
RUNS = ROOT / "results" / "runs"
ESCALATE_MODEL = next((m for m in (TIER_MODELS["group"], TIER_MODELS["master"]) if m != TIER_MODELS["worker"]),
                      TIER_MODELS["master"])
WORDS, MIN_SOURCES, MAX_SOURCES, PER_SUB_QUESTION, MAX_FETCH, MAX_SNOWBALL = (800, 1200), 5, 12, 4, 24, 8

STR = {"type": "string"}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "sub_questions": {"type": "array", "minItems": 3, "maxItems": 4, "items": {
        "type": "object", "additionalProperties": False, "properties": {
            "question": STR, "queries": {"type": "array", "minItems": 2, "maxItems": 2, "items": STR}},
        "required": ["question", "queries"]}},
    "screening_note": STR},
    "required": ["sub_questions", "screening_note"]}


def label(w):
    """'Morehead et al. (2019), Educational Psychology Review': how sources are named to models."""
    fam = [a["family"] for a in w.get("authors") or []]
    who = fam[0] if len(fam) == 1 else " & ".join(fam) if len(fam) == 2 else f"{fam[0]} et al." if fam else w["title"][:40]
    return f"{who} ({w.get('year')}), {w.get('venue')}"


def plan_problem(plan):
    """Code-level critic for the plan: every query 2 to 4 different words, no query used twice."""
    queries = [q for sq in plan["sub_questions"] for q in sq["queries"]]
    bad = [q for q in queries if not 2 <= len(q.split()) <= 4 or len(set(q.lower().split())) < len(q.split())]
    if bad:
        return f"these queries are not 2 to 4 different words: {bad}"
    dup = sorted({q for q in queries if queries.count(q) > 1})
    return f"these queries are repeated: {dup}" if dup else None


def report_problems(text, sources):
    """Code-level critic for the master's draft. Returns (hard, soft) problems.
    Hard ones fail the run: a citation with no source, or a quote that is not word for word."""
    body = text.split("\n## References")[0]
    hard, soft = [], []
    unknown = sorted({n for n in citation_numbers(body) if n not in sources})
    if unknown:
        hard.append(f"cites {unknown}, but the pack has no source with that number; use only 1 to {len(sources)}")
    for para in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)", body):
        para = para.strip()
        if not para or para.startswith("#"):
            continue
        cited = [n for n in citation_numbers(para) if n in sources]
        for q in quoted_spans(para, min_words=4):
            if not cited:
                hard.append(f'the quote "{q[:60]}" has no citation next to it')
            elif not any(is_verbatim(q, sources[n]["abstract"]) for n in cited):
                hard.append(f'the quote "{q[:60]}..." is not word for word in the abstract of {cited}; copy it exactly '
                            "from the pack or drop the quotation marks and paraphrase")
        if len(para.split()) >= 15 and not cited:
            soft.append(f"this paragraph has no citation: '{' '.join(para.split()[:10])}...'")
    n_words = len(re.findall(r"[A-Za-z0-9][\w'’-]*", re.sub(r"\[[\d,;\s–-]+\]|^#.*$", " ", body, flags=re.M)))
    if not WORDS[0] <= n_words <= WORDS[1]:
        soft.append(f"the report body is {n_words} words; it must be {WORDS[0]} to {WORDS[1]}")
    distinct = len(set(citation_numbers(body)) & set(sources))
    if distinct < min(MIN_SOURCES, len(sources)):
        soft.append(f"cite at least {min(MIN_SOURCES, len(sources))} different sources; you cited {distinct}")
    return hard, soft


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracer, tools = Tracer("research"), Toolbox()
    # The research tools run as a separate MCP server process; agentkit lists its tools like local ones.
    env = {**os.environ, "RESEARCH_CACHE_DIR": str(OUT / "cache")}
    offline = os.environ.get("RESEARCH_OFFLINE", "") not in ("", "0")
    if offline:   # replay this folder's cache (an earlier online run's), then make_cache.py's, if built
        env["RESEARCH_SEED_DIR"] = str(DATA / "cache")
    server = tools.add_mcp(MCPServer([sys.executable, str(ROOT / "mcp_servers" / "research_server.py")], env=env))

    def agent(card_name, **override):
        return Agent(dataclasses.replace(AgentCard.load(card_name), **override), tools, tracer)

    def bigger(card_name):   # escalation: the same card on the next bigger model up the hierarchy
        return agent(card_name, name=f"{card_name}_escalated", model=ESCALATE_MODEL)

    master = agent("research_master")
    search = Group("search", None, {"screen": agent("research_screener"),
                                    "screen_escalated": bigger("research_screener")}, tracer)
    lead = agent("research_evidence_lead")
    evidence = Group("evidence", lead, {"extract": agent("research_extractor"),
                                        "extract_escalated": bigger("research_extractor"),
                                        "lead": lead, "lead_escalated": bigger("research_evidence_lead")}, tracer)
    brief = (DATA / "question.md").read_text()
    question = next(line.strip() for line in brief.splitlines() if line.strip() and not line.startswith("#"))

    def fan_out_escalating(group, worker, jobs, check):
        """Workers in parallel, each checked by code and retried once; what still fails goes to the 27B."""
        results = group.fan_out(worker, jobs, check=check)
        for i, r in enumerate(results):
            if "_failed" in r:
                tracer.note(f"escalating {jobs[i]['label']} to {ESCALATE_MODEL}: {r['_failed'][:120]}")
                results[i] = group.fan_out(f"{worker}_escalated", [jobs[i]], check=check)[0]
        return results

    # 0. The master plans from the question alone: sub-questions and search queries, no answers.
    #    Code checks the queries (the search needs every word, so short and distinct) and retries once.
    plan = master.run(f"PLAN.\n{brief}", label="plan", schema=PLAN_SCHEMA)
    if problem := plan_problem(plan):
        tracer.note(f"plan check: {problem}")
        plan = master.run(f"PLAN.\n{brief}\n\nYour previous plan had a problem: {problem}\nFix it.",
                          label="plan:retry", schema=PLAN_SCHEMA)
    for sq in plan["sub_questions"]:   # still wrong after the retry: repair (drop repeated words, keep 4)
        sq["queries"] = list(dict.fromkeys(" ".join(list(dict.fromkeys(q.split()))[:4]) for q in sq["queries"]))
    subqs = [sq["question"] for sq in plan["sub_questions"]]
    sq_list = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(subqs))
    tracer.note("plan: " + " | ".join(f"{i + 1}. {sq['question'][:60]} {sq['queries']}" for i, sq in enumerate(plan["sub_questions"])))

    # 1. Screeners: one query each. The worker calls search_works itself (an MCP tool call), and may
    #    reword the query. Code checks every id against what search_works really returned: the server's
    #    cache of this run is the fetched set, so an invented id cannot slip through.
    jobs = [{"label": f"q{si + 1}.{qi + 1}", "query": q, "sub_question": si + 1,
             "task": f"Note from the master: {plan['screening_note']}\n\nThe student's question: {question}\n"
                     f"Its sub-questions:\n{sq_list}\nThis search is for sub-question {si + 1}.\n"
                     f"Make exactly this call: search_works(query={json.dumps(q)})"}
            for si, sq in enumerate(plan["sub_questions"]) for qi, q in enumerate(sq["queries"])]
    fetched = Fetcher(cache_dir=OUT / "cache", offline=True)   # read-only view of the server's cache

    def screen_check(job, res):
        got = [s["id"] for s in res.get("screened", [])]
        if not got:   # fine if the search really finds nothing (code replays it to see); else the worker skipped it
            found = tools.call("search_works", {"query": job["query"]}, tracer, "code", f"check {job['label']}")
            return None if found.get("error") or not found.get("results") else \
                "call search_works once, then screen every hit it returns"
        returned = [{h["id"] for h in hits} for hits in searched_hits(fetched)]
        if bad := [i for i in got if not any(i in ids for ids in returned)]:
            return f"these ids were never returned by search_works: {bad[:3]}; copy each id exactly from its results"
        # The search this worker ran: the smallest one containing all its ids (else the best overlap).
        ids = min((ids for ids in returned if set(got) <= ids), key=len, default=None) or \
            max(returned, key=lambda ids: len(ids & set(got)))
        if len(ids - set(got)) > len(ids) / 2:
            return f"screen every hit: search_works returned {len(ids)} hits and you screened {len(set(got) & ids)}"
        return None

    screened = fan_out_escalating(search, "screen", jobs, screen_check)

    # 2. Code: pool the relevant hits, merge duplicates (same DOI or same title), apply the credibility rule.
    start, pool, excluded = time.time(), {}, []
    hits = {h["id"]: h for search_hits in searched_hits(fetched) for h in search_hits}
    for job, res in zip(jobs, screened):
        for s in res.get("screened", []):
            h = hits.get(s["id"])
            if not (s.get("relevant") and h):
                continue
            key = h["doi"] or norm(h["title"])[:90]
            twin = next((k for k, c in pool.items() if norm(c["title"])[:90] == norm(h["title"])[:90]), None)
            c = pool.get(key) or (pool.get(twin) if twin else None)
            if c is None:
                c = pool[key] = {**h, "votes": 0, "sub_questions": set()}
            elif h["credibility"] > c["credibility"]:   # keep the better version (journal article over preprint)
                c.update({k: h[k] for k in ("id", "doi", "title", "venue", "type", "credibility", "peer_reviewed")})
            c["votes"] += 1
            c["sub_questions"].add(job["sub_question"])
    ranked = sorted(pool.values(), key=lambda c: (-c["votes"], -c["credibility"]))
    vetted = []

    def vet(wid, votes, sub_questions, why=None, snowballed=False):
        """Fetch the full record (get_work) and apply the credibility rule; record why anything is dropped."""
        w = {} if why else tools.call("get_work", {"work_id": wid}, tracer, "code", f"get {wid}")
        cred = w.get("credibility", {})
        why = why or w.get("error") or (None if cred.get("vetted") else "; ".join(cred.get("reasons", [])[:2])) or \
            (None if len(w.get("abstract", "").split()) >= 40 else "no open abstract, so its claims cannot be checked")
        if why:
            excluded.append({"id": wid, "title": w.get("title") or pool_titles.get(wid), "year": w.get("year"),
                             "type": w.get("type"), "credibility": cred.get("score"), "reason": why})
        else:
            vetted.append({**w, "votes": votes, "sub_questions": sorted(sub_questions), "snowballed": snowballed})
    pool_titles = {c["id"]: c["title"] for c in ranked}
    for k, c in enumerate(ranked):
        vet(c["id"], c["votes"], c["sub_questions"],
            ("retracted" if c["is_retracted"] else None) or
            (None if c["peer_reviewed"] else f"not a peer-reviewed journal article ({c['type']})") or
            (None if k < MAX_FETCH else "lower-ranked than the sources fetched"))
    # Snowballing: a paper that two or more vetted sources cite is a candidate too. This is how the
    # famous original study gets found even when no query matched it. Single-work lookups are free.
    seen = {c["id"] for c in ranked} | {w["id"] for w in vetted}
    cited_by = {}
    for w in list(vetted):
        for ref in w.get("references", []):
            cited_by.setdefault(ref, []).append(w)
    snow = sorted(((ref, ws) for ref, ws in cited_by.items() if len(ws) >= 2 and ref not in seen),
                  key=lambda x: -len(x[1]))[:MAX_SNOWBALL]
    for ref, ws in snow:
        vet(ref, len(ws), {k for w in ws for k in w["sub_questions"]}, snowballed=True)
    # Selection is a rule too: the best few screened works per sub-question (votes, then credibility),
    # then the most-cited snowballed works, at most MAX_SOURCES in all.
    screened_ok, snowballed = [w for w in vetted if not w.get("snowballed")], [w for w in vetted if w.get("snowballed")]
    chosen = []
    for k in range(1, len(subqs) + 1):
        chosen += [w for w in sorted((w for w in screened_ok if k in w["sub_questions"]),
                                     key=lambda w: (-w["votes"], -w["credibility"]["score"]))[:PER_SUB_QUESTION]
                   if w not in chosen]
    chosen = (chosen + sorted(snowballed, key=lambda w: -w["votes"]))[:MAX_SOURCES]
    excluded += [{"id": w["id"], "title": w["title"], "year": w["year"], "type": w["type"],
                  "credibility": w["credibility"]["score"], "reason": "vetted, but not among the top sources selected"}
                 for w in vetted if w not in chosen]
    tracer.tool("code", "dedupe_score_select", f"{len(pool)} hits + {len(snow)} cited -> {len(chosen)} sources",
                time.time() - start, True)

    # 3. Extractors: one abstract each; code checks every quote word for word.
    ejobs = [{"label": f"S{k + 1} {label(w)[:14]}", "abstract": w["abstract"],
              "task": f"Sub-questions:\n{sq_list}\n\nSource: {label(w)}\nTitle: {w['title']}\nAbstract:\n{w['abstract']}"}
             for k, w in enumerate(chosen)]

    def quote_check(job, res):
        fs = res.get("findings", [])
        if not fs and not job.get("asked_again"):   # screeners judged it relevant: one more look, then accept
            job["asked_again"] = True
            return ("findings is empty, but this source was found relevant to the question. Re-read the abstract "
                    "and give 1 to 4 findings for the sub-questions it answers; empty only if it answers none")
        if bad := [f["sub_question"] for f in fs if not 1 <= f["sub_question"] <= len(subqs)]:
            return f"sub_question must be a number from 1 to {len(subqs)}; you used {bad}"
        if bad := [f["quote"] for f in fs if not is_verbatim(f["quote"], job["abstract"], min_words=6)]:
            return (f"these quotes are not exact copies of 6+ consecutive words from the abstract: {bad[:2]}. "
                    "Copy consecutive words exactly; no '...', no changes.")
        return None

    extracted = fan_out_escalating(evidence, "extract", ejobs, quote_check)
    findings, used = [], []
    for w, res in zip(chosen, extracted):
        if "_failed" in res or not res.get("findings"):
            excluded.append({"id": w["id"], "title": w["title"], "year": w["year"], "type": w["type"],
                             "credibility": w["credibility"]["score"],
                             "reason": res.get("_failed") or "its abstract answers none of the sub-questions"})
            continue
        used.append({**w, "design": res["design"], "sample": res["sample"]})
        n = len(used)
        findings += [{"id": f"F{len(findings) + i + 1}", "source": n, **f} for i, f in enumerate(res["findings"])]
    sources = {n: w for n, w in enumerate(used, 1)}
    if len(sources) < MIN_SOURCES:   # not enough checkable evidence: say so instead of writing a thin report
        server.close()
        tracer.save(RUNS, extra={"plan": plan, "failed": "too few sources", "excluded": excluded})
        (OUT / "sources.json").write_text(json.dumps({"question": question, "sources": [], "excluded": excluded},
                                                     indent=2, ensure_ascii=False))
        sys.exit(f"FAILED: only {len(sources)} vetted sources with checkable findings (need {MIN_SOURCES}); "
                 f"see {OUT / 'sources.json'} for what was excluded and why")

    # 4. The evidence lead groups compact findings (no abstracts) and flags conflicts.
    fids = {f["id"] for f in findings}
    compact = [{"id": f["id"], "sub_question": f["sub_question"],
                "source": f"{label(sources[f['source']])}; {sources[f['source']]['design']}; {sources[f['source']]['sample']}",
                "claim": f["claim"]} for f in findings]

    def lead_check(job, res):
        cited = [i for s in res.get("sections", []) for p in s["points"] for i in p["finding_ids"]] + \
                [i for c in res.get("conflicts", []) for i in c["finding_ids"]]
        if bad := sorted(set(cited) - fids):
            return f"these finding ids do not exist: {bad[:5]}; use only {sorted(fids)[:3]}..."
        missing = {f["sub_question"] for f in findings} - {s["sub_question"] for s in res.get("sections", [])}
        return f"add a section for sub-question(s) {sorted(missing)}" if missing else None

    lead_schema = AgentCard.load("research_evidence_lead").output_schema
    n_sections = len({f["sub_question"] for f in findings})
    sized = {**lead_schema, "properties": {**lead_schema["properties"], "sections": {   # constrained decoding:
        **lead_schema["properties"]["sections"], "minItems": n_sections, "maxItems": n_sections}}}   # no empty answer
    grouped = fan_out_escalating(evidence, "lead", [{"label": "group findings", "schema": sized, "task": (
        f"Question: {question}\nSub-questions:\n{sq_list}\n\nFindings:\n{json.dumps(compact, indent=1)}")}], lead_check)[0]
    if "_failed" in grouped:   # still wrong after escalation: fall back to listing findings by sub-question
        grouped = {"sections": [{"sub_question": k, "points": [{"point": f["claim"], "finding_ids": [f["id"]]}
                                                              for f in findings if f["sub_question"] == k]}
                                for k in sorted({f["sub_question"] for f in findings})], "conflicts": [], "caveats": []}

    # 5. The master writes for a person, from the evidence pack only (sources numbered by code).
    by_id = {f["id"]: f for f in findings}

    def tags(ids):
        return "[" + ", ".join(str(n) for n in sorted({by_id[i]["source"] for i in ids if i in by_id})) + "]"
    pack = [f"Question: {question}\n\nSOURCES (cite by number; all are peer-reviewed and passed the credibility rule):"]
    pack += [f"[{n}] {label(w)}. {w['title']}. Design: {w['design']}. Sample: {w['sample']}. "
             f"Credibility {w['credibility']['score']:.0f}/100." for n, w in sources.items()]
    pack.append("\nEVIDENCE BY SUB-QUESTION (grouped by the evidence lead):")
    for s in sorted(grouped["sections"], key=lambda s: s["sub_question"]):
        if 1 <= s["sub_question"] <= len(subqs):
            pack.append(f"\n{s['sub_question']}. {subqs[s['sub_question'] - 1]}")
            pack += [f"- {p['point']} {tags(p['finding_ids'])}" for p in s["points"]]
    pack.append("\nCONFLICTS:")
    pack += [f"- {c['description']} {tags(c['finding_ids'])}" for c in grouped["conflicts"]] or ["- none flagged"]
    pack.append("\nCAVEATS:\n" + "\n".join(f"- {c}" for c in grouped["caveats"] or ["none flagged"]))
    pack.append("\nFINDINGS WITH QUOTES YOU MAY USE (copy quotes exactly):")
    pack += [f"[{f['source']}] {f['claim']} Quote: \"{f['quote']}\"" for f in findings]
    pack = "\n".join(pack)
    task = f"WRITE the evidence report.\n\nWhat the student asked for:\n{brief}\n\nEVIDENCE PACK\n{pack}"
    draft = master.run(task, label="write", schema=None, reasoning="low")
    hard, soft = report_problems(draft, sources)
    if hard or soft:   # one rewrite with the specific problems
        tracer.note(f"report check: {len(hard)} hard, {len(soft)} soft problems; asking for one rewrite")
        draft = master.run(f"{task}\n\nYOUR DRAFT:\n{draft}\n\nFix these problems and return the whole corrected report:\n"
                           + "\n".join(f"- {p}" for p in hard + soft), label="rewrite", schema=None, reasoning="low")
        hard, soft = report_problems(draft, sources)

    # 6. Code renumbers citations in order of first use and builds the reference list from fetched metadata.
    body = draft.split("\n## References")[0].strip()
    order = list(dict.fromkeys(n for n in citation_numbers(body) if n in sources))
    new = {old: i for i, old in enumerate(order, 1)}
    body = renumber(body, new)
    refs = [sources[old] for old in order]
    reference_list = "\n".join(f"{i}. {w['apa']}" for i, w in enumerate(refs, 1))
    made = (f"*How this report was made: {len(jobs)} searches; {len(pool)} relevant works after merging duplicates, "
            f"plus {len(snow)} that two or more of them cite; "
            f"{len(used)} peer-reviewed, unretracted sources with an open abstract read in full; {len(refs)} cited. "
            "Credibility is a fixed rule in code (mcp_servers/research_credibility.py) and every quote was checked "
            "word for word against its source's abstract.*") if not hard else \
        "**This report FAILED its checks:** " + "; ".join(hard)
    (OUT / "report.md").write_text(f"{body}\n\n## References\n\n{reference_list}\n\n---\n{made}\n")
    (OUT / "references.md").write_text(f"# References\n\n{reference_list}\n")
    (OUT / "sources.json").write_text(json.dumps({"question": question, "sources": [
        {"n": i, "openalex_id": w["id"], "doi": w["doi"], "title": w["title"], "year": w["year"], "venue": w["venue"],
         "type": w["type"], "design": w["design"], "sample": w["sample"], "credibility": w["credibility"],
         "abstract_source": w["abstract_source"], "updates": w["updates"], "reference": w["apa"]}
        for i, w in enumerate(refs, 1)],
        "not_cited": [{"openalex_id": w["id"], "doi": w["doi"], "title": w["title"]} for w in used if w not in refs],
        "screening": [{"query": j["query"], "sub_question": j["sub_question"],
                       "relevant": [x["id"] for x in r.get("screened", []) if x.get("relevant")],
                       "not_relevant": [x["id"] for x in r.get("screened", []) if not x.get("relevant")]}
                      for j, r in zip(jobs, screened)],
        "excluded": excluded}, indent=2, ensure_ascii=False))
    (OUT / "claims.json").write_text(json.dumps({
        "claims": [{"source_n": new[f["source"]], "sub_question": f["sub_question"], "claim": f["claim"], "quote": f["quote"]}
                   for f in findings if f["source"] in new],
        "sub_questions": subqs, "grouped": grouped,
        "checks": {"hard_problems": hard, "soft_problems": soft, "sources_cited": len(refs)}}, indent=2, ensure_ascii=False))
    server.close()
    path = tracer.save(RUNS, extra={"plan": plan, "checks": {"hard": hard, "soft": soft}})
    s = tracer.summary()
    print(f"\nDone in {s['wall_seconds']}s: {s['llm_calls']} model calls, {s['tool_calls']} tool calls, "
          f"{len(refs)} sources cited, {len(findings)} findings, "
          f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens.\nOutputs: {OUT}\nTrace: {path}")
    for p in soft:
        print(f"warning: {p}")
    if hard:   # a citation that maps to nothing, or a quote that isn't verbatim, fails the run
        sys.exit("FAILED report checks:\n" + "\n".join(f"- {p}" for p in hard))


if __name__ == "__main__":
    main()
