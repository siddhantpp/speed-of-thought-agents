"""Score an evidence report against sample_data/research/answer_key.json (local pipeline or Claude Code).

Output contract, the same for every harness (an outputs folder or a Claude Code run folder):
  report.md     Markdown. The body cites sources as [n] (also [1, 3] or [2-4]); a '## References' section
                lists the sources numbered 1..N, each entry including its DOI. Text after References is ignored.
  sources.json  {"sources": [{"n": 1, "doi": "10....", "openalex_id": "W...", "title": "..."}, ...]}
                one entry per reference (doi, or openalex_id when a work has no DOI).
  claims.json   {"claims": [{"source_n": 1, "claim": "...", "quote": "words copied exactly from source 1's abstract"}]}
  cache/        the research MCP server's response cache (RESEARCH_CACHE_DIR): the run's fetched set.
Nothing in these files is trusted: metadata, abstracts, credibility and retraction status are re-read from the
run's cache (or looked up again for anything it never fetched, unless RESEARCH_OFFLINE=1) and re-scored by code.

Usage:  .venv/bin/python score.py research [outputs_dir]
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mcp_servers"))
from research_apis import Fetcher, Offline, bare_doi, doi_resolves, full_work, works_in  # noqa: E402
from research_credibility import credibility  # noqa: E402
from research_references import citation_numbers, is_verbatim, norm, quoted_spans  # noqa: E402

KEY = ROOT / "sample_data" / "research" / "answer_key.json"


def body_of(report):
    return re.split(r"\n#+\s*References\b", report, maxsplit=1, flags=re.I)[0]


def words_in(body):
    return len(re.findall(r"[A-Za-z0-9][\w'’-]*", re.sub(r"\[[\d,;\s–-]+\]|^#.*$", " ", body, flags=re.M)))


def paragraphs(body):
    return [p.strip() for p in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)", body)
            if p.strip() and not p.strip().startswith("#")]


def seen_in_searches(fetch, ident):
    """A work the run only saw in a search result still counts as fetched; rebuild it from the hit."""
    for rec in fetch.records("search"):
        for w in works_in(rec):
            if ident and ident.lower() in ((w["id"] or "").lower(), (w["doi"] or "").lower()):
                return w
    return None


def lookup(out, src, online):
    """(record or None, how): 'fetched' = in the run's cache, 'resolved' = looked up now, 'missing'."""
    ident = bare_doi(src.get("doi")) or src.get("openalex_id")
    if not ident:
        return None, "missing"
    run_cache = Fetcher(cache_dir=out / "cache", offline=True)
    try:
        w = full_work(run_cache, ident)
        if w and (w.get("title") or w.get("doi")):
            return w, "fetched"
    except Offline:
        pass
    w = seen_in_searches(run_cache, ident)
    if w:
        return w, "fetched"
    if not online:
        return None, "missing"
    fresh = Fetcher(cache_dir=out / "cache_scorer", offline=False)   # never mixed into the run's own cache
    try:
        w = full_work(fresh, ident)
        if w:
            return w, "resolved"
        d = bare_doi(ident)
        return (None, "resolved") if d and doi_resolves(fresh, d) else (None, "missing")
    except RuntimeError:
        return None, "missing"


def overclaims(body, patterns):
    """Sentences that overstate the evidence. A sentence that denies the claim ("not proof") or
    hedges it ("an association rather than proof") is not an overclaim."""
    negation = r"\b(not|no|never|isn't|aren't|wasn't|hasn't|cannot|can't|myth|unsupported)\b"
    hedge = r"\b(?:rather than|instead of|short of)\s+(?:\w+\s+){0,2}$"
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        if re.search(negation, sent, flags=re.I):
            continue
        hits = [m for pat in patterns for m in re.finditer(pat, sent, flags=re.I)]
        if any(not re.search(hedge, sent[:m.start()], flags=re.I) for m in hits):
            out.append(sent)
    return out


def score(out):
    out = Path(out)
    key = json.loads(KEY.read_text())
    lim = key["constraints"]
    missing_files = [f for f in ("report.md", "sources.json", "claims.json") if not (out / f).exists()]
    if missing_files:
        return {"score": "0", "error": f"missing output files: {missing_files}"}
    report = (out / "report.md").read_text()
    body = body_of(report)
    refs_text = report[len(body):]
    sources = {int(s["n"]): s for s in json.loads((out / "sources.json").read_text()).get("sources", [])}
    claims = json.loads((out / "claims.json").read_text()).get("claims", [])
    online = os.environ.get("RESEARCH_OFFLINE", "") in ("", "0")

    # 1. Every reference must be a real work: fetched by the run, or at least registered (looked up now).
    looked = {n: lookup(out, s, online) for n, s in sources.items()}
    fabricated = [{"n": n, "doi": s.get("doi"), "title": s.get("title")} for n, s in sources.items()
                  if looked[n][1] == "missing"]
    records = {n: w for n, (w, _) in looked.items() if w}
    cred = {n: credibility(w) for n, w in records.items()}

    # 2. Citations: every [n] maps to a source, every source is cited, and the reference list matches sources.json.
    cited = citation_numbers(body)
    cited_set = set(cited)
    unknown = sorted(cited_set - set(sources))
    uncited = sorted(set(sources) - cited_set)
    ref_lines = [line for line in refs_text.splitlines() if re.match(r"\s*(?:\[\d+\]|\d+\.)\s", line)]
    ref_mismatch = [n for n, s in sources.items()
                    if bare_doi(s.get("doi")) and not any(bare_doi(s.get("doi")) in line.lower() for line in ref_lines)]
    # 3. Quotes: claims.json quotes against their source's abstract; report quotes against cited abstracts.
    def abstract(n):
        return (records.get(n) or {}).get("abstract") or ""
    quote_results = []
    for c in claims:
        n = int(c.get("source_n", c.get("source", 0)) or 0)
        quote_results.append({"where": "claims.json", "source": n, "quote": c.get("quote", "")[:90],
                              "ok": bool(c.get("quote")) and is_verbatim(c["quote"], abstract(n), min_words=4)})
    uncited_paras = []
    for p in paragraphs(body):
        here = [n for n in citation_numbers(p) if n in sources]
        for q in quoted_spans(p, min_words=lim["quote_min_words_checked"]):
            quote_results.append({"where": "report.md", "source": here, "quote": q[:90],
                                  "ok": any(is_verbatim(q, abstract(n)) for n in here)})
        if len(p.split()) >= 15 and not here:
            uncited_paras.append(" ".join(p.split()[:10]) + " ...")
    bad_quotes = [q for q in quote_results if not q["ok"]]
    long_paras = [p for p in paragraphs(body) if len(p.split()) >= 15]

    # 4. Source quality, recomputed by the credibility rule from fetched metadata.
    cited_ok = [n for n in sources if n in cited_set]
    peer = [n for n in cited_ok if cred.get(n, {}).get("peer_reviewed")]
    retracted = [n for n in cited_ok if cred.get(n, {}).get("retracted")]
    cited_dois = {bare_doi((records.get(n) or {}).get("doi") or sources[n].get("doi")) for n in cited_ok}

    # 5. Coverage of the key findings: the words appear AND one of the right sources is cited.
    text = f" {norm(body)} "
    findings = {}
    for kf in key["key_findings"]:
        words_ok = all(any(f" {norm(k)}" in text for k in group) for group in kf["keyword_groups"])
        source_ok = bool(cited_dois & {d.lower() for d in kf["dois"]})
        findings[kf["id"]] = words_ok and source_ok
    rn = key["replication_nuance"]
    nuance = findings.get("replications_smaller_or_none", False) and rn["original_doi"] in cited_dois

    # 6. Red flags.
    flags = []
    for rf in key["red_flags"]:
        if rf["id"] == "overstated_effect":
            flags += [f"overstated: '{sent.strip()[:140]}'" for sent in overclaims(body, rf["patterns"])]
        elif rf["id"] == "correction_called_retraction":
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                s = sent.lower()
                if rf["pattern"] in s and any(w in s for w in rf["near"]) and "not" not in s.split("retract")[0][-30:]:
                    flags.append(f"calls the 2014 study retracted: '{sent.strip()[:120]}'")
        elif rf["id"] == "corrigendum_as_study" and rf["doi"] in cited_dois:
            flags.append("cites the 2018 corrigendum as a source")
    hand = [s.strip()[:200] for s in re.split(r"(?<=[.!?])\s+|\n+", body)
            if any(sources.get(n, {}) and bare_doi(sources[n].get("doi")) == "10.3389/fpsyg.2023.1219945"
                   for n in citation_numbers(s))]

    n_words = words_in(body)
    checks = {
        "no_fabricated_references": not fabricated,
        "citations_map_to_sources": not unknown and not uncited and bool(cited),
        "reference_list_matches_sources": len(ref_lines) == len(sources) and not ref_mismatch,
        "quotes_verbatim": bool(quote_results) and not bad_quotes,
        "peer_reviewed_share_ok": bool(cited_ok) and len(peer) / len(cited_ok) >= lim["min_peer_reviewed_share"],
        "no_retracted_sources": not retracted,
        "enough_sources": len(cited_ok) >= lim["min_sources"],
        **{f"finding:{k}": v for k, v in findings.items()},
        "replication_nuance": nuance,
        "length_ok": lim["min_words"] <= n_words <= lim["max_words"],
        "claims_cited": bool(long_paras) and (1 - len(uncited_paras) / len(long_paras)) >= 0.9,
        "no_red_flags": not flags,
    }
    got = sum(checks.values())
    return {"score": f"{got}/{len(checks)}", "points": got, "max_points": len(checks), "checks": checks,
            "words": n_words, "sources_cited": len(cited_ok),
            "sources_fetched_by_run": sum(how == "fetched" for _, how in looked.values()),
            "sources_resolved_now": sum(how == "resolved" for _, how in looked.values()),
            "fabricated_references": fabricated, "unknown_citations": unknown, "uncited_sources": uncited,
            "reference_list_mismatch": ref_mismatch,
            "peer_reviewed": f"{len(peer)}/{len(cited_ok)}", "retracted_cited": retracted,
            "credibility": {n: {"score": c["score"], "tier": c["tier"], "peer_reviewed": c["peer_reviewed"]}
                            for n, c in cred.items()},
            "quotes_checked": len(quote_results), "quotes_not_verbatim": bad_quotes,
            "uncited_paragraphs": uncited_paras, "key_findings": findings,
            "key_findings_covered": f"{sum(findings.values())}/{len(findings)}", "red_flags": flags,
            "for_hand_review": {"eeg_study_citations": hand,
                                "rule": next(r["rule"] for r in key["red_flags"] if r["id"] == "eeg_overreach")}}
