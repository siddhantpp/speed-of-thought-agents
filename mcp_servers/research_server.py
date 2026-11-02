"""Research tools as an MCP server: search the scholarly record, read one work, check a DOI.

One server, three harnesses: agentkit (workflows/research_report.py), LM Studio's chat app (mcp.json)
and Claude Code (--mcp-config claude_compare/mcp/research.json) all get the same three tools.

Data: free, keyless public APIs only (OpenAlex, Crossref, ERIC, Europe PMC, doi.org; see research_apis.py).
Credibility and APA references are computed here by code (research_credibility.py, research_references.py),
so no model ever rates a source or types a reference.

Environment:
  RESEARCH_CACHE_DIR  where every API response is saved (default ./cache): the run's "fetched set"
  RESEARCH_OFFLINE=1  replay the cache only; never touch the network
  RESEARCH_SEED_DIR   optional read-only cache to replay from first (e.g. sample_data/research/cache)

Run by hand:  .venv/bin/python mcp_servers/research_server.py     (speaks MCP on stdin/stdout)
"""
import sys

from mcp.server.mcpserver import MCPServer

from research_apis import (Fetcher, Offline, as_hit, bare_doi, crossref_record, doi_resolves, from_crossref,
                           full_work, search)
from research_credibility import credibility
from research_references import apa

app = MCPServer("research", instructions=(
    "Scholarly search for evidence reports. search_works finds candidate papers; get_work returns one paper's "
    "metadata, abstract, code-computed credibility and APA reference; check_doi confirms a DOI exists and "
    "whether it was corrected or retracted. Cite only works you fetched, and quote only their abstracts."))
fetch = Fetcher()      # configured from the environment variables above


def offline_error(what):
    """Offline and not cached. When nothing at all is cached yet, say how to fill the cache."""
    hint = ("; the response cache is empty, so there is nothing to replay: run once without RESEARCH_OFFLINE "
            "(network needed) to fetch and cache the responses") if fetch.empty() else ""
    return {"error": f"offline: {what}{hint}"}


@app.tool()
def search_works(query: str, from_year: int = 2000, limit: int = 10) -> dict:
    """Search scholarly works by keywords: OpenAlex (title or abstract must contain EVERY word) and Crossref
    (titles, authors, venues), interleaved, duplicates merged. Use 2 or 3 distinctive words, e.g. 'sleep
    deprivation memory' or authors' surnames. Returns up to `limit` hits (max 25) published since `from_year`,
    peer-reviewed journal articles and reviews first. Each hit has an id (use it with get_work), title, first
    author, year, venue, type (article, review, preprint, dissertation, ...), citation count, retraction flag,
    a code-computed credibility score (0-100) and the first words of its abstract when one is available."""
    try:
        res = search(fetch, query, from_year, limit)
    except Offline:
        return offline_error(f"no cached search close to {query!r}")
    except RuntimeError as e:
        return {"error": str(e)}
    hits = [as_hit(w) for w in res["works"]]
    out = {"query": query, "total_found": res["total_found"], "results": hits}
    if res["replayed_from"]:
        out["note"] = f"offline replay of the cached search {res['replayed_from']!r}"
    elif res["notes"]:
        out["note"] = "; ".join(res["notes"])
    return out


@app.tool()
def get_work(work_id: str) -> dict:
    """Get one work by OpenAlex id (e.g. W1234567890) or DOI. Returns full metadata, the abstract (from
    OpenAlex, Crossref, ERIC or Europe PMC; empty if no open source has it), correction/retraction
    notices, the OpenAlex ids of the works it cites (references), a code-computed credibility block
    with its reasons, and the APA 7 reference."""
    try:
        w = full_work(fetch, work_id)
    except Offline:
        return offline_error(f"{work_id!r} is not in the cache")
    except RuntimeError as e:
        return {"error": str(e)}
    if not w:
        return {"error": f"no work found for {work_id!r}"}
    return {**w, "credibility": credibility(w), "apa": apa(w)}


@app.tool()
def check_doi(doi: str) -> dict:
    """Check that a DOI is real (registered with Crossref or the DOI system) and whether a correction,
    retraction or expression of concern has been published for it."""
    d = bare_doi(doi)
    if not d:
        return {"doi": doi, "exists": False, "error": "that is not a DOI (they start with 10.)"}
    try:
        m = crossref_record(fetch, d)
        if not m:
            return {"doi": d, "exists": doi_resolves(fetch, d), "registrar": "not Crossref"}
    except Offline:
        return offline_error(f"{d} is not in the cache")
    except RuntimeError as e:
        return {"error": str(e)}
    r = from_crossref(m)
    return {"doi": d, "exists": True, "registrar": "Crossref", "title": r["title"], "venue": r["venue"],
            "year": r["year"], "type": r["crossref_type"], "updates": r["updates"],
            "retracted": any(u["type"] in ("retraction", "withdrawal", "removal") for u in r["updates"]),
            "is_update_notice": r["is_update_notice"]}


if __name__ == "__main__":
    if fetch.offline and fetch.empty():
        print(f"research server: RESEARCH_OFFLINE is set, but there is nothing to replay in {fetch.cache_dir}"
              + (f" or {fetch.seed_dir}" if fetch.seed_dir else "") + "; run once without RESEARCH_OFFLINE "
              "to fill the cache", file=sys.stderr)
    app.run()   # stdio
