"""Build responses/: the fictional API responses that tests/test_research.py replays.

papers.json holds made-up papers (fictional authors, journals and abstracts; DOIs under the reserved 10.5555
prefix). This script serves them from a fake OpenAlex, Crossref, ERIC, Europe PMC and doi.org, in each API's
own shape and with the fields a real answer carries but the cache must not keep (affiliations, ORCIDs,
conflict-of-interest statements, reference lists). Then it runs the research server's own lookups against
them, exactly as make_cache.py does against the real APIs, so trim() and the Fetcher write responses/ in
the format a real run writes. No network is used. Deterministic: the tests check that a rebuild matches.

    .venv/bin/python tests/fixtures/research/make_fixtures.py
"""
import contextlib
import io
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works in the exported repo (tests/fixtures/research) and in the source tree (tools/repo_extras/tests/...).
ROOT = next(p for p in (HERE.parents[2], HERE.parents[4] / "demos") if (p / "agentkit").is_dir())
sys.path.insert(0, str(ROOT / "mcp_servers"))
import research_apis  # noqa: E402
from research_apis import Fetcher, crossref_record, doi_resolves, full_work, search  # noqa: E402

PAPERS = json.loads((HERE / "papers.json").read_text())
WORKS = {w["key"]: w for w in PAPERS["works"]}
BY_DOI = {w["doi"]: w for w in PAPERS["works"] if w.get("doi")}
BY_ID = {w["openalex_id"]: w for w in PAPERS["works"] if w.get("openalex_id")}
SEARCHES = {s["query"]: s for s in PAPERS["searches"]}
FETCHED_AT = "2026-11-02T12:00:00Z"


def words(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def full_title(w):
    return f"{w['title']}: {w['subtitle']}" if w.get("subtitle") else w["title"]


def first_last(w):
    first, _, last = (w.get("pages") or "").partition("-")
    return (first or w.get("article_number")), (last or (first if first else None))


def openalex(w):
    """One work as OpenAlex returns it."""
    online = w.get("online") or [w["year"], 1, 1]
    source = {"type": w.get("venue_type", "journal"), "display_name": w["venue"], "issn_l": w.get("issn"),
              "issn": [w["issn"]] if w.get("issn") else None, "host_organization_name": "Northvale University Press"}
    text = w["abstract"] if w.get("abstract_in") == "openalex" else None
    index = {}
    for i, word in enumerate((text or "").split(" ")):
        index.setdefault(word, []).append(i)
    first, last = first_last(w)
    return {"id": f"https://openalex.org/{w['openalex_id']}", "doi": f"https://doi.org/{w['doi']}" if w.get("doi") else None,
            "title": full_title(w), "display_name": full_title(w), "publication_year": online[0],
            "publication_date": "-".join(f"{x:02d}" for x in online), "type": w["type"],
            "cited_by_count": w["cited_by_count"], "is_retracted": bool(w.get("is_retracted")),
            "open_access": {"is_oa": False, "oa_status": "closed", "oa_url": None},
            "primary_location": {"source": source, "landing_page_url": "https://example.org/landing"},
            "locations": [{"source": source}],
            "authorships": [{"author_position": "first" if i == 0 else "middle",
                             "author": {"id": f"https://openalex.org/A{9000 + i}", "display_name": f"{g} {f}",
                                        "orcid": f"https://orcid.org/0000-0000-0000-{i:04d}"},
                             "raw_author_name": f"{g} {f}",
                             "raw_affiliation_strings": [f"Department of Psychology, Northvale University, {f.lower()}@example.edu"]}
                            for i, (g, f) in enumerate(w["authors"])],
            "biblio": {"volume": w.get("volume"), "issue": w.get("issue"), "first_page": first, "last_page": None
                       if w.get("article_number") else last},
            "abstract_inverted_index": index or None,
            "referenced_works": [f"https://openalex.org/{x}" for x in ("W0000000101", "W0000000106")
                                 if x != w["openalex_id"] and w["year"] > 2014]}


def crossref(w):
    """One work as Crossref returns it."""
    def date(parts):
        return {"date-parts": [parts]}
    m = {"DOI": w["doi"], "title": [w["title"]], "container-title": [w["venue"]], "type": w["crossref_type"],
         "publisher": "Northvale University Press", "is-referenced-by-count": w["cited_by_count"],
         "author": [{"given": g, "family": f, "sequence": "first" if i == 0 else "additional",
                     "ORCID": f"https://orcid.org/0000-0000-0000-{i:04d}", "authenticated-orcid": False,
                     "affiliation": [{"name": "Department of Psychology, Northvale University"}]}
                    for i, (g, f) in enumerate(w["authors"])]
         + ([{"name": "Northvale University", "sequence": "additional", "affiliation": []}] if w["authors"] else []),
         "assertion": [{"name": "conflict_of_interest", "label": "Conflict of interest",
                        "value": "The first author consults for a study-music app and received speaker fees."}],
         "reference": [{"key": "ref1", "unstructured": "A cited work, with its authors' affiliations."}],
         "issued": date(w.get("online") or [w["year"]]), "published-print": date([w["year"], 6])}
    for k, v in (("subtitle", [w["subtitle"]] if w.get("subtitle") else None), ("ISSN", [w["issn"]] if w.get("issn") else None),
                 ("volume", w.get("volume")), ("issue", w.get("issue")), ("page", w.get("pages")),
                 ("article-number", w.get("article_number")), ("published-online", date(w["online"]) if w.get("online") else None)):
        if v:
            m[k] = v
    if w.get("updates"):
        m["updated-by"] = [{"type": u["type"], "DOI": u["doi"], "updated": date(u["date"]), "source": "publisher",
                            "label": u["type"].title(), "record-id": f"rec-{i}"} for i, u in enumerate(w["updates"])]
    if w.get("update_to"):
        m["update-to"] = [{"type": w["update_to"]["type"], "DOI": w["update_to"]["doi"], "label": "Correction",
                           "updated": date([w["year"], 7, 31]), "source": "publisher"}]
    if w.get("abstract_in") == "crossref":
        m["abstract"] = f"<jats:p>{w['abstract']}</jats:p>"
    return m


def respond(url):
    """(status, body) for one request, like the real APIs."""
    u = urllib.parse.urlparse(url)
    q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
    path = urllib.parse.unquote(u.path)
    if u.netloc == "api.openalex.org" and path == "/works":
        s = SEARCHES[q["search.title_and_abstract"]]
        return 200, {"meta": {"count": s["openalex_count"], "db_response_time_ms": 12, "page": 1, "per_page": 40},
                     "results": [openalex(WORKS[k]) for k in s["openalex"]], "group_by": []}
    if u.netloc == "api.openalex.org":
        key = path.removeprefix("/works/")
        w = BY_DOI.get(key.removeprefix("doi:")) if key.startswith("doi:") else BY_ID.get(key)
        return (200, openalex(w)) if w and w.get("openalex_id") else (404, {"error": "not found"})
    if u.netloc == "api.crossref.org" and path == "/works":
        s = SEARCHES[q["query.bibliographic"]]
        return 200, {"status": "ok", "message-type": "work-list", "message": {
            "total-results": s["crossref_count"], "items": [crossref(WORKS[k]) for k in s["crossref"]]}}
    if u.netloc == "api.crossref.org":
        w = BY_DOI.get(path.removeprefix("/works/"))
        return (200, {"status": "ok", "message-type": "work", "message": crossref(w)}) if w else (404, None)
    if u.netloc == "api.ies.ed.gov":
        phrase = re.fullmatch(r'title:"(.*)"', q["search"]).group(1)
        docs = [{"id": w["eric_id"], "title": full_title(w), "publicationdateyear": w["year"], "description": w["abstract"],
                 "author": [f"{f}, {g}" for g, f in w["authors"]], "subject": ["Study Habits"]}
                for w in PAPERS["works"] if w.get("abstract_in") == "eric" and phrase in words(full_title(w))]
        return 200, {"response": {"numFound": len(docs), "start": 0, "numFoundExact": True, "docs": docs}}
    if u.netloc == "www.ebi.ac.uk":
        w = BY_DOI.get(re.fullmatch(r'DOI:"(.*)"', q["query"]).group(1))
        hits = [{"id": "0" + w["openalex_id"][1:], "doi": w["doi"], "title": full_title(w), "pubYear": str(w["year"]),
                 "abstractText": w["abstract"], "authorString": ", ".join(f for _, f in w["authors"]),
                 "affiliation": "Department of Psychology, Northvale University"}] \
            if w and w.get("abstract_in") == "europepmc" else []
        return 200, {"version": "6.9", "hitCount": len(hits), "resultList": {"result": hits}}
    if u.netloc == "doi.org":
        doi = path.removeprefix("/api/handles/")
        return (200, {"responseCode": 1, "handle": doi, "values": [{"index": 1, "type": "URL",
                                                                     "data": {"format": "string", "value": "https://example.org/"}}]}) \
            if doi in BY_DOI else (404, {"responseCode": 100, "handle": doi})
    raise AssertionError(f"make_fixtures has no fake for {url}")


class Response(io.BytesIO):
    def __init__(self, status, body):
        super().__init__(json.dumps(body).encode())
        self.status = status


def fake_urlopen(req, timeout=None):
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    status, body = respond(url)
    if status == 404:
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(json.dumps(body).encode()))
    return Response(status, body)


@contextlib.contextmanager
def fake_apis():
    saved = urllib.request.urlopen, research_apis.MIN_INTERVAL
    urllib.request.urlopen, research_apis.MIN_INTERVAL = fake_urlopen, 0
    try:
        yield
    finally:
        urllib.request.urlopen, research_apis.MIN_INTERVAL = saved


def build(out):
    """Write every fixture response to `out` (emptied first). Returns the number of files."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.json"):
        p.unlink()
    f = Fetcher(cache_dir=out, offline=False)
    with fake_apis():
        for w in PAPERS["works"]:
            if w["role"] != "search_only":       # what get_work and check_doi fetch
                full_work(f, w["doi"])
                crossref_record(f, w["doi"])
        crossref_record(f, PAPERS["unregistered_doi"])
        doi_resolves(f, PAPERS["unregistered_doi"])
        for s in PAPERS["searches"]:
            search(f, s["query"])
    files = sorted(out.glob("*.json"))
    for p in files:   # a fixed fetch time, so a rebuild is byte for byte the same
        rec = json.loads(p.read_text())
        rec["fetched_at"] = FETCHED_AT
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))   # the Fetcher's own format
    return len(files)


if __name__ == "__main__":
    print(f"{build(HERE / 'responses')} responses in {HERE / 'responses'}")
