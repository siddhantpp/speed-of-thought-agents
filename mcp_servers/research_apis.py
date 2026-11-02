"""The research tools' data layer: free, keyless scholarly APIs, polite and cached.

  OpenAlex    search + one work's metadata (venue, type, citations, retraction flag, abstract, references)
  Crossref    search + the DOI record: authors, print year, pages, correction and retraction notices
  ERIC        abstracts for education research that publishers withhold from OpenAlex/Crossref
  Europe PMC  abstracts for anything indexed by PubMed

Every response (including "not found") is saved under RESEARCH_CACHE_DIR, one JSON file per URL
(only the fields the code reads: no free text but titles and abstracts, author names but no
affiliations, ORCIDs or emails; see trim()),
so a run can be replayed with RESEARCH_OFFLINE=1 and the scorer can check citations against
exactly what was fetched. Requests carry only the query and a generic User-Agent: no API keys,
no email, nothing about the student.
"""
import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from itertools import zip_longest
from pathlib import Path

from research_credibility import credibility

OPENALEX, CROSSREF = "https://api.openalex.org", "https://api.crossref.org"
ERIC, EUROPEPMC = "https://api.ies.ed.gov/eric/", "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DOI_HANDLE = "https://doi.org/api/handles/"
HOSTS = {"openalex": "openalex", "crossref": "crossref", "ies.ed.gov": "eric", "ebi.ac.uk": "europepmc", "doi.org": "doi"}
USER_AGENT = "speed-of-thought-research-demo/1.0 (teaching demo)"
MIN_INTERVAL = 0.5          # seconds between two requests to the same host: well under every API's limit
POOL = 40                   # OpenAlex matches fetched per search (Crossref: half); the tool shows `limit`
# OpenAlex single-work lookups are free; a search costs credits, so ask only for the fields we read.
FIELDS = ("id,doi,title,publication_year,publication_date,type,cited_by_count,is_retracted,open_access,"
          "primary_location,locations,authorships,biblio,abstract_inverted_index")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def trim(body):
    """Keep only what the code reads. The only free text kept is titles and abstracts; authors and venues
    keep their names. Everything else is dropped: affiliations (also one a publisher deposited as an
    author), ORCIDs, emails, bios, disclosures, notes and reference lists, and a response of any other
    shape keeps no text at all. OpenAlex's word-position abstracts become plain text."""
    def src(loc):
        return {"source": {k: ((loc or {}).get("source") or {}).get(k) for k in ("type", "display_name", "issn_l")}}

    def work(w):   # an OpenAlex work
        return {"id": w.get("id"), "doi": w.get("doi"), "title": w.get("title"), "type": w.get("type"),
                "publication_year": w.get("publication_year"), "publication_date": w.get("publication_date"),
                "cited_by_count": w.get("cited_by_count"), "is_retracted": w.get("is_retracted"),
                "open_access": {"oa_status": (w.get("open_access") or {}).get("oa_status")},
                "primary_location": src(w.get("primary_location")), "locations": [src(x) for x in w.get("locations") or []],
                "authorships": [{"author": {"display_name": ((a.get("author") or {}).get("display_name")
                                                             or a.get("raw_author_name"))}} for a in w.get("authorships") or []],
                "biblio": {k: v for k, v in (w.get("biblio") or {}).items() if k in ("volume", "issue", "first_page", "last_page")},
                "referenced_works": w.get("referenced_works") or [],
                "abstract": w.get("abstract") or abstract_from_index(w.get("abstract_inverted_index"))}

    def item(m):   # a Crossref work
        keep = ("DOI", "title", "subtitle", "container-title", "ISSN", "volume", "issue", "page", "article-number", "type",
                "published-print", "published-online", "issued", "abstract", "is-referenced-by-count")
        people = [{k: a[k] for k in ("given", "family") if k in a} for a in m.get("author") or [] if a.get("family")]
        groups = [{"name": a["name"]} for a in m.get("author") or [] if a.get("name") and not a.get("family")]
        return {**{k: m[k] for k in keep if k in m},
                **{k: [{f: u[f] for f in ("type", "DOI", "updated") if f in u} for u in m[k]]   # notices, no labels
                   for k in ("updated-by", "update-to") if k in m},
                "author": people or groups}   # an organisation counts only when no person is listed (as authors_of)
    if not isinstance(body, dict):
        return body
    if "results" in body and "meta" in body:                       # OpenAlex search
        body = {"meta": {"count": body["meta"].get("count")}, "results": [work(w) for w in body["results"]]}
    elif str(body.get("id", "")).startswith("https://openalex.org/W"):   # one OpenAlex work
        body = work(body)
    elif isinstance(body.get("message"), dict) and "items" in body["message"]:   # Crossref search
        body = {"message": {"total-results": body["message"].get("total-results"),
                            "items": [item(m) for m in body["message"]["items"]]}}
    elif isinstance(body.get("message"), dict):                    # one Crossref work
        body = {"message": item(body["message"])}
    elif "resultList" in body:                                     # Europe PMC
        body = {"resultList": {"result": [{k: r.get(k) for k in ("id", "doi", "title", "pubYear", "abstractText")}
                                          for r in (body["resultList"].get("result") or [])[:1]]}}
    elif isinstance(body.get("response"), dict):                   # ERIC: its description is the abstract
        body = {"response": {"numFound": body["response"].get("numFound"), "docs": [
            {k: d.get(k) for k in ("id", "title", "publicationdateyear", "description")}
            for d in body["response"].get("docs") or []]}}
    elif "responseCode" in body:                                   # the DOI system's handle API
        body = {"responseCode": body.get("responseCode"), "handle": body.get("handle")}
    else:                                                          # any other shape: numbers and flags only
        body = {k: v for k, v in body.items() if v is None or isinstance(v, (bool, int, float))}
    # Any email left in free text (an abstract that prints one) is not ours to keep.
    return json.loads(EMAIL.sub("[email removed]", json.dumps(body, ensure_ascii=False)))


class Offline(Exception):
    """RESEARCH_OFFLINE=1 and the response is not in the cache."""


def clean(text):
    """Abstracts arrive as JATS XML (Crossref), HTML (Europe PMC) or with entities (ERIC): make them plain."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def bare_doi(text):
    m = re.search(r"10\.\d{4,9}/\S+", str(text or ""))
    return m.group(0).rstrip(".,;)").lower() if m else None


def words(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


class Fetcher:
    """GET JSON with a disk cache. Reads cache_dir, then seed_dir (read-only fixtures); writes cache_dir."""

    def __init__(self, cache_dir=None, offline=None, seed_dir=None):
        self.cache_dir = Path(cache_dir or os.environ.get("RESEARCH_CACHE_DIR") or "cache").resolve()
        seed = seed_dir or os.environ.get("RESEARCH_SEED_DIR")
        self.seed_dir = Path(seed).resolve() if seed else None
        self.offline = (os.environ.get("RESEARCH_OFFLINE", "") not in ("", "0")) if offline is None else offline
        self._lock, self._last = threading.Lock(), {}

    def _path(self, directory, url):
        netloc = urllib.parse.urlparse(url).netloc
        host = next((n for k, n in HOSTS.items() if k in netloc), netloc.replace(".", "_"))
        return directory / f"{host}-{hashlib.sha1(url.encode()).hexdigest()[:16]}.json"

    def cached(self, url):
        for d in filter(None, (self.cache_dir, self.seed_dir)):
            p = self._path(d, url)
            if p.exists():
                rec = json.loads(p.read_text())
                if d == self.seed_dir:   # copy fixtures into this run's cache: the cache IS the fetched set
                    self._save(url, rec)
                return rec
        return None

    def _save(self, url, rec):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(self.cache_dir, url).write_text(json.dumps(rec, indent=1, ensure_ascii=False))

    def get(self, url, kind, **meta):
        """Returns the cache record {"url", "kind", "status", "body", ...}; body is None for a 404."""
        rec = self.cached(url)
        if rec is not None:
            return rec
        if self.offline:
            raise Offline(url)
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(3):
            with self._lock:   # politeness: one request at a time, spaced per host
                wait = self._last.get(host, 0) + MIN_INTERVAL - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last[host] = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    status, body = r.status, json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    status, body = 404, None
                    break
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"{host} answered HTTP {e.code}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"could not reach {host}: {e}") from None
        rec = {"url": url, "kind": kind, **meta, "status": status,
               "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "body": trim(body)}
        self._save(url, rec)
        return rec

    def empty(self):
        """True when there is nothing to replay: no response in the cache or the seed folder (if any)."""
        return not any(d.is_dir() and next(d.glob("*.json"), None) for d in filter(None, (self.cache_dir, self.seed_dir)))

    def records(self, kind, prefix=""):
        """Every cached record of one kind (used for offline fallbacks and by the scorer)."""
        out = []
        for d in filter(None, (self.cache_dir, self.seed_dir)):
            if d.is_dir():
                for p in sorted(d.glob(f"{prefix}*.json")):
                    try:
                        rec = json.loads(p.read_text())
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") == kind:
                        out.append(rec)
        return out


# ---------------------------------------------------------------- OpenAlex
def abstract_from_index(inv):
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inv:
        return ""
    pos = {p: w for w, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def journal_location(w):
    """The publisher's version if there is one (some works list a repository copy first)."""
    locs = [w.get("primary_location") or {}] + (w.get("locations") or [])
    for loc in locs:
        if ((loc or {}).get("source") or {}).get("type") == "journal":
            return loc
    return locs[0] or {}


def from_openalex(w):
    src = journal_location(w).get("source") or {}
    b = w.get("biblio") or {}
    first, last = b.get("first_page"), b.get("last_page")
    pages = first if (first and last in (None, first)) else f"{first}–{last}" if first else None
    authors = []
    for a in w.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name") or a.get("raw_author_name") or ""
        parts = name.split()
        if parts:
            authors.append({"family": name_case(parts[-1]), "given": name_case(" ".join(parts[:-1]))})
    return {"id": (w.get("id") or "").rsplit("/", 1)[-1], "doi": bare_doi(w.get("doi")),
            "title": clean(w.get("title")), "authors": authors, "year": w.get("publication_year"),
            "venue": src.get("display_name"), "venue_type": src.get("type"), "issn": src.get("issn_l"),
            "volume": b.get("volume"), "issue": b.get("issue"), "pages": pages, "article_number": None,
            "type": w.get("type"), "crossref_type": None, "cited_by_count": w.get("cited_by_count") or 0,
            "is_retracted": bool(w.get("is_retracted")), "updates": [],
            "open_access": (w.get("open_access") or {}).get("oa_status"),
            "abstract": clean(w.get("abstract") or abstract_from_index(w.get("abstract_inverted_index"))),
            "abstract_source": "openalex",
            "references": [r.rsplit("/", 1)[-1] for r in w.get("referenced_works") or []]}


def search(f, query, from_year=2000, limit=10):
    """What search_works returns. Two free-to-call indexes, interleaved:
      OpenAlex  works whose title or abstract contains EVERY query word (one search = 10 credits of the
                keyless daily budget; it fetches the POOL best matches, which costs the same as 10)
      Crossref  best bibliographic matches: titles, authors and venues (free), so author names work too
    Duplicates (same DOI) are merged, then peer-reviewed works are listed first (see shown())."""
    # OpenAlex splits 'note-taking' into two loose words; as a phrase it means what the person meant.
    phrased = re.sub(r"\b(\w+)-(\w+)\b", r'"\1 \2"', query.replace('"', ""))
    urls = {"openalex": f"{OPENALEX}/works?" + urllib.parse.urlencode({
                "search.title_and_abstract": phrased, "filter": f"from_publication_date:{int(from_year)}-01-01",
                "per_page": POOL, "select": FIELDS}),
            "crossref": f"{CROSSREF}/works?" + urllib.parse.urlencode({
                "query.bibliographic": query, "rows": POOL // 2, "filter": f"from-pub-date:{int(from_year)}"})}
    recs, notes, replayed = [], [], None
    for backend, url in urls.items():
        try:
            recs.append(f.get(url, "search", query=query, backend=backend))
        except Offline:
            pass
        except RuntimeError as e:              # e.g. the keyless OpenAlex budget is used up: Crossref still works
            notes.append(f"{backend} unavailable ({e})")
    if not recs and f.offline:                 # offline and a new query: replay the closest cached one
        replayed = nearest_query(f, query)
        if replayed is None:
            raise Offline(query)
        recs = [f.cached(r["url"]) for r in f.records("search") if r.get("query") == replayed]   # copies fixtures in
    if not recs:
        raise RuntimeError("; ".join(notes) or "no search backend answered")
    found = {r.get("backend", "openalex"): ((r.get("body") or {}).get("meta") or {}).get("count")
             or (((r.get("body") or {}).get("message") or {}).get("total-results")) for r in recs}
    return {"works": merged(recs, limit), "total_found": found, "replayed_from": replayed, "notes": notes}


def merged(recs, limit=10):
    """Interleave the OpenAlex and Crossref matches (best first from each). A DOI found by both appears
    once, as the OpenAlex record (it has the abstract and the OpenAlex id)."""
    lists = [[w for r in recs if r.get("backend", "openalex") == b for w in works_in(r)] for b in ("openalex", "crossref")]
    by_doi = {w["doi"]: w for w in lists[0] if w.get("doi")}
    out, seen = [], set()
    for w in (w for pair in zip_longest(*lists) for w in pair if w):
        w = by_doi.get(w.get("doi"), w)
        key = (w.get("doi") or w.get("id") or w.get("title") or "").lower()
        if key not in seen:
            seen.add(key)
            out.append(w)
    return shown(out, limit)


def shown(works, limit=10):
    """Which matches a search shows is a rule too: peer-reviewed works first, each group in match order,
    so preprints and theses don't crowd out journal articles (code filters them later anyway)."""
    return sorted(works, key=lambda w: not credibility(w)["peer_reviewed"])[:max(1, min(int(limit), 25))]


def nearest_query(f, query):
    """Offline fallback: the cached search query that shares the most words with this one."""
    q = words(query)
    best = max(((len(q & words(r.get("query", ""))) / len(q | words(r.get("query", ""))), r.get("query"))
                for r in f.records("search") if r.get("body")), default=(0, None))
    return best[1] if best[0] >= 0.25 else None


def openalex_work(f, work_id):
    """One work by OpenAlex id (W...) or DOI. Single-work lookups are free on OpenAlex."""
    doi = bare_doi(work_id)
    wid = None if doi else re.search(r"W\d+", str(work_id), flags=re.I)
    if not doi and not wid:
        return None
    key = f"doi:{doi}" if doi else wid.group(0).upper()
    alias = cached_alias(f, key)      # a work fetched by id is also found by its DOI, and vice versa
    rec = alias or f.get(f"{OPENALEX}/works/{urllib.parse.quote(key, safe=':/')}?select={FIELDS},referenced_works",
                         "openalex_work", key=key)
    return from_openalex(rec["body"]) if rec.get("body") else None


def cached_alias(f, key):
    key = key.lower()
    for rec in f.records("openalex_work", prefix="openalex-"):
        b = rec.get("body") or {}
        if b and key in {f"doi:{bare_doi(b.get('doi'))}", (b.get("id") or "").rsplit("/", 1)[-1].lower()}:
            return f.cached(rec["url"])     # through the cache, so a fixture hit is copied into the run's cache
    return None


# ---------------------------------------------------------------- Crossref
def crossref_record(f, doi):
    rec = f.get(f"{CROSSREF}/works/{urllib.parse.quote(doi, safe='/')}", "crossref_work", doi=doi)
    return (rec.get("body") or {}).get("message")


def year_of(m):
    """APA uses the year of the issue; fall back to the online date, then the earliest date."""
    for k in ("published-print", "published-online", "issued"):
        parts = (m.get(k) or {}).get("date-parts") or [[None]]
        if parts[0] and parts[0][0]:
            return parts[0][0]
    return None


def notices(m):
    """Correction / retraction / expression-of-concern notices that point at this DOI."""
    def day(u):
        parts = ((u.get("updated") or {}).get("date-parts") or [[]])[0]
        return "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(parts))
    return [{"type": u.get("type"), "doi": bare_doi(u.get("DOI")), "date": day(u)} for u in m.get("updated-by") or []]


def name_case(name):
    """Some records shout ('KOBAYASHI'); names already in mixed case are left exactly as published."""
    return name.title() if name.isupper() and len(name) > 2 else name


def authors_of(m):
    """People first. Some publishers deposit an affiliation ('Drake University') as an author; an organisation
    counts as the author only when no person is listed (APA group author)."""
    people = [{"family": name_case(a["family"]), "given": name_case(a.get("given", ""))}
              for a in m.get("author") or [] if a.get("family")]
    return people or [{"family": a["name"], "given": ""} for a in m.get("author") or [] if a.get("name")]


def from_crossref(m):
    title = clean(" ".join(m.get("title") or []))
    sub = clean(" ".join(m.get("subtitle") or []))
    page, art = m.get("page"), m.get("article-number")
    return {"doi": bare_doi(m.get("DOI")), "title": f"{title.rstrip('.')}: {sub}" if sub else title.rstrip("."),
            "authors": authors_of(m),
            "year": year_of(m), "venue": clean(" ".join(m.get("container-title") or [])) or None,
            "issn": (m.get("ISSN") or [None])[0], "volume": m.get("volume"), "issue": m.get("issue"),
            "pages": None if (not page or page == art) else page.replace("-", "–"), "article_number": art,
            "crossref_type": m.get("type"), "updates": notices(m), "abstract": clean(m.get("abstract")),
            "is_update_notice": bool(m.get("update-to"))}


def from_crossref_hit(m):
    w = from_crossref(m)
    typ = {"journal-article": "article", "posted-content": "preprint", "dissertation": "dissertation",
           "book-chapter": "book-chapter", "proceedings-article": "article"}.get(m.get("type"), m.get("type"))
    return {**w, "id": w["doi"], "type": typ, "cited_by_count": m.get("is-referenced-by-count") or 0,
            "venue_type": "journal" if m.get("type") == "journal-article" else m.get("type"),
            "is_retracted": any(u["type"] in ("retraction", "withdrawal") for u in w["updates"]),
            "abstract_source": "crossref"}


def works_in(rec):
    """The works in one cached search response (OpenAlex, or the Crossref backup)."""
    body = rec.get("body") or {}
    if "results" in body:
        return [from_openalex(w) for w in body["results"]]
    return [from_crossref_hit(m) for m in (body.get("message") or {}).get("items", [])]


def as_hit(w):
    """A search hit as the tools show it: compact metadata plus the code-computed credibility."""
    c = credibility(w)
    words_ = (w.get("abstract") or "").split()
    return {"id": w["id"], "doi": w["doi"], "title": w["title"],
            "first_author": (w["authors"][0]["family"] if w.get("authors") else None), "year": w.get("year"),
            "venue": w.get("venue"), "type": w.get("type"), "cited_by_count": w.get("cited_by_count"),
            "is_retracted": c["retracted"], "peer_reviewed": c["peer_reviewed"], "credibility": c["score"],
            "snippet": " ".join(words_[:45]) + (" ..." if len(words_) > 45 else "") if words_ else "(no abstract available)"}


def searched_hits(f, limit=10):
    """What search_works has shown, rebuilt from the cache (the fetched set): one list of hits per query."""
    by_query = {}
    for rec in f.records("search"):
        by_query.setdefault(rec.get("query"), []).append(rec)
    return [[as_hit(w) for w in merged(recs, limit)] for recs in by_query.values()]


def doi_resolves(f, doi):
    """The DOI system's own handle API: true for any registered DOI (Crossref, DataCite, ...)."""
    rec = f.get(DOI_HANDLE + urllib.parse.quote(doi, safe="/"), "doi_handle", doi=doi)
    return (rec.get("body") or {}).get("responseCode") == 1


# ---------------------------------------------------------------- abstracts the big two lack
def eric_abstract(f, title, year):
    """ERIC (U.S. Institute of Education Sciences) by exact title, accepted only if title and year match."""
    key = " ".join(re.findall(r"[a-z0-9]+", title.lower().split(":")[0])[:12])
    if not key:
        return ""
    url = ERIC + "?" + urllib.parse.urlencode({"search": f'title:"{key}"', "format": "json", "rows": 5,
                                                 "fields": "id,title,publicationdateyear,description"})
    rec = f.get(url, "eric_search", query=key)
    for d in ((rec.get("body") or {}).get("response") or {}).get("docs", []):
        same = " ".join(re.findall(r"[a-z0-9]+", d.get("title", "").lower())).startswith(key)
        if same and year and abs((d.get("publicationdateyear") or 0) - year) <= 1 and d.get("description"):
            return clean(d["description"])
    return ""


def europepmc_abstract(f, doi):
    url = EUROPEPMC + "?" + urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json",
                                                     "resultType": "core", "pageSize": 1})
    rec = f.get(url, "europepmc_search", doi=doi)
    hits = (((rec.get("body") or {}).get("resultList") or {}).get("result")) or []
    return clean(hits[0].get("abstractText")) if hits else ""


def full_work(f, work_id):
    """OpenAlex + Crossref merged, with the first abstract any source has. None if nobody knows it.
    A source that can't be reached (or isn't cached, offline) is skipped and listed in "gaps"."""
    gaps = []

    def attempt(name, fn, default=None):
        try:
            return fn()
        except (RuntimeError, Offline):
            gaps.append(name)
            return default
    w = attempt("openalex", lambda: openalex_work(f, work_id))
    doi = (w or {}).get("doi") or bare_doi(work_id)
    m = attempt("crossref", lambda: crossref_record(f, doi)) if doi else None
    if not w and not m:
        if f.offline:
            raise Offline(work_id)
        return None
    if m:
        cr = from_crossref(m)
        w = {**(w or {"id": doi, "type": "article" if cr["crossref_type"] == "journal-article" else cr["crossref_type"],
                      "venue_type": "journal" if cr["crossref_type"] == "journal-article" else None,
                      "cited_by_count": m.get("is-referenced-by-count") or 0, "is_retracted": False,
                      "open_access": None, "abstract": "", "abstract_source": None}),
             **{k: v for k, v in cr.items() if v not in (None, "", []) and k != "abstract"}}
        if cr["article_number"] and not cr["pages"]:     # e-journals number articles instead of pages
            w["pages"] = None
        if not w.get("abstract") and cr["abstract"]:
            w["abstract"], w["abstract_source"] = cr["abstract"], "crossref"
        w["is_retracted"] = w["is_retracted"] or any(u["type"] in ("retraction", "withdrawal", "removal")
                                                     for u in cr["updates"])
    if not w.get("abstract") and w.get("title"):
        for source, get in (("eric", lambda: eric_abstract(f, w["title"], w.get("year"))),
                            ("europepmc", lambda: europepmc_abstract(f, doi) if doi else "")):
            text = attempt(source, get, "")
            if text:
                w["abstract"], w["abstract_source"] = text, source
                break
    if not w.get("abstract"):
        w["abstract_source"] = None
    w["gaps"] = gaps
    return w
