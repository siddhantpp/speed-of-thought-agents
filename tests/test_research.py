"""Tests for the research workflow's deterministic parts: credibility rule, retractions, APA references, quote
and citation checks, the MCP server's offline replay, and the scorer's fabricated-reference check.

Offline: everything replays tests/fixtures/research/responses, the recorded answers of a fake OpenAlex,
Crossref, ERIC, Europe PMC and doi.org about fictional papers (fixtures/research/papers.json; DOIs under the
reserved 10.5555 prefix). No network, no model. AnswerKeyTest also checks the real answer key; its checks
against real abstracts need the recorded API responses in sample_data/research/cache, which make_cache.py
builds and the repo does not ship, so they are skipped without them.

    .venv/bin/python -m unittest            # discovers this file
    .venv/bin/python tests/test_research.py # or run it directly
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works in the exported repo (tests/ next to agentkit/) and in the source tree (tools/repo_extras/tests).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "agentkit").is_dir())
FIXTURE_DIR = HERE / "fixtures" / "research"
sys.path[:0] = [str(ROOT), str(ROOT / "mcp_servers"), str(FIXTURE_DIR)]
from research_apis import Fetcher, authors_of, full_work, merged, searched_hits, shown, trim  # noqa: E402
from research_credibility import credibility  # noqa: E402
from research_references import apa, citation_numbers, initials, is_verbatim, quoted_spans, renumber  # noqa: E402

FIXTURES = FIXTURE_DIR / "responses"
PAPERS = json.loads((FIXTURE_DIR / "papers.json").read_text())
WORKS = {w["key"]: w for w in PAPERS["works"]}
SOURCES = {k: w for k, w in WORKS.items() if w["role"] == "source"}
FAKE_DOI = PAPERS["unregistered_doi"]
DATA = ROOT / "sample_data" / "research"
KEY = json.loads((DATA / "answer_key.json").read_text())
RECORDED = DATA / "cache"   # the real API responses: in the source project only (make_cache.py builds them)


class Replay(unittest.TestCase):
    """Each test gets an empty cache dir that replays the fixtures; the network is switched off."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fetch = Fetcher(cache_dir=self.tmp / "cache", seed_dir=FIXTURES, offline=True)
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: self.fail("offline mode tried to use the network")

    def tearDown(self):
        urllib.request.urlopen = self._urlopen
        shutil.rmtree(self.tmp, ignore_errors=True)

    def work(self, key):
        return full_work(self.fetch, WORKS[key]["doi"])


class CredibilityTest(Replay):
    def test_peer_reviewed_article_scores_high_by_the_written_rule(self):
        c = credibility(self.work("brennholt2014"), as_of=2026)
        # 45 (journal article) + 15 (venue with ISSN) + 25 (1,000+ citations, capped) + 8 (12 years old)
        self.assertEqual(c["score"], 93.0)
        self.assertEqual(c["tier"], "high")
        self.assertTrue(c["peer_reviewed"] and c["vetted"])
        self.assertTrue(any("correction" in r for r in c["reasons"]), "the 2018 correction should be reported")

    def test_every_source_is_vetted(self):
        for key in SOURCES:
            with self.subTest(key):
                self.assertTrue(credibility(self.work(key), as_of=2026)["vetted"])

    def test_preprint_thesis_and_dataset_are_not_vetted(self):
        base = {"venue": "Northvale Preprints", "venue_type": "repository", "issn": None, "cited_by_count": 3, "year": 2017}
        for typ in ("preprint", "dissertation", "dataset"):
            with self.subTest(typ):
                c = credibility({**base, "type": typ}, as_of=2026)
                self.assertFalse(c["peer_reviewed"] or c["vetted"])
                self.assertLess(c["score"], 50)

    def test_a_retracted_paper_scores_zero_whatever_else_is_true(self):
        w = self.work("drummond2016")   # a journal article with an ISSN and 95 citations, then retracted
        self.assertTrue(w["is_retracted"])
        self.assertEqual([u["type"] for u in w["updates"]], ["retraction"])
        c = credibility(w, as_of=2026)
        self.assertEqual(c["score"], 0)
        self.assertTrue(c["retracted"] and c["peer_reviewed"])
        self.assertFalse(c["vetted"])
        self.assertTrue(c["reasons"][0].startswith("RETRACTED"))

    def test_a_crossref_retraction_notice_alone_is_enough(self):
        w = {**self.work("saarinen2013"), "updates": [{"type": "retraction", "doi": "10.5555/x", "date": "2025-01-01"}]}
        c = credibility(w, as_of=2026)
        self.assertEqual(c["score"], 0)
        self.assertFalse(c["vetted"])
        self.assertTrue(c["retracted"])

    def test_a_corrigendum_is_a_notice_not_a_study(self):
        w = self.work("brennholt2014_correction")
        self.assertEqual(w["type"], "erratum")
        self.assertTrue(w["is_update_notice"])
        self.assertFalse(credibility(w, as_of=2026)["vetted"])

    def test_expression_of_concern_costs_30_points(self):
        w = self.work("saarinen2013")
        plain = credibility(w, as_of=2026)["score"]
        flagged = credibility({**w, "updates": [{"type": "expression_of_concern"}]}, as_of=2026)["score"]
        self.assertAlmostEqual(plain - flagged, 30, places=1)


class ReferenceTest(Replay):
    def test_two_authors_subtitle_issue_and_pages(self):
        self.assertEqual(apa(self.work("brennholt2014")),
                         "Brennholt, I. M., & Vanstone, T. R. (2014). Quiet rooms help: Advantages of silent study over "
                         "background music for recall. *Northvale Journal of Learning Science*, *12*(4), 301–317. "
                         "https://doi.org/10.5555/nvjls.2014.0412")

    def test_title_case_becomes_sentence_case_but_names_stay(self):
        self.assertEqual(apa(self.work("marwood2019")),
                         "Marwood, K., Ashdown, J., & Blevins, R. T. (2019). How much does silence help? A replication and "
                         "extension of Brennholt and Vanstone (2014). *Northvale Review of Study Strategies*, *8*(2), "
                         "145–171. https://doi.org/10.5555/nvrss.2019.0208")

    def test_article_number_instead_of_pages(self):
        ref = apa(self.work("abaroa2024"))
        self.assertIn("*Northvale Review of Study Strategies*, *13*(1), Article 42.", ref)
        self.assertIn("college students: A meta-analysis.", ref)

    def test_more_than_20_authors_uses_the_apa_ellipsis(self):
        ref = apa(self.work("ashworthpell2021"))
        self.assertTrue(ref.startswith("Ashworth-Pell, H. L., Crane, C. S., "))
        self.assertIn(", . . . Kaldirim, Y. (2021).", ref)
        self.assertEqual(ref.split(" (2021)")[0].count(".,"), 19, "first 19 authors, then the last one")

    def test_print_year_wins_over_online_year(self):
        # Saarinen et al. went online in 2012 but appeared in the 2013 issue; APA cites 2013.
        w = self.work("saarinen2013")
        self.assertIn("(2013).", apa(w))
        self.assertEqual(full_work(self.fetch, WORKS["saarinen2013"]["openalex_id"])["year"], 2013)

    def test_initials(self):
        self.assertEqual(initials("Pam A."), "P. A.")
        self.assertEqual(initials("Jean-Paul"), "J.-P.")


class AbstractSourceTest(Replay):
    def test_each_open_source_of_abstracts_is_used_in_turn(self):
        for key in SOURCES:
            with self.subTest(key):
                w = self.work(key)
                self.assertEqual(w["abstract_source"], WORKS[key]["abstract_in"])
                self.assertGreaterEqual(len(w["abstract"].split()), 40, "the workflow excludes shorter abstracts")

    def test_html_in_an_abstract_becomes_plain_text(self):
        abstract = self.work("abaroa2024")["abstract"]
        self.assertIn("(Hedges' g = 0.47; p < 0.001)", abstract)
        self.assertNotIn("<", abstract.replace("p < 0.001", ""))

    def test_a_notice_has_no_abstract_anywhere(self):
        w = self.work("brennholt2014_correction")
        self.assertEqual((w["abstract"], w["abstract_source"], w["gaps"]), ("", None, []))


class QuoteAndCitationTest(Replay):
    def test_source_quotes_are_verbatim_in_the_fetched_abstracts(self):
        for key, s in SOURCES.items():
            with self.subTest(key):
                self.assertTrue(is_verbatim(s["quote"], self.work(key)["abstract"], min_words=6))

    def test_paraphrase_and_spliced_quotes_fail(self):
        abstract = self.work("ashworthpell2021")["abstract"]
        self.assertTrue(is_verbatim("MUSIC participants did not perform better or worse than silence participants", abstract))
        self.assertFalse(is_verbatim("music participants performed worse than silence participants", abstract))
        self.assertFalse(is_verbatim("music participants did not perform ... by 50 percent", abstract))
        self.assertTrue(is_verbatim("As in the original study ... lower focus ratings overall than did silence participants",
                                    abstract))

    def test_curly_apostrophes_match_straight_ones(self):
        self.assertTrue(is_verbatim("Brennholt and Vanstone's (2014) Study 1", self.work("ashworthpell2021")["abstract"]))

    def test_citations_parse_and_renumber(self):
        text = "First [3]. Then [1, 3] and a range [4–6]."
        self.assertEqual(citation_numbers(text), [3, 1, 3, 4, 5, 6])
        self.assertEqual(renumber(text, {3: 1, 1: 2, 4: 3, 5: 4, 6: 5}), "First [1]. Then [1, 2] and a range [3, 4, 5].")

    def test_only_real_quotations_are_checked(self):
        self.assertEqual(quoted_spans('a "scare quote" and "a much longer quoted span here" end'),
                         ["a much longer quoted span here"])


class SearchRuleTest(unittest.TestCase):
    def work(self, i, typ="article", doi=None):
        return {"id": f"W{i}", "doi": doi, "title": f"t{i}", "type": typ, "venue": "J", "venue_type": "journal",
                "issn": "1234-5678", "year": 2020, "cited_by_count": 10, "authors": [], "abstract": ""}

    def test_peer_reviewed_hits_come_first_in_match_order(self):
        works = [self.work(1, "preprint"), self.work(2), self.work(3, "dissertation"), self.work(4)]
        self.assertEqual([w["id"] for w in shown(works)], ["W2", "W4", "W1", "W3"])

    def test_openalex_and_crossref_interleave_and_merge_by_doi(self):
        def oa(i, doi):   # minimal OpenAlex work and Crossref item, as the APIs return them
            return {"id": f"https://openalex.org/W{i}", "doi": f"https://doi.org/{doi}", "title": f"t{i}", "type": "article",
                    "primary_location": {"source": {"type": "journal", "display_name": "J", "issn_l": "1234-5678"}}}

        def cr(doi):
            return {"DOI": doi, "title": [doi], "type": "journal-article", "container-title": ["J"], "ISSN": ["1234-5678"]}
        recs = [{"backend": "openalex", "body": {"results": [oa(1, "10.5555/a"), oa(2, "10.5555/b")]}},
                {"backend": "crossref", "body": {"message": {"items": [cr("10.5555/b"), cr("10.5555/c")]}}}]
        # OpenAlex 1st, Crossref 1st (same DOI as OpenAlex 2nd, so the OpenAlex record), Crossref 2nd
        self.assertEqual([w["id"] for w in merged(recs)], ["W1", "W2", "10.5555/c"])

    def test_an_affiliation_listed_as_author_is_dropped(self):
        m = {"author": [{"name": "Northvale University"}, {"family": "Okafor-Lind", "given": "Tamsin"}]}
        self.assertEqual(authors_of(m), [{"family": "Okafor-Lind", "given": "Tamsin"}])
        self.assertEqual(authors_of({"author": [{"name": "Northvale Learning Council"}]}),
                         [{"family": "Northvale Learning Council", "given": ""}])


class CacheHygieneTest(unittest.TestCase):
    def test_cached_records_keep_names_but_no_affiliations_orcids_or_emails(self):
        raw = {"id": "https://openalex.org/W1", "title": "t", "abstract_inverted_index": {"Write": [0], "to": [1],
               "a@b.example.org": [2]}, "authorships": [{"author": {"display_name": "Ada Quillfeather",
               "orcid": "https://orcid.org/0000"}, "raw_affiliation_strings": ["Somewhere, ada@uni.example.org"]}]}
        kept = json.dumps(trim(raw))
        self.assertIn("Ada Quillfeather", kept)
        self.assertIn("Write to [email removed]", kept)
        for gone in ("orcid", "raw_affiliation", "@"):
            self.assertNotIn(gone, kept)

    def test_crossref_keeps_no_free_text_but_title_and_abstract(self):
        raw = {"message": {"DOI": "10.5555/x", "title": ["A title"], "abstract": "<jats:p>An abstract.</jats:p>",
                           "author": [{"given": "Tamsin", "family": "Okafor-Lind", "affiliation": [{"name": "Dept, Northvale"}],
                                       "ORCID": "https://orcid.org/0000"}, {"name": "Northvale University, Dept of X"}],
                           "assertion": [{"label": "Conflict of interest", "value": "Speaker fees from a company."}],
                           "reference": [{"unstructured": "Someone (2001)."}], "publisher": "Northvale Press",
                           "updated-by": [{"type": "correction", "DOI": "10.5555/y", "label": "Correction", "source": "x",
                                           "updated": {"date-parts": [[2020, 1, 2]]}}]}}
        m = trim(raw)["message"]
        self.assertEqual(m["author"], [{"given": "Tamsin", "family": "Okafor-Lind"}])
        self.assertEqual(m["updated-by"], [{"type": "correction", "DOI": "10.5555/y", "updated": {"date-parts": [[2020, 1, 2]]}}])
        self.assertEqual(set(m), {"DOI", "title", "abstract", "author", "updated-by"})
        self.assertEqual(trim({"message": {"author": [{"name": "A Group Author"}]}})["message"]["author"],
                         [{"name": "A Group Author"}], "an organisation stays when no person is listed")

    def test_eric_handle_and_unknown_answers_keep_only_what_is_read(self):
        eric = {"responseHeader": {"params": {"q": "x"}}, "response": {"numFound": 1, "docs": [
            {"id": "EJ1", "title": "T", "publicationdateyear": 2019, "description": "Abstract.", "author": ["Someone"]}]}}
        self.assertEqual(trim(eric), {"response": {"numFound": 1, "docs": [
            {"id": "EJ1", "title": "T", "publicationdateyear": 2019, "description": "Abstract."}]}})
        handle = {"responseCode": 1, "handle": "10.5555/x", "values": [{"type": "HS_ADMIN", "data": {"value": "x"}}]}
        self.assertEqual(trim(handle), {"responseCode": 1, "handle": "10.5555/x"})
        self.assertEqual(trim({"count": 3, "note": "free text", "ok": True}), {"count": 3, "ok": True})

    def test_fixtures_are_fictional_and_hold_no_emails_or_orcids(self):
        files = sorted(FIXTURES.glob("*.json"))
        self.assertTrue(files)
        for p in files:
            with self.subTest(p.name):
                text = p.read_text()
                self.assertIsNone(re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+|orcid|affiliation|"abstract_inverted_index"', text, re.I))
                self.assertEqual({d.lower() for d in re.findall(r"10\.\d{4,9}/", text)} - {"10.5555/"}, set())
                self.assertEqual(set(re.findall(r"openalex\.org/(W\d+)", text)) - {w.get("openalex_id") for w in WORKS.values()},
                                 set())

    def test_fixtures_match_make_fixtures(self):
        import make_fixtures
        tmp = Path(tempfile.mkdtemp())
        try:
            make_fixtures.build(tmp)
            built = {p.name: p.read_text() for p in tmp.glob("*.json")}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(sorted(built), sorted(p.name for p in FIXTURES.glob("*.json")),
                         "rebuild with tests/fixtures/research/make_fixtures.py")
        for name, text in built.items():
            self.assertEqual(text, (FIXTURES / name).read_text(), f"{name} is stale: rebuild with make_fixtures.py")


class AnswerKeyTest(unittest.TestCase):
    """The real answer key (sample_data/research/answer_key.json) stays short on quotes and true to its sources."""

    def test_quotes_are_short(self):
        for s in KEY["sources"]:
            if s.get("quote"):
                with self.subTest(s["key"]):
                    self.assertLessEqual(len(s["quote"].split()), 12)

    @unittest.skipUnless(RECORDED.is_dir(), "the recorded API responses are not in the repo (make_cache.py builds them)")
    def test_quotes_are_verbatim_and_sources_vetted_in_the_recorded_responses(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            f = Fetcher(cache_dir=tmp, seed_dir=RECORDED, offline=True)
            for s in KEY["sources"]:
                with self.subTest(s["key"]):
                    w = full_work(f, s["doi"])
                    self.assertTrue(credibility(w, as_of=2026)["vetted"])
                    if s.get("quote"):
                        self.assertTrue(is_verbatim(s["quote"], w["abstract"], min_words=6))
            notice = full_work(f, next(r["doi"] for r in KEY["red_flags"] if r["id"] == "corrigendum_as_study"))
            self.assertFalse(credibility(notice, as_of=2026)["vetted"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ServerReplayTest(unittest.TestCase):
    """Start the real MCP server as a subprocess, offline, and call its tools like a model would."""

    @classmethod
    def setUpClass(cls):
        from agentkit.mcp import MCPServer
        cls.tmp = Path(tempfile.mkdtemp())
        env = {**os.environ, "RESEARCH_OFFLINE": "1", "RESEARCH_SEED_DIR": str(FIXTURES),
               "RESEARCH_CACHE_DIR": str(cls.tmp / "cache")}
        cls.server = MCPServer([sys.executable, str(ROOT / "mcp_servers" / "research_server.py")], env=env)

    @classmethod
    def tearDownClass(cls):
        cls.server.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_lists_the_three_tools(self):
        self.assertEqual({t["name"] for t in self.server.tools}, {"search_works", "get_work", "check_doi"})

    def test_get_work_by_doi_and_by_openalex_id(self):
        s = SOURCES["marwood2019"]
        by_doi = self.server.call("get_work", {"work_id": s["doi"]})
        by_id = self.server.call("get_work", {"work_id": s["openalex_id"]})
        self.assertEqual(by_doi["doi"], by_id["doi"])
        self.assertEqual(by_doi["abstract_source"], "eric")
        self.assertTrue(by_doi["credibility"]["vetted"])
        self.assertIn("https://doi.org/10.5555/nvrss.2019.0208", by_doi["apa"])

    def test_check_doi_reports_corrections_retractions_and_unregistered_dois(self):
        real = self.server.call("check_doi", {"doi": "https://doi.org/" + SOURCES["brennholt2014"]["doi"]})
        self.assertTrue(real["exists"])
        self.assertFalse(real["retracted"])
        self.assertEqual([u["type"] for u in real["updates"]], ["correction"])
        retracted = self.server.call("check_doi", {"doi": WORKS["drummond2016"]["doi"]})
        self.assertTrue(retracted["exists"] and retracted["retracted"])
        notice = self.server.call("check_doi", {"doi": WORKS["brennholt2014_correction"]["doi"]})
        self.assertTrue(notice["is_update_notice"])
        fake = self.server.call("check_doi", {"doi": FAKE_DOI})
        self.assertFalse(fake["exists"])

    def test_search_replays_offline_and_a_new_query_gets_the_nearest_cached_one(self):
        queries = [r["query"] for r in Fetcher(cache_dir=FIXTURES, offline=True).records("search")]
        self.assertEqual(set(queries), {s["query"] for s in PAPERS["searches"]})
        query = PAPERS["searches"][0]["query"]
        hits = self.server.call("search_works", {"query": query})["results"]
        self.assertTrue(hits)
        reviewed = [h["peer_reviewed"] for h in hits]
        self.assertEqual(reviewed, sorted(reviewed, reverse=True), "peer-reviewed hits first")
        self.assertIn("preprint", {h["type"] for h in hits})
        retracted = next(h for h in hits if h["doi"] == WORKS["drummond2016"]["doi"])
        self.assertTrue(retracted["is_retracted"])
        self.assertEqual(retracted["credibility"], 0)
        near = self.server.call("search_works", {"query": query + " evidence"})
        self.assertIn("offline replay", near.get("note", ""))
        # what was replayed is now in the run's own cache, so code can check ids against it
        replayed = {h["id"] for hs in searched_hits(Fetcher(cache_dir=self.tmp / "cache", offline=True)) for h in hs}
        self.assertTrue({h["id"] for h in near["results"]} <= replayed)

    def test_uncached_requests_fail_cleanly_offline(self):
        res = self.server.call("get_work", {"work_id": "10.5555/not-in-the-fixtures"})
        self.assertIn("offline", res["error"])
        self.assertNotIn("empty", res["error"])

    def test_replayed_responses_land_in_the_run_cache(self):
        self.server.call("get_work", {"work_id": SOURCES["saarinen2013"]["doi"]})
        self.assertTrue(any((self.tmp / "cache").glob("openalex-*.json")))


class EmptyCacheTest(unittest.TestCase):
    """Offline with nothing cached yet (a fresh clone): every tool says why, and how to fill the cache."""

    def test_offline_with_an_empty_cache_explains_itself(self):
        from agentkit.mcp import MCPServer
        tmp = Path(tempfile.mkdtemp())
        env = {**os.environ, "RESEARCH_OFFLINE": "1", "RESEARCH_SEED_DIR": str(tmp / "not-there"),
               "RESEARCH_CACHE_DIR": str(tmp / "cache")}
        server = MCPServer([sys.executable, str(ROOT / "mcp_servers" / "research_server.py")], env=env)
        try:
            for tool, args in [("search_works", {"query": "background music recall"}),
                               ("get_work", {"work_id": SOURCES["brennholt2014"]["doi"]}),
                               ("check_doi", {"doi": SOURCES["brennholt2014"]["doi"]})]:
                with self.subTest(tool):
                    error = server.call(tool, args)["error"]
                    self.assertIn("cache is empty", error)
                    self.assertIn("without RESEARCH_OFFLINE", error)
        finally:
            server.close()
            shutil.rmtree(tmp, ignore_errors=True)


class ScorerTest(unittest.TestCase):
    def test_an_invented_reference_is_caught_offline(self):
        from scoring.research import score
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copytree(FIXTURES, tmp / "cache")
            real = SOURCES["ashworthpell2021"]
            (tmp / "report.md").write_text(
                f'A replication found that "{real["quote"]}" [1]. A second paper says the opposite [2].\n\n'
                f"## References\n\n1. Ashworth-Pell et al. https://doi.org/{real['doi']}\n"
                f"2. Made up. https://doi.org/{FAKE_DOI}\n")
            (tmp / "sources.json").write_text(json.dumps({"sources": [{"n": 1, "doi": real["doi"]}, {"n": 2, "doi": FAKE_DOI}]}))
            (tmp / "claims.json").write_text(json.dumps({"claims": [{"source_n": 1, "claim": "x", "quote": real["quote"]}]}))
            os.environ["RESEARCH_OFFLINE"] = "1"
            result = score(tmp)
        finally:
            os.environ.pop("RESEARCH_OFFLINE", None)
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual([f["doi"] for f in result["fabricated_references"]], [FAKE_DOI])
        self.assertFalse(result["checks"]["no_fabricated_references"])
        self.assertTrue(result["checks"]["quotes_verbatim"])
        self.assertTrue(result["checks"]["citations_map_to_sources"])


class OverclaimTest(unittest.TestCase):
    """The red-flag check must catch overclaims but not penalise hedged wording."""

    def test_hedged_and_negated_sentences_are_not_overclaims(self):
        from scoring.research import overclaims
        pats = [r"\bprove[sn]?\b", r"\bproof\b", "definitively"]
        body = ("It shows an association rather than proof that laptops lower grades. "
                "That is not proof of cause. It stops short of proof. "
                "This study proves handwriting is better. The results definitively show it.")
        self.assertEqual(overclaims(body, pats),
                         ["This study proves handwriting is better.", "The results definitively show it."])


if __name__ == "__main__":
    unittest.main()
