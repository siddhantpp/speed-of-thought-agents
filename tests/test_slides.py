"""Tests for the deterministic slide tools (workflows/slide_tools.py) and the slides scorer.

No model or LM Studio server is needed. The overflow and PDF tests render in headless Chrome and are
skipped cleanly when Chrome is not installed.

    .venv/bin/python -m unittest            # from the repo root
    .venv/bin/python tests/test_slides.py
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works both in the exported repo (tests/ next to workflows/) and in the source tree (tools/repo_extras/tests).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "workflows" / "slide_tools.py").exists())
sys.path.insert(0, str(ROOT))
from workflows import slide_tools as st  # noqa: E402

DATA = ROOT / "sample_data" / "slides"
KEY = json.loads((DATA / "answer_key.json").read_text())
TABLE = st.read_table(DATA / "data.csv")
SOURCES = [DATA / "brief.md", DATA / "notes.md", DATA / "data.csv"]

try:
    st.find_chrome()
    CHROME = True
except st.ChromeMissing:
    CHROME = False


def chart(**kw):
    c, problems = st.normalize_chart({"type": "bar", "x": "plan", "y": ["first_years"], "title": "First-years per plan",
                                      "highlight": "", **kw}, TABLE)
    return c, problems


def deck(slides):
    return st.render_deck({"title": "Test deck", "slides": slides}, TABLE)


GOOD = [  # a deck that should pass every check
    {"n": 1, "title": "Nineteen Meals by Default", "bullets": ["Riley Park", "ECON 140, Section 4"],
     "notes": "Hi everyone, I'm Riley, and this is about the meal plan every first-year starts on."},
    {"n": 2, "title": "Most first-years never switch", "bullets": ["31 of 40 still on 19 meals", "Only 9 switched"],
     "notes": "Here is the survey. Thirty-one of the forty first-years I asked are still on the default plan, "
              "and only nine switched to a smaller one.",
     "visual": {"kind": "chart", "chart": chart(highlight="19 meals (default)")[0]}},
    {"n": 3, "title": "Why the default sticks", "bullets": [],
     "notes": "Most people told me they figured nineteen was the recommended amount. That is endorsement, and "
              "effort comes second.",
     "visual": {"kind": "diagram", "diagram": st.normalize_diagram({
         "nodes": [{"id": "d", "label": "Default: 19 meals"}, {"id": "e", "label": "Endorsement (14)"},
                   {"id": "f", "label": "Effort (9)"}, {"id": "s", "label": "Stay on 19"}],
         "edges": [{"from": "d", "to": "e", "label": ""}, {"from": "d", "to": "f", "label": ""},
                   {"from": "e", "to": "s", "label": "top reason"}, {"from": "f", "to": "s", "label": ""}]})[0]}},
]


class Charts(unittest.TestCase):
    def test_marks_are_the_csv_values(self):
        c, problems = chart(y=["avg_meals_used"], highlight="19 meals (default)")
        self.assertEqual(problems, [])
        svg = st.chart_svg(c, TABLE)
        marks = re.findall(r'data-label="([^"]+)" data-series="([^"]+)" data-value="([^"]+)"', svg)
        self.assertEqual(marks, [("19 meals (default)", "avg_meals_used", "11.2"), ("14 meals", "avg_meals_used", "10.4"),
                                 ("10 meals", "avg_meals_used", "8.7")])
        self.assertIn(">11.2<", svg)        # the printed label is the CSV string, not a re-rounded float

    def test_type_follows_the_number_of_columns(self):
        self.assertEqual(chart(type="bar", y=["meals_per_week", "avg_meals_used"])[0]["type"], "grouped_bar")
        self.assertEqual(chart(type="grouped_bar", y=["first_years"])[0]["type"], "bar")
        c = chart(type="hbar", y=["price_usd"])[0]
        self.assertIn(">$3,150<", st.chart_svg(c, TABLE))

    def test_bad_choices_come_back_as_errors(self):
        self.assertTrue(chart(x="first_years")[1])                         # x must be the label column
        self.assertTrue(chart(y=["plan"])[1])                              # y must be numeric
        self.assertTrue(chart(y=["first_years", "price_usd"])[1])          # two very different scales
        self.assertTrue(chart(highlight="21 meals")[1])                    # no such row
        self.assertTrue(chart(title="a chart title that is far too long to read")[1])

    def test_mismatched_marks_are_caught(self):
        good = [{"label": "19 meals (default)", "series": "first_years", "value": "31"}]
        self.assertEqual(st.chart_mismatches(good, TABLE), [])
        self.assertTrue(st.chart_mismatches([{"label": "19 meals (default)", "series": "first_years", "value": "35"}], TABLE))
        self.assertTrue(st.chart_mismatches([{"label": "Block 160", "series": "first_years", "value": "3"}], TABLE))


class Diagrams(unittest.TestCase):
    def test_layered_layout(self):
        d = GOOD[2]["visual"]["diagram"]
        self.assertEqual(st.layer(d["nodes"], d["edges"]), {"d": 0, "e": 1, "f": 1, "s": 2})
        svg = st.diagram_svg(d)
        self.assertEqual(svg.count('class="node"'), 4)
        self.assertEqual(svg.count('class="edge"'), 4)

    def test_cycles_do_not_break_ranking(self):
        nodes = [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]
        self.assertEqual(st.layer(nodes, [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}]), {"a": 0, "b": 1})

    def test_invalid_specs(self):
        self.assertTrue(st.normalize_diagram({"nodes": [{"id": "a", "label": "A"}], "edges": []})[1])
        self.assertTrue(st.normalize_diagram({"nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                                              "edges": [{"from": "a", "to": "z", "label": ""}]})[1])


class Numbers(unittest.TestCase):
    def test_number_forms(self):
        self.assertEqual(st.numbers_in("$3,150 vs $2,890, 11.20 meals, 78%, 6-8 slides"),
                         ["3150", "2890", "11.2", "78", "6", "8"])
        self.assertEqual(st.numbers_in("HW7, L17 and x2 are names"), [])

    def test_struck_out_numbers_are_not_sources(self):
        allowed = st.source_numbers(SOURCES)
        self.assertNotIn("35", allowed)          # ~~35 of 40~~ was a miscount
        self.assertTrue({"31", "40", "78", "11.2", "7.8", "260", "3150"} <= allowed)
        self.assertNotIn("77.5", allowed)        # a model's own arithmetic is not traceable

    def test_answer_key_matches_the_sources(self):
        self.assertEqual({x["value"] for x in KEY["numbers"]}, st.source_numbers(SOURCES))
        for x in KEY["numbers"]:
            for src in x["sources"]:
                self.assertIn(x["value"], st.numbers_in(src.split(": ", 1)[1]), src)
        for label, cols in KEY["chart_values"]["rows"].items():
            row = next(r for r in TABLE["rows"] if r["plan"] == label)
            self.assertEqual({c: float(row[c]) for c in TABLE["numeric"]}, cols)

    def test_check_deck_flags_untraceable_numbers(self):
        bad = [dict(GOOD[1], bullets=["77.5% still on 19 meals", "35 of 40 never switched"])]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "deck.html"
            p.write_text(deck(bad))
            rep = st.check_deck(p, SOURCES, DATA / "data.csv", overflow=False)
        self.assertEqual(rep["per_slide"][0]["untraceable_numbers"], ["35", "77.5"])


class ReadingDecks(unittest.TestCase):
    def test_parse_rendered_deck(self):
        slides = st.parse_deck(deck(GOOD))
        self.assertEqual(len(slides), 3)
        self.assertEqual([s["bullets"] for s in slides], [2, 2, 0])
        self.assertIn("Thirty-one", slides[1]["notes"])
        self.assertNotIn("Thirty-one", slides[1]["visible"])
        self.assertEqual(len(slides[1]["marks"]), 3)
        self.assertEqual((slides[1]["charts"], slides[2]["diagrams"]), (1, 1))
        self.assertNotIn("2", st.numbers_in(slides[1]["visible"]))      # the slide number is not slide text

    def test_parse_a_hand_written_deck(self):
        """Another harness's deck: bare sections, its own classes, a self-closing bar."""
        text = ("<html><body><section><h1>A</h1><p>One<br>two</p><aside class='notes'>Say 31.</aside></section>"
                "<section><svg class='chart'><rect data-label='14 meals' data-series='first_years' data-value='6'/>"
                "</svg><ul><li>x</li><li>y</li></ul></section></body></html>")
        a, b = st.parse_deck(text)
        self.assertEqual((a["visible"], a["notes"]), ("A One two", "Say 31."))
        self.assertEqual((b["bullets"], b["marks"]), (2, [{"label": "14 meals", "series": "first_years", "value": "6"}]))

    def test_word_count(self):
        self.assertEqual(st.count_words("Make the 14-meal plan the default: $260 less, 11.2 used."), 10)


@unittest.skipUnless(CHROME, "Google Chrome / Chromium not installed")
class Chrome(unittest.TestCase):
    def test_overflow_is_detected_and_measured(self):
        long = "Most first-years we surveyed still keep the default nineteen meal plan every single week"
        crowded = dict(GOOD[1], bullets=[long] * 4, title="A title that is long enough to wrap onto a second line")
        rep = st.overflow_of_html(deck([GOOD[0], crowded]))
        self.assertEqual([o["slide"] for o in rep["overflow"]], [2])
        o = rep["overflow"][0]
        self.assertGreaterEqual(o["lines"], 1)
        self.assertIn("text", o["where"])
        msg = st.overflow_message(o, 4)
        self.assertRegex(msg, r"slide 2 overflows by \d+ line\(s\) of text; cut it to [123] bullet")

    def test_a_deck_that_fits_has_no_overflow(self):
        self.assertEqual(st.overflow_of_html(deck(GOOD))["overflow"], [])

    def test_long_title_is_caught(self):
        title = ("This title keeps going and going well past any reasonable length for one slide, because it "
                 "tries to say every single thing at once")
        rep = st.overflow_of_html(deck([dict(GOOD[1], title=title)]))
        self.assertTrue(rep["overflow"] and rep["overflow"][0]["where"].startswith("h2"))

    def test_pdf_has_one_page_per_slide(self):
        with tempfile.TemporaryDirectory() as d:
            html_path, pdf = Path(d) / "deck.html", Path(d) / "deck.pdf"
            html_path.write_text(deck(GOOD))
            self.assertEqual(st.export_pdf(html_path, pdf), 3)
            self.assertEqual(st.pdf_pages(pdf), 3)
            self.assertIn(b"/MediaBox [0 0 960 540]", pdf.read_bytes())      # 1280x720 px = 13.33x7.5 in
            # The CLI does the same for any deck.html.
            with contextlib.redirect_stdout(io.StringIO()) as said:
                self.assertEqual(st.main(["export", str(html_path), str(pdf)]), 0)
            self.assertIn("3 pages", said.getvalue())

    def test_scorer_on_a_code_rendered_deck(self):
        from scoring.slides import score
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "deck.html").write_text(deck(GOOD))
            st.export_pdf(Path(d) / "deck.html", Path(d) / "deck.pdf")
            r = score(d)
        for k in ("numbers_traceable", "chart_values_exact", "diagram_present", "no_overflow",
                  "pdf_one_page_per_slide", "speaker_notes", "word_limits"):
            self.assertTrue(r[k]["pass"], (k, r[k]))
        self.assertFalse(r["slide_count"]["pass"])          # 3 slides is below the brief's 6
        self.assertFalse(r["required_messages"]["pass"])    # no recommendation or caveat


class WorkflowChecks(unittest.TestCase):
    """The code-level critics in workflows/slide_deck.py (imported without calling any model)."""

    @classmethod
    def setUpClass(cls):
        from workflows import slide_deck
        cls.wf = slide_deck
        cls.brief = (DATA / "brief.md").read_text()

    def plan(self, **kw):
        slides = [{"message": "Title", "visual": "none", "covers": []}] + [
            {"message": f"m{k}", "visual": v, "covers": [k]} for k, v in
            zip(range(1, 7), ["none", "chart", "chart", "diagram", "none", "none"])]
        return {"deck_title": "Nineteen Meals by Default", "audience": "section", "minutes": 5, "min_slides": 6,
                "max_slides": 8, "max_bullets": 4, "max_words": 45, "slides": slides, **kw}

    def test_key_messages_are_parsed_from_the_brief(self):
        keys = self.wf.key_messages(self.brief)
        self.assertEqual(len(keys), 6)
        self.assertTrue(keys[1].startswith("It sticks: 31 of the 40"))

    def test_plan_check(self):
        self.assertIsNone(self.wf.check_plan(self.plan(), self.brief, 6))
        self.assertIn("max_words=50 is not a number in the brief", self.wf.check_plan(self.plan(max_words=50), self.brief, 6))
        p = self.plan()
        p["slides"][6]["covers"] = []
        self.assertIn("key messages [6] are not covered", self.wf.check_plan(p, self.brief, 6))
        p = self.plan()
        p["slides"][4]["visual"] = "none"      # the only diagram slide
        self.assertIn("asks for a diagram", self.wf.check_plan(p, self.brief, 6))

    def test_slide_limits(self):
        lim = {"max_words": 45, "notes_min": 30, "notes_max": 100, "sentences": 4}
        ok = {"n": 2, "title": "Most first-years never switch", "bullets": ["31 of 40 still on 19 meals"],
              "notes": "word " * 60, "visual": {"kind": "none"}}
        self.assertEqual(self.wf.limit_problems(ok, lim), [])
        bad = dict(ok, bullets=["Most first-years never switch plans"], notes="Say this. [Next slide's message: x]")
        problems = " ".join(self.wf.limit_problems(bad, lim))
        self.assertIn("repeat the title", problems)
        self.assertIn("prompt text", problems)
        self.assertIn("write 4 full sentences", problems)
        self.assertIn("cut them to 3 short sentences", " ".join(self.wf.limit_problems(dict(ok, notes="w " * 120), lim)))
        coded = dict(ok, visual={"kind": "diagram", "diagram": {"nodes": [{"id": "a", "label": "Stay_on_default"}],
                                                               "edges": []}})
        self.assertIn("not code", " ".join(self.wf.limit_problems(coded, lim)))

    def test_struck_out_notes_never_reach_a_worker(self):
        self.assertNotIn("35 of 40", self.wf.clean_notes((DATA / "notes.md").read_text()))


class PdfCount(unittest.TestCase):
    def test_counts_page_objects_not_the_pages_tree(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.pdf"
            p.write_bytes(b"%PDF-1.4\n1 0 obj <</Type /Pages /Count 2 /Kids [2 0 R 3 0 R]>> endobj\n"
                          b"2 0 obj <</Type /Page /Parent 1 0 R>> endobj\n3 0 obj <</Type/Page/Parent 1 0 R>> endobj\n")
            self.assertEqual(st.pdf_pages(p), 2)


if __name__ == "__main__":
    unittest.main()
