"""Tests for the deterministic class-notes helpers in workflows/class_notes_tools.py.

No model or LM Studio server is needed: the alignment tests use hand-made score matrices, and the
data tests check the sample transcripts and slide exports against sample_data/classes/answer_key.json.

    .venv/bin/python -m unittest            # from the repo root
    .venv/bin/python tests/test_class_notes.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works both in the exported repo (tests/ next to workflows/) and in the source tree (tools/repo_extras/tests).
ROOT = next(p for p in (HERE.parent, HERE.parents[2] / "demos") if (p / "workflows" / "class_notes_tools.py").exists())
sys.path.insert(0, str(ROOT))
from workflows.class_notes_tools import (accuracy, align, find_quote_time, fmt_ts, natural_key, parse_ts,  # noqa: E402
                                         parse_vtt, quote_in, quote_mismatch, runs, slide_files, spoken_form,
                                         time_ranges)

DATA = ROOT / "sample_data" / "classes"
KEY = json.loads((DATA / "answer_key.json").read_text())

ZOOM = """WEBVTT

1
00:00:03.500 --> 00:00:07.250
Leona Brandt-Okoye: Okay, good morning, everybody.

2
00:00:07.400 --> 00:00:11.000
Leona Brandt-Okoye: Can people in the back
hear me okay?

3
00:00:11.200 --> 00:00:13.000
Student: Yes &amp; thanks.
"""

TEAMS = """﻿WEBVTT
Kind: captions
Language: en-US

NOTE
Speaker names come from the roster.

00:00:02.175 --> 00:00:04.097 align:start position:10%
<v Owen Castellanos-Reid>Alright, let's <i>start</i>.</v>

00:00:04.368 --> 00:00:09.361
<v.loud Student>Is it the points at the top right?</v>

00:00:09.500 --> 00:00:10.000
and the bottom?

01:00:00,000 --> 01:00:01,500
<v Owen Castellanos-Reid>One hour in.</v>
"""


def ramp(segments, n_slides, strength=2.0, noise=None):
    """A score matrix: cue t scores `strength` on its true slide and 0 elsewhere (plus optional noise)."""
    rows = []
    for t, s in enumerate(segments):
        row = [0.0] * n_slides
        row[s] = strength
        if noise and t in noise:
            row[noise[t][0]] = noise[t][1]
        rows.append(row)
    return rows


class TimestampTest(unittest.TestCase):
    def test_parse_and_format_round_trip(self):
        self.assertEqual(parse_ts("00:01:02.500"), 62.5)
        self.assertEqual(parse_ts("01:02.5"), 62.5)               # minutes:seconds form
        self.assertEqual(parse_ts("01:00:00,250"), 3600.25)       # SRT-style comma
        self.assertEqual(fmt_ts(3723.4567), "01:02:03.457")


class ParseVttTest(unittest.TestCase):
    def test_zoom_speakers_wrapped_lines_and_entities(self):
        cues = parse_vtt(ZOOM)
        self.assertEqual(len(cues), 3)
        self.assertEqual(cues[0], {"start": 3.5, "end": 7.25, "speaker": "Leona Brandt-Okoye",
                                   "text": "Okay, good morning, everybody."})
        self.assertEqual(cues[1]["text"], "Can people in the back hear me okay?")   # two caption lines joined
        self.assertEqual(cues[2]["speaker"], "Student")
        self.assertEqual(cues[2]["text"], "Yes & thanks.")

    def test_teams_voice_tags_header_note_and_settings(self):
        cues = parse_vtt(TEAMS)
        self.assertEqual([c["speaker"] for c in cues],
                         ["Owen Castellanos-Reid", "Student", "Student", "Owen Castellanos-Reid"])
        self.assertEqual(cues[0]["text"], "Alright, let's start.")    # inline <i> tag removed
        self.assertEqual(cues[0]["end"], 4.097)                       # cue settings ignored
        self.assertEqual(cues[2]["text"], "and the bottom?")          # no speaker: inherits the previous one
        self.assertEqual(cues[3]["start"], 3600.0)

    def test_sample_transcripts_match_the_answer_key(self):
        for k in KEY["classes"]:
            cues = parse_vtt((DATA / k["transcript"]).read_text())
            self.assertEqual(len(cues), k["n_cues"], k["id"])
            self.assertEqual({c["speaker"] for c in cues}, set(k["speakers"]), k["id"])
            words = sum(len(c["text"].split()) for c in cues)
            self.assertEqual(words, k["transcript_words"], k["id"])
            self.assertTrue(all(a["start"] < a["end"] <= b["start"] for a, b in zip(cues, cues[1:])), k["id"])


class SlideOrderTest(unittest.TestCase):
    def test_natural_order_puts_slide10_last(self):
        names = ["Slide1.png", "Slide10.png", "Slide2.png", "Slide9.png"]
        self.assertEqual(sorted(names, key=natural_key), ["Slide1.png", "Slide2.png", "Slide9.png", "Slide10.png"])

    def test_slide_files_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("Slide10.PNG", "Slide2.png", "notes.txt", "Slide1.jpg"):
                (Path(d) / name).write_text("")
            self.assertEqual([p.name for p in slide_files(d)], ["Slide1.jpg", "Slide2.png", "Slide10.PNG"])

    def test_sample_decks(self):
        for k in KEY["classes"]:
            files = slide_files(DATA / k["slides_dir"])
            self.assertEqual([p.name for p in files], [s["file"] for s in k["slides"]], k["id"])


class QuoteTest(unittest.TestCase):
    TEXT = "So, um, I'm moving the quiz to next Monday, November 9th, in class. Same material."

    def test_verbatim_ignoring_case_and_punctuation(self):
        self.assertTrue(quote_in("I'm moving the quiz to next Monday, November 9th", self.TEXT))
        self.assertTrue(quote_in("im moving the QUIZ to next monday november 9th", self.TEXT))

    def test_paraphrase_short_and_partial_words_fail(self):
        self.assertFalse(quote_in("the quiz is moving to next Monday", self.TEXT))     # paraphrase
        self.assertFalse(quote_in("the quiz", self.TEXT))                              # too short to mean anything
        self.assertFalse(quote_in("oving the quiz to next Monday", self.TEXT))         # starts mid-word

    def test_mismatch_message_points_at_the_first_wrong_word(self):
        text = "It says beta 1 plus beta 3 times x1. That should be beta 1 plus beta 3 times x2."
        msg = quote_mismatch("That should be beta_1 plus beta_3 times x_2", text)
        self.assertIn("up to 'plus beta 3 times'", msg)
        self.assertIn("'x 2'", msg)
        self.assertIn("too short", quote_mismatch("Milestone 2", text))

    def test_every_key_quote_is_in_its_transcript(self):
        for k in KEY["classes"]:
            text = " ".join(c["text"] for c in parse_vtt((DATA / k["transcript"]).read_text()))
            quotes = [v["quote"] for v in k["verbal_only"]] + [c["quote"] for c in k["corrections"]] + \
                     [a["quote"] for a in k["announcements"]] + list(k["skip_quotes"].values())
            for q in quotes:
                self.assertTrue(quote_in(q, text, min_words=3), (k["id"], q))

    def test_quote_time_spans_cues(self):
        cues = parse_vtt(ZOOM)
        self.assertEqual(find_quote_time("in the back hear me okay", cues), 7.4)
        self.assertIsNone(find_quote_time("not said at all here", cues))


class SpokenFormTest(unittest.TestCase):
    def test_symbols_read_aloud(self):
        self.assertEqual(spoken_form("β1 + β3·x2"), "beta 1 plus beta 3 times x2")
        self.assertIn("percent", spoken_form("41%"))


class AlignTest(unittest.TestCase):
    def test_monotonic_path_follows_the_evidence(self):
        truth = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        self.assertEqual(align(ramp(truth, 4)), truth)

    def test_one_noisy_cue_does_not_cause_a_jump(self):
        truth = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        noisy = ramp(truth, 3, noise={4: (2, 3.0)})     # cue 4 looks like slide 3 for a moment
        self.assertEqual(align(noisy), truth)

    def test_unprompted_return_is_too_expensive_but_a_cued_one_is_found(self):
        truth = [0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2]
        scores = ramp(truth, 3, strength=3.0)
        texts = [""] * len(truth)
        self.assertNotEqual(align(scores, texts), truth)            # without words, stay monotonic
        texts[7] = "Actually, let me go back to the chart for one second."
        self.assertEqual(align(scores, texts), truth)               # "go back" makes the return cheap

    def test_resume_after_a_long_return_is_not_a_skip(self):
        # 1 2 3 4 5 6, back to 2, then forward again to 6 and on to 7
        truth = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 1, 1, 5, 5, 6, 6]
        texts = [""] * len(truth)
        texts[11] = "let me flip back to the formula"
        self.assertEqual(align(ramp(truth, 7, strength=3.0), texts), truth)

    def test_a_briefly_skipped_slide_still_gets_its_cue(self):
        truth = [0, 0, 0, 1, 2, 2, 2]                    # slide 2 is up for one cue ("skipping this one")
        self.assertEqual(align(ramp(truth, 3, strength=3.0)), truth)

    def test_runs_time_ranges_and_accuracy(self):
        cues = [{"start": float(i), "end": i + 0.9} for i in range(5)]
        labels = [1, 1, 2, 1, 1]
        self.assertEqual(runs(labels), [(1, 0, 1), (2, 2, 2), (1, 3, 4)])
        self.assertEqual(time_ranges(labels, cues), {1: [["00:00:00.000", "00:00:01.900"],
                                                         ["00:00:03.000", "00:00:04.900"]],
                                                     2: [["00:00:02.000", "00:00:02.900"]]})
        self.assertAlmostEqual(accuracy([1, 1, 1, 1, 1], labels, cues), 0.8)

    def test_answer_key_ranges_reproduce_its_cue_labels(self):
        for k in KEY["classes"]:
            cues = parse_vtt((DATA / k["transcript"]).read_text())
            got = time_ranges(k["cue_slides"], cues)
            self.assertEqual({s["slide"]: s["time_ranges"] for s in k["slides"]}, got, k["id"])
            self.assertEqual(accuracy(k["cue_slides"], k["cue_slides"], cues), 1.0)


if __name__ == "__main__":
    unittest.main()
