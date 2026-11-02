#!/usr/bin/env python3
"""Render Riley Park's Monday 2026-11-02 classes from ONE data table (CLASSES below):
slide decks (one PNG per slide, PowerPoint-export names Slide1.png ...), lecture transcripts
(WebVTT with timestamps and speaker names), schedule.json and answer_key.json.

Re-runnable and deterministic:  ../../.venv/bin/python make_slides.py

The script text, the slide text and every answer-key item live in the same dicts, so they
cannot drift apart: every quote in the key is asserted to be verbatim in the transcript, every
"image only" fact is asserted to be absent from the speech, and each slide's transcript time
range comes from the same cue timings that are written to the .vtt file.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1280, 720                                  # PowerPoint's default 16:9 export size
FONTS = "/System/Library/Fonts"
AVENIR = f"{FONTS}/Avenir Next.ttc"               # 0 Bold, 2 Demi Bold, 5 Medium, 7 Regular
HELV = f"{FONTS}/HelveticaNeue.ttc"               # 0 Regular, 1 Bold, 10 Medium
GEORGIA, GEORGIA_B = f"{FONTS}/Supplemental/Georgia.ttf", f"{FONTS}/Supplemental/Georgia Bold.ttf"
ARIAL, ARIAL_B = f"{FONTS}/Supplemental/Arial.ttf", f"{FONTS}/Supplemental/Arial Bold.ttf"
MENLO = f"{FONTS}/Menlo.ttc"


def KP(text, source, match, quote=None):
    """One key point. source: slide (printed), speech (said only aloud) or image (only in a chart).
    match: groups of regex alternatives; every group must match the notes (see scoring/class_notes.py).
    Prefix an alternative with 'c:' to match against the compact formula form (b1+b3x2)."""
    return {"text": text, "source": source, "match": match, **({"quote": quote} if quote else {})}


FAN = (r"\bfan|funnel|cone|megaphone|trumpet|wid(en|ens|ening|er)\b|spread(s|ing)? out|more spread|"
       r"spread (grows|increases|gets (larger|bigger|wider))|(increasing|growing|larger|bigger|wider) spread|"
       r"grow(s|ing)? (larger|wider|bigger)|get(s)? (larger|wider|bigger) (as|with|for)")

# ======================================================================================
# Single source of truth
# ======================================================================================
# script items: (slide number on screen, who, text). who: "P" = instructor, "S" = student,
# "PAUSE" = silence of `text` seconds (an in-class activity, or "look at this for 10 seconds").
# Class times are the weekly series in ../inbox/calendar_before.json. STAT 121 meets Mon/Wed at this slot, so the
# in-class Quiz 4 (moved to Mon Nov 9) starts at the same time.
STAT_SLOT = ("12:00", "13:15")
CLASSES = [
    dict(
        id="econ140", course="ECON 140: Behavioral Economics", code="ECON 140", session="Lecture 17",
        instructor="Prof. Leona Brandt-Okoye", speaker="Leona Brandt-Okoye",
        start="2026-11-02T10:30:00-05:00", end="2026-11-02T11:45:00-05:00",
        title="Defaults: Why They Stick, and When They Backfire", vtt="zoom", theme="econ", seed=140,
        slides=[
            dict(kind="title", title="Defaults: Why They Stick, and When They Backfire",
                 sub=["Lecture 17", "Prof. Leona Brandt-Okoye", "Monday, November 2, 2026"], keys=[]),
            dict(kind="bullets", title="Why do defaults stick? Three channels",
                 bullets=["Effort: switching takes time and attention, so people stay put (inertia)",
                          "Endorsement: the default reads as a recommendation",
                          "Reference point: the default becomes the status quo, and leaving it feels like a loss"],
                 keys=[KP("Effort: switching costs time and attention, so people stay put (inertia)", "slide",
                          [["effort|inertia"]]),
                       KP("Endorsement: people read the default as a recommendation", "slide",
                          [["endorse|recommend"]]),
                       KP("Reference point: the default becomes the status quo; leaving it feels like a loss",
                          "slide", [["reference point|status quo"]]),
                       KP("Exam hint: the final gives a scenario and asks which channel is doing the work; "
                          "name the one that matters most and say why", "speech",
                          [[r"\bexam|\bfinal\b"], ["scenario|which channel"]],
                          quote="On the final exam, I will give you a scenario")]),
            dict(kind="bars", title="Fernhill Logistics: plan participation after 12 months",
                 bars=[("Opt-in", "(2023 hires)", 41), ("Automatic enrollment at 3%", "(2024 hires)", 88)],
                 ylabel="% of new hires contributing", note="Teaching case; numbers simplified.",
                 keys=[KP("Participation was 41% under opt-in vs 88% with automatic enrollment", "image",
                          [[r"\b41\b"], [r"\b88\b"]]),
                       KP("Same plan and match; only the default changed between the 2023 and 2024 hires",
                          "speech", [["same plan|same .{0,20}match|(remained|stayed|were|was) (the )?same|unchanged|only (thing )?(that )?"
                                      "changed|only the default|only difference|nothing else changed"]],
                          quote="The only thing that changed is what happens if you do nothing"),
                       KP("Student question: the two groups of hires were similar on age and salary, "
                          "so it is not a story about who got hired", "speech",
                          [[r"\bage\b|salar"], ["similar|same|composition|who got hired|not .*different"]],
                          quote="The two groups of hires look very similar on age and salary"),
                       KP("On a return to this chart: the extra participants were mostly younger, "
                          "lower-paid workers", "speech",
                          [["young"], [r"low(er)?[- ]paid|low(er)? (income|wage|salar)"]],
                          quote="they're mostly younger, lower-paid workers")]),
            dict(kind="bullets", title="The catch: people stick to the default rate, too",
                 bullets=["Most auto-enrolled workers stayed at the 3% default rate",
                          "Under opt-in, many joiners chose more than 3%",
                          "Participation rose a lot; total savings rose by less"],
                 keys=[KP("Most auto-enrolled workers stayed at the 3% default rate", "slide",
                          [[r"\b3 ?%|3 percent|three percent"], ["stay|stuck|stick|remain|anchor"]]),
                       KP("Participation rose a lot, but total savings rose by less", "slide",
                          [["saving"], [r"\bless\b|smaller|not as much"]])]),
            dict(kind="bullets", title="Two fixes",
                 bullets=["Auto-escalation: the default rate rises every year unless you opt out",
                          "Active choice: no default; everyone must pick a rate",
                          "Trade-off: active choice respects preferences but costs effort"],
                 keys=[KP("Auto-escalation raises the default rate every year unless you opt out", "slide",
                          [["escalat"]]),
                       KP("Fernhill escalates one percentage point a year, capped at 10%", "speech",
                          [[r"one percentage point|1 percentage point|one point|\b1 point|\b1 ?%|one percent"],
                           [r"\b10 ?%|10 percent|ten percent|\bcap"]],
                          quote="the escalation is one percentage point a year, capped at 10%"),
                       KP("Active choice: no default, everyone must pick; respects preferences but costs effort",
                          "slide", [["active choice"]])]),
            dict(kind="activity", title="Your turn: design the default",
                 bullets=["First-years are defaulted into the largest campus meal plan.",
                          "Should they be? Which channel matters most? What would you change?"],
                 box="2 minutes with a neighbor",
                 keys=[KP("Class answer: make the middle plan the default, because a default reads as the "
                          "recommended amount (endorsement)", "speech",
                          [["meal plan|dining|middle plan|biggest plan|largest"], ["endorse|recommend"]],
                          quote="So that's the endorsement channel")]),
            dict(kind="formula", title="Appendix: present bias (the β–δ model)",
                 formula="U_{0} = u(c_{0}) + β Σ δ^{t} u(c_{t}),   t = 1 … T",
                 bullets=["β < 1 puts extra weight on today", "δ is the usual per-period discount factor"],
                 skipped=True, skip_quote="I'm going to skip it today", keys=[]),
            dict(kind="bullets", title="When defaults backfire",
                 bullets=["Chosen for the firm, not the person (“sludge”)",
                          "One-size defaults can be wrong for many people",
                          "Trust: a default seen as manipulative loses its endorsement power"],
                 keys=[KP("Sludge: a default chosen for the firm, not the person", "slide",
                          [["sludge|for the firm"]]),
                       KP("A default seen as manipulative loses its endorsement power (trust)", "slide",
                          [["trust|manipulat"]]),
                       KP("Example said aloud: a gym membership that auto-renews and can only be cancelled "
                          "by certified letter", "speech", [["gym"]],
                          quote="A gym membership that renews automatically every year")]),
            dict(kind="bullets", title="Essay 2 and what’s next",
                 bullets=["Essay 2 draft due Thu Nov 5, 11:59 pm on Canvas",
                          "Evaluate ONE real-world default using at least two frameworks (weeks 6–8)",
                          "Next: present bias and commitment devices"],
                 keys=[KP("Evaluate one real-world default with at least two frameworks from weeks 6-8",
                          "slide", [["two frameworks|2 frameworks|two of the frameworks|two .{0,20}frameworks"]]),
                       KP("Next: present bias and commitment devices", "slide", [["commitment device"]])]),
        ],
        corrections=[dict(slide=9, slide_said="Essay 2 draft due Thu Nov 5, 11:59 pm",
                          lecturer_said="Essay 2 draft is due Mon Nov 9, 11:59 pm (the slide has the old date)",
                          quote="That's the old date, I didn't update the slide",
                          right=[[r"essay|draft"], [r"nov(ember)?\.? 9|11/9|\b9 nov"]],
                          wrong=[[r"essay|draft"], [r"nov(ember)?\.? 5\b|11/5\b|\b5 nov|thu(rsday)?,? nov"]])],
        announcements=[
            dict(slide=9, text="Essay 2 draft due Mon Nov 9, 11:59 pm on Canvas (the slide's Nov 5 date is old)",
                 when="2026-11-09T23:59:00-05:00", source="slide (wrong) + speech (correction)",
                 match=[[r"essay|draft"], [r"nov(ember)?\.? 9|11/9|\b9 nov"]],
                 quote="the draft is due Monday, November 9th at 11:59 PM"),
            dict(slide=9, text="Essay 2 peer review: pairings go out Tue Nov 10; reviews due Mon Nov 16, 11:59 pm",
                 when="2026-11-16T23:59:00-05:00", source="speech only",
                 match=[[r"peer review|review"], [r"nov(ember)?\.? 16|11/16|\b16 nov"]],
                 quote="your review of your partner's draft is due Monday, November 16th at 11:59 PM"),
        ],
        script=[
            (1, "P", "Okay, good morning, everybody. Let's, um, let's get going. Can people in the back hear me "
                     "okay? Great. So today is Lecture 17, and it's the one your second essay is built on. We've "
                     "spent three weeks on frameworks, present bias, loss aversion, mental accounting, and today I "
                     "want to put them to work on one very specific design tool, which is the default. The default "
                     "is what happens to you when you don't make a choice. By the end of today you should be able "
                     "to look at any default out in the world and say why it sticks and, uh, when it's going to "
                     "backfire."),
            (2, "P", "So, first question. Why do defaults stick at all? Standard economics says a default shouldn't "
                     "matter much. If switching is cheap and you know what you want, you switch. But defaults "
                     "matter a lot, and there are three channels, three reasons, and they're on the slide. The "
                     "first one is effort. Switching takes time and attention, and attention is scarce, so people "
                     "stay put. That's inertia. It's the boring explanation, but it's a big one. Think about your phone. How "
                     "many of you have changed the default ringtone? Right, almost nobody. That's mostly effort. "
                     "The second is "
                     "endorsement. People read the default as a recommendation. If my employer picked this plan "
                     "for me, they probably know something, right? The default sends a signal. And the third is "
                     "the reference point. Once something is the default, it becomes the status quo, and leaving "
                     "it feels like giving something up. That's loss aversion from week 7 doing its work."),
            (2, "P", "Now, here's something I want you to hear clearly, because it matters in a few weeks. On the "
                     "final exam, I will give you a scenario, some default out in the world, and I'll ask you which "
                     "channel is doing the work. And the honest answer is usually more than one. The best answers "
                     "name the channel that matters most and say why. So, um, practice that on every example we "
                     "see today."),
            (3, "P", "Okay. Here's a case I like a lot. This is a mid-sized company, Fernhill Logistics. It's a "
                     "teaching case, so the numbers are cleaned up. In 2023, new hires had to opt in to the "
                     "retirement plan. They had to find a form and fill it out. In 2024, the company switched to "
                     "automatic enrollment. Same plan, same employer match, same everything. The only thing that "
                     "changed is what happens if you do nothing. And, um, look at the two bars. Just look at the "
                     "gap. I'm not even going to read the numbers to you, you can see it. This is one of the most "
                     "replicated results in this whole part of the course. Change the default and participation "
                     "moves by a huge amount, and nobody's preferences changed between 2023 and 2024."),
            (3, "S", "Were the 2024 people just different, though? Like, maybe the company hired different kinds "
                     "of people that year?"),
            (3, "P", "Good question, that's exactly the right worry. The case checks that. The two groups of hires "
                     "look very similar on age and salary, so it's not a story about who got hired. It's the "
                     "default."),
            (4, "P", "But. There's always a but. Here's the catch. When Fernhill auto-enrolled people, it had to "
                     "pick a contribution rate, and it picked 3% of pay. And most auto-enrolled workers stayed at "
                     "exactly 3%, year after year. Now, under opt-in, the people who did join picked their own "
                     "rate, and a lot of them picked more than 3. So some of the people who would have saved 6 or "
                     "8% on their own got anchored at 3. The default pulled them down. So participation went way "
                     "up, but total savings went up by less than you'd guess from that chart. Keep that in mind. A "
                     "default fixes one thing and can quietly break another."),
            (4, "S", "Why not just set the default at 6% then?"),
            (4, "P", "Some firms do. But a higher default also makes more people opt out entirely, because the "
                     "hit to their paycheck is bigger and they notice it. So there's a trade-off between "
                     "participation and the rate. That's exactly the kind of tension I want to see in your essays."),
            (5, "P", "So how do you fix that? There are two standard fixes on this slide. The first is "
                     "auto-escalation. The default rate goes up a little every year unless you opt out. At "
                     "Fernhill, the escalation is one percentage point a year, capped at 10%. So you start at 3, "
                     "next year 4, and so on. Inertia, which was the problem, now works for you. Some plans tie each "
                     "increase to a raise, so your take-home pay never actually goes down, which takes the loss "
                     "aversion out of it. The second fix is "
                     "active choice. No default at all. Everyone has to pick a rate on day one, you can't start the "
                     "job without picking. And the trade-off is pretty clear. Active choice respects people's "
                     "preferences, but it costs effort, and some people will pick badly or, um, just pick whatever "
                     "number is first on the form."),
            (6, "P", "Okay, let's try one. It's on the slide. Right now, first-years here are defaulted into the "
                     "largest meal plan. Should that be the default? Take 2 minutes with the person next to you. "
                     "Which channel matters most, and what would you change?"),
            (6, "PAUSE", 105),
            (6, "P", "Okay, okay, let's come back. Who wants to share? Yes, in the middle."),
            (6, "S", "We said the default should be the middle plan, not the biggest, because people treat the "
                     "default like the recommended amount. So the biggest plan makes first-years think they need "
                     "that much food."),
            (6, "P", "Nice. So that's the endorsement channel. The default is telling first-years, this is how "
                     "much a normal person eats. The other answer people usually give is effort, nobody bothers to "
                     "downgrade. Both are fine. The best answer says which one matters more and why."),
            (7, "P", "The next slide is the beta-delta model, the present bias math from the appendix of the "
                     "reading. I'm going to skip it today. We'll do present bias properly on Wednesday, so don't "
                     "worry about this one now. Skipping it."),
            (8, "P", "So, when do defaults backfire? Three ways on the slide. First, a default can be chosen for "
                     "the firm, not for the person. People call that sludge. Here's my favorite example. A gym "
                     "membership that renews automatically every year, where the only way to cancel is to mail a "
                     "certified letter. That's a default too. It's just pointed at you instead of working for you. "
                     "Second, one-size defaults can be wrong for lots of people. Remember, 3% is too low for the "
                     "person who wanted 8. And a default that's fine for the average worker can be bad for "
                     "someone at the edges, like a person with a lot of credit card debt who shouldn't be saving "
                     "for retirement yet. And third, trust. If people find out a default was set to manipulate "
                     "them, it loses its endorsement power, and sometimes people move away from it on purpose. "
                     "Actually, let me go back to the Fernhill chart for one second."),
            (3, "P", "So remember the gap between those bars. Who's in the gap? Who are the extra people who "
                     "joined? In the case, they're mostly younger, lower-paid workers, the people who are least "
                     "likely to go find a form. That's the good news part of this story. The default helped the "
                     "people who needed the help most. Okay, forward again."),
            (8, "P", "So when you evaluate a default in your essay, ask both questions. Who does it help, and who "
                     "does it hurt?"),
            (9, "P", "Okay, last slide, logistics. And I need to fix something. The slide says the Essay 2 draft "
                     "is due Thursday, November 5th. That's the old date, I didn't update the slide. As I emailed "
                     "yesterday, the draft is due Monday, November 9th at 11:59 PM. Monday the 9th. Please don't "
                     "go by the slide. Everything else is the same. Pick one real-world default and evaluate it "
                     "using at least two frameworks from weeks 6 to 8. And since a few of you asked about peer "
                     "review, I'll go through the details in section, but here are the dates. Pairings go out "
                     "Tuesday, November 10th, and your review of your partner's draft is due Monday, November 16th "
                     "at 11:59 PM. On Wednesday we start present bias and commitment devices. Have a good day, "
                     "everyone."),
        ],
    ),
    dict(
        id="stat121", course="STAT 121: Applied Regression", code="STAT 121", session="Lecture 18",
        instructor="Dr. Owen Castellanos-Reid", speaker="Owen Castellanos-Reid",
        start=f"2026-11-02T{STAT_SLOT[0]}:00-05:00", end=f"2026-11-02T{STAT_SLOT[1]}:00-05:00",
        title="Interactions in Practice, and Reading Residual Plots", vtt="teams", theme="stat", seed=121,
        slides=[
            dict(kind="title", title="Interactions in Practice, and Reading Residual Plots",
                 sub=["Lecture 18", "Dr. Owen Castellanos-Reid", "Monday, November 2, 2026"], keys=[]),
            dict(kind="formula", title="A model with an interaction",
                 formula="score = β_{0} + β_{1}·study + β_{2}·sleep + β_{3}·(study × sleep) + ε",
                 bullets=["study = hours studied in the week before the exam",
                          "sleep = average hours of sleep per night that week",
                          "β_{3} lets the payoff from studying depend on sleep",
                          "Simulated data, n = 212 students"],
                 keys=[KP("The product term study × sleep (coefficient β3) lets the effect of studying depend "
                          "on sleep", "slide", [["interaction|product"], ["depend"]])]),
            dict(kind="bullets", title="Interpreting the interaction",
                 bullets=["Effect of a 1-unit increase in x_{1}:   β_{1} + β_{3}·x_{1}",
                          "β_{1} alone = effect of x_{1} when x_{2} = 0",
                          "Does the effect change at all? Test H_{0}: β_{3} = 0"],
                 keys=[KP("β1 alone is the effect of x1 when x2 = 0", "slide",
                          [["c:x2=0|c:x2is(0|zero)|c:x2equals(0|zero)|c:(zero|0)sleep|c:sleep(=|is|of)(0|zero)|"
                            "c:sleeps?(0|zero)hours|c:(0|zero)hours"]]),
                       KP("Test H0: β3 = 0 to ask whether the effect changes at all", "slide",
                          [["c:b3=0|c:b3is(0|zero)|c:b3equals(0|zero)|c:h0"]])]),
            dict(kind="table", title="Fitted model: score ~ study × sleep",
                 rows=[("term", "estimate", "std. error"), ("(Intercept)", "52.0", "4.1"), ("study", "3.1", "0.9"),
                       ("sleep", "1.8", "0.6"), ("study × sleep", "0.4", "0.12")],
                 note="R² = 0.46     n = 212",
                 keys=[KP("Interaction estimate 0.4 (SE 0.12): clearly not zero", "slide",
                          [[r"0\.4\b|\.4\b"], [r"0?\.12\b|not (0|zero)|nonzero|non-zero|significant"]]),
                       KP("Worked example said aloud: at 8 hours of sleep, an extra study hour is worth "
                          "3.1 + 0.4 × 8 = 6.3 points", "speech", [[r"6\.3\b"]],
                          quote="so 6.3 points per extra hour of studying"),
                       KP("At 5 hours of sleep it is 3.1 + 2 = 5.1 points", "speech", [[r"5\.1\b"]],
                          quote="so 5.1 points"),
                       KP("Student question: 3.1 alone is the effect of studying at zero sleep, a person who "
                          "doesn't exist", "speech",
                          [[r"zero|\b0 hours|no sleep"],
                           [r"(doesn.t|does not|don.t) exist|nonsense|meaningless|useless|nobody|no one|"
                            r"not meaningful|unrealistic|impossible|not realistic"]],
                          quote="And that's a person who doesn't exist")]),
            dict(kind="bullets", title="Centering",
                 bullets=["Subtract the mean before forming the product: sleep_{c} = sleep − 7",
                          "Centered fit: study coefficient = 5.9 (the effect at average sleep)",
                          "β_{3}, the predictions and R² do not change"],
                 keys=[KP("Centering: subtract the mean (sleep minus 7) before forming the product", "slide",
                          [["center"], ["mean|average"]]),
                       KP("Centered study coefficient 5.9 = effect at average sleep", "slide", [[r"5\.9\b"]]),
                       KP("β3, predictions and R² do not change", "slide",
                          [["c:b3|interaction"], [r"(not|n.t) change|unchanged|same|stays"]]),
                       KP("Quiz 4 hint: you'll get an uncentered model and must interpret β1 as the effect "
                          "when the other variable is 0", "speech",
                          [["quiz"], [r"not centered|uncentered|un-centered|isn.t centered|without centering|"
                                      r"non-centered"]],
                          quote="On Quiz 4, I'm going to give you a model that is not centered")]),
            dict(kind="scatter", title="Residuals vs. fitted: Model 2", note="Model 2: score ~ study × sleep (n = 212)",
                 keys=[KP("The plot fans out: residual spread grows as fitted values increase (non-constant "
                          "variance)", "image", [[FAN]])]),
            dict(kind="bullets", title="If the error variance is not constant: three fixes",
                 bullets=["Transform the outcome (e.g., log y) and refit",
                          "Robust (HC) standard errors: same estimates, honest SEs",
                          "Weighted least squares, if you can model the variance"],
                 keys=[KP("Transform the outcome (e.g., log) and refit", "slide", [[r"\blog"]]),
                       KP("Robust (HC) standard errors: same estimates, honest SEs", "slide", [[r"robust|\bhc\b"]]),
                       KP("Weighted least squares if you can model the variance", "slide",
                          [["weighted least squares|wls"]]),
                       KP("HW7 hint: robust standard errors are all that's expected on problem 4 (no WLS needed)",
                          "speech", [["robust"], [r"problem 4|problem four|hw ?7|homework 7"]],
                          quote="robust standard errors are all I'm expecting on problem 4"),
                       KP("Student question: after a log, a coefficient is roughly a percent change", "speech",
                          [["percent|%"], [r"\blog"]], quote="a coefficient is roughly a percent change")]),
            dict(kind="formula", title="Leverage and Cook’s distance",
                 formula="D_{i} = ( e_{i}^{2} / (p · MSE) ) · h_{ii} / (1 − h_{ii})^{2}",
                 bullets=["h_{ii}: leverage of point i", "Rule of thumb: look closer when D_{i} > 4/n"],
                 skipped=True, skip_quote="I'm skipping it today", keys=[]),
            dict(kind="bullets", title="Homework 7 and next time",
                 bullets=["HW7 due Thu Nov 5, 11:59 pm: one PDF rendered from your notebook",
                          "Problems 3–4 use today’s material",
                          "Wednesday: leverage and outliers"],
                 keys=[KP("HW7 due Thu Nov 5, 11:59 pm as one PDF from the notebook", "slide",
                          [[r"hw ?7|homework 7"], [r"nov(ember)?\.? 5\b|11/5\b|thu"]]),
                       KP("Wednesday: leverage and outliers", "slide", [["leverage|outlier"]])]),
            dict(kind="bullets", title="Questions?",
                 bullets=["Office hours: Dr. Castellanos-Reid, Thu 10–11 am, Pemberton 214",
                          "TF office hours: Mina Haddad-Byrne, Tue 3–5 pm, Pemberton 210"],
                 keys=[KP("Instructor office hours Thu 10-11 am, Pemberton 214", "slide", [["214|thu.{0,12}10"]])]),
        ],
        corrections=[dict(slide=3, slide_said="Effect of a 1-unit increase in x1: β1 + β3·x1",
                          lecturer_said="The effect of x1 is β1 + β3·x2 (times the other variable)",
                          quote="That should be beta 1 plus beta 3 times x2",
                          right=[[r"c:b1\+b3x2|c:b1\+b3sleep|c:b3x2\+b1|c:b1\+x2b3"]],
                          wrong=[[r"c:b1\+b3x1(?!\d)|c:b1\+b3study"]])],
        announcements=[
            dict(slide=1, text="Quiz 4 moved from Wed Nov 4 to Mon Nov 9, in class (same material: interactions "
                               "and residual plots)", when=f"2026-11-09T{STAT_SLOT[0]}:00-05:00", source="speech only",
                 match=[["quiz"], [r"nov(ember)?\.? 9|11/9|\b9 nov"]],
                 wrong=[["quiz"], [r"nov(ember)?\.? 4\b|11/4\b|wed"]],
                 quote="I'm moving the quiz to next Monday, November 9th, in class"),
            dict(slide=1, text="TF office hours this week: Wed Nov 4, 2-4 pm, Pemberton 210 (not the usual Tuesday)",
                 when="2026-11-04T14:00:00-05:00", source="speech only (slide 10 shows the usual Tuesday slot)",
                 match=[["office hours|mina"], [r"wed|nov(ember)?\.? 4\b|11/4\b"]],
                 wrong=[["office hours|mina"], [r"tue"]],
                 quote="Mina's office hours this week are on Wednesday from 2 to 4 in Pemberton 210"),
            dict(slide=9, text="HW7 due Thu Nov 5, 11:59 pm (one PDF rendered from the notebook)",
                 when="2026-11-05T23:59:00-05:00", source="slide + speech",
                 match=[[r"hw ?7|homework 7"], [r"nov(ember)?\.? 5\b|11/5\b"]],
                 quote="Homework 7 is due Thursday at 11:59 PM"),
        ],
        script=[
            (1, "P", "Alright, let's start. Good afternoon. Before anything else, two logistics things, so, uh, "
                     "write these down. First, Quiz 4. It was going to be this Wednesday. Since Homework 7 is due "
                     "Thursday, I don't want you doing both in the same two days, so I'm moving the quiz to next "
                     "Monday, November 9th, in class. Same material, interactions and residual plots, which is "
                     "today. So Quiz 4 is Monday the 9th now, not Wednesday. Second, Mina's office hours this week "
                     "are on Wednesday from 2 to 4 in Pemberton 210, not the usual Tuesday slot. She posted that on "
                     "Slack, but I know not everyone reads Slack. Okay. Today we finish interactions, and then we "
                     "start reading residual plots, which is the other half of the homework."),
            (2, "P", "So here's the model from last time. We're predicting exam score from two things. Study is "
                     "hours studied in the week before the exam, and sleep is average hours of sleep per night that "
                     "week. And the new piece is this product term, study times sleep, with its own coefficient, "
                     "beta 3. Without that term, the model says an extra hour of studying is worth the same amount "
                     "for everybody, no matter how much they sleep. With it, the payoff from studying is allowed to "
                     "depend on sleep. That's all an interaction is. The effect of one variable depends on the level "
                     "of the other one. Sleep here is the average for the week, not the night before the exam, and "
                     "that matters for how we talk about it. And this is simulated data, 212 students, so, um, "
                     "don't go quoting it to your friends."),
            (2, "S", "Do we still need study and sleep on their own if we have the product?"),
            (2, "P", "Yes. Keep the main effects in whenever you include their product. Dropping them forces the "
                     "model through strange constraints, and the coefficients stop meaning anything sensible. "
                     "People call that the hierarchy principle."),
            (3, "P", "So how do you interpret it? The first line on this slide is the most important line today, "
                     "and, uh, it's wrong. Sorry about that. It says the effect of a one unit increase in x1 is beta "
                     "1 plus beta 3 times x1. That should be beta 1 plus beta 3 times x2. Times the other variable. "
                     "The effect of x1 depends on the value of x2, that's the whole point of an interaction. If you "
                     "copied the slide, fix it now, and I'll post a corrected version tonight. The second line, beta "
                     "1 on its own, is the effect of x1 when x2 equals 0. Hold on to that, because it causes "
                     "trouble in a minute. And third, if you want to know whether the effect changes at all, you "
                     "test whether beta 3 is 0. Let me say the first line one more time, because it's the mistake I see "
                     "on every homework. The effect of x1 is beta 1 plus beta 3 times x2."),
            (4, "P", "Okay, here's the fitted model. Intercept 52, study 3.1, sleep 1.8, and the interaction is 0.4 "
                     "with a standard error of 0.12, so it's clearly not 0. So let's use it. What's the payoff from "
                     "one more hour of studying for a student who sleeps 8 hours a night? Let me flip back to the "
                     "formula for a second."),
            (3, "P", "Right, beta 1 plus beta 3 times x2, the corrected version. Here x2 is sleep. So it's 3.1 plus "
                     "0.4 times 8."),
            (4, "P", "Back to the table. 0.4 times 8 is 3.2, plus 3.1, so 6.3 points per extra hour of studying. "
                     "And for a student who sleeps 5 hours, it's 3.1 plus 2, so 5.1 points. Same hour of studying, "
                     "worth more if you've slept. And notice the sign. The interaction is positive, so sleep makes "
                     "studying more valuable. If it were negative, the payoff from studying would shrink the more "
                     "you sleep, which would be a strange world. Yeah, question?"),
            (4, "S", "So does the 3.1 by itself mean the effect of studying for someone who sleeps zero hours?"),
            (4, "P", "Yes. Exactly. And that's a person who doesn't exist. Nobody in the data sleeps zero hours. So "
                     "3.1 is a perfectly correct number that answers a useless question. Which brings us to "
                     "centering."),
            (5, "P", "Centering means you subtract the mean before you build the product. Average sleep in this "
                     "data is 7 hours, so centered sleep is sleep minus 7. Refit, and now the study coefficient is "
                     "5.9, which is the payoff from studying at average sleep. That's a number you can actually say "
                     "out loud in a sentence. And notice what doesn't change. The interaction, beta 3, is still 0.4. "
                     "The predictions are the same and the R squared is the same. Centering changes what the main "
                     "effects mean, not what the model says. Quick self-check: if you center, refit, and beta 3 "
                     "changed, you made a mistake in your code. And a hint, since I just moved the quiz. On Quiz 4, "
                     "I'm going to give you a model that is not centered and ask you to interpret beta 1. The right "
                     "answer always says, when the other variable is 0. Every year people write at the average, and "
                     "that's only true after centering."),
            (6, "P", "Okay, second half. Residual plots. This is the residuals versus fitted plot for the model we "
                     "just fit, before any fixes. Fitted values along the bottom, residuals up the side, and the red "
                     "dashed line is zero. If the model were doing its job, you'd see a shapeless cloud around that "
                     "line. Take 10 seconds and just look at it."),
            (6, "PAUSE", 10),
            (6, "P", "I'm not going to tell you what's wrong with it, because problem 4 on Homework 7 asks you to "
                     "diagnose a plot that looks a lot like this one, and I want you to practice saying it in your "
                     "own words. But I'll tell you this much. It's not about one weird point. Look at the whole "
                     "shape, left to right."),
            (6, "S", "Is it, like, the points at the top right?"),
            (6, "P", "Not just those. Compare the left side with the right side. That's all I'll say."),
            (7, "P", "So when you see a problem like that, it usually means the variance of the errors isn't "
                     "constant, and there are three standard fixes on this slide. One, transform the outcome, often "
                     "with a log, and refit. Two, keep the model and use robust standard errors. People call them HC "
                     "standard errors, for heteroskedasticity-consistent. The estimates don't change, but the "
                     "standard errors get honest. Three, weighted least squares, if you can actually model how the "
                     "variance changes. For Homework 7, robust standard errors are all I'm expecting on problem 4. "
                     "You don't need weighted least squares. If you do it anyway, great, but it's not worth extra "
                     "points."),
            (7, "S", "If we log the outcome, do the coefficients mean something different?"),
            (7, "P", "They do. After a log, a coefficient is roughly a percent change in the outcome for a one unit "
                     "change in the predictor, at least when it's small. We'll practice that next week. A log is "
                     "nice when the outcome is positive and skewed, like income or prices. Exam scores are bounded, "
                     "so for this data I'd reach for robust standard errors first."),
            (8, "P", "The next slide is leverage and Cook's distance. I'm skipping it today. We'll do it properly "
                     "on Wednesday, and it's not on Homework 7. So, skip."),
            (9, "P", "Last thing. Homework 7 is due Thursday at 11:59 PM, one PDF rendered from your notebook, like "
                     "it says here. Problems 3 and 4 use exactly what we did today. And remember, Quiz 4 is next "
                     "Monday now, not this Wednesday. In problem 3 you fit an interaction model and interpret every "
                     "coefficient in a sentence, and problem 4 is the residual diagnosis. Please label your plots, "
                     "the graders will thank you. On Wednesday we'll do leverage and outliers."),
            (10, "P", "Any questions? The slide shows the regular office hours. Again, this week Mina is Wednesday 2 "
                      "to 4, not Tuesday. Mine are the same as always, Thursday 10 to 11. If interactions still feel "
                      "fuzzy, go to Mina's hours on Wednesday. It's the day before the homework is due, so it's a "
                      "good time. Okay, see you Wednesday."),
        ],
    ),
    dict(
        id="cs152", course="CS 152: Data Structures and Systems", code="CS 152", session="Section 6",
        instructor="Priya Lindqvist (TF)", speaker="Priya Lindqvist",
        start="2026-11-02T16:00:00-05:00", end="2026-11-02T16:50:00-05:00",
        title="Section 6: Resizing, Tombstones, and Benchmarking", vtt="zoom2", theme="cs", seed=152,
        slides=[
            dict(kind="title", title="Section 6: Resizing, Tombstones, and Benchmarking",
                 sub=["PS5 + Project Milestone 2", "Priya Lindqvist (TF)", "Monday, November 2, 2026"], keys=[]),
            dict(kind="slots", title="Tombstones, one more time",
                 slots=["pear", "", "", "", "", "†", "lime", "date"], caption="after delete(\"fig\")",
                 probe="lookup(\"lime\"): slot 5 (tombstone, keep going) → slot 6, found",
                 keys=[KP("After deleting fig, slot 5 holds a tombstone; lookup(lime) probes past it and finds "
                          "lime in slot 6", "slide", [["tombstone"], [r"lime|keep (going|probing)|continue(s)? (probing|past|to probe)|probes? past|skip"]]),
                       KP("An empty slot would stop the lookup and wrongly report lime missing", "speech",
                          [[r"empty"], [r"stop|missing|not (in|there|found)|isn.t (in|there)|break|broke"]],
                          quote="If slot 5 were just empty, you'd stop and say lime isn't in the table"),
                       KP("Student question: insert may reuse a tombstone slot only after checking the rest of the "
                          "probe chain, or the same key ends up in the table twice", "speech",
                          [["reuse"], [r"twice|duplicate|already in|further along|rest of the (probe )?chain|"
                                       r"whole (probe )?chain|(keep|continue) probing|end of the (probe )?chain"]],
                          quote="only after you've checked the rest of the probe chain")]),
            dict(kind="bullets", title="When should you resize?",
                 bullets=["Track the load factor α = n / m after every insert",
                          "Past the threshold: double m and rehash every key",
                          "CPython’s dict resizes at α ≈ 3/4",
                          "PS5 starter code: threshold = 0.7"],
                 keys=[KP("Track the load factor alpha = n/m after every insert", "slide", [["load factor"]]),
                       KP("Past the threshold, double m and rehash every key", "slide",
                          [["doubl"], [r"rehash|reinsert|re-insert"]]),
                       KP("PS5 starter code fixes the threshold at 0.7", "slide", [[r"0\.7\b|\.7\b"]]),
                       KP("Student question: doubling makes inserts amortized O(1); adding a fixed 100 slots "
                          "makes total copying grow like n squared", "speech",
                          [["amortized"], [r"n squared|n\^?2|n²|quadratic|fixed|hundred|\b100\b"]],
                          quote="doubling is what makes inserts amortized O(1)")]),
            dict(kind="bullets", title="Deletion and resizing",
                 bullets=["delete() leaves a tombstone; insert() may reuse it",
                          "Lookups probe past tombstones; they stop only at an empty slot",
                          "Resizing drops every tombstone: only live keys are reinserted"],
                 keys=[KP("Lookups probe past tombstones and stop only at an empty slot", "slide",
                          [["tombstone"], ["empty"]]),
                       KP("Resizing drops every tombstone: only live keys are reinserted", "slide",
                          [["live keys|only .{0,20}keys"], [r"reinsert|re-insert|rehash|drop|disappear|remov|clear"]]),
                       KP("PS5 hint: count tombstones as occupied when deciding to resize; a hidden test "
                          "inserts and deletes 10,000 times and times out otherwise", "speech",
                          [["tombstone"], [r"occupied|count (them|tombstones)|counting (them|tombstones)|"
                                           r"tombstones (count|are counted|as occupied|toward)|include tombstones|"
                                           r"including tombstones|counted as|as full|toward(s)? the load"]],
                          quote="count tombstones as occupied slots, not just live keys"),
                       KP("A hidden PS5 test inserts and deletes 10,000 times in a loop", "speech",
                          [[r"hidden.{0,12}test"]], quote="There's a hidden test on PS5 that does exactly this")]),
            dict(kind="lines", title="Expected probes for a lookup that misses",
                 note="Textbook estimates: linear probing ½(1 + 1/(1 − α)²); chaining 1 + α",
                 keys=[KP("Linear probing explodes near full: about 50 probes per miss at load factor 0.9", "image",
                          [[r"\b50(\.5)?\b"], [r"0\.9\b|90 ?%|90 percent|\.9\b"]]),
                       KP("At load factor 0.75, a linear-probing miss averages about 8.5 probes", "image",
                          [[r"8\.5\b"]]),
                       KP("Chaining stays flat, under 2 probes per miss even at 0.9", "image",
                          [["chain"], [r"1\.9\b|under 2\b|below 2\b|less than 2\b|fewer than 2\b|about 2\b|~ ?2\b|\b2 probes"]]),
                       KP("If your measured PS5 curve doesn't bend up like this, suspect your hash function", "speech",
                          [["hash function"]], quote="it's usually your hash function")]),
            dict(kind="bullets", title="Benchmarking for Milestone 2",
                 bullets=["Compare against a plain Python dict at ≥ 2 workload sizes",
                          "Time with time.perf_counter(); repeat 5×, report the median",
                          "Time the operations, not the setup: build keys before the timer"],
                 keys=[KP("Compare against a plain Python dict at 2 or more workload sizes", "slide",
                          [[r"dict"], [r"two|\b2\b|sizes"]]),
                       KP("Use time.perf_counter, repeat 5 times and report the median", "slide",
                          [["perf_?counter|perf counter"], ["median"]]),
                       KP("Story: a team looked 40 times slower than dict because it generated random strings "
                          "inside the timed loop", "speech", [[r"\b40\b|forty|random (string|key|number)"]],
                          quote="their table was 40 times slower than dict"),
                       KP("Being within 10x of dict at a million keys is a good result; nobody loses points for "
                          "being slower than dict", "speech",
                          [[r"10 ?x|10 ?×|10 times|ten times|within 10|order of magnitude"]],
                          quote="If you're within 10 times of dict at a million keys, that's a good result"),
                       KP("Student question: get and put must be benchmarked; benchmarking delete is optional",
                          "speech", [["delete"], ["optional|not required|don.t have to|not needed"]],
                          quote="but benchmarking it is optional")]),
            dict(kind="bullets", title="Milestone 2 checklist (due Sun Nov 8, 11:59 pm)",
                 bullets=["Repo link: working get / put / delete (no persistence yet)",
                          "Two-page write-up with the dict benchmark",
                          "One paragraph on how you split the work",
                          "One submission per team"],
                 keys=[KP("Repo with working get/put/delete, no persistence yet", "slide",
                          [["persist|get.{0,10}put"]]),
                       KP("Two-page write-up with the dict benchmark, plus a work-split paragraph", "slide",
                          [["two-page|two page|2-page|2 page"]])]),
            dict(kind="bullets", title="PS5: before Friday",
                 bullets=["Due Fri Nov 6, 5:00 pm: code + experiment write-up, submitted together",
                          "Run the local tests before you submit",
                          "3 late days per semester (24 h each)"],
                 keys=[KP("3 late days per semester, 24 hours each", "slide", [["late day"]])]),
        ],
        corrections=[dict(slide=3, slide_said="CPython's dict resizes at α ≈ 3/4",
                          lecturer_said="CPython's dict resizes at about two-thirds full; 0.75 is Java's HashMap default",
                          quote="CPython's dict resizes at about two-thirds full, not three quarters",
                          right=[[r"2/3|two.thirds|0?\.6[67]|6[67] ?%"]],
                          wrong=[[r"python|dict"], [r"3/4|three.quarters?|0?\.75|75 ?%"]], allow=r"java")],
        announcements=[
            dict(slide=8, text="PS5 autograder opens Wed Nov 4 at noon; 10 submissions per day", source="speech only",
                 when="2026-11-04T12:00:00-05:00", match=[["autograder"], [r"wed|nov(ember)?\.? 4\b|11/4\b"]],
                 quote="The autograder for PS5 opens this Wednesday at noon"),
            dict(slide=7, text="Project Milestone 2 due Sun Nov 8, 11:59 pm (one submission per team)",
                 when="2026-11-08T23:59:00-05:00", source="slide + speech",
                 match=[["milestone"], [r"nov(ember)?\.? 8\b|11/8\b|sun"]],
                 quote="It's due Sunday, November 8th at 11:59 PM"),
            dict(slide=8, text="PS5 due Fri Nov 6, 5:00 pm (code + experiment write-up together)",
                 when="2026-11-06T17:00:00-05:00", source="slide + speech",
                 match=[[r"ps ?5|problem set 5"], [r"nov(ember)?\.? 6\b|11/6\b|fri"]],
                 quote="PS5 is due Friday at 5 PM"),
        ],
        script=[
            (1, "P", "Hi everyone, welcome to section. Let's give it one more minute for people coming from the "
                     "other building. Okay. So this is Section 6, and today is all about Problem Set 5 and Milestone "
                     "2, because, uh, they're due two days apart and they use the same ideas. We'll do tombstones "
                     "one more time, then resizing, and then I want to spend real time on how to benchmark your "
                     "key-value store, because last year that's where teams lost the most points on the "
                     "milestone. Bring up questions whenever, this is section, it's supposed to be loose. Quick poll, "
                     "who's started PS5? Okay, about half. Who's started the milestone? Fewer hands. That's fine, "
                     "that's why we're here. Also, the room after us has a lab, so we have to be out at 4:50 sharp."),
            (2, "P", "Okay. Tombstones, one more time, because this is the part of PS5 people get wrong. This is "
                     "the same little table from Thursday's lecture, 8 slots. Fig was in slot 5, and we deleted it, "
                     "so now slot 5 holds a tombstone instead of being empty. Lime had collided with fig, so it "
                     "lives in slot 6. So, look up lime. You hash to 5, you see the tombstone, and you keep going, "
                     "because a tombstone means something used to be here, keep probing. Then you find lime in slot "
                     "6. If slot 5 were just empty, you'd stop and say lime isn't in the table, which is exactly the "
                     "bug from lecture. Draw this on paper for your own table before you write any code. Seriously, "
                     "five minutes with a pencil saves an hour of print statements."),
            (2, "S", "What happens if you insert a new key that hashes to 5?"),
            (2, "P", "You can reuse the tombstone slot, but only after you've checked the rest of the probe chain "
                     "to make sure the key isn't already in the table further along. Otherwise you end up with the "
                     "same key in the table twice, and that's a really annoying bug to find."),
            (3, "P", "Next. When do you resize? You track the load factor, alpha, which is n over m, after every "
                     "insert. When it passes your threshold, you double m and rehash every key, because every key's "
                     "slot depends on m. And, oh, wait, that third bullet is wrong. Sorry. I mixed up my languages "
                     "when I made this slide. CPython's dict resizes at about two-thirds full, not three quarters. "
                     "Three quarters, 0.75, is the default for Java's HashMap. Professor Vasquez-Hale said this in "
                     "lecture, so don't let my slide confuse you. For PS5, the threshold is given to you in the "
                     "starter code, it's 0.7, so you don't have to pick one. You can play with it in your own copy, but "
                     "the tests assume 0.7."),
            (3, "S", "Why do we double the table instead of just adding, like, a hundred slots each time?"),
            (3, "P", "Because doubling is what makes inserts amortized O(1). If you add a fixed hundred slots, you "
                     "resize every hundred inserts, and the total copying grows like n squared. With doubling, the "
                     "total copying stays under 2n. Same argument as dynamic arrays on PS4. It's also why Python lists "
                     "grow by a fraction of their size, not by a fixed amount."),
            (4, "P", "Okay, now the one that bites people every single year. When you decide whether to resize, "
                     "count tombstones as occupied slots, not just live keys. Think about it. If you only count live "
                     "keys, a table that does a lot of deletes never looks full, so it never resizes, and it slowly "
                     "fills up with tombstones. And then every lookup for a missing key walks the whole table. "
                     "There's a hidden test on PS5 that does exactly this. It inserts and deletes 10,000 times in a "
                     "loop, and if you don't count tombstones, it times out. So, count them. And the good news is on "
                     "the slide. When you do resize, the tombstones disappear, because you only reinsert the live "
                     "keys."),
            (4, "S", "So for the experiment part, should the load factor we report include tombstones too?"),
            (4, "P", "For deciding when to resize, yes, count them. For the experiment, report live keys over "
                     "slots, and say so in your write-up. Just be explicit about which one you mean."),
            (5, "P", "This is the picture I want you to keep in your head. It's the expected number of probes for a "
                     "lookup that misses, as the load factor goes up, for chaining and for linear probing. These are "
                     "the textbook estimates, not our measurements. Look at how flat one line stays, and then look "
                     "at what the other one does at the right edge. That's the whole reason we resize long before "
                     "the table is full. On PS5 you measure this yourself for linear probing, so you should get a curve "
                     "shaped like one of these."),
            (5, "S", "Why does linear probing get so much worse than chaining?"),
            (5, "P", "Primary clustering. Runs of filled slots grow and merge, so a miss has to walk all the way to "
                     "the end of a long run. With chaining, a miss only walks one bucket's list. It's also roughly "
                     "what your PS5 experiment should show. If your measured curve doesn't bend up like this one, "
                     "something is off, and it's usually your hash function."),
            (6, "P", "Okay, Milestone 2. Your write-up has to compare your store against a plain Python dict for at "
                     "least two workload sizes. Three rules on the slide. Use time.perf_counter, not time.time. "
                     "Repeat each measurement 5 times and report the median. And time the operations, not the "
                     "setup, so build your keys before you start the timer. Here's a story. Last year, one team "
                     "reported that their table was 40 times slower than dict. It turned out they were generating "
                     "random strings inside the timed loop, so most of what they measured was the random number "
                     "generator. And honestly, you will be slower than dict. Dict is written in C. If you're within "
                     "10 times of dict at a million keys, that's a good result, and nobody loses points for being "
                     "slower than dict. Also, run the benchmark on a quiet machine. Close the browser with a hundred tabs. "
                     "And put the machine details in the write-up, the CPU and the Python version, so we can make "
                     "sense of your numbers."),
            (6, "S", "Do we have to benchmark delete too?"),
            (6, "P", "Get and put are required in the benchmark. Delete has to work in the prototype, but "
                     "benchmarking it is optional. If you do benchmark it, run it after a big batch of inserts, so the "
                     "tombstones actually show up."),
            (7, "P", "Here's the checklist, same as the email I sent this morning. A repo link with a working get, "
                     "put and delete, no persistence yet. The two-page write-up with the benchmark. And one "
                     "paragraph on how you split the work. If someone on your team has been quiet, the split paragraph is "
                     "where we notice, so be honest. It's not graded on how even the split is. It's due Sunday, November 8th at 11:59 PM, and only one "
                     "person per team submits."),
            (8, "P", "Last slide. PS5 is due Friday at 5 PM, code and experiment write-up together. And one new "
                     "thing that's not on the slide. The autograder for PS5 opens this Wednesday at noon. You get "
                     "10 submissions a day, so please don't use it as your debugger, run the local tests first. And if "
                     "you use a late day, you don't need to email anyone, the portal tracks it. "
                     "Okay, that's it. Go get some sleep, and, um, I'll see some of you at office hours."),
        ],
    ),
]

# ======================================================================================
# Transcript: script -> timed cues -> WebVTT
# ======================================================================================
MAX_CUE_WORDS = 15


def split_cues(text):
    """Break one speaker turn into caption-sized cues, like auto-captioning does:
    whole sentences where they fit, long sentences cut into even pieces."""
    cues, cur = [], []
    for sent in re.split(r"(?<=[.?!])\s+", text.strip()):
        words = sent.split()
        if len(words) > MAX_CUE_WORDS:
            if cur:
                cues.append(" ".join(cur))
                cur = []
            parts = -(-len(words) // MAX_CUE_WORDS)
            size = -(-len(words) // parts)
            cues += [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
        elif len(cur) + len(words) > MAX_CUE_WORDS:
            cues.append(" ".join(cur))
            cur = words
        else:
            cur += words
    if cur:
        cues.append(" ".join(cur))
    return cues


def timed_cues(cls):
    rng = random.Random(cls["seed"])
    t, prev, cues = 2.0 + rng.random() * 2, None, []
    for slide, who, text in cls["script"]:
        if who == "PAUSE":
            t += text
            continue
        if prev is not None and slide != prev:
            t += rng.uniform(1.2, 2.8)                   # clicking to the next slide
        speaker = cls["speaker"] if who == "P" else "Student"
        rate = 2.55 if who == "P" else 2.3                # words per second
        for chunk in split_cues(text):
            dur = len(chunk.split()) / (rate * rng.uniform(0.88, 1.12))
            cues.append({"start": round(t, 3), "end": round(t + dur, 3), "speaker": speaker, "text": chunk,
                         "slide": slide})
            t += dur + rng.uniform(0.12, 0.5)
        t += rng.uniform(0.3, 1.0)                        # turn boundary
        prev = slide
    return cues


def ts(sec):
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def write_vtt(cls, cues, path):
    """Three real-world flavours: Zoom (numbered, 'Name: text'), Teams-like (<v Name> voice tags,
    header metadata, NOTE block, cue settings) and Zoom with long cues wrapped onto two lines."""
    out = ["WEBVTT", ""]
    if cls["vtt"] == "teams":
        out = ["WEBVTT", "Kind: captions", "Language: en-US", "",
               "NOTE", "Auto-generated captions from the lecture capture. Speaker names come from the roster.", ""]
    for i, c in enumerate(cues, 1):
        timing = f"{ts(c['start'])} --> {ts(c['end'])}"
        if cls["vtt"] == "teams":
            out += [f"{timing} align:start position:10%", f"<v {c['speaker']}>{c['text']}</v>", ""]
        elif cls["vtt"] == "zoom2" and len(c["text"].split()) > 9:
            words = f"{c['speaker']}: {c['text']}".split()
            half = len(words) // 2
            out += [str(i), timing, " ".join(words[:half]), " ".join(words[half:]), ""]
        else:
            out += [str(i), timing, f"{c['speaker']}: {c['text']}", ""]
    path.write_text("\n".join(out))


# ======================================================================================
# Slides
# ======================================================================================
THEMES = {
    "econ": dict(bg=(255, 255, 255), title=(22, 42, 84), text=(38, 40, 46), accent=(0, 128, 128),
                 muted=(118, 124, 136), band=(0, 128, 128), tfont=(AVENIR, 2), bfont=(AVENIR, 7)),
    "stat": dict(bg=(251, 249, 243), title=(27, 74, 48), text=(32, 32, 30), accent=(176, 118, 32),
                 muted=(120, 116, 104), band=(27, 74, 48), tfont=(GEORGIA_B, 0), bfont=(HELV, 0)),
    "cs": dict(bg=(255, 255, 255), title=(150, 22, 42), text=(34, 34, 38), accent=(150, 22, 42),
               muted=(120, 120, 128), band=(150, 22, 42), tfont=(HELV, 1), bfont=(HELV, 0)),
}


def font(spec, size):
    path, index = spec
    return ImageFont.truetype(path, size, index=index)


def runs(text):
    """Parse a tiny markup: x_{1} is a subscript, x^{2} a superscript. Returns [(text, 'n'|'sub'|'sup')]."""
    out, i = [], 0
    for m in re.finditer(r"([_^])\{([^}]*)\}", text):
        if m.start() > i:
            out.append((text[i:m.start()], "n"))
        out.append((m.group(2), "sub" if m.group(1) == "_" else "sup"))
        i = m.end()
    if i < len(text):
        out.append((text[i:], "n"))
    return out


def plain(text):
    """What a reader sees: x_{1} -> x1, e^{2} -> e^2."""
    return re.sub(r"\^\{([^}]*)\}", r"^\1", re.sub(r"_\{([^}]*)\}", r"\1", text))


def rich_width(text, spec, size):
    f, small = font(spec, size), font(spec, int(size * 0.62))
    return sum((f if kind == "n" else small).getlength(s) for s, kind in runs(text))


def draw_rich(dr, xy, text, spec, size, fill):
    x, y = xy
    f, small = font(spec, size), font(spec, int(size * 0.62))
    for s, kind in runs(text):
        if kind == "n":
            dr.text((x, y), s, font=f, fill=fill)
            x += f.getlength(s)
        else:
            dy = size * 0.48 if kind == "sub" else -size * 0.08
            dr.text((x, y + dy), s, font=small, fill=fill)
            x += small.getlength(s)
    return x


def wrap(text, spec, size, max_w):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if rich_width(trial, spec, size) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def base(cls, n, total, th, title):
    img = Image.new("RGB", (W, H), th["bg"])
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 10], fill=th["band"])
    if title is not None:
        y = 46
        for line in wrap(title, th["tfont"], 44, W - 150):
            draw_rich(dr, (72, y), line, th["tfont"], 44, th["title"])
            y += 56
        dr.rectangle([72, y + 10, 72 + 96, y + 15], fill=th["accent"])
    foot = font(th["bfont"], 17)
    dr.text((72, H - 40), f"{cls['code']} · {cls['session']}", font=foot, fill=th["muted"])
    label = str(n)
    dr.text((W - 72 - foot.getlength(label), H - 40), label, font=foot, fill=th["muted"])
    return img, dr


def bullets_at(dr, th, items, y, size=31, x=104, max_w=W - 210, gap=22):
    for b in items:
        lines = wrap(b, th["bfont"], size, max_w)
        dr.ellipse([x - 30, y + size * 0.4, x - 19, y + size * 0.4 + 11], fill=th["accent"])
        for line in lines:
            draw_rich(dr, (x, y), line, th["bfont"], size, th["text"])
            y += int(size * 1.32)
        y += gap
    return y


def render_title(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, None)
    dr.rectangle([0, 0, 22, H], fill=th["band"])
    head = font(th["bfont"], 26)
    dr.text((90, 170), cls["course"].upper(), font=head, fill=th["accent"])
    y = 222
    for line in wrap(s["title"], th["tfont"], 56, W - 220):
        draw_rich(dr, (90, y), line, th["tfont"], 56, th["title"])
        y += 70
    y += 26
    sub = font(th["bfont"], 28)
    for line in s["sub"]:
        dr.text((90, y), line, font=sub, fill=th["text"])
        y += 42
    return img


def render_bullets(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    bullets_at(dr, th, s["bullets"], 196)
    return img


def render_formula(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    spec = (ARIAL, 0)
    size = 38
    while rich_width(s["formula"], spec, size) > W - 220:
        size -= 2
    wbox = rich_width(s["formula"], spec, size) + 60
    dr.rounded_rectangle([82, 192, 82 + wbox, 192 + size + 54], radius=12, fill=(242, 244, 247),
                         outline=th["accent"], width=2)
    draw_rich(dr, (112, 214), s["formula"], spec, size, th["title"])
    bullets_at(dr, th, s["bullets"], 192 + size + 100, size=29)
    return img


def render_activity(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    y = bullets_at(dr, th, s["bullets"], 200, size=32)
    f = font(th["tfont"], 30)
    wb = f.getlength(s["box"]) + 60
    dr.rounded_rectangle([92, y + 30, 92 + wb, y + 100], radius=35, fill=th["accent"])
    dr.text((122, y + 47), s["box"], font=f, fill=(255, 255, 255))
    return img


def axis_font(th, size=18):
    return font(th["bfont"], size)


def render_bars(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    x0, y0, x1, y1 = 170, 190, 1110, 590            # plot area
    f, fb = axis_font(th, 18), font(th["tfont"], 30)
    for v in range(0, 101, 20):
        y = y1 - (y1 - y0) * v / 100
        dr.line([x0, y, x1, y], fill=(222, 226, 232), width=1)
        lab = f"{v}%"
        dr.text((x0 - 14 - f.getlength(lab), y - 11), lab, font=f, fill=th["muted"])
    dr.line([x0, y0, x0, y1], fill=th["muted"], width=2)
    dr.line([x0, y1, x1, y1], fill=th["muted"], width=2)
    yl = Image.new("RGBA", (360, 30), (0, 0, 0, 0))
    ImageDraw.Draw(yl).text((0, 2), s["ylabel"], font=f, fill=th["muted"])
    img.paste(yl.rotate(90, expand=True), (96, 230), yl.rotate(90, expand=True))
    colors = [(170, 178, 190), th["accent"]]
    bw = 210
    for i, (label, sub, value) in enumerate(s["bars"]):
        cx = x0 + (x1 - x0) * (i + 1) / (len(s["bars"]) + 1)
        top = y1 - (y1 - y0) * value / 100
        dr.rectangle([cx - bw / 2, top, cx + bw / 2, y1], fill=colors[i])
        val = f"{value}%"
        dr.text((cx - fb.getlength(val) / 2, top - 42), val, font=fb, fill=th["title"])
        fl = font(th["bfont"], 22)
        dr.text((cx - fl.getlength(label) / 2, y1 + 12), label, font=fl, fill=th["text"])
        dr.text((cx - f.getlength(sub) / 2, y1 + 42), sub, font=f, fill=th["muted"])
    fn = font(th["bfont"], 16)
    dr.text((W - 72 - fn.getlength(s["note"]), H - 72), s["note"], font=fn, fill=th["muted"])
    return img


def render_table(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    f, fb = font(th["bfont"], 30), font((HELV, 1), 30)
    cols, x, y = [140, 560, 800], 0, 210
    for r, row in enumerate(s["rows"]):
        if r == 0:
            dr.rectangle([120, y - 12, 1040, y + 46], fill=(232, 238, 232))
        for c, cell in enumerate(row):
            ff = fb if r == 0 else f
            xx = 140 + [0, 420, 660][c]
            dr.text((xx, y), cell, font=ff, fill=th["title"] if r == 0 else th["text"])
        y += 66
        dr.line([120, y - 12, 1040, y - 12], fill=(210, 214, 206), width=1)
    dr.text((140, y + 20), s["note"], font=font(th["bfont"], 28), fill=th["text"])
    return img


def render_scatter(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    rng = np.random.default_rng(cls["seed"])
    fitted = rng.uniform(56, 97, 212)
    resid = rng.normal(0, 1, 212) * (0.6 + 0.30 * (fitted - 56))      # spread grows with the fitted value
    x0, y0, x1, y1 = 180, 180, 1100, 600
    f = axis_font(th, 18)
    xs = lambda v: x0 + (x1 - x0) * (v - 50) / 50                    # noqa: E731
    ys = lambda v: (y0 + y1) / 2 - (y1 - y0) / 2 * v / 30            # noqa: E731
    for v in range(-30, 31, 10):
        dr.line([x0, ys(v), x1, ys(v)], fill=(228, 226, 218), width=1)
        lab = f"{v:+d}" if v else "0"
        dr.text((x0 - 12 - f.getlength(lab), ys(v) - 11), lab, font=f, fill=th["muted"])
    for v in range(50, 101, 10):
        dr.text((xs(v) - f.getlength(str(v)) / 2, y1 + 10), str(v), font=f, fill=th["muted"])
    dr.line([x0, y0, x0, y1], fill=th["muted"], width=2)
    dr.line([x0, y1, x1, y1], fill=th["muted"], width=2)
    for xx in range(x0, x1, 18):                                     # dashed zero line
        dr.line([xx, ys(0), xx + 9, ys(0)], fill=(190, 60, 50), width=2)
    for fv, rv in zip(fitted, np.clip(resid, -29, 29)):
        cx, cy = xs(fv), ys(rv)
        dr.ellipse([cx - 4.5, cy - 4.5, cx + 4.5, cy + 4.5], fill=(40, 96, 70), outline=(255, 255, 255))
    fl = font(th["bfont"], 21)
    dr.text(((x0 + x1) / 2 - fl.getlength("Fitted values") / 2, y1 + 40), "Fitted values", font=fl, fill=th["text"])
    yl = Image.new("RGBA", (200, 30), (0, 0, 0, 0))
    ImageDraw.Draw(yl).text((0, 2), "Residuals", font=fl, fill=th["text"])
    img.paste(yl.rotate(90, expand=True), (108, 330), yl.rotate(90, expand=True))
    fn = font(th["bfont"], 16)
    dr.text((W - 72 - fn.getlength(s["note"]), H - 72), s["note"], font=fn, fill=th["muted"])
    return img


def probes_lp(a):
    return 0.5 * (1 + 1 / (1 - a) ** 2)


def render_lines(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    x0, y0, x1, y1 = 170, 170, 1000, 590
    f = axis_font(th, 18)
    xs = lambda a: x0 + (x1 - x0) * a                                # noqa: E731
    ys = lambda p: y1 - (y1 - y0) * p / 55                           # noqa: E731
    for p in range(0, 56, 10):
        dr.line([x0, ys(p), x1, ys(p)], fill=(230, 230, 234), width=1)
        dr.text((x0 - 12 - f.getlength(str(p)), ys(p) - 11), str(p), font=f, fill=th["muted"])
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        lab = f"{a:g}"
        dr.text((xs(a) - f.getlength(lab) / 2, y1 + 10), lab, font=f, fill=th["muted"])
    dr.line([x0, y0, x0, y1], fill=th["muted"], width=2)
    dr.line([x0, y1, x1, y1], fill=th["muted"], width=2)
    grid = [i / 100 for i in range(5, 91)]
    series = [("Linear probing", probes_lp, th["accent"]), ("Separate chaining", lambda a: 1 + a, (60, 90, 160))]
    fb = font((HELV, 1), 20)
    for name, fn_, color in series:
        dr.line([(xs(a), ys(fn_(a))) for a in grid], fill=color, width=4)
        for a in (0.5, 0.75, 0.9):
            v = fn_(a)
            cx, cy = xs(a), ys(v)
            dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=color)
            lab = f"{v:.2f}".rstrip("0").rstrip(".") if name.startswith("Sep") else f"{v:.1f}"
            if name.startswith("Lin"):
                dr.text((cx - fb.getlength(lab) - 14, cy - 40), lab, font=fb, fill=color)
            else:
                dr.text((cx + 12, cy - 32), lab, font=fb, fill=color)
    lx, ly = 1020, 230
    for name, _, color in series:
        dr.line([lx, ly + 12, lx + 36, ly + 12], fill=color, width=5)
        dr.text((lx + 46, ly), name, font=font(th["bfont"], 19), fill=th["text"])
        ly += 40
    fl = font(th["bfont"], 21)
    lab = "Load factor α = n / m"
    dr.text(((x0 + x1) / 2 - fl.getlength(lab) / 2, y1 + 38), lab, font=fl, fill=th["text"])
    yl = Image.new("RGBA", (300, 30), (0, 0, 0, 0))
    ImageDraw.Draw(yl).text((0, 2), "Probes per lookup (miss)", font=fl, fill=th["text"])
    img.paste(yl.rotate(90, expand=True), (100, 240), yl.rotate(90, expand=True))
    fn = font(th["bfont"], 16)
    dr.text((W - 72 - fn.getlength(s["note"]), H - 72), s["note"], font=fn, fill=th["muted"])
    return img


def render_slots(cls, s, n, total, th):
    img, dr = base(cls, n, total, th, s["title"])
    f, fm = font((MENLO, 0), 26), font(th["bfont"], 20)
    bw, x0, y0 = 128, 128, 270
    for i, key in enumerate(s["slots"]):
        x = x0 + i * bw
        fill = (226, 226, 230) if key == "†" else (255, 255, 255)
        dr.rectangle([x, y0, x + bw, y0 + 96], fill=fill, outline=th["text"], width=3)
        dr.text((x + bw / 2 - fm.getlength(str(i)) / 2, y0 - 34), str(i), font=fm, fill=th["muted"])
        if key == "†":
            ft, fs = font((ARIAL, 0), 40), font(th["bfont"], 18)
            dr.text((x + bw / 2 - ft.getlength("†") / 2, y0 + 8), "†", font=ft, fill=(90, 90, 98))
            dr.text((x + bw / 2 - fs.getlength("tombstone") / 2, y0 + 62), "tombstone", font=fs, fill=(90, 90, 98))
        elif key:
            dr.text((x + bw / 2 - f.getlength(key) / 2, y0 + 30), key, font=f, fill=th["title"])
    # probe arrow 5 -> 6
    ax, ay = x0 + 5 * bw + bw / 2, y0 + 150
    bx = x0 + 6 * bw + bw / 2
    dr.line([ax, y0 + 96, ax, ay, bx, ay, bx, y0 + 110], fill=th["accent"], width=4)
    dr.polygon([(bx - 10, y0 + 114), (bx + 10, y0 + 114), (bx, y0 + 98)], fill=th["accent"])
    fa = font((ARIAL, 0), 26)
    dr.text((128, 470), s["probe"], font=fa, fill=th["text"])
    dr.text((128, 520), s["caption"], font=font((MENLO, 0), 24), fill=th["muted"])
    return img


RENDER = {"title": render_title, "bullets": render_bullets, "formula": render_formula, "activity": render_activity,
          "bars": render_bars, "table": render_table, "scatter": render_scatter, "lines": render_lines,
          "slots": render_slots}


def on_slide_text(s):
    """The slide's printed text (what a perfect reader would extract), for the answer key."""
    lines = [plain(s["title"])]
    if s["kind"] == "title":
        lines += s["sub"]
    if s.get("formula"):
        lines.append(plain(s["formula"]))
    lines += [plain(b) for b in s.get("bullets", [])]
    if s["kind"] == "activity":
        lines.append(s["box"])
    if s["kind"] == "bars":
        lines += [f"{a} {b}: {v}%" for a, b, v in s["bars"]] + [s["ylabel"], s["note"]]
    if s["kind"] == "table":
        lines += [" | ".join(r) for r in s["rows"]] + [s["note"]]
    if s["kind"] == "lines":
        lines += ["Linear probing: 2.5 at α=0.5, 8.5 at 0.75, 50.5 at 0.9",
                  "Separate chaining: 1.5 at 0.5, 1.75 at 0.75, 1.9 at 0.9", s["note"]]
    if s["kind"] == "scatter":
        lines += ["(scatter of residuals vs fitted values 55-97; the vertical spread grows from about ±3 at the "
                  "left to about ±25 at the right)", s["note"]]
    if s["kind"] == "slots":
        lines += [f"slot {i}: {k or 'empty'}" for i, k in enumerate(s["slots"])] + [s["probe"], s["caption"]]
    return lines


# ======================================================================================
# Answer key
# ======================================================================================
def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%./ ]", " ", text.lower())).strip()


def spoken(cls, slide=None):
    return " ".join(t for s, who, t in cls["script"] if who != "PAUSE" and (slide is None or s == slide))


def verify(cls, cues):
    import sys
    sys.path.insert(0, str(HERE.parent.parent))
    from scoring.class_notes import matches, stated_as_fact   # the grader's own matcher
    for s in cls["slides"]:
        for k in s["keys"]:
            assert matches(k["text"], k["match"]), (cls["id"], k["text"])
    for c in cls["corrections"]:
        assert matches(c["lecturer_said"], c["right"]) and matches(c["slide_said"], c["wrong"]), (cls["id"], c)
        assert not stated_as_fact([c["lecturer_said"]], c["wrong"], c.get("allow")), (cls["id"], c)
        assert stated_as_fact([c["slide_said"]], c["wrong"], c.get("allow")), (cls["id"], c)
    for a in cls["announcements"]:
        assert matches(a["text"], a["match"]), (cls["id"], a["text"])
    n = len(cls["slides"])
    assert 8 <= n <= 10, (cls["id"], n)
    words = len(spoken(cls).split())
    assert 1200 <= words <= 2000, (cls["id"], words)
    assert sorted({c["slide"] for c in cues}) == list(range(1, n + 1)), cls["id"]   # every slide is on screen
    for i, s in enumerate(cls["slides"], 1):
        for k in s["keys"]:
            if "quote" in k:   # quotes must be verbatim in the speech while that slide is up
                assert norm(k["quote"]) in norm(spoken(cls, i)), (cls["id"], i, k["quote"])
            if k["source"] == "image":   # an image-only fact must not be recoverable from the speech
                assert not all(any(re.search(a, norm(spoken(cls))) for a in g) for g in k["match"]), (cls["id"], i)
        if s.get("skipped"):
            assert norm(s["skip_quote"]) in norm(spoken(cls, i)), (cls["id"], i)
    for item in cls["corrections"] + cls["announcements"]:
        assert norm(item["quote"]) in norm(spoken(cls, item["slide"])), (cls["id"], item["quote"])
    # cue timings are increasing and never overlap
    assert all(a["end"] <= b["start"] for a, b in zip(cues, cues[1:])), cls["id"]


def runs_of(cues):
    """Contiguous runs of cues on the same slide: [(slide, start, end), ...]."""
    out = []
    for c in cues:
        if out and out[-1][0] == c["slide"]:
            out[-1][2] = c["end"]
        else:
            out.append([c["slide"], c["start"], c["end"]])
    return out


def ranges(cues):
    out = {}
    for slide, start, end in runs_of(cues):
        out.setdefault(slide, []).append([ts(start), ts(end)])
    return out


def build_key(cls, cues):
    rng = ranges(cues)
    slides = []
    for i, s in enumerate(cls["slides"], 1):
        slides.append({"slide": i, "file": f"Slide{i}.png", "kind": s["kind"], "title": plain(s["title"]),
                       "on_slide": on_slide_text(s), "skipped": bool(s.get("skipped")),
                       "time_ranges": rng[i], "key_points": s["keys"]})
    # A return: a run of cues on a slide that is earlier than a slide already shown.
    returns, shown, prev = [], set(), None
    for slide, start, end in runs_of(cues):
        if shown and slide < max(shown):
            returns.append({"slide": slide, "from_slide": prev, "time_range": [ts(start), ts(end)]})
        shown.add(slide)
        prev = slide
    pts = [dict(k, slide=s["slide"]) for s in slides for k in s["key_points"]]
    return {
        "id": cls["id"], "course": cls["course"], "session": cls["session"], "title": cls["title"],
        "instructor": cls["instructor"], "start": cls["start"], "end": cls["end"],
        "slides_dir": f"{cls['id']}/slides", "transcript": f"{cls['id']}/transcript.vtt",
        "speakers": [cls["speaker"], "Student"], "vtt_flavor": cls["vtt"],
        "n_slides": len(slides), "transcript_words": len(spoken(cls).split()), "n_cues": len(cues),
        "slides": slides,
        "corrections": cls["corrections"],
        "verbal_only": [{k: p[k] for k in ("slide", "text", "match", "quote")} for p in pts if p["source"] == "speech"],
        "vision_only": [{k: p[k] for k in ("slide", "text", "match")} for p in pts if p["source"] == "image"],
        "skipped_slides": [s["slide"] for s in slides if s["skipped"]],
        "skip_quotes": {str(i): s["skip_quote"] for i, s in enumerate(cls["slides"], 1) if s.get("skipped")},
        "announcements": cls["announcements"],
        "returns": returns,
        "cue_slides": [c["slide"] for c in cues],
    }


def main():
    for p in (AVENIR, HELV, GEORGIA_B, ARIAL, MENLO):
        assert Path(p).exists(), f"missing font {p}"
    key = {"student": "Riley Park", "date": "2026-11-02", "timezone": "America/New_York",
           "note": "Everything is fictional. Match groups are regexes over normalized notes text: every group "
                   "must match; an alternative starting with 'c:' is matched against the compact formula form "
                   "(beta->b, no spaces or multiplication signs), e.g. b1+b3x2.",
           "classes": []}
    schedule = {"student": "Riley Park", "date": "2026-11-02", "timezone": "America/New_York", "classes": []}
    for cls in CLASSES:
        cues = timed_cues(cls)
        verify(cls, cues)
        d = HERE / cls["id"]
        (d / "slides").mkdir(parents=True, exist_ok=True)
        th = THEMES[cls["theme"]]
        total = len(cls["slides"])
        for i, s in enumerate(cls["slides"], 1):
            RENDER[s["kind"]](cls, s, i, total, th).save(d / "slides" / f"Slide{i}.png", optimize=True)
        write_vtt(cls, cues, d / "transcript.vtt")
        key["classes"].append(build_key(cls, cues))
        schedule["classes"].append({"id": cls["id"], "course": cls["course"], "session": cls["session"],
                                    "instructor": cls["instructor"], "start": cls["start"], "end": cls["end"],
                                    "slides": f"{cls['id']}/slides", "transcript": f"{cls['id']}/transcript.vtt"})
        k = key["classes"][-1]
        print(f"{cls['id']}: {total} slides, {k['transcript_words']} words, {len(cues)} cues, "
              f"{ts(cues[-1]['end'])} long, skipped {k['skipped_slides']}, returns "
              f"{[(r['from_slide'], r['slide']) for r in k['returns']]}")
    (HERE / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (HERE / "answer_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
