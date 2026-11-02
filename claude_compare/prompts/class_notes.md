You are helping a student, Riley Park, who went to three classes today, Monday 2026-11-02 (America/New_York). For each class there is the slide deck, exported as one PNG per slide, and the transcript of the lecture recording in WebVTT (timestamps and speaker names).

Inputs in this folder:
- `schedule.json`: today's classes (id, course, session, instructor, start and end time, slide folder, transcript path).
- `<id>/slides/Slide1.png`, `Slide2.png`, ... for each class id (number order: Slide10 comes after Slide9).
- `<id>/transcript.vtt` for each class.

Look at every slide image (some facts are only in a chart or plot) and read every transcript. Lecturers sometimes correct a slide aloud, skip a slide, go back to an earlier slide, give examples or exam and homework hints that are not on the slides, and announce logistics (due dates, and changes to dates, times or rooms) only aloud. Use the lecturer's corrections, never state a slide's wrong value as a fact, and capture what was said only aloud. Resolve spoken dates ("the Tuesday after next", "this Saturday") against the class date.

Write these files into this folder (exact formats; they are machine-graded):
- `<id>.md` for each class (`econ140.md`, `stat121.md`, `cs152.md`): notes to study from: a short summary, slide-by-slide notes including what the lecturer added aloud, corrections (what the slide said and the corrected fact), skipped slides, announcements with dates, and what to review.
- `today.md`: at most 350 words: one line per class; every announcement and date change from today's classes, with its date written like `Fri Nov 20`; the corrections to fix in your notes; what to review tonight.
- `notes.json`: `{"date": "2026-11-02", "classes": [{"id": "econ140", "summary": "...", "slides": [{"slide": 1, "title": "...", "skipped": false, "time_ranges": [["00:00:02.000", "00:00:40.500"]], "points": ["..."], "lecturer_added": [{"point": "...", "quote": "5-30 consecutive words copied exactly from the transcript"}]}], "corrections": [{"slide": 2, "slide_said": "...", "lecturer_said": "...", "quote": "the lecturer's exact words"}], "announcements": [{"text": "...", "due": "2026-11-20T17:00:00-05:00"}], "skipped_slides": [5], "review": ["..."]}]}`, with one entry per class and one entry per slide. `time_ranges` are the stretches of the recording (WebVTT timestamps) while that slide was on screen; a slide the lecturer came back to has more than one range.

Plain text inside JSON (no Markdown, no LaTeX). Do not ask questions; make reasonable decisions and finish.
