You are helping a student, Riley Park, turn a presentation brief, rough notes and a small data table into the slide deck for a 5-minute class presentation.

Inputs in this folder:
- `brief.md`: the course, audience, time limit, slide and word limits, required visuals, and the key messages that must land.
- `notes.md`: Riley's rough notes. Text struck out like ~~this~~ was a mistake; never use it.
- `data.csv`: Riley's survey table, one row per meal plan (label column `plan`).
- `slide_tools.py`: a helper that needs only python3 and the installed Chrome. `python3 slide_tools.py export deck.html deck.pdf` exports a deck to PDF with headless Chrome, one page per slide. `python3 slide_tools.py check deck.html --pdf deck.pdf` reports words, bullets, speaker notes, numbers not found in the inputs, chart values and text that overflows (it renders the deck in Chrome).

Follow the brief exactly. Every slide gets speaker notes: what Riley will say, in Riley's voice. Every number on a slide, in a diagram label or in the speaker notes must appear in brief.md, notes.md or data.csv exactly as written: no new percentages, differences, averages or rounding. Each slide has at most 4 bullets and at most 45 words of visible text outside charts and diagrams (the title, bullets, captions and any other visible text all count). Speaker notes total at most 700 words. No text may overflow its box or the slide.

Write these files into this folder (exact formats; they are machine-graded):
- `deck.html`: one self-contained HTML file with no external fonts, scripts, stylesheets or images. One `<section class="slide">` per slide, in order, each exactly 1280x720 px with `overflow: hidden`, and `@page { size: 1280px 720px; margin: 0 }` with a page break after each slide. Bullets are `<li>` elements. The speaker notes for a slide go inside its section in `<aside class="notes">`, hidden on the slide and in the PDF. Every chart is an inline `<svg class="chart">` drawn from data.csv in which every bar or point has `data-label` (its row's `plan` value), `data-series` (the data.csv column it shows) and `data-value` (the exact number drawn). The diagram is an inline `<svg class="diagram">`. Slide numbers, if you show them, go in an element with class `slide-number`.
- `deck.pdf`: made with `python3 slide_tools.py export deck.html deck.pdf`, one page per slide.

Do not ask questions; make reasonable decisions and finish.
