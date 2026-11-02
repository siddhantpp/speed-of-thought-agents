You are helping a student who missed a lecture. Input: `transcript.txt` in this folder (a ~3,600-word transcript of a CS lecture on hash tables).

Produce a study pack and write these files into this folder (exact formats; they are machine-graded):
- `summary.md`: at most 450 words for a student who missed the lecture: an "In one paragraph" overview, one `##` heading per major section with bullets, "Key terms" (term: definition), "Watch out for" (2-3 common mistakes the lecture warned about), "Logistics", and a 3-day "Study plan". Only use facts from the transcript.
- `flashcards.csv`: Anki-importable, no header row, three columns: front, back, tag (use the tag `hash-tables`). About 20-25 cards, mixing why/how/compare questions; backs at most 30 words.
- `cards.json`: `{"kept": [{"front": "...", "back": "...", "source_quote": "6-25 consecutive words copied exactly from transcript.txt that support the answer"}]}` with one entry per flashcard, in the same order.
- `quiz.md`: about 9 multiple-choice questions. Format each as `**N. Question text**` on its own line followed by four lines `- A. ...` to `- D. ...`; after all questions add a line `## Answer key` and then lines like `1. C: one-sentence explanation`.
Plain text only (no LaTeX). Do not ask questions; finish the task.
