You are helping a student, Riley Park, answer a research question with a short evidence report built only on credible sources. Input: `question.md` in this folder (the question and what Riley needs).

Tools: find and read sources ONLY with the `research` MCP server: `search_works` (scholarly search), `get_work` (one work's metadata, abstract, a code-computed credibility block and its APA reference) and `check_doi`. You have no web access. The server caches every response in `cache/` in this folder; leave that folder in place.

Do all of this, then write the files below into this folder:
1. Search for the main evidence on the question: the best-known studies, attempts to replicate them, reviews or meta-analyses, and other factors that matter.
2. Read each candidate with `get_work`. Use only works whose `credibility.vetted` is true (peer-reviewed journal article or review, not retracted) and that have an abstract. Never cite a work you did not fetch, and never cite from memory.
3. Write the report: 800 to 1,200 words in the body (not counting the reference list), at least 5 different sources, a citation like [3] or [2, 5] after every factual sentence. Quotes are optional, but only words copied exactly from the cited source's abstract may go inside double quotes. Say how strong the evidence is, including whether well-known findings held up when others repeated them. No tables.

Output files (exact formats; they are machine-graded):
- `report.md`: Markdown. The body cites sources as [n]. Then a `## References` section with one line per source, numbered `1.` to `N.` in order of first citation, each line the `apa` reference that `get_work` returned (it ends with the https://doi.org/ link).
- `sources.json`: `{"question": "...", "sources": [{"n": 1, "doi": "10....", "openalex_id": "W...", "title": "...", "year": 2020, "venue": "...", "credibility": {"score": 0.0, "tier": "...", "peer_reviewed": true, "vetted": true}, "reference": "the APA line"}]}` with exactly the sources in the reference list, numbered as in the report.
- `claims.json`: `{"claims": [{"source_n": 1, "claim": "one sentence", "quote": "8 to 40 consecutive words copied exactly from source 1's abstract that support the claim"}]}` with at least one claim per cited source.
Do not ask questions; make reasonable decisions and finish.
