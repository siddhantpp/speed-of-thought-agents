"""Build cache/: recorded API responses for the answer-key sources, which offline runs replay.

Not shipped with the repo (the responses hold publishers' abstracts); the tests replay the fictional
tests/fixtures/research instead.

  1. every source in answer_key.json, fetched exactly as get_work and check_doi fetch it (free lookups);
  2. one DOI that does not exist, so "fabricated reference" checks can be tested offline;
  3. optionally, every response from a finished run (--from-run DIR), so a whole workflow run can be
     replayed offline: RESEARCH_OFFLINE=1 .venv/bin/python workflows/research_report.py

Run from the repo root:  .venv/bin/python sample_data/research/make_cache.py [--from-run outputs/research/cache]
Needs the network once; afterwards everything replays from cache/.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "mcp_servers"))
from research_apis import Fetcher, crossref_record, doi_resolves, full_work  # noqa: E402

FAKE_DOI = "10.1177/0956797699999999"   # Psychological Science prefix, but never registered


def main():
    cache = HERE / "cache"
    f = Fetcher(cache_dir=cache, offline=False)
    key = json.loads((HERE / "answer_key.json").read_text())
    for s in key["sources"]:
        w = full_work(f, s["doi"])
        crossref_record(f, s["doi"])
        print(f"{s['key']:<16} abstract from {w.get('abstract_source')}, gaps {w.get('gaps')}")
    full_work(f, next(r["doi"] for r in key["red_flags"] if r["id"] == "corrigendum_as_study"))
    crossref_record(f, FAKE_DOI)
    print(f"{FAKE_DOI} registered: {doi_resolves(f, FAKE_DOI)}")
    if "--from-run" in sys.argv:
        run = Path(sys.argv[sys.argv.index("--from-run") + 1])
        copied = 0
        for p in run.glob("*.json"):
            if not (cache / p.name).exists():
                shutil.copy(p, cache / p.name)
                copied += 1
        print(f"copied {copied} responses from {run}")
    print(f"{len(list(cache.glob('*.json')))} responses in {cache}")


if __name__ == "__main__":
    main()
