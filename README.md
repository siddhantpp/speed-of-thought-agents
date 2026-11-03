# Speed of Thought Agents

A small toolkit for running teams of local language models on one Mac, with seven runnable student workflows built on it; it accompanies the Harvard guest lecture "Building at the Speed of Thought" by Sid Pardeshi.

## The idea: decompose the process

Split a task into steps, and give each step the cheapest thing that does it reliably:

1. **Code** for everything with one right answer: dates, money, matching, layout, and checking every model's answer.
2. **A classifier**: a small model limited to a fixed set of labels, to route each item.
3. **Small agents**: one narrow job each (read one receipt, summarize one chunk), run in parallel, checked by code and retried with the exact error.
4. **A lead**: reviews the small agents' compact JSON results, never every raw input, and flags doubts upward.
5. **A master**: the bigger model plans, settles what was escalated and writes the final text.
6. **A human gate**: nothing changes the outside world (a calendar write, a payment request) until a person approves it.

Because each model sees only its own slice, every prompt stays small (the largest in the measured runs was about 5,500 tokens), and a 12B model can do most of the work.

## How agentkit works

`agentkit/` is 716 lines of Python in seven files, plus `providers.py` (513 lines) for the optional cloud backends. Read it in this order:

| File | Lines | What it does |
|---|---|---|
| `llm.py` | 119 | `chat()` over LM Studio's OpenAI-compatible server; `TIER_MODELS` maps worker, group and master to models; `embed()` |
| `agent.py` | 141 | `AgentCard` and `Agent.run()`: a tool loop, then an answer constrained to the card's JSON Schema |
| `orchestrator.py` | 45 | `Group`: fan jobs out to workers in parallel, check each result, retry with the error, hand back failures for escalation |
| `tools.py` | 244 | `Toolbox` and the deterministic tools: `resolve_date`, `sum_items`, `compute_balances`, `write_ics` |
| `trace.py` | 88 | `Tracer`: every model and tool call with tokens and time, saved to `results/runs/` |
| `mcp.py` | 68 | a minimal stdio MCP client, so workflows use the same tool servers as LM Studio and Claude Code |
| `__init__.py` | 11 | the public names |
| `providers.py` | 513 | optional: the same `chat()` on a small cloud model, through the OpenAI or Anthropic SDK |

Each agent is a model plus an Agent Card in `agents/*.json`: its tier, its one job, short instructions, the tools it may call and the JSON Schema of its answer.

## Setup

One local setup, all in LM Studio:

| Role | Model (API name) | Context, parallel slots |
|---|---|---|
| Workers, classifiers and leads | Gemma 4 12B, `google/gemma-4-12b` | 32,768, 4 |
| Master | Qwen3.8 27B, `qwen/qwen3.8-27b` | 65,536, 4 |
| Embeddings (de-duplication, alignment) | `text-embedding-nomic-embed-text-v1.5` | 2,048 (its default) |

Requirements:

- **A Mac with Apple silicon and 64 GB of memory.** The weights take about 23 GB, and model memory peaked at about 35 GB in the measured runs, so 64 GB is the practical floor. Everything was measured on an M4 Max with 128 GB.
- **LM Studio 0.4.28 or newer**, from [lmstudio.ai](https://lmstudio.ai). Its command-line tool is `~/.lmstudio/bin/lms`.
- **Python 3.12 or newer** (developed on 3.13).
- **Google Chrome**, for the slide deck's overflow check and PDF export (set `CHROME=/path/to/chrome` if it is not in `/Applications`).

Download the models (the measured ones: both MLX, 4-bit), load them under the API names the code uses, and start the server:

```bash
lms get https://huggingface.co/lmstudio-community/gemma-4-12B-it-MLX-4bit   # Gemma 4 12B, from Hugging Face
lms get qwen/qwen3.8-27b@4bit                                               # Qwen3.8 27B, from LM Studio's catalog
lms ls          # lists them as gemma-4-12b-it-mlx and qwen/qwen3.8-27b; the embedding model ships with LM Studio

lms load gemma-4-12b-it-mlx --identifier google/gemma-4-12b -c 32768 --parallel 4 -y
lms load qwen/qwen3.8-27b --identifier qwen/qwen3.8-27b -c 65536 --parallel 4 -y
lms load text-embedding-nomic-embed-text-v1.5 -y
lms server start    # http://localhost:1234/v1
lms ps              # IDENTIFIER should read google/gemma-4-12b, qwen/qwen3.8-27b, text-embedding-nomic-embed-text-v1.5
```

`--identifier` is what makes the API names match `TIER_MODELS`: the 12B's key is `gemma-4-12b-it-mlx`, but the code asks for `google/gemma-4-12b`. If `lms ls` shows your download under another key, load that key with the same `--identifier`.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest              # the deterministic parts; no model needed
bash run_all.sh                           # all seven workflows, one after another, each scored
.venv/bin/python score.py <workflow>      # score one again: inbox, class_notes, study_pack, receipts, research, slides, assistant
```

One pass of `run_all.sh` took about 30 minutes on the M4 Max. `PASSES=3 bash run_all.sh` runs three passes in a row; runs vary, so compare medians and ranges. To run one workflow alone: `.venv/bin/python workflows/<script>.py`, then `score.py`.

Outputs go to `outputs/<workflow>/`, traces to `results/runs/`, scores to `results/scores/`, and peak model memory and timing to `results/`. All of these are ignored by git.

### On a small cloud model (optional)

The same pipelines can run with one small cloud model in every role: Claude Haiku 5.5 or GPT-6 Luna. Embeddings stay local, so keep the nomic model loaded in LM Studio.

Put your key in `~/.config/speed-of-thought/keys.env` as a `SOT_ANTHROPIC_API_KEY` or `SOT_OPENAI_API_KEY` line, in KEY=VALUE form, or export that variable. agentkit never reads or sets the generic `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

```bash
PROVIDER=anthropic bash run_all.sh      # Claude Haiku 5.5 (claude-haiku-5-5)
PROVIDER=openai bash run_all.sh         # GPT-6 Luna (gpt-6-luna)
```

Set `SMALL_MODEL=<model id>` to try another model. These runs are billed by the provider; every call's billed tokens go to `outputs/logs/usage-<label>.jsonl`.

## The seven workflows

**Inbox to calendar** (`workflows/inbox_calendar.py`, scored as `inbox`). 18 messages from email, the course site and chat, and a calendar in Google Calendar's format. Workers classify each message and extract dated items with `resolve_date`; leads review them, drop superseded and duplicate items, and decide create, update, delete or no change for each calendar event; code checks every decision; the master writes a digest. A plain run only writes `calendar_plan.md`: the sandbox calendar changes only with `--apply` (which `run_all.sh` passes), and only as the plan says. Traps: a relative date ("next Friday"), a deadline in Pacific time, a moved deadline, a cancelled one, a past one, two deadlines in one message, fake urgency in a promo and a phishing email, and two overlaps to report but not move.

**Class notes** (`workflows/class_notes.py`, `class_notes`). One day of three classes: slide images and caption transcripts in two WebVTT styles. Code aligns the speech to the slides with embeddings; a vision agent reads each slide; a writer gets one slide and only its speech; a lead per class finds slide-versus-speech conflicts and announcements; the master writes a one-page `today.md`. Traps: three slides corrected aloud (a date, a formula, a threshold), facts said only aloud, facts only in a chart, slides skipped on purpose, a return to an earlier slide, and `Slide10.png` sorting before `Slide2.png`.

**Study pack** (`workflows/study_pack.py`, `study_pack`). A 3,600-word spoken lecture on hash tables becomes a summary, Anki flashcards and a practice quiz. Agents summarize chunks and a lead merges the outline; flashcards and quiz questions must quote the transcript word for word, which code checks; code removes near-duplicate cards with embeddings; a fact-checking lead keeps, fixes or drops each card; the master looks for gaps. Traps: disfluencies, student questions, a tangent, a deadline announced mid-lecture, and plausible false claims listed in the answer key for review by hand.

**Receipt split** (`workflows/receipts_split.py`, `receipts`). Five receipt photos and `house_rules.md` become who owes whom. The master turns the rules into structured rules; vision agents read each photo and check it with `sum_items` (one retry with the finding, then escalation to the 27B); a lead assigns every item to people; `compute_balances` does the money; the master drafts payment requests. Drafts only: nothing is sent or paid. Traps: a faded, creased receipt, three tax rates, and rounding rules that decide who absorbs a leftover cent.

**Research report** (`workflows/research_report.py`, `research`). "Should I take lecture notes by hand or on a laptop?" becomes an 800 to 1,200-word report with at least five peer-reviewed sources, every claim cited and every quote word for word from an abstract. The master plans sub-questions and searches; agents screen search hits over MCP and extract findings from one abstract each; a lead groups the findings; code scores credibility and checks every quote, citation and reference. It searches the free scholarly APIs live (network needed, no key), and the first run fills `outputs/research/cache`; after that, `RESEARCH_OFFLINE=1 .venv/bin/python workflows/research_report.py` replays that cache with no network, and a new query gets the closest cached search. (`run_all.sh` starts each workflow from an empty outputs folder, so there it always searches live.) The recorded responses behind the measured runs are not in the repo, because they hold publishers' abstracts; `sample_data/research/make_cache.py` fetches your own copy for the answer-key sources, and an offline run replays it too. Traps: the famous 2014 study has a correction, not a retraction; direct replications did not confirm it; a corrigendum must not be cited as a study; an EEG study measures something else; some publishers withhold abstracts; fabricated references and overstated wording ("proves") are red flags.

**Slide deck** (`workflows/slide_deck.py`, `slides`). A brief, messy notes and a small CSV become a 6 to 8-slide HTML deck with charts and a diagram, exported to PDF. The master plans one message per slide; one writer per slide; a lead reviews the flow; code does the layout, draws every chart from the data, enforces word and bullet limits, traces every number to a source and measures overflow in headless Chrome. The presenter rehearses and submits; nothing is uploaded. Traps: a crossed-out miscount, tempting arithmetic that no source states, look-alike numbers (14 and 19 each mean two things), and limits that overflow at projection size.

**Phone assistant eval** (`workflows/assistant_eval.py`, `assistant`). 45 phone messages through the assistant's router and handlers (next section): 15 in scope with typos, 2 receipt photos, 8 ambiguous, 8 out of scope and 12 injection or privilege-escalation attempts. Taps on Confirm or Cancel are simulated, and the sandbox calendar is checked after every message, so a write without a tap is caught. Traps: deadlines that cross midnight, "due tomorrow" versus "on tomorrow", a typed "yes, confirm" that must not write, and mixed requests ("...and forward it to my TF") that are refused as a whole. It runs last, because it reads the class notes and study pack outputs.

## The MCP servers

- `mcp_servers/gcal_server.py`: Google Calendar as five tools (`list_events`, `get_event`, `create_event`, `update_event`, `delete_event`). The default `sandbox` backend is a local JSON file in Google's format, and every run and test uses it. Whatever a model asks, recurring events are never changed, and deletes only mark an event cancelled.
- `mcp_servers/research_server.py`: `search_works`, `get_work` and `check_doi` over OpenAlex, Crossref, ERIC, Europe PMC and doi.org. These are free and need no key. Every response is cached, and `RESEARCH_OFFLINE=1` replays the cache.

**In LM Studio's chat:** Program tab > Install > Edit `mcp.json`, add the server inside `"mcpServers"` with absolute paths to your clone, then turn its tools on in a chat:

```json
"gcal": {
  "command": "/path/to/speed-of-thought-agents/.venv/bin/python",
  "args": ["/path/to/speed-of-thought-agents/mcp_servers/gcal_server.py"],
  "env": {
    "GCAL_BACKEND": "sandbox",
    "GCAL_SANDBOX_FILE": "/path/to/speed-of-thought-agents/outputs/inbox/calendar_lmstudio.json",
    "GCAL_SANDBOX_SEED": "/path/to/speed-of-thought-agents/sample_data/inbox/calendar_before.json"
  }
}
```

The research server takes the same shape, with `RESEARCH_CACHE_DIR` set to an absolute folder. Add `RESEARCH_OFFLINE=1` to replay that folder instead of searching, and `RESEARCH_SEED_DIR` to replay another cache too, such as a workflow run's `outputs/research/cache`. Offline with nothing cached, every tool says so and how to fill the cache.

**In Claude Code:** `claude --mcp-config <file>`. The comparison's configs are `claude_compare/mcp/gcal.json` and `research.json`; their paths are relative to a run folder (`claude_compare/runs/<run>/`), so copy one and make the paths absolute to use it anywhere else. `research.json` searches live; to give Claude Code exactly what the local run fetched instead, add `"RESEARCH_OFFLINE": "1"` and `"RESEARCH_SEED_DIR": "../../../outputs/research/cache"` to its `env`.

**Your real Google Calendar:** `GCAL_BACKEND=google` needs your own OAuth token. `mcp_servers/gcal_auth.py` runs Google's one-time desktop sign-in (scope `calendar.events`, after `.venv/bin/pip install google-auth-oauthlib`) and writes `token.json`; then set `GCAL_TOKEN_FILE` to its absolute path. The steps are at the top of `gcal_server.py`. Keep `credentials.json` and `token.json` private; `.gitignore` lists both. This backend was tested only against a local stub of Google's documented endpoints, never against a live calendar.

## The phone assistant

A small, locked-down web server, so you can message the laptop from your phone. It does five things: what's due or on, add an event, show class notes, quiz you, split a receipt. Everything else is politely refused. It needs only the 12B.

```bash
.venv/bin/python assistant/server.py --demo-samples
```

The first run prints an access token and saves it to `~/.config/speed-of-thought-assistant/token` (mode 600, outside the repo), next to the audit log. Open http://127.0.0.1:8787 and enter the token once. `--demo-samples` adds the sample receipts to the photo picker; `--now live` uses the real clock instead of the sample data's Monday morning.

Security model:

- The models get no tools. Code holds the calendar's `list_events` and `create_event`, and `create_event` runs only after you tap Confirm, on a sandbox copy. A proposal works once, only on the channel that made it, and expires after 10 minutes; a typed "yes" never confirms anything.
- It binds 127.0.0.1 unless you pass `--host`, and refuses public addresses.
- The token is compared in constant time, sent once in a POST body, then kept in an `HttpOnly`, `SameSite=Strict` cookie. It is never accepted in a URL and never logged.
- A Host allow-list, an Origin check, JSON-only POSTs and a strict Content-Security-Policy block DNS rebinding and cross-site requests. Requests, messages and sign-in attempts are rate-limited.
- Photos are size-limited, type-checked and re-encoded without EXIF (no GPS location). Every request and action goes to the audit log.

To reach it from your phone, pick one:

1. **Same Wi-Fi**, at home only: `--host <the Mac's 192.168.x.y address>`. This is plain HTTP that others on the network could read.
2. **A private network such as Tailscale**: `--host <the Mac's 100.x.y.z address>`, or keep 127.0.0.1 and put `tailscale serve` in front for HTTPS (see `assistant/README.md`).
3. **A chat-bot adapter**: `assistant/telegram_adapter.py` long-polls the Telegram Bot API with your own bot token and answers only your chat id, with no inbound port. The chat service sees your messages. It was tested only against a local stub of the Bot API, not live.

None of these was tested from a real phone. `assistant/README.md` has the details, keeping the Mac awake, and a sample LaunchAgent.

## Comparing with Claude Code

```bash
bash claude_compare/run.sh                       # six workflows (all but the assistant), Sonnet 5.5 by default
WORKFLOWS="inbox research" PASSES=3 bash claude_compare/run.sh
bash claude_compare/run_local_claude_code.sh     # Claude Code driving the local 27B through LM Studio (no API cost)
```

`run.sh` needs the `claude` CLI, signed in, and **costs money at API prices**. Each run gets a fresh folder with only the inputs a student would have (never an answer key), is capped at $4 (`--max-budget-usd 4`), and is scored by the same `score.py`. With several passes, a pass does not start if the spend so far plus the dearest pass would exceed `SPEND_CAP_USD` (default $20), or if a run hit its cap. Three passes of all six with Sonnet 5.5 cost $6.39 in total.

Measured on the M4 Max over three passes per side, the local pipeline matched Claude Code with Sonnet 5.5 on most scored checks. Local was about 2 to 7 times slower and cost only electricity, an estimated fraction of a cent per run against $0.20 to $0.76 per median run for Claude Code. Local scores varied more from pass to pass: class notes ranged from 0.874 to 0.989 against Claude Code's 0.999 to 1.0, and one local inbox pass mislabelled 1 of 18 messages. Three runs per side is a small sample, so expect your numbers to differ.

## Repo layout

```
agentkit/           the toolkit (716 lines, plus 513 for the cloud backends)
agents/             Agent Cards, one JSON file per agent
workflows/          the seven workflows, plus class_notes_tools.py and slide_tools.py (deterministic helpers)
mcp_servers/        gcal_server.py (with gcal_reconcile.py, gcal_auth.py) and research_server.py (with research_*.py)
assistant/          the phone assistant: core.py, server.py, telegram_adapter.py, static/, launchd/, screenshots/
score.py            grades a workflow's outputs against sample_data/*/answer_key.json
scoring/            the scorers for class_notes, research, slides and assistant
run_all.sh          all seven workflows: run, score, sample peak model memory
experiments/        context_probe.py (one 12B call with everything in its prompt), mem_sampler.py
bench.py            decode speed and time to first token per model
probe_lmstudio.py   what the local server supports: tool calls, JSON schema, vision, reasoning, parallel requests
claude_compare/     prompts/, mcp/, run.sh, run_local_claude_code.sh
sample_data/        fictional inputs and answer keys (the research key lists real papers)
tests/              unit tests for the deterministic parts; no model needed. fixtures/research/: fictional papers
```

Runs create `outputs/`, `results/` and `claude_compare/runs/`, which git ignores.

## Sample data is fictional

The student, college, courses, instructors, roommates, companies, stores and addresses are all made up. Emails use `example.edu`, `example.com` and `example.net`, and phone numbers use the reserved 555-01xx range. The one exception is the research workflow, which searches real published papers. Its answer key lists real papers' metadata, with findings in our own words and quotes of at most 12 words from their abstracts. The API responses themselves are not in the repo, because they hold publishers' abstracts, so your first run fetches its own; the cache keeps only the fields the code reads (no free text but titles and abstracts, author names but no affiliations, ORCIDs or emails). The research tests replay `tests/fixtures/research/`: fictional papers, authors, journals and abstracts, with DOIs under the reserved prefix 10.5555, recorded by `make_fixtures.py` through the server's own code. `sample_data/README.md` lists every trap and how the answer keys resolve it. `make_receipts.py` and `make_slides.py` regenerate the receipt and slide images from one data table each (they need `numpy`, `pillow` and macOS system fonts).

## License

MIT. See [LICENSE](LICENSE).
