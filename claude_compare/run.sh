#!/usr/bin/env bash
# Run the local workflows' tasks through Claude Code (headless) for a cost and quality comparison.
# Each run gets a fresh folder with ONLY the inputs a student would have (never an answer key), then is
# scored by the same scorer as the local run (score.py <wf> <run folder>).
#
# Usage:  bash claude_compare/run.sh [model ...]                    default: claude-sonnet-5-5
#         WORKFLOWS="inbox research" bash claude_compare/run.sh     a subset (default: all six below)
#         PASSES=3 bash claude_compare/run.sh                       three passes, one after another (runs vary)
# Writes  claude_compare/runs/<wf>-<model>[-p<N>]/   inputs, outputs, _claude_result.json, _stderr.txt, _score.txt
#         results/runs/<wf>-<model>[-p<N>].json       cost, time, turns and tokens as Claude Code reports them
#         results/scores/<wf>-<model>[-p<N>].json     the score
# (-p<N> only with PASSES > 1). The workflows of one pass run in parallel; passes run in sequence. A pass is
# not started when the spend so far plus the dearest pass so far would pass SPEND_CAP_USD (default $20), or
# when a run of the last pass hit its $4 cap.
#
# Tools: every run may Read/Write/Edit/Glob/Grep and run python3. inbox adds the Google Calendar MCP server
# (sandbox backend) and research the research MCP server, which searches the free scholarly APIs live and
# caches every response in the run folder, with web search and fetch disallowed. The MCP configs in
# claude_compare/mcp/ use paths relative to the run folder: Claude Code starts stdio servers there.
# Each run is capped at $4 (--max-budget-usd). The phone-assistant eval (assistant) has no Claude Code version.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
DATA="$ROOT/sample_data"
RUNS_OUT="$ROOT/results/runs"
MODELS=("${@:-claude-sonnet-5-5}")
read -r -a MODELS <<< "${MODELS[*]}"
ALL_WORKFLOWS="inbox class_notes study_pack receipts research slides"
read -r -a WFS <<< "${WORKFLOWS:-$ALL_WORKFLOWS}"
BUDGET_USD=4
PASSES="${PASSES:-1}"
SPEND_CAP_USD="${SPEND_CAP_USD:-20}"
BASE_TOOLS="Read Write Edit Glob Grep Bash(python3:*)"
mkdir -p "$RUNS_OUT"

inputs() {  # copy only what a student would have
  local wf="$1" dir="$2" id
  case "$wf" in
    inbox)       cp "$DATA/inbox/messages.jsonl" "$DATA/inbox/calendar_before.json" "$dir/" ;;
    class_notes) cp "$DATA/classes/schedule.json" "$dir/"
                 for id in $(python3 -c 'import json,sys; print(" ".join(c["id"] for c in json.load(open(sys.argv[1]))["classes"]))' \
                             "$DATA/classes/schedule.json"); do   # econ140 stat121 cs152
                   mkdir -p "$dir/$id/slides"
                   cp "$DATA/classes/$id"/slides/Slide*.png "$dir/$id/slides/"
                   cp "$DATA/classes/$id/transcript.vtt" "$dir/$id/"
                 done ;;
    study_pack)  cp "$DATA/lecture/transcript.txt" "$dir/" ;;
    receipts)    cp "$DATA"/receipts/r*.png "$DATA/receipts/house_rules.md" "$dir/" ;;
    research)    cp "$DATA/research/question.md" "$dir/" ;;
    slides)      cp "$DATA/slides/brief.md" "$DATA/slides/notes.md" "$DATA/slides/data.csv" \
                    "$ROOT/workflows/slide_tools.py" "$dir/" ;;
    *)           echo "unknown workflow $wf" >&2; return 1 ;;
  esac
  if [ -n "$(find "$dir" -iname '*answer_key*' -o -iname 'make_*.py')" ]; then
    echo "refusing to run $wf: an answer key or generator was copied into $dir" >&2; return 1
  fi
}

run_one() {
  local wf="$1" model="$2" sfx="$3"
  local dir="$HERE/runs/$wf-$model$sfx"
  local -a opts=(--allowedTools "$BASE_TOOLS")
  case "$wf" in
    inbox)    opts=(--mcp-config "$HERE/mcp/gcal.json" --allowedTools "$BASE_TOOLS mcp__gcal") ;;
    research) opts=(--mcp-config "$HERE/mcp/research.json"
                    --allowedTools "$BASE_TOOLS mcp__research__search_works mcp__research__get_work mcp__research__check_doi"
                    --disallowedTools "WebSearch WebFetch") ;;
  esac
  rm -rf "$dir" && mkdir -p "$dir"
  inputs "$wf" "$dir"
  local listing start end
  listing="$(cd "$dir" && find . -type f | sed 's|^\./||' | sort | tr '\n' ' ')"
  start=$(date +%s)
  (cd "$dir" && claude -p "$(cat "$HERE/prompts/$wf.md")" \
      --model "$model" --output-format json --no-session-persistence --strict-mcp-config \
      --permission-mode acceptEdits "${opts[@]}" \
      --max-budget-usd "$BUDGET_USD" > "$dir/_claude_result.json" 2> "$dir/_stderr.txt") || true
  end=$(date +%s)
  python3 - "$dir/_claude_result.json" "$RUNS_OUT/$wf-$model$sfx.json" "$wf" "$model" "$((end - start))" \
      "$BUDGET_USD" "$listing" "${opts[*]}" "$sfx" <<'PY'
import json, sys
src, dst, wf, model, secs, budget, listing, opts, sfx = sys.argv[1:]
try:
    r = json.load(open(src))
except Exception as e:
    r = {"error": f"could not parse result: {e}"}
cost = r.get("total_cost_usd")
over = bool(cost is not None and cost > float(budget)) or "budget" in str(r.get("subtype", ""))
json.dump({"workflow": wf, "harness": "claude-code", "model": model, "wall_seconds_measured": int(secs),
           "total_cost_usd": cost, "duration_ms": r.get("duration_ms"),
           "duration_api_ms": r.get("duration_api_ms"), "num_turns": r.get("num_turns"),
           "usage": r.get("usage"), "model_usage": r.get("modelUsage"), "is_error": r.get("is_error"),
           "subtype": r.get("subtype"), "max_budget_usd": float(budget), "over_budget": over,
           "pass": int(sfx[2:]) if sfx else 1, "run_folder": f"claude_compare/runs/{wf}-{model}{sfx}",
           "inputs": listing.split(), "claude_options": opts,
           "result_excerpt": (r.get("result") or r.get("error") or "")[:600]}, open(dst, "w"), indent=2)
print(f"{wf:<11} {model}{sfx:<4} ${cost}  {secs}s  turns={r.get('num_turns')}  subtype={r.get('subtype')}"
      + ("  OVER BUDGET" if over else ""))
PY
  RESEARCH_OFFLINE=1 "$ROOT/.venv/bin/python" "$ROOT/score.py" "$wf" "$dir" --tag "$model$sfx" > "$dir/_score.txt" 2>&1 \
    || echo "score failed for $wf $model$sfx (see $dir/_score.txt)"
}

spend() {   # prints: total spent so far, dearest pass so far, and "over" if a run hit its cap
  python3 - "$RUNS_OUT" "${MODELS[*]}" "${WFS[*]}" "$PASSES" "$1" <<'PY'
import json, sys
runs, models, wfs, passes, upto = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), int(sys.argv[4]), int(sys.argv[5])
total, per_pass, over = 0.0, {}, False
for p in range(1, upto + 1):
    sfx = f"-p{p}" if passes > 1 else ""
    for m in models:
        for wf in wfs:
            try:
                r = json.load(open(f"{runs}/{wf}-{m}{sfx}.json"))
            except (OSError, ValueError):
                print(f"missing result for {wf} {m}{sfx}", file=sys.stderr)
                continue
            c = r.get("total_cost_usd") or 0
            total += c
            per_pass[p] = per_pass.get(p, 0) + c
            if r.get("over_budget") or r.get("is_error"):
                over = over or bool(r.get("over_budget"))
                print(f"CHECK {wf} {m}{sfx}: subtype={r.get('subtype')} is_error={r.get('is_error')} cost={c}",
                      file=sys.stderr)
print(f"{total:.4f} {max(per_pass.values(), default=0):.4f} {'over' if over else 'ok'}")
PY
}

for pass in $(seq 1 "$PASSES"); do
  sfx=""; [ "$PASSES" -gt 1 ] && sfx="-p$pass"
  if [ "$pass" -gt 1 ]; then
    read -r spent dearest flag <<< "$(spend $((pass - 1)))"
    echo "after pass $((pass - 1)): \$$spent spent, dearest pass \$$dearest"
    if [ "$flag" = over ]; then echo "STOP: a run of pass $((pass - 1)) hit its \$$BUDGET_USD cap"; exit 2; fi
    if python3 -c "import sys; sys.exit(0 if $spent + $dearest > $SPEND_CAP_USD else 1)"; then
      echo "STOP: another pass could take the total past \$$SPEND_CAP_USD"; exit 2
    fi
  fi
  [ "$PASSES" -gt 1 ] && echo "##### pass $pass of $PASSES"
  for model in "${MODELS[@]}"; do
    for wf in "${WFS[@]}"; do
      run_one "$wf" "$model" "$sfx" &
    done
  done
  wait
done
read -r spent dearest flag <<< "$(spend "$PASSES")"
echo "all Claude Code runs finished; total \$$spent"
