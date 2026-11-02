#!/usr/bin/env bash
# Run every workflow on the local setup, one after another, score each one, and record peak model memory.
#
# The local setup (agentkit/llm.py TIER_MODELS), all loaded in LM Studio:
#   workers and group leads  google/gemma-4-12b   (32,768-token context, 4 parallel slots)
#   master                   qwen/qwen3.8-27b     (65,536-token context, 4 parallel slots)
#   embeddings               text-embedding-nomic-embed-text-v1.5
#
# Usage:  bash run_all.sh               one pass over every workflow
#         PASSES=3 bash run_all.sh      three passes in a row (runs vary: report the median and the range)
#           traces  results/runs/<wf>-local-<stamp>.json         (one per workflow per pass; all are kept)
#           scores  results/scores/<wf>-local.json               (one pass)
#                   results/scores/<wf>-local-p<N>.json          (pass N of several)
#           memory  results/memory-local.json  (macOS footprint of the model processes every 2 s, across all passes)
#           timing  results/run-all-local.json (each pass and workflow: start, end, exit status, trace, score file
#                                               and peak model memory while it ran)
#           logs    outputs/logs/<wf>[-p<N>].log and score-<wf>[-p<N>].txt; with several passes each pass's
#                   outputs are copied to outputs/passes/p<N>/<wf>/
#         bash run_all.sh think    opt-in experiment, not part of the standard run: the same models, but the
#           master thinks ("high") before every plan, judgment and review (MASTER_REASONING=high). Files are
#           tagged think: <wf>-local-think-<stamp>.json, scores <wf>-local-think.json, memory-think.json,
#           run-all-think.json. Measured once on the first three workflows: the same scores at about 2.8x the time.
#         PROVIDER=anthropic|openai [SMALL_MODEL=<model id>] [PASSES=n] bash run_all.sh
#           the same pipelines on one small cloud model in every role (worker, group lead, master), through
#           agentkit/providers.py. SMALL_MODEL defaults to claude-haiku-5-5 (anthropic) or gpt-6-luna (openai).
#           Keys: SOT_ANTHROPIC_API_KEY / SOT_OPENAI_API_KEY, or KEY=VALUE lines in ~/.config/speed-of-thought/keys.env.
#           Embeddings stay local (the nomic model in LM Studio must be loaded; the chat models need not be).
#           Files are tagged with the run label (RUN_LABEL, default haiku / gpt6luna / the model id) instead of
#           local, in the same folders: traces <wf>-<label>-<stamp>.json, scores <wf>-<label>[-p<N>].json,
#           timing run-all-<label>.json (no memory sampler), logs outputs/logs/<wf>-<label>[-p<N>].log, each
#           pass's outputs in outputs/passes/<label>/p<N>/<wf>/. Every model call's billed tokens (uncached input,
#           cache reads and writes, output) are in its trace, so cost = usage x the provider's prices, and are
#           also appended as they happen to outputs/logs/usage-<label>.jsonl.
#
# Every workflow starts from an empty outputs/<wf>/ (the previous contents move to outputs/previous/<wf>/).
# The research workflow searches OpenAlex and Crossref live (free, no key) and caches every response in
# outputs/research/cache; to replay that run with no network: RESEARCH_OFFLINE=1 .venv/bin/python
# workflows/research_report.py (run_all.sh itself always searches live).
# To re-run one workflow alone:  .venv/bin/python workflows/<script> && .venv/bin/python score.py <wf>
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-local}"
PROVIDER="${PROVIDER:-lmstudio}"
case "$MODE" in
  local) unset RUN_TAG MASTER_REASONING; LABEL=local ;;
  think) export RUN_TAG=think MASTER_REASONING=high; LABEL=think ;;
  *) echo "usage: [PASSES=n] [PROVIDER=anthropic|openai [SMALL_MODEL=id]] bash run_all.sh [think]"; exit 1 ;;
esac
PASSES="${PASSES:-1}"
case "$PASSES" in ''|*[!0-9]*|0) echo "PASSES must be a positive whole number"; exit 1 ;; esac
unset WORKER_MODEL GROUP_MODEL MASTER_MODEL   # one configuration: the TIER_MODELS defaults above
case "$PROVIDER" in
  lmstudio|local) CLOUD=""; unset AGENTKIT_PROVIDER SMALL_MODEL AGENTKIT_RUN_LABEL AGENTKIT_USAGE_LOG ;;
  anthropic|openai)
    if [ "$MODE" != local ]; then echo "PROVIDER=$PROVIDER runs the standard setup only (no think mode)"; exit 1; fi
    CLOUD=1
    if [ -z "${SMALL_MODEL:-}" ]; then
      if [ "$PROVIDER" = anthropic ]; then SMALL_MODEL=claude-haiku-5-5; else SMALL_MODEL=gpt-6-luna; fi
    fi
    case "$SMALL_MODEL" in
      claude-haiku-5-5) LABEL=haiku ;; gpt-6-luna) LABEL=gpt6luna ;;
      *) LABEL="$(printf %s "$SMALL_MODEL" | tr -c 'A-Za-z0-9_.' _)" ;;
    esac
    LABEL="${RUN_LABEL:-$LABEL}"
    case "$LABEL" in local|think|*[!A-Za-z0-9_.]*|'') echo "RUN_LABEL must be letters, digits, _ or . (not local/think)"; exit 1 ;; esac
    # every call's billed tokens, appended as it happens (a crashed run's spend is still known)
    export AGENTKIT_PROVIDER="$PROVIDER" SMALL_MODEL AGENTKIT_RUN_LABEL="$LABEL" \
           AGENTKIT_USAGE_LOG="$PWD/outputs/logs/usage-$LABEL.jsonl" ;;
  *) echo "PROVIDER must be lmstudio, anthropic or openai"; exit 1 ;;
esac
export RESEARCH_OFFLINE="${RESEARCH_OFFLINE:-0}"

# key, then the script and its arguments, in run order
STEPS=(
  "inbox workflows/inbox_calendar.py --apply"
  "class_notes workflows/class_notes.py"
  "study_pack workflows/study_pack.py"
  "receipts workflows/receipts_split.py"
  "research workflows/research_report.py"
  "slides workflows/slide_deck.py"
)
if [ -f workflows/assistant_eval.py ] && [ -f scoring/assistant.py ]; then
  STEPS+=("assistant workflows/assistant_eval.py")   # last: it reads the class_notes and study_pack outputs
fi

LMS="$HOME/.lmstudio/bin/lms"
NEEDED="google/gemma-4-12b qwen/qwen3.8-27b text-embedding-nomic-embed-text-v1.5"
[ -n "$CLOUD" ] && NEEDED="text-embedding-nomic-embed-text-v1.5"   # cloud runs embed locally, nothing else
for model in $NEEDED; do
  if ! "$LMS" ps 2>/dev/null | grep -q "^$model "; then
    echo "LM Studio has no model loaded as $model; load the local setup first (see the header)."; exit 1
  fi
done
[ -n "$CLOUD" ] && echo "### $PROVIDER $SMALL_MODEL in every role, label $LABEL, RESEARCH_OFFLINE=$RESEARCH_OFFLINE"

mkdir -p outputs/logs results
TIMES="$(mktemp)"
STOP="$(mktemp -u)"
SAMPLER=""
if [ -z "$CLOUD" ]; then   # model memory only means something when the models run here
  .venv/bin/python experiments/mem_sampler.py "$LABEL" "$STOP" &
  SAMPLER=$!
fi
SAMPLER_T0=$(date +%s)
failed=()
for pass in $(seq 1 "$PASSES"); do
  if [ "$PASSES" -gt 1 ]; then sfx="-p$pass"; echo "##### pass $pass of $PASSES"; else sfx=""; fi
  if [ "$LABEL" = local ]; then SCORE_TAG="local$sfx"; else SCORE_TAG="local-$LABEL$sfx"; fi
  lsfx="$sfx"   # log and pass-copy names: cloud runs keep apart from the local ones
  if [ -n "$CLOUD" ]; then SCORE_TAG="$LABEL$sfx"; lsfx="-$LABEL$sfx"; fi
  for step in "${STEPS[@]}"; do
    read -r wf script args <<< "$step"
    echo "=== $wf ($LABEL${sfx:+, pass $pass}): $script ${args:-}"
    # Start from an empty outputs folder (the last run's moves to outputs/previous/), so nothing from an
    # earlier run, such as research's response cache (its fetched set), carries over into this one.
    if [ -d "outputs/$wf" ]; then
      mkdir -p outputs/previous && rm -rf "outputs/previous/$wf" && mv "outputs/$wf" "outputs/previous/$wf"
    fi
    t0=$(date +%s)
    # shellcheck disable=SC2086   # $args is deliberately split into words
    .venv/bin/python "$script" ${args:-} > "outputs/logs/$wf$lsfx.log" 2>&1
    status=$?
    t1=$(date +%s)
    tail -4 "outputs/logs/$wf$lsfx.log"
    echo "$pass $wf $t0 $t1 $status $SCORE_TAG" >> "$TIMES"
    if [ "$status" -ne 0 ]; then
      failed+=("$wf$sfx")
      echo "!!! $wf exited with status $status after $((t1 - t0)) s (log: outputs/logs/$wf$lsfx.log)"
      continue
    fi
    if .venv/bin/python score.py "$wf" --tag "$SCORE_TAG" > "outputs/logs/score-$wf$lsfx.txt" 2>&1; then
      .venv/bin/python - "results/scores/$wf-$SCORE_TAG.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
keys = ["full_score", "ok", "overall", "score", "passed", "key_concepts_covered", "settlement_exact", "router_correct"]
print(f"scored {s['workflow']} -> results/scores/{sys.argv[1].rsplit('/', 1)[-1]}: "
      + ", ".join(f"{k}={s[k]}" for k in keys if k in s))
PY
    else
      failed+=("$wf$sfx (scoring)")
      echo "!!! scoring $wf failed (log: outputs/logs/score-$wf$lsfx.txt)"
    fi
    if [ "$PASSES" -gt 1 ]; then
      pdir="outputs/passes/p$pass"; [ -n "$CLOUD" ] && pdir="outputs/passes/$LABEL/p$pass"
      mkdir -p "$pdir" && rm -rf "${pdir:?}/$wf" && cp -R "outputs/$wf" "$pdir/$wf"
    fi
  done
done
if [ -n "$SAMPLER" ]; then touch "$STOP"; wait "$SAMPLER"; fi
rm -f "$STOP"

.venv/bin/python - "$TIMES" "$SAMPLER_T0" "$LABEL" "$PASSES" "${CLOUD:-}" <<'PY'
import glob, json, os, sys
times, sampler_t0, label, passes, cloud = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
mem_file = f"results/memory-{label}.json"
mem = {"samples": []} if cloud else json.load(open(mem_file))
tag = "" if label == "local" or cloud else label + "-"
side = label if cloud else "local"   # trace names: <wf>-local-[<tag>-]<stamp> or <wf>-<label>-<stamp>
rows = []
for line in open(times):
    p, wf, t0, t1, status, score_tag = line.split()
    t0, t1 = int(t0), int(t1)
    pattern = f"results/runs/{wf}-local-{tag}[0-9]*.json".replace("-local-", f"-{side}-", 1)
    traces = [x for x in glob.glob(pattern) if t0 - 1 <= os.path.getmtime(x) <= t1 + 5]
    inside = [s["total_gb"] for s in mem["samples"] if t0 - sampler_t0 - 2 <= s["t"] <= t1 - sampler_t0 + 2]
    score = f"results/scores/{wf}-{score_tag}.json"
    rows.append({"pass": int(p), "workflow": wf, "start_epoch": t0, "end_epoch": t1, "seconds": t1 - t0,
                 "exit_status": int(status),
                 "trace": os.path.basename(max(traces, key=os.path.getmtime)) if traces else None,
                 "score": score if os.path.exists(score) else None,
                 "peak_model_memory_gb": max(inside) if inside else None})
out = {"label": label, "passes": passes, "memory_file": f"results/memory-{label}.json",
       "memory_sampler_started_epoch": sampler_t0,
       "how": "start and end of each workflow (date +%s); peak = max memory sample (every 2 s) from 2 s before "
              "its start to 2 s after its end", "workflows": rows}
if cloud:
    out.update(memory_file=None, memory_sampler_started_epoch=None, provider=os.environ.get("AGENTKIT_PROVIDER"),
               small_model=os.environ.get("SMALL_MODEL"), research_offline=os.environ.get("RESEARCH_OFFLINE"),
               embeddings="local: text-embedding-nomic-embed-text-v1.5 in LM Studio (de-duplication only)",
               how="start and end of each workflow (date +%s); no memory sampler (the models run in the cloud)")
json.dump(out, open(f"results/run-all-{label}.json", "w"), indent=2)
for r in rows:
    print(f"  p{r['pass']} {r['workflow']:<12} {r['seconds']:>5} s  exit {r['exit_status']}  "
          f"peak {r['peak_model_memory_gb']} GB  {r['trace']}")
print(f"wrote results/run-all-{label}.json")
PY
rm -f "$TIMES"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "FAILED: ${failed[*]}"; exit 1
fi
echo "all ${#STEPS[@]} workflows ran and were scored, $PASSES pass(es)"
