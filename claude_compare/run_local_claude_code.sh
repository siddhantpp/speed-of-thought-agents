#!/usr/bin/env bash
# Claude Code (the harness) driving a LOCAL model through LM Studio's Anthropic-compatible /v1/messages.
# Same inbox task and grading as the other runs; the only change is where the model lives.
#
# Usage:  bash claude_compare/run_local_claude_code.sh [model]     default: qwen/qwen3.8-27b
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
MODEL="${1:-qwen/qwen3.8-27b}"
SLUG="$(echo "$MODEL" | tr '/' '_')"
DIR="$HERE/runs/inbox-claude-code-local-$SLUG"
OUT="$ROOT/results/runs/inbox-claude-code-local-$SLUG.json"
rm -rf "$DIR" && mkdir -p "$DIR" "$(dirname "$OUT")"
cp "$ROOT/sample_data/inbox/messages.jsonl" "$DIR/"
start=$(date +%s)
(cd "$DIR" && env -u ANTHROPIC_API_KEY ANTHROPIC_BASE_URL=http://localhost:1234 ANTHROPIC_AUTH_TOKEN=lmstudio \
    CLAUDE_CODE_ATTRIBUTION_HEADER=0 ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
    timeout 2400 claude -p "$(cat "$HERE/prompts/inbox.md")" --model "$MODEL" --output-format json \
    --no-session-persistence --strict-mcp-config --permission-mode acceptEdits \
    --allowedTools "Read Write Edit Glob Grep Bash(python3:*)" > "$DIR/_claude_result.json" 2> "$DIR/_stderr.txt")
status=$?
end=$(date +%s)
python3 - "$DIR/_claude_result.json" "$OUT" "$MODEL" "$((end - start))" "$status" <<'PY'
import json, sys
src, dst, model, secs, status = sys.argv[1:]
try:
    r = json.load(open(src))
except Exception as e:
    r = {"error": f"could not parse result: {e}"}
json.dump({"workflow": "inbox", "harness": "claude-code", "backend": "LM Studio /v1/messages (local)", "model": model,
           "wall_seconds_measured": int(secs), "exit_status": int(status), "num_turns": r.get("num_turns"),
           "usage": r.get("usage"), "is_error": r.get("is_error"), "subtype": r.get("subtype"),
           "result_excerpt": (r.get("result") or r.get("error") or "")[:800]}, open(dst, "w"), indent=2)
print(f"claude-code + {model}: exit {status}, {secs}s, turns={r.get('num_turns')}, error={r.get('is_error')}")
PY
"$ROOT/.venv/bin/python" "$ROOT/score.py" inbox "$DIR" --tag "claude-code-local-$SLUG" > "$DIR/_score.txt" 2>&1 \
  && head -20 "$DIR/_score.txt" || echo "scoring failed (missing or malformed outputs); see $DIR"
