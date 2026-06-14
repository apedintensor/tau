#!/usr/bin/env bash
# A/B pool-fill experiment: temp=1 vs shell_tools on google-vertex/global.
set -euo pipefail

REPO=/home/const/subnet66/tau
VENV=/home/const/subnet66/.venv/bin/python
POOL_ROOT="$REPO/workspace/validate/netuid-66"
PHASE_MINUTES="${PHASE_MINUTES:-8}"
MIN_LLM_CALLS="${MIN_LLM_CALLS:-40}"

clear_pools() {
  rm -f "$POOL_ROOT/task-pool"/*.json "$POOL_ROOT/task-pool-retest"/*.json 2>/dev/null || true
  echo "cleared pools: primary=$(ls "$POOL_ROOT/task-pool"/*.json 2>/dev/null | wc -l) retest=$(ls "$POOL_ROOT/task-pool-retest"/*.json 2>/dev/null | wc -l)"
}

analyze_phase() {
  local label=$1 since_iso=$2
  "$VENV" <<PY
import json, glob
from collections import Counter
from datetime import datetime, timezone

label = ${label@Q}
since = datetime.fromisoformat(${since_iso@Q})
files = sorted(glob.glob("$REPO/workspace/rollouts/tasks/validate-*/records/*.json"))
llm_calls = []
rollouts = []
for fp in files:
    d = json.load(open(fp))
    started = d.get("started_at") or ""
    try:
        ts = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except Exception:
        continue
    if ts < since:
        continue
    rollouts.append(d)
    for step in d.get("trajectory") or []:
        if step.get("type") == "llm_call":
            llm_calls.append((d.get("exit_reason"), d.get("success"), step))

print(f"\n=== {label} ===")
print(f"rollouts={len(rollouts)} llm_calls={len(llm_calls)}")
if not llm_calls:
    print("(no data yet)")
    raise SystemExit(0)

temps = Counter(); tc = Counter(); bash_req = 0; text_only_req = 0
malformed = empty = 0; finish = Counter()
for _, _, step in llm_calls:
    req = step.get("request") or {}
    resp = step.get("response") or {}
    temps[req.get("temperature")] += 1
    tc[req.get("tool_choice")] += 1
    tools = req.get("tools") or []
    if tools and any((t.get("function") or {}).get("name") == "bash" for t in tools):
        bash_req += 1
    if req.get("tool_choice") == "none" and not tools:
        text_only_req += 1
    native = resp.get("native_finish_reason") or resp.get("finish_reason") or ""
    finish[str(native)] += 1
    if "MALFORMED" in str(native):
        malformed += 1
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    if not str(msg.get("content") or "").strip() and not msg.get("tool_calls"):
        empty += 1

outcomes = Counter((d.get("exit_reason"), d.get("success")) for d in rollouts)
print("temperature", dict(temps))
print("tool_choice", dict(tc))
print(f"bash_tool_requests={bash_req}/{len(llm_calls)} text_only_requests={text_only_req}/{len(llm_calls)}")
print("finish_reasons", dict(finish))
print(f"malformed={malformed} empty={empty}")
print("rollout_outcomes", dict(outcomes))
PY
}

run_phase() {
  local label=$1
  shift
  local since
  since="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
  echo "\n>>> phase $label starting at $since"
  clear_pools

  doppler run -p arbos -c dev -- bash -lc "
set -euo pipefail
cd $REPO
export PYTHONPATH=src
export OPENROUTER_UPSTREAM_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_PROVIDER_ONLY=google-ai-studio
export OPENROUTER_PROVIDER_ALLOW_FALLBACKS=false
export SOLVER_EMPTY_RESPONSE_RETRIES=5
export GENERATOR_MODEL=google/gemini-3.1-flash-lite
export EVAL_MODEL=google/gemini-3.1-flash-lite
export TAU_POOL_GENERATION_CONCURRENCY=6
$*
exec $VENV -m cli pool-manager \\
  --workspace-root $REPO \\
  --solver-model google/gemini-3.1-flash-lite \\
  --solver-provider-only google-vertex/global \\
  --solver-provider-disable-fallbacks \\
  --poll-interval-seconds 10 \\
  --task-pool-target 20 \\
  --task-pool-static \\
  --record-rollouts \\
  --rollout-root $REPO/workspace/rollouts \\
  --pool-filler-concurrency 16 \\
  --docker-solver-start-concurrency 16
" &
  local pid=$!

  local deadline=$((SECONDS + PHASE_MINUTES * 60))
  while (( SECONDS < deadline )); do
    sleep 30
    count=$("$VENV" <<PY
import json, glob
from datetime import datetime
since = datetime.fromisoformat("$since".replace("Z", "+00:00"))
n = 0
for fp in glob.glob("$REPO/workspace/rollouts/tasks/validate-*/records/*.json"):
    d = json.load(open(fp))
    try:
        ts = datetime.fromisoformat((d.get("started_at") or "").replace("Z", "+00:00"))
    except Exception:
        continue
    if ts < since:
        continue
    n += sum(1 for s in (d.get("trajectory") or []) if s.get("type") == "llm_call")
print(n)
PY
)
    echo "phase $label: llm_calls=$count / target=$MIN_LLM_CALLS elapsed=$((SECONDS))s"
    if (( count >= MIN_LLM_CALLS )); then
      break
    fi
  done

  kill "$pid" 2>/dev/null || true
  pkill -f "python -m cli pool-manager" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 2
  analyze_phase "$label" "$since"
}

pm2 stop pool-manager 2>/dev/null || true

run_phase "A_temp1_text_only" \
  "export SOLVER_TEMPERATURE=1" \
  "export SOLVER_TEXT_ONLY=true" \
  "unset SOLVER_SHELL_TOOLS 2>/dev/null || true"

run_phase "B_temp0_shell_tools" \
  "export SOLVER_TEMPERATURE=0" \
  "export SOLVER_SHELL_TOOLS=true" \
  "unset SOLVER_TEXT_ONLY 2>/dev/null || true"

echo "\nDone. Restart production pool-manager manually if needed."
