#!/bin/bash
# ============================================================================
# Generate rubric-carrying rollout TREES for the provenance-credit POC.
#
# One arm only (structured child-return = best citation survival), on the
# dr-tulu RL training distribution (drtulu_rl benchmark; rubrics ride in each row).
# gpt-5-mini agent (best recurser per G1), live web backend, depth 1, capture
# full per-node trajectories + evidence ledger. No grader here — credit + LOCO
# are computed offline by analysis/provenance_credit_stage1.py.
# ============================================================================
set -euo pipefail

REPO=/gpfs/home5/lgehringer/Dr-RLM
DRRLM=$REPO/dr-rlm
VENV=${VENV:-/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers}

MODEL=${MODEL:-gemini-2.5-flash}
MODEL_BASE_URL=${MODEL_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
MODEL_API_KEY_ENV=${MODEL_API_KEY_ENV:-GEMINI_API_KEY}
TOKENIZER=${TOKENIZER:-Qwen/Qwen3-8B}
BENCH=drtulu_rl
SUBSET=${SUBSET:-$DRRLM/data/subsets/drtulu_rl_decompose40.jsonl}
NUMEX=${NUMEX:-}
MAX_DEPTH=${MAX_DEPTH:-1}
MAX_ITERS=${MAX_ITERS:-12}
MAX_TOKENS=${MAX_TOKENS:-12000}
TEMP=${TEMP:-0.7}                                       # gemini accepts <1; gpt-5-class needs 1
SEARCH_BACKEND=${SEARCH_BACKEND:-web}
MCP_PORT=${MCP_PORT:-8040}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-80}
MAX_CHILDREN=${MAX_CHILDREN:-6}
CHILD_MODE=${CHILD_MODE:-structured}
OUT=${OUT:-$DRRLM/runs/provenance_poc}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD=${SHARD:-0}

# per-shard output file when sharded (shards must not write the same jsonl); the
# sharded launcher concatenates drtulu_rl.shard*.jsonl afterwards. Trajectories share
# the dir (per-id filenames never collide).
if [[ "$NUM_SHARDS" -gt 1 ]]; then
  OUTFILE="$OUT/drtulu_rl.shard${SHARD}.jsonl"
else
  OUTFILE="$OUT/drtulu_rl.jsonl"
fi

mkdir -p "$OUT/trajectories" "$DRRLM/runs/logs"
source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
[[ -f "$DRRLM/keys.sh" ]] && source "$DRRLM/keys.sh"
export DRTULU_RL_LOCAL_PATH="$SUBSET"
# Jina browse is out of quota; disable it so its 402 error text never leaks into the agent's
# context (a leaked "InsufficientBalanceError" made an orchestrator abort synthesis). Search
# (Serper) still grounds the trees and carries the provenance ids credit needs.
export DR_RLM_DISABLE_BROWSE=${DR_RLM_DISABLE_BROWSE:-1}
# ISOLATION: use a PRIVATE prompt (original + a "never abandon synthesis on tool error" clause)
# so the shared prompts/system_prompt.txt is untouched for parallel experiments. See
# docs/provenance_poc_CHANGES.md.
export DR_RLM_PROMPT_FILE=${DR_RLM_PROMPT_FILE:-$DRRLM/prompts/_provenance_poc_system_prompt.txt}
# ISOLATION: ledger capture in infer_driver is opt-in (default off) so other --driver new runs
# are byte-identical; this POC needs the per-tree evidence ledger for ledger_support credit.
export DR_RLM_CAPTURE_LEDGER=${DR_RLM_CAPTURE_LEDGER:-1}

AGENT=$REPO/dr-tulu/agent       # web MCP expects this cwd
GEN=$DRRLM/agent/generate.py
cd "$AGENT"

MCP_PID=""
cleanup() { [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT
if [[ "$SEARCH_BACKEND" == "web" ]]; then
  echo "[mcp] launching ONLINE web backend on :$MCP_PORT"
  python -m dr_agent.mcp_backend.main --port "$MCP_PORT" > "$DRRLM/runs/logs/poc_mcp_${SLURM_JOB_ID:-local}_${SHARD}.log" 2>&1 &
  MCP_PID=$!
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$MCP_PORT/health" >/dev/null 2>&1 && { echo "[mcp] ready after ~$((i*2))s"; break; }
    sleep 2; kill -0 "$MCP_PID" 2>/dev/null || { echo "[FATAL] MCP died"; exit 1; }
  done
fi

NUMEX_ARG=""; [[ -n "$NUMEX" ]] && NUMEX_ARG="--num-examples $NUMEX"
PROVIDER_ARG=""; [[ -n "$MODEL_BASE_URL" ]] && PROVIDER_ARG="--base-url $MODEL_BASE_URL --api-key-env $MODEL_API_KEY_ENV"

python "$GEN" \
  --driver new --benchmark "$BENCH" --model "$MODEL" --tokenizer-path "$TOKENIZER" \
  --max-recursion-depth "$MAX_DEPTH" --max-iterations "$MAX_ITERS" \
  --max-completion-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  --max-tool-calls "$MAX_TOOL_CALLS" --max-children "$MAX_CHILDREN" \
  --child-return-mode "$CHILD_MODE" \
  --search-backend "$SEARCH_BACKEND" --mcp-port "$MCP_PORT" \
  --save-trajectories "$OUT/trajectories" \
  --num-shards "$NUM_SHARDS" --shard "$SHARD" \
  $PROVIDER_ARG $NUMEX_ARG \
  --output "$OUTFILE" < /dev/null

echo "[done] -> $OUTFILE + $OUT/trajectories/"
