#!/bin/bash
# ============================================================================
# W1 / Arm A2 : GPT-5-mini in the rlm/ RECURSIVE REPL harness. 0 GPUs.
#
#   The recursive counterpart of A3 (GPT-5-mini FLAT). Held identical to A1/A3 on
#   everything except the engine + recursion:
#     - SAME online MCP backend (Serper + Jina), launched WITHOUT --local-searcher-type
#     - SAME items (dr-tulu loaders via RESEARCHQA_LOCAL_PATH / DRB_LOCAL_PATH env)
#     - SAME graders (scripts/evaluate.py researchqa / deep_research_bench)
#     - SAME output row schema (example_id / original_data / problem / final_response)
#   DIFFERENCE: engine = rlm/ recursive REPL driven by gpt-5-mini, run DEEP
#   (MAX_DEPTH, default 5) so we see what a frontier model does with recursive
#   decomposition. Model = gpt-5-mini (same as A3) => A2-vs-A3 isolates the scaffold;
#   A1 (DR-Tulu-8B) is the trained-8B reference.
#
#   Parameterized by env vars (defaults = full run).
# ============================================================================
set -euo pipefail

REPO=/gpfs/home5/lgehringer/Dr-RLM
VENV=/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers
AGENT=$REPO/dr-tulu/agent
DRRLM=$REPO/drrlm

ARM=${ARM:-A2}
OUT=${OUT:-$DRRLM/runs/$ARM}
MCP_PORT=${MCP_PORT:-8030}
MODEL=${MODEL:-gpt-5-mini}
MODEL_BASE_URL=${MODEL_BASE_URL:-}                       # set to route the agent to a non-OpenAI provider (e.g. Gemini OpenAI-compat)
MODEL_API_KEY_ENV=${MODEL_API_KEY_ENV:-OPENAI_API_KEY}   # env var holding the key for MODEL_BASE_URL
MAX_DEPTH=${MAX_DEPTH:-5}
MAX_ITERS=${MAX_ITERS:-12}
MAX_SUBCALLS=${MAX_SUBCALLS:-4}
MAX_TOKENS=${MAX_TOKENS:-12000}
ITEM_TIMEOUT=${ITEM_TIMEOUT:-900}    # per-item wall-clock guard (s)
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-80} # search+browse budget per item (scope below)
TOOL_BUDGET_SCOPE=${TOOL_BUDGET_SCOPE:-tree}  # 'tree' (legacy A2) or 'node' (per-agent budget == flat baseline)
ORCHESTRATOR=${ORCHESTRATOR:-true}            # 'false' drops the decompose/delegate addendum (flat arm)
NUMEX=${NUMEX:-}                 # empty => all of the subset (49 RQA / 50 DRB); else an int
BENCH=${BENCH:-both}             # researchqa | deep_research_bench | both
NUM_SHARDS=${NUM_SHARDS:-1}      # >1 => run that many shard PROCESSES in parallel (clean per-item instrumentation)
GRADE=${GRADE:-1}                # 1 => run graders after generation
mkdir -p "$OUT" "$DRRLM/runs/logs"

source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
source "$DRRLM/keys.sh"
export RESEARCHQA_LOCAL_PATH=$DRRLM/data/subsets/researchqa_strat49.jsonl
export DRB_LOCAL_PATH=$DRRLM/data/subsets/drb_en50.jsonl
export SQAV2_LOCAL_PATH=${SQAV2_LOCAL_PATH:-$DRRLM/data/subsets/sqav2_cs100.jsonl}   # frozen ScholarQA-CSv2 (all 100, breadth-tagged); env-overridable
export HEALTHBENCH_LOCAL_PATH=${HEALTHBENCH_LOCAL_PATH:-$DRRLM/data/subsets/healthbench_hard_repr.jsonl}  # representative difficulty-weighted hard subset (72); env-overridable

NUMEX_ARG=""
[[ -n "$NUMEX" ]] && NUMEX_ARG="--num-examples $NUMEX"

cd "$AGENT"   # graders + MCP expect the agent dir as cwd

MCP_PID=""
cleanup() { echo "[cleanup] stopping MCP"; [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[mcp] launching ONLINE MCP backend on :$MCP_PORT"
python -m dr_agent.mcp_backend.main --port $MCP_PORT > "$DRRLM/runs/logs/${ARM}_mcp_${SLURM_JOB_ID:-local}.log" 2>&1 &
MCP_PID=$!
for i in $(seq 1 60); do
  curl -sf "http://localhost:$MCP_PORT/health" >/dev/null 2>&1 && { echo "[mcp] ready after ~$((i*2))s"; break; }
  sleep 2
  kill -0 "$MCP_PID" 2>/dev/null || { echo "[FATAL] MCP died; see ${ARM}_mcp log"; exit 1; }
done

GEN=$DRRLM/agent/generate.py
ORCH_ARG=$([[ "$ORCHESTRATOR" == "true" ]] && echo "--orchestrator" || echo "--no-orchestrator")
PROVIDER_ARG=""
[[ -n "$MODEL_BASE_URL" ]] && PROVIDER_ARG="--base-url $MODEL_BASE_URL --api-key-env $MODEL_API_KEY_ENV"
LOG_DIR=${LOG_DIR:-}                                     # set => per-item full trajectory JSONL capture
LOGDIR_ARG=""
[[ -n "$LOG_DIR" ]] && LOGDIR_ARG="--log-dir $LOG_DIR"
COMMON="--mcp-port $MCP_PORT --model $MODEL --max-depth $MAX_DEPTH --max-iterations $MAX_ITERS \
        --max-concurrent-subcalls $MAX_SUBCALLS --max-completion-tokens $MAX_TOKENS \
        --item-timeout $ITEM_TIMEOUT --max-tool-calls $MAX_TOOL_CALLS \
        --tool-budget-scope $TOOL_BUDGET_SCOPE $ORCH_ARG $PROVIDER_ARG $NUMEX_ARG $LOGDIR_ARG"

run_rqa() { [[ "$BENCH" == "both" || "$BENCH" == "researchqa" ]]; }
run_drb() { [[ "$BENCH" == "both" || "$BENCH" == "deep_research_bench" ]]; }
run_sqa() { [[ "$BENCH" == "sqav2" ]]; }
run_hb()  { [[ "$BENCH" == "healthbench" ]]; }

# Generate one benchmark, sharded across NUM_SHARDS parallel processes (each runs its
# items SEQUENTIALLY → its global token/recursion instrumentation stays correctly isolated;
# parallelism is across processes). Shard stdout is redirected to per-shard logs (keeps the
# main job log clean + avoids interleaved-stdout garble). Shard files are concatenated into
# the final $OUT/$bench.jsonl that the grader reads.
gen_bench() {
  local bench=$1
  local t0; t0=$(date +%s)
  echo "[gen] $bench ($MODEL, depth=$MAX_DEPTH, timeout=${ITEM_TIMEOUT}s, tool_budget=$MAX_TOOL_CALLS, shards=$NUM_SHARDS) ..."
  if [[ "$NUM_SHARDS" -le 1 ]]; then
    python "$GEN" --benchmark "$bench" --output "$OUT/$bench.jsonl" $COMMON < /dev/null
  else
    local pids=() s fail=0
    for s in $(seq 0 $((NUM_SHARDS-1))); do
      python "$GEN" --benchmark "$bench" --output "$OUT/${bench}.shard${s}.jsonl" \
        --num-shards "$NUM_SHARDS" --shard "$s" $COMMON < /dev/null \
        > "$DRRLM/runs/logs/${ARM}_${bench}_shard${s}_${SLURM_JOB_ID:-local}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    cat "$OUT/${bench}.shard"*.jsonl > "$OUT/$bench.jsonl"
    echo "[gen] $bench: assembled $(wc -l < "$OUT/$bench.jsonl") rows from $NUM_SHARDS shards (any_shard_fail=$fail)"
  fi
  echo "{\"$bench\": {\"wall_clock_s\": $(( $(date +%s) - t0 )), \"num_shards\": $NUM_SHARDS}}" > "$OUT/timing_$bench.json"
}

if run_rqa; then gen_bench researchqa; fi
if run_drb; then gen_bench deep_research_bench; fi
if run_sqa; then gen_bench sqav2; fi
if run_hb;  then gen_bench healthbench; fi

if [[ "$GRADE" == "1" ]]; then
  if run_rqa; then
    echo "[grade] ResearchQA coverage (gpt-4.1-mini) ..."
    python scripts/evaluate.py researchqa "$OUT/researchqa.jsonl" 2>&1 | tee "$OUT/researchqa_score.txt"
  fi
  if run_drb; then
    echo "[grade] Deep Research Bench RACE+FACT (Gemini) ..."
    python scripts/evaluate.py deep_research_bench "$OUT/deep_research_bench.jsonl" \
        --save_path "$OUT/drb_eval" 2>&1 | tee "$OUT/drb_score.txt"
  fi
  if run_sqa; then
    echo "[grade] ScholarQA-CSv2 rubric coverage (Gemini-2.5-flash, sqa_cs_v2; astabench/inspect via uv) ..."
    export GOOGLE_API_KEY="${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}"   # sqa grader checks GOOGLE_API_KEY
    # best-effort: a grader-dep/key failure must NOT discard the (expensive) generation
    { python scripts/evaluate.py sqa_cs_v2 "$OUT/sqav2.jsonl" --save_path "$OUT/sqa_eval" 2>&1 | tee "$OUT/sqav2_score.txt"; } \
        || echo "[warn] sqa grading failed (astabench/inspect_ai via uv, or GOOGLE_API_KEY) — generation saved; grade later"
  fi
  if run_hb; then
    echo "[grade] HealthBench rubric coverage (Gemini-2.5-flash grader via OpenAI-compat) ..."
    # ChatCompletionSampler uses a bare OpenAI() -> route it to Gemini via env + --grader-model
    { OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
      OPENAI_API_KEY="${GEMINI_API_KEY:-}" \
      python scripts/evaluate.py healthbench "$OUT/healthbench.jsonl" \
        --save_path "$OUT/hb_eval" --grader-model gemini-2.5-flash 2>&1 | tee "$OUT/healthbench_score.txt"; } \
        || echo "[warn] healthbench grading failed — generation saved; grade later"
  fi
fi

echo "[done] A2 ($BENCH) outputs in $OUT"
