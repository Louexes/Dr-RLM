#!/bin/bash
# ============================================================================
# A/B: child->parent return contract  (prose  vs  structured)
#
#   Tests the tweet's hypothesis (neural_avb): an RLM orchestrator that synthesizes
#   over sub-agents' FREE-TEXT mini-reports reconciles conflicting prose and can
#   fabricate; handing it STRUCTURED claim+citation atoms it merges in code preserves
#   provenance and improves attribution at equal token cost.
#
#   Both arms use the UNIFIED driver (--driver new = SkyRL DrRlmEnv, the code RL trains),
#   recursion ON, and are byte-identical except --child-return-mode:
#     arm PROSE       -> rlm_query() returns the child's rendered report STRING  (v1)
#     arm STRUCTURED  -> rlm_query() returns {content, citations:[{id,claim}]} dict
#   Same model, same items, same search backend, same grader => isolates the contract.
#
#   Headline metric: DRB FACT valid_rate (citation-attribution fidelity) + RACE overall
#   (quality), at matched completion-token cost. Parameterized by env vars.
# ============================================================================
set -euo pipefail

REPO=/gpfs/home5/lgehringer/Dr-RLM
DRRLM=$REPO/dr-rlm
VENV=${VENV:-/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers}

# --- knobs (identical across arms except the contract) ---
MODEL=${MODEL:-gpt-5-mini}
MODEL_BASE_URL=${MODEL_BASE_URL:-}                       # OpenAI-compat base for non-OpenAI providers
MODEL_API_KEY_ENV=${MODEL_API_KEY_ENV:-OPENAI_API_KEY}
BENCH=${BENCH:-deep_research_bench}                      # DRB has FACT; researchqa = coverage only
NUMEX=${NUMEX:-}                                         # empty => full subset
MAX_DEPTH=${MAX_DEPTH:-2}                                # MUST be >=1 (no children => no A/B signal)
MAX_ITERS=${MAX_ITERS:-12}
MAX_TOKENS=${MAX_TOKENS:-12000}
TEMP=${TEMP:-0.7}
SEARCH_BACKEND=${SEARCH_BACKEND:-web}                    # web (online) | mcp_http | bm25 | bm25s | faiss | local_jsonl
SEARCH_ENDPOINT=${SEARCH_ENDPOINT:-http://localhost:8003/mcp}
SEARCH_CORPUS_PATH=${SEARCH_CORPUS_PATH:-$DRRLM/data/corpus.jsonl}   # local_jsonl/bm25/faiss corpora (e.g. data/offline_corpus/drb.jsonl)
SEARCH_INDEX_PATH=${SEARCH_INDEX_PATH:-$DRRLM/data/frozen_corpus/bm25s_index}   # bm25s/bm25/faiss index dir
CITATION_VERIFY=${CITATION_VERIFY:-0}                    # 1 => pre-submit citation verification (drop unsupported)
CHECK_TOOL=${CHECK_TOOL:-0}                              # 1 => expose check_citations() verdict oracle in the REPL (advertise via prompt variant)
CITE_BOUNCE=${CITE_BOUNCE:-0}                            # 1 => one-time empty-citations submission gate (env-side contract enforcement)
MCP_PORT=${MCP_PORT:-8030}
TOKENIZER=${TOKENIZER:-$MODEL}
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-80}                     # per-ITEM (whole tree) Serper/web call budget
MAX_CHILDREN=${MAX_CHILDREN:-6}                          # per-NODE sub-agent fan-out cap (bounds tree width)
SAVE_TRAJ=${SAVE_TRAJ:-1}                                # 1 => capture full per-node trajectories (parseable JSONL)
OUT=${OUT:-$DRRLM/runs/ab_child_return}
GRADE=${GRADE:-1}

if [[ "$MAX_DEPTH" -lt 1 ]]; then
  echo "[FATAL] MAX_DEPTH=$MAX_DEPTH < 1: no sub-agents spawn, so prose vs structured is identical. Set MAX_DEPTH>=1." >&2
  exit 2
fi

mkdir -p "$OUT" "$DRRLM/runs/logs"
source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
[[ -f "$DRRLM/keys.sh" ]] && source "$DRRLM/keys.sh"
export RESEARCHQA_LOCAL_PATH=${RESEARCHQA_LOCAL_PATH:-$DRRLM/data/subsets/researchqa_strat49.jsonl}
export DRB_LOCAL_PATH=${DRB_LOCAL_PATH:-$DRRLM/data/subsets/drb_en50.jsonl}

AGENT=$REPO/dr-tulu/agent       # graders + (web) MCP expect this cwd
GEN=$DRRLM/agent/generate.py
cd "$AGENT"

# --- online web backend: launch the dr_agent MCP server (Serper + Jina) once, shared by both arms ---
MCP_PID=""
cleanup() { [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT
if [[ "$SEARCH_BACKEND" == "web" ]]; then
  echo "[mcp] launching ONLINE web backend on :$MCP_PORT"
  python -m dr_agent.mcp_backend.main --port "$MCP_PORT" > "$DRRLM/runs/logs/ab_mcp_${SLURM_JOB_ID:-local}.log" 2>&1 &
  MCP_PID=$!
  for i in $(seq 1 60); do
    curl -sf "http://localhost:$MCP_PORT/health" >/dev/null 2>&1 && { echo "[mcp] ready after ~$((i*2))s"; break; }
    sleep 2; kill -0 "$MCP_PID" 2>/dev/null || { echo "[FATAL] MCP died"; exit 1; }
  done
fi

NUMEX_ARG=""; [[ -n "$NUMEX" ]] && NUMEX_ARG="--num-examples $NUMEX"
PROVIDER_ARG=""; [[ -n "$MODEL_BASE_URL" ]] && PROVIDER_ARG="--base-url $MODEL_BASE_URL --api-key-env $MODEL_API_KEY_ENV"

COMMON="--driver new --benchmark $BENCH --model $MODEL --tokenizer-path $TOKENIZER \
        --max-recursion-depth $MAX_DEPTH --max-iterations $MAX_ITERS \
        --max-completion-tokens $MAX_TOKENS --temperature $TEMP --max-tool-calls $MAX_TOOL_CALLS \
        --max-children $MAX_CHILDREN \
        --search-backend $SEARCH_BACKEND --search-endpoint $SEARCH_ENDPOINT --mcp-port $MCP_PORT \
        --search-corpus-path $SEARCH_CORPUS_PATH --search-index-path $SEARCH_INDEX_PATH \
        $([[ "$CITATION_VERIFY" == "1" ]] && echo --citation-verification || true) \
        $([[ "$CHECK_TOOL" == "1" ]] && echo --check-citations-tool || true) \
        $([[ "$CITE_BOUNCE" == "1" ]] && echo --citations-bounce || true) \
        $PROVIDER_ARG $NUMEX_ARG"

run_arm() {
  local mode=$1 ; local armdir="$OUT/$mode"
  mkdir -p "$armdir"
  local traj_arg=""
  [[ "$SAVE_TRAJ" == "1" ]] && traj_arg="--save-trajectories $armdir/trajectories"
  echo "[gen] arm=$mode ($MODEL, depth=$MAX_DEPTH, backend=$SEARCH_BACKEND, save_traj=$SAVE_TRAJ) ..."
  python "$GEN" $COMMON --child-return-mode "$mode" --output "$armdir/$BENCH.jsonl" $traj_arg < /dev/null
  if [[ "$GRADE" == "1" ]]; then
    if [[ "$BENCH" == "deep_research_bench" ]]; then
      echo "[grade $mode] DRB RACE+FACT ..."
      python scripts/evaluate.py deep_research_bench "$armdir/$BENCH.jsonl" --save_path "$armdir/drb_eval" \
        2>&1 | tee "$armdir/drb_score.txt"
    elif [[ "$BENCH" == "researchqa" ]]; then
      echo "[grade $mode] ResearchQA coverage ..."
      python scripts/evaluate.py researchqa "$armdir/$BENCH.jsonl" 2>&1 | tee "$armdir/researchqa_score.txt"
    elif [[ "$BENCH" == "healthbench" ]]; then
      echo "[grade $mode] HealthBench rubric ..."
      # The HB grader builds a PLAIN OpenAI() client; with a gemini judge it must be routed to
      # Gemini's OpenAI-compatible endpoint or every call 404s (model_not_found) forever.
      # Scoped to THIS command so other graders' env is untouched.
      OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai" \
      OPENAI_API_KEY="$GEMINI_API_KEY" \
      python scripts/evaluate.py healthbench "$armdir/$BENCH.jsonl" \
        --save_path "$armdir/hb_eval" --grader-model gemini-2.5-flash 2>&1 | tee "$armdir/healthbench_score.txt"
    elif [[ "$BENCH" == "sqav2" ]]; then
      echo "[grade $mode] SQA-CS-v2 ..."
      python scripts/evaluate.py sqa_cs_v2 "$armdir/$BENCH.jsonl" --save_path "$armdir/sqa_eval" \
        2>&1 | tee "$armdir/sqav2_score.txt"
    fi
  fi
}

# ARMS lets a job run a single arm (e.g. ARMS=structured for the budget-lift/FACT-fix run);
# default runs the full A/B. The cross-arm compare only runs when both arms are present.
ARMS=${ARMS:-"prose structured"}
for arm in $ARMS; do
  run_arm "$arm"
done

echo "[ab] generation+grading done -> $OUT/{$(echo $ARMS | tr ' ' ',')}"
if [[ -s "$OUT/prose/$BENCH.jsonl" && -s "$OUT/structured/$BENCH.jsonl" ]]; then
  echo "[ab] comparing ..."
  python "$DRRLM/analysis/ab_child_return.py" --root "$OUT" --bench "$BENCH" --tokenizer "$TOKENIZER" \
    --out "$OUT/ab_summary.json" || echo "[warn] analysis failed; rows are saved, run analysis/ab_child_return.py manually"
else
  echo "[ab] single arm ($ARMS) — skipping cross-arm compare"
fi
echo "[done] A/B outputs in $OUT"
