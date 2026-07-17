#!/bin/bash
# ============================================================================
# W1 / Arm A3 (FULL) : GPT-5-mini FLAT, in the DR-Tulu agent harness. 0 GPUs.
#
#   Same harness as A1 (online MCP serper/jina, browse on, max_tool_calls=10,
#   --use-cache, 49 RQA + 50 DRB subset, same graders). Differences from A1:
#     - MODEL: gpt-5-mini (commercial API, litellm) for BOTH search + browse agents
#     - tool_calling_mode=native  (OpenAI function-calling; parser-mode degraded
#       ~12% of items for gpt-5-mini because it rejects `stop` and drifts on the
#       tag format. native verified 0/4 degraded. A1 stays parser-mode/untouched.)
#     - config base = workflows/auto_search_sft-oai.yaml (ships gpt-4.1; overridden)
#   Online Serper/Jina (frozen wii corpus still gated) — same as A1.
#
#   This is the FLAT frontier control. A2 (RLM, same model) vs A3 = recursion effect.
# ============================================================================
set -euo pipefail

REPO=/gpfs/home5/lgehringer/Dr-RLM
VENV=/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers
AGENT=$REPO/dr-tulu/agent
W1=$REPO/drrlm
OUT=$REPO/drrlm/runs/A3
mkdir -p "$OUT" "$REPO/drrlm/runs/logs"

ARM=A3
N=final_run               # loader caps to our 49/50 local subset
MCP_PORT=8020             # avoid colliding with any other run
MODEL=gpt-5-mini
OVERRIDES="use_browse_agent=true,browse_tool_name=jina,search_agent_max_tool_calls=10,search_agent_model_name=${MODEL},browse_agent_model_name=${MODEL},search_agent_temperature=1,browse_agent_temperature=1,tool_calling_mode=native,mcp_port=${MCP_PORT}"

source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
source "$REPO/drrlm/keys.sh"

export RESEARCHQA_LOCAL_PATH=$REPO/drrlm/data/subsets/researchqa_strat49.jsonl
export DRB_LOCAL_PATH=$REPO/drrlm/data/subsets/drb_en50.jsonl

cd "$AGENT"

MCP_PID=""
cleanup() { echo "[cleanup] stopping MCP"; [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[mcp] launching ONLINE MCP backend on :$MCP_PORT"
python -m dr_agent.mcp_backend.main --port $MCP_PORT > "$REPO/drrlm/runs/logs/${ARM}_mcp.log" 2>&1 &
MCP_PID=$!
for i in $(seq 1 60); do
  curl -sf "http://localhost:$MCP_PORT/health" >/dev/null 2>&1 && { echo "[mcp] ready after ~$((i*2))s"; break; }
  sleep 2
  kill -0 "$MCP_PID" 2>/dev/null || { echo "[FATAL] MCP died; see ${ARM}_mcp.log"; exit 1; }
done

# ---- generate (flat, native tool-calling, online search+browse) + wall-clock ----
t_rqa_start=$(date +%s)
echo "[gen] ResearchQA (N=$N, $MODEL, native) ..."
python workflows/auto_search_sft.py generate-dataset researchqa \
    --num-examples $N --max-concurrent 8 --batch-size 20 --use-cache \
    --config workflows/auto_search_sft-oai.yaml \
    --config-overrides "$OVERRIDES" \
    --output "$OUT/researchqa.jsonl" < /dev/null
t_rqa_end=$(date +%s)

t_drb_start=$(date +%s)
echo "[gen] Deep Research Bench (N=$N, $MODEL, native) ..."
python workflows/auto_search_sft.py generate-dataset deep_research_bench \
    --num-examples $N --max-concurrent 8 --batch-size 20 --use-cache \
    --config workflows/auto_search_sft-oai.yaml \
    --config-overrides "$OVERRIDES" \
    --output "$OUT/deep_research_bench.jsonl" < /dev/null
t_drb_end=$(date +%s)

python - "$OUT/timing.json" $t_rqa_start $t_rqa_end $t_drb_start $t_drb_end <<'PY'
import json, sys
out, rs, re, ds, de = sys.argv[1], *map(int, sys.argv[2:6])
json.dump({
    "researchqa":         {"wall_clock_s": re-rs, "concurrency": 8},
    "deep_research_bench":{"wall_clock_s": de-ds, "concurrency": 8},
}, open(out, "w"), indent=2)
print("[timing] wrote", out)
PY

# ---- grade (same graders as A1) ----
echo "[grade] ResearchQA coverage (gpt-4.1-mini) ..."
python scripts/evaluate.py researchqa "$OUT/researchqa.jsonl" 2>&1 | tee "$OUT/researchqa_score.txt"

echo "[grade] Deep Research Bench RACE+FACT (Gemini, self-contained) ..."
python scripts/evaluate.py deep_research_bench "$OUT/deep_research_bench.jsonl" \
    --save_path "$OUT/drb_eval" 2>&1 | tee "$OUT/drb_score.txt"

echo "[done] A3 outputs in $OUT"
