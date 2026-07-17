#!/bin/bash
# ============================================================================
# W1 / Arm A3 SMOKE (cost probe) : GPT-5-mini FLAT, in the DR-Tulu agent harness.
#   0 GPUs (commercial API, litellm-routed). 2 items/bench. Generation only
#   (cost question is about generation; grader cost is separate + known).
#
#   Harness = A1-faithful: same MCP online tools (serper/jina), browse ON,
#   search_agent_max_tool_calls=10. Only the MODEL changes: gpt-5-mini for both
#   the search agent AND the browse/reader agent (so it's pure GPT-5-mini, 0 GPU).
#   Base config = workflows/auto_search_sft-oai.yaml (ships gpt-4.1; we override).
#
#   Purpose: read real `usage` token blocks -> $ per item -> project A2/A3 cost.
# ============================================================================
set -euo pipefail

REPO=/gpfs/home5/lgehringer/Dr-RLM
VENV=/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers
AGENT=$REPO/dr-tulu/agent
W1=$REPO/drrlm
OUT=$REPO/drrlm/runs/A3_smoke
mkdir -p "$OUT" "$REPO/drrlm/runs/logs"

MCP_PORT=8011   # avoid colliding with any :8000 left around
N=2
MODEL=gpt-5-mini
# A1-faithful harness, but model swapped to gpt-5-mini for BOTH agents; temp=1 (GPT-5 only supports 1)
OVERRIDES="use_browse_agent=true,browse_tool_name=jina,search_agent_max_tool_calls=10,search_agent_model_name=${MODEL},browse_agent_model_name=${MODEL},search_agent_temperature=1,browse_agent_temperature=1"

source "$VENV/bin/activate"
export TOKENIZERS_PARALLELISM=false
source "$REPO/drrlm/keys.sh"

export RESEARCHQA_LOCAL_PATH=$REPO/drrlm/data/subsets/researchqa_strat49.jsonl
export DRB_LOCAL_PATH=$REPO/drrlm/data/subsets/drb_en50.jsonl

cd "$AGENT"

MCP_PID=""
cleanup() { [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "[mcp] launching ONLINE MCP backend on :$MCP_PORT"
python -m dr_agent.mcp_backend.main --port $MCP_PORT \
    > "$REPO/drrlm/runs/logs/A3_smoke_mcp.log" 2>&1 &
MCP_PID=$!
for i in $(seq 1 60); do
  curl -sf "http://localhost:$MCP_PORT/health" >/dev/null 2>&1 && { echo "[mcp] ready after ~$((i*2))s"; break; }
  sleep 2
  kill -0 "$MCP_PID" 2>/dev/null || { echo "[FATAL] MCP died; see A3_smoke_mcp.log"; exit 1; }
done

echo "[gen] ResearchQA (N=$N, $MODEL) ..."
python workflows/auto_search_sft.py generate-dataset researchqa \
    --num-examples $N --max-concurrent 2 --batch-size 2 --use-cache \
    --config workflows/auto_search_sft-oai.yaml \
    --config-overrides "$OVERRIDES,mcp_port=${MCP_PORT}" \
    --output "$OUT/researchqa.jsonl" < /dev/null

echo "[gen] Deep Research Bench (N=$N, $MODEL) ..."
python workflows/auto_search_sft.py generate-dataset deep_research_bench \
    --num-examples $N --max-concurrent 2 --batch-size 2 --use-cache \
    --config workflows/auto_search_sft-oai.yaml \
    --config-overrides "$OVERRIDES,mcp_port=${MCP_PORT}" \
    --output "$OUT/deep_research_bench.jsonl" < /dev/null

echo "[done] A3 smoke generation in $OUT"
