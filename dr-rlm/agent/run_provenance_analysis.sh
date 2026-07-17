#!/bin/bash
# Run Stage 0 (non-degeneracy) + Stage 1 (credit + LOCO alignment + verdict) on the
# captured provenance-POC trees. Gemini judge + synthesizer. Idempotent; re-runnable.
set -uo pipefail
DRRLM=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
VENV=${VENV:-/gpfs/home5/lgehringer/venvs/rlm_vllm311_headers}
RUN=${RUN:-$DRRLM/runs/provenance_poc}
CONC=${CONC:-8}

source "$VENV/bin/activate"
[[ -f "$DRRLM/keys.sh" ]] && source "$DRRLM/keys.sh"
cd "$DRRLM"

echo "=== STAGE 0: non-degeneracy (judge-free) ==="
python analysis/provenance_credit_stage0.py \
  --runs "$RUN" \
  --out results/provenance_credit_stage0.json 2>&1 | tee results/_stage0.log

echo "=== STAGE 1: credit + LOCO alignment + verdict (gemini) ==="
python analysis/provenance_credit_stage1.py \
  --runs "$RUN" \
  --subset "$DRRLM/data/subsets/drtulu_rl_decompose40.jsonl" \
  --concurrency "$CONC" \
  --out results/provenance_credit_stage1.json 2>&1 | tee results/_stage1.log

echo "=== DONE -> results/provenance_credit_stage{0,1}.json ==="
