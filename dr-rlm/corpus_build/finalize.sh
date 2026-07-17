#!/bin/bash
# Phases 4-7 (normalize -> index -> audit -> freeze) as one chained SLURM step.
# Runs after fetch (afterok). No internet needed — pure GPFS + CPU.
set -euo pipefail
VENV=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/.venv-crawl/bin/python
CB=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/corpus_build
ROOT=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/data/frozen_corpus
DATE="${1:-2026-06-26}"

echo "===== [finalize] PHASE 4: normalize ====="
$VENV "$CB/normalize.py" --root "$ROOT"
echo "===== [finalize] PHASE 5: index (bm25s) ====="
$VENV "$CB/index_bm25.py" --root "$ROOT"
echo "===== [finalize] PHASE 6: coverage audit ====="
$VENV "$CB/audit.py" --root "$ROOT"
echo "===== [finalize] PHASE 7: freeze ====="
$VENV "$CB/freeze.py" --root "$ROOT" --date "$DATE" \
  --note "self-crawled frozen corpus: ddgs(auto) discovery + trafilatura reading + bm25s index; RL(4852)+SFT(744)+eval(drb/researchqa/healthbench/sqa/simpleqa/2wiki/webwalker)"
echo "===== [finalize] DONE ====="
