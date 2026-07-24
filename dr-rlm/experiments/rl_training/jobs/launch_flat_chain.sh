#!/bin/bash
# Flat (DR-Tulu) arm chain: 3 x (judge_flat + train) afterany pairs, done-guard@75.
# Own judge endpoint => parallel-safe with the recursive bc/cc arms. Prints ids for the watchdog.
# ponytail: 3 chunks covers worst-case ~25 steps/chunk; done-guard makes extras ~free.
set -euo pipefail
cd "$(dirname "$0")"
PREV=""
FIRST_TRAIN=""
for i in 1 2 3; do
  if [ -z "$PREV" ]; then
    J=$(sbatch --parsable judge_server_flat.job)
    T=$(sbatch --parsable rl_drtulu_flat.job)
    FIRST_TRAIN=$T
  else
    J=$(sbatch --parsable --dependency=afterany:$PREV judge_server_flat.job)
    T=$(sbatch --parsable --dependency=afterany:$PREV rl_drtulu_flat.job)
  fi
  echo "chunk$i: judge=$J train=$T"
  PREV=$T
done
echo "FIRST_TRAIN=$FIRST_TRAIN"
