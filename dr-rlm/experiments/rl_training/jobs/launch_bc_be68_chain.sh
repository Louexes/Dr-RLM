#!/bin/bash
# Broadcast (RQ2 control) chain matched to B_e68 main arm: 3 x (judge_bc + train) afterany pairs, done-guard@75.
# ponytail: 4 chunks covers worst-case 25 steps/chunk; done-guard makes extras ~free.
set -euo pipefail
cd "$(dirname "$0")"
PREV=""
for i in 1 2 3; do
  if [ -z "$PREV" ]; then
    J=$(sbatch --parsable judge_server_bc.job)
    T=$(sbatch --parsable rl_bc_be68.job)
  else
    J=$(sbatch --parsable --dependency=afterany:$PREV judge_server_bc.job)
    T=$(sbatch --parsable --dependency=afterany:$PREV rl_bc_be68.job)
  fi
  echo "chunk$i: judge=$J train=$T"
  PREV=$T
done
