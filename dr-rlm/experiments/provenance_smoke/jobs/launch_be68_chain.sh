#!/bin/bash
# Launch the fresh100 chain: 4 x (judge + train) chunk pairs, afterany-chained.
# ponytail: 4 chunks covers worst-case 25 steps/chunk; done-guard makes extras ~free.
set -euo pipefail
cd "$(dirname "$0")"
PREV=""
for i in 1 2 3; do
  if [ -z "$PREV" ]; then
    J=$(sbatch --parsable judge_server.job)
    T=$(sbatch --parsable rl_be68.job)
  else
    J=$(sbatch --parsable --dependency=afterany:$PREV judge_server.job)
    T=$(sbatch --parsable --dependency=afterany:$PREV rl_be68.job)
  fi
  echo "chunk$i: judge=$J train=$T"
  PREV=$T
done
