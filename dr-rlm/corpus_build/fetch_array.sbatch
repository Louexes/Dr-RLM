#!/bin/bash
#SBATCH --job-name=drlm_fetch
#SBATCH --partition=staging
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/data/frozen_corpus/logs/fetch_%a.out
/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/.venv-crawl/bin/python /gpfs/home5/lgehringer/Dr-RLM/dr-rlm/corpus_build/fetch.py --root /gpfs/home5/lgehringer/Dr-RLM/dr-rlm/data/frozen_corpus --workers 8 --num-shards 8 --shard $SLURM_ARRAY_TASK_ID
