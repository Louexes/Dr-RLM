#!/bin/bash
#SBATCH --job-name=drlm_finalize
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --dependency=afterany:24243508
#SBATCH --output=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/data/frozen_corpus/logs/finalize.out
bash /gpfs/home5/lgehringer/Dr-RLM/dr-rlm/corpus_build/finalize.sh 2026-06-27
