#!/bin/bash
# DR-RLM recursive cold-start SFT launcher.
#
# Thin wrapper that (1) registers the dr_rlm_recursive_sft dataset by merging
# sft/dataset_info_entry.json into LLaMA-Factory's dataset_info.json, then (2) runs
# `llamafactory-cli train dr_rlm_sft.yaml`. Mirrors dr-tulu/sft/llama-factory/train/train.sh.
#
# Usage:
#   # 1. generate the recursive SFT data (rejection sampling from the rlm inference arm):
#   python -m examples.train.dr_rlm.sft.gen_recursive_sft \
#       --prompts ~/data/dr_rlm_rl/train.parquet \
#       --out_jsonl ~/data/dr_rlm_sft/dr_rlm_recursive_sft.jsonl \
#       --n_samples 4 --max_depth 2 --threshold 0.5 \
#       --model rl-research/dr-tulu-8b --base_url http://localhost:8000/v1
#
#   # 2. train (DATA_DIR must contain dr_rlm_recursive_sft.jsonl):
#   DATA_DIR=~/data/dr_rlm_sft LF_DIR=/path/to/LLaMA-Factory bash sft/run_sft.sh sft/dr_rlm_sft.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load a local .env if present (mirrors dr-tulu/train.sh).
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export WANDB_PROJECT="${WANDB_PROJECT:-rl-research}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

# Config to train (default: the dr_rlm SFT yaml next to this script).
CONFIG="${1:-${SCRIPT_DIR}/dr_rlm_sft.yaml}"

# Where the generated jsonl + dataset_info.json live (LLaMA-Factory's dataset_dir).
# Defaults to the directory holding the generated data.
DATA_DIR="${DATA_DIR:-${HOME}/data/dr_rlm_sft}"
DATASET_INFO="${DATA_DIR}/dataset_info.json"
ENTRY="${SCRIPT_DIR}/dataset_info_entry.json"

if [ ! -f "${DATA_DIR}/dr_rlm_recursive_sft.jsonl" ]; then
    echo "ERROR: ${DATA_DIR}/dr_rlm_recursive_sft.jsonl not found." >&2
    echo "       Run gen_recursive_sft.py first (see usage at the top of this script)." >&2
    exit 1
fi

# Register the dataset: merge our entry into ${DATA_DIR}/dataset_info.json (create if absent).
# Uses python so an existing dataset_info.json is updated in place rather than clobbered.
mkdir -p "${DATA_DIR}"
python - "${DATASET_INFO}" "${ENTRY}" <<'PY'
import json, os, sys
info_path, entry_path = sys.argv[1], sys.argv[2]
info = {}
if os.path.exists(info_path):
    with open(info_path) as f:
        info = json.load(f)
with open(entry_path) as f:
    entry = json.load(f)
info.update(entry)
with open(info_path, "w") as f:
    json.dump(info, f, indent=2)
print(f"[run_sft] registered {list(entry.keys())} into {info_path}")
PY

# Train. Point LLaMA-Factory at our dataset_dir so it reads the merged dataset_info.json
# and resolves dr_rlm_recursive_sft -> ${DATA_DIR}/dr_rlm_recursive_sft.jsonl.
llamafactory-cli train "${CONFIG}" dataset_dir="${DATA_DIR}"
