#!/bin/bash
# Full SFT-column panel on the chosen flat init (dr_tulu_flat_sft_lowlr epoch 1 = global_step_13).
# Mirrors panel_b_e68.sh for the flat arm: reuses the env-parametrized DR-Tulu flat eval jobs,
# pointing GEN at the already-grafted served_e13 dir. Grades off-GPU (Gemini, staging).
# ResearchQA already measured by the gate (cov 0.295) -> not re-run here.
set -euo pipefail
REPO=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
SERVED=/scratch-shared/lgehringer/ckpts/dr_tulu_flat_sft_lowlr/served_e13
NAME=drtulu_flat_e13
TOK=Qwen/Qwen3.5-4B
LFOUT=$REPO/runs/flat_e13_longform
SFOUT=$REPO/runs/flat_e13_shortform
mkdir -p "$LFOUT" "$SFOUT"

# --- longform gen (HB120 + DRB50 + SQA100), thinking-ON ---
LF=$(sbatch --parsable --job-name=flate13_lf \
     --export=ALL,GEN_MODEL_PATH="$SERVED",SERVED_NAME="$NAME",TOKENIZER="$TOK",OUT="$LFOUT" \
     "$REPO/agent/eval_drtulu_longform_gen.job")
# --- shortform gen (SimpleQA + 2Wiki + WebWalker), thinking-ON ---
SF=$(sbatch --parsable --job-name=flate13_sf \
     --export=ALL,GEN_MODEL_PATH="$SERVED",SERVED_NAME="$NAME",TOKENIZER="$TOK",OUT="$SFOUT" \
     "$REPO/agent/eval_drtulu_shortform_thinkon.job")

# --- grades off-GPU (staging), afterok their gens ---
GLF=$(sbatch --parsable --dependency=afterok:$LF --job-name=flate13_grade_lf \
      --export=ALL,OUT="$LFOUT",TAG="$NAME" "$REPO/agent/eval_drtulu_longform_grade.job")
GSQA=$(sbatch --parsable --dependency=afterok:$LF --job-name=flate13_grade_sqa \
      --export=ALL,SQA_INPUT="$LFOUT/sqav2.jsonl" "$REPO/agent/eval_sqa_grade.job")
GSF=$(sbatch --parsable --dependency=afterok:$SF --job-name=flate13_grade_sf \
      --export=ALL,OUT="$SFOUT" "$REPO/agent/eval_drtulu_shortform_grade.job")

echo "GEN:   longform=$LF shortform=$SF"
echo "GRADE: longform(HB+DRB)=$GLF SQA=$GSQA shortform=$GSF"
echo "$LF $SF $GLF $GSQA $GSF" > /tmp/flat_e13_panel_ids.txt
echo "RQA: reuse gate result cov=0.295 (runs/drtulu_flat_gate_e13)"
