#!/bin/bash
# Full RL-column panel on the FINAL flat DR-Tulu RL ckpt (rl_drtulu_flat step 75).
# Grafts the text export -> served (staging, afterok), then runs the flat eval panel
# (longform HB+DRB+SQA gen, shortform, RQA gen+grade) and grades off-GPU. Mirror of
# panel_rl_be68.sh for the flat arm; reuses the env-parametrized DR-Tulu eval jobs.
#   FIRE THIS once rl_drtulu_flat reaches step 75 (export present under global_step_75/policy).
# Anchors: flat untrained + flat SFT e13 (avg 0.287). Does RL beat SFT+untrained on the flat arm?
set -euo pipefail
REPO=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
STEP=${STEP:-75}
SERVED=/scratch-shared/lgehringer/rl_drtulu_flat_exports/global_step_${STEP}/served
NAME=drtulu_flat_rl
TOK=Qwen/Qwen3.5-4B
LFOUT=$REPO/runs/rl_drtulu_flat_final_longform
SFOUT=$REPO/runs/rl_drtulu_flat_final_shortform
RQAOUT=$REPO/runs/rl_drtulu_flat_final_rqa
mkdir -p "$LFOUT" "$SFOUT" "$RQAOUT"

# --- graft first (staging, free); everything else afterok it ---
GRAFT=$(sbatch --parsable --export=ALL,STEP="$STEP" "$REPO/agent/graft_rl_drtulu_flat.job")
echo "graft(staging)=$GRAFT"
DEP="--dependency=afterok:$GRAFT"

# --- gen: longform (HB+DRB+SQA), shortform, RQA(gen+grade self-contained) ---
LF=$(sbatch --parsable $DEP --job-name=rldtf_lf \
     --export=ALL,GEN_MODEL_PATH="$SERVED",SERVED_NAME="$NAME",TOKENIZER="$TOK",OUT="$LFOUT" \
     "$REPO/agent/eval_drtulu_longform_gen.job")
SF=$(sbatch --parsable $DEP --job-name=rldtf_sf \
     --export=ALL,GEN_MODEL_PATH="$SERVED",SERVED_NAME="$NAME",TOKENIZER="$TOK",OUT="$SFOUT" \
     "$REPO/agent/eval_drtulu_shortform_thinkon.job")
RQA=$(sbatch --parsable $DEP --job-name=rldtf_rqa \
     --export=ALL,SERVED="$SERVED",GEN_NAME="$NAME",OUT="$RQAOUT" \
     "$REPO/agent/eval_drtulu_flat_sft_gate.job")

# --- grades off-GPU (staging), afterok their gens ---
GLF=$(sbatch --parsable --dependency=afterok:$LF --job-name=rldtf_grade_lf \
      --export=ALL,OUT="$LFOUT",TAG="$NAME" "$REPO/agent/eval_drtulu_longform_grade.job")
GSQA=$(sbatch --parsable --dependency=afterok:$LF --job-name=rldtf_grade_sqa \
      --export=ALL,SQA_INPUT="$LFOUT/sqav2.jsonl" "$REPO/agent/eval_sqa_grade.job")
GSF=$(sbatch --parsable --dependency=afterok:$SF --job-name=rldtf_grade_sf \
      --export=ALL,OUT="$SFOUT" "$REPO/agent/eval_drtulu_shortform_grade.job")

echo "GEN:   longform(HB+DRB+SQA)=$LF shortform=$SF RQA(gen+grade)=$RQA  (all afterok graft $GRAFT)"
echo "GRADE: longform(HB+DRB)=$GLF SQA=$GSQA shortform=$GSF"
echo "$GRAFT $LF $SF $RQA $GLF $GSQA $GSF" > /tmp/rl_drtulu_flat_panel_ids.txt
echo "RQA result -> $RQAOUT/researchqa_eval_results.json (self-graded thinking-off)"
