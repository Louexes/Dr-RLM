#!/bin/bash
# Full RL-column panel on the FINAL main-arm ckpt (rl_bc_be68 step_75, provenance credit).
# Grafts the text export -> served (staging, afterok), then reuses the recursive fresh100 panel
# jobs byte-for-byte, retargeted to the RL served dir. Grades off-GPU (Gemini/local). The headline
# RL column: does RL beat SFT (B_e68) and untrained across the board.
set -euo pipefail
REPO=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
SERVED=/scratch-shared/lgehringer/rl_bc_be68_exports/global_step_75/served
OUT=$REPO/runs/rl_bc_be68_final_thinkon
SFOUT=$REPO/runs/rl_bc_be68_final_shortform_thinkon
JOBDIR=/gpfs/scratch1/nodespecific/int6/74327/panel_scratch/rl_bc_be68_panel_jobs
mkdir -p "$JOBDIR" "$OUT" "$SFOUT"
F100=dr_rlm_fresh100_final_served

# --- graft first (staging, free), everything else afterok it ---
GRAFT=$(sbatch --parsable "$REPO/agent/graft_rl_bc_be68.job")
echo "graft(staging)=$GRAFT"

clone(){  # $1=src job basename -> retargeted job path
  local src=$REPO/agent/$1.job dst=$JOBDIR/$1.rlbe68.job
  sed -e "s|/scratch-shared/lgehringer/ckpts/$F100|$SERVED|g" \
      -e "s|runs/drrlm_fresh100_thinkon|runs/rl_bc_be68_final_thinkon|g" \
      "$src" > "$dst"
  echo "$dst"
}

DEP="--dependency=afterok:$GRAFT"
HB=$(sbatch --parsable $DEP --job-name=rlbc68_hb  "$(clone eval_drrlm_fresh100_healthbench)")
DRB=$(sbatch --parsable $DEP --job-name=rlbc68_drb "$(clone eval_drrlm_fresh100_drb_gen)")
SQA=$(sbatch --parsable $DEP --job-name=rlbc68_sqa "$(clone eval_drrlm_fresh100_sqav2_gen)")
RQA=$(sbatch --parsable $DEP --job-name=rlbc68_rqa "$(clone eval_drrlm_fresh100_thinkon)")
SF=$(sbatch --parsable $DEP --job-name=rlbc68_sf \
     --export=ALL,GEN_MODEL_PATH="$SERVED",SERVED_NAME=rl_bc_be68_final,TOKENIZER=Qwen/Qwen3.5-4B,OUT="$SFOUT" \
     "$REPO/agent/eval_drrlm_shortform_thinkon.job")

# grades off-GPU, afterok their gens
GLF=$(sbatch --parsable --dependency=afterok:$HB:$DRB --job-name=rlbc68_grade_lf \
      --export=ALL,OUT="$OUT",TAG=rl_bc_be68_final "$REPO/agent/eval_drtulu_longform_grade.job")
GSQA=$(sbatch --parsable --dependency=afterok:$SQA --job-name=rlbc68_grade_sqa \
      --export=ALL,SQA_INPUT="$OUT/sqav2.jsonl" "$REPO/agent/eval_sqa_grade.job")
GSF=$(sbatch --parsable --dependency=afterok:$SF --job-name=rlbc68_grade_sf \
      --export=ALL,OUT="$SFOUT" "$REPO/agent/eval_drrlm_shortform_grade.job")

echo "GEN:   HB=$HB DRB=$DRB SQA=$SQA RQA=$RQA(gen+grade) SF=$SF  (all afterok graft $GRAFT)"
echo "GRADE: longform(HB+DRB)=$GLF SQA=$GSQA shortform=$GSF"
echo "$GRAFT $HB $DRB $SQA $RQA $SF $GLF $GSQA $GSF" > /tmp/rl_bc_be68_panel_ids.txt
