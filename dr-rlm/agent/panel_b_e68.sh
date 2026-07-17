#!/bin/bash
# Full 6-benchmark SFT-column panel on the chosen sweep init B_lowlr epoch-2 (global_step_68).
# Reuses the VALIDATED fresh100 panel jobs byte-for-byte, only retargeting MODEL path + OUT dir
# (clones to a temp dir; canonical jobs untouched). B_e68 already grafted -> served dir reused.
# Anchors to compare against (same protocol): untrained / 2ep-canonical(=RL-init) columns.
set -euo pipefail
REPO=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
B68=/scratch-shared/lgehringer/ckpts/sft_recipe_sweep/B_lowlr/hf/global_step_68/served
OUT=$REPO/runs/drrlm_b_e68_thinkon
SFOUT=$REPO/runs/drrlm_b_e68_shortform_thinkon
JOBDIR=/gpfs/scratch1/nodespecific/int6/74327/panel_scratch/b_e68_jobs
mkdir -p "$JOBDIR" "$OUT" "$SFOUT"
F100=dr_rlm_fresh100_final_served

clone(){  # $1=src job basename  -> retargeted job path
  local src=$REPO/agent/$1.job dst=$JOBDIR/$1.b_e68.job
  sed -e "s|/scratch-shared/lgehringer/ckpts/$F100|$B68|g" \
      -e "s|runs/drrlm_fresh100_thinkon|runs/drrlm_b_e68_thinkon|g" \
      "$src" > "$dst"
  echo "$dst"
}

# --- gen-only (HB / DRB / SQA): MODEL=served path, OUT retargeted ---
HB=$(sbatch --parsable --job-name=be68_hb  "$(clone eval_drrlm_fresh100_healthbench)")
DRB=$(sbatch --parsable --job-name=be68_drb "$(clone eval_drrlm_fresh100_drb_gen)")
SQA=$(sbatch --parsable --job-name=be68_sqa "$(clone eval_drrlm_fresh100_sqav2_gen)")
# --- ResearchQA (gen + inline thinking-OFF judge, self-contained) ---
RQA=$(sbatch --parsable --job-name=be68_rqa "$(clone eval_drrlm_fresh100_thinkon)")
# --- short-form (env-parametrized, no clone) ---
SF=$(sbatch --parsable --job-name=be68_sf \
     --export=ALL,GEN_MODEL_PATH="$B68",SERVED_NAME=drrlm_b_e68,TOKENIZER=Qwen/Qwen3.5-4B,OUT="$SFOUT" \
     "$REPO/agent/eval_drrlm_shortform_thinkon.job")

# --- grades off-GPU (staging), afterok their gens ---
GLF=$(sbatch --parsable --dependency=afterok:$HB:$DRB --job-name=be68_grade_lf \
      --export=ALL,OUT="$OUT",TAG=drrlm_b_e68 "$REPO/agent/eval_drtulu_longform_grade.job")
GSQA=$(sbatch --parsable --dependency=afterok:$SQA --job-name=be68_grade_sqa \
      --export=ALL,SQA_INPUT="$OUT/sqav2.jsonl" "$REPO/agent/eval_sqa_grade.job")
GSF=$(sbatch --parsable --dependency=afterok:$SF --job-name=be68_grade_sf \
      --export=ALL,OUT="$SFOUT" "$REPO/agent/eval_drrlm_shortform_grade.job")

echo "GEN:   HB=$HB DRB=$DRB SQA=$SQA RQA=$RQA(gen+grade) SF=$SF"
echo "GRADE: longform(HB+DRB)=$GLF SQA=$GSQA shortform=$GSF"
echo "$HB $DRB $SQA $RQA $SF $GLF $GSQA $GSF" > /tmp/b_e68_panel_ids.txt
