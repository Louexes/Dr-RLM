#!/bin/bash
# Depth-0 (single REPL agent, --max-recursion-depth 0) panel: untrained / SFT(B_e68) / RL(rl_be68 s75)
# on all 6 benchmarks. Untrained = fair no-recursion substrate baseline (base model is native to
# nothing); SFT/RL = lesion ablation (both were trained WITH recursion available, so depth-0
# amputates a trained behaviour rather than baselining a fairly-trained single agent).
# ALL gens use prompts/system_prompt_thinkon_depth0.txt: the depth-0 WORKER band reframed as a
# standalone REPL research agent (no "sub-agent at the leaf" identity, no delegation mentions).
# The 07-12 banked RL long-form gen used the old framing -> archived to *_promptv1, regenerated.
# Mirrors panel_b_e68.sh: clone the validated jobs, retarget only model/out/port/depth/prompt.
set -euo pipefail
REPO=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
UNTR=Qwen/Qwen3.5-4B
SFT=/scratch-shared/lgehringer/ckpts/sft_recipe_sweep/B_lowlr/hf/global_step_68/served
RL=/scratch-shared/lgehringer/rl_be68_exports/global_step_75/served
JOBDIR=/gpfs/scratch1/nodespecific/int6/74327/panel_scratch/depth0_jobs
mkdir -p "$JOBDIR"

lf_clone(){  # $1=model $2=run-dir-name $3=port -> retargeted long-form gen job (DRB50+SQA100+RQA120)
  local dst=$JOBDIR/depth0_lf_$2.job
  sed -e "s|MODEL=/scratch-shared/lgehringer/rl_be68_exports/global_step_75/served|MODEL=$1|" \
      -e "s|runs/rl_be68_depth0_thinkon|runs/$2|g" \
      -e "s|PORT=8016|PORT=$3|" \
      -e "s|--time=04:00:00|--time=05:00:00|" \
      -e "s|system_prompt_thinkon.txt|system_prompt_thinkon_depth0.txt|" \
      "$REPO/agent/eval_drrlm_depth0_efficiency.job" > "$dst"
  echo "$dst"
}

rqa_clone(){  # $1=run-dir-name $2=port -> RQA grade job (local base-Qwen thinking-OFF judge)
  local dst=$JOBDIR/depth0_rqa_grade_$1.job
  sed -e "s|runs/rl_be68_final_thinkon|runs/$1|g" \
      -e "s|researchqa_clean.jsonl|researchqa.jsonl|" \
      -e "s|PORT=8017|PORT=$2|" \
      "$REPO/agent/grade_rqa_rl_be68.job" > "$dst"
  echo "$dst"
}

# short-form job stays env-parametrized; depth + prompt swapped via clone
SFJOB=$JOBDIR/depth0_shortform.job
sed -e 's|--max-recursion-depth 1|--max-recursion-depth 0|' \
    -e 's|system_prompt_thinkon.txt|system_prompt_thinkon_depth0.txt|' \
  "$REPO/agent/eval_drrlm_shortform_thinkon.job" > "$SFJOB"

# archive the 07-12 old-prompt RL long-form gen (regenerated below with the depth-0 prompt)
if [ -d "$REPO/runs/rl_be68_depth0_thinkon" ] && [ ! -d "$REPO/runs/rl_be68_depth0_thinkon_promptv1" ]; then
  mv "$REPO/runs/rl_be68_depth0_thinkon" "$REPO/runs/rl_be68_depth0_thinkon_promptv1"
fi

# --- long-form gens x3 ---
ULF=$(sbatch --parsable --job-name=d0_untr_lf "$(lf_clone "$UNTR" drrlm_untrained_depth0_thinkon 8021)")
SLF=$(sbatch --parsable --job-name=d0_sft_lf  "$(lf_clone "$SFT"  drrlm_b_e68_depth0_thinkon    8022)")
RLF=$(sbatch --parsable --job-name=d0_rl_lf   "$(lf_clone "$RL"   rl_be68_depth0_thinkon        8020)")

# --- short-form gens x3 ---
USF=$(sbatch --parsable --job-name=d0_untr_sf --export=ALL,GEN_MODEL_PATH="$UNTR",SERVED_NAME=Qwen/Qwen3.5-4B,TOKENIZER=Qwen/Qwen3.5-4B,PORT=8023,OUT=$REPO/runs/drrlm_untrained_depth0_shortform_thinkon "$SFJOB")
SSF=$(sbatch --parsable --job-name=d0_sft_sf  --export=ALL,GEN_MODEL_PATH="$SFT",SERVED_NAME=drrlm_b_e68_d0,TOKENIZER=Qwen/Qwen3.5-4B,PORT=8024,OUT=$REPO/runs/drrlm_b_e68_depth0_shortform_thinkon "$SFJOB")
RSF=$(sbatch --parsable --job-name=d0_rl_sf   --export=ALL,GEN_MODEL_PATH="$RL",SERVED_NAME=rl_be68_d0,TOKENIZER=Qwen/Qwen3.5-4B,PORT=8025,OUT=$REPO/runs/rl_be68_depth0_shortform_thinkon "$SFJOB")

# --- grades: DRB Gemini (staging), SQA astabench/Gemini (staging), RQA local judge (GPU) ---
GLF_U=$(sbatch --parsable --dependency=afterok:$ULF --job-name=d0_untr_glf --export=ALL,OUT=$REPO/runs/drrlm_untrained_depth0_thinkon,TAG=untr_depth0 "$REPO/agent/eval_drtulu_longform_grade.job")
GLF_S=$(sbatch --parsable --dependency=afterok:$SLF --job-name=d0_sft_glf  --export=ALL,OUT=$REPO/runs/drrlm_b_e68_depth0_thinkon,TAG=b_e68_depth0 "$REPO/agent/eval_drtulu_longform_grade.job")
GLF_R=$(sbatch --parsable --dependency=afterok:$RLF --job-name=d0_rl_glf --export=ALL,OUT=$REPO/runs/rl_be68_depth0_thinkon,TAG=rl_be68_depth0 "$REPO/agent/eval_drtulu_longform_grade.job")

GSQA_U=$(sbatch --parsable --dependency=afterok:$ULF --job-name=d0_untr_gsqa --export=ALL,SQA_INPUT=$REPO/runs/drrlm_untrained_depth0_thinkon/sqav2.jsonl "$REPO/agent/eval_sqa_grade.job")
GSQA_S=$(sbatch --parsable --dependency=afterok:$SLF --job-name=d0_sft_gsqa  --export=ALL,SQA_INPUT=$REPO/runs/drrlm_b_e68_depth0_thinkon/sqav2.jsonl "$REPO/agent/eval_sqa_grade.job")
GSQA_R=$(sbatch --parsable --dependency=afterok:$RLF --job-name=d0_rl_gsqa --export=ALL,SQA_INPUT=$REPO/runs/rl_be68_depth0_thinkon/sqav2.jsonl "$REPO/agent/eval_sqa_grade.job")

GRQA_U=$(sbatch --parsable --dependency=afterok:$ULF --job-name=d0_untr_grqa "$(rqa_clone drrlm_untrained_depth0_thinkon 8026)")
GRQA_S=$(sbatch --parsable --dependency=afterok:$SLF --job-name=d0_sft_grqa  "$(rqa_clone drrlm_b_e68_depth0_thinkon 8027)")
GRQA_R=$(sbatch --parsable --dependency=afterok:$RLF --job-name=d0_rl_grqa "$(rqa_clone rl_be68_depth0_thinkon 8028)")

GSF_U=$(sbatch --parsable --dependency=afterok:$USF --job-name=d0_untr_gsf --export=ALL,OUT=$REPO/runs/drrlm_untrained_depth0_shortform_thinkon "$REPO/agent/eval_drrlm_shortform_grade.job")
GSF_S=$(sbatch --parsable --dependency=afterok:$SSF --job-name=d0_sft_gsf  --export=ALL,OUT=$REPO/runs/drrlm_b_e68_depth0_shortform_thinkon "$REPO/agent/eval_drrlm_shortform_grade.job")
GSF_R=$(sbatch --parsable --dependency=afterok:$RSF --job-name=d0_rl_gsf   --export=ALL,OUT=$REPO/runs/rl_be68_depth0_shortform_thinkon "$REPO/agent/eval_drrlm_shortform_grade.job")

# --- SBU guard: stall watchdogs on every GPU job (liveness = vllm log + slurm .out) ---
wd(){ nohup "$REPO/agent/kill_on_stall.sh" "$1" 30 $2 > "$JOBDIR/wd_$1.log" 2>&1 & }
wd "$ULF" "$REPO/runs/logs/drrlm_depth0_vllm_${ULF}.log $REPO/runs/logs/drrlm_depth0_${ULF}.out"
wd "$SLF" "$REPO/runs/logs/drrlm_depth0_vllm_${SLF}.log $REPO/runs/logs/drrlm_depth0_${SLF}.out"
wd "$RLF" "$REPO/runs/logs/drrlm_depth0_vllm_${RLF}.log $REPO/runs/logs/drrlm_depth0_${RLF}.out"
wd "$USF" "$REPO/runs/logs/drrlm_sf_ton_vllm_${USF}.log $REPO/runs/logs/drrlm_sf_ton_${USF}.out"
wd "$SSF" "$REPO/runs/logs/drrlm_sf_ton_vllm_${SSF}.log $REPO/runs/logs/drrlm_sf_ton_${SSF}.out"
wd "$RSF" "$REPO/runs/logs/drrlm_sf_ton_vllm_${RSF}.log $REPO/runs/logs/drrlm_sf_ton_${RSF}.out"
wd "$GRQA_U" "$REPO/runs/logs/grade_rqa_rlbe68_vllm_${GRQA_U}.log $REPO/runs/logs/grade_rqa_rlbe68_${GRQA_U}.out"
wd "$GRQA_S" "$REPO/runs/logs/grade_rqa_rlbe68_vllm_${GRQA_S}.log $REPO/runs/logs/grade_rqa_rlbe68_${GRQA_S}.out"
wd "$GRQA_R" "$REPO/runs/logs/grade_rqa_rlbe68_vllm_${GRQA_R}.log $REPO/runs/logs/grade_rqa_rlbe68_${GRQA_R}.out"

echo "GEN:    untrLF=$ULF sftLF=$SLF rlLF=$RLF  untrSF=$USF sftSF=$SSF rlSF=$RSF"
echo "GRADE:  drb+gemini U/S/R=$GLF_U/$GLF_S/$GLF_R  sqa U/S/R=$GSQA_U/$GSQA_S/$GSQA_R  rqa U/S/R=$GRQA_U/$GRQA_S/$GRQA_R  sf U/S/R=$GSF_U/$GSF_S/$GSF_R"
echo "$ULF $SLF $RLF $USF $SSF $RSF $GLF_U $GLF_S $GLF_R $GSQA_U $GSQA_S $GSQA_R $GRQA_U $GRQA_S $GRQA_R $GSF_U $GSF_S $GSF_R" > "$JOBDIR/panel_depth0_ids.txt"
