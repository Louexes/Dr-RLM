#!/bin/bash
# SFT recipe sweep — anti-narrowing probe (2026-07-11).
# Q: can a cheap full-FT recipe change lift the SFT init to >= untrained on DRB/SQA
#    while keeping the recursion/finalization gate? If yes -> 50-step RL x3 plan.
#    If no -> narrowing is intrinsic, fall back to matched 100-step plan (banked run intact).
#
# EACH config trains to 4 EPOCHS with one HF export PER EPOCH (sft_trainer.py:913 subfolders by
# global_step_{step}/policy), so we get the epoch axis for free: gate/DRB-eval epochs 2/3/4 post-hoc,
# pick the least-narrowed one that keeps recursion. Epoch-1 skipped (known to under-install: 15% rec).
# LR = cosine over full num_steps -> intermediate epochs are warmer/less-annealed than a dedicated
# k-epoch run (more entropy retained = aligned with anti-narrowing; valid RL init, RL re-inits its LR).
#
# All overrides are Hydra last-wins args to the VALIDATED job (agent/sft_train_drrlm_sweep.job);
# job file untouched. LoRA excluded (not a flag in fsdp2 main_sft; round-2 escalation only).
# Baseline anchor = banked 2ep-canonical (data=266, lr=4e-5, wd=0); NOT re-run here.
set -euo pipefail
REPO=/gpfs/home5/lgehringer/Dr-RLM/dr-rlm
JOB=$REPO/agent/sft_train_drrlm_sweep.job
P266=$REPO/data/sft/primary/dr_rlm_recursive_primary.jsonl   # 266 rows -> 34 steps/epoch @ bs8
P448=$REPO/data/sft/ablation/dr_rlm_recursive_full_clean.jsonl     # 448 rows -> 56 steps/epoch @ bs8, same schema
CKROOT=/scratch-shared/lgehringer/ckpts/sft_recipe_sweep

submit(){  # $1=tag  $2..=extra hydra overrides
  local tag="$1"; shift
  local ck="$CKROOT/$tag"; mkdir -p "$ck"
  local jid
  jid=$(sbatch --parsable --time=05:00:00 --job-name="sftrec_$tag" "$JOB" \
      run_name="sft_recipe_$tag" \
      ckpt_path="$ck" export_path="$ck/hf" \
      "$@")
  echo "  [$tag] job=$jid  ckpts -> $ck/hf/global_step_*/policy"
}

echo "=== SFT recipe sweep: 3 configs x 4 epochs, per-epoch HF export (~3-4h each, 2xH100) ==="
# A: more data, same register. 448 rows, 4ep=224 steps, save every 56 -> ep 1/2/3/4 @ 56/112/168/224
submit A_moredata dataset_name="$P448" num_steps=224 optimizer_config.num_warmup_steps=22 \
                  optimizer_config.lr=4e-5 optimizer_config.weight_decay=0.0 hf_save_interval=56
# B: gentler update. 266 rows, lr 2e-5, 4ep=136 steps, save every 34 -> ep 1/2/3/4 @ 34/68/102/136
submit B_lowlr     dataset_name="$P266" num_steps=136 optimizer_config.num_warmup_steps=14 \
                  optimizer_config.lr=2e-5 optimizer_config.weight_decay=0.0 hf_save_interval=34
# C: explicit regularizer. 266 rows, wd 0.1, lr 4e-5, 4ep=136 steps, save every 34
submit C_wd        dataset_name="$P266" num_steps=136 optimizer_config.num_warmup_steps=14 \
                  optimizer_config.lr=4e-5 optimizer_config.weight_decay=0.1 hf_save_interval=34
echo "=== submitted. next: graft+gate epochs 2/3/4 (behavior-first), DRB-24 on survivors ==="
