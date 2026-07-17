set -x

# =============================================================================
# DR-Tulu FLAT BASELINE RL — faithful Open-Instruct recipe on SkyRL
# -----------------------------------------------------------------------------
# The flat arm of the controlled comparison: DR-Tulu trained as its authors
# trained it (docs/DRTULU_RL_FAITHFULNESS_AUDIT.md), inside the SAME SkyRL
# trainer as the recursive arm.
#
#   env        : dr_tulu (dr_tulu_env.DrTuluEnv) — ReAct <call_tool>/<answer>,
#                snippet_search over the frozen bm25s corpus, tool outputs as
#                loss-masked user turns (byte-identical to SFT/eval format)
#   reward     : original composite 0.5·rubric + 0.2·citation + 0.2·format +
#                0.1·search_turns (citation via the local judge; static rubrics
#                = their apply_adaptive_rubric_reward=false ablation)
#   optimizer  : stock GRPO — std-norm, k3 KL-in-loss, clip 0.2, token-mean,
#                on-policy 1 pass (all SkyRL defaults = original's exact shape);
#                lr/KL-coef/batch held to the RECURSIVE arm's values for
#                cross-arm parity (audit §2B)
#   rollout    : stop on </call_tool> (kept in output, EOS appended), T=1.0,
#                top_p=1.0, thinking-ON, max 12 tool calls (= eval convention)
#
# RUN ORDER: run_judge.sh (judge endpoint) -> data (data/rl_full_600_drtulu)
#   -> THIS script. No retriever server needed (bm25s is in-process).
# =============================================================================

: "${DATA_DIR:=$HOME/Dr-RLM/dr-rlm/data/rl_fresh_400_drtulu}"  # paired 400 prompts (= recursive rl_fresh_400) -> 100 steps
: "${MODEL_PATH:=/scratch-shared/lgehringer/ckpts/dr_tulu_flat_sft_CANONICAL_1ep_served}"
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${TRAIN_GPUS:=4}"
: "${INFERENCE_BACKEND:=vllm}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

# ----- held-constant judge + retriever (IDENTICAL to the recursive arm) -----
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
: "${JUDGE_BASE_URL:=http://localhost:8100/v1}"
: "${JUDGE_MODEL:=Qwen/Qwen3.5-4B}"
export JUDGE_BASE_URL JUDGE_MODEL
: "${SEARCH_BACKEND:=bm25s}"
: "${SEARCH_CORPUS_PATH:=$HOME/Dr-RLM/dr-rlm/data/frozen_corpus/corpus.jsonl}"
: "${SEARCH_INDEX_PATH:=$HOME/Dr-RLM/dr-rlm/data/frozen_corpus/bm25s_index}"
export SEARCH_BACKEND SEARCH_CORPUS_PATH SEARCH_INDEX_PATH
# ----- flat-arm knobs (audit §2B: 12 calls = our eval convention) -----
export DR_TULU_MAX_TOOL_CALLS="${DR_TULU_MAX_TOOL_CALLS:-12}"
export DR_TULU_SEARCH_K="${DR_TULU_SEARCH_K:-10}"
export DR_TULU_JUDGE_CONCURRENCY="${DR_TULU_JUDGE_CONCURRENCY:-5}"

uv run --extra fsdp --python 3.12 -m examples.train.dr_rlm.main_dr_tulu \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=dr_tulu \
  `# ---- ReAct rollout: stop at tool-call closers, 12 calls + answer margin ----` \
  generator.max_turns=14 \
  generator.batched=false \
  generator.sampling_params.stop="['</call_tool>','</call>']" \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.max_input_length=32768 \
  generator.chat_template_kwargs.enable_thinking=true \
  generator.n_samples_per_prompt=4 \
  `# ---- stock GRPO = original optimization shape (defaults: std-norm, k3, 0.2 clip, token-mean) ----` \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.05 \
  `#  ^ matches the recursive thesis run rl_fresh100 (probe-validated; 0.01 caused quality contraction)` \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  `# ---- infra (same as recursive arm) ----` \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAIN_GPUS \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=32768 \
  trainer.eval_before_train=false \
  trainer.eval_interval=1000 \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=25 \
  trainer.max_ckpts_to_keep=2 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="dr-rlm" \
  trainer.run_name="${RUN_NAME:-dr_tulu_flat_rl}" \
  trainer.log_path="${LOG_PATH:-$(pwd)/.neer/artifacts/skyrl-logs}" \
  trainer.ckpt_path="${CKPT_PATH:-/scratch-shared/lgehringer/dr_tulu_flat_rl_ckpts}" \
  trainer.export_path="${EXPORT_PATH:-/scratch-shared/lgehringer/dr_tulu_flat_rl_exports}" \
  trainer.dump_eval_results=true \
  "$@"
