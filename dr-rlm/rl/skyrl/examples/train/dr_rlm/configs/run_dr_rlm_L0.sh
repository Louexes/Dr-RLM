set -x

# =============================================================================
# DR-RLM ablation ladder — RUNG L0 : FLAT BASELINE (no recursion)
# -----------------------------------------------------------------------------
# Proposal §6 ladder (Recursive Deep Research Proposal.md, "Ablation ladder"):
#   | L0 | Flat RLER = DR Tulu-8B (free, downloadable) | flat baseline |
#
# L0 isolates: the flat (single-agent) policy. NO child agents, NO recursion,
# stock GRPO over one root trajectory scored by the held-constant rubric judge.
# This is the SkyRL/DR-Tulu-shaped control that L1..L4 are measured against.
#
# Mechanism knobs that make this rung L0 (vs the others):
#   generator.max_recursion_depth=0      -> recursion disabled entirely (env withholds subcall_fn)
#   generator.enable_child_agents=false  -> no subcall_fn injection at all (single agent)
#   generator.per_node_credit=false      -> stock flatten+broadcast (one root trajectory)
#   generator.reward_mode=inherited      -> only the root is scored (children would get 0 anyway)
#   trainer.algorithm.advantage_estimator=grpo
#
# RUN ORDER (see configs/README.md): run_judge.sh -> [run_retriever.sh] ->
#   data convert (-> $DATA_DIR/{train,validation}.parquet) -> THIS script.
# =============================================================================

# ----- cluster / engine sizing (override via env; same knobs as run_multi_paper_rlm.sh) -----
: "${DATA_DIR:=$HOME/data/dr-rlm}"
: "${MODEL_PATH:=Qwen/Qwen3-8B}"          # SFT/base policy for the controlled comparison (held constant L0..L4)
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${TRAIN_GPUS:=4}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

# ----- held-constant judge + retriever env (IDENTICAL across L0..L4) -----
# Judge: local Qwen3-8B served as an OpenAI-compatible endpoint (see run_judge.sh).
# JUDGE_API_KEY is read by JudgeConfig.from_env_payload via judge_api_key_env=JUDGE_API_KEY.
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
: "${JUDGE_BASE_URL:=http://localhost:8100/v1}"   # MUST match the port run_judge.sh serves on
: "${JUDGE_MODEL:=hosted_vllm/Qwen/Qwen3-8B}"      # litellm-style prefix is stripped by JudgeConfig
# Retriever: held constant across arms (see run_retriever.sh for the two options).
: "${SEARCH_BACKEND:=local_jsonl}"                 # local_jsonl | bm25 | faiss | mcp_http | none
: "${SEARCH_CORPUS_PATH:=$DATA_DIR/corpus.jsonl}"
: "${SEARCH_INDEX_PATH:=$DATA_DIR/bm25}"
: "${SEARCH_ENDPOINT:=http://localhost:8003/mcp}"

uv run --extra fsdp --python 3.12 -m examples.train.dr_rlm.main_dr_rlm \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=dr_rlm \
  generator.step_wise_trajectories=true \
  `# ---- L0 mechanism selector ----` \
  generator.max_recursion_depth=0 \
  generator.enable_child_agents=false \
  generator.train_child_trajectories=false \
  generator.per_node_credit=false \
  generator.reward_mode=inherited \
  trainer.algorithm.advantage_estimator="grpo" \
  `# ---- held-constant judge ----` \
  generator.judge_model="$JUDGE_MODEL" \
  generator.judge_base_url="$JUDGE_BASE_URL" \
  generator.judge_api_key_env=JUDGE_API_KEY \
  `# ---- held-constant retriever ----` \
  generator.search_backend="$SEARCH_BACKEND" \
  generator.search_corpus_path="$SEARCH_CORPUS_PATH" \
  generator.search_index_path="$SEARCH_INDEX_PATH" \
  generator.search_endpoint="$SEARCH_ENDPOINT" \
  `# ---- canonical RLM/SkyRL hyperparameters (copied from run_multi_paper_rlm.sh) ----` \
  generator.max_turns=6 \
  generator.batched=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAIN_GPUS \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3DecoderLayer']" \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=16 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=100 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=32768 \
  generator.sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.01 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.max_input_length=32768 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.inference_engine.enforce_eager=false \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.n_samples_per_prompt=8 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="dr-rlm" \
  trainer.run_name="dr_rlm_L0_flat" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.ckpt_path="$(pwd)/.neer/artifacts/ckpts/dr_rlm_L0" \
  trainer.export_path="$(pwd)/.neer/artifacts/dr_rlm_exports" \
  trainer.dump_eval_results=true \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  "$@"
