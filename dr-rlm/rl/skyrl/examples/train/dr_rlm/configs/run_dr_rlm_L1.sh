set -x

# =============================================================================
# DR-RLM ablation ladder — RUNG L1 : RECURSION + INHERITED SCALAR (SkyRL-style)
# -----------------------------------------------------------------------------
# Proposal §6 ladder (Recursive Deep Research Proposal.md, "Ablation ladder"):
#   | L1 | Recursive + root-only RLER + advantage inheritance | recursion alone (SkyRL) |
#
# L1 isolates: RECURSION ALONE. The root delegates to child agents, but the tree
# is flattened into ONE root trajectory and the single root rubric scalar is
# broadcast to every node (advantage inheritance) — exactly the SkyRL/NovaSky
# scheme. Comparing L1 vs L0 measures the value of recursion under scalar credit;
# comparing L3 vs L1 is RQ2 (does provenance-attributed credit beat inheritance).
#
# Mechanism knobs that make this rung L1:
#   generator.max_recursion_depth=2      -> grandchildren allowed (the proposal target tree)
#   generator.enable_child_agents=true   -> subcall_fn injected so the root can delegate
#   generator.train_child_trajectories=true -> child trajectories enter the training batch...
#   generator.per_node_credit=false      -> ...but stay flattened+broadcast (NO un-flattening)
#   generator.reward_mode=inherited      -> only the root is scored; children get 0 then inherit
#   trainer.algorithm.advantage_estimator=grpo
#
# RUN ORDER (see configs/README.md): run_judge.sh -> [run_retriever.sh] ->
#   data convert (-> $DATA_DIR/{train,validation}.parquet) -> THIS script.
# =============================================================================

# ----- cluster / engine sizing -----
: "${DATA_DIR:=$HOME/data/dr-rlm}"
: "${MODEL_PATH:=Qwen/Qwen3-8B}"
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${TRAIN_GPUS:=4}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

# ----- held-constant judge + retriever env (IDENTICAL across L0..L4) -----
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
: "${JUDGE_BASE_URL:=http://localhost:8100/v1}"   # MUST match the port run_judge.sh serves on
: "${JUDGE_MODEL:=hosted_vllm/Qwen/Qwen3-8B}"
: "${SEARCH_BACKEND:=local_jsonl}"
: "${SEARCH_CORPUS_PATH:=$DATA_DIR/corpus.jsonl}"
: "${SEARCH_INDEX_PATH:=$DATA_DIR/bm25}"
: "${SEARCH_ENDPOINT:=http://localhost:8003/mcp}"

uv run --extra fsdp --python 3.12 -m examples.train.dr_rlm.main_dr_rlm \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=dr_rlm \
  generator.step_wise_trajectories=true \
  `# ---- L1 mechanism selector ----` \
  generator.max_recursion_depth=2 \
  generator.enable_child_agents=true \
  generator.train_child_trajectories=true \
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
  trainer.run_name="dr_rlm_L1_recursion_inherited" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.ckpt_path="$(pwd)/.neer/artifacts/ckpts/dr_rlm_L1" \
  trainer.export_path="$(pwd)/.neer/artifacts/dr_rlm_exports" \
  trainer.dump_eval_results=true \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  "$@"
