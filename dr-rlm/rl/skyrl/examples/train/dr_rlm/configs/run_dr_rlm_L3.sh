set -x

# =============================================================================
# DR-RLM ablation ladder — RUNG L3 : RER PROVENANCE PER-NODE CREDIT  (THE CORE)
# -----------------------------------------------------------------------------
# Proposal §6 ladder (Recursive Deep Research Proposal.md, "Ablation ladder"):
#   | L3 | + provenance-attributed per-node credit | RER core (RQ2) |
#
# L3 isolates: THE THESIS CONTRIBUTION (C1). Each node's terminal reward r_a is
# the report rubric mass attributed to it through the citation/provenance graph:
#   r_a = Σ_{c : a supports c} w_c · s_c · share(a,c)  −  gamma · cost(a)
# share(a,c) splits each criterion's mass across the nodes whose cited evidence
# supports it; because shares sum to 1, Σ_a r_a = R (credit is conserved, just
# redistributed by who-earned-it). This is the cheap *structural difference
# reward* (no counterfactual rollouts). RQ2 = L3 vs {L1, L2}.
#
# NOTE on the estimator: once the generator un-flattens the tree, every node is its
# own step-wise trajectory carrying its own r_a, and ALL of a prompt's nodes share the
# prompt uid as instance_id (this is forced — SkyRL builds GRPO groups AND mini-batch
# boundaries from instance_id, asserting each value is contiguous and #distinct ==
# train_batch_size; depth-encoding the instance_id would break that). So stock GRPO
# baselines each node against the *mean credit of all the prompt's nodes* — a per-prompt
# difference-reward baseline. L3 uses advantage_estimator=grpo to isolate the credit
# *scheme* (rer vs rao/inherited). The depth-cohort baseline + RAO depth-weighting is the
# rer_pernode estimator (+ DrRlmTrainer), the RQ4 variant exercised by L2.
#
# Mechanism knobs that make this rung L3:
#   generator.per_node_credit=true       -> un-flatten: each node its own trajectory + r_a
#   generator.reward_mode=rer            -> provenance-attributed credit (the core)
#   generator.share_mode=citation_count  -> split each criterion ∝ #cited snippets per node
#                                           (ledger_support is an off-by-default ablation; see
#                                            dr_rlm_config.py — gate it on an orphan-rate check)
#   generator.max_recursion_depth=2      -> grandchildren allowed (proposal target tree)
#   trainer.algorithm.advantage_estimator=grpo  -> stock per-prompt GRPO baseline over nodes
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

# NODE-FIX (2026-06-24): launch via the venv python DIRECTLY, not `uv run`. On the current
# gpu_h100 nodes Ray detects `uv run` (process-tree) and re-launches EVERY worker via
# `uv run`, which re-resolves the env per-worker and HANGS ("workers not registered within
# timeout" / ray._private). Proven on-node by ray_diag2: uv-run hangs, ./.venv/bin/python works.
# The .venv already carries the --extra fsdp deps from prior syncs; workers read code+venv off /gpfs.
env -u VIRTUAL_ENV ./.venv/bin/python -m examples.train.dr_rlm.main_dr_rlm \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=dr_rlm \
  generator.step_wise_trajectories=true \
  `# ---- L3 mechanism selector (THE CORE) ----` \
  generator.max_recursion_depth=2 \
  generator.enable_child_agents=true \
  generator.train_child_trajectories=true \
  generator.per_node_credit=true \
  generator.reward_mode=rer \
  generator.share_mode=citation_count \
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
  trainer.run_name="dr_rlm_L3_rer_provenance" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.ckpt_path="$(pwd)/.neer/artifacts/ckpts/dr_rlm_L3" \
  trainer.export_path="$(pwd)/.neer/artifacts/dr_rlm_exports" \
  trainer.dump_eval_results=true \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  "$@"
