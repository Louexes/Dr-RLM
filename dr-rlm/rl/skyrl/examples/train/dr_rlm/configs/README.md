# DR-RLM launch scripts — the L0-L4 ablation ladder

These scripts launch the controlled ablation ladder for **DR-RLM** (recursive
deep-research RLM training on SkyRL). Each rung isolates exactly one mechanism so
the experimental spine answers one research question at a time (proposal §6,
"Ablation ladder"; *Recursive Deep Research Proposal.md*). Everything held
constant across arms — the **same** rubric/citation judge, the **same** frozen
corpus + retriever, the **same** base policy and hyperparameters — so any quality
difference is attributable to the credit-assignment mechanism, not the substrate.

The entry point is `examples.train.dr_rlm.main_dr_rlm` with
`environment.env_class=dr_rlm`; the ladder selector knobs live on
`cfg.generator` (`dr_rlm_config.py::DrRlmGeneratorConfig`) and are threaded into
the env (and every child env) via `env_payload()`.

## Ladder: which rung isolates which mechanism

| Rung | Script | Proposal rung (§6) — isolates | Key generator knobs | Advantage estimator |
|------|--------|-------------------------------|---------------------|---------------------|
| **L0** | `run_dr_rlm_L0.sh` | **Flat baseline** (DR Tulu-8B-shaped): no recursion at all | `max_recursion_depth=0` `enable_child_agents=false` `per_node_credit=false` `reward_mode=inherited` | `grpo` |
| **L1** | `run_dr_rlm_L1.sh` | **Recursion alone (SkyRL)**: root delegates, but tree is flattened and the single root rubric scalar is broadcast (advantage inheritance) | `max_recursion_depth=2` `enable_child_agents=true` `train_child_trajectories=true` `per_node_credit=false` `reward_mode=inherited` | `grpo` |
| **L2** | `run_dr_rlm_L2.sh` | **RAO coarse per-node scalar**: own-answer quality + λ·mean-child-success (delegation bonus) + RAO depth inverse-frequency weighting. Per-node, but NOT provenance-aware | `per_node_credit=true` `reward_mode=rao` `rao_lambda=0.5` (env `DR_RLM_DEPTH_WEIGHTING=1`) | `rer_pernode` |
| **L3** | `run_dr_rlm_L3.sh` | **RER provenance per-node credit (THE CORE, RQ2)**: r_a = Σ w_c·s_c·share(a,c) − γ·cost(a), attributed through the citation graph; Σ_a r_a = R (credit conserved) | `per_node_credit=true` `reward_mode=rer` `share_mode=citation_count` `max_recursion_depth=2` | `grpo` |
| **L4** | `run_dr_rlm_L4.sh` | **+ structural rubric channel (RQ3)**: L3 plus a co-evolving penalty on the tree (orphan / redundant / over-fragmented children) | = L3 but `reward_mode=rer_structural` (+ `structural_*` penalties) | `grpo` |

Research-question mapping (proposal §RQs): **RQ2** = L3 vs {L1, L2} (does
provenance-attributed credit beat inherited scalar and the RAO delegation bonus);
**RQ3** = L4 vs L3 (does the structural channel improve decomposition);
**RQ4** = gradient diagnostics across L1 ↔ L3 (variance / credit alignment).

### Notes on the estimator choice
- **L3/L4 use stock `grpo`, not `rer_pernode`.** Once `DrRlmGenerator` un-flattens
  the tree, every node is its own step-wise trajectory carrying its own `r_a`, and all
  of a prompt's nodes share the prompt `uid` as `instance_id` (this is required: SkyRL
  builds GRPO groups *and* mini-batch boundaries from `instance_id`, asserting each
  value is contiguous and `#distinct == train_batch_size`). So stock GRPO baselines
  each node against the **mean credit of all the prompt's nodes** — a per-prompt
  difference-reward baseline. The depth-**cohort** baseline + RAO depth-weighting is the
  `rer_pernode` estimator (+ `DrRlmTrainer`, which threads per-row `node_depth`), the
  RQ4 variant exercised by **L2** (env `DR_RLM_DEPTH_WEIGHTING=1`,
  `DR_RLM_DEPTH_WEIGHT_ALPHA`).
- `per_node_credit=true` (L2-L4) forces `train_child_trajectories=true` in
  `DrRlmGeneratorConfig.__post_init__`, so child trajectories enter the batch.

## Held-constant substrate

- **Judge** (`run_judge.sh`): local Qwen3-8B served by vLLM as an OpenAI-compatible
  endpoint on `:8100`. The L* scripts set `generator.judge_base_url` to match and
  export `JUDGE_API_KEY=EMPTY`. `generator.judge_model=hosted_vllm/Qwen/Qwen3-8B`
  (the `hosted_vllm/` prefix is stripped by `JudgeConfig.from_env_payload`, so the
  wire model id is `Qwen/Qwen3-8B`, exactly what `vllm serve Qwen/Qwen3-8B` registers).
- **Retriever** (`run_retriever.sh`): the offline corpus `search()`/`get_doc()`
  tools (`corpus_search.py`). Two options, pick one and set
  `generator.search_backend` to match:
  - **Option A — in-process** `bm25` / `faiss` / `local_jsonl` (no server;
    `get_backend()` loads it per worker). `bm25` needs Java on PATH (pyserini/Lucene);
    `local_jsonl` needs nothing (good for smoke tests).
  - **Option B — server**: `bash run_retriever.sh mcp` launches the dr_agent FastMCP
    server; set `generator.search_backend=mcp_http`
    `generator.search_endpoint=http://localhost:8003/mcp`.

## Run order

1. **Judge** (own GPU/node, leave running):
   ```bash
   bash configs/run_judge.sh
   ```
2. **Retriever** (only if using Option B / `mcp_http`; in-process backends need no server):
   ```bash
   bash configs/run_retriever.sh mcp          # Option B server, OR
   bash configs/run_retriever.sh build-bm25   # build the pyserini index for Option A bm25
   ```
3. **Data convert** — produce the parquet the trainer reads and the frozen corpus the
   retriever reads. The dataset row schema the env consumes is
   `prompt`, `env_class=dr_rlm`, `reward_spec={"rubrics":[{description,title,weight}]}`,
   `question`, `max_turns`, `extra_info` — `reward_spec.rubrics` is what the judge
   scores in both the L1 in-env path (`score_report_sync`) and the L2-L4 generator
   path (`compute_rer_rewards`). Outputs:
   `$DATA_DIR/train.parquet`, `$DATA_DIR/validation.parquet`, `$DATA_DIR/corpus.jsonl`
   (defaults: `DATA_DIR=$HOME/data/dr-rlm`).
4. **Train one rung** (judge + retriever must already be up):
   ```bash
   bash configs/run_dr_rlm_L0.sh      # flat baseline
   bash configs/run_dr_rlm_L1.sh      # recursion + inherited scalar (SkyRL)
   bash configs/run_dr_rlm_L2.sh      # RAO local-node + depth weighting
   bash configs/run_dr_rlm_L3.sh      # RER provenance per-node credit (the core)
   bash configs/run_dr_rlm_L4.sh      # + structural channel
   ```

### Common overrides (env vars, same names across all L* scripts)
| Env var | Default | Meaning |
|---------|---------|---------|
| `DATA_DIR` | `$HOME/data/dr-rlm` | parquet + corpus location |
| `MODEL_PATH` | `Qwen/Qwen3-8B` | base/SFT policy (held constant L0..L4) |
| `TRAIN_GPUS` / `TP_SIZE` / `NUM_ENGINES` | `4` / `4` / `1` | trainer + inference engine sizing |
| `JUDGE_BASE_URL` | `http://localhost:8100/v1` | must match `run_judge.sh` port |
| `JUDGE_MODEL` | `hosted_vllm/Qwen/Qwen3-8B` | judge model id (prefix stripped on the wire) |
| `SEARCH_BACKEND` | `local_jsonl` | `local_jsonl` \| `bm25` \| `faiss` \| `mcp_http` \| `none` |
| `SEARCH_CORPUS_PATH` / `SEARCH_INDEX_PATH` / `SEARCH_ENDPOINT` | under `$DATA_DIR` / `:8003/mcp` | retriever paths/endpoint |

Any flag can also be appended on the CLI (forwarded via `"$@"`), e.g.:
```bash
bash configs/run_dr_rlm_L3.sh generator.search_backend=mcp_http trainer.epochs=2
```
