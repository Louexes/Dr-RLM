# DR-RLM — Recursive Deep-Research RLM on SkyRL

DR-RLM trains and evaluates a **recursive** deep-research agent end-to-end on
SkyRL. One shared policy is instantiated recursively as a tree of agents: a root
*orchestrator* decomposes a research question, delegates focused sub-questions to
*coordinator*/*worker* sub-agents (each running its own recursive research over a
**frozen offline corpus**), and synthesizes a long-form, `<cite>`-tagged report.
The contribution is the credit-assignment scheme: instead of broadcasting one
scalar report reward across the whole tree (the stock SkyRL/RAO behavior, which is
diluted and high-variance for long-form synthesis), DR-RLM decomposes the
held-constant rubric reward **across the recursion tree via the agent's own
citation/provenance graph** — yielding dense, low-variance, difference-reward-style
per-node credit `r_a`, plus an optional structural-rubric channel that scores the
*decomposition itself*. The package is built as an **ablation ladder** (L0–L4) so
each mechanism is isolated as a single config flip on top of an otherwise identical
harness.

> **Why is it built this way?** See `ARCHITECTURE.md` for the mapping onto the
> thesis proposal's L0–L4 ladder, the step-wise un-flattening credit contract, and
> the per-node advantage derivation. This README is the *practical* entry doc.

---

## File map

All config lives on `cfg.generator.*` (a single source of truth, auto-exposed as
`generator.<field>` CLI overrides). The env-relevant subset is threaded into
`env_extras["dr_rlm"]` via `DrRlmGeneratorConfig.env_payload()`.

| Path | Purpose |
| --- | --- |
| `dr_rlm_config.py` | `DrRlmGeneratorConfig`: every DR-RLM knob (ladder selector `reward_mode`/`per_node_credit`, recursion depth, RER share/cost params, structural penalties, judge endpoint, search backend) + `env_payload()`. |
| `prompts.py` | Depth-banded system prompts (orchestrator / coordinator / worker) + the shared evidence/citation/`answer`-dict contract. This is what lets depth>1 trees form. |
| `judge.py` | Held-constant rubric/citation judge: `RubricJudge` (async), `score_report_sync` (L1 in-env path), `weighted_report_reward`, `extract_claims_and_corresponding_citation_ids`, `JudgeConfig.from_env_payload`. A dependency-light verbatim port of DR Tulu's rubric scoring. |
| `corpus_search.py` | Offline corpus `search()`/`get_doc()` REPL tools with **provenance-encoded snippet ids** (`"{node_rid}-{n}"`). `make_corpus_tools`, `owner_rid_of`, `get_backend`; backends `bm25` / `faiss` / `mcp_http` / `local_jsonl` / `none`. |
| `rer_reward.py` | The RER core: `RerNode`, `compute_rer_rewards(nodes, rubrics, question, payload) -> RerResult{rewards, report_reward, metrics}`. Implements all four credit modes (inherited / rao / rer / rer_structural). |
| `dr_rlm_env.py` | `DrRlmEnv` (gym id `dr_rlm`): supplies the three `BaseRLMEnv` hooks + a `_build_system_prompt` that advertises a **grounded-only action space** — `search`/`get_doc` and (when the node can delegate) `rlm_query`/`rlm_query_batched`; `llm_query` is intentionally not advertised. `reward_spec` must carry `"rubrics": [{description, title, weight}]`. |
| `dr_rlm_generator.py` | `DrRlmGenerator`: threads config + enforces the depth ceiling, and **un-flattens** the tree into per-node step-wise trajectories (each node keeps the prompt `uid` as `instance_id` + a unique `repetition_id`) carrying per-node reward `r_a`. |
| `rer_advantage.py` | Optional `rer_pernode` advantage estimator (depth-cohort baseline + RAO depth inverse-frequency weighting; RQ4). Registered at import. |
| `dr_rlm_trainer.py` | Optional `DrRlmTrainer` that threads per-row `node_depth` to `rer_pernode`; auto-selected when `advantage_estimator=rer_pernode`, else stock `RayPPOTrainer`. |
| `main_dr_rlm.py` | Training entry point. Module path: `examples.train.dr_rlm.main_dr_rlm`. |
| `main_dr_rlm_eval.py` | Eval-only (generate-only) entry point. Module path: `examples.train.dr_rlm.main_dr_rlm_eval`. |
| `__init__.py` | Registers the `dr_rlm` gym env id and imports `rer_advantage` (registers `rer_pernode`). |
| `data/` | Corpus + RL-prompt prep: `convert_drtulu_rl.py` (DR Tulu RL rows → SkyRL parquet), `build_corpus.py` (corpus.jsonl / index), and a tiny `corpus.jsonl` for smoke tests. |
| `sft/` | [optional] Recursive cold-start: scripts to build/run recursive SFT (or warm-start straight from DR Tulu-8B). |
| `eval/` | `eval_recursive.py` (per-axis latency/compute/quality eval of the untrained recursive arm via `rlm/` inference) + helpers. |
| `configs/` | One launch script per rung: `run_judge.sh`, `run_dr_rlm_L1.sh` … `run_dr_rlm_L4.sh`, plus `configs/README.md` (rung table, kept consistent with this file and `ARCHITECTURE.md`). |
| `tests/` | `pytest` unit tests (RER credit conservation, citation parsing, depth-ceiling, local_jsonl backend) — runnable with no GPU/judge. |
| `README.md` | This file. |
| `ARCHITECTURE.md` | The "why": design rationale, proposal mapping, credit contract. |

---

## Quickstart

All commands run **inside SkyRL's `uv` environment** from the SkyRL repo root
(`/gpfs/home5/lgehringer/Dr-RLM/SkyRL`). DR-RLM is invoked as a module so the
package import side effects (env + estimator registration) fire in the Ray driver
and workers.

### 0. Environment

No separate venv. Use SkyRL's `uv`-managed env with the vLLM extra:

```bash
cd /gpfs/home5/lgehringer/Dr-RLM/SkyRL
uv run --isolated --extra vllm -m examples.train.dr_rlm.main_dr_rlm --help
```

(The reference `run_multi_paper_rlm.sh` uses `--extra fsdp` for the FSDP2 trainer
path; pick the extra that matches your trainer strategy. The launch scripts in
`configs/` pin the right combination.)

### 1. Launch the held-constant judge

The reward judge is an OpenAI-compatible vLLM endpoint serving a local Qwen
(zero external API calls). The **same** judge is used on every arm.

```bash
bash examples/train/dr_rlm/configs/run_judge.sh        # serves judge_model on judge_base_url
export JUDGE_API_KEY=EMPTY                              # dummy key for local vLLM
```

Defaults (`dr_rlm_config.py`): `judge_model=hosted_vllm/Qwen/Qwen3-8B`,
`judge_base_url=http://localhost:8100/v1`, `judge_api_key_env=JUDGE_API_KEY`.

### 2. Get a corpus + pick a search backend

Retrieval is **held constant across arms**: both the flat and recursive policies
search the SAME frozen corpus with the SAME retriever. Corpus rows are
`{id, contents, url}` (DR Tulu's wii schema).

```bash
# real corpus: download the gated wii indices (needs `huggingface-cli login`)
#   -> yields data/corpus.jsonl, data/bm25 (Lucene), data/qwen3-8b/corpus.pkl* (FAISS)
python examples/train/dr_rlm/data/build_corpus.py      # build/fetch the index
```

Choose `generator.search_backend`:
- `local_jsonl` — dependency-free TF-IDF-lite over `search_corpus_path` (smoke/CI; no Java, no embed model).
- `bm25` — in-process dr_agent `BM25Searcher` over `search_index_path=data/bm25` (needs Java/pyserini).
- `faiss` — in-process dr_agent `FaissSearcher` (needs `search_embed_model`, e.g. `Qwen/Qwen3-Embedding-8B`).
- `mcp_http` — call a running dr_agent FastMCP `local_search` at `search_endpoint` (default `http://localhost:8003/mcp`).
- `none` — disable retrieval (context-only debugging).

### 3. Convert the RL data

Reuse DR Tulu's RL prompts verbatim — no new data is authored. The converter emits
SkyRL parquet rows whose `reward_spec` carries `rubrics: [{description, title,
weight}]` (the contract `DrRlmEnv` reads) plus `dataset` / `question_type`.

```bash
python examples/train/dr_rlm/data/convert_drtulu_rl.py \
    --out_dir $HOME/data/dr-rlm
# -> $HOME/data/dr-rlm/{train,validation}.parquet
```

### 4. [optional] Recursive SFT cold-start / warm-start

Warm-start the policy so it already searches and emits `<cite>` tags (non-zero
RL start). Either warm-start directly from the public **DR Tulu-8B**
(`rl-research/DR-Tulu-SFT-8B`) as `trainer.policy.model.path`, or run the
recursive SFT under `sft/` first.

### 5. Train a rung

Each rung is the same harness with the ladder flags flipped (see the table below).
Example — L3 (the RER core):

```bash
bash examples/train/dr_rlm/configs/run_dr_rlm_L3.sh
```

That script ultimately runs the module entry point. The load-bearing flags
(matching `main_dr_rlm.py`'s docstring and `run_multi_paper_rlm.sh`):

```bash
uv run --isolated --extra vllm -m examples.train.dr_rlm.main_dr_rlm \
    data.train_data="['$HOME/data/dr-rlm/train.parquet']" \
    data.val_data="['$HOME/data/dr-rlm/validation.parquet']" \
    environment.env_class=dr_rlm \
    generator.step_wise_trajectories=true \
    generator.per_node_credit=true \
    generator.reward_mode=rer \
    generator.max_recursion_depth=2 \
    generator.search_backend=bm25 \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path=rl-research/DR-Tulu-SFT-8B
```

Notes that follow from the source:
- `per_node_credit=true` implies `train_child_trajectories=true` (`__post_init__`)
  and requires `step_wise_trajectories=true`.
- With `per_node_credit=true`, every node is its own trajectory carrying its own
  `r_a`, and all of a prompt's nodes share the prompt `uid` as `instance_id` (forced by
  SkyRL's mini-batch/GRPO contiguity invariant), so the stock `grpo` estimator baselines
  each node against the mean credit of all the prompt's nodes (a per-prompt
  difference-reward baseline). Use `trainer.algorithm.advantage_estimator=rer_pernode`
  for the RQ4 variant — a depth-**cohort** baseline + RAO depth inverse-frequency
  weighting (tune via `DR_RLM_DEPTH_WEIGHT_ALPHA`, `DR_RLM_DEPTH_WEIGHTING`); it is wired
  through `DrRlmTrainer` automatically when selected.

### 6. Evaluate

In-harness rollout eval (scores the report; `per_node_credit=false` — eval does not
need per-node credit):

```bash
uv run --isolated --extra vllm -m examples.train.dr_rlm.main_dr_rlm_eval \
    data.val_data="['$HOME/data/dr-rlm/validation.parquet']" \
    environment.env_class=dr_rlm \
    generator.step_wise_trajectories=true \
    generator.per_node_credit=false \
    trainer.policy.model.path=<ckpt-or-base>
```

Per-axis latency/compute/quality eval of the untrained recursive arm (via `rlm/`
inference):

```bash
python examples/train/dr_rlm/eval/eval_recursive.py
```

### 7. Run the tests

No GPU or judge required (the framework-free RER/citation/backend units):

```bash
uv run --isolated --extra vllm -m pytest examples/train/dr_rlm/tests
```

---

## The L0–L4 ladder

Each rung isolates one mechanism by flipping a small set of `generator.*` flags on
top of an otherwise identical harness (same data, corpus, judge). Kept consistent
with `configs/README.md` and `ARCHITECTURE.md`.

| Rung | Flags | What it isolates |
| --- | --- | --- |
| **L0** | Flat RLER = **DR Tulu-8B** (downloaded; no recursion) | flat baseline (no harness recursion) |
| **L1** | `per_node_credit=false`, `reward_mode=inherited` | recursion alone (stock SkyRL: flatten tree, score root, broadcast scalar) |
| **L2** | `per_node_credit=true`, `reward_mode=rao` (`rao_lambda`) | coarse scalar per-node (RAO local reward = own quality + λ·mean-child) |
| **L3** | `per_node_credit=true`, `reward_mode=rer` (`share_mode`, `gamma_cost`) | **provenance-attributed per-node credit `r_a` — the RER core (RQ2)** |
| **L4** | `per_node_credit=true`, `reward_mode=rer_structural` (+ `structural_*_penalty`) | the decomposition reward / structural-rubric channel (RQ3) |

RQ2 = L3 vs {L1, L2}; RQ3 = L4 vs L3; RQ4 = gradient diagnostics (`grpo` vs
`rer_pernode` depth weighting) across L1↔L3. All recursive rungs default to
`max_recursion_depth=2` (grandchildren permitted, the proposal's target).

---

## Controlled experiment

This is a controlled A/B against the existing recursive-RL credit schemes. The
**reward substrate is held constant** so any difference is attributable to the
credit-assignment mechanism, not to a different reward/corpus/judge:

- **Held constant across all rungs:** the rubric/citation **judge**
  (`judge.py`, same `judge_model`/endpoint), the frozen **corpus + retriever**
  (`corpus_search.py`, same `search_backend`/index), the **data** (DR Tulu RL
  prompts, reused verbatim), the report reward formula `R = Σ s_c·w_c / Σ_{w>0} w`
  (`weighted_report_reward`, byte-identical on the sync L1 and async L2–L4 paths),
  and the depth-banded prompts/protocol.
- **The two baselines being beaten:**
  1. **DR Tulu / RLER (flat single-agent)** — owns the reward axis but does not
     recurse (L0).
  2. **SkyRL- / RAO-style scalar tree credit** — recursion with a single scalar
     broadcast across the tree (L1) and the RAO coarse per-node reward (L2).

  DR-RLM's L3/L4 add provenance-attributed credit and a structural channel on top
  of that same substrate. Proprietary deep-research systems are reference-only
  (frozen corpus ⇒ not leaderboard-comparable, by design).

---

## Status & limitations

- **Smoke-runnable today** with `generator.search_backend=local_jsonl` (a small
  bundled `data/corpus.jsonl`) + a locally served judge. The RER/citation/backend
  logic is framework-free and unit-tested (`tests/`, no GPU/judge needed).
- **Real runs need** the gated **wii corpus + indices** (`s42chen/wii-indexes`,
  requires `huggingface-cli login`; yields `data/corpus.jsonl`, `data/bm25`,
  `data/qwen3-8b/corpus.pkl*`) for `bm25`/`faiss`, and a **served judge** endpoint.
  Corpus availability (gated HF) is the standing blocker.
- The judge is a dependency-light **port** of DR Tulu's rubric scoring; a
  format-mismatch on the local judge silently zeros scores (a failed/parse-error
  judge call returns `0.0`, exactly like the DR Tulu original — monitor judge
  health). Citation-as-reward is off by default (`citation_reward_weight=0.0`): the
  citation graph is used for *credit*, not the reward, to avoid double-counting.
- Per the proposal's design risk, **flat DR Tulu-8B already saturates** the standard
  benchmarks — recursion's *quality* gains are expected only in the high-breadth /
  source-overflow regime (latency/compute gains show everywhere). See
  `ARCHITECTURE.md` for the full limitations list and the high-breadth eval stratum.
