# DR-RLM: Reinforcement Learning with Provenance Credit for Recursive Language Models

Code repository accompanying the MSc Artificial Intelligence thesis
*"Reinforcement Learning with Provenance Credit for Recursive Language Models"*
(University of Amsterdam, 2026).

DR-RLM post-trains a small language model (Qwen3.5-4B) as a *recursive* deep-research agent:
the root agent works in a persistent Python REPL, delegates sub-questions to child agents, and
composes their structured `{content, citations}` returns into a cited research report. The core
methodological contribution is **provenance credit** (Recursive Provenance Credit, RPC): instead
of broadcasting one scalar reward to every agent in the tree, credit is routed along the
citation graph, so each sub-agent is rewarded in proportion to how much of its evidence actually
supports the final report. Training and evaluation run against a frozen, self-crawled retrieval
corpus, which makes the flat-vs-recursive comparison controlled and the pipeline reproducible on
academic budgets.

## What this repository contains

- **The recursive agent as an RL environment**: REPL-based recursion substrate (root +
  sub-agents, tree-global provenance ledger, strict finalization) implemented on SkyRL
  (`dr-rlm/rl/skyrl/examples/train/dr_rlm/`, `dr-rlm/rl/skyrl/skyrl-gym/skyrl_gym/envs/rlm/`).
- **Provenance credit**: per-node credit from ledger support, validated against a
  leave-one-child-out counterfactual.
- **A matched flat baseline**: a faithful DR-Tulu-style ReAct arm with its own SFT and RL
  environment (`dr_tulu_env.py`).
- **The frozen corpus pipeline**: 139k-document web corpus + BM25 index, self-crawled with a
  free, ungated stack (`dr-rlm/corpus_build/`).
- **The evaluation harness**: one inference driver for both substrates over seven benchmarks,
  plus analysis scripts and generated figures (`dr-rlm/agent/`, `dr-rlm/analysis/`).

## Repository layout

| Path | What |
|---|---|
| `dr-rlm/agent/` | Inference + eval driver for both substrates; SLURM jobs for every experiment |
| `dr-rlm/rl/skyrl/examples/train/dr_rlm/` | Training package: RER reward, provenance credit, judge, corpus tools, entry points `main_dr_rlm.py` / `main_dr_rlm_eval.py` / `main_dr_tulu.py` |
| `dr-rlm/rl/skyrl/skyrl-gym/skyrl_gym/envs/rlm/` | Recursive REPL environment |
| `dr-rlm/corpus_build/` | Frozen-corpus pipeline (discover, fetch, normalize, index, audit, freeze) |
| `dr-rlm/sft/`, `dr-rlm/scripts/` | SFT data generation and filtering; checkpoint grafting utilities |
| `dr-rlm/analysis/` | Behavioral metrics, training-curve and ablation figures |
| `dr-rlm/results/` | Curated result tables and generated paper assets |
| `dr-rlm/prompts/`, `dr-rlm/experiments/` | Canonical system prompt; one directory per experiment |
| `dr-rlm/rl/skyrl/` | SkyRL, the RL backbone (vendored) |
| `dr-tulu/` | DR Tulu, the deep-research baseline we extend (vendored) |
| `rlm/` | RLM, the original recursive-language-model implementation (vendored) |
| `dr-rlm/sft/llama-factory/` | LLaMA-Factory, SFT reference; final SFT uses SkyRL's native trainer (vendored) |

## Setup

Developed on a SLURM cluster with H100 GPUs. The `.job` files under `dr-rlm/agent/` are the
exact as-run configurations; paths inside them are cluster-specific and need adapting.

```bash
# RL / SFT training environment (SkyRL, uv-managed)
cd dr-rlm/rl/skyrl && uv sync
```

Models are served with vLLM. The policy is served thinking-on; the judge endpoint must be served
thinking-off (`--default-chat-template-kwargs '{"enable_thinking": false}'`). API keys (only
needed for live-web crawling and the optional Gemini grader) are sourced from a local `keys.sh`,
which is not committed.

## Reproducing the pipeline

1. **Frozen corpus (once).** `dr-rlm/corpus_build/` crawls, normalizes, indexes, and freezes the
   corpus (`seeds.py` → `discover.py` → `fetch.py` → `normalize.py` → `index_bm25.py` →
   `audit.py` → `freeze.py`). All later stages point at it via `SEARCH_BACKEND=bm25s`. The built
   corpus is not in the repo; the pipeline regenerates it, or the frozen snapshot is available
   from the author.
2. **Untrained baselines.** `dr-rlm/agent/generate.py` runs both substrates through the same
   driver: recursive REPL (depth 1, structured child returns, strict finalization) and flat
   ReAct. Eval jobs: `dr-rlm/agent/eval_*.job`; panels: `panel_*.sh`.
3. **SFT cold start.** Teacher trajectories are distilled under each substrate's own harness,
   filtered and count-matched across arms (`dr-rlm/scripts/`, `dr-rlm/sft/`), then trained with
   SkyRL's native SFT trainer (`sft_*.job`). Text-only checkpoints are grafted back into the
   multimodal base for serving (`scripts/graft_text_into_multimodal.py`).
4. **RL.** Entry point `dr-rlm/rl/skyrl/examples/train/dr_rlm/main_dr_rlm.py` (GRPO-family,
   rubric reward, provenance credit `share_mode=ledger_support`, train harness mirroring the
   eval harness). The credit-rule ablation arms differ only in credit routing; the flat arm
   trains in `dr_tulu_env.py` on the identical prompt set.
5. **Evaluation and grading.** Generation runs on GPU; grading runs off GPU. Figures and
   behavioral metrics are regenerated by the `dr-rlm/analysis/` scripts.

Model checkpoints and raw run outputs are too large for GitHub and are available from the author
on request.

## Acknowledgments

This project builds directly on three open-source efforts: [DR Tulu](https://github.com/rlresearch/dr-tulu),
[RLM](https://github.com/alexzhang13/rlm), and [SkyRL](https://github.com/NovaSky-AI/SkyRL).
Vendored copies retain their upstream licenses.

```bibtex
@misc{recursivelanguage,
  title         = {Recursive Language Models},
  author        = {Alex L. Zhang and Tim Kraska and Omar Khattab},
  year          = {2026},
  eprint        = {2512.24601},
  archivePrefix = {arXiv},
}

@article{shao2025dr,
  title   = {DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research},
  author  = {Shao, Rulin and Asai, Akari and Shen, Shannon Zejiang and Ivison, Hamish and others},
  journal = {arXiv preprint arXiv:2511.19399},
  year    = {2025}
}

@misc{cao2025skyrl,
  title  = {SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning},
  author = {Shiyi Cao and Sumanth Hegde and Dacheng Li and Tyler Griggs and others},
  year   = {2025}
}
```
