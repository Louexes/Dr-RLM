# DR-RLM: Reinforcement Learning with Provenance Credit for Recursive Language Models

Code accompanying the MSc Artificial Intelligence thesis
*"Reinforcement Learning with Provenance Credit for Recursive Language Models"*
(University of Amsterdam, 2026).

DR-RLM post-trains a small language model (Qwen3.5-4B) as a *recursive* deep-research agent:
the root agent works in a persistent Python REPL, delegates sub-questions to child agents, and
composes their structured `{content, citations}` returns into a cited research report. The core
contribution is **provenance credit**: instead of broadcasting one scalar reward to every agent
in the tree, credit is routed along the citation graph, so each sub-agent is rewarded in
proportion to how much of its evidence survives into the final report. Training and evaluation
run against a frozen, self-crawled retrieval corpus, making the flat-vs-recursive comparison
controlled and reproducible on academic budgets.

## Layout

| Path | What |
|---|---|
| `dr-rlm/rl/skyrl/examples/train/dr_rlm/` | Training package: recursive env, RER reward, provenance credit, judge, corpus tools. Entry points `main_dr_rlm.py` (RL), `main_dr_tulu.py` (flat baseline arm) |
| `dr-rlm/rl/skyrl/skyrl-gym/skyrl_gym/envs/rlm/` | Base recursive REPL environment |
| `dr-rlm/agent/` | Inference/eval drivers: `infer_driver.py` (unified, trains+evals), `generate.py` (legacy engine), `run.sh` |
| `dr-rlm/prompts/` | The system prompts (depth-banded, shared between inference and RL) |
| `dr-rlm/corpus_build/` | Frozen-corpus pipeline: `seeds → discover → fetch → normalize → index_bm25 → audit → freeze` |
| `dr-rlm/scripts/` | SFT data pipeline and checkpoint grafting |
| `dr-rlm/analysis/` | Behavioral metrics + headline training figures |
| `dr-rlm/data/`, `dr-rlm/results/` | Benchmark-subset manifests; credit-validation results |
| `dr-rlm/rl/skyrl/`, `dr-tulu/`, `rlm/`, `dr-rlm/sft/llama-factory/` | Vendored: SkyRL, DR Tulu, RLM, LLaMA-Factory |

## Setup

Developed on a SLURM cluster with H100 GPUs; models served with vLLM (policy thinking-on, judge
thinking-off via `--default-chat-template-kwargs '{"enable_thinking": false}'`).

```bash
cd dr-rlm/rl/skyrl && uv sync   # RL / SFT training environment
```

## Reproducing

1. **Corpus**: run the `dr-rlm/corpus_build/` pipeline; everything downstream reads it via
   `SEARCH_BACKEND=bm25s`.
2. **Baselines**: `dr-rlm/agent/run.sh` drives both substrates (recursive REPL and flat ReAct)
   over the benchmark subsets built by `dr-rlm/analysis/build_subsets.py`.
3. **SFT**: distill teacher trajectories per substrate (`dr-rlm/scripts/`), train with SkyRL's
   SFT trainer, graft for serving (`scripts/graft_text_into_multimodal.py`).
4. **RL**: `main_dr_rlm.py` with `share_mode=ledger_support` (credit-rule ablations switch only
   this); flat arm via `main_dr_tulu.py` on the identical prompts.
5. **Eval**: same driver as step 2 on the trained checkpoints; grading runs off-GPU.

Checkpoints, corpus snapshot, and raw run outputs are available from the author on request.

## Acknowledgments

Builds on [DR Tulu](https://github.com/rlresearch/dr-tulu), [RLM](https://github.com/alexzhang13/rlm),
and [SkyRL](https://github.com/NovaSky-AI/SkyRL); vendored copies retain their upstream licenses.

```bibtex
@misc{recursivelanguage,
  title = {Recursive Language Models},
  author = {Alex L. Zhang and Tim Kraska and Omar Khattab},
  year = {2026}, eprint = {2512.24601}, archivePrefix = {arXiv},
}
@article{shao2025dr,
  title = {DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research},
  author = {Shao, Rulin and Asai, Akari and Shen, Shannon Zejiang and Ivison, Hamish and others},
  journal = {arXiv preprint arXiv:2511.19399}, year = {2025}
}
@misc{cao2025skyrl,
  title = {SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning},
  author = {Shiyi Cao and Sumanth Hegde and Dacheng Li and Tyler Griggs and others},
  year = {2025}
}
```
