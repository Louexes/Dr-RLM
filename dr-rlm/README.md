# dr-rlm — Deep-Research RLM project

The **DR-RLM** deep-research agent — a single self-contained project (RL / SFT / inference
together), laid out dr-tulu-style. Reference deps (`rlm/`, `dr-tulu/`) stay at the repo root;
the RL backbone (**SkyRL**) and SFT framework (**llama-factory**) are vendored INSIDE this
folder (`rl/skyrl/`, `sft/llama-factory/`).

## Layout
| Dir | What |
|-----|------|
| `rl/` | **Reinforcement learning.** SkyRL backbone vendored at `rl/skyrl/`; DR-RLM RL code at `rl/skyrl/examples/train/dr_rlm/`. |
| `sft/` | **Supervised fine-tuning (cold start).** llama-factory vendored at `sft/llama-factory/`; recursive-trajectory distillation. |
| `agent/` | Inference + eval driver (`generate.py`, `infer_driver.py`, `run.sh`) and the SLURM jobs for every eval/SFT/RL stage. |
| `prompts/` | The canonical inference system prompt (`system_prompt.txt`), depth-banded, shared with the RL env (`rl/skyrl/…/dr_rlm/prompts.py`). |
| `data/subsets/` | Frozen benchmark subsets (+ provenance in `data/README.md`). |
| `corpus_build/` | The frozen-corpus pipeline (discover → fetch → normalize → index → audit → freeze). |
| `experiments/` | One self-contained dir of launch jobs per experiment (`rl_training/jobs/` holds the RL runs). |
| `analysis/` | Analysis scripts (behavioral metrics, training curves, ablation figures) + generated figures. |
| `results/` | Curated result tables. |
| `runs/` | Heavy eval outputs + SLURM logs (gitignored). |
| `keys.sh` | API keys (gitignored). `source dr-rlm/keys.sh` before running anything. |

## Quickstart (a recursive inference run)
```bash
source dr-rlm/keys.sh
sbatch dr-rlm/experiments/<exp>/jobs/<job>.job   # jobs set OUT=dr-rlm/runs/<ARM> and call agent/run.sh
```
`run.sh` launches the search backend (default: the frozen BM25 corpus, `SEARCH_BACKEND=bm25s`;
live Serper+Jina MCP also supported), then the driver, then graders.

Paths are anchored on `REPO=/gpfs/home5/lgehringer/Dr-RLM` and `DRRLM=$REPO/dr-rlm`; adapt for
your cluster.
