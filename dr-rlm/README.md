# dr-rlm — Deep-Research RLM project

Our working home for building, running, and training the **DR-RLM** deep-research agent —
a single self-contained project (RL / SFT / inference together), laid out dr-tulu-style.
Reference deps (`rlm/`, `dr-tulu/`) stay at the repo root; the RL backbone (**SkyRL**) and SFT
framework (**llama-factory**) are vendored INSIDE this folder (`rl/skyrl/`, `sft/llama-factory/`).
Our edits to them are logged in [`docs/patches_to_deps.md`](docs/patches_to_deps.md).

## Layout
| Dir | What |
|-----|------|
| `rl/` | **Reinforcement learning.** SkyRL backbone vendored at `rl/skyrl/`; DR-RLM RL code at `rl/skyrl/examples/train/dr_rlm/`. |
| `sft/` | **Supervised fine-tuning (cold start).** llama-factory vendored at `sft/llama-factory/`; recursive-trajectory distillation. |
| `agent/` | Inference + eval driver: `generate.py`, `run.sh`, A3 helpers. |
| `prompts/` | Editable inference prompt (`system_prompt.txt`). _Being unified with the RL prompt (`rl/skyrl/…/dr_rlm/prompts.py`) — see `docs/decisions.md`._ |
| `data/subsets/` | Frozen benchmark subsets (+ provenance in `data/README.md`). |
| `experiments/` | One self-contained dir per experiment. Start at [`experiments/REGISTRY.md`](experiments/REGISTRY.md). |
| `analysis/` | Reusable analysis scripts (recursion rate, breadth, score matching, aggregation). |
| `results/` | Curated canonical tables (the scoreboard). |
| `runs/` | Heavy eval outputs + SLURM logs (gitignored). |
| `docs/` | Project documentation — start at [`docs/README.md`](docs/README.md). |
| `keys.sh` | API keys (gitignored). `source dr-rlm/keys.sh` before running anything. |

## Quickstart (an A2-rec inference run)
```bash
source dr-rlm/keys.sh
sbatch dr-rlm/experiments/<exp>/jobs/<job>.job   # jobs set OUT=dr-rlm/runs/<ARM> and call agent/run.sh
```
`run.sh` launches the search backend (default: the frozen BM25 corpus, `SEARCH_BACKEND=bm25s`;
live Serper+Jina MCP also supported), then `generate.py` (the RLM REPL agent), then graders.

## Conventions
- **New experiment** = copy `experiments/_TEMPLATE/`, add a row to `experiments/REGISTRY.md`, write `NOTES.md` with the question + result.
- **Prompt change** = edit `prompts/system_prompt.txt`, bump its CHANGELOG.
- Paths are anchored on `REPO=/gpfs/home5/lgehringer/Dr-RLM` and `DRRLM=$REPO/dr-rlm`.
