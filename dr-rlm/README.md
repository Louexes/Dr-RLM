# dr-rlm

The DR-RLM project: RL, SFT, and inference in one place, laid out dr-tulu-style.

| Dir | What |
|-----|------|
| `rl/skyrl/` | Vendored SkyRL backbone; DR-RLM training code at `rl/skyrl/examples/train/dr_rlm/` |
| `sft/` | Cold-start SFT (llama-factory vendored; final runs use SkyRL's native trainer) |
| `agent/` | Inference/eval drivers (`infer_driver.py`, `generate.py`, `run.sh`) |
| `prompts/` | Canonical system prompts, shared between inference and RL |
| `corpus_build/` | Frozen-corpus pipeline |
| `scripts/` | SFT data pipeline + checkpoint grafting |
| `analysis/` | Behavioral metrics + headline figures |
| `data/`, `results/` | Subset manifests; credit-validation results |

Paths are anchored on `REPO=/gpfs/home5/lgehringer/Dr-RLM`; adapt for your cluster.
`keys.sh` (gitignored) holds API keys for live crawling / the optional Gemini grader.
