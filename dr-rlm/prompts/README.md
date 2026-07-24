# prompts/

- **`system_prompt.txt`** — the canonical prompt for the unified `DrRlmEnv` driver (and RL).
  Three depth bands (orchestrator / coordinator / worker) plus shared fragments, assembled by
  `rl/skyrl/examples/train/dr_rlm/prompts.py`. Per-benchmark task format lives in
  `agent/infer_driver.py:_TASK_FRAMING`, not here.
- **`system_prompt_thinkon.txt`** / **`system_prompt_thinkon_depth0.txt`** — thinking-on variant
  and its delegation-free (depth-0) counterpart.
- **`legacy_system_prompt.txt`** — frozen prompt for the legacy `rlm/`-engine driver
  (`generate.py`), kept for reproducing the early baseline runs.
