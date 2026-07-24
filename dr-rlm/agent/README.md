# agent/ — inference & evaluation

Two drivers, selected by `--driver`:

- **`infer_driver.py`** (`--driver new`) — the unified path: runs the same SkyRL `DrRlmEnv` that
  RL trains, for both the recursive and flat substrates. System prompt from
  `../prompts/system_prompt.txt`; per-benchmark task format in `_TASK_FRAMING`.
- **`generate.py`** (`--driver legacy`) — the original `rlm/`-engine REPL driver, kept for the
  early baseline runs. Loads `../prompts/legacy_system_prompt.txt`.

`run.sh` wires it together: starts the search backend (default `SEARCH_BACKEND=bm25s` over the
frozen corpus; live Serper+Jina also supported), shards generation, runs graders.
`build_shortform_evalsets.py` builds the short-form benchmark subsets; `api_client.py` is the
shared model-endpoint client.
