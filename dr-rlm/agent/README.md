# agent/ — DR-RLM inference harness

Two drivers, selected by `--driver`:
- **`new`** (the unified path — same code RL trains): `infer_driver.py` runs the SkyRL `DrRlmEnv`.
  The system prompt is `../prompts/system_prompt.txt` (depth-banded, parsed by
  `rl/skyrl/examples/train/dr_rlm/prompts.py`); the per-benchmark TASK FORMAT is appended to the
  user message from `infer_driver.py:_TASK_FRAMING`; the `context`-metadata and per-turn messages
  come from `DrRlmEnv._get_context_metadata_text` / `_get_user_prompt`. This is what to iterate on.
- **`legacy`** (the original `rlm/`-engine path, below): loads `../prompts/legacy_system_prompt.txt`.

## Legacy driver

- **`generate.py`** — the driver. Loads `../prompts/legacy_system_prompt.txt` as the RLM `custom_system_prompt`
  (`orchestrator=False`, addendum baked in), builds the per-benchmark user prologue (`_user_prologue`:
  search/browse intro + decompose bullet + cite rule + TASK FORMAT), injects `search()`/`browse()` tools,
  runs the recursive REPL, writes one JSONL row per item. Key flags: `--benchmark --model --base-url
  --api-key-env --max-depth --max-iterations --max-tool-calls --tool-budget-scope --num-shards`.
- **`run.sh`** — runner. Reads env (MODEL, MAX_DEPTH, ORCHESTRATOR, TOOL_BUDGET_SCOPE, MAX_TOOL_CALLS,
  BENCH, NUM_SHARDS, GRADE, OUT, *_LOCAL_PATH); launches the MCP backend (Serper+Jina); shards generation;
  runs graders. Anchors: `REPO=/gpfs/home5/lgehringer/Dr-RLM`, `DRRLM=$REPO/drrlm`.
- **`a3_gpt5mini.sh`, `a3_pilot.sh`** — A3 (flat DR-Tulu ReAct) helpers.

The full prompt the model sees = `prompts/system_prompt.txt` (system) + the `_user_prologue` (user, in
`generate.py`) + runtime metadata/turn messages. See `../prompts/README.md`.

Note: `--orchestrator/--no-orchestrator` and the `ORCHESTRATOR` env are vestigial (addendum always baked).
