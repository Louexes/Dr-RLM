# rl/ — DR-RLM reinforcement learning

**SkyRL is the RL backbone**, vendored at **`skyrl/`**. Our DR-RLM code lives inside it:

- `skyrl/skyrl-gym/skyrl_gym/envs/rlm/` — the base recursive REPL env (`BaseRLMEnv` + `PersistentREPL`);
  final-answer contract = the upstream `answer["ready"]` dict.
- `skyrl/examples/train/dr_rlm/` — the DR-RLM training package: RER reward, per-node credit, ladder
  L1–L4, judge, corpus-search REPL tools, generator un-flatten, advantage estimator. Entry points
  `main_dr_rlm.py` (train) / `main_dr_rlm_eval.py` (eval).
- `skyrl/examples/train/rlm/` — the base RLM generator (recursion engine, tree flatten) + multi-paper env.

## Pipeline
1. **SFT warm-up** (→ `../sft/`): distill recursive trajectories into the Qwen3.5-4B student.
2. **RL:** GRPO + RER on SkyRL over the frozen corpus, with provenance credit
   (`share_mode=ledger_support`) routing per-node reward along the citation graph.
3. **Eval:** run the trained model through `../agent/` on the benchmark subsets against the
   matched flat DR-Tulu-style arm (`dr_tulu_env.py`).

`configs/` — RL launch configs (or pointers into `skyrl/examples/train/dr_rlm`).
