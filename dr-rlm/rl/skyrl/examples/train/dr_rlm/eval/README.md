# DR-RLM evaluation

Two complementary eval paths:

## 1. `eval_recursive.py` — the untrained / inference arm + flat-vs-recursive frontier (C2)

Runs the `rlm/` inference library directly (`RLM(...).completion(question)`) over an eval set,
with the **same corpus retriever and the same held-constant rubric judge** as training, and
compares a **flat** arm (`max_depth=1`, no recursion) against a **recursive** arm (`max_depth>=2`)
on the proposal's three axes:

- **quality** — report rubric score `R` from `judge.score_report_sync` (identical judge on every arm);
- **compute** — total calls + input/output tokens across the whole tree (`RLMChatCompletion.usage_summary`);
- **latency** — wall-clock per completion (reported with the concurrency setting; do not read latency
  as a recursion win under serial sub-calls).

This is the characterization study (RQ1/C2): it needs no trained checkpoint, so it can run before any
RL, and it produces the flat-vs-recursive frontier that motivates the method.

```bash
python examples/train/dr_rlm/eval/eval_recursive.py \
  --eval_data eval.jsonl \           # rows: {"question": ..., "rubrics": [{description,title,weight}]}
  --model Qwen/Qwen3-8B --base_url http://localhost:8000/v1 \
  --judge_model Qwen/Qwen3-8B --judge_base_url http://localhost:8100/v1 \
  --max_depth 2 --out frontier.jsonl
```

## 2. `main_dr_rlm_eval.py` — trained-policy, in-harness rollout eval

Generate-only run of a trained DR-RLM policy *inside* the SkyRL harness (same env/generator hooks as
training), for measuring the trained recursive agent on the eval split. Set
`generator.per_node_credit=false` for eval (it scores the report; per-node credit is a training-time
concern). See `../main_dr_rlm_eval.py` and `../configs/` for invocation.

## Note on the two baselines (controlled vs external)

The **controlled** baseline for the causal claim is the *flat-mode run in this same harness* (ladder
L0/L1) on identical prompts/reward/corpus — not the externally released DR Tulu-8B. The released
DR Tulu-8B is an *external SOTA reference* only (it trained on a larger, partly-private mix). Keep the
comparison honest by holding the judge/corpus/rubrics constant within the harness.
