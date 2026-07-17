# DR-RLM SFT: warm-start (default) vs recursive cold-start (this pipeline)

This directory holds the **optional recursive cold-start SFT** for DR-RLM. Read this
before running it — for the controlled comparison against DR Tulu the default is **not**
to run SFT here at all.

## TL;DR — which path do I take?

| Path | What | When |
| --- | --- | --- |
| **Warm-start (DEFAULT)** | Initialize **both** arms (flat L1 baseline and recursive L2–L4) from the released **DR Tulu-8B** checkpoint and go straight to RL. No SFT in this directory. | Always try first. Keeps the A/B controlled (same init, same judge, same corpus, same retriever) so any delta is attributable to the per-node credit scheme, not to a different cold-start. |
| **Recursive cold-start (this pipeline)** | Rejection-sample recursive RLM trajectories, span-mask the injected text, SFT on them. | Only if RL-from-warm-start fails to induce genuine depth>1 trees (e.g. the policy collapses to flat single-agent behavior and never delegates). Mirrors NovaSky/Sky-T1's mandatory SFT cold-start before RL. |

The reason the warm-start is the default: DR Tulu-8B is already a competent
search-augmented long-form research model with the `<cite id=...>` + `<tool_output>`
tag protocol baked in. Starting both arms from it isolates the contribution of the
recursive credit assignment (RER) from any confound introduced by a bespoke cold-start.

## Recursive cold-start pipeline (only if needed)

1. **Generate trajectories** — `gen_recursive_sft.py` samples `--n_samples` recursive
   rollouts per training prompt from the `rlm/` **inference arm** (`RLM(backend='vllm',
   environment='local', max_depth>=2)` against a locally served policy), scores each
   **root report** with the **held-constant RER rubric judge**
   (`judge.score_report_sync`, byte-identical to the L1 RL reward), and keeps the best
   rollout iff its report score `R >= --threshold` (rejection sampling).

   ```bash
   python -m examples.train.dr_rlm.sft.gen_recursive_sft \
       --prompts ~/data/dr_rlm_rl/train.parquet \
       --out_jsonl ~/data/dr_rlm_sft/dr_rlm_recursive_sft.jsonl \
       --n_samples 4 --max_depth 2 --threshold 0.5 \
       --model rl-research/dr-tulu-8b --base_url http://localhost:8000/v1
   ```

   The corpus `search()`/`get_doc()` tools and the rubric judge are the **same** held
   constant pieces the RL arm uses (`corpus_search.make_corpus_tools`,
   `JudgeConfig.from_env_payload`), so SFT-time retrieval/judging matches RL-time.
   Defaults for both come from `DrRlmGeneratorConfig`, the single config source of truth.

2. **Train** — register the local dataset and launch LLaMA-Factory:

   ```bash
   DATA_DIR=~/data/dr_rlm_sft bash sft/run_sft.sh sft/dr_rlm_sft.yaml
   ```

   `run_sft.sh` merges `dataset_info_entry.json` into
   `${DATA_DIR}/dataset_info.json` (key `dr_rlm_recursive_sft`, `formatting: sharegpt`,
   `columns.messages = conversations`) and runs `llamafactory-cli train dr_rlm_sft.yaml
   dataset_dir=${DATA_DIR}`.

## Data format

Each output row is one sharegpt example:

```json
{"conversations": [
  {"role": "system",    "content": "<depth-banded ORCHESTRATOR system prompt>"},
  {"role": "user",      "content": "<the research question>"},
  {"role": "assistant", "content": "<repl think/code>\n\n<tool_output>...search hits...</tool_output>\n\n<subagent_output>...child mini-report with <cite id=...>...</subagent_output>"},
  {"role": "assistant", "content": "...answer[\"content\"] = report (with <cite id=\"...\"> citations); answer[\"ready\"] = True..."}
]}
```

- The **system** turn is the depth-banded orchestrator prompt from `prompts.py`
  (`depth_system_prompt(0, max_recursion_depth)`), so the SFT model learns the exact
  decompose→delegate→synthesize contract the RL env serves.
- Each REPL iteration becomes one **assistant** turn that inlines the model's
  think/code response, then the turn's retrieved snippets wrapped in
  `<tool_output>...</tool_output>` and any delegated sub-agent answers wrapped in
  `<subagent_output>...</subagent_output>`.
- `<cite id=...>` tags are preserved verbatim (provenance flows up from sub-agents),
  matching the citation contract the RER credit assignment reads.

## Span-masking rationale

`dr_rlm_sft.yaml` sets `use_span_masking: true` and
`mask_span_types: "tool_output,subagent_output"`. Both spans are content the policy
**did not generate** but which appears inline in the assistant turns (the dr-tulu
convention: tool outputs are not separate tool-role messages — they live inside
assistant content and are masked). Masking them sets `labels = IGNORE_INDEX` over those
token spans so:

- **`tool_output`** — the model is not trained to memorize/hallucinate retrieved corpus
  snippets; it is trained on *when and how to search* and *how to cite* what came back.
- **`subagent_output`** — the model is not trained to reproduce a sub-agent's mini-report
  verbatim; it is trained to *decide what to delegate* and to *synthesize over* the
  returned answers (keeping their `<cite id=...>` tags), which is exactly the
  orchestrator behavior the recursion + RER credit reward rewards.

The dr-tulu baseline masks only `tool_output`; DR-RLM extends the mask to
`subagent_output` because depth>1 recursion injects a second class of non-policy text
(sub-agent answers) that must likewise be excluded from the loss.
