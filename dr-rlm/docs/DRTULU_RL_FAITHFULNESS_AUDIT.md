# DR-Tulu RL faithfulness audit — original Open-Instruct recipe vs our SkyRL flat arm

**Date:** 2026-07-07. **Purpose:** pin down, file:line-verified, how the original authors train
DR-Tulu (`dr-tulu/rl/open-instruct`, launched by `train_dr_tulu.sh` → `open_instruct/grpo_fast.py`),
and classify every difference in our SkyRL reimplementation (flat baseline arm) as
**match / forced deviation / decision**. This is the build spec for `DrTuluEnv`.

## 1. The original recipe (verified against code, not the paper)

### Rollout (`tool_utils/tool_vllm.py::ToolUseLLM`)
- Protocol: `<call_tool name="X" ...>query</call_tool>`; parser `v20250824`
  (`UnifiedToolCallParserV20250824`); stop strings `["</call_tool>", "</call>"]`,
  stop string kept in the output.
- Tool result injected as `"\n<tool_output>" + <snippet id=..>/<webpage id=..> items + "</tool_output>\n\n"`,
  **appended to the same continuous token stream** (no chat-template user turn) with
  loss-mask 0 (`mask_tool_use=True` default). Only the newest `<call_tool>` block is parsed.
- Termination: EOS or token budget. **`<answer>` is not a stop string.**
- `max_tool_calls=10` = **global per-rollout budget across all tools** (off-by-one: 11 real
  calls possible); on exhaustion injects `<output>\nMax tool calls exceeded.\n</output>` (masked)
  and lets the model continue to EOS.
- Budget: `response_length=16384` = total response tokens **including injected tool output**;
  tool output truncated to fit. `max_prompt_token_length=2048` and `max_token_length=10240`
  are dataset pre-filters, not rollout knobs.
- Sampling: train T=1.0, top_p=1.0; eval T=0.6, n=1.
- Truncated (`finish_reason=="length"`) rollouts are **kept and rewarded normally**
  (`non_stop_penalty=False`, `mask_truncated_completions=False`).
- System prompt: `unified_tool_calling_v20250907.yaml` + per-question-type
  `additional_instructions` (exact_answer / short_form / long_form). Tools:
  `snippet_search, google_search, browse_webpage` (live web via MCP).

### Reward (`search_rewards/longform_rubric_rewards.py::compute_weighted_rubric_reward_with_citation_and_format_reward`)
All DR-Tulu long-form rows route to verifier `general_rubric`. Per trajectory:

```
score = 10 × (0.5·R_rubric + 0.2·R_cite + 0.2·R_format + 0.1·R_search)
```

- **R_rubric**: per-rubric judge (env `RUBRIC_JUDGE_MODEL`=gpt-4.1-mini, temp 0) scores 0/1/2,
  normalized /2 → [0,1]; weight-weighted mean `Σ(s·w)/max(Σw⁺,1)`. Rubrics = **persistent**
  (the dataset GT rubrics, always on, never pruned) + **adaptive** (RLER; see below).
- **R_cite** = `0.6·avg_F1 + 0.4·(fraction of cited ids that exist in retrieved snippets)`.
  F1 per `<cite id=..>` claim = harmonic(recall, precision), each LLM-judged
  (env `CITATION_JUDGE_MODEL`=gpt-4o-mini): recall Fully/Partially/No → 1/0.5/0,
  precision Relevant/Not → 1/0; uncited factual claims scored `1 − needs_citation`.
- **R_format** = `0.5·has(<answer>) + 0.3·has(well-formed <cite>) + 0.2·has(≥1 tool call)` (regex only).
- **R_search** = `min(valid_tool_calls / 3, 1.0)`.
- No `<answer>` parsed ⇒ R_rubric=R_cite=0, format+search still credited. No truncation penalty.
- **Evolving rubrics (RLER)**: each step, gpt-4.1 generates ≤5 discriminative ± rubrics per
  prompt group from the 8 rollouts (+1/−1 weights); buffer keeps ≤5 active by reward-std
  pruning; persistent GT rubrics always included. `apply_adaptive_rubric_reward=false` in
  their own code = static-rubrics-only ablation.

### Optimization (`grpo_fast.py`)
- GRPO: group of 8/prompt, advantage `(r−mean)/(std+1e-8)` (**std-norm ON**), scalar per
  trajectory broadcast per-token.
- **Zero-variance groups dropped** (std==0 groups filtered before the update).
- KL: **k3** estimator, **added to loss** (`beta=0.001`), reference = frozen init, never updated.
- Clip ±0.2 two-sided; loss = **token-mean** over all response tokens in the micro-batch;
  `num_epochs=1`, `num_mini_batches=1` → strictly on-policy, 1 optimizer step per batch.
- lr 5e-7 constant; batch 32 prompts × 8 samples = 256 trajectories/step; 2 nodes × 8 GPUs.

## 2. Classified deltas for our SkyRL flat arm

### A. Exact matches (free — enforce in `DrTuluEnv` build)
| dimension | both |
|---|---|
| protocol + parser | `<call_tool>` / `v20250824` parser class reused verbatim (`dr_agent.tool_interface.tool_parsers`) |
| stop strings | `</call_tool>`, `</call>` (kept in output) |
| tool-output wrap | `\n<tool_output><snippet id=..>…</tool_output>\n\n`, loss-masked |
| system prompt | same `unified_tool_calling_v20250907.yaml` + per-type additional_instructions (already used in SFT + eval) |
| R_rubric machinery | our `judge.py` is a verbatim port (same prompt, 0-2 scale, /2, weighted mean) |
| train sampling | T=1.0, top_p=1.0 |
| truncation | keep + reward normally (no non-stop penalty) |
| GRPO shape | std-norm ON (SkyRL default `grpo_norm_by_std=true`), k3 (`kl_estimator_type` default), KL in loss (`use_kl_loss=true`), clip 0.2/0.2 (default), token-mean loss (default), on-policy 1 pass (`update_epochs_per_batch=1`) |
| tool-call budget message | inject masked "Max tool calls exceeded." style observation, let model finish |

### B. Forced / already-locked deviations (document, don't relitigate)
| dimension | original | ours | why |
|---|---|---|---|
| base model | DR-Tulu-SFT-8B | Qwen3.5-4B + our SFT | compute; held constant across arms |
| judges | gpt-4.1-mini (+gpt-4o-mini cites) | local Qwen3.5-4B think-off | $0, reproducible, held constant across arms + eval |
| tools / corpus | live web (google, browse, snippet_search via MCP) | `snippet_search` over frozen bm25s corpus (matches our SFT+eval; browse off, as in gate eval) | frozen-corpus design (reproducibility, $0); same retrieval for both arms |
| turn encoding | raw single-stream continuation | chat-template multi-turn (tool output = user turn, masked via all_assistant_messages) | matches DR-Tulu's own *agent library* (`client.py`), our SFT data, and our eval harness — self-consistent pipeline beats matching their trainer's internal encoding |
| batch scale | 32×8=256 traj/step, 16 GPUs | 4×4=16 traj/step, 4 GPUs | compute; identical across our two arms |
| tool-call cap | 10 (11 w/ off-by-one) | 12 (= our eval baseline convention) | RL≡eval consistency rule beats matching 10 |
| length budget | 16384 total incl. tool tokens | 8192/turn within 32k context | = our eval COMMON; more generous |
| eval-time temp | 0.6 | 1.0 | our locked eval convention |
| evolving rubrics (RLER adaptive) | ON (gpt-4.1 generation each step) | OFF — persistent GT rubrics only | needs a strong API generator model per step; equals `apply_adaptive_rubric_reward=false` ablation of their own code; static rubrics held constant across arms |
| optimizer scale | lr 5e-7, KL β=0.001, n=8 | lr 1e-6, KL 0.05, n=4 | cross-arm parity with the recursive **thesis run** `rl_fresh100` (100 steps, `rl_fresh_400`, KL 0.05 probe-validated) outranks matching original scale |
| zero-variance group filter | ON | OFF (SkyRL default; both arms) | parity; minor (only removes no-gradient groups) |
| reward scale | ×10 (`verification_reward`) | ×1 (`DR_TULU_REWARD_SCALE`, set 10 to restore) | provably no-op under std-normalized GRPO advantages; ×1 keeps W&B dashboards comparable across arms |

### C. The one open decision — reward composition
Original: `0.5·rubric + 0.2·cite + 0.2·format + 0.1·search` (×10). Our recursive recipe
(validated, rising-R trifecta): **rubric-only** root R. Options:

1. **Rubric-only, both arms** — max cross-arm parity; flat arm = "DR-Tulu RL, rubric term
   only" (their own static-rubric ablation, minus shaping terms). Note format/search terms
   are near-constant ≈1 at our SFT init (100% finalize, all search) ⇒ they mostly cancel in
   the GRPO group baseline anyway; the material omission is **R_cite**.
2. **Full composite (local-judge R_cite), flat arm only** — most faithful "DR-Tulu as
   intended"; asymmetric rewards across arms = each method trained as its authors intended
   (their citation reward vs our provenance credit) — defensible but weakens the
   "only-credit-differs" causal claim.
3. **Full composite, both arms** — faithful + symmetric, but invalidates the validated
   recursive recipe (would need re-run; grounding-aware reward previously caused judge-load
   failures).

**Chosen (2026-07-07, Louis): option 2 — full composite, flat arm only.** The flat baseline
trains on the faithful DR-Tulu reward (rubric + citation + format + search-turns, citation
judged by the same local Qwen judge); the recursive arm keeps its validated rubric-only root R
with provenance credit. Framing: each method is trained with its own grounding mechanism as
its authors intended (DR-Tulu's citation reward ↔ our provenance credit).

## 3. Provenance
Verified from source at file:line granularity; key files:
`open_instruct/grpo_fast.py` (advantages 1351-1375, KL/loss 963-997, zero-var filter 1450-56,
masking 910-41), `tool_utils/tool_vllm.py` (rollout loop 244-408), `search_rewards/
longform_rubric_rewards.py` (weights 15-25, composition 244-53), `search_rewards/utils/
{format_utils,search_utils,citation_utils,rubric_utils}.py` (term formulas),
`search_utils/system_prompts/unified_tool_calling_v20250907.yaml`.
