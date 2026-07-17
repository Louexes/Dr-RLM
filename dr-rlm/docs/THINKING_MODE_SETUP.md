# Thinking-ON vs Thinking-OFF — exact setup, differences, and revert recipe

**Purpose.** We ran an untrained DR-RLM thinking-on/off comparison (2026-06-30). This file records
**every** difference between the two setups so we can reproduce either exactly, and in particular
**revert to the thinking-OFF setup (our canonical / pre-2026-06-30 configuration) byte-for-byte.**

> TL;DR revert to thinking-OFF: use `prompts/system_prompt.txt` (canonical, unchanged), serve with
> `enable_thinking=false`, and set **none** of the `DR_RLM_SALVAGE_ON_MAX_TURNS` /
> `DR_RLM_FINALIZE_NUDGE_TURNS` env vars. Everything else is already identical. See §4.

---

## 1. What is identical across both arms (the locked config)

These do **not** change between thinking-on and thinking-off — same in both jobs:

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3.5-4B` |
| `--max-recursion-depth` | `1` (orchestrator + worker) |
| `--max-iterations` (max_turns) | `10` |
| `--max-children` | `10` |
| `--max-completion-tokens` | `8192` (raised for the think block; off arm uses 8192 too) |
| `--max-tool-calls` | `80` |
| `--child-return-mode` | `structured` |
| share_mode | `ledger_support` |
| Search backend | `bm25s`, frozen corpus (`data/frozen_corpus/`, 139,257 docs) |
| Sampling | `temp 1.0 / top_p 1.0 / top_k -1(off) / min_p 0.0 / presence_penalty 0.0` (DR-Tulu harness defaults) |
| N | 120 (ResearchQA `researchqa_strat120`) |
| Judge | Qwen3.5-4B served `enable_thinking=false`, aliased `gpt-4.1-mini-2025-04-14` (judge is **always** thinking-off) |

## 2. The ONLY runtime differences (thinking-OFF → thinking-ON)

There are exactly **two** differences for the plain on/off comparison, plus **two more** env-var
flags that were added ONLY for the empty-rescue follow-up (§3).

| | Thinking-OFF (canonical / previous) | Thinking-ON |
|---|---|---|
| **vLLM serve flag** | `--default-chat-template-kwargs '{"enable_thinking": false}'` | `--default-chat-template-kwargs '{"enable_thinking": true}'` |
| **System-prompt file** (`DR_RLM_PROMPT_FILE`) | *(unset)* → `prompts/system_prompt.txt` (canonical) | `prompts/system_prompt_thinkon.txt` |
| Job file | `agent/eval_drrlm_untrained_thinkoff.job` | `agent/eval_drrlm_untrained_thinkon.job` |

### 2a. The prompt-file difference, spelled out

`system_prompt_thinkon.txt` = an exact copy of canonical `system_prompt.txt` with **two added
blocks** (everything else byte-identical). `diff system_prompt.txt system_prompt_thinkon.txt`:

**Block A — added to `REPL_RULES`** (the "reason inside `<think>`" instruction; added when the
thinking-ON de-risk probe was built):
```
- THINK FIRST: reason about your plan — decomposition, what to `search`, which sub-questions to
  delegate, how to synthesize — inside the `<think>...</think>` block. AFTER `</think>`, emit your
  work. Keep ALL reasoning in `<think>`; the ```repl``` block holds only executable code
  (search/delegate/answer assembly) and `answer["content"]` holds only the report — never plans
  or narration.
```

**Block B — added to `FINISHING`** (the empty-rescue "never run out unsubmitted" tweak, §3; added
2026-06-30):
```
- NEVER let your turn budget run out without submitting. Reasoning inside `<think>` does NOT
  submit anything — only writing `answer["content"]` and setting `answer["ready"] = True` in a
  ```repl``` block does. The moment you have a defensible answer, or you are running low on turns,
  SUBMIT: a rough submitted report always beats an unsubmitted one, which scores zero.
```

So there are effectively two versions of the thinking-ON prompt:
- **probe version** = canonical + Block A only (the original 0.237 run).
- **rescue version** = canonical + Block A + Block B (the empty-53 rescue run). This is the current
  content of `system_prompt_thinkon.txt`.

The canonical thinking-OFF prompt (`system_prompt.txt`) has **neither** block and is the single
source of truth — **do not edit it for thinking experiments.** Steer thinking behavior in the
variant file only.

## 3. Empty-rescue follow-up — the gated code changes (default OFF)

The thinking-ON probe scored 0 on 53/116 items purely on **finalization failure** (100% hit
max_turns, 0% ever set `answer["ready"]=True`; 24/53 had a written `answer["content"]` that was
silently discarded). To test how much of that is rescuable, three fixes were added — **all
default-OFF and behavior-byte-identical unless explicitly turned on:**

| Fix | Mechanism | How to turn ON | Default |
|---|---|---|---|
| **Salvage** | `BaseRLMEnv._try_salvage()` captures a written-but-unsubmitted `answer["content"]` on max_turns (reuses the normal-path `_finalize_answer`→`get_metrics`→`final_response`) | env var `DR_RLM_SALVAGE_ON_MAX_TURNS=1` | OFF → report discarded as before |
| **Earlier+forceful nudge** | `DrRlmEnv._get_user_prompt` fires a "STOP searching/reasoning, submit NOW" message at `remaining <= max(2, N)` | env var `DR_RLM_FINALIZE_NUDGE_TURNS=N` (rescue used `3`) | `0` → threshold stays `<=2` with the original soft text |
| **Prompt tweak** | Block B above | use the rescue version of `system_prompt_thinkon.txt` | n/a (variant-only) |

Code locations (nested git repo `dr-rlm/rl/skyrl`, anchor commit `77c7947c`):
- `skyrl-gym/skyrl_gym/envs/rlm/env.py` — `import os`; two flag reads in `__init__`; `_try_salvage()`;
  the two `_try_salvage()` calls in the max_turns done-branches of `step()`.
- `examples/train/dr_rlm/dr_rlm_env.py` — the `_get_user_prompt` nudge now branches on
  `getattr(self, "_finalize_nudge_turns", 0)`; with the flag at 0 it emits the **original** text
  (byte-identical).

Rescue job: `agent/eval_drrlm_thinkon_empty53_fix.job`; subset `data/subsets/researchqa_thinkon_empty53.jsonl` (the 53 zero-scoring ids, mapped via `original_data.orig_id`).

## 4. EXACT REVERT RECIPE → thinking-OFF (our canonical setup)

To get back to the pre-2026-06-30 thinking-OFF configuration exactly:

1. **Serve thinking-off:** `--default-chat-template-kwargs '{"enable_thinking": false}'`.
2. **Prompt:** leave `DR_RLM_PROMPT_FILE` unset (→ canonical `prompts/system_prompt.txt`).
   *Do not* point it at `system_prompt_thinkon.txt`.
3. **Do NOT export** `DR_RLM_SALVAGE_ON_MAX_TURNS` or `DR_RLM_FINALIZE_NUDGE_TURNS` (or set them to
   `0`). With both unset, the salvage no-ops and the nudge uses the original `<=2` soft text — the
   env behaves byte-identically to before this session.
4. Sampling / depth / turns / children / tool-calls / child-return-mode / search backend: already
   identical — nothing to change.

The canonical reference job is `agent/eval_drrlm_untrained_thinkoff.job` (it already does 1–4). The
gated code additions can stay in the tree; they are inert when the flags are unset, so no code revert
is needed to recover the previous behavior. (If a *literal source* revert is ever wanted: only the
salvage block + the `_finalize_nudge_turns` branch in `_get_user_prompt` are this session's additions;
everything else in `git diff 77c7947c` predates it — trajectory capture, the bounce gate, etc.)

## 5. Thesis-relevant finding (why the empties matter)

The thinking-ON empties were **not** a quality failure — they were a **harness-finalization**
failure: the untrained model reasons well but never reliably executes the submit contract
(`answer["ready"]=True`). This is direct evidence that **SFT on rejection-sampled, correctly-finalized
trajectories is needed** to teach the submit protocol — exactly the demonstrations
`gen_recursive_sft.py` produces (every kept SFT trajectory finalized with `ready=True`). The rescue
run quantifies the ceiling that finalization-only fixes (no training) can recover; the gap that
remains motivates SFT. See `EXPERIMENT_LOG.md` (thinking-ON 0.237 row + empty-rescue row).

## 6. Thinking-ON is SAFE for RL training — audited + smoke-validated (2026-07-01)

Full SkyRL audit + a 4×H100 plumbing smoke (`experiments/provenance_smoke/jobs/smoke_thinkon_plumbing.job`,
job 24346258, 287 SBU) confirm `enable_thinking=true` does not break RL:
- **Plumbing verified:** `chat_template_kwargs.enable_thinking` is applied at every `apply_chat_template`
  site; training is token-in-token-out (the eval-only api_client double-templating is NOT in the
  training path); think tokens are loss-masked=1 (`get_turn_loss_mask` over `output_ids`); multi-turn
  history strips `<think>` identically at generate and train time (Qwen3 default template) → consistent.
- **Smoke result (thinking-ON):** first rollout batch 16/16 trajectories; reports finalize
  un-truncated (longest 10,196 chars); report_reward mean 0.510; **0 length-aborts, 0 crashes**;
  `convert_to_training_input` (response_ids+loss_mask assembly) and `fwd_logprobs` both ran clean.

**Required changes to flip RL/SFT to thinking-ON (do all three):**
1. `generator.chat_template_kwargs.enable_thinking=false → true` in all 5 `run_dr_rlm_L*.sh`.
2. **Raise `max_generate_length` 1024 → ≥4096** (the L*.sh default is 1024, too small once a `<think>`
   block precedes the repl/report → truncates the report mid-write → broken finalization; the smoke
   used 4096 with 0 aborts). `max_generate_length` was thinking-neutral in eval (compact CoT), so ~5%
   cost, not 2×.
3. **Regenerate SFT data thinking-ON** (`gen_recursive_sft.py` served thinking-on) so demos contain
   `<think>` reasoning; watch the multi-turn think-mask (default template strips think from non-final
   turns — RL step-wise avoids this, SFT needs a per-turn split or the `qwen3_acc_thinking.jinja2` template).
