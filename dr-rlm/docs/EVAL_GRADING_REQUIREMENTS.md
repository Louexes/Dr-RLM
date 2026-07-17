# Eval grading requirements — what each benchmark needs captured to grade correctly

**Purpose.** Grading silently returns 0 / wrong scores if the reports don't carry the
fields the grader reads. This is the checklist so an eval run captures everything BEFORE
we spend GPU on it — untrained AND every trained (SFT/RL) checkpoint. Verify with
`analysis/` ad-hoc or the snippet at the bottom.

## Per-benchmark requirements

| Benchmark | Grader | Report fields the grader reads | Capture flag needed | Grade command |
|---|---|---|---|---|
| **ResearchQA** | LOCAL Qwen judge (thinking-OFF), coverage | `final_response` | none | `scripts/evaluate.py researchqa <jsonl>` (OPENAI_BASE_URL→local judge) |
| **HealthBench** | **Gemini flash** (OpenAI-compat), rubric — **NOT local judge** | `final_response` + `original_data.rubrics` (carried from the `*_LOCAL_PATH` subset) | none (grade off-GPU) | `OPENAI_API_KEY=$GEMINI_API_KEY OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ scripts/evaluate.py healthbench <jsonl> --grader-model gemini-2.5-flash --save_path <json>` |
| **DRB** | **Gemini**: RACE=gemini-2.5-pro, FACT=gemini-2.5-flash | RACE: `final_response`. **FACT: `full_traces.tool_calls`** (source set FACT validates `<cite id>` against) | **`DR_RLM_CAPTURE_LEDGER=1`** (else `tool_calls=[]` → FACT=0) | `evaluation/deep_research_bench_eval/run_eval.py --input_file <jsonl> --task_name <name> --only_en --output_dir <dir>` |
| **SQA v2** | **Gemini** flash (inspect_ai/astabench) | global_avg + `<snippet id=…>` blocks in `full_traces.generated_text` for cite-precision/recall | **`DR_RLM_CAPTURE_LEDGER=1`** (else no `<snippet>` blocks → cite metrics=0) | `evaluation/sqa_eval/run_eval.py run --input_file <jsonl> --scorer_model google/gemini-2.5-flash --max_connections 16 --output_dir <dir>` (`GOOGLE_API_KEY=$GEMINI_API_KEY`) |

## Pre-launch checklist for ANY DR-RLM eval job (`--driver new`)

- [ ] `DR_RLM_CAPTURE_LEDGER=1` exported — **required for DRB-FACT and SQA-cite**; harmless elsewhere. (Baked into `agent/eval_drrlm_{drb,sqav2,healthbench}.job` — don't strip it.) See [[project-fact-zero-toolcalls-bug]].
- [ ] `--save-trajectories` set (writes the ledger to disk for the credit pipeline; not strictly needed for `tool_calls` in the output jsonl, but keep for provenance).
- [ ] Thinking mode matches training regime: `enable_thinking=true` + `DR_RLM_PROMPT_FILE=system_prompt_thinkon.txt` (current direction = thinking-ON). Strict = **no** `DR_RLM_SALVAGE_ON_MAX_TURNS` / `DR_RLM_FINALIZE_NUDGE_TURNS`. See [[project-thinking-onoff-baselines]], `docs/decisions.md`.
- [ ] **Judge is ALWAYS thinking-OFF** — HB/ResearchQA local judge served `enable_thinking=false`; for thinking-ON gen use the two-serve pattern (gen ON → judge OFF). Never alias a thinking-ON endpoint as the grader. See [[feedback-judge-always-thinking-off]].
- [ ] Judge serve capped `--max-num-seqs 32` (the HB rubric grader fires ~1000 concurrent calls; uncapped → vLLM KV-cache deadlock, 0 tok/s). Learned 2026-07-01 (job 24350042 deadlocked).
- [ ] Right subset via `*_LOCAL_PATH` (researchqa_strat120 / healthbench_strat120 / drb_en50 / sqav2_cs100) — same subsets as the DR-Tulu arm for a fair head-to-head.
- [ ] `--mem` ≤180G so it bills 1 GPU, not 2 (shared-node cap).

## SBU discipline — grading NEVER holds a GPU
Grading is a light, latency-bound judge workload; a dedicated H100 sits ~99% idle during it. **Grade off-GPU via Gemini flash** (DRB, SQA, HB all do). The local Qwen-4B judge for HB **retry-storms**: `evaluate.py`'s `max_tokens=1000` cap (only bumped to 4096 for `_is_gemini`) truncates the rubric JSON → the HealthBench harness retries forever → hours of GPU burn, **0 score**. Cost this cost us: ~6.7 GPU-hr across 4 dead jobs (24346833/24350042/24350290/24351063) before switching to Gemini (3s for 5 reports, pennies). Rule: **split gen (GPU) from grade (Gemini, no GPU)** — HB job should be gen-only + a separate Gemini pass, like `eval_drrlm_{drb,sqav2}_gen.job`. Only ResearchQA still uses the local judge (coverage JSON is short, no truncation, it completes) — cap its grade wall and watch for non-draining.

## DR-Tulu (ReAct) arm
DR-Tulu populates `full_traces.tool_calls` natively (its ReAct search calls) → DRB-FACT & SQA-cite work with no flag. Same subsets + same Gemini/local graders as above. Only difference: no ledger flag needed.

## One-line verify (run over any reports jsonl before grading)
```python
import json; rows=[json.loads(l) for l in open(F) if l.strip()]
print("nonempty", sum(1 for r in rows if (r.get('final_response') or '').strip()), "/", len(rows))
print("DRB-FACT tool_calls", sum(1 for r in rows if (r.get('full_traces') or {}).get('tool_calls')), "/", len(rows))   # DRB only
print("SQA snippet-blocks", sum(1 for r in rows if '<snippet id=' in ((r.get('full_traces') or {}).get('generated_text') or '')), "/", len(rows))  # SQA only
print("HB rubrics", sum(1 for r in rows if (r.get('original_data') or {}).get('rubrics')), "/", len(rows))  # HB only
```
