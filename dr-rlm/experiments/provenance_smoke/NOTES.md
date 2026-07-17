# provenance_smoke — small-scale RL smoke test of provenance credit (G4 de-risk)

**Goal:** first end-to-end RL evidence that provenance-attributed per-node credit
(`reward_mode=rer`, `share_mode=ledger_support` — the mode the offline POC validated,
see `results/provenance_credit_POC.md`) **functions as a training signal**: it survives
the online pipeline and applies differential pressure the broadcast baseline cannot.
NOT a quality benchmark — at ~40 steps no L3>L1 report-quality claim is possible
(DR-Tulu's own ablations separate after 1000+ steps).

## Design

| | |
|---|---|
| Policy / warm start | `rl-research/DR-Tulu-8B` (Qwen3-based; flat-trained, already searches+cites) |
| Data | 32 train / 8 val decompose-shaped dr-tulu-rl prompts (`data/poc_corpus/*.parquet`, rubrics in reward_spec) |
| Corpus | frozen snapshot of the POC web evidence (5,079 docs, `data/poc_corpus/corpus.jsonl`, local_jsonl backend) — stationary, guaranteed-relevant |
| Judge | gemini-2.5-flash via OpenAI-compat (same judge that validated the POC signal; held constant across arms) |
| Arms | **L3** = per-node `ledger_support` credit; **L1** = flatten+broadcast (stock SkyRL). Same seeds/steps/everything else |
| Scale | batch 4 × n_samples 4 = 16 trees/step; 5 epochs ≈ 40 steps; depth 1, ≤4 children/node, 10 turns |
| Hardware | 1× gpu_a100 node (4 GPU; TP4 colocated engine + FSDP2) per arm |

## Run order
1. `sbatch jobs/rgate_drtulu8b.job` — recursion gate (1 GPU, ~1–2 h): does DR-Tulu-8B
   delegate under the env prompt? **Gate: recursion rate ≥ ~0.3.** If ~0 → light SFT on
   the POC trees first (`sft/gen_recursive_sft.py`; gemini is a valid recursive teacher
   under the unified env — POC trees fan out 2–6 children).
2. `sbatch jobs/smoke_L3.job` then `sbatch jobs/smoke_L1.job` (can run concurrently;
   independent nodes).
3. `python analysis/provenance_smoke_curves.py --l3 runs/provenance_smoke/credit_metrics_L3.jsonl
   --l1 runs/provenance_smoke/credit_metrics_L1.jsonl --bucket 16 --out results/provenance_smoke_curves.json`

## Pre-registered evidence criteria

**A. Plumbing must-pass (L3, from step 1 onward)** — "the mechanism runs":
- per-node r_a populated and finite; un-flatten produces #trajectories == #nodes (trainer
  runs without contiguity asserts firing);
- conservation: |Σ_a r_a − R| ≈ 0 per tree (`mean_abs_conservation_gap` small);
- within-tree advantage spread: `mean_within_tree_ra_std` > 0 (L1 is structurally 0) —
  the credit DIFFERENTIATES siblings online;
- judge health: `frac_R_positive` well above 0; no judge-zero collapse.

**B. Differential learning evidence (L3 vs L1 slopes over ~40 steps)** — "it DOES something":
- `frac_children_credit0` (orphan fraction) trends DOWN under L3; its analog
  (`frac_children_cite0`) flat-ish under L1 — broadcast applies no per-child pressure;
- `mean_cites_per_child` trends UP under L3 vs flat under L1 (children learn to return
  cited evidence — the behavior ledger_support pays);
- `mean_report_reward` non-decreasing in both (sanity).

**C. Reward-hacking watch:** cites/child must not explode (citation stuffing); if it
does, that motivates `gamma_cost` > 0 — itself a useful thesis finding.

Pass = A holds and ≥1 of the B slopes separates in the predicted direction. That, stacked
on the offline LOCO alignment (within-tree ρ̄=0.425, p=0.0025), is the
"working + worth naming as a contribution" evidence package: the signal is (i) aligned
with the true counterfactual, (ii) computable online at training scale, (iii) the only
arm applying differential per-child pressure, and (iv) behavior moves along it.

## Isolation (per 2026-06-10 policy — see docs/provenance_poc_CHANGES.md)
- Credit telemetry hook in `dr_rlm_generator.py` is **opt-in** via
  `DR_RLM_CREDIT_METRICS_PATH` (default off = byte-identical; never raises).
- `convert_drtulu_rl.py` empty-`extra_info` parquet crash fix (pyarrow cannot write
  childless structs) — unconditional BUG fix, no semantic change.
- Everything else here is additive (new files under experiments/, analysis/, data/poc_corpus/).
- Smoke jobs do NOT use the POC robustness prompt (policy prompt = canonical
  `prompts/system_prompt.txt` via the env) — the trained policy must learn robustness;
  only the POC generation used the private prompt.
