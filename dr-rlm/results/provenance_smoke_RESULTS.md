# Provenance-credit RL smoke — FINAL VERDICT (L3 job 23688527 + L1 job 23706302, 2026-06-12)

> **ATTRIBUTION CONFIRMED — the credit assignment, specifically, drives evidence-grounding.**
> Matched 24-step window, identical configs except the credit scheme:
>
> | | **L3 (provenance credit)** | **L1 (broadcast)** |
> |---|---|---|
> | children citing evidence | 0.04 → **0.52** (ρ=+0.94) | 0.09 → **0.00** (ρ=−0.82) |
> | cites per child | 0.13 → **1.23** (ρ=+0.90) | 0.26 → **0.00** (ρ=−0.81) |
> | last-5-step cites/child | **1.31** | **0.00** (10 consecutive zero steps) |
> | report reward trend | 0.62 → 0.46 (declining) | 0.29 → 0.50 (**rising**) |
> | judge health | healthy through step 25 | healthy throughout (0 failures) |
>
> Same environment, same reward function, same data/judge/model — the arms diverge
> **monotonically in opposite directions** on the credited behavior. Under broadcast, RL
> *extinguishes* citation behavior entirely (it costs tokens and earns nothing differential)
> while climbing the rubric score with fluent, **ungrounded** reports — the rubric judge
> cannot tell the difference (the FACT≈0 failure mode as an RL attractor). Under provenance
> credit, grounding grows ~12× and is sustained.
>
> **The trade-off this exposes (the key G4 design input):** broadcast optimizes judged
> quality directly but destroys verifiability; provenance credit buys grounding at some
> short-run rubric cost (root under-incentivization — see Endgame/causes section; the hybrid
> root←R / children←credit scheme is the indicated fix, plus citation-validity metrics
> (FACT-style) as the quality axis the rubric judge is blind to).
>
> L1 was cut at step 24 (evidence saturated: 10 consecutive zero-citing steps; remaining
> wall-time would have added ~13 H100-GPU-hours of statistical redundancy).

**Headline: the provenance-credit mechanism WORKS as an RL training signal.** Over the
judge-healthy window (steps 1–25) of the first sustained DR-RLM training run, the policy's
behavior moved strongly and near-monotonically along the credit gradient, with stable
optimization. Combined with the offline LOCO validation (within-tree ρ̄=0.425, p=0.0025),
the mechanism-evidence package for naming Recursive Provenance Credit as the thesis
contribution is complete. Attribution (credit-specific vs generic-RL) awaits the paired L1
run (job 23706302, queued).

## Setup (canonical run)
Qwen3.5-4B (untrained, clean slate) on 4×H100; DrRlmEnv (REPL + delegation + ledger);
`reward_mode=rer, share_mode=ledger_support` (post scale-fix — Σr_a=R conservation verified
live, max r_a=1.0); 32 decompose-shaped dr-tulu-rl prompts over the frozen snapshot corpus;
gemini-2.5-flash judge; GRPO, batch 4 × 4 samples = 16 trees/step; depth 1, ≤4 children;
29 steps (TIMEOUT at 10h wall as designed), 480 trees, telemetry one row/tree.

## Evidence (bucket = one step = 16 trees)

| metric | step 1 | steps 18–19 (peak, judge healthy) | Spearman vs step (n=30) |
|---|---|---|---|
| children citing (>0 cited ids) | 0.043 | **0.55–0.64** | **+0.858** |
| cites per child | 0.13 | **1.68** | **+0.817** |
| children earning credit (r_a>0) | 0.022 | **0.40–0.42** | +0.063 (see note) |
| within-tree r_a spread (vs L1's structural 0) | 0.024 | 0.079–0.108 | — |
| report reward R̄ | 0.52 | 0.57–0.60 | −0.726 (endgame artifact, see below) |

- **Citing behavior is the cleanest signal**: ~13× rise in children-citing fraction,
  near-monotone (ρ=+0.86). This is precisely the behavior `ledger_support` pays and
  broadcast (L1) does not differentially reward.
- **Credited fraction peaked at 19× baseline** (0.022→0.42 by step 18) before the judge
  outage zeroed the credit mass (credit requires R>0 to distribute — hence the deceptively
  flat full-series Spearman; over steps 1–19 the rise is unambiguous).
- **Optimization stable** throughout: entropy 0.67–0.88 (no collapse), policy KL ≈ 0.07–0.08.
- **No citation stuffing**: cites/child plateaued ≈1.4 (bounded), not runaway.

## Endgame artifact (steps ~26–30): judge outage, not policy collapse
311 `judge giving up` warnings cluster at the end of the run — the Gemini judge hit
rate-limit/quota after ~9h and silently returned 0.0 (the DR-Tulu-inherited failure
semantics). R̄ collapsed 0.4→0.0 in 4 steps; with zero reward mass, credit zeroed and the
policy trained ~4 steps on dead rewards (late root answer-rate degraded). The evidence
window excludes these buckets.

**Two real findings for the full G4 runs:**
1. **Judge is a single point of failure with silent-zero semantics.** Required hardening:
   local vLLM judge (no quota) for long runs, and/or an all-zero-R step guard that skips the
   policy update (treat as judge outage, not signal).
2. **R did not rise with citing** over the healthy window (R̄ ≈ 0.5 flat). Citing behavior is
   necessary-but-not-sufficient for report quality; motivates the `gamma_cost` term and
   citation-validity weighting at scale. At smoke scale (25 steps, 4B model) a quality rise
   was not expected (DR-Tulu effects emerge after 1000+ steps).

## Pre-registered criteria (NOTES.md)
- **A. Plumbing must-pass: PASS.** Per-node r_a finite + conserved (post-fix), within-tree
  advantage spread > 0 every step (L1's is structurally 0), judge healthy through step 25,
  recursion persisted under training (~3–4 children/tree), 29 full GRPO steps end-to-end.
- **B. Differential slopes: L3 side PASS** (citing ρ=+0.86, cites/child ρ=+0.82, credited
  ×19 to peak). L1 contrast pending (job 23706302).
- **C. Stuffing watch: PASS** (bounded plateau; quality flag recorded above).

## Verdict
**The smoke's goal is met:** provenance graph credit assignment is computable online at
training scale, conserves and differentiates as designed, and demonstrably moves policy
behavior along the credit gradient under stable RL — *evidence that it is doing something*,
on top of the offline proof that the signal tracks ground-truth counterfactual contribution.
Worth naming as the thesis contribution's mechanism. Final attribution claim (credit-specific
vs generic RL) lands with the L1 comparison; final value claim (better models) is G4's job.

## Artifacts
- Telemetry: `runs/provenance_smoke/credit_metrics_L3.jsonl` (480 trees) + partials
  `_L3_run*.jsonl` from earlier attempts (incl. pre-scale-fix run1 corroboration).
- Curves: `results/provenance_smoke_curves.json`; W&B project `dr-rlm`, run `smoke_L3_ledger`.
- Infra changelog: `docs/provenance_poc_CHANGES.md` (#1–#11 + run-failure post-mortems).
