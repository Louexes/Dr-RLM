# healthbench_gemini

**Question:** A2-rec (REPL+recursion) vs A3 (flat DR-Tulu ReAct) on HealthBench-hard-72, gemini-2.5-flash
on both arms (agent + grader), within-benchmark comparison.

**Setup:** `jobs/a2rec_healthbench.job`, `a3_healthbench.job` (+ smokes). Subset
`data/subsets/healthbench_hard_repr.jsonl` (72, theme-proportional, difficulty-weighted). Config:
max_depth=2, orchestrator on, per-node budget=10. Outputs in `drrlm/runs/{A2rec,A3}/`.

**Result:** A2-rec **0.228** vs A3 **0.078** on the matched 51 items both completed (paired bootstrap
95% CI [+0.089, +0.213]; A2 wins 35 / ties 3 / A3 wins 13). Full-set nearly identical (0.221 N69 vs
0.077 N52) → the item mismatch is cosmetic.

**Caveats:** (1) **recursion fired 0/69** for A2-rec → this is REPL-vs-flat, NOT a recursion result.
(3) Drops were gemini 1M-tok/min quota: A2-rec client retries (lost 3); A3 litellm drops whole batch (lost 20).
(4) Grader fix required (gemini thinking truncated JSON → infinite retry); see docs/patches_to_deps.md.

**LENGTH CONTROL + MECHANISM (2026-06-06) — the +0.15 is length-MEDIATED but reflects genuine broader
sourcing, NOT padding.** `analysis/healthbench_length_control.py` → `results/healthbench_length_control.md`.
A2 writes **2.22× longer** answers (395 vs 178 words). Paired within-item regression Δscore~Δlen: overall
*score* gap at equal length b0 = −0.014 [−0.099, +0.073] (pooled ANCOVA +0.065 [−0.014, +0.141]; nearest-
length |Δw|<100 n=10 gap −0.03). So the headline number is delivered *through* length. **But length is the
substrate's channel, not a confound** — three tells it's real, not verbosity: (a) A2 cites **9.6 distinct
corpus docs vs 6.3** at *equal prose density* (~25 vs 21 words/cite), low repeat ratio (1.62 vs 1.37) →
broad sourcing not spam; (b) A2 wins the **non-length-fakeable accuracy axis** (+0.066 raw, +0.074 at equal
length, length-independent though CI wide); (c) A2 does NOT gain on communication-quality (−0.001) or
instruction-following (−0.086) → not grader-gaming. Win concentrates in **completeness** (+0.302, ~80%
length-mediated; A3 *negative* = genuinely missing content). ⇒ REPL does more genuine research a coverage
rubric rewards via length; "controlling for length" over-controls. **STILL OPEN: citation validity** —
structure≠validity; judging claim-support needs the wii corpus, which is **NOT downloaded** (HF ref stub),
so the airtight check is blocked. DRB FACT can't substitute (A2 0/32 parsed; A2 cites corpus-ids, A3 web URLs).
