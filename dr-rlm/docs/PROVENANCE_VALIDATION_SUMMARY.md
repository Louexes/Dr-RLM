# Provenance-Credit Validation — Summary of RL Findings

**Claim under test:** per-node *provenance credit* (distribute the report reward to tree nodes by
which node's retrieved-and-cited evidence supports the answer) is a working RL training signal that
teaches a recursive deep-research model to ground its answers.

**Verdict: VALIDATED.** Three independent lines of evidence (figures in `docs/figures/`).

---

> **Reward note (important):** all three validation runs below use **`rubric_reward` ONLY** —
> DR Tulu's **`citation_reward` was OFF** (`DR_RLM_CITATION_REWARD` unset). The grounding behavior
> therefore comes entirely from the **credit attribution** (`ledger_support` routing reward to
> children whose evidence is cited), *not* from any citation-specific reward term. We ported DR
> Tulu's citation_reward (and re-expressed format/num_search_turns), but they are env-gated and
> were **not active in these results**. The one run that turned citation_reward ON (Run A) is
> summarized separately below — it OOM'd at step 20 and is *not* part of this validation.

## Evidence 1 — It works as a training signal (clean run)
`v1-clean` (ledger_support credit, credit_share root, **rubric-only** reward, GRPO; Qwen3.5-4B;
judge-uncontaminated via checkpointed chunks on fresh quota). Over ~24 GRPO steps:

| metric | start | end | direction |
|---|---:|---:|---|
| orphan children (cited nothing) | 93% | **43%** | ↓ learns to cite |
| citations / child | 0.13 | **1.68** | ↑ ~13× |
| report R (judge-alive trees) | ~0.75 | ~0.76 | **flat — no quality cost** |

→ Provenance credit teaches grounded retrieval **at no cost to report quality.**
Figure: `fig1_v1clean_trajectory.png`.

## Evidence 2 — It's the *credit* specifically, not generic RL (dissociation)
Same everything, only the credit scheme differs:

| arm | orphan % | cites / child | citing learned? |
|---|---:|---:|---|
| **L1 broadcast** (no per-node credit; root R copied to all) | 97.9% | 0.05 | **no — extinction** |
| **L3 provenance credit** (start) | 93% | 0.13 | — |
| **L3 provenance credit** (end) | 43% | 1.68 | **yes** |

→ With broadcast credit the model never learns to cite; only **per-node provenance credit** drives
grounding. (Consistent with the earlier smoke double-dissociation: citing ρ=+0.94 under L3 vs
ρ=−0.82 under L1.) Figure: `fig2_dissociation.png`.

## Evidence 3 — The credit lands on the right nodes (attribution mechanism)
Per-node credit `r_a` in the clean run, children grouped by whether their evidence was cited:

| node type | mean credit r_a | n |
|---|---:|---:|
| citing children (evidence cited in report) | **0.170** | 442 |
| orphan children (no cited evidence) | **0.000** | 1042 |

→ Credit flows **exactly** to the nodes whose evidence the report uses; orphans get nothing. This is
the provenance-attribution mechanism doing what it claims. (Offline LOCO check: within-tree ρ̄=0.425,
p=0.003.) Figure: `fig3_node_attribution.png`.

---

## Aside — Run A (citation_reward ON): the mirror-image tradeoff (NOT in the validation)
The one run with **`citation_reward` ON** (`DR_RLM_CITATION_REWARD=1`, R = 0.5·rubric + 0.2·citation),
also full_R root + rer_pernode estimator. OOM'd at step 20, but ~21 steps show a clear pattern:

| | orphan % | cites/child | citation_R | blended R |
|---|---:|---:|---:|---:|
| Run A start→end | 90→85 (**plateau, weak**) | 0.21→~0.4 | ~0.33 (flat) | **0.44→0.52 (↑)** |
| v1-clean (ref) | 93→**43** (strong) | 0.13→**1.68** | (off) | flat |

→ Run A is the **mirror image** of v1-clean: it made **R rise** (the root got a synthesis signal
from full_R + citation) **but citing barely learned** (orphans stuck ~85% vs v1-clean's 43%). So
neither config was the "complete package" (citing↑ AND R↑); they were the two halves.

**Root cause found & fixed (2026-06-23) — it was an estimator bug, not a fundamental tradeoff.**
Run A (and v2/v3) ran the `rer_pernode` estimator with its depth **inverse-frequency weighting** on.
In our *shallow* trees (depth-0 root + depth-1 children) that weighting up-weights the rare ROOT and
down-weights the common CHILDREN, **halving the children's collective gradient share (~83%→50%)** and
starving the citing signal — while the depth-cohort baseline (the actual per-role separation) was
fine all along. Turning the weighting off (now the default, `DR_RLM_DEPTH_WEIGHTING=0`) restores
children to ~80% gradient share (strong citing, like v1-clean) while the root keeps ~20% (so R still
trains). Numerically verified + unit-tested (`tests/test_advantage.py`, 107 passed). The locked
complete-package config (`jobs/smoke_L3_complete.job`) is expected to deliver citing↑ AND R↑ together;
budget-blocked. Other Run A caveats remain: single config, OOM'd at step 20, some judge throttle.
See `RL_TRAINING_LOG.md` "✅ RESOLVED" for the full diagnosis.

## Honest scope / caveats
- It's a **smoke-scale** result (4B model, 32-prompt decompose-shaped subset of dr-tulu-rl-data,
  frozen snapshot corpus) — evidence the mechanism *works*, not absolute benchmark numbers.
- Citing learns to a **~45% orphan plateau**, not to zero.
- **R is stable, not rising** — by design: under credit_share the root earns ~0 reward 74% of the
  time (its synthesis is untrained). Raising R is the open extension (train the root: per-role
  baseline separation + grounding-aware root reward), not a matter of more steps.
- The original v1's apparent "R-decline" was a **judge-outage artifact** (Gemini rate-limit), not a
  real quality loss — which is why the judge-uncontaminated v1-clean rerun was needed.

See `RL_TRAINING_LOG.md` for the full run history, SBU costs, and infra notes.
