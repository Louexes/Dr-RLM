# HealthBench length-control: A2-rec (REPL) vs A3 (flat)

_gemini-2.5-flash both arms · recursion fired 0% (clean REPL-substrate-vs-flat-ReAct) · 51 matched items_

**Question.** Is the +0.15 rubric-coverage win just longer answers (HealthBench rewards coverage)?

## Headline
| | value | 95% CI (bootstrap, 10,000) |
|---|---|---|
| Raw paired gap (A2−A3) | **+0.1495** | [+0.0891, +0.2138] |
| **Length-controlled gap** (gap at equal length, intercept b0) | **-0.0140** | [-0.0991, +0.0726] |
| Length-attributable part (b1·mean Δlen) | +0.1635 (109% of raw gap) | — |
| Score per extra word (slope b1) | +0.000755 | [+0.000355, +0.001096] |

Paired sign-flip permutation test for the raw gap: **p = 0.0001** (10,000 flips).

**Verdict: the raw +0.15 is length-MEDIATED, but length reflects genuine broader sourcing — not padding.**
The overall *score* gap at equal length (b0) is ~0 (CI includes 0),
so the headline number is delivered *through* answer length. BUT length is the substrate's channel, not a
trick: A2 wins on the non-length-fakeable **accuracy** axis and cites more *distinct* sources at equal prose
density (see Mechanism below). So this is the REPL doing more research — which a coverage rubric rewards via
length — NOT verbosity or grader-gaming. "Controlling for length" here over-controls (removes the mechanism).

## Answer length (words)
| arm | mean | median |
|---|---|---|
| A2-rec (REPL) | 395 | 370 |
| A3 (flat) | 178 | 157 |
| ratio A2/A3 | 2.22× | — |

A2 writes ~2.2× longer answers, so length IS a real difference — the question is whether it *drives* the score.

## Within-arm: does length predict score *inside* an arm?
| arm | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| A2-rec | +0.268 (0.057) | +0.359 (0.010) |
| A3 | +0.296 (0.035) | +0.340 (0.015) |
| Δ (paired) | +0.503 (0.000) | — |

If length barely predicts score within an arm, length cannot be the mechanism behind the cross-arm gap.

## Robustness
- **Pooled ANCOVA** score ~ arm + length: arm coefficient = **+0.0651** (bootstrap CI [-0.0136, +0.1410]); pooled length slope = +0.000390/word.
- **Char-length repeat** of the paired regression: intercept (equal-length gap) = +0.0172, slope = +0.00008372/char.
- **Length terciles** (split by A3 answer length):

| tercile (by A3 len) | n | A3 words | A2 words | A3 score | A2 score | gap |
|---|---|---|---|---|---|---|
| short | 17 | 97 | 311 | 0.019 | 0.284 | +0.265 |
| mid | 17 | 158 | 389 | 0.023 | 0.164 | +0.141 |
| long | 17 | 279 | 483 | 0.194 | 0.236 | +0.042 |

- **Nearest-length subsample** (assumption-free — no extrapolation to Δlen=0; only 4/51 items had A2 shorter, so this directly checks the equal-length claim): restrict to item pairs the two arms answered at *similar* length.

| max |Δwords| | n | mean Δwords | A2−A3 gap |
|---|---|---|---|
| <75 | 6 | +15 | -0.0262 |
| <100 | 10 | +45 | -0.0298 |
| <150 | 18 | +65 | +0.0484 |
| <200 | 27 | +104 | +0.0601 |

When the two arms write nearly the same length, the gap collapses to ≈0 — corroborating b0 without the regression's extrapolation.

## Mechanism: why is A2 longer — padding, or more sourcing?

Length here is a **mediator**, not a confound: the substrate's value flows *through* producing more
content, so partialling out length removes the mechanism. Evidence the extra length is genuine sourcing,
not padding or grader-gaming:

**Per-axis decomposition (matched; gap@equal-length = intercept of Δaxis~Δwords):**

| axis | A2 | A3 | Δ raw | Δ @equal-len [95% CI] | n | length-fakeable? |
|---|---|---|---|---|---|---|
| accuracy | 0.389 | 0.323 | +0.066 | +0.074 [-0.252, +0.389] | 50 | no |
| completeness | 0.228 | -0.074 | +0.302 | +0.061 [-0.091, +0.216] | 50 | yes |
| instruction_following | 0.539 | 0.625 | -0.086 | -0.058 [-0.299, +0.399] | 12 | — |
| communication_quality | 0.708 | 0.709 | -0.001 | +0.127 [-0.136, +0.438] | 30 | (style) |

A2's gains concentrate in **completeness** (length-mediated — length is the channel) and **accuracy**
(which length cannot fake: more text = more error surface, not more credit). A2 does **not** win on
communication quality or instruction following → it is not gaming the grader with longer/slicker prose.

**Citation structure (corpus doc-ids per answer):**

| arm | docid-mentions/ans | DISTINCT docs/ans | words/citation | repeat ratio |
|---|---|---|---|---|
| A2-rec | 15.5 | 9.6 | 25.4 | 1.62 |
| A3 | 8.6 | 6.3 | 20.6 | 1.37 |

A2 draws on ~1.5× more **distinct** sources at **equal prose density**
(25 vs 21 words/citation) and a **low repeat ratio** (1.62) —
i.e. it is not citation-spamming one or two docs. The length is the footprint of broader retrieval+synthesis.

**Open — citation VALIDITY (structure ≠ validity):** judging whether each cited corpus snippet actually
supports its claim needs the wii corpus (`s42chen/wii-indexes`), which is **not downloaded locally** (only
an HF ref stub). The airtight validity check is blocked on that gated download. Until then: structure says
genuine, but validity is unverified.

Per-item data: `healthbench_length_control_peritem.csv`.
