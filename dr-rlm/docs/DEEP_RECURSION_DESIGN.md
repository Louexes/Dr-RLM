# Deep Recursion (depth ≥ 2) — Design & Future Work

**Status:** future work / extension (not scheduled). The committed near-term result is the
depth-1 complete-package run (`jobs/smoke_L3_complete.job`). This document fully develops the
deeper-recursion direction so it is ready to pick up.

**Why it matters.** Our validated result is *2-level* (root + children). A skeptic can call that
"hierarchical," not genuinely "recursive." Demonstrating provenance credit flowing cleanly through
**≥ 2 levels** (root → child → grandchild) closes that gap and is the natural strong form of the
thesis claim. Concurrent work (RAO, Gandhi et al. 2026 — see `wiki/sources/recursive-agent-optimization.md`)
shows end-to-end RL of recursive agents *beyond depth 1* is viable on deep research with **Qwen-3-4B
(our exact base)**: on DeepDive, depth-4 RAO went 0.24 → 0.40 and the model *adapted depth to task
difficulty*. So the direction is de-risked in principle; the open work is the **credit design**.

---

## 1. Does the harness support it? — YES (traced, code-only)

`max_recursion_depth=2` already exists (`dr_rlm_config.py:51`; the config docstring calls depth>1
"the proposal's target"); our smoke jobs just gate it to 1. The mechanism is depth-agnostic:

- **Tree-global ledger.** Minted once at the root and threaded down **by reference** through
  `dict(env_extras)` at every subcall (`dr_rlm_generator.py:181–192`). Because the *same* subcall
  path runs at each level, a **grandchild's** `search()` writes into the **same** ledger, with id
  `{grandchild_rid}-n`. Membership is automatic at all depths — no merge, no re-minting.
- **Owner-encoded ids.** Every snippet id is `{node_rid}-{n}`; `owner_rid_of` decodes the prefix
  regardless of depth (`corpus_search.py:245`). The credit loop iterates **all** nodes by rid.
- **Multi-hop forwarding exists too.** The structured child-return hands the parent
  `{content, citations:[{id,claim}]}` and the prompt instructs it to "preserve provenance by copying
  the relevant citations" — so a grandchild's id *can* ride up two hops into the root report.

**Correction to an earlier claim.** I initially framed the risk as "a grandchild's snippet must
survive two hops to the root's *report citations*." Reading `rer_reward.py:206–232`, the
`ledger_support` credit does **not** require that. A node's credit is judged on **its own cited
evidence vs the report's *rubric criteria*** (`judge.supports(criterion, node_evidence_text[rid])`),
where "own evidence" = `[c for c in n.cited_ids if owner_rid_of(c)==n.rid and c in ledger]`. So a
grandchild that *locally* cites its own supporting evidence earns credit **even if the root never
cites it**. The coupling to the final report is via `s_c` (the report's per-criterion score): the
grandchild only earns credit for a criterion the **report actually covered** (`if s_c<=0: continue`).
That is an *outcome-chain* dependence, not a citation-propagation one — materially milder.

**Net:** turning on depth-2 is a flag flip mechanically. The open questions are (a) whether the
credit *signal* stays strong at depth, and (b) whether the data benefits from depth at all.

---

## 2. Does our data support it? — WEAKLY (flagged)

Structural scan of the 32 training prompts (`data/poc_corpus/train.parquet`):

- **Predominantly explicit depth-1.** Most are a top question + a *flat list of aspects*
  (e.g. #9 EU inputs → {soybean, maize, ethanol, fertilizer}; #4 → {legal, social, ethical,
  professional}). That is **breadth**, served well by a *wide* depth-1 tree (many parallel children).
- **A few have latent depth-2 structure:** #26 (Mastodon SWOT, 19 sub-points → naturally
  {S,W,O,T} → points), #9 (each input → {import share, price, source country}), #21 (5 Foucault
  readings → {proponents, texts, critiques}), #4, #30 (review paper). But the prompt never *forces*
  the second level — the model would have to *choose* to decompose a sub-question.
- **None are DeepDive-style.** DeepDive's value comes from KG-walk "blurry-entity" multi-hop queries
  where depth is *required* to even find the answer (sequential chaining). Ours are
  *parallelizable-breadth*, not *sequential-depth*.

**Implication.** On the current corpus, depth-2 mostly means "a sub-aspect is big enough to warrant
its own sub-team" (recursive breadth) — most of which the wide depth-1 tree already captures. RAO's
own negative result (Appendix A.1: recursion only helps on hard/long-horizon tasks; matches
single-agent on easy ones) applies directly. **To test deep recursion meaningfully we need
depth-requiring data:** DeepDive-style multi-hop QA, or genuinely hierarchical survey/SWOT tasks.
This is a data-acquisition prerequisite, not an afterthought.

---

## 3. The credit design at depth

### 3.1 What we have (root-anchored, node-local evidence)
Current `ledger_support` per-node credit:

```
r_a^evidence(n) = Σ_c  (w_c · s_c / w_den) · ( sup(c, n) / Σ_{n'∈contrib} sup(c, n') )
   sup(c, n) = judge.supports(criterion c, n's OWN cited evidence text)
   s_c       = the ROOT REPORT's score on criterion c   (the anchor to the final reward)
   contrib   = nodes that cited ≥1 of their own ledger snippets
```

This already iterates all nodes, so it is **depth-extensible as written**. Two things degrade at
depth, though:

1. **Outcome-chain dilution.** A deep node earns credit only for criteria the *report* covered
   (`s_c>0`). The more levels between a grandchild and the report, the more ways its good local work
   fails to show up in `s_c` (the child/root didn't synthesize it). Credit becomes chain-dependent.
2. **Support-share dilution.** The denominator `Σ_{n'} sup(c,n')` runs over *all* contributors. As
   the tree widens/deepens, each node's share of a criterion's mass shrinks → sparser per-node
   signal exactly where (deep) we can least afford it.

### 3.2 Three credit options for depth

- **A — Keep root-anchored (conservative).** Children *and* grandchildren stay on `ledger_support`;
  every node's credit traces to the final deliverable. Cleanest, conserved thesis story. **Risk:**
  weak/diluted deep signal (the two dilutions above).

- **B — Recursive-local delegation bonus (RAO-style).** Add to each node a bonus for its *immediate*
  children's contribution — already stubbed as `reward_mode=rao` / `rao_lambda`:
  ```
  r_a(n) = r_a^evidence(n) + λ · mean_{c ∈ children(n)} r_a^evidence(c)
  ```
  This is RAO's Eq. 1 (own-success + λ·mean-child) lifted to *evidence* credit, propagated up the
  tree, so **middle nodes get a dense signal** for spawning useful sub-teams regardless of whether
  their evidence reached the root. Use **mean (a rate), not sum** — RAO's explicit guard against
  spawning useless children to farm the bonus. **Risk:** a node can be "locally useful, globally
  useless" (rewarded for children that didn't help the report).

- **C — Hybrid (recommended).** Root-anchored evidence credit (A) for grounding + a *small*
  delegation bonus (B) to densify the orchestration/middle-node signal + grounding-aware root reward.
  This is the **direct deep generalization of the depth-1 complete-package config** (children
  `ledger_support`, root `full_R` + citation, per-role separation).

### 3.3 The novel framing: *recursive* provenance credit
RAO's deep signal is node **success**; ours is **provenance** (which evidence grounded the answer).
The clean recursive lift is: **anchor each node's evidence to its *parent's* answer-criteria, one
hop at a time, composed up the tree** — a grandchild is credited by whether its evidence supported
the *child's* answer, the child by the *root's* report. This (i) removes the outcome-chain dilution
(each hop is local), (ii) stays a provenance signal (distinct from RAO's success bonus), and (iii) is
a genuinely *recursive* generalization of the validated depth-1 rule (where parent == root). It is
the strongest thesis-facing version and the recommended target for the eventual write-up. It is a
real change to `rer_reward.py` (per-node parent-anchored `s_c` and a per-parent contributor pool),
sketched in §3.4.

### 3.4 Concrete code change (sketch, against current `rer_reward.py`)
- Today the criterion loop uses the global `rubrics` + report `s_list` and a single `contributors`
  pool. For recursive-local credit:
  - Compute, per **non-leaf** node `p`, a local "answer-quality vector" `s_c^(p)` (reuse the rubric
    judge on `p.final_answer` instead of only the root). Leaves keep evidence-only credit.
  - Restrict each node's support-share competition to **siblings under the same parent**
    (`_children(nodes, p.rid)`), so a grandchild competes with its grandchild-siblings, not the whole
    tree — this fixes the support-share dilution.
  - `r_a(n) = Σ_c (w_c · s_c^(parent(n)) / w_den) · (sup(c,n)/Σ_{sib} sup(c,sib))`.
  - Conservation: per-parent the shares sum to the parent's covered mass, so credit telescopes up the
    tree (a clean conserved story, parametrizable toward root-anchored by mixing `s_c^(parent)` with
    the global `s_c`). Cost: one extra rubric-judge pass per non-leaf node (judge load grows with
    internal nodes — bounded, but a real Gemini-cost line item).

---

## 4. The estimator at depth (the weighting comes back)

The depth-cohort **baseline** already handles ≥3 depths (one cohort per (prompt, depth)); roots vs
roots, children vs children, grandchildren vs grandchildren. No change needed there.

The depth inverse-frequency **weighting** I defaulted off (`DR_RLM_DEPTH_WEIGHTING`) is **literally
RAO's Eq. 4–5** (`w_d = α/N_d`), and RAO's ablation (§5, Fig. 8) found it **necessary** for deep
training ("unweighted trains slower and plateaus lower"). It was right to default off for depth-1
(there the populous level is the *children* we want to dominate). At depth ≥2 it comes back, but
**calibrated, not full equalization**:

- Full RAO equalization gives each depth equal total gradient mass — at depth 2 that's root 33% /
  children 33% / grandchildren 33%, which over-protects the rare root.
- Better target: a **profile** that keeps grounding (children+grandchildren) dominant while giving
  the root a real-but-minority share, e.g. root ≈15–20% / children ≈40% / grandchildren ≈40%.
  Implement as a tunable per-depth weight vector (generalize `α/N_d` to `α_d/N_d`) and sweep it; log
  the realized per-depth gradient share (we already compute child/root share in the diagnostic).
- The loud `node_depth`-threading warning (added in the per-role-separation fix) guards the silent
  degrade-to-prompt-baseline failure at any depth.

---

## 5. Experimental design (when picked up)

**Prerequisite:** depth-requiring data (DeepDive subset, or hierarchical survey/SWOT tasks).

### 5.1 Staged ablation sequence (confound control — do NOT combine untested changes)

Both the depth jump *and* the complete-package credit config are **untested**, and depth-2 makes the
depth-1-optimal estimator setting **wrong** (`DR_RLM_DEPTH_WEIGHTING=0` is right only for shallow
trees — at depth-2 the weighting must come back, §4). Combining all of this in one run would stack
three unknowns: a failure couldn't be attributed (credit mechanism? depth? mistuned estimator?), and
a success couldn't be claimed for any one cause. On scarce budget that is the worst outcome — spend
and learn nothing causal. So stage it, changing ~one thing per step and building on a validated base:

| Stage | Change vs prior stage | Held fixed | What it isolates | Status |
|---|---|---|---|---|
| **S0 — depth-1 complete-package** | (baseline) | — | the credit / root-reward mechanism at the *known* depth | committed (`smoke_L3_complete.job`); budget-blocked |
| **S1 — depth-2, same credit rule** | `max_recursion_depth` 1→2 **+ the forced estimator recalibration** (`DR_RLM_DEPTH_WEIGHTING` 0→1, α-tuned, §4) | root-anchored `ledger_support` rule; `full_R`+citation root reward | the **depth effect** — do grandchildren occur? does credit reach them (per-depth-2 mean credit > 0, grandchild orphan% ↓)? does depth help on hard data? | future |
| **S2 — depth-2, recursive-local credit** | credit *rule*: root-anchored → recursive-local (§3.3) | depth=2; estimator settings from S1 | the **credit-rule effect** — does recursive-local densify the deep signal (higher depth-2 credit, lower grandchild orphan% than S1)? | future |

**The one irreducible bundle vs the avoidable confound.** Depth-2 *forces* an estimator change (the
weighting), so "go to depth-2 **with the estimator that depth requires**" is a single conceptual
change — that bundle is unavoidable and fine (S1). What is *separable* and must NOT be bundled in is
the credit-**rule** swap (root-anchored → recursive-local): that gets its own step (S2 vs S1) so we
know which one mattered. Concretely: do **not** flip `max_recursion_depth=2` on the S0 job and *also*
switch to recursive-local credit in the same run.

**Trap to avoid (specific):** running depth-2 on the S0 config *unchanged* (i.e. leaving
`DR_RLM_DEPTH_WEIGHTING=0`) is not "controlled" — it would likely fail for pure estimator reasons
(grandchildren washing out children) that have nothing to do with whether deep credit works, and the
failure would be uninterpretable. S1 must recalibrate the weighting as part of the depth move.

### 5.2 Per-stage configs, metrics, success criteria

**Configs:** `max_recursion_depth=2`; reduce `max_children_per_node` (4→3) to contain blowup;
credit variant per stage (S1 root-anchored, S2 recursive-local §3.3); `DR_RLM_DEPTH_WEIGHTING=1` with
an α-sweep (S1+); checkpointing + judge guard (as in the complete-package job).

**Metrics:** per-depth orphan% and cites/node; **per-depth mean credit** (does credit actually reach
grandchildren?); **depth-usage distribution** (does the model adapt depth to difficulty, like RAO
Fig. 7?); R(alive); compute/wall-clock and judge-call count.

**Ablations (these ARE the staged sequence §5.1):** (1) **S0 vs S1** — depth-2 vs depth-1 on the
*same hard data*, does depth help? (2) the weighting α-sweep *within* S1 (on/off + calibration);
(3) **S1 vs S2** — root-anchored vs recursive-local credit, does recursive-local densify the deep
signal (higher per-depth-2 credit, lower grandchild orphan%)? Run them in S0→S1→S2 order; never fold
(1) and (3) into a single run.

**Success criteria:** depth-2 beats depth-1 on hard data; per-depth-2 mean credit > 0 and grandchild
orphan% drops over training (credit demonstrably reaches grandchildren); model adapts depth to
difficulty.

---

## 6. Cost / risk / sequencing

- **Compute blows up:** depth-2 / branch-4 ⇒ up to 64 leaves/tree (×4 samples = 256 leaf rollouts
  per prompt). Mitigate with branch≤3, child caps, checkpointing. We are already OOM-sensitive at
  depth-1 (micro_batch=1 is the only safe setting).
- **Judge cost multiplies** (more nodes → more support/rubric calls; recursive-local adds a rubric
  pass per internal node). Gemini quota is already the binding constraint.
- **Budget-blocked** (project ~170 SBU). This is prep, not a runnable plan.
- **Sequencing:** land the depth-1 complete-package run first (committed result + clean baseline this
  builds on), *then* deep recursion as the next arc.

---

## 7. Positioning vs RAO (for the thesis)

| | RAO (Gandhi et al. 2026) | This work (deep extension) |
|---|---|---|
| Deep signal | node **success** + λ·mean-child-success | **provenance** (evidence→criteria), recursively |
| Baseline | shared root leave-one-out + depth weighting | **per-depth cohort** baseline + calibrated weighting |
| Domain | TextCraft / Oolong / DeepDive | document-intensive deep research |
| Depth shown | 4–6 (Qwen-3-4B) | target ≥2, same base model |

RAO is the closest prior art and the existence proof that deep recursion trains on our base model and
domain. **Our differentiation is the credit signal:** *recursive provenance credit* — attributing the
report's grounded quality to the specific (possibly deep) sub-agent whose retrieval made it possible —
is novel relative to RAO's success-based delegation bonus, and the depth extension is where that
recursion becomes more than 2-level.

See also: `RL_TRAINING_LOG.md` (per-role-separation fix — the estimator this builds on),
`PROVENANCE_VALIDATION_SUMMARY.md` (the depth-1 result), `wiki/sources/recursive-agent-optimization.md`,
`wiki/sources/deepdive.md`.
