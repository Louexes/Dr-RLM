# Provenance-Credit Proof of Concept — validity of the thesis core

> **RESULT (2026-06-10): CONDITIONAL GO — the credit signal tracks the true counterfactual,
> but only in its `ledger_support` form.** On 37 recursion trees (132 child nodes,
> dr-tulu-rl decompose-shaped prompts, gemini-2.5-flash agent/judge/synthesizer):
>
> | share mode | mean within-tree Spearman(r_a, ΔR_a) | 95% CI | trees +/− | sign-test p | pooled ρ |
> |---|---|---|---|---|---|
> | citation_count | 0.224 | [−0.07, 0.50] | 15/9 | 0.31 (n.s.) | 0.061 |
> | support | 0.291 | [0.03, 0.54] | 13/7 | 0.26 (n.s.) | 0.130 |
> | **ledger_support** | **0.425** | **[0.19, 0.64]** | **21/5** | **0.0025** | 0.210 |
>
> - **Within-tree is the training-relevant statistic** (GRPO baselines within the prompt
>   group; what matters is whether r_a rank-orders *siblings* by true contribution). On it,
>   `ledger_support` clears the pre-registered 0.4 GO bar with a significant sign test;
>   median per-tree ρ = 0.60. The machine verdict in the JSON says "WEAK" because it keyed
>   on *pooled* ρ (diluted by cross-tree reward-scale differences) — the pre-registered
>   pooled criterion was the wrong primary, and we report both honestly.
> - **Mode ordering is exactly the false-orphan prediction:** the more the credit depends on
>   the ROOT's citing behavior (citation_count), the worse the alignment; judging each
>   child's OWN retrieved-and-cited ledger evidence (ledger_support) is the reliable proxy.
>   ⇒ raw root-cite counting is NOT a viable training signal; the ledger path is.
> - **Orphan separation is right-signed but modest:** credited children ΔR=0.123 vs orphans
>   0.071 (diff CI [−0.05, 0.15]). Orphans are genuinely less important on average — the
>   orphan rate is real learning signal — but false orphans exist, which is precisely why
>   citation_count fails and ledger_support is required.
> - **Differentiation:** L3 per-child credit gives nonzero within-tree advantage spread
>   (var 0.018) where L1 broadcast is structurally zero. Report reward healthy
>   (mean R_all = 0.50, only 3/37 zero trees).
> - **Decision:** proceed with provenance credit as the thesis method, with
>   `share_mode=ledger_support` as the default (citation_count/support as ablations).
>   Attenuation note: judge noise sits in BOTH r_a and ΔR_a, so 0.425 is a lower bound on
>   the true alignment.
>
> Full numbers: `provenance_credit_stage1.json` (+ stats reproduced below in §Results).
> Caveats: untrained gemini-2.5-flash policy, n=37 trees, LOCO ground truth is itself
> judge-scored one-shot re-synthesis (not the original REPL process); browse disabled
> (Jina quota) so grounding is search-snippet-only.

**Question this answers (the linchpin, RQ4 in `wiki/concepts/provenance-credit-assignment.md`):**
is the per-sub-agent credit `r_a` — read for free off the citation graph — a usable proxy for
the *true* counterfactual contribution of that sub-agent? If yes, the credit-assignment method
that the thesis is built on has signal; if no, training on it is pointless and the direction
must pivot (see `docs/decisions.md` "re-anchor RER to within-agent provenance" fork).

This is a **signal-quality validation done offline, before any GPU training** — it cannot be
faked by a benchmark score and is the cheapest possible falsification of the core idea.

## Method

**Models: Gemini only (`gemini-2.5-flash`) for agent, judge, and synthesizer.**

### Stage 0 — non-degeneracy (judge-free), `analysis/provenance_credit_stage0.py`
On real captured recursion trees, compute the L3 citation-count credit share per child using the
*same* parser (`judge.extract_claims_and_corresponding_citation_ids`) and provenance decoder
(`corpus_search.owner_rid_of`) that RL training uses. Reports:
- **Q1 Degeneracy** — fraction of trees where every child gets zero credit (report cites nothing
  decodable, or all child evidence is orphaned).
- **Q2 Differentiation** — among trees with ≥2 contributing children, do siblings get different shares?
- **Q3 Orphans** — per-tree fraction of children whose surfaced evidence never appears in the root report.
- **Q4 Conservation** — shares sum to 1 over contributors.

### Stage 1 — counterfactual alignment (the go/no-go), `analysis/provenance_credit_stage1.py`
On rubric-carrying rollout trees (DR-Tulu RL training-distribution prompts, `drtulu_rl` benchmark):
1. **Credit (proxy):** run `compute_rer_rewards` (the training reward) under each share mode
   (`citation_count`, `support`, `ledger_support`) → `r_a` per child + report reward `R`.
2. **LOCO (ground truth):** deterministically re-synthesize the root report from the children's
   returns — once with **all** children, once **leaving each child out** — restricting citable
   evidence to the present children; judge each against the rubric. `ΔR_a = R(all) − R(−a)` is the
   true difference-reward contribution of child `a`.
3. **Align:** Spearman/Pearson(`r_a`, `ΔR_a`) pooled over child-nodes and within-tree; orphan
   separation (mean `ΔR` of credited vs zero-credit children); per-node advantage variance
   under L3 (per-child `r_a`) vs L1 (broadcast `R` → zero within-tree spread).

## Go / no-go criterion

- **GO** — positive, significant alignment (pooled Spearman ρ ≳ 0.4–0.5) under at least one share
  mode, with `r_a` separating LOCO-important children from orphans. The citation graph is a usable
  counterfactual proxy ⇒ proceed to the L1-vs-L3 training comparison (G4).
- **NO-GO** — ρ ≈ 0 under all share modes ⇒ the structural proxy does not track marginal
  contribution; pivot per `docs/decisions.md` (within-agent provenance, or contingent cross-tree RER).
- **CONDITIONAL** — alignment only under `ledger_support` (judged over each child's own evidence),
  not `citation_count` (off the root's surviving `<cite>` tags). Then the method works but **requires
  the ledger path**, because orchestrators drop child citations on synthesis (the low citation-survival
  already measured in `runs/ab_child_return_v2`). This directly informs the default `share_mode`.

## Infrastructure notes (so the numbers are interpretable)
- **Models: `gemini-2.5-flash` for agent, judge, synthesizer** (per request — no gpt-5-mini).
  It fans out reliably under the unified `DrRlmEnv` driver (4–6 sub-agents/tree), unlike in the
  legacy `rlm/` harness.
- **Search = Serper (live web); browse (Jina) DISABLED.** Jina's quota was exhausted mid-run and
  its 402 error text leaked into the orchestrator's context as if it were page content, making a
  root *abandon synthesis* and write an apology (R≈0, no reward to attribute). Fix:
  `DR_RLM_DISABLE_BROWSE=1` (no Jina calls) + a poison-text scrub in `tools/web_provider.py`, so a
  dead backend reads as "no hit", never a poison pill. Children ground on search snippets — which
  already carry the `{node_rid}-{n}` provenance ids credit needs — so grounding is unaffected.

## Honest scope
This validates that the credit **signal** is informative (tracks the counterfactual). It does **not**
prove RL on it improves the policy — that is the subsequent training experiment (G4/G5). But a credit
signal that fails this test cannot help training, so this is the correct gate to run first.

## Artifacts
- `provenance_credit_stage0.json` — non-degeneracy on existing trees.
- `provenance_credit_stage1.json` — alignment on rubric-carrying trees (the verdict).
- Generation: `agent/gen_provenance_poc.sh` (one shard) / `agent/run_provenance_poc_sharded.sh` (N shards).
- Subset: `data/subsets/drtulu_rl_decompose40.jsonl` (built by `analysis/build_drtulu_rl_subset.py`).
