# Paper assets — "training improves Dr-RLM's raw outputs"

> **Status: PLACEHOLDER (2026-06-30).** Built from the **v1-clean provenance-smoke RL run**
> (`runs/provenance_smoke/credit_metrics_L3_v1clean.jsonl`) — relative numbers on the frozen
> smoke corpus, *not* absolute scores. **These are scaffolds for the final paper figures/table.
> Regenerate from the FINAL SFT+RL checkpoint's `credit_metrics` jsonl before submission — the
> tooling is run-agnostic (see "Regenerate" below); only the input file changes.**

## Why this exists (supervisor request)

Literally show how the *raw model outputs* improve over training — **the primary ask is the verbatim
text snippets (#1): the reader sees the actual generated text change.** The rest are support.

0. **★★ Consolidated full-trajectory figure** (`fig-trajectory-full.tex`, full-width `figure*`) — the
   one to put in the paper. Features **instance 24** (construction product from air-filter waste),
   chosen for the **clearest visible before/after**: one coherent two-column flow — user query → root
   distills the sprawling ~15-question brief into 4 focused sub-questions → sub-agents (BEFORE | AFTER)
   → root final report (BEFORE | AFTER). Raw model text at each stage, `<cite id>` spans highlighted.
   BEFORE: *every* sub-agent returns **(empty)** so the root fabricates an ungrounded "Aero-Skin"
   product with **0 citations**; AFTER: the sub-agents ground their findings and the root produces a
   **cited** feasibility report. Left column has no orange (empty + uncited), right has grounded content
   + citations — "after is better" at a glance. Also illustrates the grounding-blind judge ($R$ only
   0.80→0.90 despite hallucination→grounded). **Two distinct improvements, no citation hand-off:** the
   sub-agents' own-evidence cite counts (from the ledger) go `0/0/0/0 → 3/3/1/5`, shown as right-aligned
   count chips on each sub-agent line (their final summaries drop the `<cite>` tags, so we show counts
   not inline tags); and *separately* the root's report citations trace to owner `d42178f8`=the **root's
   own** retrieval (verified) — so the figure explicitly does *not* claim the report cites the
   sub-agents. Final-checkpoint version (`generate.py --log-dir`) will show inline child cites.
   **Known trade-off (see chat 2026-06-30):** no smoke instance shows BOTH a clear visible improvement
   AND the root *inventing* its decomposition from an open question — visible-jump instances (24, 27)
   have pre-listed sub-questions; open-question instances (4, 18) improve only subtly. inst 24 distills
   a messy brief (a real selection step) but doesn't invent from scratch. The final trained-checkpoint
   figure should use an open query that shows both. Root sub-question wording reconstructed (RL logs
   dump only final answers; final version gets verbatim delegation via `generate.py --log-dir`).
   NB the single-stage snippet figures + table + tree below still use **instance 27**.
1. **★ Single-stage before/after snippets** (the building blocks of #0, if you want a smaller figure):
   VERBATIM model text, first vs. last rollout, `<cite id>` spans highlighted, at two levels:
   `fig-trajectory-snippets.tex` (**sub-agent** output) and `fig-trajectory-snippets-root.tex`
   (**root report** — ungrounded draft names the wrong six elements; grounded one cites CHNOPS).
   `headline_example_raw_inst27.md` has the longer raw text.
2. **Multi-query before/after table** (`tbl-training-delta.tex` / `training_delta_table.md`) —
   several queries, start-of-training vs. end-of-training, on the metrics that move.
3. **Headline TikZ figure** (`fig-training-delta.tex`) — one instance, schematic of the whole tree
   going from orphan sub-agents → grounded sub-agents at no quality cost (the at-a-glance summary).

## What the data shows (honest framing)

The observable improvement is **grounding behavior**, which is exactly what provenance credit
trains — *not* a report-quality jump:

| signal | start → end (mean over 32 queries) | meaning |
|---|---|---|
| orphan sub-agents % | **91% → 50%** | sub-agents stop surfacing evidence they never cite |
| own-evidence cites / sub-agent | **0.20 → 1.43** (~7×) | each sub-agent grounds its findings |
| report reward R (alive) | **0.69 → 0.81** | quality holds (stable-to-slightly-up) |

**Caveats to preserve in the final version:**
- The clean signal is the **child/sub-agent grounding** (orphan→grounded). `total_report_cites`
  (root `<cite>` tags) is noisier — under `credit_share` the root is barely trained. In the
  headline instance the 6 "cited claims" are the root grounding **its own** retrieved evidence,
  not provenance edges to children (verified) — the figure is drawn faithfully to that.
- ~43% of grounded sub-agents record the citation in the harness *ledger* but summarize
  ("Based on the retrieved document…") in their final answer. A *fully-visible* raw-text trajectory
  benefits from the full REPL rollout (router logs / `generate.py --log-dir` + `agent/viz_trajectory.py`),
  not just the final-answer dump. Instance 27 is one where it is fully visible — that's why it's the headline.
- The per-query table rows are illustrative; `orphan%` and `own-evidence cites` improve on **all 32**
  queries, so they are representative, not cherry-picked. For the final version consider a held-out /
  random query sample to pre-empt the cherry-pick critique.

## Files

| file | what | goes where in thesis |
|---|---|---|
| `fig-trajectory-full.tex` | **★★ consolidated full-width trajectory** (query→decompose→sub-agents→report, before/after) | `\input{sections/fig-trajectory-full}` in Results (`figure*`) |
| `fig-trajectory-full_PREVIEW.png` | rendered preview of the above | (preview only) |
| `fig-trajectory-snippets.tex` | verbatim before/after raw text (sub-agent), cites highlighted | `\input{sections/fig-trajectory-snippets}` in Results |
| `fig-trajectory-snippets-root.tex` | **★ verbatim before/after raw text (root report)**, cites highlighted | `\input{sections/fig-trajectory-snippets-root}` in Results |
| `fig-trajectory-snippets*_PREVIEW.png` | rendered previews of the above | (preview only) |
| `fig-training-delta.tex` | headline TikZ (before/after tree) | `\input{sections/fig-training-delta}` in Results |
| `fig-training-delta_PREVIEW.png` | rendered preview of the above | (preview only, not for thesis) |
| `tbl-training-delta.tex` | booktabs multi-query before/after table | `\input` in Results |
| `training_delta_table.md` | same table, markdown (read now) | — |
| `headline_example.json` | the inst-27 per-node before/after numbers | source of truth for the figure |
| `headline_example_raw_inst27.md` | raw report + sub-agent text | appendix qualitative-example box |

Both `.tex` files assume `\usepackage{tikz}` + `\usetikzlibrary{arrows.meta}` (figure) and
`\usepackage{booktabs}` (table) — already in `msc_thesis.tex` — and the thesis palette
(`cI/cII/cIII/cBad`). Build target is **Overleaf** (this HPC node lacks the thesis body fonts);
the figure was syntax-validated locally with the `article` class.

## Regenerate (placeholder → final checkpoint)

```bash
python analysis/training_delta_examples.py \
    --metrics runs/<FINAL_RUN>/credit_metrics_<final>.jsonl \
    --prompts data/<final_eval_or_train>.parquet \
    --headline <instance_id> --k 3 \
    --table-instances <id1,id2,...> \
    --outdir docs/paper_assets
```

Then update `fig-training-delta.tex` from the new `headline_example.json` (per-node counts) and
drop the "PLACEHOLDER / v1-clean" wording in the captions. The figure is small/explicit by design
so re-keying the 8 numbers by hand is trivial; if it's worth automating, have the generator emit
the figure too.

For the **raw-trajectory snippets** (`fig-trajectory-snippets.tex`), pull fresh verbatim text:

```bash
python analysis/training_delta_examples.py \
    --metrics runs/<FINAL_RUN>/credit_metrics_<final>.jsonl \
    --dump-snippets <instance_id> --k 3
```

This prints the early orphan sub-agent text (BEFORE) and the late grounded sub-agents' verbatim
`<cite>...</cite>` spans (AFTER). Paste the strongest into the `\trajbox{}` bodies, normalize
whitespace, mark elisions with `[...]`, and escape LaTeX specials (`& % _ # $`). Keep them VERBATIM
otherwise — the whole point is that the reader is seeing real model output.
