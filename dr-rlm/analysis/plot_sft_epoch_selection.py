#!/usr/bin/env python3
"""Combined SFT low-LR sweep figure: how BOTH RL initializations were chosen.

One figure, two panels, both arms overlaid (recursive DR-RLM, flat DR-Tulu):
  (a) predictive entropy (% of base) vs SFT epochs, with the RL-workable band;
  (b) out-of-domain quality relative to each arm's untrained baseline vs epochs.

The chosen init for each arm is the earliest epoch that clears its untrained
baseline on quality WHILE staying in the workable entropy band -- the two axes
have to be balanced, and later (higher-quality) epochs get entropy-vetoed. The
recursive arm balances at two epochs, the flat arm one epoch earlier because its
smaller cold-start set over-narrows sooner.

All numbers are measured sweep scalars (seed-42), verbatim from:
  runs/sft_sweep/B_lowlr/e*/race/.../race_result.txt   (recursive DRB-24 RACE)
  runs/sft_sweep/.../behavioral.txt                     (recursion %)
  docs/EXPERIMENT_LOG.md 2026-07-11 (recursive) and 2026-07-12 (flat)

Run:  uv run --with matplotlib python analysis/plot_sft_epoch_selection.py
Out:  ../thesis_obsidian/thesis_final/figures/fig-sft-epoch-selection.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
    })


C_REC  = "#1b8a6b"   # recursive DR-RLM
C_FLAT = "#3b6ea5"   # flat DR-Tulu
C_SEL  = "#e34948"   # selected-checkpoint ring
C_BASE = "#2a2a2a"
C_GOOD = "#1b8a6b"   # workable-entropy band
C_BAD  = "#b03a2e"   # collapse band

WORKABLE = 65        # entropy % of base: >= is RL-workable; < ~55 collapses

OUT = Path(__file__).resolve().parents[2] / "thesis_obsidian/thesis_final/figures"

# --- low-LR sweep scalars, both arms -----------------------------------------
# recursive (DR-RLM): the gentler-LR recipe swept over 2-4 epochs
REC_EP   = [2, 3, 4]
REC_ENT  = [69, 55, 46]                 # entropy % of base
REC_DRB  = [0.335, 0.344, 0.336]        # out-of-domain DRB-24 RACE coverage
REC_BASE = 0.312                        # untrained recursive DRB-24
REC_SEL  = 2                            # chosen: 2 epochs

# flat (DR-Tulu): the gentler-LR recipe swept over 1-4 epochs
FLAT_EP   = [1, 2, 3, 4]
FLAT_ENT  = [89, 75, 48, 48]            # entropy % of base
FLAT_COV_EP = [1, 2]                    # only 1+2 epochs cleared entropy to be gated
FLAT_COV  = [0.295, 0.232]             # ResearchQA judge coverage
FLAT_BASE = 0.268                       # untrained flat coverage
FLAT_SEL  = 1                           # chosen: 1 epoch

REC_DQ  = [q - REC_BASE for q in REC_DRB]                    # quality above baseline
FLAT_DQ = [q - FLAT_BASE for q in FLAT_COV]

# sanity: chosen point clears baseline AND stays workable; later epochs do not (entropy or quality)
assert REC_ENT[0] >= WORKABLE and REC_DRB[0] > REC_BASE, "recursive 2ep must pass both"
assert REC_ENT[1] < WORKABLE, "recursive 3ep (higher DRB) must be entropy-vetoed"
assert FLAT_ENT[0] >= WORKABLE and FLAT_COV[0] > FLAT_BASE, "flat 1ep must pass both"
assert FLAT_COV[1] < FLAT_BASE, "flat 2ep must fail the quality gate"


def fig_selection():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    # (a) predictive entropy vs epochs, both arms
    a.axhspan(WORKABLE, 108, color=C_GOOD, alpha=0.06, zorder=0)
    a.axhspan(0, 55, color=C_BAD, alpha=0.05, zorder=0)
    a.axhline(100, color=C_BASE, ls=":", lw=1, alpha=0.6, zorder=1)
    a.axhline(WORKABLE, color=C_GOOD, ls=":", lw=1, alpha=0.7, zorder=1)
    a.annotate("untrained base", xy=(4, 100), xytext=(0, 2), textcoords="offset points",
               ha="right", va="bottom", fontsize=7, color=C_BASE)
    a.annotate("workable floor", xy=(1, WORKABLE), xytext=(0, 3), textcoords="offset points",
               ha="left", va="bottom", fontsize=7, color=C_GOOD)
    a.plot(REC_EP, REC_ENT, "-o", color=C_REC, ms=5, zorder=3, label="DR-RLM (recursive)")
    a.plot(FLAT_EP, FLAT_ENT, "-s", color=C_FLAT, ms=5, zorder=3, label="DR-Tulu (flat)")
    a.scatter([REC_SEL], [REC_ENT[0]], s=220, facecolor="none", edgecolor=C_SEL,
              linewidth=1.8, zorder=4)
    a.scatter([FLAT_SEL], [FLAT_ENT[0]], s=220, facecolor="none", edgecolor=C_SEL,
              linewidth=1.8, zorder=4)
    a.annotate("collapse", xy=(3.5, 48), xytext=(0, 5), textcoords="offset points",
               ha="center", fontsize=7, color=C_BAD)
    a.set_title("(a) Predictive entropy")
    a.set_ylabel("entropy (% of base)")
    a.set_xlabel("SFT epochs")
    a.set_xticks(FLAT_EP)
    a.set_ylim(0, 108)
    a.legend(loc="lower left", handletextpad=0.4, borderpad=0.3)

    # (b) out-of-domain quality above each arm's untrained baseline
    b.axhline(0, color=C_BASE, ls="--", lw=1, zorder=1)
    b.annotate("untrained baseline (pass line)", xy=(1, 0), xytext=(0, -4),
               textcoords="offset points", ha="left", va="top", fontsize=7, color=C_BASE)
    b.plot(REC_EP, REC_DQ, "-o", color=C_REC, ms=5, zorder=3, label="DR-RLM (recursive)")
    b.plot(FLAT_COV_EP, FLAT_DQ, "-s", color=C_FLAT, ms=5, zorder=3, label="DR-Tulu (flat)")
    b.scatter([REC_SEL], [REC_DQ[0]], s=220, facecolor="none", edgecolor=C_SEL,
              linewidth=1.8, zorder=4)
    b.scatter([FLAT_SEL], [FLAT_DQ[0]], s=220, facecolor="none", edgecolor=C_SEL,
              linewidth=1.8, zorder=4)
    b.annotate("higher quality,\nbut entropy-vetoed", xy=(3, REC_DQ[1]), xytext=(4, 2),
               textcoords="offset points", fontsize=6.8, color=C_BAD, va="bottom")
    b.set_title("(b) Out-of-domain quality")
    b.set_ylabel("quality above untrained baseline")
    b.set_xlabel("SFT epochs")
    b.set_xticks(FLAT_EP)
    b.set_ylim(-0.06, 0.05)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig-sft-epoch-selection.{ext}")
    plt.close(fig)
    print("wrote fig-sft-epoch-selection.{pdf,png}")


if __name__ == "__main__":
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_selection()
    print(f"-> {OUT}")
