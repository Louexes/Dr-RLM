#!/usr/bin/env python3
"""Cost-vs-quality Pareto frontier: flat (DR-Tulu) vs recursive (DR-RLM), across training.

One figure for the "recursion wins quality, at what cost --- and does training pay the cost
down?" argument (results ch., Efficiency Frontier). x = compute cost per query (total tree
tokens, log scale); y = avg report quality over the three long-form benchmarks. Each arm is a
trajectory Untrained -> SFT -> RL; the hypothesis is that RL moves the recursive point DOWN
(cheaper) and UP (better) --- a frontier that improves with learned delegation.

Numbers come from `analysis/behavioral_metrics.py --paired` (cost, macro-averaged over
ResearchQA / DeepResearch Bench / ScholarQA-CS, paired over shared example_ids) and the
results quality tables (avg). RL cells are pending held-out eval of the final checkpoints ---
set them below and re-run; the figure auto-completes the arm.

Run:  uv run --with matplotlib python analysis/plot_cost_quality_pareto.py
Out:  analysis/figures/fig-cost-quality-pareto.{pdf,png}
"""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- measured points: (tokens_per_query, avg_quality). RL = None until the eval lands. -------
# cost   = behavioral_metrics.py --paired, "total tokens (tree)", macro-avg over 3 long-form benches
# quality= results tab:progression "Average (3)" per stage
POINTS = {
    "Flat (DR-Tulu)": {
        "Untrained": (6800.0, 0.278),
        "SFT":       (6200.0, 0.245),
        "RL":        None,               # pending flat RL checkpoint
    },
    "Recursive (DR-RLM)": {
        "Untrained": (165000.0, 0.349),
        "SFT":       (161400.0, 0.338),
        "RL":        None,               # pending: recompute over 3-bench avg (RQA RL=0.404 only so far)
    },
}
STAGES = ["Untrained", "SFT", "RL"]
COLORS = {"Flat (DR-Tulu)": "#0072B2", "Recursive (DR-RLM)": "#D55E00"}
MARKERS = {"Untrained": "o", "SFT": "s", "RL": "*"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for arm, stages in POINTS.items():
        c = COLORS[arm]
        pts = [(s, stages[s]) for s in STAGES if stages.get(s) is not None]
        xs = [p[1][0] for p in pts]
        ys = [p[1][1] for p in pts]
        # trajectory line + directional arrows between consecutive measured stages
        for i in range(len(pts) - 1):
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.3, alpha=0.8))
        for stage, (x, y) in pts:
            ax.scatter([x], [y], s=90 if stage != "RL" else 200, marker=MARKERS[stage],
                       color=c, edgecolor="white", linewidth=0.6, zorder=5,
                       label=f"{arm}" if stage == "Untrained" else None)
            ax.annotate(stage, (x, y), textcoords="offset points", xytext=(6, 5),
                        fontsize=7, color=c)
        # hypothesized RL direction if RL not yet measured (recursive arm carries the claim)
        if stages.get("RL") is None and pts:
            lx, ly = xs[-1], ys[-1]
            ax.annotate("RL (pending)", (lx, ly), textcoords="offset points", xytext=(6, -12),
                        fontsize=7, color=c, alpha=0.6)
            ax.annotate("", xy=(lx * 0.72, ly + 0.03), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.0, ls=":", alpha=0.5))

    ax.set_xscale("log")
    ax.set_xlabel("compute cost  (total tree tokens / query, log scale)")
    ax.set_ylabel("avg. report quality  (3 long-form benchmarks)")
    ax.set_title("Cost--quality frontier across training")
    ax.grid(True, which="both", axis="both", alpha=0.15)
    # "better" corner annotation (up-left = cheaper + higher quality)
    ax.annotate("better", xy=(0.04, 0.96), xycoords="axes fraction", fontsize=8,
                style="italic", color="#444", ha="left", va="top")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()

    outdir = Path(__file__).resolve().parent / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig-cost-quality-pareto.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {outdir}/fig-cost-quality-pareto.pdf/.png")


def _selfcheck():
    # recursion must be ~orders more tokens than flat, and every measured point well-formed
    for arm, stages in POINTS.items():
        for s, v in stages.items():
            assert v is None or (v[0] > 0 and 0.0 <= v[1] <= 1.0), f"bad point {arm}/{s}: {v}"
    fu = POINTS["Flat (DR-Tulu)"]["Untrained"][0]
    ru = POINTS["Recursive (DR-RLM)"]["Untrained"][0]
    assert ru / fu > 10, "expected recursion >10x flat token cost"
    assert math.isclose(POINTS["Recursive (DR-RLM)"]["Untrained"][1], 0.349)


if __name__ == "__main__":
    _selfcheck()
    main()
