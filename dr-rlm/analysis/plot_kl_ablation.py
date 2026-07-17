#!/usr/bin/env python3
"""KL anchor-strength ablation figure: splice the coef=0.01 phase (steps 1-50
of rl_full150) with the coef=0.05 resume (rl_klprobe, global steps 51-95).

Reuses plot_training_curves.py's loading/metric/styling helpers so the figure
is visually consistent with the other thesis training-curve figures (same
palette, same window=15 rolling mean, no block overlays beyond the splice
marker this ablation specifically calls for).

Usage:
  python analysis/plot_kl_ablation.py \
      --phase1 runs/provenance_smoke/credit_metrics_rl_full150.jsonl \
      --phase2 runs/provenance_smoke/credit_metrics_rl_klprobe.jsonl \
      --out analysis/figures/kl_ablation --window 15
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_training_curves as base
from plot_training_curves import (
    C_LENGTH, C_R_FIN, C_CITES, C_RAW, SFT_INIT_LEN,
    load_metrics, bucket_by_step, rolling_mean, _style, TREES_PER_STEP,
)

PHASE1_STEPS = 50  # rl_full150 steps 1-50 kept; steps 51-53 dropped (re-rolled by the probe)
C_PHASE1 = C_LENGTH   # reuse existing hues per metric; phase distinguished by shading, not color
SPLICE_COLOR = "#6b6b63"


def splice(phase1_path: str, phase2_path: str):
    """Load both phases, trim phase1 to its first 50 steps, and return the
    per-step stats list plus the step index (1-based) at which coef changed."""
    rows1 = load_metrics(phase1_path)
    rows2 = load_metrics(phase2_path)

    stats1 = bucket_by_step(rows1)
    assert len(stats1) == 53, f"expected 53 steps in phase1, got {len(stats1)}"
    stats1 = stats1[:PHASE1_STEPS]  # drop steps 51-53 (re-rolled by the probe's resume)

    stats2 = bucket_by_step(rows2)
    assert len(stats2) == 45, f"expected 45 steps in phase2 (klprobe), got {len(stats2)}"

    stats = stats1 + stats2
    steps = list(range(1, len(stats) + 1))
    return steps, stats, len(stats1)  # splice index = 50


def _smoothed_per_phase(vals: List, splice_idx: int, window: int):
    """Roll each phase separately so the discontinuity at the splice is honest
    (no smoothing across the coef change)."""
    left = rolling_mean(vals[:splice_idx], window)
    right = rolling_mean(vals[splice_idx:], window)
    import numpy as np
    return np.concatenate([left, right])


def _panel(ax, steps, vals, splice_idx, color, ylabel, window, label=None):
    raw = [v for v in vals]
    ax.plot(steps, raw, color=C_RAW, linewidth=0.8, alpha=0.6, zorder=1)
    smoothed = _smoothed_per_phase(vals, splice_idx, window)
    ax.plot(steps, smoothed, color=color, linewidth=1.6, zorder=2, label=label)
    ax.set_ylabel(ylabel)
    # phase shading: coef=0.01 (left) vs coef=0.05 (right), disclosed in caption
    ax.axvspan(steps[0] - 0.5, steps[splice_idx - 1] + 0.5, color=C_RAW, alpha=0.08, zorder=0)
    ax.axvline(steps[splice_idx - 1] + 0.5, color=SPLICE_COLOR, linestyle="--", linewidth=1.1, zorder=3)


def fig_kl_ablation(steps, stats, splice_idx, out_dir: Path, window: int):
    fig, axes = plt.subplots(3, 1, figsize=(5, 7.2), sharex=True)

    length = [s["median_len"] for s in stats]
    r_fin = [s["r_finalized"] for s in stats]
    cites = [s["cites_per_child"] for s in stats]

    _panel(axes[0], steps, length, splice_idx, C_PHASE1, "median report length (chars)", window,
           label="median report length")
    axes[0].axhline(SFT_INIT_LEN, color=C_RAW, linestyle=":", linewidth=1.2,
                     label="SFT init (~4.9k chars)")
    axes[0].legend(loc="upper right", fontsize=7)

    _panel(axes[1], steps, r_fin, splice_idx, C_R_FIN, "R(finalized)", window)

    _panel(axes[2], steps, cites, splice_idx, C_CITES, "citations per child", window)
    axes[2].set_xlabel("training step")

    # single splice annotation on the top panel only
    axes[0].annotate(
        "KL 0.01 → 0.05",
        xy=(splice_idx + 0.5, axes[0].get_ylim()[1]),
        xytext=(splice_idx + 1.5, axes[0].get_ylim()[1]),
        fontsize=7.5, va="top", ha="left", color=SPLICE_COLOR,
    )

    fig.tight_layout()
    fig.savefig(str(out_dir / "fig_kl_ablation") + ".pdf")
    fig.savefig(str(out_dir / "fig_kl_ablation") + ".png")
    plt.close(fig)


def thirds(vals: List, n_parts: int = 3):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    part = max(1, n // n_parts)
    chunks = [vals[i * part:(i + 1) * part] for i in range(n_parts)]
    chunks[-1] = vals[(n_parts - 1) * part:]
    return [st.mean(c) if c else None for c in chunks]


def phase_prose_stats(rows: List[dict], label: str, n_parts: int = 3, part_name: str = "thirds"):
    """Per-phase first/last-part stats for the prose, computed directly over
    trees in file order (matches base.step_stats definitions)."""
    n = len(rows)
    part = max(1, n // n_parts)
    chunks = [rows[i * part:(i + 1) * part] for i in range(n_parts)]
    chunks[-1] = rows[(n_parts - 1) * part:]

    print(f"\n[{label}] {n} trees, {n_parts} {part_name} of ~{part} trees each")
    for i, chunk in enumerate(chunks):
        s = base.step_stats(chunk)
        print(
            f"  {part_name[:-1]} {i+1}: medlen={s['median_len']:.0f}  "
            f"R(fin)={s['r_finalized']:.3f}  strict0={100*s['strict_zero_pct']:.0f}%  "
            f"cites/kid={s['cites_per_child']:.2f}  orphan={100*s['orphan_pct']:.0f}%  "
            f"E[R]={s['e_r']:.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", required=True, help="rl_full150 metrics (coef=0.01)")
    ap.add_argument("--phase2", required=True, help="rl_klprobe metrics (coef=0.05, resumed step 50)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=15)
    args = ap.parse_args()

    _style()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, stats, splice_idx = splice(args.phase1, args.phase2)
    print(f"[splice] phase1 (coef=0.01) steps 1-{splice_idx}, "
          f"phase2 (coef=0.05) steps {splice_idx+1}-{len(steps)} "
          f"(global steps 51-{50+len(stats)-splice_idx})")

    fig_kl_ablation(steps, stats, splice_idx, out_dir, args.window)

    # prose stats, computed on raw trees per phase (not on the bucketed/rolled stats)
    rows1_full = load_metrics(args.phase1)
    rows1 = rows1_full[: splice_idx * TREES_PER_STEP]
    rows2 = load_metrics(args.phase2)

    phase_prose_stats(rows1, "phase1 coef=0.01 (steps 1-50)", n_parts=3, part_name="thirds")
    phase_prose_stats(rows2, "phase2 coef=0.05 (global steps 51-95)", n_parts=4, part_name="quarters")

    readme = README_TEMPLATE.format(
        phase1=args.phase1, phase2=args.phase2,
        n1=len(rows1), n2=len(rows2), splice_idx=splice_idx,
        splice_idx_p1=splice_idx + 1,
        n_total=len(stats), n_total_global=50 + (len(stats) - splice_idx),
        window=args.window,
    )
    (out_dir / "README.md").write_text(readme)
    print(f"[done] figure + README written to {out_dir}")


README_TEMPLATE = """# KL anchor-strength ablation figure

Generated by `analysis/plot_kl_ablation.py`, splicing two lineages of the
SAME training run:

- `{phase1}` (rl_full150, KL-to-init coefficient 0.01) -- steps 1-{splice_idx}
  kept ({n1} trees); steps 51-53 dropped (the probe re-rolled from the
  step-50 checkpoint, so steps 51+ exist in both lineages and the probe's
  continuation is the one analyzed).
- `{phase2}` (rl_klprobe, coefficient 0.05, resumed from the step-50
  checkpoint -- the only change) -- plotted as steps {splice_idx_p1}-{n_total}
  (global steps 51-{n_total_global}), {n2} trees.

Both files use the same per-tree telemetry schema as
`plot_training_curves.py` (16 trees/step) and are deduplicated on
(instance_id, repetition_id) with the same keep-last rule before bucketing.

## `fig_kl_ablation.{{pdf,png}}`

Three vertically stacked panels sharing the step axis (1-{n_total}):
(a) median finalized-report length (chars), with the ~4,900-char SFT-init
    dotted reference line;
(b) R(finalized) -- mean report reward over finalized trees only;
(c) citations per child node.

Each panel shows the faint raw per-step series plus a centered rolling mean
(window={window}), smoothed SEPARATELY within each phase -- the mean never
rolls across the splice, so the visible discontinuity at step {splice_idx}
is real, not a smoothing artifact. A vertical dashed line at step {splice_idx}
marks the coefficient change ("KL 0.01 -> 0.05"); a faint background tint
distinguishes the coef=0.01 region (steps 1-{splice_idx}) from the coef=0.05
region (steps {splice_idx_p1}-{n_total}).

Suggested caption: "Spliced training-dynamics lineage across the KL anchor-
strength ablation: steps 1-{splice_idx} under KL-to-init coefficient 0.01
(rl_full150), followed by steps {splice_idx_p1}-{n_total} resumed from the
step-{splice_idx} checkpoint under coefficient 0.05 (rl_klprobe, global
steps 51-{n_total_global}) -- the only changed hyperparameter. Faint lines
are raw per-step values; colored lines are centered rolling means (window
{window}), smoothed independently within each phase so the splice
discontinuity is not a smoothing artifact. n=16 rollout trees per step.
(a) Median finalized-report length falls under coef=0.01 and recovers under
coef=0.05, relative to the SFT-initialization median of ~4,900 characters
(dotted line). (b) R(finalized) falls then recovers alongside length. (c)
Citations per child node rise on both sides of the splice -- attribution
improves throughout even as (a)-(b) contract and recover."
"""


if __name__ == "__main__":
    main()
