#!/usr/bin/env python3
"""Publication-quality RL training-curve figures from a completed run.

Reads per-tree credit-metrics telemetry (one JSON row per rollout tree) plus
optional console-log entropy/KL fragments, buckets trees into training steps
(16 trees/step = 4 prompts x 4 samples), computes the metric definitions below,
and writes six figures (PDF + 300-dpi PNG) plus a README describing them.

Metric definitions (per step):
  root            = node with depth==0 (first match) in a tree
  finalized       = root exists and root.ans_len > 0
  E[R]            = mean report_reward over ALL trees (null -> 0)
  R(finalized)    = mean report_reward over finalized trees only
  finalized%      = share of trees finalized
  strict-zero%    = share of FINALIZED trees with report_reward == 0
  median length   = median root.ans_len over finalized trees
  children/tree   = mean count of depth>0 nodes per tree
  cites/child     = mean n_cited over all depth>0 nodes in the step (missing -> 0)
  orphan%         = share of depth>0 nodes with falsy n_cited

Usage:
  python analysis/plot_training_curves.py \
      --metrics runs/rl_training/credit_metrics_rl_fresh100.jsonl \
      --logs runs/logs/rl_fresh100_24502441.err runs/logs/rl_fresh100_24502443.err \
             runs/logs/rl_fresh100_24502445.err \
      --out analysis/figures/rl_fresh100 --window 15
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TREES_PER_STEP = 16
ROLL_WINDOW = 9  # centered rolling-mean window over noisy per-step series (--window)
SFT_INIT_LEN = 4900

# Colorblind-safe categorical palette (dataviz skill reference palette),
# one fixed hue per metric, reused across every figure it appears in.
C_E_R = "#2a78d6"          # blue      -- E[R]
C_R_FIN = "#184f95"        # dark blue -- R(finalized), same family as E[R]
C_CITES = "#1baf7a"        # aqua      -- cites/child
C_ORPHAN = "#e34948"        # red       -- orphan%
C_LENGTH = "#eb6834"        # orange    -- median report length
C_FINAL = "#008300"        # green     -- finalized%
C_STRICT0 = "#e87ba4"        # magenta   -- strict-zero%
C_ENTROPY = "#4a3aa7"        # violet    -- entropy
C_KL = "#eda100"        # yellow    -- policy KL
C_RAW = "#c3c2b7"        # muted grey-- faint raw per-step series


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.4,
    })


def load_metrics(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["ts"])

    # Crash-duplicated rollouts from chunked checkpoint resumption: chunk 1
    # timed out and chunk 2 crashed mid-step, so their in-flight rollouts were
    # logged but never trained on, and the resumed chunk re-rolled the same
    # (instance_id, repetition_id) prompts. Keep the LAST occurrence -- the
    # re-roll is the copy that actually produced a gradient.
    dedup: Dict[tuple, dict] = {}
    for r in rows:
        dedup[(r["instance_id"], r["repetition_id"])] = r
    rows = sorted(dedup.values(), key=lambda r: r["ts"])

    assert len(rows) % TREES_PER_STEP == 0, (
        f"{len(rows)} deduped trees not divisible by {TREES_PER_STEP}/step"
    )
    print(f"[dedup] {len(rows)} trees -> {len(rows) // TREES_PER_STEP} steps")
    return rows


def _root(tree: dict) -> Optional[dict]:
    for n in tree.get("nodes") or []:
        if int(n.get("depth", 0) or 0) == 0:
            return n
    return None


def _children(tree: dict) -> List[dict]:
    return [n for n in (tree.get("nodes") or []) if int(n.get("depth", 0) or 0) > 0]


def step_stats(trees: List[dict]) -> dict:
    rr_all = [t.get("report_reward") or 0.0 for t in trees]
    roots = [(_root(t), t.get("report_reward") or 0.0) for t in trees]
    finalized = [(r, rr) for r, rr in roots if r is not None and (r.get("ans_len") or 0) > 0]

    kids = [c for t in trees for c in _children(t)]

    out: Dict[str, Optional[float]] = {
        "n_trees": len(trees),
        "e_r": st.mean(rr_all) if rr_all else None,
        "r_finalized": st.mean([rr for _, rr in finalized]) if finalized else None,
        "finalized_pct": (len(finalized) / len(trees)) if trees else None,
        "strict_zero_pct": (
            sum(1 for _, rr in finalized if rr == 0) / len(finalized) if finalized else None
        ),
        "median_len": (
            st.median([r.get("ans_len") or 0 for r, _ in finalized]) if finalized else None
        ),
        "children_per_tree": (len(kids) / len(trees)) if trees else None,
        "cites_per_child": st.mean([c.get("n_cited") or 0 for c in kids]) if kids else None,
        "orphan_pct": (
            sum(1 for c in kids if not c.get("n_cited")) / len(kids) if kids else None
        ),
    }
    return out


def bucket_by_step(rows: List[dict]) -> List[dict]:
    buckets = [rows[i:i + TREES_PER_STEP] for i in range(0, len(rows), TREES_PER_STEP)]
    return [step_stats(b) for b in buckets]


_ENTROPY_RE = re.compile(r"entropy':\s*'([0-9.eE+-]+)'")
_KL_RE = re.compile(r"policy_kl':\s*'([0-9.eE+-]+)'")


def extract_log_series(paths: List[str]) -> Dict[str, List[float]]:
    """Extract entropy/policy_kl in chronological file order (one match/step)."""
    entropy: List[float] = []
    kl: List[float] = []
    for p in paths:
        text = Path(p).read_text(errors="replace")
        entropy.extend(float(x) for x in _ENTROPY_RE.findall(text))
        kl.extend(float(x) for x in _KL_RE.findall(text))
    return {"entropy": entropy, "kl": kl}


def rolling_mean(vals: List[Optional[float]], window: Optional[int] = None) -> np.ndarray:
    window = window or ROLL_WINDOW
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    half = window // 2
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        window_vals = arr[lo:hi]
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals):
            out[i] = window_vals.mean()
    return out


def _plot_series(ax, x, raw, color, label):
    ax.plot(x, raw, color=C_RAW, linewidth=0.8, alpha=0.6, zorder=1)
    ax.plot(x, rolling_mean(raw), color=color, linewidth=1.6,
             label=label, zorder=2)


def _finish(ax_or_fig, out_stem: Path):
    fig = ax_or_fig if hasattr(ax_or_fig, "savefig") else ax_or_fig.figure
    fig.tight_layout()
    fig.savefig(str(out_stem) + ".pdf")
    fig.savefig(str(out_stem) + ".png")
    plt.close(fig)


def fig_reward_curve(steps, stats, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    e_r = [s["e_r"] for s in stats]
    r_fin = [s["r_finalized"] for s in stats]
    _plot_series(ax, steps, e_r, C_E_R, "E[R] (all trees)")
    ax.plot(steps, rolling_mean(r_fin), color=C_R_FIN, linewidth=1.6,
            label="R(finalized)")
    ax.plot(steps, r_fin, color=C_RAW, linewidth=0.8, alpha=0.4, zorder=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean report reward")
    ax.legend(loc="lower right")
    _finish(fig, out_dir / "fig_reward_curve")


def fig_trifecta(steps, stats, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(5, 6.4), sharex=True)
    e_r = [s["e_r"] for s in stats]
    cites = [s["cites_per_child"] for s in stats]
    orphan = [100 * s["orphan_pct"] if s["orphan_pct"] is not None else None for s in stats]

    _plot_series(axes[0], steps, e_r, C_E_R, "E[R]")
    axes[0].set_ylabel("mean report reward")

    _plot_series(axes[1], steps, cites, C_CITES, "cites/child")
    axes[1].set_ylabel("citations per child")

    _plot_series(axes[2], steps, orphan, C_ORPHAN, "orphan%")
    axes[2].set_ylabel("orphan rate (%)")
    axes[2].set_xlabel("training step")

    for ax in axes:
        ax.legend(loc="best")
    _finish(fig, out_dir / "fig_trifecta")


def fig_report_length(steps, stats, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    length = [s["median_len"] for s in stats]
    _plot_series(ax, steps, length, C_LENGTH, "median report length")
    ax.axhline(SFT_INIT_LEN, color=C_RAW, linestyle="--", linewidth=1.0,
               label="SFT init (~4.9k chars)")
    ax.set_xlabel("training step")
    ax.set_ylabel("median report length (chars)")
    ax.legend(loc="best")
    _finish(fig, out_dir / "fig_report_length")


def fig_finalization(steps, stats, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fin = [100 * s["finalized_pct"] if s["finalized_pct"] is not None else None for s in stats]
    zero = [100 * s["strict_zero_pct"] if s["strict_zero_pct"] is not None else None for s in stats]
    _plot_series(ax, steps, fin, C_FINAL, "finalized%")
    ax.plot(steps, rolling_mean(zero), color=C_STRICT0, linewidth=1.6, label="strict-zero%")
    ax.plot(steps, zero, color=C_RAW, linewidth=0.8, alpha=0.4, zorder=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of trees (%)")
    ax.legend(loc="best")
    _finish(fig, out_dir / "fig_finalization")


def fig_guardrails(log_steps, entropy, kl, out_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(5, 4.4), sharex=True)
    _plot_series(axes[0], log_steps, entropy, C_ENTROPY, "policy entropy")
    axes[0].set_ylabel("policy entropy")

    _plot_series(axes[1], log_steps, kl, C_KL, "policy KL")
    axes[1].set_ylabel("policy KL")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("training step (approx.)")

    for ax in axes:
        ax.legend(loc="best")
    _finish(fig, out_dir / "fig_guardrails")


def fig_training_overview(steps, stats, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    e_r = [s["e_r"] for s in stats]
    cites = [s["cites_per_child"] for s in stats]
    orphan = [100 * s["orphan_pct"] if s["orphan_pct"] is not None else None for s in stats]
    length = [s["median_len"] for s in stats]
    fin = [100 * s["finalized_pct"] if s["finalized_pct"] is not None else None for s in stats]

    panels = [
        (axes[0, 0], e_r, C_E_R, "mean report reward"),
        (axes[0, 1], cites, C_CITES, "citations per child"),
        (axes[0, 2], orphan, C_ORPHAN, "orphan rate (%)"),
        (axes[1, 0], length, C_LENGTH, "median report length (chars)"),
        (axes[1, 1], fin, C_FINAL, "finalized (%)"),
    ]
    for ax, series, color, ylabel in panels:
        _plot_series(ax, steps, series, color, None)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("training step")

    axes[1, 0].axhline(SFT_INIT_LEN, color=C_RAW, linestyle="--", linewidth=0.9)

    # entropy goes in the 6th panel if log-derived series is available and
    # roughly matches the step count; fall back to leaving it blank.
    axes[1, 2].axis("off")

    _finish(fig, out_dir / "fig_training_overview")


def fig_training_overview_with_entropy(steps, stats, log_steps, entropy, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    e_r = [s["e_r"] for s in stats]
    cites = [s["cites_per_child"] for s in stats]
    orphan = [100 * s["orphan_pct"] if s["orphan_pct"] is not None else None for s in stats]
    length = [s["median_len"] for s in stats]
    fin = [100 * s["finalized_pct"] if s["finalized_pct"] is not None else None for s in stats]

    panels = [
        (axes[0, 0], steps, e_r, C_E_R, "mean report reward"),
        (axes[0, 1], steps, cites, C_CITES, "citations per child"),
        (axes[0, 2], steps, orphan, C_ORPHAN, "orphan rate (%)"),
        (axes[1, 0], steps, length, C_LENGTH, "median report length (chars)"),
        (axes[1, 1], steps, fin, C_FINAL, "finalized (%)"),
        (axes[1, 2], log_steps, entropy, C_ENTROPY, "policy entropy"),
    ]
    for ax, x, series, color, ylabel in panels:
        _plot_series(ax, x, series, color, None)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("training step" if ylabel != "policy entropy" else "training step (approx.)")

    axes[1, 0].axhline(SFT_INIT_LEN, color=C_RAW, linestyle="--", linewidth=0.9)
    _finish(fig, out_dir / "fig_training_overview")


README_TEMPLATE = """# RL training-curve figures

Generated by `analysis/plot_training_curves.py` from
`{metrics_path}` (per-tree telemetry, {n_trees} trees, {n_steps} steps of
{trees_per_step} trees/step) and {n_logs} console log(s) for entropy/KL.

Data note: the raw telemetry contains 119 crash-duplicated rollouts from
chunked checkpoint resumption (a timed-out/crashed chunk's in-flight
rollouts were logged but never trained on, then re-rolled on resume). These
are deduplicated on (instance_id, repetition_id), keeping the later
(actually-trained) occurrence, before any of the figures below are computed.

Noise note: each step aggregates only {trees_per_step} rollout trees, so raw
per-step values are noisy (per-step SE of E[R] ~0.07). Every figure shows the
raw per-step series as a faint grey line and a centered rolling mean
(window={roll_window}) as the prominent colored line; that is the curve to
read. The statistically reliable trend statement is the first-vs-last-fifth
comparison printed by the script (320 trees per fifth).

- `fig_reward_curve` -- Mean report reward per training step, over all
  rollout trees (E[R]) and over finalized trees only (R(finalized)).
  Caption: "Mean report reward across all sampled rollout trees (E[R]) and
  restricted to trees that finalized a report (R(finalized)) over the course
  of RL training, 16 trees per step."
- `fig_trifecta` -- The headline result: reward, citation density, and
  orphan rate over training, stacked on a shared step axis.
  Caption: "Joint evolution of mean report reward, citations per child node,
  and the orphan rate (share of child nodes with zero citations) over RL
  training, showing reward gains alongside improved attribution."
- `fig_report_length` -- Median finalized-report length per step, with a
  reference line at the SFT-init length.
  Caption: "Median length (characters) of finalized reports per training
  step, relative to the SFT-initialization median of approximately 4,900
  characters (dashed line)."
- `fig_finalization` -- Share of rollout trees that finalized a report, and
  the share of finalized reports scored exactly zero.
  Caption: "Finalization rate (trees that emitted a non-empty root report)
  and strict-zero rate (finalized reports scored 0) per training step."
- `fig_guardrails` -- Policy entropy and policy KL divergence from the
  training console logs, in log-file chronological order (approximate step
  alignment).
  Caption: "Policy entropy and KL divergence from the reference policy over
  the course of RL training, log scale on KL."
- `fig_training_overview` -- Compact 2x3 grid combining reward, cites/child,
  orphan rate, median report length, finalized%, and policy entropy.
  Caption: "Overview of reward, attribution, and stability metrics over the
  course of RL training."

Each figure is exported as both `.pdf` (vector, for LaTeX) and `.png`
(300 dpi, for previews/slides).
"""


def sanity_fifths(steps, stats, key, transform=lambda v: v):
    vals = [transform(s[key]) if s[key] is not None else None for s in stats]
    n = len(vals)
    fifth = max(1, n // 5)
    chunks = [vals[i * fifth: (i + 1) * fifth] for i in range(5)]
    chunks[-1] = vals[4 * fifth:]  # last chunk absorbs remainder
    means = []
    for c in chunks:
        c = [v for v in c if v is not None]
        means.append(st.mean(c) if c else None)
    return means


def mann_whitney_p(x: List[float], y: List[float]) -> float:
    """Two-sided Mann-Whitney U p-value, normal approximation with tie
    correction (no scipy dependency)."""
    n1, n2 = len(x), len(y)
    n = n1 + n2
    # rank all values together (average rank for ties)
    tagged = [(v, "x") for v in x] + [(v, "y") for v in y]
    tagged.sort(key=lambda t: t[0])
    ranks = [0.0] * n
    i = 0
    tie_term = 0
    while i < n:
        j = i
        while j < n and tagged[j][0] == tagged[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed ranks i+1..j
        for k in range(i, j):
            ranks[k] = avg_rank
        t = j - i
        tie_term += t ** 3 - t
        i = j
    r1 = sum(r for r, (v, tag) in zip(ranks, tagged) if tag == "x")
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    if sigma == 0:
        return 1.0
    z = (u1 - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def fifths_report(rows: List[dict]) -> None:
    """Per-tree reward std/SE and first-fifth-vs-last-fifth E[R] comparison
    on deduped data (steps 1-20 vs steps 81-100, 320 trees each)."""
    rr_all = [r.get("report_reward") or 0.0 for r in rows]
    tree_std = st.stdev(rr_all)
    step_se = tree_std / math.sqrt(TREES_PER_STEP)
    print(f"[stats] per-tree report_reward std={tree_std:.4f}, "
          f"per-step SE (n=16)={step_se:.4f}")

    fifth = len(rows) // 5
    first = rr_all[:fifth]
    last = rr_all[-fifth:]
    m1, m2 = st.mean(first), st.mean(last)
    s1, s2 = st.stdev(first), st.stdev(last)
    diff = m2 - m1
    se_diff = math.sqrt(s1 ** 2 / len(first) + s2 ** 2 / len(last))
    p = mann_whitney_p(first, last)
    print(f"[stats] E[R] first fifth (n={len(first)}) mean={m1:.4f}  "
          f"last fifth (n={len(last)}) mean={m2:.4f}  "
          f"diff={diff:+.4f}  SE(diff)={se_diff:.4f}  "
          f"Mann-Whitney p={p:.4g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=9,
                    help="centered rolling-mean window (steps)")
    args = ap.parse_args()

    global ROLL_WINDOW  # ponytail: module global beats threading window through 8 signatures
    ROLL_WINDOW = args.window

    _style()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(args.metrics)
    fifths_report(rows)
    stats = bucket_by_step(rows)
    steps = list(range(1, len(stats) + 1))

    fig_reward_curve(steps, stats, out_dir)
    fig_trifecta(steps, stats, out_dir)
    fig_report_length(steps, stats, out_dir)
    fig_finalization(steps, stats, out_dir)

    log_series = extract_log_series(args.logs) if args.logs else {"entropy": [], "kl": []}
    entropy, kl = log_series["entropy"], log_series["kl"]
    if entropy or kl:
        log_steps = list(range(1, max(len(entropy), len(kl)) + 1))
        fig_guardrails(log_steps, entropy, kl, out_dir)
        fig_training_overview_with_entropy(steps, stats, log_steps, entropy, out_dir)
    else:
        fig_training_overview(steps, stats, out_dir)

    readme = README_TEMPLATE.format(
        metrics_path=args.metrics, n_trees=len(rows), n_steps=len(stats),
        trees_per_step=TREES_PER_STEP, n_logs=len(args.logs), roll_window=ROLL_WINDOW,
    )
    (out_dir / "README.md").write_text(readme)

    # --- sanity report (fifths) ---
    print(f"[info] {len(rows)} trees -> {len(stats)} steps of {TREES_PER_STEP} trees")
    print(f"[info] log series: entropy={len(entropy)} kl={len(kl)}")
    for key, xf, label in [
        ("e_r", lambda v: v, "E[R]"),
        ("cites_per_child", lambda v: v, "cites/child"),
        ("orphan_pct", lambda v: 100 * v, "orphan% "),
        ("median_len", lambda v: v, "median length"),
    ]:
        fifths = sanity_fifths(steps, stats, key, xf)
        print(f"[sanity] {label}: " + " -> ".join(f"{v:.3f}" if v is not None else "NA" for v in fifths))

    print(f"[done] figures written to {out_dir}")


if __name__ == "__main__":
    main()
