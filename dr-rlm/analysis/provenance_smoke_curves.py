#!/usr/bin/env python3
"""Smoke-run evidence curves: is provenance credit DOING something during RL?

Parses the opt-in credit telemetry (DR_RLM_CREDIT_METRICS_PATH jsonl; one row per
rollout tree, written in generation order) from the L3 and L1 arms and computes the
pre-registered evidence metrics over training progress (rows bucketed sequentially
into pseudo-steps of train_batch_size x n_samples_per_prompt trees):

  A. plumbing (L3 must-pass, per bucket):
     - conservation: |sum_a r_a^evidence - R| (from rer_metrics; ledger mode conserves)
     - within-tree advantage spread: std of children's r_a (L3 > 0; L1 structurally 0)
     - judge health: frac trees with report_reward > 0
  B. learning-signal slopes (L3 vs L1, the differential prediction):
     - orphan fraction: frac children with r_a ~ 0 (L3) / with n_cited == 0 (both arms)
     - cites per child (both arms)
     - report reward trend (both arms; sanity, not the differentiator)
  C. reward-hacking watch: cites/child should not explode (citation stuffing)

Usage:
  python analysis/provenance_smoke_curves.py \
      --l3 runs/provenance_smoke/credit_metrics_L3.jsonl \
      --l1 runs/provenance_smoke/credit_metrics_L1.jsonl \
      --bucket 16 --out results/provenance_smoke_curves.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional


def load(path: str) -> List[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def bucketize(rows: List[dict], k: int) -> List[List[dict]]:
    return [rows[i:i + k] for i in range(0, len(rows), k)]


def _children(tree: dict) -> List[dict]:
    return [n for n in (tree.get("nodes") or []) if int(n.get("depth", 0) or 0) > 0]


def bucket_stats(bucket: List[dict]) -> dict:
    out: Dict[str, Optional[float]] = {"n_trees": len(bucket)}
    kids = [c for t in bucket for c in _children(t)]
    out["n_children"] = len(kids)
    out["recursion_rate"] = (sum(1 for t in bucket if _children(t)) / len(bucket)) if bucket else None
    if kids:
        out["frac_children_cite0"] = sum(1 for c in kids if not c.get("n_cited")) / len(kids)
        out["mean_cites_per_child"] = st.mean([c.get("n_cited", 0) for c in kids])
    # L3-only fields (r_a present)
    ra_kids = [c for c in kids if c.get("r_a") is not None]
    if ra_kids:
        out["frac_children_credit0"] = sum(1 for c in ra_kids if abs(c["r_a"]) < 1e-9) / len(ra_kids)
        spreads = []
        for t in bucket:
            cs = [c["r_a"] for c in _children(t) if c.get("r_a") is not None]
            if len(cs) >= 2:
                spreads.append(st.pstdev(cs))
        out["mean_within_tree_ra_std"] = st.mean(spreads) if spreads else None
    rr = [t.get("report_reward") for t in bucket if t.get("report_reward") is not None]
    if rr:
        out["mean_report_reward"] = st.mean(rr)
        out["frac_R_positive"] = sum(1 for r in rr if r > 0) / len(rr)
    cons = []
    for t in bucket:
        m = t.get("rer_metrics") or {}
        R = m.get("report_reward")
        if R is None:
            continue
        ev = sum(c.get("r_a", 0.0) or 0.0 for c in (t.get("nodes") or []))
        # gamma_cost=0 in the smoke -> Σ r_a should equal R when any contributor exists
        cons.append(abs(ev - R))
    if cons:
        out["mean_abs_conservation_gap"] = st.mean(cons)
    return out


def slope(vals: List[Optional[float]]) -> Optional[float]:
    """OLS slope over bucket index (None-tolerant)."""
    pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def arm_summary(rows: List[dict], bucket: int) -> dict:
    buckets = [bucket_stats(b) for b in bucketize(rows, bucket)]
    keys = sorted({k for b in buckets for k in b})
    series = {k: [b.get(k) for b in buckets] for k in keys}
    return {
        "n_rows": len(rows),
        "n_buckets": len(buckets),
        "buckets": buckets,
        "slopes": {k: slope(series[k]) for k in
                   ("frac_children_cite0", "frac_children_credit0", "mean_cites_per_child",
                    "mean_report_reward", "mean_within_tree_ra_std") if k in series},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3", required=True)
    ap.add_argument("--l1", default=None)
    ap.add_argument("--bucket", type=int, default=16,
                    help="trees per pseudo-step (train_batch_size x n_samples_per_prompt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {"L3": arm_summary(load(args.l3), args.bucket)}
    if args.l1:
        res["L1"] = arm_summary(load(args.l1), args.bucket)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "buckets"} for k, v in res.items()}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
