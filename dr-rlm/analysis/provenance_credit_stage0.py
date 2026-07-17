#!/usr/bin/env python3
"""Stage 0 of the provenance-credit POC: is the credit signal non-degenerate on REAL trees?

Loads captured recursion trees (ab_child_return_v2 and friends), rebuilds the RerNode
tree, and computes the L3 *structural* credit shares exactly as training would —
using the SAME citation parser (`judge.extract_claims_and_corresponding_citation_ids`)
and the SAME provenance decoder (`corpus_search.owner_rid_of`) that `rer_reward.py`
uses. In `citation_count` mode r_a = R * share(a), so the structural questions are
judge-free:

  Q1  Degeneracy: what fraction of trees give ZERO credit to every child
      (no report cites / all child evidence orphaned)?
  Q2  Differentiation: among trees with >=2 contributing children, do siblings get
      DIFFERENT shares (std / max-min of sibling shares)?
  Q3  Orphans: per-tree orphan-child fraction (children whose surfaced evidence never
      appears in the root report).
  Q4  Conservation: shares sum to 1 over contributors (sanity on real cite strings).

Absolute scale (optional): joins per-item RACE overall (drb_eval/race/.../raw_results.jsonl)
when present so r_a = RACE * share is also reported.

Usage:
  python analysis/provenance_credit_stage0.py \
      --runs runs/ab_child_return_v2/prose runs/ab_child_return_v2/structured \
      --out results/provenance_credit_stage0.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]  # .../Dr-RLM
_SKYRL = _REPO / "dr-rlm" / "rl" / "skyrl"
for p in (str(_SKYRL), str(_SKYRL / "skyrl-gym")):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.train.dr_rlm.judge import extract_claims_and_corresponding_citation_ids  # noqa: E402
from examples.train.dr_rlm.corpus_search import owner_rid_of  # noqa: E402


# ----------------------------- tree loading -----------------------------
def load_tree(traj_path: Path) -> Optional[dict]:
    """Parse one trajectory JSONL -> {meta, nodes:[{rid,depth,parent_rid,final_answer,cited_ids}]}."""
    rows = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    meta = next((r for r in rows if r.get("type") == "metadata"), {})
    nodes = [r for r in rows if r.get("type") == "node"]
    if not nodes:
        return None
    return {"meta": meta, "nodes": nodes, "path": str(traj_path)}


def race_scores(arm_dir: Path) -> Dict[int, float]:
    """example_id -> RACE overall, defensively parsed from the DRB grader sidecar."""
    out: Dict[int, float] = {}
    for p in arm_dir.glob("drb_eval/race/*/raw_results.jsonl"):
        with open(p) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = j.get("id", j.get("example_id"))
                score = j.get("overall_score", j.get("score"))
                if isinstance(score, dict):
                    score = score.get("overall_score")
                if eid is not None and isinstance(score, (int, float)):
                    try:
                        out[int(eid)] = float(score)
                    except (TypeError, ValueError):
                        pass
    return out


# ----------------------------- credit shares -----------------------------
def tree_credit(tree: dict) -> dict:
    nodes = tree["nodes"]
    root = next((n for n in nodes if n.get("depth") == 0), nodes[0])
    rid_set = {n["rid"] for n in nodes}
    children = [n for n in nodes if n.get("parent_rid") == root["rid"]]

    claims = extract_claims_and_corresponding_citation_ids(root.get("final_answer") or "")
    cite_count: Dict[str, int] = defaultdict(int)
    total_cites = 0
    unknown_owner = 0
    for _claim, ids in claims.items():
        for cid in ids:
            owner = owner_rid_of(cid)
            if owner in rid_set:
                cite_count[owner] += 1
                total_cites += 1
            else:
                unknown_owner += 1

    contributors = [r for r in rid_set if cite_count[r] > 0]
    shares = {r: (cite_count[r] / total_cites if total_cites else 0.0) for r in rid_set}

    child_shares = [shares[c["rid"]] for c in children]
    contributing_children = [c for c in children if cite_count[c["rid"]] > 0]
    res = {
        "example_id": tree["meta"].get("example_id"),
        "n_nodes": len(nodes),
        "n_children": len(children),
        "n_claims": len(claims),
        "total_root_cites_decoded": total_cites,
        "cites_unknown_owner": unknown_owner,
        "root_share": shares[root["rid"]],
        "child_shares": child_shares,
        "n_contributing_children": len(contributing_children),
        "n_orphan_children": len(children) - len(contributing_children),
        "degenerate_all_children_zero": len(children) > 0 and len(contributing_children) == 0,
        "share_sum_over_contributors": sum(shares[r] for r in contributors),
    }
    if len(child_shares) >= 2:
        res["sibling_share_std"] = st.pstdev(child_shares)
        res["sibling_share_spread"] = max(child_shares) - min(child_shares)
    return res


# ----------------------------- main -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="arm dirs containing trajectories/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_rows: List[dict] = []
    for arm in args.runs:
        arm_dir = Path(arm)
        races = race_scores(arm_dir)
        tdir = arm_dir / "trajectories"
        if not tdir.is_dir():
            print(f"[skip] no trajectories/ in {arm_dir}")
            continue
        for tp in sorted(tdir.glob("traj_*.jsonl")):
            tree = load_tree(tp)
            if tree is None:
                continue
            row = tree_credit(tree)
            row["arm"] = str(arm_dir)
            eid = row.get("example_id")
            if eid is not None and int(eid) in races:
                row["race_overall"] = races[int(eid)]
                row["child_r_a"] = [races[int(eid)] * s for s in row["child_shares"]]
            all_rows.append(row)

    if not all_rows:
        print("[FATAL] no trees parsed")
        sys.exit(1)

    # ---------- aggregate ----------
    n = len(all_rows)
    with_children = [r for r in all_rows if r["n_children"] > 0]
    degenerate = [r for r in with_children if r["degenerate_all_children_zero"]]
    no_cites = [r for r in all_rows if r["total_root_cites_decoded"] == 0]
    diff_rows = [r for r in with_children if r.get("sibling_share_std") is not None
                 and r["n_contributing_children"] >= 2]
    orphan_fracs = [r["n_orphan_children"] / r["n_children"] for r in with_children]
    cons = [r["share_sum_over_contributors"] for r in all_rows if r["total_root_cites_decoded"] > 0]

    summary = {
        "n_trees": n,
        "n_trees_with_children": len(with_children),
        "mean_children_per_tree": st.mean([r["n_children"] for r in with_children]) if with_children else 0,
        "Q1_frac_trees_no_decoded_cites": len(no_cites) / n,
        "Q1_frac_trees_all_children_zero_credit": (len(degenerate) / len(with_children)) if with_children else None,
        "Q2_n_trees_with_>=2_contributing_children": len(diff_rows),
        "Q2_mean_sibling_share_std": st.mean([r["sibling_share_std"] for r in diff_rows]) if diff_rows else None,
        "Q2_mean_sibling_share_spread": st.mean([r["sibling_share_spread"] for r in diff_rows]) if diff_rows else None,
        "Q3_mean_orphan_child_fraction": st.mean(orphan_fracs) if orphan_fracs else None,
        "Q4_share_sum_over_contributors_min": min(cons) if cons else None,
        "Q4_share_sum_over_contributors_max": max(cons) if cons else None,
        "mean_root_self_share": st.mean([r["root_share"] for r in all_rows]),
    }

    by_arm: Dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in all_rows}):
        rows = [r for r in all_rows if r["arm"] == arm]
        wc = [r for r in rows if r["n_children"] > 0]
        deg = [r for r in wc if r["degenerate_all_children_zero"]]
        by_arm[arm] = {
            "n_trees": len(rows),
            "frac_no_decoded_cites": sum(1 for r in rows if r["total_root_cites_decoded"] == 0) / len(rows),
            "frac_all_children_zero": (len(deg) / len(wc)) if wc else None,
            "mean_orphan_child_fraction": st.mean([r["n_orphan_children"] / r["n_children"] for r in wc]) if wc else None,
            "mean_root_self_share": st.mean([r["root_share"] for r in rows]),
            "mean_contributing_children": st.mean([r["n_contributing_children"] for r in wc]) if wc else None,
        }

    print(json.dumps({"summary": summary, "by_arm": by_arm}, indent=2))
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump({"summary": summary, "by_arm": by_arm, "trees": all_rows}, f, indent=2)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
