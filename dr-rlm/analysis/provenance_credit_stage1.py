#!/usr/bin/env python3
"""Stage 1 of the provenance-credit POC: does the credit signal track the TRUE counterfactual?

The thesis bet: per-child credit r_a (read off the citation graph, zero extra rollouts)
approximates the leave-one-child-out difference reward ΔR_a = R(all children) − R(−child a)
— "what the report reward loses without child a". If r_a correlates with ΔR_a, the cheap
structural proxy is a usable training signal; if not, the core mechanism is invalid.

For each captured tree (one rollout, root + sub-agents, from
agent/gen_provenance_poc.sh trajectories) we:

  1. CREDIT (the proxy, judge-scored): run the SAME `compute_rer_rewards` RL uses, under
     each share mode (citation_count / support / ledger_support), to get r_a per child and
     the report reward R. Judge = a held-constant OpenAI-compatible endpoint (judge.py port).

  2. LOCO (the ground truth, re-synthesis): deterministically re-synthesize the root report
     from the children's returns — once with ALL children, once leaving each child out —
     restricting citable evidence to the present children. Judge each → R(all), R(−a).
     ΔR_a = R(all) − R(−a). This is the actual difference reward the proxy claims to estimate.

  3. ALIGN: Spearman/Pearson(r_a, ΔR_a), pooled over child-nodes and within-tree; orphan
     separation (do LOCO-important children get nonzero r_a?); and the per-node advantage
     variance under L3 (per-child r_a) vs L1 (broadcast R) — the second RQ4 claim.

Usage:
  python analysis/provenance_credit_stage1.py \
      --runs runs/provenance_poc \
      --judge-model gpt-4.1-mini --judge-base-url https://api.openai.com/v1 --judge-api-key-env OPENAI_API_KEY \
      --synth-model gpt-4.1-mini --synth-base-url https://api.openai.com/v1 --synth-api-key-env OPENAI_API_KEY \
      --out results/provenance_credit_stage1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

_REPO = Path(__file__).resolve().parents[2]
_SKYRL = _REPO / "dr-rlm" / "rl" / "skyrl"
for p in (str(_SKYRL), str(_SKYRL / "skyrl-gym")):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.train.dr_rlm.judge import RubricJudge, JudgeConfig, weighted_report_reward  # noqa: E402
from examples.train.dr_rlm.rer_reward import RerNode, compute_rer_rewards  # noqa: E402
from examples.train.dr_rlm.corpus_search import owner_rid_of  # noqa: E402

_CITE_RE = re.compile(r'<cite\s+id="([^"]+)">', re.I)


# ----------------------------- tree loading -----------------------------
def load_tree(traj_path: Path) -> Optional[dict]:
    rows = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    meta = next((r for r in rows if r.get("type") == "metadata"), {})
    nodes = [r for r in rows if r.get("type") == "node"]
    ledger = next((r for r in rows if r.get("type") == "ledger"), None)
    if not nodes:
        return None
    return {"meta": meta, "nodes": nodes, "ledger": ledger, "path": str(traj_path)}


def rubrics_for(example_id, subset_rows: Dict[int, dict]) -> List[dict]:
    row = subset_rows.get(int(example_id)) if example_id is not None else None
    return (row or {}).get("rubrics", []) or []


# ----------------------------- credit (proxy) -----------------------------
def to_rernodes(tree: dict) -> Tuple[List[RerNode], dict]:
    nodes = tree["nodes"]
    objs: List[RerNode] = []
    for n in nodes:
        objs.append(RerNode(
            rid=str(n["rid"]),
            depth=int(n.get("depth", 0) or 0),
            parent_rid=(str(n["parent_rid"]) if n.get("parent_rid") else None),
            final_answer=n.get("final_answer") or "",
            cost=0.0,
            cited_ids=list(n.get("cited_ids") or []),
        ))
    ledger = None
    if tree.get("ledger"):
        ledger = {"snippets": tree["ledger"].get("snippets", {}), "_lock": _DummyLock()}
    return objs, ledger


class _DummyLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ----------------------------- LOCO (ground truth) -----------------------------
_SYNTH_SYSTEM = (
    "You are a research orchestrator. You are given a question and a set of findings produced "
    "by sub-agents. Each finding contains claims with inline citation tags of the form "
    '<cite id="...">claim</cite>. Write a comprehensive, well-organized report that answers the '
    "question by synthesizing ONLY the provided findings.\n"
    "STRICT RULES (this is a controlled evidence-attribution test):\n"
    "- Use ONLY information contained in the provided findings. Do NOT add facts, figures, or "
    "claims from your own knowledge. If the findings do not cover an aspect of the question, "
    "OMIT it rather than filling it in.\n"
    "- Preserve the inline <cite id=\"...\"> tags verbatim on every claim you take from a finding; "
    "do NOT invent citation ids and do NOT cite ids absent from the provided findings.\n"
    "- Be thorough about everything the findings DO support, and silent about everything they do not."
)


def _child_block(idx: int, content: str) -> str:
    return f"<finding source=\"sub-agent-{idx}\">\n{content}\n</finding>"


async def _synthesize(client: httpx.AsyncClient, cfg: dict, question: str,
                      child_contents: List[str]) -> str:
    findings = "\n\n".join(_child_block(i + 1, c) for i, c in enumerate(child_contents) if c.strip())
    if not findings.strip():
        return ""
    user = f"<question>{question}</question>\n\nFINDINGS:\n{findings}\n\nWrite the synthesized report now."
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": _SYNTH_SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    for attempt in range(4):
        try:
            r = await client.post(url, json=body, headers=headers, timeout=180.0)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception:
            if attempt == 3:
                return ""
            await asyncio.sleep(2 ** attempt)
    return ""


async def _report_reward(judge: RubricJudge, report: str, question: str, rubrics: List[dict]) -> float:
    if not rubrics:
        return await judge.score_quality(report, question)
    weights = [float(r.get("weight", 1.0)) for r in rubrics]
    s_list = await asyncio.gather(*[judge.score_criterion(report, question, r["description"]) for r in rubrics])
    return weighted_report_reward(list(s_list), weights)


async def loco_for_tree(tree: dict, question: str, rubrics: List[dict],
                        synth_cfg: dict, judge: RubricJudge) -> dict:
    """Re-synthesize all-vs-leave-one-out, judge each, return {rid: ΔR_a, _R_all, _R_minus}."""
    nodes = tree["nodes"]
    root = next((n for n in nodes if int(n.get("depth", 0) or 0) == 0), nodes[0])
    children = [n for n in nodes if n.get("parent_rid") == root["rid"]]
    child_ids = [str(c["rid"]) for c in children]
    child_content = {str(c["rid"]): (c.get("final_answer") or "") for c in children}
    non_empty = [cid for cid in child_ids if child_content[cid].strip()]
    if len(non_empty) < 2:
        return {"deltas": {}, "R_all": None, "R_minus": {}, "n_children_used": len(non_empty)}

    async with httpx.AsyncClient() as client:
        # all-children synthesis
        rep_all = await _synthesize(client, synth_cfg, question, [child_content[c] for c in non_empty])
        R_all = await _report_reward(judge, rep_all, question, rubrics)
        # leave-one-out
        R_minus: Dict[str, float] = {}
        rep_minus: Dict[str, str] = {}
        loo_reports = await asyncio.gather(*[
            _synthesize(client, synth_cfg, question, [child_content[c] for c in non_empty if c != drop])
            for drop in non_empty
        ])
        R_minus_vals = await asyncio.gather(*[
            _report_reward(judge, rep, question, rubrics) for rep in loo_reports
        ])
        for drop, rep, rmv in zip(non_empty, loo_reports, R_minus_vals):
            rep_minus[drop] = rep
            R_minus[drop] = rmv

    deltas = {cid: (R_all - R_minus[cid]) for cid in non_empty}
    return {"deltas": deltas, "R_all": R_all, "R_minus": R_minus, "n_children_used": len(non_empty)}


# ----------------------------- stats -----------------------------
def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _verdict(align: dict, go_rho: float = 0.40, weak_rho: float = 0.20) -> dict:
    """Apply the pre-registered go/no-go logic (see results/provenance_credit_POC.md).

    GO          : some share mode reaches pooled Spearman >= go_rho AND its credited children
                  have higher mean ΔR than its orphans (credit points the right way).
    CONDITIONAL : only the ledger path (ledger_support) clears the bar, not citation_count
                  (orchestrators drop <cite> tags on synthesis → must train on the ledger signal).
    WEAK        : best mode in [weak_rho, go_rho) — a real but soft signal; more data / tuning.
    NO-GO       : no share mode shows positive alignment (all < weak_rho or wrong-signed orphan gap).
    """
    rows = {}
    for sm, a in align.items():
        rho = a.get("pooled_spearman")
        mc, mo = a.get("mean_deltaR_credited"), a.get("mean_deltaR_orphan")
        sep = (mc - mo) if (mc is not None and mo is not None) else None
        rows[sm] = {"pooled_spearman": rho, "credited_minus_orphan_deltaR": sep,
                    "passes": bool(rho is not None and rho >= go_rho and (sep is None or sep > 0))}
    passing = [sm for sm, r in rows.items() if r["passes"]]
    best = max((sm for sm in rows if rows[sm]["pooled_spearman"] is not None),
               key=lambda sm: rows[sm]["pooled_spearman"], default=None)
    best_rho = rows[best]["pooled_spearman"] if best else None

    if passing:
        if passing == ["ledger_support"]:
            decision = "CONDITIONAL-GO (ledger_support only)"
        else:
            decision = "GO"
    elif best_rho is not None and best_rho >= weak_rho:
        decision = "WEAK"
    else:
        decision = "NO-GO"
    return {"decision": decision, "best_mode": best, "best_pooled_spearman": best_rho,
            "by_mode": rows, "thresholds": {"go_rho": go_rho, "weak_rho": weak_rho}}


# ----------------------------- main -----------------------------
async def amain(args) -> None:
    subset_rows: Dict[int, dict] = {}
    if args.subset and os.path.exists(args.subset):
        with open(args.subset) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    subset_rows[int(r["id"])] = r

    judge = RubricJudge(JudgeConfig(
        model=args.judge_model, base_url=args.judge_base_url,
        api_key=os.environ.get(args.judge_api_key_env, "EMPTY"),
        max_concurrency=args.concurrency,
    ))
    synth_cfg = {
        "model": args.synth_model, "base_url": args.synth_base_url,
        "api_key": os.environ.get(args.synth_api_key_env, ""),
    }
    base_payload = {
        "judge_model": args.judge_model, "judge_base_url": args.judge_base_url,
        "judge_api_key_env": args.judge_api_key_env, "gamma_cost": 0.0,
    }

    trees = []
    for run in args.runs:
        tdir = Path(run) / "trajectories"
        if not tdir.is_dir():
            print(f"[skip] no trajectories/ in {run}")
            continue
        for tp in sorted(tdir.glob("traj_*.jsonl")):
            t = load_tree(tp)
            if t is not None:
                trees.append(t)
    print(f"[stage1] loaded {len(trees)} trees")

    share_modes = ["citation_count", "support", "ledger_support"]
    per_child_rows: List[dict] = []   # one row per (tree, child)
    tree_summaries: List[dict] = []

    for ti, tree in enumerate(trees):
        eid = tree["meta"].get("example_id")
        question = tree["meta"].get("problem") or ""
        rubrics = rubrics_for(eid, subset_rows)
        nodes = tree["nodes"]
        root = next((n for n in nodes if int(n.get("depth", 0) or 0) == 0), nodes[0])
        children = [n for n in nodes if n.get("parent_rid") == root["rid"]]
        if len(children) < 2:
            print(f"[tree {ti} id={eid}] <2 children — skip (no within-tree counterfactual)")
            continue

        objs, ledger = to_rernodes(tree)
        # credit under each share mode
        credit: Dict[str, Dict[str, float]] = {}
        report_R = None
        for sm in share_modes:
            payload = dict(base_payload, reward_mode="rer", share_mode=sm)
            try:
                res = await compute_rer_rewards(objs, rubrics, question, payload, ledger=ledger)
                credit[sm] = dict(res.rewards)
                report_R = res.report_reward
            except Exception as e:
                print(f"[tree {ti}] credit({sm}) failed: {e}")
                credit[sm] = {}

        # LOCO ground truth
        loco = await loco_for_tree(tree, question, rubrics, synth_cfg, judge)
        deltas = loco["deltas"]

        child_rids = [str(c["rid"]) for c in children]
        for cid in child_rids:
            row = {
                "tree": ti, "example_id": eid, "child_rid": cid,
                "delta_R": deltas.get(cid),
                "R_minus": loco["R_minus"].get(cid),
                "report_R": report_R,
                "R_all": loco["R_all"],
            }
            for sm in share_modes:
                row[f"r_a_{sm}"] = credit.get(sm, {}).get(cid)
            per_child_rows.append(row)

        # within-tree spearman per share mode (only children with a LOCO delta)
        usable = [c for c in child_rids if deltas.get(c) is not None]
        tsum = {"tree": ti, "example_id": eid, "n_children": len(children),
                "n_children_loco": len(usable), "R_all": loco["R_all"], "report_R": report_R}
        for sm in share_modes:
            xs = [credit.get(sm, {}).get(c, 0.0) for c in usable]
            ys = [deltas[c] for c in usable]
            tsum[f"spearman_{sm}"] = _spearman(xs, ys) if len(usable) >= 2 else None
        tree_summaries.append(tsum)
        print(f"[tree {ti} id={eid}] children={len(children)} loco={len(usable)} "
              f"R_all={loco['R_all']} " +
              " ".join(f"ρ_{sm[:3]}={tsum[f'spearman_{sm}']}" for sm in share_modes))

    # ---------- pooled stats ----------
    def pooled(sm: str) -> dict:
        pairs = [(r[f"r_a_{sm}"], r["delta_R"]) for r in per_child_rows
                 if r.get(f"r_a_{sm}") is not None and r.get("delta_R") is not None]
        if len(pairs) < 2:
            return {"n": len(pairs)}
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        # orphan separation: mean ΔR for children with r_a==0 vs r_a>0
        pos = [y for x, y in pairs if x > 1e-9]
        zero = [y for x, y in pairs if x <= 1e-9]
        wt = [t[f"spearman_{sm}"] for t in tree_summaries if t.get(f"spearman_{sm}") is not None]
        return {
            "n": len(pairs),
            "pooled_spearman": _spearman(xs, ys),
            "pooled_pearson": _pearson(xs, ys),
            "mean_within_tree_spearman": st.mean(wt) if wt else None,
            "n_trees_with_spearman": len(wt),
            "mean_deltaR_credited": st.mean(pos) if pos else None,
            "mean_deltaR_orphan": st.mean(zero) if zero else None,
            "n_credited": len(pos), "n_orphan": len(zero),
        }

    align = {sm: pooled(sm) for sm in share_modes}

    # advantage variance: L1 (broadcast R to every child) vs L3 (per-child r_a)
    # within each tree, variance of the per-child advantage signal; report mean over trees.
    def adv_variance() -> dict:
        l1_vars, l3_vars = [], []
        by_tree = defaultdict(list)
        for r in per_child_rows:
            by_tree[r["tree"]].append(r)
        for ti2, rows in by_tree.items():
            ra = [r.get("r_a_citation_count") for r in rows if r.get("r_a_citation_count") is not None]
            if len(ra) >= 2:
                l3_vars.append(st.pvariance(ra))
                R = rows[0].get("report_R") or 0.0
                l1_vars.append(st.pvariance([R] * len(ra)))  # broadcast => 0 within-tree spread
        return {
            "mean_within_tree_var_L3_credit": st.mean(l3_vars) if l3_vars else None,
            "mean_within_tree_var_L1_broadcast": st.mean(l1_vars) if l1_vars else None,
            "note": "L1 broadcasts identical R to all children => 0 within-tree spread (no differentiation); "
                    "L3 spread>0 means credit differentiates siblings.",
        }

    verdict = _verdict(align)

    out = {
        "config": vars(args),
        "n_trees_loaded": len(trees),
        "n_trees_analyzed": len(tree_summaries),
        "n_child_rows": len(per_child_rows),
        "alignment_by_share_mode": align,
        "advantage_variance": adv_variance(),
        "verdict": verdict,
        "tree_summaries": tree_summaries,
        "per_child_rows": per_child_rows,
    }
    print("\n==== ALIGNMENT (r_a vs LOCO ΔR_a) ====")
    print(json.dumps(align, indent=2))
    print("==== ADVANTAGE VARIANCE ====")
    print(json.dumps(out["advantage_variance"], indent=2))
    print("==== VERDICT ====")
    print(json.dumps(verdict, indent=2))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {outp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--subset", default=str(_REPO / "dr-rlm" / "data" / "subsets" / "drtulu_rl_decompose40.jsonl"))
    ap.add_argument("--judge-model", default="gemini-2.5-flash")
    ap.add_argument("--judge-base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    ap.add_argument("--judge-api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--synth-model", default="gemini-2.5-flash")
    ap.add_argument("--synth-base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    ap.add_argument("--synth-api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(_REPO / "dr-rlm" / "results" / "provenance_credit_stage1.json"))
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
