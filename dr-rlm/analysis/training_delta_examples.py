#!/usr/bin/env python3
"""Before/after training-delta assets for the paper (raw-output improvement).

PURPOSE
-------
Show, on the SAME prompts, how Dr-RLM's *raw outputs* change from the start of
training to the end: orphan sub-agents (surface evidence, cite none of it) -> grounded
sub-agents (cite their own retrieved evidence), and the root report's provenance
citations densifying -- at no quality cost (report reward R held).

This reads the opt-in credit telemetry (one JSON row per rollout tree, written in
generation order; same file `analysis/provenance_smoke_curves.py` consumes) and emits:
  - a markdown before/after table (viewable now),
  - a LaTeX booktabs table (drop-in `\input` for the thesis),
  - headline-example JSON (the per-node before/after for ONE instance, feeds the TikZ).

RUN-AGNOSTIC: this is a PLACEHOLDER built from the v1-clean smoke run. To produce the
FINAL paper version, re-point `--metrics` at the final SFT+RL checkpoint's credit_metrics
jsonl -- nothing else changes.

Metric semantics (verified against rl/skyrl/examples/train/dr_rlm/rer_reward.py):
  * ledger_support mode recomputes node_cite_count to each node's OWN-evidence cites.
  * child `n_cited`       = # of a sub-agent's own retrieved snippets it cited (grounding).
  * `n_orphan_children`   = # children that cited NONE of their own evidence (orphans).
  * `total_report_cites`  = # <cite id> tags in the ROOT report (provenance-tagged claims).
  * `report_reward` (R)   = rubric report score (the quality signal).

Usage:
  python analysis/training_delta_examples.py \
      --metrics runs/provenance_smoke/credit_metrics_L3_v1clean.jsonl \
      --prompts data/poc_corpus/train.parquet \
      --headline 27 --k 3 \
      --table-instances 27,9,4,7,20,3 \
      --outdir docs/paper_assets
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CITE_RE = re.compile(r'<cite id="([^"]+)"')


def load_metrics(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows.sort(key=lambda r: r.get("ts", 0.0))  # generation order == training progress
    return rows


def short_label(prompt_field) -> str:
    """Best-effort short topic label from a chat-style prompt array."""
    try:
        import numpy as np  # noqa
        msgs = prompt_field
        text = ""
        if isinstance(msgs, (list, tuple)) or hasattr(msgs, "__iter__"):
            for m in msgs:
                if isinstance(m, dict) and m.get("content"):
                    text = m["content"]
                    break
        text = re.sub(r"\s+", " ", str(text)).strip()
        text = re.sub(r"^\s*(\d+[.)]\s*|[•\-\*]\s*)", "", text)  # strip leading list marker
        # first clause / question; but don't return a tiny fragment
        clause = re.split(r"[:.•?]", text)[0].strip()
        if len(clause) < 14:
            clause = text
        return clause[:58].strip()
    except Exception:
        return "(query)"


def load_labels(parquet: str | None) -> dict[str, str]:
    if not parquet:
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(parquet)
        return {str(i): short_label(df.iloc[i]["prompt"]) for i in range(len(df))}
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not load prompt labels: {e}")
        return {}


def children(tree: dict) -> list[dict]:
    return [n for n in (tree.get("nodes") or []) if int(n.get("depth", 0) or 0) >= 1]


def root(tree: dict) -> dict:
    rs = [n for n in (tree.get("nodes") or []) if int(n.get("depth", 0) or 0) == 0]
    return rs[0] if rs else {}


def agg(trees: list[dict]) -> dict:
    n_orph = n_kids = child_cites = root_cites = 0
    Rs: list[float] = []
    for t in trees:
        ks = children(t)
        n_orph += sum(1 for n in ks if n.get("n_cited", 0) == 0)
        n_kids += len(ks)
        child_cites += sum(n.get("n_cited", 0) for n in ks)
        root_cites += t.get("rer_metrics", {}).get("total_report_cites", 0)
        Rs.append(t.get("report_reward", 0.0))
    alive = [x for x in Rs if x > 0]
    return {
        "n_trees": len(trees),
        "orphan_pct": 100 * n_orph / max(n_kids, 1),
        "child_cites_per_kid": child_cites / max(n_kids, 1),
        "root_cites_per_tree": root_cites / max(len(trees), 1),
        "R_mean": sum(Rs) / max(len(Rs), 1),
        "R_alive": sum(alive) / max(len(alive), 1),
    }


def per_instance(rows: list[dict], k: int) -> dict[str, dict]:
    byi: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        byi[str(r["instance_id"])].append(r)
    out = {}
    for inst, rr in byi.items():
        out[inst] = {"before": agg(rr[:k]), "after": agg(rr[-k:]), "n": len(rr)}
    return out


def md_table(insts: list[str], pi: dict, labels: dict, overall: dict) -> str:
    L = []
    L.append("| Query | Orphan sub-agents % | Own-evidence cites / sub-agent | Report reward R |")
    L.append("|---|---|---|---|")
    L.append("| | before → after | before → after | before → after |")
    for inst in insts:
        d = pi.get(inst)
        if not d:
            continue
        lab = labels.get(inst, f"query {inst}")
        b, a = d["before"], d["after"]
        L.append(
            f"| {lab} | {b['orphan_pct']:.0f}% → {a['orphan_pct']:.0f}% "
            f"| {b['child_cites_per_kid']:.2f} → {a['child_cites_per_kid']:.2f} "
            f"| {b['R_alive']:.2f} → {a['R_alive']:.2f} |"
        )
    ob, oa = overall["before"], overall["after"]
    L.append(
        f"| **All 32 queries (mean)** | **{ob['orphan_pct']:.0f}% → {oa['orphan_pct']:.0f}%** "
        f"| **{ob['child_cites_per_kid']:.2f} → {oa['child_cites_per_kid']:.2f}** "
        f"| **{ob['R_alive']:.2f} → {oa['R_alive']:.2f}** |"
    )
    return "\n".join(L)


def latex_table(insts: list[str], pi: dict, labels: dict, overall: dict, metrics_path: str) -> str:
    rows = []
    for inst in insts:
        d = pi.get(inst)
        if not d:
            continue
        lab = labels.get(inst, f"query {inst}").replace("&", "\\&").replace("%", "\\%")
        b, a = d["before"], d["after"]
        rows.append(
            f"  {lab} & {b['orphan_pct']:.0f}\\,\\textrightarrow\\,{a['orphan_pct']:.0f} "
            f"& {b['child_cites_per_kid']:.2f}\\,\\textrightarrow\\,{a['child_cites_per_kid']:.2f} "
            f"& {b['R_alive']:.2f}\\,\\textrightarrow\\,{a['R_alive']:.2f} \\\\"
        )
    ob, oa = overall["before"], overall["after"]
    overall_row = (
        f"  \\textbf{{All 32 queries (mean)}} "
        f"& \\textbf{{{ob['orphan_pct']:.0f}\\,\\textrightarrow\\,{oa['orphan_pct']:.0f}}} "
        f"& \\textbf{{{ob['child_cites_per_kid']:.2f}\\,\\textrightarrow\\,{oa['child_cites_per_kid']:.2f}}} "
        f"& \\textbf{{{ob['R_alive']:.2f}\\,\\textrightarrow\\,{oa['R_alive']:.2f}}} \\\\"
    )
    body = "\n".join(rows)
    return f"""% ============================================================================
% PLACEHOLDER TABLE -- training before/after on the SAME prompts.
% Generated by analysis/training_delta_examples.py from:
%   {metrics_path}
% This is the v1-clean PROVENANCE SMOKE run (relative numbers, frozen corpus).
% >>> REPLACE with the FINAL SFT+RL checkpoint's credit_metrics jsonl before submission. <<<
% Requires \\usepackage{{booktabs}} (already in msc_thesis.tex).
% ============================================================================
\\begin{{table}}[t]
\\centering
\\caption{{\\textbf{{Training improves Dr-RLM's raw outputs (placeholder, v1-clean smoke run).}}
Same prompts, first vs.\\ last rollouts. Sub-agents move from \\emph{{orphans}} (they surface
evidence but cite none of it) to \\emph{{grounded}} (they cite their own retrieved evidence),
while report quality $R$ holds. \\textit{{Final version uses the SFT+RL checkpoint.}}}}
\\label{{tab:training-delta}}
\\small
\\begin{{tabular}}{{@{{}}l ccc@{{}}}}
\\toprule
& Orphan & Own-evidence cites & Report \\\\
Query & sub-agents (\\%) & per sub-agent & reward $R$ \\\\
\\midrule
{body}
\\midrule
{overall_row}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def headline_json(rows: list[dict], inst: str) -> dict:
    rr = [r for r in rows if str(r["instance_id"]) == inst]
    if not rr:
        return {}
    def snap(t):
        ks = children(t)
        rt = root(t)
        owners = CITE_RE.findall(rt.get("ans_text", ""))
        return {
            "R": round(t.get("report_reward", 0.0), 2),
            "n_children": len(ks),
            "orphans": int(t.get("rer_metrics", {}).get("n_orphan_children", 0)),
            "child_cites": [int(n.get("n_cited", 0)) for n in ks],
            "root_cite_tags": len(owners),
            "root_cite_owners": owners[:8],
        }
    return {"instance_id": inst, "before": snap(rr[0]), "after": snap(rr[-1])}


CITE_SPAN_RE = re.compile(r'<cite id="[^"]+">.*?</cite>', re.S)


def dump_snippets(rows: list[dict], inst: str, k: int) -> str:
    """Verbatim before/after sub-agent text for the raw-trajectory listing.
    BEFORE = early orphan sub-agent (ungrounded assertions).
    AFTER  = late grounded sub-agents' own <cite>...</cite> spans (verbatim).
    Whitespace normalized; truncate by hand when pasting into fig-trajectory-snippets.tex."""
    rr = [r for r in rows if str(r["instance_id"]) == inst]
    if not rr:
        return f"(no rows for instance {inst})"
    early, late = rr[:k], rr[-k:]
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    L = [f"==== INSTANCE {inst} -- verbatim snippets for fig-trajectory-snippets.tex ====", ""]
    L.append("---- BEFORE: early orphan sub-agent(s) (ungrounded) ----")
    shown = 0
    for t in early:
        for n in children(t):
            if n.get("n_cited", 0) == 0 and n.get("ans_len", 0) > 200 and shown < 2:
                L.append(norm(n["ans_text"])[:600]); L.append(""); shown += 1
    if not shown:
        L.append("(no early orphan with text > 200 chars; widen --k)\n")
    L.append("---- AFTER: late grounded sub-agents' own <cite> spans (verbatim) ----")
    spans = 0
    for t in late:
        for n in children(t):
            for sp in CITE_SPAN_RE.findall(n.get("ans_text", "")):
                if spans < 8:
                    L.append(norm(sp)); L.append(""); spans += 1
    if not spans:
        L.append("(no visible <cite> spans in late sub-agent final answers; "
                 "cites may be ledger-only -- use full-rollout capture)")
    # ---- ROOT level (for fig-trajectory-snippets-root.tex) ----
    L += ["", "==== ROOT node (synthesis) ====", "",
          "---- BEFORE: early root report (head, ungrounded) ----"]
    rE = root(early[0])
    L.append(norm(rE.get("ans_text", ""))[:700] or "(empty)")
    L += ["", "---- AFTER: late root report's own <cite> spans (verbatim) ----"]
    rspans = 0
    for t in late:
        for sp in CITE_SPAN_RE.findall(root(t).get("ans_text", "")):
            if rspans < 8:
                L.append(norm(sp)); L.append(""); rspans += 1
    if not rspans:
        L.append("(late root report has no visible <cite> spans -- root may be untrained "
                 "under this config; pick a run where the root is trained)")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/provenance_smoke/credit_metrics_L3_v1clean.jsonl")
    ap.add_argument("--prompts", default="data/poc_corpus/train.parquet")
    ap.add_argument("--k", type=int, default=3, help="trees per instance to average at each end")
    ap.add_argument("--headline", default="27")
    ap.add_argument("--table-instances", default="27,9,4,7,20,3")
    ap.add_argument("--outdir", default="docs/paper_assets")
    ap.add_argument("--dump-snippets", metavar="INSTANCE_ID", default=None,
                    help="print verbatim before/after sub-agent text for one instance and exit")
    args = ap.parse_args()

    rows = load_metrics(args.metrics)
    if args.dump_snippets is not None:
        print(dump_snippets(rows, args.dump_snippets, args.k))
        return
    labels = load_labels(args.prompts)
    pi = per_instance(rows, args.k)
    insts = [s for s in args.table_instances.split(",") if s]
    overall = {"before": agg(rows[: len(rows) // 8]), "after": agg(rows[-len(rows) // 8:])}

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    md = md_table(insts, pi, labels, overall)
    (out / "training_delta_table.md").write_text(md + "\n")
    (out / "tbl-training-delta.tex").write_text(latex_table(insts, pi, labels, overall, args.metrics))
    hl = headline_json(rows, args.headline)
    (out / "headline_example.json").write_text(json.dumps(hl, indent=2) + "\n")

    print("== markdown table ==")
    print(md)
    print("\n== headline example (inst", args.headline, ") ==")
    print(json.dumps(hl, indent=2))
    print(f"\nwrote: {out}/training_delta_table.md, {out}/tbl-training-delta.tex, {out}/headline_example.json")


if __name__ == "__main__":
    main()
