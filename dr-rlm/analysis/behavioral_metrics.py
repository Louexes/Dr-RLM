#!/usr/bin/env python3
"""Behavioral / efficiency metrics over eval-run outputs (Untrained -> SFT -> RL).

PURPOSE
-------
The thesis has a *quality* stage-wise table (tab:progression) but the behavioral /
efficiency metrics only appear one-off (tab:efficiency, substrate comparison) and in
prose. This script produces the missing **stage-wise behavioral table**: how tool use,
tokens, recursion, finalization and provenance-flow change from Untrained -> SFT -> RL,
on the SAME eval benchmark.

Everything here is computed POST-HOC from the eval `.jsonl` outputs already written by
`agent/infer_driver.py` (DR-RLM, recursive REPL) and the DR-Tulu ReAct harness. Nothing
needs to be logged at run time. Point it at each stage's eval output and it emits:
  - a JSON summary (all aggregates, per run),
  - a markdown table (metrics as rows, runs/stages as columns),
  - a LaTeX booktabs table (drop-in \input, placeholder-styled like training_delta).

HARNESS AUTO-DETECT (per record):
  * DR-RLM  = `additional_output_data.recursion` present (or `driver` set).
  * DR-Tulu = `additional_output_data.searched_links` / `total_tool_calls`, no recursion.
A run may be homogeneous (all DR-RLM, all DR-Tulu) or mixed; each record is classified
independently and per-harness metrics are averaged only over the records that carry them.

METRIC PROVENANCE (field semantics verified against agent/infer_driver.py L197-202):
  * n_child_cited            = distinct evidence ids the sub-agents cited.
  * n_child_cited_survived   = ...of those, how many reached the ROOT report.
  * child_citation_survival_rate = survived / n_child_cited (delegation "stickiness").
  * n_child_orphan_cited     = child-cited ids the orchestrator dropped (child_cited - root_cited).
  * n_root_cited             = distinct ids the root report cited.
  * n_root_own_cited         = root's own searches cited (root_cited - child_cited).
  So provenance_share = n_child_cited_survived / n_root_cited = fraction of the report's
  citations that originated in a delegated sub-agent (the "delegation pays off" metric).

KNOWN LIMITATION — peak per-agent context:
  The RLM decoupling headline ("bounded per-agent peak context while searching 20+ docs")
  needs per-node peak token counts. Trajectory files (`--log-dir` -> trajectory_path) record
  per-turn text but NOT token counts, so a faithful peak-per-agent number requires
  re-tokenization / the dedicated vLLM-usage probe (see docs/PROGRESS.md ~L212-263). This
  script therefore reports what IS in the records: DR-Tulu peak single-context tokens (ReAct
  is one growing context, so max total_tokens == the peak) and DR-RLM whole-tree total tokens
  (a compute measure, NOT a per-agent peak). Do not present tree-total as "peak context".

Usage (stage-wise — column order = arg order):
  python analysis/behavioral_metrics.py \
      --run Untrained:runs/drrlm_untrained_thinkoff/researchqa.jsonl \
      --run SFT:runs/drrlm_sft/researchqa.jsonl \
      --run RL:runs/drrlm_rl/researchqa.jsonl \
      --outdir docs/paper_assets

  # single run, or a DR-Tulu vs DR-RLM substrate comparison, work the same:
  python analysis/behavioral_metrics.py \
      --run "DR-Tulu:runs/drtulu_untrained_thinkon/researchqa.jsonl" \
      --run "DR-RLM:runs/drrlm_untrained_thinkoff/researchqa.jsonl"

>>> RUN THIS OVER ALL RUNS ONCE THE SFT + RL EVAL OUTPUTS LAND — see docs/BEHAVIORAL_METRICS_TODO.md <<<
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

CITE_RE = re.compile(r'<cite id="?([^">]+)"?>')


# --------------------------------------------------------------------------- IO
def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def detect_harness(rec: dict) -> str:
    aod = rec.get("additional_output_data") or {}
    if "recursion" in aod or aod.get("driver"):
        return "dr_rlm"
    if "searched_links" in aod or "total_tool_calls" in aod:
        return "dr_tulu"
    # fall back: recursion present anywhere -> rlm; else tulu
    return "dr_rlm" if "recursion" in aod else "dr_tulu"


def walk_max_tokens(o) -> int:
    """Max `total_tokens` anywhere in a (possibly nested) full_traces object.
    For DR-Tulu ReAct (single growing context) this is the peak single-context size."""
    best = 0
    stack = [o]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            v = cur.get("total_tokens")
            if isinstance(v, int):
                best = max(best, v)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return best


# ------------------------------------------------------------- per-record metrics
def n_cites(text: str) -> int:
    """Distinct grounded <cite id> tags in a report (harness-agnostic)."""
    return len(set(CITE_RE.findall(text or "")))


def record_metrics(rec: dict) -> dict:
    """Everything computable from a single eval record. Missing per-harness fields -> None."""
    harness = detect_harness(rec)
    aod = rec.get("additional_output_data") or {}
    ft = rec.get("full_traces") or {}
    resp = rec.get("final_response") or ""

    m: dict = {
        "harness": harness,
        "finalized": 1.0 if resp.strip() else 0.0,      # proxy: non-empty submitted report
        "report_words": len(resp.split()),
        "report_cites": n_cites(resp),
        "wall_clock_s": aod.get("wall_clock_s"),
    }

    if harness == "dr_tulu":
        m["tool_calls"] = aod.get("total_tool_calls")
        ftc = aod.get("total_failed_tool_calls")
        ttc = aod.get("total_tool_calls")
        m["failed_tool_rate"] = (ftc / ttc) if (ftc is not None and ttc) else None
        m["evidence_surfaced"] = len(aod.get("searched_links") or [])   # unique URLs
        m["peak_context_tok"] = walk_max_tokens(ft)                     # ReAct peak == max
        m["tree_total_tok"] = walk_max_tokens(ft)                       # single agent
        # recursion / provenance: N/A for flat ReAct
    else:  # dr_rlm
        # NOTE: DR-RLM full_traces.tool_call_count mirrors the surfaced-snippet count
        # (== n_surfaced_snippets), NOT a search-invocation count -> NOT comparable to
        # DR-Tulu's total_tool_calls. A true per-node search-call count needs trajectory
        # parsing (count search() calls across nodes); left None to avoid a false compare.
        m["tool_calls"] = None
        m["failed_tool_rate"] = None                                    # not recorded per-tree
        m["evidence_surfaced"] = aod.get("n_surfaced_snippets")         # snippets across tree
        m["peak_context_tok"] = None                                    # needs probe (see header)
        m["tree_total_tok"] = ft.get("total_tokens")                    # tree compute, NOT peak
        m["prompt_tok"] = ft.get("prompt_tokens")
        m["completion_tok"] = ft.get("completion_tokens")

        rec_r = aod.get("recursion") or {}
        n_sub = rec_r.get("n_subagents")
        m["n_subagents"] = n_sub
        m["recursed"] = 1.0 if (n_sub or 0) > 0 else 0.0
        m["max_depth"] = rec_r.get("max_depth_reached")
        m["n_nodes"] = rec_r.get("n_nodes")
        by_depth = rec_r.get("subagents_by_depth") or {}
        m["tree_width"] = max((v for v in by_depth.values()), default=0)

        prov = aod.get("provenance") or {}
        m["child_grounding"] = prov.get("n_child_cited")               # distinct ids kids cited
        m["child_survived"] = prov.get("n_child_cited_survived")       # ...that reached the report
        m["survival_rate"] = prov.get("child_citation_survival_rate")
        m["orphan_child_cites"] = prov.get("n_child_orphan_cited")     # kid-cited, root dropped
        m["root_cites"] = prov.get("n_root_cited")
        m["root_own_cites"] = prov.get("n_root_own_cited")
        rc = prov.get("n_root_cited")
        cs = prov.get("n_child_cited_survived")
        m["provenance_share"] = (cs / rc) if (rc and cs is not None) else None
    return m


# ----------------------------------------------------------------- aggregation
# metric key -> (display label, format spec, section)
SPEC = [
    ("__general__", "General", None, None),
    ("n", "records", "{:.0f}", "general"),
    ("finalized", "finalized (non-empty report) %", "{:.0%}", "general"),
    ("report_words", "report length (words)", "{:.0f}", "general"),
    ("wall_clock_s", "latency (s)", "{:.1f}", "general"),
    ("__retrieval__", "Retrieval & tool use", None, None),
    ("tool_calls", "tool calls / query", "{:.1f}", "retrieval"),
    ("failed_tool_rate", "failed tool-call rate", "{:.0%}", "retrieval"),
    ("evidence_surfaced", "evidence surfaced (links|snippets)", "{:.1f}", "retrieval"),
    ("report_cites", "grounded cites / report", "{:.1f}", "retrieval"),
    ("__tokens__", "Compute (tokens)", None, None),
    ("peak_context_tok", "DR-Tulu peak context (tok)", "{:.0f}", "tokens"),
    ("tree_total_tok", "total tokens (tree|ReAct)", "{:.0f}", "tokens"),
    ("__recursion__", "Recursion (DR-RLM)", None, None),
    ("recursed", "recursion rate %", "{:.0%}", "recursion"),
    ("n_subagents", "sub-agents / query", "{:.2f}", "recursion"),
    ("max_depth", "recursion depth (mean)", "{:.2f}", "recursion"),
    ("tree_width", "tree width (mean breadth)", "{:.2f}", "recursion"),
    ("n_nodes", "nodes / tree", "{:.2f}", "recursion"),
    ("__provenance__", "Provenance flow (DR-RLM)", None, None),
    ("child_grounding", "sub-agent cites / query", "{:.2f}", "provenance"),
    ("survival_rate", "child->root survival rate", "{:.2f}", "provenance"),
    ("provenance_share", "report cites from sub-agents %", "{:.0%}", "provenance"),
    ("root_cites", "root report cites", "{:.2f}", "provenance"),
    ("root_own_cites", "root's own cites", "{:.2f}", "provenance"),
    ("orphan_child_cites", "orphaned child cites (dropped)", "{:.2f}", "provenance"),
]
METRIC_KEYS = [k for k, *_ in SPEC if not k.startswith("__")]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.fmean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def aggregate(recs: list[dict]) -> dict:
    per = [record_metrics(r) for r in recs]
    n = len(per)
    n_rlm = sum(1 for p in per if p["harness"] == "dr_rlm")
    n_tulu = n - n_rlm
    out = {"n": float(n), "_n_dr_rlm": n_rlm, "_n_dr_tulu": n_tulu,
           "_harness": "dr_rlm" if n_rlm >= n_tulu else "dr_tulu"}
    for key in METRIC_KEYS:
        if key == "n":
            continue
        vals = [p.get(key) for p in per]
        out[key] = mean(vals)
        out[f"{key}__median"] = median(vals)
    return out


# --------------------------------------------------------------------- rendering
def fmt(val, spec):
    if val is None:
        return "--"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def md_table(runs: list[tuple[str, dict]]) -> str:
    names = [n for n, _ in runs]
    L = ["| Metric | " + " | ".join(names) + " |",
         "|" + "---|" * (len(names) + 1)]
    for key, label, spec, _sec in SPEC:
        if key.startswith("__"):                       # section header row
            L.append(f"| **{label}** |" + " |" * len(names))
            continue
        cells = [fmt(agg.get(key), spec) for _, agg in runs]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def latex_table(runs: list[tuple[str, dict]]) -> str:
    names = [n.replace("_", r"\_").replace("&", r"\&") for n, _ in runs]
    cols = "l" + "r" * len(names)
    header = " & " + " & ".join(names) + r" \\"
    body = []
    for key, label, spec, _sec in SPEC:
        lab = label.replace("%", r"\%").replace("&", r"\&").replace("->", r"$\rightarrow$")
        lab = lab.replace("|", r"$\vert$")
        if key.startswith("__"):
            body.append(r"\midrule")
            body.append(rf"\multicolumn{{{len(names)+1}}}{{@{{}}l}}{{\textit{{{lab}}}}} \\")
            continue
        cells = [fmt(agg.get(key), spec).replace("%", r"\%") for _, agg in runs]
        body.append(f"  {lab} & " + " & ".join(cells) + r" \\")
    body_s = "\n".join(body)
    return rf"""% ============================================================================
% PLACEHOLDER — stage-wise behavioral / efficiency table.
% Generated by analysis/behavioral_metrics.py (post-hoc over eval .jsonl outputs).
% >>> Re-run over the FINAL Untrained / SFT / RL eval outputs before submission. <<<
% See docs/BEHAVIORAL_METRICS_TODO.md. Requires \usepackage{{booktabs}}.
% ============================================================================
\begin{{table}}[t]
\centering
\caption{{\textbf{{How Dr-RLM's behavior changes across training (Untrained $\rightarrow$ SFT $\rightarrow$ RL).}}
Behavioral / efficiency metrics on the same eval benchmark, complementing the quality
progression (Table~\ref{{tab:progression}}). Provenance-flow rows are DR-RLM-only.
DR-Tulu peak context is the ReAct single-context peak; the DR-RLM per-agent peak-context
headline comes from the dedicated decoupling probe, not this table.}}
\label{{tab:behavior-progression}}
\small
\begin{{tabular}}{{@{{}}{cols}@{{}}}}
\toprule
{header}
{body_s}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ---------------------------------------------------------------------- driver
def parse_run(spec: str) -> tuple[str, str]:
    """`Name:path` -> (Name, path). Path may contain ':' after the first."""
    if ":" not in spec:
        p = Path(spec)
        return p.parent.name or p.stem, spec
    name, path = spec.split(":", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME:PATH",
                    help="a run to aggregate; repeat for a stage-wise table (column order = arg order)")
    ap.add_argument("--outdir", default="docs/paper_assets")
    ap.add_argument("--stem", default="behavioral_metrics",
                    help="output filename stem (.json/.md) ; the .tex is tbl-behavior-progression.tex")
    ap.add_argument("--paired", action="store_true",
                    help="restrict EVERY run to the intersection of example_ids present in ALL runs, "
                         "so the columns are paired over the same questions (use for the untrained-vs-RL "
                         "cost frontier; else a policy that finalizes a different item mix skews the compare).")
    args = ap.parse_args()

    # Load first so --paired can intersect example_ids across all runs before aggregating.
    loaded: list[tuple[str, str, list[dict]]] = []
    for spec in args.run:
        name, path = parse_run(spec)
        if not Path(path).exists():
            print(f"[warn] skipping missing run '{name}': {path}")
            continue
        loaded.append((name, path, load_jsonl(path)))

    if args.paired and len(loaded) > 1:
        idsets = [set(str(r.get("example_id")) for r in recs) for _, _, recs in loaded]
        shared = set.intersection(*idsets)
        print(f"[paired] shared example_ids across {len(loaded)} runs: {len(shared)} "
              f"(per-run: {[len(s) for s in idsets]})")
        loaded = [(n, p, [r for r in recs if str(r.get("example_id")) in shared])
                  for n, p, recs in loaded]

    runs: list[tuple[str, dict]] = []
    full: dict = {}
    for name, path, recs in loaded:
        agg = aggregate(recs)
        runs.append((name, agg))
        full[name] = {"path": path, "paired": bool(args.paired), **agg}
        print(f"[{name}] n={len(recs)} "
              f"(dr_rlm={agg['_n_dr_rlm']}, dr_tulu={agg['_n_dr_tulu']}) from {path}")

    if not runs:
        print("[error] no valid runs; nothing to write.")
        return

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    md = md_table(runs)
    (out / f"{args.stem}.json").write_text(json.dumps(full, indent=2) + "\n")
    (out / f"{args.stem}.md").write_text(md + "\n")
    (out / "tbl-behavior-progression.tex").write_text(latex_table(runs))

    print("\n== behavioral metrics ==")
    print(md)
    print(f"\nwrote: {out}/{args.stem}.json, {out}/{args.stem}.md, {out}/tbl-behavior-progression.tex")


if __name__ == "__main__":
    main()
