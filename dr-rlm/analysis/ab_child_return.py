#!/usr/bin/env python3
"""Compare the child->parent return-contract A/B: PROSE vs STRUCTURED.

Reads the two arms produced by `agent/ab_child_return.sh`:
  <root>/prose/<bench>.jsonl       + <root>/prose/drb_eval/...
  <root>/structured/<bench>.jsonl  + <root>/structured/drb_eval/...

Hypothesis (neural_avb): handing the orchestrator STRUCTURED claim+citation atoms it merges
in code — instead of FREE-TEXT mini-reports it must reconcile — preserves provenance and
improves citation-attribution fidelity, at matched token cost.

Headline (arm-level, from the grader sidecars): DRB FACT `valid_rate` (attribution fidelity)
and RACE `overall_score` (quality). Plus PAIRED per-item proxies present in every row
(cited-id count, completion tokens, #sub-agents, surfaced snippets) with a bootstrap CI on
the structured-minus-prose delta, so we can confirm the token-cost parity the claim assumes.

Defensive: missing graders / fields degrade to None, never crash. Bootstrap is seeded
(deterministic) so reruns match.

Usage:
  python analysis/ab_child_return.py --root dr-rlm/runs/ab_child_return --bench deep_research_bench
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------- IO -----------------------------
def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _parse_kv_floats(path: Path) -> Dict[str, float]:
    """Plain 'key: value' metrics file (DRB race_result.txt / fact_result.txt)."""
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z][\w ()./-]*?)\s*[:=]\s*([0-9]+\.?[0-9]*)\s*$", line)
        if m:
            try:
                out[m.group(1).strip().lower().replace(" ", "_")] = float(m.group(2))
            except ValueError:
                pass
    return out


def parse_quality(arm_dir: Path, bench: str) -> Dict[str, float]:
    """Grader sidecars: DRB -> fact_* + race_*; researchqa -> coverage."""
    out: Dict[str, float] = {}
    if bench == "researchqa":
        for name in ("researchqa_eval_results.json",):
            p = arm_dir / name
            if p.exists():
                try:
                    j = json.loads(p.read_text())
                    if isinstance(j.get("score"), (int, float)):
                        out["coverage"] = float(j["score"])
                    m = j.get("metrics") or {}
                    if isinstance(m.get("coverage"), (int, float)):
                        out["coverage"] = float(m["coverage"])
                except Exception:
                    pass
    else:
        for rf in (arm_dir / "drb_eval" / "race").glob("*/race_result.txt"):
            for k, v in _parse_kv_floats(rf).items():
                out[f"race_{k}"] = v
        for ff in (arm_dir / "drb_eval" / "fact").glob("*/fact_result.txt"):
            for k, v in _parse_kv_floats(ff).items():
                out[f"fact_{k}"] = v
    return out


# ----------------------------- per-item proxies -----------------------------
def _completion_tokens(row: dict, tok=None) -> Optional[float]:
    ft = row.get("full_traces") or {}
    for k in ("completion_tokens", "total_tokens"):
        if isinstance(ft.get(k), (int, float)) and ft[k]:
            return float(ft[k])
    text = row.get("final_response") or ""
    if not text:
        return 0.0
    if tok is not None:
        try:
            return float(len(tok.encode(text)))
        except Exception:
            pass
    return float(len(text.split()))


def per_item(rows: List[dict], tok=None) -> Dict[str, Dict[str, float]]:
    """Map example_id -> per-item metrics (None where a field is absent/undefined)."""
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        eid = str(r.get("example_id"))
        aod = r.get("additional_output_data") or {}
        rec = aod.get("recursion") or {}
        prov = aod.get("provenance") or {}   # absent on pre-survival-metric runs -> {}
        resp = r.get("final_response") or ""
        sr = prov.get("child_citation_survival_rate", None)
        out[eid] = {
            # headline attribution metric: fraction of child-cited evidence kept in the root report
            "survival_rate": (float(sr) if isinstance(sr, (int, float)) else None),
            "n_child_cited": float(prov.get("n_child_cited", 0) or 0),
            "n_child_orphan_cited": float(prov.get("n_child_orphan_cited", 0) or 0),
            "n_cited": float(aod.get("n_cited_ids", 0) or 0),
            "tokens": _completion_tokens(r, tok) or 0.0,
            "n_subagents": float(rec.get("n_subagents", 0) or 0),
            "max_depth": float(rec.get("max_depth_reached", 0) or 0),
            "n_surfaced": float(aod.get("n_surfaced_snippets", 0) or 0),
            "len": float(len(resp)),
            "has_cite": 1.0 if "<cite" in resp else 0.0,
        }
    return out


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def _bootstrap_ci(deltas: List[float], n: int = 5000, seed: int = 0):
    """Percentile bootstrap 95% CI of the mean paired delta. Deterministic (seeded)."""
    deltas = [d for d in deltas if d is not None]
    if len(deltas) < 2:
        return None
    import random
    rng = random.Random(seed)
    k = len(deltas)
    means = []
    for _ in range(n):
        s = sum(deltas[rng.randrange(k)] for _ in range(k)) / k
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (st.mean(deltas), lo, hi)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding prose/ and structured/ arm subdirs")
    ap.add_argument("--bench", default="deep_research_bench")
    ap.add_argument("--tokenizer", default=None, help="HF id for an accurate token count (optional)")
    ap.add_argument("--out", default=None, help="write the comparison JSON here")
    args = ap.parse_args()

    root = Path(args.root)
    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            print(f"[warn] tokenizer load failed ({e}); falling back to word/char counts")

    arms = {}
    for mode in ("prose", "structured"):
        adir = root / mode
        rows = read_jsonl(adir / f"{args.bench}.jsonl")
        arms[mode] = {
            "n_rows": len(rows),
            "quality": parse_quality(adir, args.bench),
            "items": per_item(rows, tok),
        }

    # survival_rate first: it's the headline attribution metric (FACT is unusable with our ids)
    PROXIES = ["survival_rate", "n_child_cited", "n_child_orphan_cited", "n_cited",
               "tokens", "n_subagents", "max_depth", "n_surfaced", "len", "has_cite"]
    summary = {"bench": args.bench, "arms": {}, "paired_delta": {}, "quality_delta": {}}

    # arm-level means
    for mode in ("prose", "structured"):
        items = arms[mode]["items"]
        summary["arms"][mode] = {
            "n_rows": arms[mode]["n_rows"],
            "quality": arms[mode]["quality"],
            "means": {p: _mean([v[p] for v in items.values()]) for p in PROXIES},
        }

    # paired per-item deltas (structured - prose) on the shared example_ids. Pairs where either
    # arm's value is None (e.g. survival_rate undefined: no child cited) are skipped per-metric.
    shared = sorted(set(arms["prose"]["items"]) & set(arms["structured"]["items"]))
    summary["n_paired"] = len(shared)
    for p in PROXIES:
        deltas = []
        for e in shared:
            sv, pv = arms["structured"]["items"][e][p], arms["prose"]["items"][e][p]
            if sv is None or pv is None:
                continue
            deltas.append(sv - pv)
        ci = _bootstrap_ci(deltas)
        summary["paired_delta"][p] = (
            {"mean": round(ci[0], 4), "ci95": [round(ci[1], 4), round(ci[2], 4)], "n": len(deltas)}
            if ci else {"mean": None, "ci95": None, "n": len(deltas)}
        )

    # arm-level quality deltas (no per-item grader breakdown available from sidecars)
    qkeys = sorted(set(arms["prose"]["quality"]) | set(arms["structured"]["quality"]))
    for k in qkeys:
        pv = arms["prose"]["quality"].get(k)
        sv = arms["structured"]["quality"].get(k)
        if isinstance(pv, (int, float)) and isinstance(sv, (int, float)):
            summary["quality_delta"][k] = round(sv - pv, 4)

    # ---- print ----
    print(f"\n=== child_return A/B — {args.bench} ===")
    print(f"paired items: {summary['n_paired']}  (prose={arms['prose']['n_rows']} rows, "
          f"structured={arms['structured']['n_rows']} rows)")
    print("\nGRADER (arm-level)            prose     structured   delta(struct-prose)")
    headline = [k for k in qkeys if k in ("fact_valid_rate", "race_overall_score", "coverage")] or qkeys
    for k in headline:
        pv = arms["prose"]["quality"].get(k)
        sv = arms["structured"]["quality"].get(k)
        d = summary["quality_delta"].get(k)
        print(f"  {k:<26} {str(pv):>8}   {str(sv):>10}   {('' if d is None else f'{d:+.4f}'):>10}")

    # arm-level survival means (the headline; FACT is unusable with our internal ids)
    print("\nHEADLINE — child→root citation SURVIVAL (mean of per-item rate)")
    for mode in ("prose", "structured"):
        m = summary["arms"][mode]["means"].get("survival_rate")
        nc = summary["arms"][mode]["means"].get("n_child_cited")
        print(f"  {mode:<11} survival_rate={('n/a' if m is None else f'{m:.3f}')}   "
              f"(mean child-cited ids/item={('n/a' if nc is None else f'{nc:.1f}')})")

    print("\nPAIRED (struct - prose, 95% bootstrap CI; n = pairs with the metric defined)")
    for p in PROXIES:
        d = summary["paired_delta"][p]
        if d and d.get("mean") is not None:
            print(f"  {p:<20} {d['mean']:+.3f}   CI[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  (n={d['n']})")
        else:
            print(f"  {p:<20} n/a  (n={d.get('n', 0) if d else 0})")
    print("\nNote: survival_rate ↑ is the attribution win the hypothesis predicts; token CI should "
          "straddle 0 for 'equal cost'. (survival needs the provenance field — rerun after the metric was added.)\n")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"[ab] wrote {args.out}")


if __name__ == "__main__":
    main()
