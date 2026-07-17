#!/usr/bin/env python3
"""Select a decompose-shaped subset of dr-tulu-rl-data for the provenance-credit POC.

We want prompts where (a) the report reward DECOMPOSES into several rubric criteria
(so per-criterion credit can differentiate children) and (b) the question is breadth-y
enough that a root would plausibly fan out into sub-agents. Selection:

  * keep rows with >= MIN_RUBRICS rubric criteria (multi-criterion reward);
  * score "decompose-shape" by facet cues in the query (compare/contrast, list/types,
    multiple wh-questions, "and"-joined sub-asks, "across/various aspects", enumerations);
  * rank by (decompose_score, n_rubrics) and take the top N.

Emits rows in the infer_driver `drtulu_rl` loader schema:
  {id, problem, query, rubrics:[{description,title,weight}], dataset, question_type, source}

Usage:
  python analysis/build_drtulu_rl_subset.py --n 40 --min-rubrics 3 \
      --out data/subsets/drtulu_rl_decompose40.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_FACET_PATTERNS = [
    r"\bcompare\b", r"\bcontrast\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bdifference[s]?\b", r"\bsimilar", r"\btrade[- ]?off",
    r"\btypes? of\b", r"\bkinds? of\b", r"\bcategor", r"\bvarious\b",
    r"\bdifferent\b", r"\bseveral\b", r"\bmultiple\b", r"\baspects?\b",
    r"\bfactors?\b", r"\bdimensions?\b", r"\bacross\b", r"\brespectively\b",
    r"\beach\b", r"\blist\b", r"\benumerate\b", r"\bbreak\s*down\b",
    r"\badvantages? and disadvantages?\b", r"\bpros and cons\b",
    r"\bcauses? and\b", r"\bhow .* and how\b",
]
_FACET_RE = [re.compile(p, re.I) for p in _FACET_PATTERNS]


def decompose_score(q: str) -> float:
    s = 0.0
    s += sum(1 for r in _FACET_RE if r.search(q))                  # facet cue hits
    s += 1.0 * min(q.count("?") - 1, 4) if q.count("?") > 1 else 0  # multiple questions (capped)
    s += 0.3 * min(q.lower().count(" and "), 4)                    # and-joined asks (capped)
    # numbered / bulleted sub-asks (capped — long pasted docs game this)
    s += 1.0 * min(len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", q)), 5)
    return s


def load_rows():
    from datasets import load_dataset
    ds = load_dataset("rl-research/dr-tulu-rl-data", split="train")
    for i, r in enumerate(ds):
        try:
            gt = json.loads(r["ground_truth"])
        except Exception:
            continue
        query = gt.get("query") or ""
        rubrics = []
        for rb in gt.get("rubrics", []) or []:
            try:
                w = float(rb.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            rubrics.append({
                "description": rb.get("description", ""),
                "title": rb.get("title", ""),
                "weight": w,
            })
        yield {
            "id": i,
            "problem": query,
            "query": query,
            "rubrics": rubrics,
            "dataset": r.get("dataset"),
            "question_type": r.get("question_type"),
            "source": r.get("source"),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-rubrics", type=int, default=3)
    ap.add_argument("--max-rubrics", type=int, default=12,
                    help="drop pathological taxonomy rows with too many criteria")
    ap.add_argument("--min-query-chars", type=int, default=60)
    ap.add_argument("--max-query-chars", type=int, default=1200,
                    help="drop prompts with a large pasted document (not web-research-decomposable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cand = []
    for row in load_rows():
        nr = len(row["rubrics"])
        if nr < args.min_rubrics or nr > args.max_rubrics:
            continue
        if not (args.min_query_chars <= len(row["problem"]) <= args.max_query_chars):
            continue
        row["_decompose_score"] = decompose_score(row["problem"])
        row["_n_rubrics"] = nr
        cand.append(row)

    # rank by decompose-shape then criterion count; stable + deterministic
    cand.sort(key=lambda r: (r["_decompose_score"], r["_n_rubrics"], -r["id"]), reverse=True)
    picked = cand[: args.n]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")

    print(f"[subset] candidates={len(cand)} picked={len(picked)} -> {outp}")
    import statistics as st
    print(f"[subset] mean rubrics={st.mean([r['_n_rubrics'] for r in picked]):.1f} "
          f"mean decompose_score={st.mean([r['_decompose_score'] for r in picked]):.1f}")
    print("[subset] sample queries:")
    for r in picked[:6]:
        print(f"   id={r['id']} nr={r['_n_rubrics']} ds={r['_decompose_score']:.1f} :: {r['problem'][:100]}")


if __name__ == "__main__":
    main()
