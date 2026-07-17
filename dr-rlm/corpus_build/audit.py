"""Phase 6 — coverage audit (THE GATE, DATA_PREP_PLAN §10).

With DDG's thinner coverage this is mandatory: for each benchmark we sample its questions, run
the *real* BM25 retrieval against the frozen index, and check that top-k comes back **non-empty
and on-topic**. Silent thin coverage = invalid eval, so we report a per-source hit-rate and flag
any benchmark below threshold for re-crawl (more queries/depth, or a bulk-corpus pull).

"on-topic" proxy = token overlap between the question and the best of its top-k docs (no judge
needed for a retrieval-starvation check). Run on the login node; needs only the bm25s index +
questions.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_TOK = re.compile(r"[A-Za-z0-9]+")


def _toks(s: str) -> set:
    return {t for t in _TOK.findall((s or "").lower()) if len(t) > 2}


def main() -> None:
    import bm25s
    import Stemmer

    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sample", type=int, default=60, help="questions sampled per source")
    ap.add_argument("--min-overlap", type=float, default=0.06, help="on-topic threshold (token overlap)")
    ap.add_argument("--hitrate-gate", type=float, default=0.85, help="flag sources below this on-topic rate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    index_dir = args.index_dir or str(root / "bm25s_index")
    qpath = Path(args.questions) if args.questions else root / "questions.jsonl"
    out = Path(args.out) if args.out else root / "coverage_audit.json"

    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    ndocs = len(getattr(retriever, "corpus", []) or [])

    by_src: Dict[str, List[dict]] = defaultdict(list)
    for line in qpath.read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            by_src[q["source"]].append(q)

    rnd = random.Random(args.seed)
    report = {"index_dir": index_dir, "n_docs": ndocs, "k": args.k,
              "min_overlap": args.min_overlap, "per_source": {}, "flags": []}
    print(f"[audit] index={index_dir} docs={ndocs} k={args.k}\n")
    print(f"{'source':12s} {'n':>4s} {'nonempty':>9s} {'ontopic':>8s} {'mean_ov':>8s} {'mean_top1':>10s}")
    for src in sorted(by_src):
        qs = by_src[src]
        sample = qs if len(qs) <= args.sample else [qs[i] for i in sorted(rnd.sample(range(len(qs)), args.sample))]
        n = len(sample)
        nonempty = ontopic = 0
        sum_ov = sum_top1 = 0.0
        for q in sample:
            qt = _toks(q["question"])
            toks = bm25s.tokenize(q["question"], stopwords="en", stemmer=stemmer, show_progress=False)
            docs, scores = retriever.retrieve(toks, k=min(args.k, ndocs), show_progress=False)
            ncols = docs.shape[1] if hasattr(docs, "shape") else 0
            if ncols > 0:
                nonempty += 1
                sum_top1 += float(scores[0, 0])
                best = 0.0
                for i in range(ncols):
                    d = docs[0, i]
                    if isinstance(d, dict):
                        dt = _toks(d.get("contents", ""))
                        ov = len(qt & dt) / (len(qt) + 1.0)
                        best = max(best, ov)
                sum_ov += best
                if best >= args.min_overlap:
                    ontopic += 1
        ne_rate = nonempty / n if n else 0.0
        ot_rate = ontopic / n if n else 0.0
        mean_ov = sum_ov / n if n else 0.0
        mean_top1 = sum_top1 / nonempty if nonempty else 0.0
        report["per_source"][src] = {
            "n": n, "nonempty_rate": round(ne_rate, 3), "ontopic_rate": round(ot_rate, 3),
            "mean_overlap": round(mean_ov, 3), "mean_top1_score": round(mean_top1, 2),
        }
        flag = "  <-- THIN" if ot_rate < args.hitrate_gate else ""
        if ot_rate < args.hitrate_gate:
            report["flags"].append(src)
        print(f"{src:12s} {n:4d} {ne_rate:9.2f} {ot_rate:8.2f} {mean_ov:8.3f} {mean_top1:10.2f}{flag}")

    out.write_text(json.dumps(report, indent=2))
    print(f"\n[audit] flags (thin, re-crawl): {report['flags'] or 'NONE — all benchmarks covered'}")
    print(f"[audit] report -> {out}")


if __name__ == "__main__":
    main()
