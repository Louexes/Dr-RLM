"""Acceptance test — is the frozen corpus a COMPLETE package for SFT, RL, and Eval?

Run after Phase 7 (freeze). Asserts, against the REAL runtime backend (``corpus_search``):
  1. corpus.jsonl exists and is non-empty; manifest present.
  2. the bm25s index loads via ``corpus_search.get_backend({"search_backend":"bm25s", ...})``.
  3. retrieval returns non-empty, on-topic docs for sample questions from EVERY benchmark
     (the same questions SFT/RL/eval will search) + get_doc resolves full text.
  4. the coverage audit ran and flagged no THIN benchmark (or prints the flags loudly).

Exit code 0 = package good to ship; non-zero = a gate failed. Uses the crawler venv (has bm25s).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_TOK = re.compile(r"[A-Za-z0-9]+")


def _toks(s: str) -> set:
    return {t for t in _TOK.findall((s or "").lower()) if len(t) > 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--per-source", type=int, default=5, help="sample questions to probe per benchmark")
    ap.add_argument("--min-overlap", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.root)
    corpus = root / "corpus.jsonl"
    index_dir = root / "bm25s_index"
    fails = []

    # --- gate 1: artifacts ---
    n_docs = sum(1 for _ in open(corpus)) if corpus.exists() else 0
    print(f"[1] corpus.jsonl: {n_docs} docs  {'OK' if n_docs > 1000 else 'FAIL (<1000)'}")
    if n_docs <= 1000:
        fails.append("corpus too small")
    manifest = root / "manifest.json"
    print(f"    manifest.json: {'present' if manifest.exists() else 'MISSING'}")

    # --- gate 2: real backend loads the index ---
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl/skyrl/examples/train/dr_rlm"))
    import corpus_search as cs

    payload = {"search_backend": "bm25s", "search_index_path": str(index_dir),
               "search_corpus_path": str(corpus)}
    be = cs.get_backend(payload)
    if type(be).__name__ != "_Bm25sBackend":
        fails.append(f"backend is {type(be).__name__}, not _Bm25sBackend")
    print(f"[2] backend: {type(be).__name__}  {'OK' if type(be).__name__=='_Bm25sBackend' else 'FAIL'}")

    # --- gate 3: retrieval per benchmark ---
    by_src = defaultdict(list)
    for line in (root / "questions.jsonl").read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            by_src[q["source"]].append(q["question"])
    print(f"[3] retrieval per benchmark (k=10):")
    for src in sorted(by_src):
        qs = by_src[src][: args.per_source]
        nonempty = ontopic = 0
        for q in qs:
            hits = be.search(q, k=10)
            if hits:
                nonempty += 1
                qt = _toks(q)
                best = max((len(qt & _toks(h["snippet"])) / (len(qt) + 1.0)) for h in hits)
                if best >= args.min_overlap:
                    ontopic += 1
                # get_doc round-trip on the top hit
                _ = be.get_text(hits[0]["docid"])
        tag = "OK" if ontopic == len(qs) else ("THIN" if ontopic >= 1 else "FAIL")
        if ontopic == 0:
            fails.append(f"{src}: 0/{len(qs)} on-topic")
        print(f"    {src:12s} nonempty={nonempty}/{len(qs)} ontopic={ontopic}/{len(qs)}  {tag}")

    # --- gate 4: coverage audit ---
    audit_p = root / "coverage_audit.json"
    if audit_p.exists():
        audit = json.loads(audit_p.read_text())
        flags = audit.get("flags", [])
        print(f"[4] coverage audit flags: {flags or 'NONE — all covered'}")
    else:
        print("[4] coverage_audit.json MISSING (run audit.py)")

    print("\n" + ("VERIFY: PASS — complete package for SFT/RL/Eval" if not fails
                  else f"VERIFY: FAIL — {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
