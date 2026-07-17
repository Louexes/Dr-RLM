"""Phase 1-2 — DISCOVERY: decompose each seed question -> DDG queries -> ranked hits.

This is the **IP-rate-limited** half of the crawl, so it runs as a *small* number of paced
processes sharded across nodes (distinct IPs), NOT massively parallel. It does no page reading
(that's ``fetch.py``, which is not IP-limited and uses all cores) — decoupling the two is what
lets each scale on its own limit and also sidesteps lxml's thread-unsafety.

Per question it writes a record ``{qid, source, question, queries, hits[], read_urls[]}`` where
``read_urls`` is the relevance-gated top-``read_k`` URLs to later fetch full-text — the single
source of truth for what fetch/normalize will read.

Shardable (``sha1(qid) % num_shards == shard``), resumable (``shard_{i}.done``), cached
(discovery cache shared across shards). See DATA_PREP_PLAN §6/§7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discovery import Discovery  # noqa: E402
from queries import make_queries  # noqa: E402

_TOK = re.compile(r"[A-Za-z0-9]+")


def _toks(s: str) -> set:
    return {t for t in _TOK.findall((s or "").lower()) if len(t) > 2}


def _shard_of(qid: str, n: int) -> int:
    return int(hashlib.sha1(qid.encode()).hexdigest(), 16) % n


def _relevance(qtoks: set, title: str, snippet: str) -> float:
    ht = _toks(title) | _toks(snippet)
    if not ht:
        return 0.0
    return len(qtoks & ht) / (len(qtoks) + 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--questions", default=None, help="default <root>/questions.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=8, help="discovery results requested per query")
    ap.add_argument("--read-k", type=int, default=6, help="full-text URLs to mark for reading")
    ap.add_argument("--min-rel", type=float, default=0.04)
    ap.add_argument("--min-interval", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.root)
    qpath = Path(args.questions) if args.questions else root / "questions.jsonl"
    sdir = root / "shards"
    sdir.mkdir(parents=True, exist_ok=True)
    rec_path = sdir / f"shard_{args.shard}.jsonl"
    done_path = sdir / f"shard_{args.shard}.done"

    questions = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()]
    mine = [q for q in questions if _shard_of(q["qid"], args.num_shards) == args.shard]
    done = {l.strip() for l in done_path.read_text().splitlines()} if done_path.exists() else set()
    todo = [q for q in mine if q["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[discover] shard {args.shard}/{args.num_shards}: {len(mine)} mine, {len(done)} done, {len(todo)} todo", flush=True)

    disc = Discovery(str(root / "cache/discovery"), max_results=args.top_k, min_interval=args.min_interval)
    t0 = time.time()
    rec_f = open(rec_path, "a", buffering=1)
    done_f = open(done_path, "a", buffering=1)
    try:
        for i, q in enumerate(todo):
            qtoks = _toks(q["question"])
            queries = make_queries(q["question"], q.get("qtype"), n=args.n_queries)
            hits, seen = [], set()
            for sq in queries:
                for h in disc.search(sq):
                    if h["url"] in seen:
                        continue
                    seen.add(h["url"])
                    h = dict(h)
                    h["query"] = sq
                    h["rel"] = round(_relevance(qtoks, h.get("title", ""), h.get("snippet", "")), 4)
                    hits.append(h)
            ranked = sorted(hits, key=lambda h: h["rel"], reverse=True)
            read_urls = [h["url"] for h in ranked if h["rel"] >= args.min_rel][: args.read_k]
            rec = {
                "qid": q["qid"], "source": q["source"], "question": q["question"],
                "queries": queries, "hits": hits, "read_urls": read_urls,
            }
            rec_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done_f.write(q["qid"] + "\n")
            if (i + 1) % args.progress_every == 0 or (i + 1) == len(todo):
                el = time.time() - t0
                print(f"[discover] {i+1}/{len(todo)} ({(i+1)/el*60:.1f}/min) {disc.stats} el={el/60:.1f}m", flush=True)
    finally:
        rec_f.close(); done_f.close(); disc.close()
    print(f"[discover] shard {args.shard} DONE in {(time.time()-t0)/60:.1f}m {disc.stats}", flush=True)


if __name__ == "__main__":
    main()
