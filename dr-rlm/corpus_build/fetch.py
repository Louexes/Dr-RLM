"""Phase 3 — FETCH: read the full text of every discovery-selected URL into the page cache.

This is the **not-IP-limited** half (diverse hosts, I/O-bound), so it uses *all cores*: the
deduped URL set is split across ``--workers`` single-threaded processes (multiprocessing, not
threads — trafilatura/lxml is not thread-safe; processes are). Each worker shares the on-disk
page cache (atomic per-URL writes), so the whole phase is naturally resumable — an already-cached
URL is skipped, a kill is harmless, and a re-run only fetches what's missing.

URL-level shardable too (``sha1(url) % num_shards == shard``) so FETCH can itself be a SLURM array
across nodes, on top of the per-node process pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reader import Reader  # noqa: E402

_CACHE_DIR = None
_READER_KW = {}


def _init(cache_dir: str, reader_kw: dict) -> None:
    global _CACHE_DIR, _READER_KW
    _CACHE_DIR, _READER_KW = cache_dir, reader_kw


def _read_batch(urls: List[str]) -> dict:
    rdr = Reader(_CACHE_DIR, **_READER_KW)
    for u in urls:
        rdr.read(u)
    rdr.close()
    return dict(rdr.stats)


def _collect_urls(root: Path, shards_glob: str) -> List[str]:
    seen, urls = set(), []
    for sf in sorted(root.glob(shards_glob)):
        for line in sf.read_text().splitlines():
            if not line.strip():
                continue
            for u in json.loads(line).get("read_urls", []):
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
    return urls


def main() -> None:
    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--shards-glob", default="shards/shard_*.jsonl")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0, help="URL-level shard (for multi-node fetch)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    cache_dir = str(root / "cache/pages")
    urls = _collect_urls(root, args.shards_glob)
    if args.num_shards > 1:
        urls = [u for u in urls if int(hashlib.sha1(u.encode()).hexdigest(), 16) % args.num_shards == args.shard]
    if args.limit:
        urls = urls[: args.limit]
    print(f"[fetch] {len(urls)} unique URLs to ensure-cached "
          f"(shard {args.shard}/{args.num_shards}, {args.workers} workers)", flush=True)
    if not urls:
        return

    batches = [urls[i:i + args.batch] for i in range(0, len(urls), args.batch)]
    reader_kw = dict(timeout=args.timeout, max_chars=args.max_chars)
    agg = Counter()
    done = 0
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(cache_dir, reader_kw)) as pool:
        for st in pool.imap_unordered(_read_batch, batches):
            agg.update(st)
            done += 1
            if done % 20 == 0 or done == len(batches):
                el = time.time() - t0
                got = agg["ok"] + agg["cache_hits"]
                print(f"[fetch] batch {done}/{len(batches)} | ok={agg['ok']} cache={agg['cache_hits']} "
                      f"fail={agg['fail']} skip={agg['skip']} | {got} usable | el={el/60:.1f}m", flush=True)
    print(f"[fetch] DONE {dict(agg)} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
