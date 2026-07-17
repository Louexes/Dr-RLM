"""Phase 4 — fold discovery shards + the page cache into the frozen ``corpus.jsonl``.

Schema + ids are byte-compatible with ``analysis/build_snapshot_corpus.py`` and what
``corpus_search``'s local_jsonl/bm25 backends read: ``id = "snap-"+sha1(url\\0normtext)[:16]``,
carrying **no** provenance (the harness mints ``{node_rid}-{n}`` at the tool facade, so the RER
credit mechanism is untouched).

Inputs: ``shards/shard_*.jsonl`` (hits + ``read_urls`` from ``discover.py``) and
``cache/pages/*.json`` (full text from ``fetch.py``). Tiering (DATA_PREP_PLAN §11):
  * a URL with cached full text -> emit its **passages** (the get_doc tier; the facade truncates
    the search() view to ``snippet_max_chars`` for free, so one ``contents`` serves both tiers).
  * a URL with no full text (blocked/failed/not selected) -> emit its discovery **snippet**.

Dedup by ``(url, normalized-text)``; a doc's source set is the union of every benchmark whose
question surfaced it (drives the per-source splits the coverage audit reads).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", (t or "").strip().lower())


def did_of(url: str, contents: str) -> str:
    return "snap-" + hashlib.sha1((url + "\x00" + norm(contents)).encode()).hexdigest()[:16]


def _load_page_cache(cache_dir: Path) -> Dict[str, List[str]]:
    """url -> passages, for successfully-read pages."""
    url2pass: Dict[str, List[str]] = {}
    if not cache_dir.exists():
        return url2pass
    for cf in cache_dir.glob("*.json"):
        try:
            obj = json.loads(cf.read_text())
        except Exception:
            continue
        if obj.get("ok") and obj.get("passages"):
            url2pass[obj.get("url", "")] = obj["passages"]
    return url2pass


def main() -> None:
    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--shards-glob", default="shards/shard_*.jsonl")
    ap.add_argument("--out", default=None, help="default <root>/corpus.jsonl")
    ap.add_argument("--min-chars", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "corpus.jsonl"
    by_dir = root / "by_source"
    by_dir.mkdir(parents=True, exist_ok=True)

    url2pass = _load_page_cache(root / "cache/pages")
    shard_files = sorted(root.glob(args.shards_glob))
    if not shard_files:
        raise SystemExit(f"[normalize] no shard files at {root/args.shards_glob}")

    docs: Dict[str, Dict] = {}
    n_rec = n_pass = n_snip = 0

    def add(url: str, contents: str, source: str) -> None:
        contents = (contents or "").strip()
        if len(contents) < args.min_chars:
            return
        did = did_of(url, contents)
        d = docs.get(did)
        if d is None:
            docs[did] = {"id": did, "contents": contents, "url": url, "_sources": {source}}
        else:
            d["_sources"].add(source)

    for sf in shard_files:
        for line in sf.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rec += 1
            src = rec.get("source", "?")
            for url in rec.get("read_urls", []):
                for p in url2pass.get(url, []):
                    add(url, p, src)
                    n_pass += 1
            for h in rec.get("hits", []):
                url = h.get("url", "")
                if url in url2pass:
                    continue  # full text already in corpus -> snippet redundant
                title, snip = h.get("title", ""), h.get("snippet", "")
                text = (title + ". " + snip).strip(". ").strip() if title else snip
                add(url, text, src)
                n_snip += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    src_counts: Dict[str, int] = defaultdict(int)
    with open(out, "w") as f:
        for d in docs.values():
            for s in d["_sources"]:
                src_counts[s] += 1
            f.write(json.dumps({"id": d["id"], "contents": d["contents"], "url": d["url"]}, ensure_ascii=False) + "\n")

    handles: Dict[str, object] = {}
    for d in docs.values():
        for s in d["_sources"]:
            h = handles.get(s) or handles.setdefault(s, open(by_dir / f"{s}.jsonl", "w"))
            h.write(json.dumps({"id": d["id"], "contents": d["contents"], "url": d["url"]}, ensure_ascii=False) + "\n")
    for h in handles.values():
        h.close()

    lens = sorted(len(d["contents"]) for d in docs.values()) or [0]
    print(f"[normalize] {n_rec} records, {len(url2pass)} read pages -> "
          f"{n_pass} passage-docs + {n_snip} snippet-docs considered")
    print(f"[normalize] {len(docs)} deduped docs -> {out}")
    print(f"[normalize] doc chars: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    print(f"[normalize] per-source docs: {dict(src_counts)}")


if __name__ == "__main__":
    main()
