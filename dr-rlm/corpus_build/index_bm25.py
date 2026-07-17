"""Phase 5 — build the BM25 retrieval index over ``corpus.jsonl``.

Uses **bm25s** (pure-Python numpy/scipy BM25) rather than pyserini/Lucene: same BM25 algorithm
(so the controlled-comparison invariant "same retriever across arms" holds), but no Java, no
gated dr_agent index format, builds in seconds and loads memory-mapped in ms. The matching
runtime reader is ``corpus_search._Bm25sBackend`` (``search_backend="bm25s"``).

Tokenization (english stopwords + Snowball stemmer) must be identical here and at query time;
the backend reconstructs the same stemmer, so this file is the single definition of record.
The full ``{id, contents, url}`` rows are saved alongside the index (``corpus=...``) so the
backend can return snippets/urls and resolve get_doc full text without a second corpus file.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build(corpus_path: str, index_dir: str) -> int:
    import bm25s
    import Stemmer

    corpus = [json.loads(l) for l in open(corpus_path) if l.strip()]
    if not corpus:
        raise SystemExit(f"[index] empty corpus at {corpus_path}")
    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    tokens = bm25s.tokenize([d["contents"] for d in corpus], stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(tokens, show_progress=False)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    retriever.save(index_dir, corpus=corpus)
    print(f"[index] {len(corpus)} docs -> {index_dir} in {time.time()-t0:.1f}s")
    return len(corpus)


def main() -> None:
    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--corpus", default=None, help="default <root>/corpus.jsonl")
    ap.add_argument("--index-dir", default=None, help="default <root>/bm25s_index")
    args = ap.parse_args()
    root = Path(args.root)
    corpus = args.corpus or str(root / "corpus.jsonl")
    index_dir = args.index_dir or str(root / "bm25s_index")
    build(corpus, index_dir)


if __name__ == "__main__":
    main()
