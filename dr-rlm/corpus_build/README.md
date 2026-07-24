# corpus_build — the frozen retrieval corpus

Builds the single frozen corpus + BM25 index that backs SFT-gen, RL, and eval: crawl the open
web once with a free stack, freeze it, and train/eval offline against that one substrate
(controlled comparison, reproducible, no API cost).

Stack: `ddgs` for discovery, `httpx` + `trafilatura` for reading (no browser), `bm25s` for the
index (runtime reader: `corpus_search._Bm25sBackend`).

Pipeline (run in order):

| Phase | Script | Output |
|---|---|---|
| 0 | `seeds.py` | benchmark/train question text → `questions.jsonl` |
| 1–2 | `discover.py` (uses `discovery.py`) | search hits + `read_urls` shards |
| 3 | `fetch.py` | full page text → `cache/pages/` |
| 4 | `normalize.py` | `corpus.jsonl` (`{id, contents, url}`) |
| 5 | `index_bm25.py` | `bm25s_index/` |
| 6 | `audit.py` | per-benchmark coverage gate |
| 7 | `freeze.py` | checksums + `manifest.json` |

`verify_package.py` checks the frozen package end-to-end. Discovery is IP-rate-limited (pace it,
shard across nodes); fetching is I/O-bound (parallelize freely).
