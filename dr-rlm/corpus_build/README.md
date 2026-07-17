# corpus_build — the self-crawled frozen retrieval corpus

Builds the **single frozen corpus + BM25 index** that backs DR-RLM **SFT-gen, RL, and eval**, by
crawling the open web once with a **free stack** and freezing it. Train and eval offline against
this one substrate → controlled comparison (`experiment_matrix.md §2`: same corpus + same
retriever, no live web), reproducible, ungated, ~0 SBU. Design doc: `../docs/DATA_PREP_PLAN.md`.

## Tool stack (free, validated 2026-06-26 from the cluster)
- **Discovery** (Serper's job): `ddgs` with `backend="auto"` — validated to return on-topic
  results where the default upstream returned SEO spam. (`discovery.py`)
- **Reading** (Jina's job): `httpx` + `trafilatura` main-content extraction — **no browser**
  (chosen over Crawl4AI/Playwright for headless robustness at scale). (`reader.py`)
- **Index**: `bm25s` (pure-Python BM25, no Java/pyserini) — same BM25 algorithm, so the
  controlled-comparison invariant holds. Runtime reader = `corpus_search._Bm25sBackend`. (`index_bm25.py`)

## Pipeline (Phases 0–7)
| Phase | Script | What |
|---|---|---|
| 0 | `seeds.py` | question text (no gold answers) from RL + SFT + 7 evals → `questions.jsonl` |
| 1–2 | `discover.py` | decompose → DDG discovery (IP-rate-limited; sharded across nodes/IPs) → `shards/shard_*.jsonl` (hits + `read_urls`) |
| 3 | `fetch.py` | read `read_urls` full text (NOT IP-limited; all cores) → `cache/pages/*.json` |
| 4 | `normalize.py` | fold shards + page cache → `corpus.jsonl` (`{id, contents, url}`) + `by_source/` |
| 5 | `index_bm25.py` | BM25 index → `bm25s_index/` |
| 6 | `audit.py` | per-benchmark coverage gate → `coverage_audit.json` |
| 7 | `freeze.py` | checksums + `manifest.json` + the runtime config to pin |

**Why discover/fetch are split:** discovery is IP-rate-limited (~30/min/IP → few paced procs,
sharded across nodes for distinct IPs); reading is I/O-bound across diverse hosts (no IP limit →
many procs, all cores). Decoupling lets each scale on its own limit *and* sidesteps lxml's
thread-unsafety (fetch uses processes, not threads).

## Run it (Snellius `staging`, distinct-IP sharding)
```bash
VENV=dr-rlm/.venv-crawl/bin/python            # ddgs + trafilatura + bm25s
ROOT=dr-rlm/data/frozen_corpus

# Phase 0 (login): seeds
$VENV dr-rlm/corpus_build/seeds.py            # -> questions.jsonl (6,543 q)

# Phases 1-2: discover — one shard per node (distinct outbound IP); --nodelist pins nodes
for i in 0..5: sbatch --partition=staging --nodelist=srv$N --cpus-per-task=2 \
  --wrap "$VENV corpus_build/discover.py --root $ROOT --num-shards 6 --shard $i --min-interval 2.0"
# Phase 3: fetch (afterany the shards) — 1 node, many workers
sbatch --partition=staging --cpus-per-task=32 --dependency=afterany:<disc-ids> \
  --wrap "$VENV corpus_build/fetch.py --root $ROOT --workers 28"
# Phases 4-7: finalize (afterok fetch)
sbatch --partition=staging --dependency=afterok:<fetch-id> corpus_build/finalize.sh 2026-06-26
```
All three phases are **shardable, resumable (`.done` + per-URL cache), and cached** — a kill or
re-run never re-issues a completed query or re-fetches a cached URL.

## Harvest loop (re-crawl thin benchmarks, DATA_PREP_PLAN §7.1/§10)
If `audit.py` flags a source as THIN: re-run `seeds.py` for just that source, `discover.py` with
more `--n-queries`, then `fetch.py`/`normalize.py`/`index_bm25.py`. The caches make this cheap.

## Point SFT-gen / RL / eval at the corpus (no script changes)
The run scripts (`configs/run_dr_rlm_L*.sh`) already map env → hydra. Set:
```bash
export SEARCH_BACKEND=bm25s
export SEARCH_CORPUS_PATH=dr-rlm/data/frozen_corpus/corpus.jsonl
export SEARCH_INDEX_PATH=dr-rlm/data/frozen_corpus/bm25s_index
```
`freeze.py` prints this exact block; `manifest.json` records it. The corpus carries **no**
provenance ids — the harness mints `{node_rid}-{n}` at the tool facade, so the RER credit
mechanism is untouched. (`bm25s`+`PyStemmer` are installed in `rl/skyrl/.venv`.)
