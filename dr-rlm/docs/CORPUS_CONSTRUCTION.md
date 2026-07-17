# Frozen Retrieval Corpus — Construction Method

*Methods writeup for the thesis. Every quantity below is measured from the actual build
(2026-06-27); the running code is in `dr-rlm/corpus_build/` and the immutable build record in
`data/frozen_corpus/manifest.json`. Companion docs: design/decisions in `DATA_PREP_PLAN.md`,
operational usage in `corpus_build/README.md`.*

---

## 1. Motivation

DR-RLM and the DR-Tulu baseline are compared as a **relative** difference (the post-training
delta, and the provenance-credit ablation). For that comparison to be valid, the retrieval
substrate must be held constant across arms and across stages: *same corpus, same retriever, no
live web.* Live search violates this in three ways — it is non-stationary (results drift between
runs, a confound), it is slow and rate-limited (web latency in the rollout loop inflates wall-clock
and, on a billed cluster, SBU for otherwise-idle GPUs), and it is paywalled/gated. We therefore
build the retrieval environment **once**, off the GPU, and **freeze** it: a single corpus plus a
single index that SFT data generation, RL training, and evaluation all retrieve from. The crawl is
network- and CPU-bound, so it runs on a data-staging partition at effectively zero GPU cost.

This trades one property for another, stated explicitly for the thesis: a question-seeded crawl
guarantees that relevant evidence exists in the corpus, so **absolute** scores are inflated
relative to open-web deep research and are *not* comparable to published live-search numbers. The
**relative** delta between arms — which is the thesis claim — is unaffected, because both arms hit
the identical frozen substrate.

## 2. Tool stack

Any research crawl has two distinct jobs, *discovery* ("which pages are relevant to this query")
and *reading* ("the full text of one page"), conventionally served by a paid search API and a paid
reader API respectively. We replace both with a free, reproducible stack and an offline index:

| Job | Component | Notes |
|---|---|---|
| Discovery | DuckDuckGo via `ddgs`, `backend="auto"` | aggregates upstreams; returns `{title, url, snippet}` |
| Reading | `httpx` + `trafilatura` main-content extraction | no browser; clean article text → passages |
| Index | `bm25s` (Okapi BM25, Lucene-style variant) | pure-Python; English stop-words + Snowball stemmer |

Two implementation choices differ from the obvious defaults, each for robustness rather than
preference, and each preserves the controlled-comparison invariant:

- **Reading uses `trafilatura`, not a headless browser** (Crawl4AI/Playwright). For the
  article/encyclopedia/scientific pages a research corpus is made of, `trafilatura`'s
  main-content extractor produces equivalent clean text as a pure-Python dependency, and runs
  reliably under heavy process-level parallelism on a compute node where a headless browser is
  fragile. JavaScript-heavy pages that it cannot render degrade gracefully to their discovery
  snippet, which is always retained.
- **Indexing uses `bm25s`, not pyserini/Lucene.** It is the same BM25 ranking function — so the
  "same retriever across arms" invariant holds — but with no Java dependency and no gated index
  format, building in seconds and loading memory-mapped in milliseconds. The discovery
  backend was selected empirically: `backend="auto"` returns on-topic results (e.g. Springer /
  ResearchGate for a chemistry query) where the library's default upstream returned unrelated SEO
  pages.

## 3. Seed questions

The corpus must cover the union of every question that will ever be searched, across the three
stages. **Crawl seeds use question text only; the gold answer or rubric never enters the crawl**
(no leakage; mirrors deployment). The seed set is assembled deterministically (fixed sampling
seed) from:

| Source | Stage | Count |
|---|---|---|
| `rl-research/dr-tulu-rl-data` | RL training prompts | 4,852 |
| `rl-research/dr-tulu-sft-data` (sampled) | SFT-gen prompts | 744 |
| HealthBench (`oss_eval`, sampled) | eval (long-form, medical) | 250 |
| ResearchQA (`test`, sampled) | eval (long-form) | 250 |
| SimpleQA (OpenAI public set, sampled) | eval (short-form, factoid) | 300 |
| ScholarQA-CS (`sqa/test`) | eval (long-form, scientific) | 97 |
| DeepResearchBench (`drb_en50`) | eval (long-form, open-web) | 50 |
| **Total (deduplicated)** | | **6,543** |

Eval counts are dev-scale; per-source caps are configurable for a final-scale eval without
recrawling the training portion.

## 4. Pipeline

Seven phases (`corpus_build/`), each shardable, resumable, and cached:

```
Phase 0  seeds.py      seed questions (no gold answers)          → questions.jsonl
Phase 1  queries.py    decompose into ≤3 discovery queries       (heuristic; no LLM dependency)
Phase 2  discover.py   DDG discovery, ranked, relevance-gated    → shards/shard_*.jsonl
Phase 3  fetch.py      read selected URLs → passages             → cache/pages/*.json
Phase 4  normalize.py  fold shards + pages → corpus              → corpus.jsonl {id, contents, url}
Phase 5  index_bm25.py build BM25 index                          → bm25s_index/
Phase 6  audit.py      per-benchmark coverage gate               → coverage_audit.json
Phase 7  freeze.py     checksum + manifest + runtime config      → manifest.json
```

**Query generation (Phase 1).** Corpus-*building* needs far fewer queries than the agent issues at
runtime (≈20 iterative searches per question): the raw natural-language question is itself the
strongest single query, so short factual questions emit it alone, while long research prompts emit
a first-clause query plus entity/keyword variants for recall. Heuristic decomposition keeps the
build free of any LLM/GPU dependency and fully reproducible.

**Relevance gate + tiering.** Each discovered URL is scored by title/snippet token-overlap with
the question; only the top-ranked on-topic URLs are read in full. A read page is chunked into
~1.8k-character passages (the BM25 retrieval unit); a URL that cannot be read contributes its
discovery snippet instead. So the corpus stores **full text where it could be fetched and snippets
elsewhere**, deduplicated by `(url, normalized-text)`.

**Two-rate decoupling.** Discovery and reading are run as *separate* phases because they obey
different limits. Discovery is rate-limited per source IP, so it runs as a small number of paced
processes; reading is I/O-bound across diverse hosts (not IP-limited) and is therefore run as many
single-threaded processes using all available cores. Decoupling lets each phase scale on its own
constraint, and — because the HTML/XML parser is not thread-safe — keeps reading in separate
processes rather than threads.

**Provenance invariant.** The corpus stores no provenance identifiers. Snippet ids of the form
`{node_rid}-{n}` are minted at the agent's tool facade at retrieval time, exactly as for live web,
so the recursive provenance-credit (RER) mechanism is byte-identical whether retrieval is offline
or online.

## 5. Infrastructure and cost

The build ran on the cluster's data-staging partition (CPU only, internet egress, no GPU →
effectively zero SBU). Discovery ran as a 6-shard array across distinct nodes; a single source IP
sustained ≈40 discovery queries/min with zero errors over a 100-query probe, so the per-IP limit
was not a practical constraint at this scale. Measured wall-clock:

- **Discovery:** 6,543 questions across 6 shards, 1 h 13 min, yielding 37,523 unique URLs marked
  for reading.
- **Reading:** the URL set, sharded 8 ways with 8 worker processes each (64-way), ≈12 min,
  31,340 pages fetched (the remainder blocked/non-HTML, covered by snippets).
- **Normalize + index + audit + freeze:** a few minutes, single CPU node.

Total monetary cost: **\$0**; GPU cost: **≈0 SBU**. The build is resumable end-to-end — discovery
is cached per `(query, k)` and reading per URL, so a re-run or a node failure never re-issues a
query or re-fetches a page.

## 6. Resulting corpus

**139,257 deduplicated documents** (`corpus_sha256` recorded in the manifest), distributed across
the seven sources:

| Source | Docs | Source | Docs |
|---|---|---|---|
| rl | 104,248 | simpleqa | 5,514 |
| sft | 17,542 | researchqa | 5,016 |
| healthbench | 6,373 | sqa | 2,131 |
| | | drb | 1,660 |

(Sources sum to >139,257 because a document surfaced by questions from multiple benchmarks is
counted under each.) The BM25 index is 344 MB; corpus text is 206 MB.

## 7. Validation

Coverage is the hard gate: thin retrieval would silently invalidate eval. Two independent checks
pass.

**Coverage audit (Phase 6).** For a sample of each benchmark's questions, the real BM25 index is
queried (`k=10`) and the best top-`k` document is scored for on-topic token overlap. Every
benchmark returns non-empty results for 100% of sampled questions, and the on-topic rate clears
the 0.85 gate for all seven:

| Source | non-empty | on-topic | mean top-1 BM25 |
|---|---|---|---|
| drb | 1.00 | 1.00 | 44.0 |
| healthbench | 1.00 | 0.92 | 28.4 |
| researchqa | 1.00 | 1.00 | 22.5 |
| rl | 1.00 | 1.00 | 25.1 |
| sft | 1.00 | 0.97 | 22.8 |
| simpleqa | 1.00 | 1.00 | 22.4 |
| sqa | 1.00 | 1.00 | 13.2 |

No benchmark was flagged thin. (Medical questions, HealthBench, have the lowest lexical overlap, as
expected, and still clear the gate.)

**Acceptance test (`verify_package.py`).** A separate end-to-end check loads the index through the
*actual runtime retrieval backend* (`corpus_search._Bm25sBackend`, the same code path RL and eval
use), confirms on-topic retrieval and full-text resolution for sample questions from all seven
benchmarks, and verifies the coverage gate. Result: **PASS**.

## 8. Reproducibility and integration

`manifest.json` pins the build date, the tool stack, the corpus SHA-256, the index file digest,
and the exact runtime configuration. All three downstream stages consume the corpus through one
switch:

```
search_backend   = bm25s
search_corpus_path = data/frozen_corpus/corpus.jsonl
search_index_path  = data/frozen_corpus/bm25s_index
```

SFT-gen, RL (`run_dr_rlm_L4.sh`), and evaluation (`generate.py --driver new`) all read these three
fields, so there is zero retrieval mismatch across the SFT → RL → eval pipeline.

## 9. Methods paragraph (liftable into the thesis)

> All retrieval — for SFT trajectory generation, RL training, and evaluation — is served by a
> single frozen corpus, so that DR-RLM and the DR-Tulu baseline are compared under an identical,
> stationary retrieval environment. We build it by crawling the open web once, off-GPU: for each
> of 6,543 seed questions (the RL training prompts, an SFT subset, and the seven evaluation
> benchmarks, using question text only), we issue up to three DuckDuckGo queries, read the most
> relevant returned pages with a main-content extractor, and segment them into ~1.8k-character
> passages, retaining discovery snippets for pages that cannot be fetched. The resulting 139,257
> documents are indexed with Okapi BM25. We verify coverage by querying the index with held-out
> questions from each benchmark: every benchmark returns on-topic evidence for ≥92% of sampled
> questions (100% non-empty), so no benchmark suffers retrieval starvation. The corpus is content-
> addressed and checksummed; because a question-seeded crawl guarantees that relevant evidence
> exists, absolute scores are higher than open-web deep research and are not comparable to
> live-search numbers, whereas the relative difference between arms — the quantity we report — is
> controlled by construction.

---

## Addendum — 2026-07-06: short-form guardrail extension (2wiki + webwalker)

The original build covered SimpleQA but omitted the other two short-form guardrail benchmarks.
Both were added by extending `seeds.py` (loaders `load_2wiki`, `load_webwalker` over
`akariasai/2wiki_rand1k` and `rl-research/webwalker_test`, full test sets) and re-running the
resumable pipeline — discovery/fetch caches meant only the ~1,680 new questions crawled; existing
6,543 seeds were byte-identical and skipped. Corpus grew **139,257 → 159,371 docs**
(`corpus_sha256=d759bb4a…`); reindexed and re-frozen.

- **2wiki:** covered, on-topic **1.00** (+13,803 docs).
- **webwalker:** the full 680-question set is **64% non-ASCII (CJK)**; the frozen corpus is an
  English-web substrate, so CJK questions are structurally uncovered (and the English-token
  on-topic metric penalizes them). The guardrail is therefore evaluated on the **English subset**
  `data/subsets/webwalker_en246.jsonl` (n=246), whose on-topic rate is **1.00**. The full-set
  audit flag is expected, not a crawl defect (see `coverage_audit.json` → `notes.webwalker`).
- **Merge-verify:** the added docs did not degrade long-form retrieval — all long-form benchmarks
  remain on-topic 1.00 (HealthBench 0.92→0.95), so the pre-existing untrained baselines stand.
