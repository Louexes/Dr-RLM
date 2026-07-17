# DR-RLM Data Preparation Plan

**Purpose.** One reference for how *all* data for the thesis is prepared: the RL training
data, the SFT data, and the 7 evaluation benchmarks — all sitting on a **single frozen,
self-crawled retrieval corpus**. Written 2026-06-26, consolidating the data/backend design
discussion. Status: **BUILDING — launched 2026-06-26** (full crawl running on Snellius `staging`).
Implementation: `corpus_build/` (see its `README.md`). Numbers below are grounded in our own run
ledgers where noted.

> **Build decisions locked (validated 2026-06-26):** reader = `trafilatura` (httpx, no browser)
> not Crawl4AI/Playwright; index = `bm25s` (pure-Python BM25, backend `corpus_search._Bm25sBackend`)
> not pyserini; discovery = `ddgs backend="auto"` (on-topic where the default upstream returned
> spam); venue = `staging` partition (egress + in-budget, ~0 SBU); 6,543 seed questions
> (RL 4,852 + SFT 744 + DRB/ResearchQA/HealthBench/SQA/SimpleQA). SFT-gen/RL/eval all wired to
> `search_backend=bm25s`. Single-IP DDG rate measured at ~40/min, 0 errors.

---

## TL;DR — the decision

Build **one frozen corpus + BM25 index** covering every question we will ever search
(RL prompts + SFT subset + all 7 benchmarks), by **crawling the open web once** with a
**free tool stack** (`ddgs`/SearXNG for discovery + Crawl4AI for reading), then freezing it.
Train and evaluate offline against that single substrate. This is reproducible, controllable,
ungated, and — with the rate-limit strategy in §7 — buildable in ~an afternoon for **$0**.

We do **not** use live web search in the training loop (slow → burns SBU; non-reproducible →
time-drift confound) and we do **not** depend on DR-Tulu's gated `wii` corpus (see §2).

---

## 1. Why offline + frozen (not live, not gated)

The thesis claim is the **relative** delta (DR-RLM vs DR-Tulu, and the credit ablation), held
under the `experiment_matrix.md §2` invariant: *same corpus + same retriever, no live web.*

- **Live-in-the-loop is out for training.** SBU is billed on wall-clock (768/hr on 4×H100,
  charged on elapsed time regardless of GPU idle). Live Serper/Jina makes rollouts I/O-bound
  (seconds per tool call × dozens of calls × thousands of trajectories) → step time inflates
  ~1.5–5× → 1.5–5× the SBU. It is also non-reproducible (results drift run-to-run = a
  confound) and rate-limited/paywalled.
- **Frozen corpus is the fix.** Pay the web latency **once**, off the GPU node (the crawl is
  network/CPU-bound → ~0 SBU), then every train/eval rollout is GPU-bound and reproducible.
- **Build order matters:** corpus FIRST → then SFT-gen → RL → eval, all consuming the frozen
  version. This is the only way to guarantee zero retrieval mismatch across the three stages —
  a property neither `wii` nor DR-Tulu's own pipeline provides (they train offline-BM25 but
  eval live).

---

## 2. Why NOT the `wii` corpus (verified)

`s42chen/wii-indexes` is referenced in DR-Tulu's own repo (`rl/open-instruct/local.md`,
`train_local/*.sh`), and `s42chen` = **Steven Chen**, a DR-Tulu team member (HF orgs: Ai2,
RL-ReSearch, Castorini, Tevatron — the IR groups that build the BM25/Qwen3-Embed indices).
So the link is real. **But:**

- It is the **offline LOCAL-REPRODUCTION corpus**, not the paper's substrate. The official
  `train_dr_tulu.sh` uses `rl-research/dr-tulu-rl-data` + an internal cloud datastore
  (`MASSIVE_DS_URL`); the paper's **eval uses live search/browse** (`browse_tool_name=jina`,
  `create_mcp_sampler_with_websearch`). `wii` only appears in the toy `train_local` (0.6B
  "mini") path.
- It is **small** (~30 GB, ~100K–1M docs), **gated** (personal account), **undocumented**
  (empty card), and almost certainly **does not cover our 7 benchmark domains**
  (medical/scientific/multi-hop) — DR-Tulu never used it for eval.
- **wii-indexes covers ~⅓ of one stage** (RL retrieval) and is exactly the piece our self-crawl
  replaces — with better, controllable coverage *and* eval coverage.

What IS worth taking from DR-Tulu, **ungated**: the RL prompts `rl-research/dr-tulu-rl-data`
and the SFT data `rl-research/dr-tulu-sft-data` (+ the `convert_drtulu_rl` schema step we have).
The retriever (BM25) is the same algorithm whether the corpus is theirs or ours.

> Outreach to Steven for `wii-indexes` access is a cheap nice-to-have, **not a dependency**.
> Ask only: (1) is it the paper's corpus or local-repro? (2) does it cover our benchmarks?

---

## 3. The tool stack — Serper/Jina vs the free alternative

Two distinct jobs in any research-crawl:

| Job | Paid | **Free (chosen)** | Returns |
|---|---|---|---|
| **Discovery** ("which pages are relevant") | Serper (Google API) | **`ddgs` → SearXNG** | `{title, url, snippet}` |
| **Reading** ("full text of one page") | Jina Reader | **Crawl4AI** (Playwright→markdown, no-LLM mode) | url → clean markdown |

- **Discovery is essential; reading is the upgrade.** A "snippet-only" corpus (Serper/`ddgs`
  results, ~300 chars) is enough for short-form factoid; **full-text (Crawl4AI) matters for the
  long-form/deep-research benchmarks and for citation grounding** (citing a real passage, not a
  2-sentence preview). For a provenance thesis, do both.
- **Cost reality (why free):**

| Stack | Full-scale cost | Reliability | Meets "free"? |
|---|---|---|---|
| Tavily (one API, search+extract) | ~$900–1,450 | ★★★ | ❌ priciest |
| Serper + Jina | ~$150 | ★★★ | ❌ (can't pay) |
| **`ddgs`/SearXNG + Crawl4AI** | **$0** | ★ (mitigated §7) | ✅ |

- **Tavily** (free tier = 1,000 credits/mo, then $0.005–0.008/credit): too small for the full
  build (~1% of one crawl), but **use the free tier as a reliable PILOT** — validate the whole
  pipeline on ~50 questions against a clean API before committing days to the free stack, and/or
  for the small eval slice where reliability matters most.

---

## 4. Data inventory — what gets crawled

The corpus must cover the **union of all questions** across the three stages. Crawl seeds use
**question text only — never the gold answer.**

| Layer | Source | Count | Notes |
|---|---|---|---|
| **RL training prompts** | `rl-research/dr-tulu-rl-data` (ungated) | ~4,881 | carries rubrics (`ground_truth`); needs `convert_drtulu_rl` schema step |
| **SFT prompts** | subset of the RL prompts | ~few hundred–1k | **no new questions** → already covered by the RL crawl |
| **Eval — 4 long-form** | DeepResearchBench, ResearchQA, ScholarQA-CS-V2, HealthBench | ~100–500 each | rubric/physician-graded; open-web/scientific/medical |
| **Eval — 3 short-form** | SimpleQA, 2WikiMultihop, WebWalker | ~100–500 each | exact-match/F1; wiki/multi-hop (guardrail / RQ5) |

Eval sets live in `dr-tulu/agent/evaluation/{...}_eval/`; W1 dev subsets in `data/subsets/`
(`drb_en50`, `researchqa_strat49`, `healthbench_hard_repr`, `sqav2_cs100`).

**Total unique questions: ~6,000 (dev-scale eval) – ~10,000 (full eval).**

---

## 5. Grounded usage (from our own ledgers)

Measured from `runs/provenance_poc` (Serper) and `runs/base5`,`quiet5` (browsed pages):

| Signal | Value | Source |
|---|---|---|
| Agent **search calls / question** | mean **19.7**, median 15.5 (4–52) | `n_search_calls` × 40 trees |
| Results / search call | ~9.3 | snippets ÷ search calls |
| Snippet size | ~303 chars (~76 tok) | offline corpus |
| Browsed full-page size | mean ~2,449 tok, median ~1,377, capped ~5k | live deep-research blocks |

**Key correction for crawl design:** the ~20 searches/question is *agent runtime* behavior
(iterative). Corpus-**building** needs far fewer — **~3–5 deliberate decomposed queries per
question** (see §7). That is the single biggest cost/time lever.

---

## 6. The build pipeline (Phase 0–7)

Extends `analysis/build_snapshot_corpus.py` (already emits the target schema) and the
`local_jsonl`/`bm25` backend in `corpus_search.py`. The corpus carries **no provenance ids** —
the harness mints `{node_rid}-{n}` at the tool facade, so the credit mechanism is untouched.

- **Phase 0 — Assemble seed questions.** Extract question text from all sources → `questions.jsonl`
  `{qid, question, source, split, qtype}`. Dedup. **Gold answers never enter the crawl.**
- **Phase 1 — Query generation.** LLM-decompose each question into **~3–5 sub-queries** (ideally
  the agent's own model+prompt, so the crawl's query distribution ≈ the agent's). Global dedup of
  near-identical queries. → `queries.jsonl`.
- **Phase 2 — Discovery (`ddgs`/SearXNG).** Each query → top-10 results `{title, url, snippet}`.
  Dedup URLs globally. Snippets become the `search()`-tier docs.
- **Phase 3 — Reading (Crawl4AI).** For the top ~10–20 unique URLs/question, Crawl4AI →
  clean markdown; chunk long pages into passages. These become the `get_doc()`-tier full text.
  Graceful timeout/paywall/retry handling.
- **Phase 4 — Normalize → `corpus.jsonl`.** Map to `{id: snap-<sha1[:16]>, contents, url}`,
  dedup by `(url, normalized-text)`. Decide snippet-only vs full-text in `contents` (see §11).
  Emit per-benchmark + `combined.jsonl` (mirrors `data/offline_corpus/`).
- **Phase 5 — Index (BM25).** Pyserini/Lucene over `corpus.jsonl` → `data/bm25/`. CPU, minutes,
  **0 GPU.** (Dense/FAISS optional, only if BM25 recall is insufficient — needs one-time GPU embed.)
- **Phase 6 — Wire + coverage audit (THE GATE).** `SEARCH_BACKEND=bm25`/`local_jsonl` +
  `SEARCH_CORPUS_PATH`; `make_corpus_tools` serves `search()`/`get_doc()`. Then audit per
  benchmark (§10). Re-crawl thin benchmarks.
- **Phase 7 — Freeze + version + document.** Checksum + snapshot corpus+index, pin in configs,
  record build params/date/coverage stats for the thesis methods + the offline-relative caveat.

---

## 7. Beating the DDG rate limit (the smart moves)

The limit is ~30 req/min/IP. Don't fight it — shrink the volume and use your own IPs. Two moves
turn a ~4.6-day naïve crawl into **~5 hours, $0**.

1. **Cut query volume ~5×** (Phase 1): ~3–5 decomposed queries/question (not the agent's ~20) +
   global query dedup + **harvest-loop** (start at 1–2 queries/q, run the coverage audit, expand
   only the questions that came back thin). → ~200k queries become **~40k**.
2. **Shard across cluster nodes (distinct IPs).** The limit is per-IP; SLURM nodes each have their
   own IP. N shards on N nodes → N× throughput. Legitimate (own compute, parallel jobs are what
   the cluster is for), no proxies. *Math:* 40k ÷ (30/min × 4 nodes) ≈ **5.5 hr**.
3. **SearXNG (self-hosted) instead of raw `ddgs`.** Aggregates many engines (load spreads across
   upstreams), you control pacing, far less fragile than the `ddgs` HTML scraper. Robust discovery.
4. **Skip live search for benchmarks whose evidence is in bulk free corpora:** Wikipedia dump
   (2wiki/SimpleQA), S2ORC/OpenAlex (ScholarQA/scientific), PubMed/PMC (HealthBench). No rate
   limit, better domain coverage. Reserve live discovery for the open-web ones (DRB/ResearchQA).
5. **Rotate free tiers** for residual live queries: `ddgs` + Brave (2k/mo) + Tavily (1k/mo).
6. **Hygiene (always):** cache at query *and* URL level; resumable checkpointing (survive a
   multi-hour/day run); backoff + jitter + UA rotation; stay <30/min/IP.

**Avoid:** free proxy lists / Tor (unreliable, ToS/security risk) and multi-account games —
the cluster-IP sharding gives the same parallelism cleanly.

---

## 8. Cost & time

| | Free stack (chosen) | Paid reference (Serper+Jina) | Tavily |
|---|---|---|---|
| **$ (full build)** | **$0** | ~$150 | ~$900–1,450 |
| **SBU (GPU)** | ~0 (crawl off-node; BM25 CPU) | ~0 | ~0 |
| **Wall-clock** | ~5 hr (with §7 moves) to ~1 day | ~3–12 hr | minimal |
| FAISS option | +few GPU-hrs one-time | — | — |

Either way the crawl is ~free in SBU; the free stack trades dollars for ~half a day of crawl +
crawler engineering (resumable/shardable).

---

## 9. Eval methodology & caveats (carry into the thesis)

- **Fairness (automatic):** both arms (DR-Tulu R0b, DR-RLM R-Rd) hit the *same* frozen corpus →
  controlled. The train→eval setup is common-mode across arms.
- **Build eval the SAME way as training** — question-seeded only, same crawl depth + retriever →
  no train/eval density mismatch, no gold leakage (mirrors deployment).
- **Absolute-score caveat:** a question-seeded crawl guarantees relevant docs exist → absolute
  scores run higher than open-web and are **not comparable to live published numbers**; the
  **relative delta is valid**. State this explicitly.
- **Short-form note:** for SimpleQA/2wiki a question-seeded crawl often contains the answer page
  (realistic; both arms equal) → RQ5 reads as closed-book-over-snapshot; document it.
- **If eval ever moves to live (optional):** fine *iff* (1) both arms eval on the same live
  backend in the same window / cached results, and (2) the offline corpus returns the same
  *format* (length/granularity) as the live tools, so the policy isn't input-shocked. Best
  presented as a **both-regimes robustness check** (eval on frozen *and* cached-live; show the
  delta survives) — turns the offline/live question from a confound into a result.

---

## 10. Coverage audit — the hard gate (Phase 6)

With DDG's thinner coverage this is mandatory, not optional. For each benchmark, on a sample:
- Does top-k retrieval return **non-empty, on-topic** docs? (catch retrieval starvation)
- For factoid: does the answer-supporting evidence appear in top-k?
- Report per-benchmark hit-rate; **flag + re-crawl** thin benchmarks (more queries/depth, or
  pull from the bulk corpus for that domain). Silent thin coverage = invalid eval.

---

## 11. Reuse vs build-new

**Reuse:**
- `analysis/build_snapshot_corpus.py` — schema `{id, contents, url}`, `snap-<sha1>` ids, dedup.
- `corpus_search.py` `_LocalJsonlBackend` / `_DrAgentBackend(bm25)` + `make_corpus_tools` — serves
  `search()`/`get_doc()`, mints provenance ids. *(Note: local_jsonl currently returns the same
  `contents` for both search and get_doc → for a snippet+full-text two-tier, store full text in
  `contents` and add a short preview-truncation for `search()`. Minor backend task.)*
- existing Serper/Jina client code (swap to `ddgs`/Crawl4AI calls), `data/offline_corpus/` layout.

**Build new:**
- Crawler driver: Phase 1–3 (decompose → discover → read → passage-chunk), **async, shardable,
  resumable, cached**.
- Pyserini BM25 index build script.
- Coverage-audit script (§10).
- Bulk-corpus loaders (Wikipedia/S2ORC/PubMed) for the benchmarks in §7.4.

---

## 12. Open decisions / next steps

- [ ] **Retriever:** pin **BM25** (no GPU, matches DR-Tulu local path). Revisit dense only on audit gap.
- [ ] **Corpus tier:** snippet-only (cheapest, short-form-OK) vs **+ Crawl4AI full-text**
  (recommended for grounding/long-form). Default: full-text.
- [ ] **Discovery backend:** start `ddgs`, stand up **SearXNG** as the robustness upgrade.
- [ ] **Eval N per benchmark:** dev-scale (~100, ~$0/fast) first; scale to ~500 for final.
- [ ] **Tavily pilot:** use free 1k credits to validate the pipeline on ~50 questions before the bulk crawl.
- [ ] **SFT init question** (separate `project-rising-r-curriculum` thread): self-distill SFT seeds
  child-citing → may collapse the two-phase curriculum; SFT-gen must run against THIS frozen corpus.

**Critical path:** stand up the shardable/resumable crawler → Tavily pilot (~50 q) → full free
crawl (§7) → BM25 index → coverage audit → freeze. Then SFT-gen → RL → eval, all on the frozen corpus.
