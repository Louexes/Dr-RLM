"""Phase 0 — assemble the union of seed questions across RL + SFT + the eval benchmarks.

Emits ``questions.jsonl`` rows ``{qid, question, source, split, qtype}``. **Crawl seeds use the
question text only — the gold answer/rubric never enters the crawl** (no leakage; mirrors
deployment). Each source loader is wrapped so a missing/un-parseable source degrades to a warning
instead of killing the build. Sampling is deterministic (fixed seed) so the seed set is
reproducible and a re-run is idempotent.

Sources (DATA_PREP_PLAN §4):
  rl         rl-research/dr-tulu-rl-data        (4881, HF cache)  -> the training prompts
  sft        rl-research/dr-tulu-sft-data       (13062, sample)   -> SFT-gen prompts
  drb        data/subsets/drb_en50.jsonl        (50)              -> DeepResearchBench (long)
  researchqa research_qa_eval/data/test.json    (3750, sample)    -> ResearchQA (long)
  healthbench health_bench_eval .../oss_eval.jsonl (5000, sample) -> HealthBench (long)
  sqa        eval-data/sqa/test.jsonl           (100)             -> ScholarQA-CS (long)
  simpleqa   openai public CSV                  (sample)          -> SimpleQA (short)
  2wiki      akariasai/2wiki_rand1k             (1000)            -> 2WikiMultihopQA (short, guardrail)
  webwalker  rl-research/webwalker_test         (680)             -> WebWalkerQA (short, guardrail)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]          # /gpfs/home5/lgehringer/Dr-RLM
DRRLM = REPO / "dr-rlm"
DRTULU = REPO / "dr-tulu"
EVAL_SNAP = Path.home() / ".cache/huggingface/hub/datasets--rl-research--dr-tulu-eval-data/snapshots"


def _last_user_text(prompt) -> str:
    """HealthBench/RL ``messages``/``prompt`` -> joined user-turn text (the searchable topic)."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = [str(m.get("content", "")) for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
        return " ".join(p for p in parts if p).strip() or " ".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return str(prompt or "")


def _sample(items: List, n: Optional[int], seed: int) -> List:
    if n is None or n >= len(items):
        return items
    rnd = random.Random(seed)
    idx = sorted(rnd.sample(range(len(items)), n))
    return [items[i] for i in idx]


# --- loaders: each yields (question, qtype, ext_id) -------------------------

def load_rl(cap: Optional[int], seed: int):
    from datasets import load_dataset

    ds = load_dataset("rl-research/dr-tulu-rl-data", split="train")
    rows = list(range(len(ds)))
    rows = _sample(rows, cap, seed)
    for i in rows:
        ex = ds[i]
        msgs = ex.get("messages") or []
        q = msgs[0]["content"] if msgs else ""
        yield q, ex.get("question_type", ""), f"{i}"


def load_sft(cap: Optional[int], seed: int):
    from datasets import load_dataset

    ds = load_dataset("rl-research/dr-tulu-sft-data", split="train")
    rows = _sample(list(range(len(ds))), cap, seed)
    for i in rows:
        ex = ds[i]
        yield ex.get("question", ""), ex.get("type", ""), str(ex.get("id", i))[:16]


def load_drb(cap: Optional[int], seed: int):
    p = DRRLM / "data/subsets/drb_en50.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in _sample(rows, cap, seed):
        yield r.get("prompt", ""), r.get("topic", "long_form"), str(r.get("id", ""))


def load_researchqa(cap: Optional[int], seed: int):
    p = DRTULU / "agent/evaluation/research_qa_eval/data/test.json"
    rows = json.loads(p.read_text())
    for r in _sample(rows, cap, seed):
        yield r.get("query", ""), r.get("field", "long_form"), str(r.get("id", ""))


def load_healthbench(cap: Optional[int], seed: int):
    p = DRTULU / "agent/evaluation/health_bench_eval/data/2025-05-07-06-14-12_oss_eval.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in _sample(rows, cap, seed):
        yield _last_user_text(r.get("prompt")), "health", str(r.get("prompt_id", ""))


def load_sqa(cap: Optional[int], seed: int):
    snaps = sorted(EVAL_SNAP.glob("*/sqa/test.jsonl")) if EVAL_SNAP.exists() else []
    if not snaps:
        return
    rows = [json.loads(l) for l in snaps[0].read_text().splitlines() if l.strip()]
    for r in _sample(rows, cap, seed):
        yield r.get("problem", ""), "long_form", str(r.get("example_id", ""))


def load_simpleqa(cap: Optional[int], seed: int):
    import httpx

    cache = DRRLM / "data/frozen_corpus/_simpleqa_test_set.csv"
    if cache.exists():
        text = cache.read_text()
    else:
        url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        text = httpx.get(url, timeout=60, follow_redirects=True).text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    for i, r in enumerate(_sample(rows, cap, seed)):
        yield r.get("problem", ""), "exact_answer", f"sqa{i}"


# short-form guardrail eval sets (HF; question = messages[-1]["content"], same as DR-Tulu's
# load_shortformqa_data). Seed the FULL test sets so any eval subset is covered.
def _load_shortform_hf(repo: str, tag: str):
    def loader(cap: Optional[int], seed: int):
        from datasets import load_dataset

        ds = load_dataset(repo, split="test")
        for i in _sample(list(range(len(ds))), cap, seed):
            msgs = ds[i].get("messages") or []
            q = msgs[-1]["content"] if msgs else ""
            yield q, "exact_answer", f"{tag}{i}"
    return loader


load_2wiki = _load_shortform_hf("akariasai/2wiki_rand1k", "2wiki")
load_webwalker = _load_shortform_hf("rl-research/webwalker_test", "webwalker")


LOADERS = {
    "rl": load_rl, "sft": load_sft, "drb": load_drb, "researchqa": load_researchqa,
    "healthbench": load_healthbench, "sqa": load_sqa, "simpleqa": load_simpleqa,
    "2wiki": load_2wiki, "webwalker": load_webwalker,
}
# default per-source caps (None = all). Eval kept dev-scale; RL fully covered.
DEFAULT_CAPS = {
    "rl": None, "sft": 1000, "drb": None, "researchqa": 250,
    "healthbench": 250, "sqa": None, "simpleqa": 300, "2wiki": None, "webwalker": None,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DRRLM / "data/frozen_corpus/questions.jsonl"))
    ap.add_argument("--sources", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--caps", default="", help="override caps, e.g. 'rl=200,sft=50' (use 'all' for no cap)")
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--min-chars", type=int, default=12)
    args = ap.parse_args()

    caps = dict(DEFAULT_CAPS)
    for tok in (args.caps or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            caps[k.strip()] = None if v.strip().lower() == "all" else int(v)

    seen_q: set = set()
    rows: List[Dict] = []
    counts: Dict[str, int] = {}
    for src in args.sources:
        cap = caps.get(src, DEFAULT_CAPS.get(src))
        try:
            n = 0
            for q, qtype, ext in LOADERS[src](cap, args.seed):
                q = " ".join((q or "").split())
                if len(q) < args.min_chars:
                    continue
                key = q.lower()[:300]
                if key in seen_q:
                    continue
                seen_q.add(key)
                rows.append({
                    "qid": f"{src}-{ext}", "question": q, "source": src,
                    "split": "eval" if src not in ("rl", "sft") else src, "qtype": qtype,
                })
                n += 1
            counts[src] = n
            print(f"[seeds] {src:12s} -> {n}")
        except Exception as e:
            counts[src] = 0
            print(f"[seeds] {src:12s} FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[seeds] TOTAL {len(rows)} unique questions -> {outp}")
    print(f"[seeds] per-source: {counts}")


if __name__ == "__main__":
    main()
