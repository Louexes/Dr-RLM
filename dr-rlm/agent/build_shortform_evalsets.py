"""Build coverage-safe short-form eval subsets for the untrained DR-RLM guardrail eval.

Eval questions MUST be a subset of the corpus-seeded questions, or retrieval goes thin.
We therefore derive each subset directly from the frozen corpus's questions.jsonl (the
exact seeded question text, whitespace-normalized as the crawler saw it) and attach gold
answers by matching that text back to the HF/CSV source. Output schema matches what
DR-Tulu's grader reads from a row (evaluate.py convert_to_evaluate_format -> row):
  simpleqa        -> {id, problem, answer}   (SimpleQAEval reads row["answer"], singular)
  2wiki/webwalker -> {id, problem, answers}  (ShortFormQAEval reads row["answers"], list)

Run on a login/CPU node (needs the crawl venv for `datasets`/`pandas`). Deterministic.
"""
from __future__ import annotations

import csv, hashlib, io, json, sys
from pathlib import Path

DRRLM = Path("/gpfs/home5/lgehringer/Dr-RLM/dr-rlm")
QJSONL = DRRLM / "data/frozen_corpus/questions.jsonl"
SUBSETS = DRRLM / "data/subsets"
N = 120  # locked dev-scale, parity with researchqa_strat120; reused for the trained eval


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _seeded(source: str) -> list[str]:
    """Seeded question texts for a source, in questions.jsonl order (already normalized)."""
    out = []
    for line in QJSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["source"] == source:
            out.append(r["question"])
    return out


def _mid(problem: str) -> str:
    return hashlib.md5(problem.encode()).hexdigest()


SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def build_simpleqa() -> Path:
    # avoid importing dr_agent (drags in litellm); the crawl venv already cached this CSV
    cache = DRRLM / "data/frozen_corpus/_simpleqa_test_set.csv"
    if cache.exists():
        text = cache.read_text()
    else:
        import httpx
        text = httpx.get(SIMPLEQA_URL, timeout=60, follow_redirects=True).text
    gold = {_norm(r["problem"]): r["answer"] for r in csv.DictReader(io.StringIO(text))}
    rows = []
    for q in _seeded("simpleqa"):
        if q in gold:
            rows.append({"id": _mid(q), "problem": q, "answer": gold[q]})
        if len(rows) >= N:
            break
    return _write("simpleqa_eval120.jsonl", rows, "simpleqa")


def _build_hf(source: str, repo: str, out_name: str, english_only: bool) -> Path:
    import datasets
    ds = datasets.load_dataset(repo, split="test")
    gold = {}
    for ex in ds:
        q = _norm(ex["messages"][-1]["content"])
        gt = ex["ground_truth"]
        gold[q] = json.loads(gt) if gt and gt[0] == "[" else [gt]
    rows = []
    for q in _seeded(source):
        if english_only and any(ord(c) > 127 for c in q):
            continue
        if q in gold:
            rows.append({"id": _mid(q), "problem": q, "answers": gold[q]})
        if len(rows) >= N:
            break
    return _write(out_name, rows, source)


def _write(name: str, rows: list[dict], source: str) -> Path:
    p = SUBSETS / name
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # coverage guard: every problem must be a seeded question for this source
    seeded = set(_seeded(source))
    missing = [r["problem"] for r in rows if r["problem"] not in seeded]
    assert not missing, f"{name}: {len(missing)} problems NOT in seeded corpus (thin-retrieval risk)"
    print(f"[ok] {name}: {len(rows)} rows, all ⊆ seeded {source} corpus -> {p}")
    return p


if __name__ == "__main__":
    build_simpleqa()
    _build_hf("2wiki", "akariasai/2wiki_rand1k", "2wiki_eval120.jsonl", english_only=False)
    _build_hf("webwalker", "rl-research/webwalker_test", "webwalker_eval120.jsonl", english_only=True)
    print("[done] short-form eval subsets built")
