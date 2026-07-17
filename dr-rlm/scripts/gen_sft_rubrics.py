"""Draft initial search-based rubrics for the frozen SFT prompts (SFT_PLAN Step 1).

Mirrors DR-Tulu App. F.2 ("initial rubrics from top-10 retrieved docs") with local pieces:
retrieval = the SAME bm25s frozen-corpus backend the harness uses; rubric writer = the
local Qwen3.6-27B teacher already being served. Rubrics here only gate SFT rejection
sampling (data selection), never a training reward.

Reads  data/subsets/sft_prompts_manifest.json
Writes data/sft/sft_prompts_rubrics.jsonl   (SkyRL dr_rlm row schema, reward_spec.rubrics
                                             filled — direct input to gen_recursive_sft.py)

Usage:
  python scripts/gen_sft_rubrics.py --base_url http://localhost:8006/v1 \
      --model Qwen/Qwen3.6-27B [--limit N]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests

DRRLM = Path(__file__).resolve().parents[1]
MANIFEST = DRRLM / "data/subsets/sft_prompts_manifest.json"
INDEX = DRRLM / "data/frozen_corpus/bm25s_index"
OUT = DRRLM / "data/sft/sft_prompts_rubrics.jsonl"

PROMPT = """You are designing grading rubrics for a research question. Based on the question and the retrieved reference material below, write 5-8 rubric criteria that a comprehensive, well-grounded research report answering this question should satisfy. Ground the criteria in specific facts/aspects from the reference material where possible; make them discriminative (a shallow report fails them, a thorough one passes).

Question:
{question}

Reference material (top retrieved snippets):
{snippets}

Return ONLY a JSON array, each element {{"title": <short name>, "description": <one-sentence checkable criterion>, "weight": 1.0}}. No prose, no markdown fence."""


def load_retriever():
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25.load(str(INDEX), load_corpus=True)

    def search(query: str, k: int = 10):
        toks = bm25s.tokenize(str(query), stopwords="en", stemmer=stemmer, show_progress=False)
        docs, scores = retriever.retrieve(toks, k=k, show_progress=False)
        return [d.get("contents", "")[:1500] for d in docs[0] if isinstance(d, dict)]

    return search


def draft_rubrics(base_url: str, model: str, question: str, snippets: list) -> list:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(
            question=question, snippets="\n\n---\n\n".join(snippets) or "(none retrieved)")}],
        "temperature": 0.2,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(base_url.rstrip("/") + "/chat/completions",
                      json=body, headers={"Authorization": "Bearer EMPTY"}, timeout=300)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"] or ""
    m = re.search(r"\[.*\]", text, re.S)
    rubrics = json.loads(m.group(0)) if m else []
    out = []
    for rb in rubrics:
        if isinstance(rb, dict) and rb.get("description"):
            out.append({"title": str(rb.get("title", ""))[:120],
                        "description": str(rb["description"]),
                        "weight": float(rb.get("weight", 1.0) or 1.0)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    entries = json.loads(MANIFEST.read_text())["entries"]
    if args.limit:
        entries = entries[: args.limit]
    search = load_retriever()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with OUT.open("w") as f:
        for i, e in enumerate(entries):
            q = e["question"]
            try:
                rubrics = draft_rubrics(args.base_url, args.model, q, search(q))
            except Exception as ex:
                print(f"[{i}] {e['qid']}: rubric draft FAILED ({ex}); writing empty rubrics", file=sys.stderr)
                rubrics = []
            if len(rubrics) >= 3:
                n_ok += 1
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": q}],
                "question": q,
                "env_class": "dr_rlm",
                "reward_spec": {"ground_truth": "", "rubrics": rubrics},
                "uid": e["qid"],
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(entries)} done ({n_ok} with >=3 rubrics)")
    print(f"wrote {OUT}: {len(entries)} rows, {n_ok} with >=3 rubrics")


if __name__ == "__main__":
    main()
