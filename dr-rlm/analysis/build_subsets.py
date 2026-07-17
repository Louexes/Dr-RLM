"""Build deterministic W1 evaluation subsets:
  - DRB: all 50 English-language items (drops 50 Chinese)
  - ResearchQA: 7 items per general_domain × 7 domains = 49 items, prefer high-rubric-count within each domain

Inputs come from HF; outputs are written to ./subsets/ as JSONL for downstream eval scripts.
Run from anywhere; uses absolute output paths.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
OUT_DIR = Path(__file__).parent / "subsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_drb_en50():
    ds = load_dataset("rl-research/deep_research_bench_eval", split="test")
    en = [dict(x) for x in ds if x["language"] == "en"]
    en.sort(key=lambda x: x["id"])
    out_path = OUT_DIR / "drb_en50.jsonl"
    with out_path.open("w") as f:
        for item in en:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return en, out_path


def _rubric_count(rubric_field) -> int:
    if isinstance(rubric_field, list):
        return len(rubric_field)
    if isinstance(rubric_field, str):
        try:
            return len(json.loads(rubric_field))
        except json.JSONDecodeError:
            return -1
    return -1


def build_researchqa_strat49():
    ds = load_dataset("realliyifei/ResearchQA", split="test_mini")
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for x in ds:
        item = dict(x)
        item["_rubric_count"] = _rubric_count(item["rubric"])
        by_domain[item["general_domain"]].append(item)

    rng = random.Random(SEED)
    picked = []
    for domain in sorted(by_domain.keys()):
        candidates = by_domain[domain]
        candidates.sort(key=lambda it: (-it["_rubric_count"], it["id"]))
        top_pool_size = max(7, min(20, len(candidates)))
        top_pool = candidates[:top_pool_size]
        rng.shuffle(top_pool)
        chosen = top_pool[:7]
        chosen.sort(key=lambda it: it["id"])
        picked.extend(chosen)

    for item in picked:
        item.pop("_rubric_count", None)

    out_path = OUT_DIR / "researchqa_strat49.jsonl"
    with out_path.open("w") as f:
        for item in picked:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    return picked, out_path


def summarize(name, items, group_key):
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it[group_key]] += 1
    print(f"\n{name}: {len(items)} items, grouped by {group_key}")
    for k in sorted(counts):
        print(f"  {k:35s}  {counts[k]}")


def main():
    drb, drb_path = build_drb_en50()
    summarize("DRB (English only)", drb, "topic")
    print(f"  -> {drb_path}")

    rqa, rqa_path = build_researchqa_strat49()
    summarize("ResearchQA (stratified)", rqa, "general_domain")
    print(f"  -> {rqa_path}")

    print(f"\nTotal eval items: {len(drb) + len(rqa)}")


if __name__ == "__main__":
    main()
