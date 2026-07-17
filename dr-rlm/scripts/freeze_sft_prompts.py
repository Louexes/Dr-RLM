"""Freeze the SFT prompt manifest (SFT_PLAN_2026-07-03.md Step 0.1).

Selects N long-form prompts from the frozen-corpus question pool's reserved
``split=sft`` questions (which came from rl-research/dr-tulu-sft-data at corpus-build
time, so they are corpus-covered by construction and disjoint from dr-tulu's RL pool
by dr-tulu's own construction). Verifies disjointness against the 40-prompt RL train
set and the locked eval subsets anyway, then writes:

  data/subsets/sft_prompts_manifest.json   (the frozen manifest, eval_locked style)
  data/subsets/sft_probe_smoke3.jsonl      (3 rows, SkyRL dr_rlm schema, for the
                                            gen_recursive_sft.py serialization smoke)

Deterministic: seed 42 over the qid-sorted pool.
"""

import hashlib
import json
import random
import re
from datetime import date
from pathlib import Path

DRRLM = Path(__file__).resolve().parents[1]
QUESTIONS = DRRLM / "data/frozen_corpus/questions.jsonl"
RL40 = DRRLM / "data/subsets/drtulu_rl_decompose40.jsonl"
EVAL_MANIFEST = DRRLM / "data/subsets/eval_locked_manifest.json"
OUT_MANIFEST = DRRLM / "data/subsets/sft_prompts_manifest.json"
OUT_SMOKE = DRRLM / "data/subsets/sft_probe_smoke3.jsonl"

SEED, N = 42, 200


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def main() -> None:
    pool = []
    for line in QUESTIONS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("split") == "sft" and d.get("qtype") == "long_form":
            pool.append({"qid": d["qid"], "question": d["question"]})
    pool.sort(key=lambda d: d["qid"])

    rl_texts = {
        norm(json.loads(l).get("problem", ""))
        for l in RL40.read_text().splitlines()
        if l.strip()
    }
    eval_texts = set()
    ev = json.loads(EVAL_MANIFEST.read_text())
    for bench in ev.get("benchmarks", {}).values():
        if "file" not in bench:  # drb/sqav2 entries carry only metadata; different sources anyway
            continue
        f = Path(bench["file"])
        if f.exists():
            text = f.read_text()
            try:  # some subsets are a single JSON array, not jsonl
                rows = json.loads(text)
                if not isinstance(rows, list):
                    rows = [rows]
            except json.JSONDecodeError:
                rows = []
                for l in text.splitlines():
                    if l.strip():
                        try:
                            rows.append(json.loads(l))
                        except json.JSONDecodeError:
                            continue
            for r in rows:
                    q = r.get("question") or r.get("prompt") or r.get("query") or ""
                    if isinstance(q, list):  # healthbench: prompt = chat message list
                        q = next((m.get("content", "") for m in reversed(q)
                                  if isinstance(m, dict) and m.get("role") == "user"), "")
                    eval_texts.add(norm(q))

    clean = [
        p for p in pool
        if norm(p["question"]) not in rl_texts and norm(p["question"]) not in eval_texts
    ]
    dropped = len(pool) - len(clean)

    rng = random.Random(SEED)
    picked = rng.sample(clean, N)
    picked.sort(key=lambda d: d["qid"])

    manifest = {
        "created": str(date.today()),
        "purpose": "SFT teacher-generation prompts (SFT_PLAN_2026-07-03.md, D3: held out from RL + eval)",
        "seed": SEED,
        "N": N,
        "rule": "questions.jsonl split=sft qtype=long_form, qid-sorted, minus RL-40/eval-locked text collisions, random.Random(seed).sample",
        "source_file": str(QUESTIONS),
        "source_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "pool_size": len(pool),
        "collisions_dropped": dropped,
        "entries": picked,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    with OUT_SMOKE.open("w") as f:
        for p in picked[:3]:
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": p["question"]}],
                "question": p["question"],
                "env_class": "dr_rlm",
                "reward_spec": {"ground_truth": "", "rubrics": []},
                "uid": p["qid"],
            }, ensure_ascii=False) + "\n")

    print(f"pool={len(pool)} dropped_collisions={dropped} picked={len(picked)}")
    print(f"wrote {OUT_MANIFEST}")
    print(f"wrote {OUT_SMOKE}")


if __name__ == "__main__":
    main()
