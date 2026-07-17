"""Freeze the SHORT-FORM SFT prompt manifest with reference answers (SFT_PLAN Step 1b, D3b).

Takes the corpus-reserved sft-split short-form questions (263 exact_answer + 42 short_form,
already corpus-covered and RL/eval-disjoint by construction), matches each back to its
rl-research/dr-tulu-sft-data row by question text, and extracts the trajectory's final
answer as the REFERENCE answer (their kept trajectories passed gold-match rejection
sampling, so trajectory answer ~= gold). Questions without a recoverable answer are dropped.

Writes data/subsets/sft_prompts_shortform_manifest.json
"""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

DRRLM = Path(__file__).resolve().parents[1]
QUESTIONS = DRRLM / "data/frozen_corpus/questions.jsonl"
OUT = DRRLM / "data/subsets/sft_prompts_shortform_manifest.json"


def extract_answer(conversations) -> str:
    """Final answer from a dr-tulu SFT trajectory: <answer>...</answer>, \\boxed{...} inside
    it if present, else the tail of the last assistant turn."""
    last = ""
    for turn in conversations:
        role = turn.get("role") or turn.get("from")
        if role in ("assistant", "gpt"):
            last = turn.get("content") or turn.get("value") or ""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", last, re.S)
    ans = (m.group(1) if m else last[-500:]).strip()
    b = re.search(r"\\boxed\{(.*?)\}", ans, re.S)
    return (b.group(1) if b else ans).strip()


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def main() -> None:
    wanted = {}
    for line in QUESTIONS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("split") == "sft" and d.get("qtype") in ("exact_answer", "short_form"):
            wanted[norm(d["question"])] = {"qid": d["qid"], "question": d["question"],
                                           "qtype": d["qtype"]}
    print(f"short-form pool from corpus sft split: {len(wanted)}")

    from datasets import load_dataset
    ds = load_dataset("rl-research/dr-tulu-sft-data", split="train")
    matched = 0
    for ex in ds:
        key = norm(ex.get("question", ""))
        if key in wanted and "reference_answer" not in wanted[key]:
            ans = extract_answer(ex.get("conversations") or [])
            if ans:
                wanted[key]["reference_answer"] = ans
                wanted[key]["source"] = ex.get("source", "")
                matched += 1

    entries = sorted((v for v in wanted.values() if v.get("reference_answer")),
                     key=lambda v: v["qid"])
    manifest = {
        "created": str(date.today()),
        "purpose": "Step 1b short-form SFT teacher prompts + reference answers (D3b guardrail mix)",
        "rule": "questions.jsonl split=sft qtype in {exact_answer, short_form}, reference answer recovered from rl-research/dr-tulu-sft-data by question-text match",
        "source_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "pool": len(wanted), "matched": matched, "kept": len(entries),
        "entries": entries,
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"matched {matched}/{len(wanted)}; wrote {len(entries)} entries -> {OUT}")


if __name__ == "__main__":
    main()
