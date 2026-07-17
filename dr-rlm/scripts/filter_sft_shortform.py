"""Answer-equivalence rejection filter for SHORT-FORM SFT rows (SFT_PLAN Step 1b, D3b).

Mirrors DR-Tulu's short-form gold-match rejection sampling (App. F.1.2): keep a teacher
trajectory iff its final answer is equivalent to the reference answer (LLM-judge verdict,
local thinking-OFF judge). Works on both arms:
  * recursive rows (gen_recursive_sft jsonl): candidate answer = row["_response"]
    (roots only; child rows of kept roots pass through with the root's verdict)
  * flat rows (auto_search_sft output jsonl): candidate = <answer>...</answer> from the
    reconstructed trace (\\boxed{} unwrapped)
Also applies the <=max_tokens drop rule and (recursive) the <think> repair.

Usage:
  python scripts/filter_sft_shortform.py --arm recursive|flat --in gen.jsonl --out sft.jsonl \
      --manifest data/subsets/sft_prompts_shortform_manifest.json \
      --judge_base_url http://localhost:8100/v1 --judge_model gpt-4.1-mini-2025-04-14
"""

import argparse
import json
import re
import statistics as stats
import sys

import requests

JUDGE_PROMPT = """Question: {question}

Reference answer: {reference}

Candidate answer: {candidate}

Is the candidate answer equivalent to the reference answer for this question (same fact/entity/value, wording may differ)? Reply with exactly one word: YES or NO."""


def judge_equivalent(base_url, model, question, reference, candidate) -> bool:
    r = requests.post(base_url.rstrip("/") + "/chat/completions", json={
        "model": model, "temperature": 0.0, "max_tokens": 8,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            question=question[:2000], reference=reference[:1000], candidate=candidate[:1000])}],
    }, headers={"Authorization": "Bearer EMPTY"}, timeout=120)
    r.raise_for_status()
    return "YES" in (r.json()["choices"][0]["message"]["content"] or "").upper()


def flat_candidate(row) -> str:
    node, chain = row.get("full_traces") or {}, []
    while node:
        gt = node.get("generated_text", "") or ""
        if gt:
            chain.append(gt)
        kids = node.get("tool_calls") or []
        node = kids[0] if kids else None
    trace = "".join(reversed(chain))
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", trace, re.S)
    ans = (m.group(1) if m else "").strip()
    b = re.search(r"\\boxed\{(.*?)\}", ans, re.S)
    return (b.group(1) if b else ans).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["recursive", "flat"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--judge_base_url", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max_tokens", type=int, default=32768)
    args = ap.parse_args()

    ref = {e["qid"]: e for e in json.load(open(args.manifest))["entries"]}
    ref_by_q = {re.sub(r"\s+", " ", e["question"].strip().lower()): e for e in ref.values()}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    kept, drops = [], {"no_ref": 0, "no_answer": 0, "not_equivalent": 0, "too_long": 0}
    verdict_by_qid = {}

    def repair_think(conv):
        for m in conv:
            if m["role"] == "assistant" and "</think>" in m["content"] \
                    and not m["content"].lstrip().startswith("<think>"):
                m["content"] = "<think>\n" + m["content"].lstrip()

    for r in rows:
        if args.arm == "recursive":
            # root rows carry the manifest qid verbatim; child rows append "-c{i}".
            # (naive split("-c") mangled qids whose hex hash starts with 'c')
            qid = str(r.get("_qid", ""))
            if r.get("_node") == "child":
                qid = qid.rsplit("-c", 1)[0]
            e = ref.get(qid)
            if not e:
                drops["no_ref"] += 1
                continue
            if r.get("_node") == "child":
                if verdict_by_qid.get(qid):
                    repair_think(r["conversations"])
                    n_tok = len(tok.apply_chat_template(r["conversations"], tokenize=True,
                                                        add_generation_prompt=False))
                    if n_tok <= args.max_tokens:
                        r["_n_tokens"] = n_tok
                        kept.append(r)
                    else:
                        drops["too_long"] += 1
                continue
            cand = (r.get("_response") or "").strip()
        else:
            od = r.get("original_data") or {}
            qid = str(od.get("orig_id") or od.get("id") or "")
            e = ref.get(qid) or ref_by_q.get(
                re.sub(r"\s+", " ", str(od.get("problem", "")).strip().lower()))
            if not e:
                drops["no_ref"] += 1
                continue
            cand = flat_candidate(r)
        if not cand:
            drops["no_answer"] += 1
            continue
        try:
            ok = judge_equivalent(args.judge_base_url, args.judge_model,
                                  e["question"], e["reference_answer"], cand)
        except Exception as ex:
            print(f"judge failed on {qid}: {ex}; dropping", file=sys.stderr)
            ok = False
        verdict_by_qid[e["qid"]] = ok
        if not ok:
            drops["not_equivalent"] += 1
            continue
        if args.arm == "recursive":
            conv = r["conversations"]
            repair_think(conv)
        else:
            sys.path.insert(0, "/gpfs/home5/lgehringer/Dr-RLM/dr-tulu/agent")
            from dr_agent.shared_prompts import UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS
            node, chain = r.get("full_traces") or {}, []
            while node:
                gt = node.get("generated_text", "") or ""
                if gt:
                    chain.append(gt)
                kids = node.get("tool_calls") or []
                node = kids[0] if kids else None
            conv = [
                {"role": "system",
                 "content": UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS["v20250907"]["system_prompt"]},
                {"role": "user", "content": (r.get("original_data") or {}).get("problem", "")
                    + "\n\n" + (r.get("original_data") or {}).get("additional_instructions", "")},
                {"role": "assistant", "content": "".join(reversed(chain))},
            ]
            r = {"conversations": conv, "_qid": e["qid"]}
        n_tok = len(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False))
        if n_tok > args.max_tokens:
            drops["too_long"] += 1
            continue
        r["_n_tokens"] = n_tok
        r["_qtype"] = e["qtype"]
        kept.append(r)

    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    n_eq = sum(verdict_by_qid.values())
    print(f"in={len(rows)} kept={len(kept)} equivalent={n_eq}/{len(verdict_by_qid)} drops={drops}")
    toks = [k["_n_tokens"] for k in kept if "_n_tokens" in k]
    if toks:
        print(f"token lengths: median={int(stats.median(toks))} max={max(toks)}")


if __name__ == "__main__":
    main()
