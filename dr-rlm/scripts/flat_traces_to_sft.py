"""Convert flat (DR-Tulu ReAct) generate-dataset output rows into LLaMA-Factory SFT rows,
with the same D6 rejection filters as the recursive arm (SFT_PLAN Step 1).

Target format = dr-tulu's own SFT convention (verified against rl-research/dr-tulu-sft-data):
  [system (workflow's UNIFIED prompt), user (question + framing), assistant (ONE turn: the
  full inline trace with <think>/<call_tool>/<tool_output>/<answer>)]
Their LF fork masks the <tool_output> spans; loss = model's own reasoning + cited answer.

Trace reconstruction: full_traces nests one level per tool round; generated_text of the
DEEPEST non-empty node is the earliest context, the root's is the final segment — concat
deepest -> root.

Filters: <answer> present, <cite present, judge rubric coverage >= threshold (same local
judge + rubrics as the recursive arm), tokenized length <= max_tokens (DROP, never truncate).

Run from dr-tulu/agent (imports dr_agent) in the eval venv:
  python .../flat_traces_to_sft.py --in gen.jsonl --out flat_sft.jsonl \
      --rubrics .../sft_prompts_rubrics.jsonl --framing "<task statement>" \
      --judge_base_url http://localhost:8100/v1 --judge_model gpt-4.1-mini-2025-04-14
"""

import argparse
import importlib.util
import json
import os
import re
import statistics as stats
import sys
from pathlib import Path

DRRLM = Path(__file__).resolve().parents[1]


def load_judge_module():
    """Import judge.py by file path (its package __init__ needs skyrl_gym; the file doesn't)."""
    p = DRRLM / "rl/skyrl/examples/train/dr_rlm/judge.py"
    spec = importlib.util.spec_from_file_location("dr_rlm_judge", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dr_rlm_judge"] = mod  # dataclasses resolve cls.__module__ via sys.modules
    spec.loader.exec_module(mod)
    return mod


def full_trace(ft: dict) -> str:
    """Concatenate generated_text deepest-first (earliest context -> final segment)."""
    chain = []
    node = ft
    while node:
        gt = node.get("generated_text", "") or ""
        if gt:
            chain.append(gt)
        kids = node.get("tool_calls") or []
        node = kids[0] if kids else None
    return "".join(reversed(chain))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rubrics", required=True, help="sft_prompts_rubrics.jsonl (qid -> rubrics)")
    ap.add_argument("--framing", required=True, help="task statement appended to the user turn (must match gen-time DR_TULU_FRAMING_JSON)")
    ap.add_argument("--judge_base_url", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--prompt_version", default=None, help="UNIFIED prompt key; default = workflow default")
    args = ap.parse_args()

    sys.path.insert(0, str(DRRLM.parent / "dr-tulu/agent"))
    from dr_agent.shared_prompts import UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS
    if args.prompt_version is None:
        args.prompt_version = "v20250907"  # auto_search_sft.py default (all agent classes)
    system_prompt = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[args.prompt_version]["system_prompt"]

    judge = load_judge_module()
    jcfg = judge.JudgeConfig.from_env_payload({
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_api_key_env": "JUDGE_API_KEY",
    })

    rubrics_by_qid, rubrics_by_q = {}, {}
    for l in open(args.rubrics):
        if l.strip():
            r = json.loads(l)
            rb = (r.get("reward_spec") or {}).get("rubrics") or []
            rubrics_by_qid[str(r.get("uid"))] = rb
            rubrics_by_q[r.get("question", "").strip()] = rb

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    kept, drops = [], {"no_answer": 0, "no_cite": 0, "low_R": 0, "too_long": 0, "no_rubrics": 0}
    lengths, Rs = [], []
    for r in rows:
        od = r.get("original_data") or {}
        question = od.get("problem") or r.get("problem") or ""
        qid = str(od.get("orig_id") or od.get("id") or "")
        trace = full_trace(r.get("full_traces") or {})
        if "<answer>" not in trace:
            drops["no_answer"] += 1
            continue
        if "<cite" not in trace:
            drops["no_cite"] += 1
            continue
        report = re.split(r"<answer>", trace, 1)[1]
        report = re.split(r"</answer>", report, 1)[0]
        rubrics = rubrics_by_qid.get(qid) or rubrics_by_q.get(question.strip()) or []
        if len(rubrics) < 3:
            drops["no_rubrics"] += 1
            continue
        try:
            R, _ = judge.score_report_sync(report, question, rubrics, jcfg)
        except Exception as ex:
            print(f"judge failed on {qid}: {ex}; R=0", file=sys.stderr)
            R = 0.0
        Rs.append(R)
        if R < args.threshold:
            drops["low_R"] += 1
            continue
        conv = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n\n" + args.framing.strip()},
            {"role": "assistant", "content": trace},
        ]
        n_tok = len(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False))
        lengths.append(n_tok)
        if n_tok > args.max_tokens:
            drops["too_long"] += 1
            continue
        kept.append({"conversations": conv, "_qid": qid, "_R": round(float(R), 4), "_n_tokens": n_tok})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"in={len(rows)} kept={len(kept)} drops={drops}")
    if Rs:
        print(f"judge R: median={stats.median(Rs):.3f} mean={stats.mean(Rs):.3f}")
    if lengths:
        lengths.sort()
        print(f"token lengths: median={int(stats.median(lengths))} p90={lengths[int(len(lengths)*0.9)]} max={max(lengths)}")


if __name__ == "__main__":
    main()
