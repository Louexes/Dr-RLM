"""Post-filter serialized SFT rows (SFT_PLAN Step 1, D6 filters + length rule).

Filters (per row, sharegpt jsonl from gen_recursive_sft.py):
  * tokenized length (student tokenizer, chat template applied) <= --max_tokens
    (rows over are DROPPED, never truncated — truncation cuts the final cited report)
  * final assistant turn contains a <cite  (the citing behavior SFT must install)
  * root rows additionally contain <subagent_output> (the tree actually recursed)
R-threshold and REPL-error filtering already happened inside gen_recursive_sft.

Usage:
  python scripts/filter_sft_rows.py --in merged.jsonl --out filtered.jsonl \
      --tokenizer Qwen/Qwen3.5-4B [--max_tokens 32768]
"""

import argparse
import json
import statistics as stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max_tokens", type=int, default=32768)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    kept, drops = [], {"no_cite": 0, "not_finalized": 0, "no_recursion": 0, "too_long": 0}
    lengths = []
    for r in rows:
        conv = r["conversations"]
        node = r.get("_node", "root")
        # repair rows serialized before the opening-<think> restoration (tag lived in the
        # generation prompt, so completions carry only '</think>')
        for m in conv:
            if m["role"] == "assistant" and "</think>" in m["content"] \
                    and not m["content"].lstrip().startswith("<think>"):
                m["content"] = "<think>\n" + m["content"].lstrip()
        asst = "\n".join(m["content"] for m in conv if m["role"] == "assistant")
        # structured contract: citations are answer["citations"]=[{"id":...}] entries in the
        # model's own REPL code (the <cite> tags are assembled downstream); accept inline
        # <cite too for prose-mode compat
        cited = "<cite" in asst or ('answer["citations"]' in asst and '"id"' in asst)
        if not cited:
            drops["no_cite"] += 1
            continue
        if 'answer["ready"]' not in asst and "FINAL" not in asst:
            drops["not_finalized"] += 1
            continue
        if node == "root" and not any("<subagent_output>" in m["content"] for m in conv):
            drops["no_recursion"] += 1
            continue
        n_tok = len(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False))
        lengths.append(n_tok)
        if n_tok > args.max_tokens:
            drops["too_long"] += 1
            continue
        r["_n_tokens"] = n_tok
        kept.append(r)

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_node = {"root": 0, "child": 0}
    for r in kept:
        by_node[r.get("_node", "root")] = by_node.get(r.get("_node", "root"), 0) + 1
    print(f"in={len(rows)} kept={len(kept)} (root={by_node['root']}, child={by_node['child']}) drops={drops}")
    if lengths:
        lengths.sort()
        print(f"token lengths (post-cite/recursion filters): median={int(stats.median(lengths))} "
              f"p90={lengths[int(len(lengths)*0.9)]} max={max(lengths)}")


if __name__ == "__main__":
    main()
