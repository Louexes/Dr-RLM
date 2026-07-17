"""Re-serialize flat DR-Tulu SFT rows from single-assistant-fulltrace into true multi-turn,
so SkyRL's message-level loss masking (ALL_ASSISTANT_MESSAGES) excludes the retrieved
<tool_output> snippets — matching (a) DR-Tulu's LF span-masking design choice and (b) the
recursive arm's masking exactly. Changes only serialization, not the SFT design.

Input rows:  {conversations:[{system},{user:question},{assistant:FULLTRACE}], ...}
  where FULLTRACE = reasoning</think> <call_tool ...>q</call_tool> <tool_output>...</tool_output>
                    reasoning</think> ... <answer>...</answer>
Output rows: {conversations:[{system},{user:question},
                             {assistant: reasoning+call_tool}, {user: <tool_output>...},
                             ..., {assistant: reasoning+answer}], ...}

Assistant turns are formatted like the recursive arm: prepend "<think>\n" when a turn has a
</think> but no opening tag (same repair gen_recursive_sft / filter_sft_rows apply), so both
arms render identically under the Qwen3 chat template.

Usage:
  python scripts/flat_to_multiturn.py --in data/sft/primary/dr_tulu_flat_primary.jsonl \
      --out data/sft/primary/dr_tulu_flat_primary_multiturn.jsonl
"""

import argparse
import json
import re

TOOL_OUTPUT_RE = re.compile(r"<tool_output>.*?</tool_output>", re.S)


def repair_think(text: str) -> str:
    t = text.strip()
    if "</think>" in t and not t.lstrip().startswith("<think>"):
        t = "<think>\n" + t
    return t


def split_fulltrace(asst):
    """Split one assistant fulltrace into alternating assistant/user turns."""
    turns, pos = [], 0
    for m in TOOL_OUTPUT_RE.finditer(asst):
        seg = asst[pos:m.start()].strip()          # reasoning + <call_tool>...</call_tool>
        if seg:
            turns.append({"role": "assistant", "content": repair_think(seg)})
        turns.append({"role": "user", "content": m.group(0).strip()})  # <tool_output>...  (masked)
        pos = m.end()
    tail = asst[pos:].strip()                        # final reasoning + <answer>...</answer>
    if tail:
        turns.append({"role": "assistant", "content": repair_think(tail)})
    return turns


def reconvert(row):
    conv = row["conversations"]
    sys_u = [m for m in conv if m["role"] in ("system", "user")]
    asst = next((m["content"] for m in conv if m["role"] == "assistant"), None)
    if asst is None:
        return None
    turns = split_fulltrace(asst)
    if not turns or turns[-1]["role"] != "assistant":
        return None  # must end on an assistant turn (SkyRL requirement)
    row = dict(row)
    row["conversations"] = sys_u + turns
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    out, dropped = [], 0
    for r in rows:
        rc = reconvert(r)
        if rc is None:
            dropped += 1
            continue
        out.append(rc)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # quick shape report
    import statistics as st
    turns = [len(r["conversations"]) for r in out]
    print(f"in={len(rows)} out={len(out)} dropped={dropped}")
    print(f"turns/convo: min={min(turns)} median={int(st.median(turns))} max={max(turns)}")


if __name__ == "__main__":
    main()
