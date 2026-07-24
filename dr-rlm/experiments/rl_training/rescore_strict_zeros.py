#!/usr/bin/env python3
"""Diagnose the strict-zero reports: re-score them through the SAME judge path and
capture RAW judge outputs, so we can tell parse/serving failures apart from genuinely
harsh scoring. Usage: rescore_strict_zeros.py <judge_base_url> [n_zero] [n_ctrl]"""
import asyncio, json, sys
import httpx
import pandas as pd

BASE = sys.argv[1].rstrip("/")
N_ZERO = int(sys.argv[2]) if len(sys.argv) > 2 else 40
N_CTRL = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MODEL = "rl-judge-qwen35-4b"
SYS = (
    "You will be given a question someone asked (in <question></question> tags) and the "
    "corresponding response (in <response></response> tags) given to them by an assistant.  "
    "You will then be given a specific criterion of the response to evaluate (in "
    "<criterion></criterion> tags).\n"
    "Return a score on a scale of 0 to 2 indicating how appropriate the response is based on "
    'the given criterion. Judge only the specified aspect(s), not any other qualities of the '
    'answer.  Output JSON in the format: {"score": x}.'
)

rows = [json.loads(l) for l in open("runs/rl_training/credit_metrics_rl_full150.jsonl") if l.strip()]
df = pd.read_parquet("data/rl_full_600/train.parquet")

def root(r):
    for n in (r.get("nodes") or []):
        if (n.get("depth") or 0) == 0:
            return n

def sample(pred, k):
    out = []
    for r in rows:
        rt = root(r)
        if rt and rt.get("ans_len", 0) > 500 and pred(float(r.get("report_reward") or 0)):
            out.append((r, rt))
            if len(out) == k:
                break
    return out

cases = [(r, rt, "zero") for r, rt in sample(lambda R: R == 0, N_ZERO)] + \
        [(r, rt, "ctrl") for r, rt in sample(lambda R: R > 0.3, N_CTRL)]

async def score_one(client, question, report, criterion):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": f"<question>{question}</question>\n<response>{report}</response>\n<criterion>{criterion}</criterion>"}]}
    try:
        resp = await client.post(BASE + "/chat/completions", json=body,
                                 headers={"Authorization": "Bearer EMPTY"}, timeout=180)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"raw": f"<HTTP-ERROR {e}>", "parsed": None}
    import re
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    parsed = None
    if m:
        try:
            parsed = float(json.loads(m.group(0))["score"])
        except Exception:
            parsed = None
    return {"raw": raw[:400], "parsed": parsed}

async def main():
    out = []
    async with httpx.AsyncClient() as client:
        for r, rt, tag in cases:
            iid = int(r["instance_id"])
            spec = df.iloc[iid]["reward_spec"]
            q = spec.get("query") or ""
            rubrics = list(spec["rubrics"])
            crits = []
            for ru in rubrics:
                res = await score_one(client, q, rt["ans_text"], ru["description"])
                crits.append({"criterion": str(ru["description"])[:100], "weight": float(ru.get("weight", 1.0)), **res})
            num = sum((c["parsed"] or 0) / 2.0 * c["weight"] for c in crits)
            den = max(sum(c["weight"] for c in crits if c["weight"] > 0), 1.0)
            out.append({"instance_id": iid, "tag": tag, "orig_R": float(r.get("report_reward") or 0),
                        "rescored_R": min(1.0, num / den), "n_parse_fail": sum(1 for c in crits if c["parsed"] is None),
                        "ans_len": rt["ans_len"], "criteria": crits})
            print(f"[{tag}] iid={iid} orig={out[-1]['orig_R']:.2f} rescored={out[-1]['rescored_R']:.2f} "
                  f"parse_fail={out[-1]['n_parse_fail']}/{len(crits)}")
    with open("runs/rl_training/judge_rescore_diag.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    z = [o for o in out if o["tag"] == "zero"]
    c = [o for o in out if o["tag"] == "ctrl"]
    import statistics as st
    print("\n===== SUMMARY =====")
    print(f"strict-zeros rescored: mean {st.mean(o['rescored_R'] for o in z):.3f} | now >0: {sum(o['rescored_R']>0 for o in z)}/{len(z)} | any parse-fail: {sum(o['n_parse_fail']>0 for o in z)}")
    print(f"controls   rescored: mean {st.mean(o['rescored_R'] for o in c):.3f} vs orig {st.mean(o['orig_R'] for o in c):.3f}")
    print("verdict: high rescored-mean + few parse fails => original zeros were TRANSIENT (serving); ")
    print("         rescored ~0 with sane raw outputs   => judge is genuinely strict; ")
    print("         many parse fails                    => format/parse bug.")

asyncio.run(main())
