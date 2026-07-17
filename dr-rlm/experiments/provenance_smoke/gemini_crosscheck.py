#!/usr/bin/env python3
"""Oracle cross-check: score the SAME strict-zero reports (that the local 4B judge zeroed)
with Gemini, using the IDENTICAL criterion prompt. Resolves 'genuinely bad reports' vs
'4B judge miscalibrated'. Off-GPU, one-off diagnostic — NOT re-introducing Gemini to training.

Runs on int-node with internet. Reads the same instance_ids the local rescore used so the
two judges score byte-identical (report, criterion) pairs."""
import json, os, re, time
from openai import OpenAI

# same instance_ids the local diagnostic scored, split by tag
diag = [json.loads(l) for l in open("runs/provenance_smoke/judge_rescore_diag.jsonl")]
zero_ids = [(d["instance_id"], d["orig_R"]) for d in diag if d["tag"] == "zero"]
ctrl_ids = [(d["instance_id"], d["orig_R"]) for d in diag if d["tag"] == "ctrl"]

import pandas as pd
df = pd.read_parquet("data/rl_full_600/train.parquet")
rows = [json.loads(l) for l in open("runs/provenance_smoke/credit_metrics_rl_full150.jsonl") if l.strip()]

def root(r):
    for n in (r.get("nodes") or []):
        if (n.get("depth") or 0) == 0:
            return n

# map instance_id -> the FIRST substantive report we can find (same selection order as rescore)
def report_for(iid, want_zero):
    for r in rows:
        if int(r["instance_id"]) != iid:
            continue
        rt = root(r)
        if rt and rt.get("ans_len", 0) > 500:
            is_zero = float(r.get("report_reward") or 0) == 0
            if is_zero == want_zero:
                return rt["ans_text"]
    return None

SYS = (
    "You will be given a question someone asked (in <question></question> tags) and the "
    "corresponding response (in <response></response> tags) given to them by an assistant.  "
    "You will then be given a specific criterion of the response to evaluate (in "
    "<criterion></criterion> tags).\n"
    "Return a score on a scale of 0 to 2 indicating how appropriate the response is based on "
    'the given criterion. Judge only the specified aspect(s), not any other qualities of the '
    'answer.  Output JSON in the format: {"score": x}.'
)

import urllib.request
key = None
for line in open("keys.sh"):
    if "GEMINI_API_KEY" in line and line.strip().startswith("export"):
        m0 = re.search(r'"([^"]+)"', line)
        if m0 and m0.group(1) != "$GEMINI_API_KEY":
            key = m0.group(1).strip()
            break
assert key and "\n" not in key and '"' not in key, f"bad key: {key!r}"
# NOTE: the AQ.* key format is rejected by the OpenAI-compat Bearer path; native ?key= works.
# thinkingBudget=0 disables 2.5-flash reasoning so content isn't eaten by thinking tokens.
_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"

def gem_score(q, report, crit):
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": SYS}]},
        "contents": [{"parts": [{"text": f"<question>{q}</question>\n<response>{report}</response>\n<criterion>{crit}</criterion>"}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                             "maxOutputTokens": 2048, "thinkingConfig": {"thinkingBudget": 0}},
    }).encode()
    last = ""
    for att in range(6):
        try:
            req = urllib.request.Request(_URL, data=body, headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
            cand = (resp.get("candidates") or [{}])[0]
            try:
                txt = cand["content"]["parts"][0]["text"]
            except Exception:
                txt = ""
            m = re.search(r"\{.*?\}", txt, re.DOTALL)
            if m:
                return float(json.loads(m.group(0))["score"])
            last = f"empty/no-json finish={cand.get('finishReason')}"  # retryable, not a hard 0
        except Exception as e:
            last = str(e)[:120]
        time.sleep(1.5 * (att + 1))
    print("  gem FAIL:", last); return None

def score_set(ids, want_zero, label):
    out = []
    for iid, origR in ids:
        rep = report_for(iid, want_zero)
        if rep is None:
            continue
        spec = df.iloc[iid]["reward_spec"]
        q = spec.get("query") or ""
        crits = list(spec["rubrics"])
        s = [gem_score(q, rep, c["description"]) for c in crits]
        w = [float(c.get("weight", 1.0)) for c in crits]
        num = sum((sv or 0) / 2.0 * wv for sv, wv in zip(s, w))
        den = max(sum(wv for wv in w if wv > 0), 1.0)
        R = min(1.0, num / den)
        out.append({"iid": iid, "local_R": origR, "gemini_R": R,
                    "per": [(round((sv or -1) / 2, 2)) for sv in s]})
        print(f"[{label}] iid={iid} local={origR:.2f} gemini={R:.2f} per={out[-1]['per']}")
    return out

print("=== scoring 40 local-strict-ZEROS with Gemini ===")
z = score_set(zero_ids, True, "zero")
print("\n=== scoring 10 controls with Gemini ===")
c = score_set(ctrl_ids, False, "ctrl")

import statistics as st
json.dump({"zero": z, "ctrl": c}, open("runs/provenance_smoke/gemini_crosscheck.json", "w"), indent=1)
print("\n===== CROSS-CHECK SUMMARY =====")
print(f"local judge zeroed these (mean local_R = 0.00 by construction)")
print(f"GEMINI on same reports: mean {st.mean(o['gemini_R'] for o in z):.3f} | now >0: {sum(o['gemini_R']>0 for o in z)}/{len(z)} | >0.3: {sum(o['gemini_R']>0.3 for o in z)}/{len(z)}")
print(f"controls: local mean {st.mean(o['local_R'] for o in c):.3f} vs gemini mean {st.mean(o['gemini_R'] for o in c):.3f}")
print("\nINTERPRET: gemini also ~0  => reports genuinely weak, local signal HONEST, continue.")
print("           gemini >>0       => 4B judge MISCALIBRATED, full run trains on degraded reward.")
