"""Rate each locked-eval query for DECOMPOSABILITY (1-5) and freeze it into the subsets + manifest.

Decomposability = how much a query benefits from being split into independent sub-questions that
separate sub-agents research in parallel — i.e. the axis on which a decompose-and-delegate RLM is
predicted to beat a flat DR-Tulu agent ([[mismanaged-geniuses-hypothesis]] / scaffold-pattern).
This is a per-query DIFFICULTY axis for the difficulty-CONDITIONED analysis (gap-vs-difficulty),
NOT a filter — the sets stay representative.

Rater: the unified Qwen3.5-4B (greedy/thinking-off, served by the wrapper job). Values are frozen
into the manifest, so reproducibility is via the frozen file. We VALIDATE the rating against the
benchmarks' OWN built-in difficulty signals (SQA _breadth_stratum, HealthBench _rubric_count) via
Spearman rho — if the LLM axis tracks those, it's trustworthy.
"""
import json, re, sys, hashlib, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

SUB = Path(__file__).parent.parent / "data" / "subsets"
MANIFEST = SUB / "eval_locked_manifest.json"
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-4B"

SYS = ("You rate how DECOMPOSABLE a research question is — how much it would benefit from being "
       "split into independent sub-questions that separate researchers investigate in parallel.")
RUBRIC = (
    "Rate the REQUEST below on a 1-5 scale by its intrinsic structure (NOT how hard each part is):\n"
    "1 = single focused question; one fact or one tightly-scoped topic; no real sub-questions.\n"
    "2 = two closely-related aspects.\n"
    "3 = a few (~3) distinct sub-topics.\n"
    "4 = several (~4-5) distinct sub-topics, possibly spanning areas.\n"
    "5 = many (6+) distinct sub-topics spanning different areas; highly multi-faceted.\n\n"
    "REQUEST:\n\"\"\"{q}\"\"\"\n\n"
    "Respond with EXACTLY one line: SCORE: <n>   (n is 1, 2, 3, 4, or 5)")

# benchmark -> (filename, id_key, query_fn, built_in_signal_fn or None)
def hb_query(it): return "\n".join(m["content"] for m in it["prompt"] if m.get("role") == "user")
BENCHES = {
    "researchqa":  ("researchqa_strat120.jsonl", "id",        lambda it: it["query"],  None),
    "deep_research_bench": ("drb_en50.jsonl",     "id",        lambda it: it["prompt"], None),
    "healthbench": ("healthbench_strat120.jsonl", "prompt_id", hb_query,                lambda it: it["_rubric_count"]),
    "sqav2":       ("sqav2_cs100.jsonl",          "case_id",   lambda it: it["question"],
                    lambda it: {"low": 1, "medium": 2, "med": 2, "high": 3}.get(str(it.get("_breadth_stratum","")).lower())),
}


def rate_one(query):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 16,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": RUBRIC.format(q=query[:4000])}]}
    for _ in range(3):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=120)
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"SCORE:\s*([1-5])", txt) or re.search(r"\b([1-5])\b", txt)
            if m:
                return int(m.group(1))
        except Exception:
            pass
    return None


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v); s = 0
        while s < len(v):
            e = s
            while e + 1 < len(v) and v[order[e+1]] == v[order[s]]:
                e += 1
            avg = (s + e) / 2 + 1
            for i in range(s, e+1):
                rk[order[i]] = avg
            s = e + 1
        return rk
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    ra, rb = ranks(a), ranks(b)
    n = len(a); ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    va = sum((x-ma)**2 for x in ra) ** .5; vb = sum((x-mb)**2 for x in rb) ** .5
    return round(cov/(va*vb), 3) if va and vb else None


def main():
    manifest = json.loads(MANIFEST.read_text())
    for bench, (fname, idk, qfn, signalfn) in BENCHES.items():
        path = SUB / fname
        # split on "\n" ONLY (read_text already normalizes \r\n); NOT .splitlines(), which also
        # breaks on U+2028/U+2029/U+0085 that appear raw inside HealthBench strings (ensure_ascii=False).
        items = [json.loads(l) for l in path.read_text().split("\n") if l.strip()]
        queries = [qfn(it) for it in items]
        with ThreadPoolExecutor(max_workers=8) as ex:
            scores = list(ex.map(rate_one, queries))
        n_fail = sum(s is None for s in scores)
        for it, s in zip(items, scores):
            it["_decomposability"] = s
        path.write_text("".join(json.dumps(it, ensure_ascii=False, default=str) + "\n" for it in items))

        dist = {k: scores.count(k) for k in range(1, 6)}
        valid = [s for s in scores if s is not None]
        block = {"dist": dist, "mean": round(statistics.mean(valid), 2) if valid else None,
                 "n_failed": n_fail,
                 "by_id": {str(it[idk]): it["_decomposability"] for it in items}}
        if signalfn is not None:
            rho = spearman(scores, [signalfn(it) for it in items])
            sig_name = "_rubric_count" if bench == "healthbench" else "_breadth_stratum"
            block["validation_spearman_vs_" + sig_name] = rho
        manifest.setdefault("benchmarks", {}).setdefault(bench, {})
        manifest["benchmarks"][bench]["decomposability"] = block
        manifest["benchmarks"][bench]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{bench:20s} dist={dist} mean={block['mean']} fail={n_fail}"
              + (f"  rho_vs_builtin={block.get('validation_spearman_vs_'+ ('_rubric_count' if bench=='healthbench' else '_breadth_stratum'))}" if signalfn else ""))

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nfrozen -> {MANIFEST}")


if __name__ == "__main__":
    main()
