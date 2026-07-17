"""Build the LOCKED thesis eval subsets (N=120 each) for ResearchQA + HealthBench-hard.

Design (see EXPERIMENT_LOG 2026-06-28):
  * N=120 per benchmark — minimal-but-significant: with per-item coverage SD~0.21, gives a
    95% CI half-width of +-0.038 per arm and detects a paired arm-vs-arm delta ~0.05 at 80%
    power (W1 deltas were ~0.15). DRB(50 en, full) + SQAv2(100, full) need no resampling.
  * PROPORTIONAL stratified sampling (representative: aggregate = unbiased estimate of the
    full-benchmark metric), largest-remainder allocation, floor 7/stratum so every stratum
    is present for per-group glances.
      - ResearchQA: stratify on general_domain (7) over test_mini (776).
      - HealthBench: stratify on theme (7) over hard split (1000).
  * Within a stratum: keep gradeable items (rubric_count >= 3), then uniform-random sample
    (seed 42). Deterministic + reproducible; a manifest records params, per-stratum counts,
    sha256, and the full id list so the set is LOCKED.
"""
import json, hashlib, random, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
N = 120
MIN_RUBRICS = 3
FLOOR = 7
OUT = Path(__file__).parent.parent / "data" / "subsets"
HB_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"


def largest_remainder(counts: dict, n: int, floor: int) -> dict:
    """Proportional allocation of n across strata by population, with a per-stratum floor."""
    total = sum(counts.values())
    raw = {k: n * v / total for k, v in counts.items()}
    alloc = {k: max(floor, int(raw[k])) for k in counts}
    # fix the sum to exactly n via fractional remainders (respecting the floor on decreases)
    while sum(alloc.values()) != n:
        diff = n - sum(alloc.values())
        if diff > 0:  # hand out to largest fractional remainders
            k = max(counts, key=lambda k: raw[k] - int(raw[k]) if alloc[k] == int(raw[k]) or alloc[k] > floor else -1)
            order = sorted(counts, key=lambda k: -(raw[k] - alloc[k]))
            alloc[order[0]] += 1
        else:        # take back from the most over-allocated above floor
            order = sorted((k for k in counts if alloc[k] > floor), key=lambda k: (alloc[k] - raw[k]), reverse=True)
            alloc[order[0]] -= 1
    return alloc


def rubric_count(item, field="rubric"):
    r = item.get(field, item.get("rubrics"))
    if isinstance(r, list):
        return len(r)
    if isinstance(r, str):
        try: return len(json.loads(r))
        except Exception: return 0
    return 0


def sample_stratified(pool_by_stratum: dict, alloc: dict, id_key: str) -> list:
    rng = random.Random(SEED)
    picked = []
    for stratum in sorted(alloc):
        cands = pool_by_stratum[stratum]
        gradeable = [c for c in cands if rubric_count(c) >= MIN_RUBRICS] or cands
        gradeable = sorted(gradeable, key=lambda c: str(c[id_key]))  # stable order before sampling
        k = min(alloc[stratum], len(gradeable))
        picked.extend(rng.sample(gradeable, k))
    picked.sort(key=lambda c: str(c[id_key]))
    return picked


def write_locked(items, name, strata_key, id_key, extra_annotate=None):
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for it in items:
            if extra_annotate:
                extra_annotate(it)
            f.write(json.dumps(it, ensure_ascii=False, default=str) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    comp = Counter(it[strata_key] for it in items)
    return {
        "file": str(path), "n": len(items), "sha256": sha,
        "strata_key": strata_key, "composition": dict(sorted(comp.items())),
        "ids": [str(it[id_key]) for it in items],
    }


def main():
    manifest = {"seed": SEED, "N": N, "min_rubrics": MIN_RUBRICS, "floor": FLOOR, "benchmarks": {}}

    # ---------- ResearchQA (test_mini, stratify general_domain) ----------
    from datasets import load_dataset
    ds = load_dataset("realliyifei/ResearchQA", split="test_mini")
    by_dom = defaultdict(list)
    for x in ds:
        by_dom[x["general_domain"]].append(dict(x))
    counts = {k: len(v) for k, v in by_dom.items()}
    alloc = largest_remainder(counts, N, FLOOR)
    print(f"ResearchQA pool {counts} (total {sum(counts.values())})")
    print(f"  -> alloc {alloc} (sum {sum(alloc.values())})")
    rqa = sample_stratified(by_dom, alloc, id_key="id")
    manifest["benchmarks"]["researchqa"] = write_locked(rqa, "researchqa_strat120", "general_domain", "id")

    # ---------- HealthBench-hard (stratify theme) ----------
    raw = urllib.request.urlopen(HB_URL, timeout=120).read().decode()
    hb = [json.loads(l) for l in raw.splitlines() if l.strip()]
    def theme_of(ex):
        for t in ex.get("example_tags", []):
            if t.startswith("theme:"): return t.split(":", 1)[1]
        return "unknown"
    def nturns(ex):
        return sum(1 for m in ex.get("prompt", []) if m.get("role") == "user")
    by_theme = defaultdict(list)
    for ex in hb:
        ex["_theme"] = theme_of(ex)
        by_theme[ex["_theme"]].append(ex)
    counts = {k: len(v) for k, v in by_theme.items()}
    alloc = largest_remainder(counts, N, FLOOR)
    print(f"HealthBench pool {counts} (total {sum(counts.values())})")
    print(f"  -> alloc {alloc} (sum {sum(alloc.values())})")
    def annotate_hb(ex):
        ex["_rubric_count"] = rubric_count(ex)
        ex["_n_turns"] = nturns(ex)
    hbsel = sample_stratified(by_theme, alloc, id_key="prompt_id")
    manifest["benchmarks"]["healthbench"] = write_locked(
        hbsel, "healthbench_strat120", "_theme", "prompt_id", extra_annotate=annotate_hb)

    # turn / rubric representativeness check for HB
    manifest["benchmarks"]["healthbench"]["n_turns_dist"] = dict(sorted(Counter(nturns(e) for e in hbsel).items()))
    manifest["benchmarks"]["healthbench"]["rubric_count_mean"] = round(sum(rubric_count(e) for e in hbsel)/len(hbsel), 1)

    man_path = OUT / "eval_locked_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nLOCKED manifest -> {man_path}")
    for b, m in manifest["benchmarks"].items():
        print(f"  {b}: n={m['n']} sha256={m['sha256'][:12]}.. comp={m['composition']}")


if __name__ == "__main__":
    main()
