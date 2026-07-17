#!/usr/bin/env python3
"""Extract DR-Tulu-style sub-metrics (already computed, never surfaced) into one table.

SQAv2 (.eval header): rubric=ingredient_recall, answer=answer_precision,
                      cite-p=citation_precision, cite-r=citation_recall, (global_avg)
DRB/RACE (race_result.txt): Comp=comprehensiveness, Depth=insight,
                      Instruction=instruction_following, Readability=readability, (overall)
DRB/FACT (fact/*/*.json): valid_rate, avg_citations_per_article  [optional]

Pure parsing of existing eval outputs — zero compute. Prefers sqa_eval_pro/ (strong-grader
regrade) over sqa_eval/ when present, and labels which grader produced the SQA row.
"""
import glob, json, os, re, sys, zipfile

# (arm, stage) -> run dir. RL-flat absent until the flat arm trains.
RUNS = {
    ("recursive", "untrained"): "runs/drrlm_untrained_thinkon",
    ("recursive", "SFT"):       "runs/drrlm_sft2ep_thinkon",
    ("recursive", "RL"):        "runs/rl_be68_final_thinkon",   # main provenance arm (step 75); replaces deprecated fresh100
    ("flat",      "untrained"): "runs/drtulu_untrained_thinkon",
    ("flat",      "SFT"):       "runs/drtulu_sft1ep_longform_thinkon",
    ("flat",      "RL"):        "runs/drtulu_flat_rl_thinkon",   # not yet produced
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sqa(run):
    # prefer the strong-grader regrade dir if it has a scored .eval
    for sub, grader in [("sqa_eval_pro", "pro"), ("sqa_eval*", "flash")]:
        for f in sorted(glob.glob(os.path.join(ROOT, run, "**", sub, "*.eval"), recursive=True)):
            try:
                h = json.loads(zipfile.ZipFile(f).read("header.json"))
                sc = {s["name"]: (s.get("metrics") or {}).get("mean", {}).get("value")
                      for s in ((h.get("results") or h).get("scores") or [])}
                if sc.get("global_avg") is not None:
                    return {"rubric": sc.get("ingredient_recall"), "answer": sc.get("answer_precision"),
                            "cite_p": sc.get("citation_precision"), "cite_r": sc.get("citation_recall"),
                            "global_avg": sc.get("global_avg"), "grader": grader}
            except Exception:
                continue
    return None


def _race(run):
    hits = glob.glob(os.path.join(ROOT, run, "**", "race", "**", "race_result.txt"), recursive=True)
    if not hits:
        return None
    vals = {}
    for line in open(hits[0]):
        m = re.match(r"\s*(\w+):\s*([\d.]+)", line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    if not vals:
        return None
    return {"Comp": vals.get("comprehensiveness"), "Depth": vals.get("insight"),
            "Instruction": vals.get("instruction_following"), "Readability": vals.get("readability"),
            "overall": vals.get("overall_score")}


def _fact(run):
    for f in glob.glob(os.path.join(ROOT, run, "**", "fact", "**", "fact_result.txt"), recursive=True):
        vals = {}
        for line in open(f):
            m = re.match(r"\s*([\w]+):\s*([\d.]+)", line)
            if m:
                vals[m.group(1)] = float(m.group(2))
        if "valid_rate" in vals:
            return {"valid_rate": vals["valid_rate"], "avg_cites": vals.get("avg_citations_per_article")}
    return None


def fmt(x):
    return " -- " if x is None else f"{x:.3f}"


def block(title, cols, getter):
    print(f"\n### {title}")
    print("arm/stage".ljust(22) + "".join(c.ljust(13) for c in cols))
    for arm in ("recursive", "flat"):
        for stage in ("untrained", "SFT", "RL"):
            d = getter(RUNS[(arm, stage)])
            row = f"{arm:9s} {stage:9s}".ljust(22)
            if d is None:
                row += "  (not available)"
            else:
                row += "".join(fmt(d.get(c if c != 'grader' else 'grader')).ljust(13)
                                if c != 'grader' else str(d.get('grader', '')).ljust(13) for c in cols)
            print(row)
        print()


if __name__ == "__main__":
    block("SQAv2 (rubric / answer / cite-p / cite-r / global_avg)  [+grader]",
          ["rubric", "answer", "cite_p", "cite_r", "global_avg", "grader"], _sqa)
    block("DRB / RACE (Comp / Depth / Instruction / Readability / overall)",
          ["Comp", "Depth", "Instruction", "Readability", "overall"], _race)
    block("DRB / FACT (citation validity)",
          ["valid_rate", "avg_cites"], _fact)
