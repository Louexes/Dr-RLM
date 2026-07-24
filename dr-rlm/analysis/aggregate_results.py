#!/usr/bin/env python3
"""Aggregate W1 arm results into one comparison table (3 thesis axes).

Reads, for each arm (A1/A2/A3) and benchmark (researchqa/deep_research_bench):
  - generation JSONL   (eval_output/<ARM>/<bench>.jsonl)   -> compute axis
  - grader score files (eval_output/<ARM>/...score.txt / drb_eval/) -> quality axis
  - timing.json        (eval_output/<ARM>/timing.json) -> latency axis

Defensive by design: the generation schema only materialises after a run, so we
introspect whatever fields exist rather than assume an exact schema. Missing
pieces degrade to None, never crash.

Compute axis:
  - tool_calls / failed_tool_calls : read directly if present
  - completion_tokens              : prefer any usage/token field in the row;
                                     else tokenize final_response with the Qwen
                                     tokenizer (--tokenizer, default Qwen/Qwen3-8B);
                                     else fall back to whitespace word count.

Usage:
  python aggregate_results.py --root drrlm/runs \
      --arms A1 A2 A3 --out drrlm/results/w1_summary.csv
"""
import argparse
import csv
import json
import os
import statistics as st
from pathlib import Path

BENCHES = ["researchqa", "deep_research_bench"]


# ----------------------------- IO helpers -----------------------------
def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def first_present(d, keys):
    """Look for any of `keys` at the top level OR nested under the workflow's
    `additional_output_data` block (where auto_search_sft.py stashes extra return keys)."""
    if not isinstance(d, dict):
        return None
    scopes = [d, d.get("additional_output_data") or {}]
    for scope in scopes:
        for k in keys:
            if isinstance(scope, dict) and k in scope and scope[k] is not None:
                return scope[k]
    return None


def deep_find_usage(obj, depth=0):
    """Walk a nested dict/list looking for a token usage block; return total tokens or None."""
    if depth > 6 or obj is None:
        return None
    if isinstance(obj, dict):
        # OpenAI/vLLM-style usage
        for key in ("usage", "token_usage"):
            u = obj.get(key)
            if isinstance(u, dict):
                tot = first_present(u, ["total_tokens", "completion_tokens", "output_tokens"])
                if isinstance(tot, (int, float)):
                    return int(tot)
        for k in ("total_tokens", "completion_tokens", "output_tokens"):
            v = obj.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        for v in obj.values():
            r = deep_find_usage(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        s = 0
        found = False
        for v in obj:
            r = deep_find_usage(v, depth + 1)
            if r is not None:
                s += r
                found = True
        if found:
            return s
    return None


# ----------------------------- tokenizer -----------------------------
class TokCounter:
    def __init__(self, name):
        self.tok = None
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"[warn] tokenizer '{name}' unavailable ({e}); using word-count fallback")

    def count(self, text):
        if not text:
            return 0
        if self.tok is not None:
            try:
                return len(self.tok.encode(text))
            except Exception:
                pass
        return len(str(text).split())


# ----------------------------- per-row metrics -----------------------------
def row_response_text(row):
    # final_response is top-level in the workflow's EvalOutput schema
    return first_present(row, ["final_response", "response_text", "answer", "prediction"]) or ""


def row_metrics(row, tc: TokCounter):
    tool_calls = first_present(row, ["total_tool_calls", "num_tool_calls", "tool_calls_count"])
    failed = first_present(row, ["total_failed_tool_calls", "failed_tool_calls"])
    wall_s = first_present(row, ["wall_clock_s"])
    # tokens: prefer a usage block anywhere in the row; else tokenize the response
    toks = deep_find_usage(row)
    if toks is None:
        toks = tc.count(row_response_text(row))
        toks_source = "tokenized"
    else:
        toks_source = "usage"
    return {
        "tool_calls": tool_calls,
        "failed_tool_calls": failed,
        "wall_clock_s": wall_s,
        "tokens": toks,
        "tokens_source": toks_source,
        "resp_chars": len(row_response_text(row)),
    }


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": round(st.mean(vals), 2), "median": round(st.median(vals), 2), "n": len(vals)}


# ----------------------------- quality parsing -----------------------------
def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_kv_floats(path: Path):
    """Parse a clean 'key: value' metrics file (DRB race_result.txt / fact_result.txt
    are plain text lines like 'overall_score: 0.4326', NOT JSON)."""
    import re
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z][\w ()./-]*?)\s*[:=]\s*([0-9]+\.?[0-9]*)\s*$", line)
        if m:
            try:
                out[m.group(1).strip().lower().replace(" ", "_")] = float(m.group(2))
            except ValueError:
                pass
    return out


def parse_quality(arm_dir: Path, bench: str):
    """Extract clean quality metrics per (arm, bench) from the graders' STRUCTURED
    sidecars (not the log-polluted *_score.txt tees, which also contain INFO lines,
    progress bars, and the fastmcp deprecation warning).

    ResearchQA -> researchqa_eval_results.json  (key: score / metrics.coverage)
    DRB        -> drb_eval/race/*/race_result.txt  (JSON: overall_score, 4 dims)
                  drb_eval/fact/*/fact_result.txt  (JSON: valid_rate, citations)
    Falls back to a tolerant text scan only if the structured files are absent.
    """
    import re

    out = {}

    if bench == "researchqa":
        j = _load_json(arm_dir / "researchqa_eval_results.json")
        if isinstance(j, dict):
            if isinstance(j.get("score"), (int, float)):
                out["coverage"] = j["score"]
            metrics = j.get("metrics") or {}
            if isinstance(metrics.get("coverage"), (int, float)):
                out["coverage"] = metrics["coverage"]
    else:
        # RACE (article quality): drb_eval/race/<task>/race_result.txt  (plain 'key: val' text)
        for rf in (arm_dir / "drb_eval" / "race").glob("*/race_result.txt"):
            for k, v in _parse_kv_floats(rf).items():
                out[f"race_{k}"] = v
        # FACT (citation verification): drb_eval/fact/<task>/fact_result.txt
        for ff in (arm_dir / "drb_eval" / "fact").glob("*/fact_result.txt"):
            for k, v in _parse_kv_floats(ff).items():
                out[f"fact_{k}"] = v

    # Fallback: tolerant scan of the tee'd score file ONLY if nothing structured was found.
    if not out:
        score_txt = arm_dir / ("researchqa_score.txt" if bench == "researchqa" else "drb_score.txt")
        if score_txt.exists():
            for m in re.finditer(r"([A-Za-z][\w ().-]{1,40}?)\s*[:=]\s*([0-9]+\.?[0-9]*)", score_txt.read_text()):
                key = m.group(1).strip().lower()
                # drop obvious log noise
                if any(bad in key for bad in ("jwt", "providers", "site-packages", "/", "step", "total", "process", "done")):
                    continue
                try:
                    out[key] = float(m.group(2))
                except ValueError:
                    pass
    return out


def headline_quality(bench: str, q: dict):
    """Single comparable number per (arm, bench): RQA->coverage, DRB->RACE overall."""
    if not isinstance(q, dict):
        return None
    if bench == "researchqa":
        return q.get("coverage")
    return q.get("race_overall_score")


def parse_timing(arm_dir: Path):
    t = arm_dir / "timing.json"
    if t.exists():
        try:
            return json.loads(t.read_text())
        except Exception:
            return {}
    return {}


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="drrlm/runs")
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="drrlm/results/w1_summary.csv")
    args = ap.parse_args()

    tc = TokCounter(args.tokenizer)
    root = Path(args.root)
    table = []

    for arm in args.arms:
        arm_dir = root / arm
        timing = parse_timing(arm_dir)
        for bench in BENCHES:
            rows = read_jsonl(arm_dir / f"{bench}.jsonl")
            mets = [row_metrics(r, tc) for r in rows]
            quality = parse_quality(arm_dir, bench)
            rec = {
                "arm": arm,
                "benchmark": bench,
                "n_items": len(rows),
                # compute axis
                "tool_calls_mean": agg([m["tool_calls"] for m in mets])["mean"],
                "failed_tool_calls_mean": agg([m["failed_tool_calls"] for m in mets])["mean"],
                "tokens_mean": agg([m["tokens"] for m in mets])["mean"],
                "tokens_source": (mets[0]["tokens_source"] if mets else None),
                "resp_chars_mean": agg([m["resp_chars"] for m in mets])["mean"],
                # latency axis: per-item timer (preferred) + coarse job-level wall clock
                "sec_per_item_peritem": agg([m["wall_clock_s"] for m in mets])["mean"],
                "wall_clock_s": timing.get(bench, {}).get("wall_clock_s") if isinstance(timing.get(bench), dict) else timing.get(f"{bench}_wall_clock_s"),
                "concurrency": timing.get(bench, {}).get("concurrency") if isinstance(timing.get(bench), dict) else timing.get("concurrency"),
                # quality axis: one comparable headline + the full clean dict
                "headline_quality": headline_quality(bench, quality),
                "quality": json.dumps(quality, ensure_ascii=False),
            }
            if rec["wall_clock_s"] and rec["n_items"]:
                rec["sec_per_item"] = round(rec["wall_clock_s"] / rec["n_items"], 2)
            else:
                rec["sec_per_item"] = None
            table.append(rec)

    # write CSV
    cols = ["arm", "benchmark", "n_items", "tool_calls_mean", "failed_tool_calls_mean",
            "tokens_mean", "tokens_source", "resp_chars_mean", "sec_per_item_peritem",
            "wall_clock_s", "concurrency", "sec_per_item", "headline_quality", "quality"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in table:
            w.writerow(rec)

    # pretty print to stdout
    print(f"\nW1 summary  ->  {out}\n")
    hdr = f"{'arm':<9} {'benchmark':<20} {'n':>3} {'tools':>6} {'tokens':>8} {'src':>9} {'s/item':>7} {'quality':>8}"
    print(hdr)
    print("-" * len(hdr))
    for rec in table:
        latency = rec["sec_per_item_peritem"] if rec["sec_per_item_peritem"] is not None else rec["sec_per_item"]
        hq = rec["headline_quality"]
        hq_s = f"{hq:.3f}" if isinstance(hq, (int, float)) else str(hq)
        print(f"{rec['arm']:<9} {rec['benchmark']:<20} {rec['n_items']:>3} "
              f"{str(rec['tool_calls_mean']):>6} {str(rec['tokens_mean']):>8} "
              f"{str(rec['tokens_source']):>9} {str(latency):>7} {hq_s:>8}")
    print("\nQuality scores (per arm/bench) are in the 'quality' CSV column (raw grader keys).")


if __name__ == "__main__":
    main()
