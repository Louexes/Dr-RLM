#!/usr/bin/env python3
"""Build a FROZEN snapshot corpus from the provenance-POC trajectory ledgers.

RL training needs an offline, stationary corpus (live web is non-stationary and burns
Serper quota at rollout scale; the wii corpus is gated). The POC run already retrieved
~100 web snippets per tree for exactly the 40 training prompts — so we freeze those
ledgers into a wii-schema corpus (rows ``{id, contents, url}``, what
``corpus_search``'s local_jsonl/bm25 backends read). Training rollouts then search the
snapshot of the same evidence the POC trees grounded on. Deliberate overfit to the 40
prompts — this corpus exists for the MECHANISM pilot run, not for benchmarking.

Dedup: by (url, normalized-text) — identical snippets surfaced by many nodes/trees
collapse to one doc. Ids are stable content hashes (``snap-<sha1[:16]>``) so rebuilds
are idempotent and ids carry no provenance (training mints its own {node_rid}-{n} ids
at the tool facade, as always).

Usage:
  python analysis/build_snapshot_corpus.py \
      --runs runs/provenance_poc \
      --out data/poc_corpus/corpus.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", (t or "").strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs containing trajectories/")
    ap.add_argument("--min-chars", type=int, default=40, help="drop trivially short snippets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen: set = set()
    rows = []
    n_files = n_snips = 0
    for run in args.runs:
        for tp in sorted(Path(run).glob("trajectories/traj_*.jsonl")):
            ledger = None
            with open(tp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") == "ledger":
                        ledger = r
            if not ledger:
                continue
            n_files += 1
            for sid, rec in (ledger.get("snippets") or {}).items():
                text = (rec.get("text") or "").strip()
                url = rec.get("url") or ""
                n_snips += 1
                if len(text) < args.min_chars:
                    continue
                key = (url, norm(text))
                if key in seen:
                    continue
                seen.add(key)
                did = "snap-" + hashlib.sha1((url + "\x00" + norm(text)).encode()).hexdigest()[:16]
                rows.append({"id": did, "contents": text, "url": url})

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[snapshot] {n_files} ledgers, {n_snips} raw snippets -> {len(rows)} deduped docs -> {outp}")
    if rows:
        lens = sorted(len(r["contents"]) for r in rows)
        print(f"[snapshot] doc chars: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")


if __name__ == "__main__":
    main()
