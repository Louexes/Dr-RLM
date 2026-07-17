"""Build / validate the frozen DR-RLM search corpus (``corpus.jsonl``).

The corpus is held constant across every arm of the controlled comparison and is
read by the dr_agent retrievers as well as the dependency-free ``local_jsonl``
fallback in ``corpus_search.py``. Both BM25 and FAISS load it via
``BM25Searcher._load_dataset`` (dr-tulu/.../bm25_retriever.py), which builds
``docid_to_metadata[str(row["id"])] = {"snippet": row["contents"], "url": row["url"]}``
and ``url_to_text[row["url"]] = row["contents"]``. So EVERY corpus row MUST carry the
three fields, with these exact names:

  * ``id``       — the docid (cited as the snippet-owning corpus document).
  * ``contents`` — the passage text returned as the snippet.
  * ``url``      — provenance url (also the key for local_browse / get_text_by_url).

``corpus_search._LocalJsonlBackend`` reads the SAME ``{id, contents, url}`` schema
(``row.get("contents")`` / ``row.get("url")`` / ``row.get("id")``), so a corpus.jsonl
produced here drives the smoke-test path with zero extra deps.

Modes
-----
* ``validate``      — read an existing corpus (jsonl or HF dataset), check every row
                      has non-empty ``id`` + ``contents`` (and an ``url``, filled with
                      ``""`` if absent), report problems, and (re)write a clean
                      canonical ``corpus.jsonl`` with only those three columns.
* ``synthetic``     — write a tiny toy ``corpus.jsonl`` (a handful of self-contained
                      passages) so ``search_backend=local_jsonl`` returns real hits in
                      end-to-end smoke tests without the gated wii indices.
* ``download_help`` — print the exact gated-download command (from dr-tulu local.md)
                      for the real wii corpus + prebuilt BM25 / Qwen3-Embed indices.

CLI
---
    # validate a raw corpus and emit the canonical 3-column jsonl
    uv run -- python examples/train/dr_rlm/data/build_corpus.py \
        --mode validate --input ~/raw_corpus.jsonl --output data/corpus.jsonl

    # tiny toy corpus for local_jsonl smoke tests
    uv run -- python examples/train/dr_rlm/data/build_corpus.py \
        --mode synthetic --output data/corpus.jsonl

    # show the command to fetch the real gated corpus + indices
    uv run -- python examples/train/dr_rlm/data/build_corpus.py --mode download_help
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

# The exact field names both dr_agent retrievers (_load_dataset) and the
# local_jsonl fallback require. Keep this list authoritative.
REQUIRED_FIELDS = ("id", "contents", "url")

# Verbatim from dr-tulu/rl/open-instruct/local.md (the gated wii corpus + indices).
_DOWNLOAD_CMD = (
    "huggingface-cli download s42chen/wii-indexes --local-dir ./data --repo-type dataset"
)


# ---------------------------------------------------------------------------
# Loading raw corpus rows (jsonl file or HF dataset spec)
# ---------------------------------------------------------------------------


def _iter_input_rows(input_spec: str) -> Iterable[Dict[str, Any]]:
    """Yield raw rows from a local ``.jsonl``/``.json`` file or an HF dataset id.

    Matches how the dr_agent ``_load_dataset`` decides (``.jsonl`` -> json loader,
    else HF hub), so 'validate' accepts exactly the inputs the real retrievers do."""
    path = os.path.expanduser(input_spec)
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        import datasets  # local import so synthetic/download_help need no datasets dep

        dataset_name, has_split, split = input_spec.partition(":")
        ds_dict = datasets.load_dataset(dataset_name)
        split = split if has_split else ("train" if "train" in ds_dict else list(ds_dict.keys())[0])
        for row in ds_dict[split]:
            yield row


def _clean_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Coerce a raw row into the canonical ``{id, contents, url}`` shape, or ``None``
    if it lacks a usable id or contents (the two fields accessed unconditionally
    downstream — a missing/empty either one would crash or surface an empty snippet).
    ``url`` is optional and defaulted to ``""`` (filled into url_to_text harmlessly)."""
    rid = row.get("id")
    contents = row.get("contents")
    if rid is None or contents is None:
        return None
    rid = str(rid)
    contents = str(contents)
    if not rid.strip() or not contents.strip():
        return None
    url = row.get("url")
    url = "" if url is None else str(url)
    return {"id": rid, "contents": contents, "url": url}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _write_jsonl(rows: List[Dict[str, Any]], output: str) -> None:
    output = os.path.expanduser(output)
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mode_validate(input_spec: str, output: str) -> None:
    if not input_spec:
        raise SystemExit("--input is required for --mode validate")
    print(f"Validating corpus from {input_spec} ...")
    clean: List[Dict[str, Any]] = []
    n_total = 0
    n_bad = 0
    n_no_url = 0
    seen_ids = set()
    n_dup = 0
    for row in _iter_input_rows(input_spec):
        n_total += 1
        cleaned = _clean_row(row)
        if cleaned is None:
            n_bad += 1
            if n_bad <= 5:
                missing = [k for k in REQUIRED_FIELDS if not str(row.get(k, "")).strip()]
                print(f"  drop row {n_total}: missing/empty {missing}")
            continue
        if not cleaned["url"]:
            n_no_url += 1
        if cleaned["id"] in seen_ids:
            n_dup += 1  # duplicate docid would collide in docid_to_metadata; keep last
        seen_ids.add(cleaned["id"])
        clean.append(cleaned)

    print(
        f"Validated {n_total} rows: kept {len(clean)}, dropped {n_bad} "
        f"(missing id/contents); {n_no_url} had empty url; {n_dup} duplicate ids."
    )
    if not clean:
        raise SystemExit("No valid rows; nothing written.")
    _write_jsonl(clean, output)
    print(f"Wrote canonical {{id, contents, url}} corpus ({len(clean)} rows) to {os.path.expanduser(output)}")


# A few self-contained passages with distinct vocabulary so the local_jsonl TF-IDF-lite
# retriever returns sensible, separable hits for queries like "octopus camouflage" etc.
_SYNTHETIC_DOCS: List[Dict[str, str]] = [
    {
        "id": "syn-0001",
        "contents": (
            "The mimic octopus (Thaumoctopus mimicus) is a species of octopus capable of "
            "impersonating other marine animals such as lionfish, sea snakes, and flatfish. "
            "It changes both skin color and body posture to camouflage and deter predators."
        ),
        "url": "https://example.org/marine/mimic-octopus",
    },
    {
        "id": "syn-0002",
        "contents": (
            "Photosynthesis in C4 plants concentrates carbon dioxide around RuBisCO using "
            "PEP carboxylase, reducing photorespiration. Maize and sugarcane are C4 crops "
            "that maintain high efficiency under hot, dry, high-light conditions."
        ),
        "url": "https://example.org/biology/c4-photosynthesis",
    },
    {
        "id": "syn-0003",
        "contents": (
            "The Antikythera mechanism is an ancient Greek analog computer used to predict "
            "astronomical positions and eclipses decades in advance. Recovered from a Roman-era "
            "shipwreck, its bronze gear trains model the irregular motion of the Moon."
        ),
        "url": "https://example.org/history/antikythera",
    },
    {
        "id": "syn-0004",
        "contents": (
            "Reinforcement learning from human feedback (RLHF) trains a reward model on human "
            "preference comparisons, then optimizes a policy against that reward with a method "
            "such as PPO or GRPO. Group-relative baselines reduce variance without a value head."
        ),
        "url": "https://example.org/ml/rlhf-overview",
    },
    {
        "id": "syn-0005",
        "contents": (
            "Citation-graph credit assignment attributes a final report's reward to the "
            "evidence snippets it cites. By tracing each <cite id=...> tag back to the node "
            "that surfaced the snippet, per-node credit conserves the report reward across the "
            "recursion tree instead of broadcasting a single scalar."
        ),
        "url": "https://example.org/ml/citation-credit",
    },
    {
        "id": "syn-0006",
        "contents": (
            "Deep-sea hydrothermal vents host chemosynthetic ecosystems where bacteria oxidize "
            "hydrogen sulfide to fix carbon, supporting giant tube worms and vent crabs without "
            "any sunlight. These vents form along mid-ocean ridges at tectonic spreading centers."
        ),
        "url": "https://example.org/marine/hydrothermal-vents",
    },
]


def mode_synthetic(output: str) -> None:
    rows = [_clean_row(d) for d in _SYNTHETIC_DOCS]
    rows = [r for r in rows if r is not None]
    _write_jsonl(rows, output)
    print(f"Wrote synthetic smoke-test corpus ({len(rows)} rows) to {os.path.expanduser(output)}")
    print("Use it with: cfg.generator.search_backend=local_jsonl "
          f"search_corpus_path={os.path.expanduser(output)}")


def mode_download_help() -> None:
    print("To fetch the REAL gated wii corpus + prebuilt BM25 / Qwen3-Embed indices:\n")
    print("  1. Authenticate (the repo is gated):")
    print("       huggingface-cli login\n")
    print("  2. Download the corpus + indices into ./data:")
    print(f"       {_DOWNLOAD_CMD}\n")
    print("This yields (per dr-tulu/rl/open-instruct/local.md):")
    print("  data/corpus.jsonl              -> set cfg.generator.search_corpus_path")
    print("  data/bm25/                     -> Lucene index; search_backend=bm25, search_index_path=data/bm25")
    print("  data/qwen3-8b/corpus.pkl*      -> FAISS shards; search_backend=faiss, search_index_path='data/qwen3-8b/corpus.pkl*'")
    print("\nNote: bm25 needs Java 16+ (pyserini LuceneSearcher); faiss needs the embed model "
          f"{'Qwen/Qwen3-Embedding-8B'!r}. corpus.jsonl rows are {list(REQUIRED_FIELDS)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["validate", "synthetic", "download_help"], default="validate")
    parser.add_argument("--input", default=None, help="validate: source corpus (.jsonl/.json file or HF dataset id[:split]).")
    parser.add_argument("--output", default="data/corpus.jsonl", help="Destination corpus.jsonl (validate/synthetic).")
    args = parser.parse_args()

    if args.mode == "validate":
        mode_validate(args.input, args.output)
    elif args.mode == "synthetic":
        mode_synthetic(args.output)
    elif args.mode == "download_help":
        mode_download_help()


if __name__ == "__main__":
    main()
