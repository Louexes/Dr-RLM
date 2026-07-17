"""Phase 7 — freeze + version + document the corpus (DATA_PREP_PLAN §7).

Checksums ``corpus.jsonl`` + the bm25s index, records build params / date / doc + per-source
counts / coverage-audit summary into ``manifest.json``, and prints the exact runtime config to
pin in the RL/eval payloads. After this the corpus is the immutable substrate for SFT-gen, RL,
and eval (the controlled-comparison invariant: same corpus + same retriever, no live web).

Date is passed in (``--date``) so this stays reproducible/offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path, cap_mb: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_digest(d: Path) -> dict:
    files = sorted(p for p in d.rglob("*") if p.is_file())
    return {
        "n_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": {str(p.relative_to(d)): _sha256(p) for p in files},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    root_default = str(Path(__file__).resolve().parents[1] / "data/frozen_corpus")
    ap.add_argument("--root", default=root_default)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD build date (kept out of code for reproducibility)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    root = Path(args.root)
    corpus = root / "corpus.jsonl"
    index_dir = root / "bm25s_index"
    n_docs = sum(1 for _ in open(corpus))

    by_source = {}
    by_dir = root / "by_source"
    if by_dir.exists():
        for f in sorted(by_dir.glob("*.jsonl")):
            by_source[f.stem] = sum(1 for _ in open(f))

    audit = {}
    ap_path = root / "coverage_audit.json"
    if ap_path.exists():
        audit = json.loads(ap_path.read_text())

    manifest = {
        "name": "dr_rlm_frozen_corpus",
        "build_date": args.date,
        "note": args.note,
        "stack": {"discovery": "ddgs(backend=auto)", "reader": "httpx+trafilatura",
                  "index": "bm25s(english stopwords + Snowball stemmer)"},
        "n_docs": n_docs,
        "by_source": by_source,
        "corpus_sha256": _sha256(corpus),
        "corpus_bytes": corpus.stat().st_size,
        "index": _dir_digest(index_dir) if index_dir.exists() else None,
        "coverage_audit": audit,
        "runtime_config": {
            "search_backend": "bm25s",
            "search_index_path": str(index_dir.resolve()),
            "search_corpus_path": str(corpus.resolve()),
        },
    }
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[freeze] {n_docs} docs, corpus sha256={manifest['corpus_sha256'][:16]}…")
    print(f"[freeze] by_source: {by_source}")
    print(f"[freeze] manifest -> {out}")
    print("\n[freeze] pin this in RL/eval payloads:")
    print(json.dumps(manifest["runtime_config"], indent=2))


if __name__ == "__main__":
    main()
