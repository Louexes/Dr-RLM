"""Convert DR Tulu RL rows into SkyRL DR-RLM parquet rows.

This is the **data substrate** for the controlled A/B against DR Tulu (see MEMORY:
project-drrlm-harness-substrate): we reuse the DR Tulu RL prompts/rubrics *verbatim*
so the corpus, judge, and rubric set are held constant across arms — only the credit
scheme (the L1..L4 ladder in ``dr_rlm_config.py``) changes. No new data is authored
here; this is a pure schema transform.

Input
-----
The DR Tulu RL data — HF dataset ``rl-research/dr-tulu-rl-data`` (a.k.a. the gated
``s42chen/wii`` prompts) and/or a local ``.jsonl`` of the same shape. Each row has
(confirmed in dr-tulu/.../dataset_transformation.py key constants @L632 and the
rubric reward field access in longform_rubric_rewards.py):

  * ``messages``       : list[{role, content}]; the user query is in the user turn(s).
  * ``ground_truth``   : a JSON STRING that ``json.loads`` to
                         ``{"query": str, "rubrics": [{"description": str,
                          "title": str, "weight": number}], "rubrics_types"?: ...}``.
                         (Per-task verifiers in dr-tulu json.loads this string; the
                         rubric path accesses rubric['description'] and rubric['weight']
                         unconditionally and rubric.get('title','') optionally. RaR-style
                         rows may omit 'weight' — we default it to 1.0.)
  * ``dataset``        : source-mix string.
  * ``question_type``  : e.g. short_form / long_form / exact_answer.
  * ``source``         : provenance string (optional).

Output
------
SkyRL parquet rows whose columns match the dataset schema consumed by
``skyrl/train/dataset/dataset.py::PromptDataset.__getitem__`` and produced by the
reference converter ``examples/train/rlm/rlm_dataset_synthetic_multi.py::convert``:

  * ``prompt``      : the chat message list (env_class routes it to ``dr_rlm``).
  * ``env_class``   : ``"dr_rlm"``.
  * ``reward_spec`` : ``{"rubrics": [...], "query": <query>}`` — exactly what
                      ``DrRlmEnv._get_reward`` (``self.extras["reward_spec"]["rubrics"]``)
                      and ``DrRlmGenerator.generate`` (``env_extras[bi]["reward_spec"]
                      ["rubrics"]``) read.
  * ``max_turns``   : REPL turn budget.
  * ``extra_info``  : ``{}`` — the corpus is GLOBAL (configured via
                      ``cfg.generator.search_*``), not per-row, so no per-row context.
  * extra columns   : ``question_type``, ``source``, ``dataset`` carried through for
                      slicing/metrics. (``PromptDataset.__getitem__`` passes every
                      non-prompt/non-env_class column through as ``extra``; the GRPO
                      ``uid`` is the row index ``str(item)``, so NO uid column is
                      emitted here — emitting one would just be ignored.)

CLI
---
    uv run -- python examples/train/dr_rlm/data/convert_drtulu_rl.py \
        --hf_path rl-research/dr-tulu-rl-data \
        --output_dir ~/data/dr-rlm-rl --n_val 64 --max_turns 12

    # or from a local jsonl dump of the same rows:
    uv run -- python examples/train/dr_rlm/data/convert_drtulu_rl.py \
        --input_jsonl ~/dumps/dr_tulu_rl.jsonl --output_dir ~/data/dr-rlm-rl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import datasets


_DEFAULT_HF_PATH = "rl-research/dr-tulu-rl-data"


def _parse_ground_truth(gt: Any) -> Dict[str, Any]:
    """``ground_truth`` may be a JSON string (the common case) or already a dict.

    Mirrors the dr-tulu verifiers, which ``json.loads`` the label string into a dict
    of shape ``{"query", "rubrics":[...], "rubrics_types"?}``. Returns ``{}`` on a
    missing / unparseable label (caller decides whether to drop the row)."""
    if gt is None:
        return {}
    if isinstance(gt, dict):
        return gt
    if isinstance(gt, (bytes, bytearray)):
        gt = gt.decode("utf-8", errors="replace")
    if isinstance(gt, str):
        s = gt.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except (ValueError, json.JSONDecodeError):
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def _normalize_rubrics(rubrics: Any) -> List[Dict[str, Any]]:
    """Coerce the rubric list into the schema DrRlmEnv / RubricJudge expect:
    a list of ``{"description": str, "title": str, "weight": float}``.

    ``weight`` defaults to 1.0 (RaR-derived rubrics frequently lack weights); ``title``
    defaults to an empty string (it is accessed via ``.get(..., '')`` downstream).
    Drops entries that carry no usable ``description``."""
    out: List[Dict[str, Any]] = []
    if not isinstance(rubrics, list):
        return out
    for i, r in enumerate(rubrics):
        if not isinstance(r, dict):
            continue
        desc = r.get("description")
        if desc is None or not str(desc).strip():
            continue
        try:
            weight = float(r.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        out.append(
            {
                "description": str(desc),
                "title": str(r.get("title", "") or ""),
                "weight": weight,
            }
        )
    return out


def _coerce_messages(messages: Any) -> List[Dict[str, str]]:
    """Validate/normalize the chat message list into ``[{role, content}, ...]``.

    DR Tulu stores the query in the user turn; we pass the messages through as the
    SkyRL ``prompt`` so the policy sees the exact same conversation. Non-string content
    (rare) is JSON-stringified so the parquet column stays a clean list[{str,str}]."""
    norm: List[Dict[str, str]] = []
    if not isinstance(messages, list):
        return norm
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role is None or content is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        norm.append({"role": str(role), "content": content})
    return norm


def _first_user_query(messages: List[Dict[str, str]]) -> str:
    """Fallback query if ``ground_truth['query']`` is absent: first user turn."""
    for m in messages:
        if m.get("role") == "user" and m.get("content", "").strip():
            return m["content"]
    return ""


def convert(row: Dict[str, Any], max_turns: int, env_class: str = "dr_rlm") -> Optional[Dict[str, Any]]:
    """One DR Tulu RL row -> one SkyRL DR-RLM parquet row, or ``None`` to drop it
    (no usable messages or no usable rubrics — a row with no rubric can't be scored
    by the held-constant judge, so it would only inject zero-signal noise)."""
    messages = _coerce_messages(row.get("messages"))
    if not messages:
        return None

    gt = _parse_ground_truth(row.get("ground_truth"))
    rubrics = _normalize_rubrics(gt.get("rubrics"))
    if not rubrics:
        return None

    query = str(gt.get("query") or "").strip() or _first_user_query(messages)

    return {
        "prompt": messages,
        "env_class": env_class,
        # reward_spec is read as self.extras["reward_spec"]["rubrics"] in DrRlmEnv and
        # env_extras[bi]["reward_spec"]["rubrics"] in DrRlmGenerator. query is carried
        # so the judge can use the canonical question even if it differs from the turn.
        "reward_spec": {
            "rubrics": rubrics,
            "query": query,
        },
        "max_turns": int(max_turns),
        # corpus is global (cfg.generator.search_*), so there is no per-row context.
        # NB: pyarrow cannot write a struct with NO child fields to parquet
        # (ArrowNotImplementedError) — carry the source string as a harmless child.
        # Nothing downstream reads extra_info fields (DrRlmEnv reads reward_spec).
        "extra_info": {"source": str(row.get("source") or "")},
        # passthrough columns for slicing / aggregate metrics (str-coerced so the
        # parquet schema is stable even when a source column is missing/None).
        "question_type": str(row.get("question_type") or ""),
        "source": str(row.get("source") or ""),
        "dataset": str(row.get("dataset") or ""),
    }


def _load_rows(hf_path: Optional[str], input_jsonl: Optional[str], split: str) -> List[Dict[str, Any]]:
    """Load raw DR Tulu RL rows from a local jsonl (preferred if given) or HF hub."""
    if input_jsonl:
        path = os.path.expanduser(input_jsonl)
        print(f"Loading local jsonl {path} ...")
        ds = datasets.load_dataset("json", data_files=path, split="train")
        return list(ds)
    print(f"Loading HF dataset {hf_path} (split={split}) ...")
    ds_dict = datasets.load_dataset(hf_path)
    if split not in ds_dict:
        # many DR Tulu RL repos ship only a train split
        split = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
        print(f"  requested split not found; using '{split}'")
    return list(ds_dict[split])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--hf_path", default=_DEFAULT_HF_PATH, help="HF dataset id of the DR Tulu RL data.")
    src.add_argument("--input_jsonl", default=None, help="Local jsonl of DR Tulu RL rows (overrides --hf_path).")
    parser.add_argument("--split", default="train", help="HF split to read when using --hf_path (default: train).")
    parser.add_argument("--output_dir", default="~/data/dr-rlm-rl")
    parser.add_argument("--n_val", type=int, default=64, help="Validation set size carved from the shuffled head.")
    parser.add_argument("--n_train", type=int, default=None, help="Cap on training rows (default: all remaining).")
    parser.add_argument("--max_turns", type=int, default=12, help="REPL turn budget written into each row.")
    parser.add_argument("--env_class", default="dr_rlm", help="SkyRL env id to stamp on each row (dr_rlm | dr_tulu).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    raw_rows = _load_rows(args.hf_path, args.input_jsonl, args.split)
    print(f"Loaded {len(raw_rows)} raw rows.")

    converted: List[Dict[str, Any]] = []
    n_drop_msg = 0
    n_drop_rubric = 0
    for row in raw_rows:
        # peek at why a row is dropped, for a useful summary line
        if not _coerce_messages(row.get("messages")):
            n_drop_msg += 1
            continue
        if not _normalize_rubrics(_parse_ground_truth(row.get("ground_truth")).get("rubrics")):
            n_drop_rubric += 1
            continue
        rec = convert(row, args.max_turns, env_class=args.env_class)
        if rec is not None:
            converted.append(rec)
    print(
        f"Converted {len(converted)} rows "
        f"(dropped {n_drop_msg} no-messages, {n_drop_rubric} no-rubrics)."
    )
    if not converted:
        raise SystemExit("No convertible rows — check the input schema (messages / ground_truth).")

    rng = random.Random(args.seed)
    rng.shuffle(converted)

    n_val = min(args.n_val, max(0, len(converted) - 1))
    val_rows = converted[:n_val]
    train_rows = converted[n_val:]
    if args.n_train is not None:
        train_rows = train_rows[: args.n_train]

    print(f"Split: train={len(train_rows)}, val={len(val_rows)}")

    splits = {
        "train": datasets.Dataset.from_list(train_rows),
        "validation": datasets.Dataset.from_list(val_rows),
    }

    # quick eyeball of the first few converted rows
    n_show = 3
    for split_name, ds in splits.items():
        print(f"\nFirst {min(n_show, len(ds))} {split_name} examples ({len(ds)} total):")
        for i in range(min(n_show, len(ds))):
            ex = ds[i]
            rubrics = ex["reward_spec"]["rubrics"]
            q = ex["reward_spec"]["query"]
            print(f"  [{i}] env_class={ex['env_class']} qtype={ex['question_type']!r} dataset={ex['dataset']!r}")
            print(f"       query: {q[:120]!r}")
            print(f"       rubrics: {len(rubrics)} (e.g. title={rubrics[0]['title']!r} weight={rubrics[0]['weight']})")

    os.makedirs(args.output_dir, exist_ok=True)
    splits["train"].to_parquet(os.path.join(args.output_dir, "train.parquet"))
    splits["validation"].to_parquet(os.path.join(args.output_dir, "validation.parquet"))

    total = sum(len(ds) for ds in splits.values())
    print(f"\nWrote train.parquet/validation.parquet ({total} total rows) to {args.output_dir}")


if __name__ == "__main__":
    main()
