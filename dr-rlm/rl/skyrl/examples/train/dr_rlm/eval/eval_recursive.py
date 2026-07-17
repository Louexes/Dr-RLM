"""Untrained / inference-arm eval for DR-RLM: the flat-vs-recursive frontier.

This is the *characterization* eval (proposal C2 + the controlled flat-vs-recursive
comparison). It does NOT touch SkyRL's training stack: it drives the ``rlm`` library
inference core directly (``RLM(...).completion(question)``), so it measures what a
*stock, untrained* policy buys you from recursion — the empirical motivation for
learning per-node credit. The trained-policy in-harness eval lives in
``main_dr_rlm_eval.py`` (generate-only over ``DrRlmGenerator``); see the README in this
directory for how the two complement each other.

For each eval item ``{question, rubrics}`` we run two arms on the SAME question, with
the SAME corpus retriever and the SAME held-constant rubric judge:

  * FLAT      arm: ``RLM(max_depth=1)``  — a single agent with the corpus ``search()``
    tool but NO recursion. (The rlm core only injects ``subcall_fn`` when
    ``max_depth > 1``, so ``rlm_query``/``rlm_query_batched`` degrade to no-ops at
    ``max_depth=1`` — verified against ``rlm/rlm/core/rlm.py`` and
    ``rlm/rlm/environments/local_repl.py``.)
  * RECURSIVE arm: ``RLM(max_depth=D)`` with ``D >= 2`` — the orchestrator may
    decompose and delegate (depth>1 trees form), exactly like the training env.

We measure the proposal's three axes per arm:

  * QUALITY: ``judge.score_report_sync(report, question, rubrics, cfg)`` — the SAME
    judge config on EVERY arm (held constant), giving the weighted rubric reward R.
  * COMPUTE: ``RLMChatCompletion.usage_summary`` aggregated over the WHOLE tree
    (``total_calls`` summed across models, ``total_input_tokens``,
    ``total_output_tokens``). usage_summary on the root completion already aggregates
    children because the rlm ``LMHandler`` tracks every (root + sub) call routed
    through it for that completion.
  * LATENCY: wall-clock seconds for the ``completion(...)`` call. NOTE: the recursive
    arm's sub-calls run concurrently (up to ``max_concurrent_subcalls``), so wall-clock
    latency is NOT a serial sum of sub-call times — we record the concurrency setting
    alongside every latency number and never claim latency under serial sub-calls.

Output: a JSONL of per-item per-arm records plus an aggregate-means summary, and a
small printed frontier table (flat vs recursive on the three axes).

The depth-banded system prompt and the corpus tools are imported from the training
package so the eval policy/retriever match training byte-for-byte:
  * ``prompts.depth_system_prompt(0, D)`` — the orchestrator band for the root.
  * ``corpus_search.make_corpus_tools(payload, node_rid, [])`` — same retriever.
  * ``judge.score_report_sync`` / ``judge.JudgeConfig`` — same judge.
  * ``dr_rlm_config.DrRlmGeneratorConfig().env_payload()`` — same config surface, so
    every knob (search backend/endpoint/top_k, judge model/url/scale) is identical to
    what the env threads into ``extras["dr_rlm"]`` at train time.

We load these four modules by file path (``importlib``) instead of as
``examples.train.dr_rlm.*`` submodules so importing this eval does NOT trigger the
package ``__init__`` (which registers the SkyRL gym env / advantage estimator and needs
the full ``skyrl_gym`` stack). The inference eval needs none of that — only the four
leaf modules (judge / corpus_search / prompts, and optionally the config dataclass).

Path bootstrap: the ``rlm`` library is imported from its repo checkout. We look for it
at ``$DR_RLM_RLM_HOME``, else next to ``Dr-RLM/rlm``, else assume it is already
importable (e.g. pip-installed). Override with ``DR_RLM_RLM_HOME=/path/to/rlm``.

CLI:
  python -m examples.train.dr_rlm.eval.eval_recursive \
    --eval_data data/eval.jsonl --model Qwen/Qwen3-8B \
    --base_url http://localhost:8000/v1 --max_depth 2 \
    --judge_base_url http://localhost:8100/v1 --judge_model Qwen/Qwen3-8B \
    --out results/frontier.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from statistics import mean
from types import ModuleType
from typing import Any, Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DR_RLM_DIR = os.path.dirname(_THIS_DIR)  # .../examples/train/dr_rlm


def _bootstrap_rlm_lib() -> None:
    """Put the ``rlm`` library on ``sys.path`` if it is not already importable.

    Honors ``$DR_RLM_RLM_HOME``; otherwise tries the conventional sibling checkout
    ``<...>/Dr-RLM/rlm`` (the dir whose ``rlm/`` subpackage holds ``core/rlm.py``).
    If neither is found we leave sys.path alone and let a pip-installed ``rlm`` resolve.
    """
    if importlib.util.find_spec("rlm") is not None:
        return
    candidates: List[str] = []
    env_home = os.environ.get("DR_RLM_RLM_HOME")
    if env_home:
        candidates.append(env_home)
    # _DR_RLM_DIR = .../Dr-RLM/drrlm/rl/skyrl/examples/train/dr_rlm -> climb 6 to Dr-RLM, then /rlm
    # (the dr-tulu-style reorg put SkyRL under drrlm/rl/skyrl/, two levels deeper than the old top-level SkyRL/).
    drrlm_root = os.path.abspath(os.path.join(_DR_RLM_DIR, "..", "..", "..", "..", "..", ".."))
    candidates.append(os.path.join(drrlm_root, "rlm"))
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "rlm", "core", "rlm.py")) and cand not in sys.path:
            sys.path.insert(0, cand)
            return


def _load_sibling(module_name: str, filename: str) -> ModuleType:
    """Import a leaf module from the dr_rlm dir BY FILE PATH so we don't execute the
    package ``__init__`` (which registers the gym env and needs the full skyrl_gym
    stack the inference eval doesn't use). Registered under a private name to avoid
    colliding with a real ``examples.train.dr_rlm.<mod>`` import elsewhere."""
    path = os.path.join(_DR_RLM_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {filename} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_bootstrap_rlm_lib()
from rlm.core.rlm import RLM  # noqa: E402  (rlm inference core — the thing under test)

# leaf modules of the training package, loaded by path (see _load_sibling docstring).
_judge = _load_sibling("_drrlm_eval_judge", "judge.py")
_corpus = _load_sibling("_drrlm_eval_corpus", "corpus_search.py")
_prompts = _load_sibling("_drrlm_eval_prompts", "prompts.py")
JudgeConfig = _judge.JudgeConfig
score_report_sync = _judge.score_report_sync
make_corpus_tools = _corpus.make_corpus_tools
depth_system_prompt = _prompts.depth_system_prompt


# ---------------------------------------------------------------------------
# per-arm result record
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One arm (flat or recursive) on one eval item — the three axes plus context."""

    arm: str  # "flat" | "recursive"
    report_reward: float  # QUALITY: weighted rubric R from the held-constant judge
    per_criterion: Dict[str, float]  # per-rubric s_c (for inspection)
    n_calls: int  # COMPUTE: total LM calls across the whole tree
    input_tokens: int  # COMPUTE
    output_tokens: int  # COMPUTE
    total_tokens: int  # COMPUTE: input + output
    latency_s: float  # LATENCY: wall-clock of completion() (see concurrency note)
    max_concurrent_subcalls: int  # concurrency context for the latency number
    depth_used: int  # max_depth ceiling this arm was allowed (1 = flat)
    report_chars: int  # length of the produced report (sanity / inspection)
    error: Optional[str] = None  # set if the arm crashed (record, don't abort the run)


@dataclass
class ItemResult:
    item_index: int
    question: str
    n_rubrics: int
    flat: Optional[ArmResult] = None
    recursive: Optional[ArmResult] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_eval_data(path: str) -> List[Dict[str, Any]]:
    """Read the eval JSONL. Each row must carry ``question`` and ``rubrics``.

    ``rubrics`` is the held-constant rubric list ``[{description, title, weight}]``
    (the same shape the env carries in ``reward_spec["rubrics"]``). We tolerate a
    couple of common aliases for the question field so eval data prepared for the
    SkyRL env (which nests the prompt) can be reused without rewriting.
    """
    items: List[Dict[str, Any]] = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question")
            if question is None:
                # tolerate {prompt: "..."} or {prompt: [{role,content}, ...]}
                p = row.get("prompt")
                if isinstance(p, list):
                    question = "\n".join(m.get("content", "") for m in p if isinstance(m, dict))
                elif isinstance(p, str):
                    question = p
            if not question:
                raise ValueError(f"{path}:{ln}: row is missing a 'question' (or 'prompt') field")
            rubrics = row.get("rubrics")
            if rubrics is None:
                # tolerate eval rows shaped like the env's reward_spec
                rubrics = (row.get("reward_spec") or {}).get("rubrics")
            items.append({"question": str(question), "rubrics": list(rubrics or [])})
    return items


def usage_totals(completion) -> Dict[str, int]:
    """Aggregate the whole tree's compute off ``RLMChatCompletion.usage_summary``.

    ``usage_summary`` is a ``UsageSummary`` whose ``model_usage_summaries`` dict maps a
    model name to a ``ModelUsageSummary(total_calls, total_input_tokens,
    total_output_tokens)``. ``total_input_tokens`` / ``total_output_tokens`` are
    properties that already sum across models; ``total_calls`` is per-model, so we sum
    it ourselves. Because the rlm LMHandler tracks every call (root + every sub-call)
    routed through it during a completion, these totals span the ENTIRE recursion tree.
    """
    usage = getattr(completion, "usage_summary", None)
    if usage is None:
        return {"n_calls": 0, "input_tokens": 0, "output_tokens": 0}
    n_calls = sum(
        getattr(m, "total_calls", 0) or 0 for m in usage.model_usage_summaries.values()
    )
    return {
        "n_calls": int(n_calls),
        "input_tokens": int(usage.total_input_tokens),
        "output_tokens": int(usage.total_output_tokens),
    }


def build_rlm(
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_depth: int,
    max_iterations: int,
    max_concurrent_subcalls: int,
    payload: Dict[str, Any],
) -> RLM:
    """Construct an ``RLM`` for one arm, wiring the SAME corpus retriever as training.

    A fresh per-run ``node_rid`` is minted and baked into every surfaced snippet id by
    ``make_corpus_tools`` (so a ``<cite id="{rid}-{n}">`` traces back to its node).
    ``surfaced_ids`` (the third arg) is a throwaway list here — the inference eval scores
    the report with the judge directly and does not need to walk the provenance graph.

    The root uses the orchestrator band of ``depth_system_prompt(0, max_depth)``; the
    rlm core hands this same ``custom_system_prompt`` down to children (see ``RLM._subcall``
    ``custom_system_prompt=self.system_prompt``), so every node in the tree shares the
    depth-0 banded prompt — matching how the training env supplies a depth-banded prompt
    per node. ``custom_tools`` are injected into the REPL globals and propagated to
    sub-agents (``custom_sub_tools`` inherits from ``custom_tools`` when unset).
    """
    node_rid = uuid.uuid4().hex  # dash-free uuid, matching the env's rlm_rollout_id scheme
    tools = make_corpus_tools(payload, node_rid, [])
    return RLM(
        backend="vllm",
        backend_kwargs={"model_name": model, "base_url": base_url, "api_key": api_key},
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_concurrent_subcalls=max_concurrent_subcalls,
        custom_system_prompt=depth_system_prompt(0, max_depth),
        custom_tools=tools,
        verbose=False,
    )


def run_arm(
    *,
    arm: str,
    question: str,
    rubrics: List[dict],
    judge_cfg: JudgeConfig,
    max_depth: int,
    args: argparse.Namespace,
    payload: Dict[str, Any],
) -> ArmResult:
    """Run one arm end-to-end: completion -> measure 3 axes -> score with the judge."""
    rlm = build_rlm(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_depth=max_depth,
        max_iterations=args.max_iterations,
        max_concurrent_subcalls=args.max_concurrent_subcalls,
        payload=payload,
    )
    try:
        t0 = time.perf_counter()
        completion = rlm.completion(question)  # the call under test
        latency_s = time.perf_counter() - t0
        report = completion.response or ""
        totals = usage_totals(completion)
        # QUALITY: identical judge for every arm (held constant).
        R, per_criterion = score_report_sync(report, question, rubrics, judge_cfg)
        return ArmResult(
            arm=arm,
            report_reward=float(R),
            per_criterion=per_criterion,
            n_calls=totals["n_calls"],
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            total_tokens=totals["input_tokens"] + totals["output_tokens"],
            latency_s=float(latency_s),
            max_concurrent_subcalls=args.max_concurrent_subcalls,
            depth_used=max_depth,
            report_chars=len(report),
        )
    except Exception as e:  # record the failure on this arm; keep the run going
        return ArmResult(
            arm=arm,
            report_reward=0.0,
            per_criterion={},
            n_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_s=0.0,
            max_concurrent_subcalls=args.max_concurrent_subcalls,
            depth_used=max_depth,
            report_chars=0,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        rlm.close()


# ---------------------------------------------------------------------------
# aggregation + reporting
# ---------------------------------------------------------------------------


def _mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def aggregate(results: List[ItemResult]) -> Dict[str, Dict[str, float]]:
    """Per-arm means over items where the arm succeeded (errored arms excluded)."""
    agg: Dict[str, Dict[str, float]] = {}
    for arm in ("flat", "recursive"):
        arms = [getattr(r, arm) for r in results]
        ok = [a for a in arms if a is not None and a.error is None]
        n_err = sum(1 for a in arms if a is not None and a.error is not None)
        agg[arm] = {
            "n_items": float(len(ok)),
            "n_errors": float(n_err),
            "mean_report_reward": _mean([a.report_reward for a in ok]),
            "mean_n_calls": _mean([float(a.n_calls) for a in ok]),
            "mean_input_tokens": _mean([float(a.input_tokens) for a in ok]),
            "mean_output_tokens": _mean([float(a.output_tokens) for a in ok]),
            "mean_total_tokens": _mean([float(a.total_tokens) for a in ok]),
            "mean_latency_s": _mean([a.latency_s for a in ok]),
        }
    return agg


def print_frontier_table(agg: Dict[str, Dict[str, float]], max_concurrent_subcalls: int) -> None:
    """Print the small flat-vs-recursive frontier table on the three axes."""
    flat, rec = agg["flat"], agg["recursive"]
    cols = [
        ("quality (R)", "mean_report_reward", "{:.4f}"),
        ("latency (s)", "mean_latency_s", "{:.2f}"),
        ("calls", "mean_n_calls", "{:.1f}"),
        ("tokens", "mean_total_tokens", "{:.0f}"),
    ]
    print("\n=== DR-RLM flat-vs-recursive frontier (untrained / inference arm) ===")
    print(f"(latency is wall-clock; recursive sub-calls run with "
          f"max_concurrent_subcalls={max_concurrent_subcalls} — NOT a serial sum)\n")
    header = f"{'axis':<14}{'flat (d=1)':>16}{'recursive':>16}{'Δ (rec-flat)':>16}"
    print(header)
    print("-" * len(header))
    for label, key, fmt in cols:
        f_v, r_v = flat[key], rec[key]
        delta = r_v - f_v
        print(f"{label:<14}{fmt.format(f_v):>16}{fmt.format(r_v):>16}{('%+.4f' % delta) if 'R' in label else ('%+.2f' % delta):>16}")
    # quality-per-1k-tokens: the actual frontier tradeoff (quality bought per compute).
    def per_kt(a):
        return a["mean_report_reward"] / (a["mean_total_tokens"] / 1000.0) if a["mean_total_tokens"] else 0.0
    print(f"\n{'R / 1k tok':<14}{per_kt(flat):>16.5f}{per_kt(rec):>16.5f}")
    print(f"\nflat: {int(flat['n_items'])} ok / {int(flat['n_errors'])} err   "
          f"recursive: {int(rec['n_items'])} ok / {int(rec['n_errors'])} err")


# ---------------------------------------------------------------------------
# config / payload
# ---------------------------------------------------------------------------


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the dr_rlm env payload — the held-constant config surface the env threads
    into ``extras["dr_rlm"]`` at train time — so the eval retriever and judge are wired
    EXACTLY like training (same keys, same defaults).

    Preferred path: instantiate the real ``DrRlmGeneratorConfig`` and call its
    ``env_payload()`` (guaranteed identical to training). That config subclasses a SkyRL
    training config, so if the full ``skyrl`` stack is not importable in this (inference)
    env we fall back to constructing the same dict by hand. The keys below are kept in
    lockstep with ``DrRlmGeneratorConfig.env_payload()``.
    """
    overrides = dict(
        max_recursion_depth=args.max_depth,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key_env=args.judge_api_key_env,
        judge_score_scale=args.judge_score_scale,
        judge_max_concurrency=args.judge_max_concurrency,
        search_backend=args.search_backend,
        search_endpoint=args.search_endpoint,
        search_corpus_path=args.search_corpus_path,
        search_index_path=args.search_index_path,
        search_embed_model=args.search_embed_model,
        search_top_k=args.search_top_k,
        snippet_max_chars=args.snippet_max_chars,
    )
    try:
        from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig  # noqa: E402

        return DrRlmGeneratorConfig(**overrides).env_payload()
    except Exception:
        # skyrl training stack unavailable: build the env_payload() dict directly.
        # reward_mode/per_node_credit are eval-irrelevant (the eval scores the report
        # with the judge directly) but kept for parity with the training payload.
        payload = {
            "reward_mode": "inherited",
            "per_node_credit": False,
            "citation_reward_weight": 0.0,
        }
        payload.update(overrides)
        return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DR-RLM untrained flat-vs-recursive frontier eval")
    # required eval/policy/judge endpoints
    p.add_argument("--eval_data", required=True, help="JSONL with {question, rubrics} rows")
    p.add_argument("--model", required=True, help="policy model id served by the vLLM endpoint")
    p.add_argument("--base_url", required=True, help="OpenAI-compatible base URL of the policy vLLM server")
    p.add_argument("--max_depth", type=int, default=2, help="recursive-arm depth ceiling (>=2); flat arm is always 1")
    p.add_argument("--judge_base_url", required=True, help="OpenAI-compatible base URL of the judge endpoint")
    p.add_argument("--judge_model", required=True, help="judge model id (SAME judge for every arm)")
    p.add_argument("--out", required=True, help="output JSONL path for per-item per-arm records")
    # policy knobs
    p.add_argument("--api_key", default=os.environ.get("RLM_API_KEY", "EMPTY"),
                   help="API key for the policy endpoint (dummy for local vLLM)")
    p.add_argument("--max_iterations", type=int, default=30, help="max REPL iterations per node")
    p.add_argument("--max_concurrent_subcalls", type=int, default=4,
                   help="parallel sub-call threads (recursive arm). Recorded with every latency number.")
    p.add_argument("--limit", type=int, default=0, help="cap #eval items (0 = all)")
    p.add_argument("--arms", default="flat,recursive", help="comma-separated arms to run")
    # judge knobs (kept identical across arms; defaults mirror DrRlmGeneratorConfig)
    p.add_argument("--judge_api_key_env", default="JUDGE_API_KEY")
    p.add_argument("--judge_score_scale", type=float, default=2.0)
    p.add_argument("--judge_max_concurrency", type=int, default=16)
    # retriever knobs (held constant across arms; defaults mirror DrRlmGeneratorConfig)
    p.add_argument("--search_backend", default="local_jsonl",
                   help="bm25|faiss|mcp_http|local_jsonl|none (default local_jsonl: dependency-free)")
    p.add_argument("--search_endpoint", default="http://localhost:8003/mcp")
    p.add_argument("--search_corpus_path", default="data/corpus.jsonl")
    p.add_argument("--search_index_path", default="data/bm25")
    p.add_argument("--search_embed_model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--search_top_k", type=int, default=10)
    p.add_argument("--snippet_max_chars", type=int, default=2000)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.max_depth < 2 and "recursive" in arms:
        raise SystemExit("--max_depth must be >= 2 for the recursive arm (flat arm is fixed at depth 1)")

    items = load_eval_data(args.eval_data)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no eval items loaded from {args.eval_data}")

    # held-constant config + judge built ONCE and reused on every arm/item.
    payload = build_payload(args)
    judge_cfg = JudgeConfig.from_env_payload(payload)
    print(f"[eval_recursive] {len(items)} items | arms={arms} | policy={args.model} @ {args.base_url}")
    print(f"[eval_recursive] judge={judge_cfg.model} @ {judge_cfg.base_url} (scale={judge_cfg.score_scale}) "
          f"| search_backend={payload.get('search_backend')}")
    print(f"[eval_recursive] flat depth=1, recursive depth={args.max_depth}, "
          f"max_concurrent_subcalls={args.max_concurrent_subcalls}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    results: List[ItemResult] = []
    # arm -> (label, depth ceiling): flat is ALWAYS depth 1 (no recursion).
    arm_depths = {"flat": 1, "recursive": args.max_depth}

    with open(args.out, "w") as fout:
        for idx, item in enumerate(items):
            question, rubrics = item["question"], item["rubrics"]
            ir = ItemResult(item_index=idx, question=question, n_rubrics=len(rubrics))
            for arm in arms:
                if arm not in arm_depths:
                    raise SystemExit(f"unknown arm {arm!r}; choose from {list(arm_depths)}")
                res = run_arm(
                    arm=arm,
                    question=question,
                    rubrics=rubrics,
                    judge_cfg=judge_cfg,
                    max_depth=arm_depths[arm],
                    args=args,
                    payload=payload,
                )
                setattr(ir, arm, res)
                tag = f"err={res.error}" if res.error else (
                    f"R={res.report_reward:.3f} calls={res.n_calls} "
                    f"tok={res.total_tokens} lat={res.latency_s:.1f}s"
                )
                print(f"  [item {idx}] {arm:<9} {tag}")
            results.append(ir)
            # stream one JSONL record per item (arms nested) so a long run is resumable.
            rec = {
                "item_index": ir.item_index,
                "question": ir.question,
                "n_rubrics": ir.n_rubrics,
                "flat": asdict(ir.flat) if ir.flat else None,
                "recursive": asdict(ir.recursive) if ir.recursive else None,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

    agg = aggregate(results)
    # append the aggregate-means summary as the final JSONL line (tagged).
    with open(args.out, "a") as fout:
        fout.write(json.dumps({"summary": agg, "config": {
            "model": args.model, "base_url": args.base_url,
            "judge_model": judge_cfg.model, "judge_base_url": judge_cfg.base_url,
            "max_depth": args.max_depth, "max_concurrent_subcalls": args.max_concurrent_subcalls,
            "search_backend": payload.get("search_backend"), "n_items": len(items),
        }}) + "\n")

    print_frontier_table(agg, args.max_concurrent_subcalls)
    print(f"\n[eval_recursive] wrote {len(results)} item records + summary to {args.out}")


if __name__ == "__main__":
    main()
