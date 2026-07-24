"""DR-RLM inference driver (Part B2): run the UNIFIED env for inference.

Drives the exact SkyRL ``DrRlmGenerator`` used in RL training — same depth-banded prompt, same
silent tool facade (online/offline via ``--search-backend``), same answer-dict + ledger + RER
provenance — but routes each turn's generation to an OpenAI-compatible API via the B1
``OpenAICompatInferenceClient`` instead of the trained policy. No Ray, no trainer: we build the
config tree with ``make_config().from_cli_overrides(...)``, construct ``DrRlmGenerator`` directly
(exactly as ``main_dr_rlm_eval``'s ``get_generator``), and ``await gen.generate(batch)`` per item.
``agent_loop`` owns the turn loop — we only build the input batch and read the rendered report back
from the root's last-step ``env_metrics``.

Invoked from ``generate.py --driver new`` (the legacy ``rlm/`` path stays the default). Reuses
generate.py's dataset loaders, sharding, resume-by-example_id, and OUTPUT ROW SCHEMA so the
downstream grader reads identical rows.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from typing import Any, Dict, List

# --- make the vendored SkyRL stack + dr_agent importable -------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # .../Dr-RLM
_SKYRL = os.path.join(_REPO, "dr-rlm", "rl", "skyrl")
for _p in (_SKYRL, os.path.join(_SKYRL, "skyrl-gym"), os.path.join(_REPO, "dr-tulu", "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer  # noqa: E402

from skyrl.train.config import make_config  # noqa: E402
from skyrl.train.generators.base import TrajectoryID  # noqa: E402
import examples.train.dr_rlm  # noqa: E402,F401  (registers gym env id "dr_rlm" -> DrRlmEnv)
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig  # noqa: E402
from examples.train.dr_rlm.dr_rlm_generator import DrRlmGenerator  # noqa: E402
from examples.train.dr_rlm._run_guard import get_hard_error as _get_hard_error  # noqa: E402

from api_client import OpenAICompatInferenceClient  # noqa: E402  (same dir; B1)

_DrRlmConfig = make_config(generator_cls=DrRlmGeneratorConfig)


# ---------------------------------------------------------------------------
# Construction (no Ray) — mirrors main_dr_rlm_eval.DrRlmEvalEntrypoint.get_generator
# ---------------------------------------------------------------------------
def _build_generator(args, tokenizer):
    recursive = args.max_recursion_depth > 0
    overrides: Dict[str, Any] = {
        # --- mandatory generator invariants ---
        "generator.step_wise_trajectories": True,      # RLMGymGenerator.__init__ raises otherwise
        "generator.batched": False,
        "generator.max_turns": args.max_iterations,
        "generator.max_input_length": args.max_input_length,   # MUST be int (no SkyRLTrainConfig.__post_init__ here)
        "generator.use_conversation_multi_turn": True,
        "generator.sampling_params.max_generate_length": args.max_completion_tokens,
        # --- DR-RLM eval knobs ---
        "generator.per_node_credit": False,            # score the report (external grader), not per-node RER
        "generator.enable_child_agents": recursive,     # inject subcall_fn iff recursive
        "generator.train_child_trajectories": recursive,  # inline child steps into env_metrics so recursion telemetry populates
        "generator.max_recursion_depth": args.max_recursion_depth,
        "generator.child_return_mode": getattr(args, "child_return_mode", "prose"),  # prose-vs-structured A/B
        "generator.citation_verification": bool(getattr(args, "citation_verification", False)),  # pre-submit cite check
        "generator.check_citations_tool": bool(getattr(args, "check_citations_tool", False)),    # in-REPL verdict oracle
        "generator.citations_bounce": bool(getattr(args, "citations_bounce", False)),            # one-time empty-citations gate
        "generator.max_search_calls_per_tree": getattr(args, "max_tool_calls", 0),   # per-item Serper/web budget
        "generator.max_children_per_node": getattr(args, "max_children", 0),         # per-node fan-out cap
        "generator.frozen_openrouter_model": None,      # in-REPL llm_query (unused by DR-RLM) hits the engine, not OpenRouter
        # --- the unified tool facade: online/offline by one flag (each backend reads a different field) ---
        "generator.search_backend": args.search_backend,
        "generator.search_endpoint": args.search_endpoint,        # mcp_http reads this
        "generator.search_corpus_path": args.search_corpus_path,  # local_jsonl / bm25 / faiss read this
        "generator.search_index_path": getattr(args, "search_index_path", "data/bm25"),  # bm25s / bm25 / faiss read this
        "generator.web_mcp_port": args.mcp_port,                  # web backend (Serper/Jina) reads this
    }
    cfg = _DrRlmConfig.from_cli_overrides(overrides)
    client = OpenAICompatInferenceClient.from_endpoint(
        model=args.model,
        tokenizer=tokenizer,
        base_url=args.base_url or "https://api.openai.com/v1",
        api_key=os.environ.get(args.api_key_env, ""),
    )
    gen = DrRlmGenerator(
        generator_cfg=cfg.generator,
        skyrl_gym_cfg=cfg.environment.skyrl_gym,
        inference_engine_client=client,
        tokenizer=tokenizer,
    )
    return gen, client


# ---------------------------------------------------------------------------
# Per-benchmark task framing, appended to the problem in the user message (root-only —
# children receive bare sub-questions, as in the legacy driver). Mirrors generate.py's
# legacy framings, which adapt the dr-tulu loaders' `additional_instructions` to a
# retrieved-snippet (rather than preloaded-passage) setting. Deliberately NOT in
# prompts/system_prompt.txt: the system prompt steers behavior; this states the task.
# ---------------------------------------------------------------------------
_SHORTFORM_FRAMING = (
    "Find the single factual answer to the question, grounded in retrieved snippets. Your "
    "final response MUST end with the answer on its own line in exactly this form:\n"
    "Exact Answer: <your succinct answer>\n"
    "Keep it short — a name, date, number, or brief phrase, not an essay. Do not stop until "
    "you have written the Exact Answer line."
)
_TASK_FRAMING = {
    # ADOPTED from the offline length-compliance A/B (runs/offline_ab_rqa_length): REPL
    # self-measurement framing -> word counts cluster on target (mean 257 vs 198), zero empty
    # answers (control had one), rubric coverage 0.441->0.602.
    "researchqa": (
        "Answer the question completely and precisely in 240-260 words — this is a HARD "
        "requirement, not a suggestion. Before submitting, MEASURE your draft in the REPL with "
        "len(answer[\"content\"].split()) and revise until it is within 240-260; an answer "
        "outside the range loses points no matter how good its content is. Write one to three "
        "paragraphs (do not enumerate the facts), and support every statement with an in-line "
        "citation to a retrieved snippet."
    ),
    "deep_research_bench": (
        "Write a COMPREHENSIVE, well-structured, multi-section research report that thoroughly "
        "develops every distinct aspect of the research question — depth, coverage, and length "
        "matter here; do not compress your findings into a thin summary. Support every substantive "
        "claim with an in-line citation to a retrieved snippet."
    ),
    "sqav2": (
        "Write a well-structured, data-driven report that thoroughly answers the scientific "
        "research question, grounded in the literature. Support every claim with an in-line "
        "citation to a retrieved snippet."
    ),
    "healthbench": (
        "Answer the patient's medical question thoroughly, accurately, and safely, grounded in "
        "retrieved evidence with in-line citations. Seek or acknowledge missing context where it "
        "matters, hedge appropriately under uncertainty, flag emergencies and when to seek "
        "in-person care, and communicate clearly for the reader's apparent expertise level."
    ),
    # DR Tulu RL prompts (training distribution; rubric-judged long-form research answers).
    "drtulu_rl": (
        "Write a comprehensive, well-organized long-form answer that covers every distinct "
        "aspect of the question, grounded in retrieved evidence. Support every substantive "
        "claim with an in-line citation to a retrieved snippet."
    ),
    # Short-form QA guardrails (verifiable, top-1 accuracy: SimpleQA/2Wiki/WebWalker). The
    # final line is the scored unit; the closing nudge counters the untrained thinking-ON
    # finalization deficit (a model that never submits scores zero here).
    "simpleqa": _SHORTFORM_FRAMING,
    "2wiki": _SHORTFORM_FRAMING,
    "webwalker": _SHORTFORM_FRAMING,
}


# ---------------------------------------------------------------------------
# Per-item: build the batch, run, read the rendered report from env_metrics
# ---------------------------------------------------------------------------
def _recursion_summary(env_rows: List[Dict[str, Any]], cap: int) -> Dict[str, Any]:
    """Reconstruct a recursion profile from the per-step env_metrics rows."""
    by_node: Dict[str, int] = {}
    for r in env_rows:
        rid = (r or {}).get("rlm_rollout_id")
        if rid is not None:
            by_node[str(rid)] = int((r or {}).get("depth", 0))
    depths = list(by_node.values())
    subs_by_depth: Dict[int, int] = {}
    for d in depths:
        if d > 0:
            subs_by_depth[d] = subs_by_depth.get(d, 0) + 1
    return {
        "n_subagents": sum(1 for d in depths if d > 0),
        "max_depth_reached": max(depths) if depths else 0,
        "max_depth_cap": cap,
        "subagents_by_depth": subs_by_depth,
        "n_nodes": len(by_node),
    }


def _provenance_survival(out: Dict[str, Any]) -> Dict[str, Any]:
    """Child->root citation SURVIVAL — the direct test of the structured-vs-prose hypothesis.

    Of the evidence the SUB-AGENTS surfaced-and-cited (their own ``<cite>`` ids), what fraction
    did the ORCHESTRATOR keep in the final report? Free-text returns are predicted to drop more
    (the orchestrator reconciles prose and loses ids); structured returns hand it id atoms to
    merge, so survival should be higher.

    Reads per-NODE ``cited_ids`` from env_metrics (each node's own report citations), unions
    them per node across steps, then splits by depth. Survival is undefined (None) when no child
    cited anything (flat run, or every child returned uncited)."""
    rows = out.get("env_metrics") or []
    node_depth: Dict[str, int] = {}
    node_cited: Dict[str, set] = {}
    for r in rows:
        r = r or {}
        rid = r.get("rlm_rollout_id")
        if rid is None:
            continue
        rid = str(rid)
        node_depth[rid] = int(r.get("depth", node_depth.get(rid, 0)))
        cids = r.get("cited_ids") or []
        if cids:
            node_cited.setdefault(rid, set()).update(str(c) for c in cids)
    root_cited: set = set()
    child_cited: set = set()
    for rid, cids in node_cited.items():
        (root_cited if node_depth.get(rid, 0) == 0 else child_cited).update(cids)
    survived = child_cited & root_cited
    n_child = len(child_cited)
    return {
        "n_child_cited": n_child,                                   # distinct ids the children cited
        "n_child_cited_survived": len(survived),                    # ...that reached the root report
        "child_citation_survival_rate": (len(survived) / n_child) if n_child else None,
        "n_child_orphan_cited": len(child_cited - root_cited),      # children cited, orchestrator dropped
        "n_root_cited": len(root_cited),
        "n_root_own_cited": len(root_cited - child_cited),          # root's own searches (not from a child)
    }


def _ledger_to_snippet_blocks(ledger: Optional[Dict[str, Any]], cited_only: Optional[set] = None) -> str:
    """Serialize the evidence ledger into ``<snippet id="sid">text</snippet>`` blocks for
    graders that rebuild the id->snippet map from ``full_traces.generated_text`` rather than
    ``tool_calls`` — sqav2's ASTA converter (convert_to_asta_format.py parse_answer) looks up
    every ``<cite id=...>`` against these tags; without them our citations are INVISIBLE to
    citation_precision/recall (measured 0 despite valid cites). Same measurement-layer fix
    family as ``_ledger_to_tool_calls`` below. ``cited_only`` (a set of sids) bounds output
    size to the snippets a report actually cites; None emits all surfaced snippets."""
    snaps = (ledger or {}).get("snippets") or {}
    blocks = []
    for sid, rec in snaps.items():
        if cited_only is not None and sid not in cited_only:
            continue
        text = str((rec or {}).get("text") or "").strip()
        if text:
            blocks.append(f'<snippet id="{sid}">\n{text}\n</snippet>')
    return "\n".join(blocks)


def _ledger_to_tool_calls(ledger: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Serialize the evidence ledger into DRB-FACT's ``tool_calls`` shape so the grader can
    validate our ``<cite id="...">`` tags.

    DRB (``format_drb_data`` in deep_research_bench_eval/run_eval.py) builds the set of
    citable sources ONLY from ``full_traces["tool_calls"]``: for each call it parses the
    ``output`` text with ``parse_search_results`` and assigns each result the id
    ``f"{call_id}-{i}"`` (i = its position in that call's results). A report ``<cite id="X">``
    is validated only if ``X`` equals one of those ids.

    Our snippet ids are ``f"{node_rid}-{n}"`` minted in provider.py (``counter['n']``
    increments once per surfaced hit within a node, so n is CONTIGUOUS from 0 per node). So we
    group ledger snippets by ``node_rid``, order by n, and emit one call per node with
    ``call_id = node_rid`` — then DRB's ``{call_id}-{i}`` reproduces our sids exactly. Gaps
    (should not occur, but be defensive) are padded so positions stay aligned. Each snippet's
    ``text`` is already a ``Title: ...\\nURL: ...\\nSnippet: ...`` block (web backend); if not,
    we synthesize that shape so ``parse_search_results`` splits it cleanly."""
    snaps = (ledger or {}).get("snippets") or {}
    by_node: Dict[str, Dict[int, Dict[str, str]]] = {}
    for sid, rec in snaps.items():
        node, _, n = str(sid).rpartition("-")
        if not node or not n.isdigit():
            continue
        by_node.setdefault(node, {})[int(n)] = rec or {}

    def _block(rec: Dict[str, str]) -> str:
        text = str((rec or {}).get("text") or "")
        if text.lstrip().startswith("Title:"):
            return text
        url = str((rec or {}).get("url") or "")
        return f"Title: \nURL: {url}\nSnippet: {text}"

    _PAD = "Title: \nURL: \nSnippet: "
    tool_calls: List[Dict[str, str]] = []
    for node, idx_rec in by_node.items():
        hi = max(idx_rec)
        blocks = [_block(idx_rec[i]) if i in idx_rec else _PAD for i in range(hi + 1)]
        tool_calls.append({"call_id": node, "output": "\n\n".join(blocks)})
    return tool_calls


def _pop_tree_ledger(gen) -> Optional[Dict[str, Any]]:
    """Snapshot-and-clear the per-tree evidence ledger after one item's generate().

    With ``per_node_credit=false`` (this driver) the generator's RER branch never runs, so
    the ledger minted in ``_setup_env_extras`` is never popped — it stays in
    ``gen._tree_ledgers``. We pop ALL entries (one item in flight at a time) and return a
    JSON-safe copy ``{snippets: {sid: {node_rid, text, url, docid, query}}}`` so the
    trajectory file carries the evidence the credit pipeline (``ledger_support``) and the
    LOCO counterfactual need."""
    ledgers = getattr(gen, "_tree_ledgers", None)
    if not ledgers:
        return None
    lock = getattr(gen, "_ledger_lock", None)
    try:
        if lock is not None:
            lock.acquire()
        items = list(ledgers.items())
        ledgers.clear()
    finally:
        if lock is not None:
            lock.release()
    if not items:
        return None
    _tid, ledger = items[-1]
    snippets = dict((ledger or {}).get("snippets") or {})
    return {"snippets": snippets, "n_search_calls": int((ledger or {}).get("_search_calls", 0))}


def _save_trajectory(out: Dict[str, Any], ex: Dict[str, Any], args, traj_dir: str,
                     ledger: Optional[Dict[str, Any]] = None) -> str:
    """Dump the item's FULL recursion-tree trajectory to a parseable JSONL (+ run config),
    so every design choice's effect on agent behavior is recorded and visualizable.

    Schema (one file per item):
      line 0     : {type:"metadata", example_id, benchmark, problem, config:{...}, n_nodes}
      line 1..N  : {type:"node", rid, parent_rid, depth, child_index, n_turns, cited_ids,
                    final_answer, trajectory:[{turn, response, code, stdout, stderr, submitted}]}
    Each node's terminal env_metrics row carries its trajectory (env.get_metrics); rlm_metadata
    gives rid/parent_rid/depth to stitch the tree. Rendered by agent/viz_trajectory_v2.py."""
    rows = out.get("env_metrics") or []
    nodes: List[Dict[str, Any]] = []
    for r in rows:
        r = r or {}
        if "trajectory" not in r:   # only the terminal per-node row carries the full record
            continue
        meta = r.get("rlm_metadata") or {}
        nodes.append({
            "type": "node",
            "rid": meta.get("rid") or r.get("rlm_rollout_id"),
            "parent_rid": meta.get("parent_rid"),
            "depth": int(meta.get("depth", r.get("depth", 0)) or 0),
            "child_index": meta.get("child_index"),
            "n_turns": len(r.get("trajectory") or []),
            "cited_ids": list(r.get("cited_ids") or []),
            "final_answer": r.get("final_answer") or "",
            # per-node verification telemetry ({"checked","dropped"}), present when the
            # citation_verification flag ran at this node's finalize
            "citation_verification": r.get("citation_verification"),
            # per-node check_citations REPL-tool telemetry ({"calls","checked","supported",
            # "unsupported"}), present when the node actually called the oracle
            "check_citations_tool": r.get("check_citations_tool"),
            "trajectory": r.get("trajectory") or [],
        })
    nodes.sort(key=lambda n: (n["depth"], str(n.get("child_index") if n.get("child_index") is not None else "")))
    os.makedirs(traj_dir, exist_ok=True)
    path = os.path.join(traj_dir, f"traj_{args.benchmark}_{ex['id']}.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({
            "type": "metadata", "example_id": ex["id"], "benchmark": args.benchmark,
            "problem": ex.get("problem", ""),
            "config": {
                "model": args.model,
                "child_return_mode": getattr(args, "child_return_mode", "prose"),
                "max_recursion_depth": args.max_recursion_depth,
                "max_children": getattr(args, "max_children", 0),
                "max_tool_calls": getattr(args, "max_tool_calls", 0),
                "search_backend": args.search_backend,
                "temperature": args.temperature,
            },
            "n_nodes": len(nodes),
        }) + "\n")
        for n in nodes:
            f.write(json.dumps(n) + "\n")
        if ledger is not None:
            f.write(json.dumps({"type": "ledger", **ledger}) + "\n")
    return path


def _root_last_step(out: Dict[str, Any]) -> Dict[str, Any]:
    """The root trajectory's terminal env_metrics row (depth==0, last is_last_step)."""
    rows = out.get("env_metrics") or []
    is_last = out.get("is_last_step") or []
    idxs = [i for i in range(len(is_last))
            if is_last[i] and (rows[i] or {}).get("depth", 0) == 0]
    if not idxs:  # fallback: last flagged step of any trajectory
        idxs = [i for i in range(len(is_last)) if is_last[i]]
    return rows[idxs[-1]] if idxs else {}


async def _run_one(gen, ex: Dict[str, Any], args) -> Dict[str, Any]:
    problem = ex["problem"]
    # The user message = bare problem + per-benchmark TASK FORMAT. The env builds the depth-banded
    # system prompt internally (DrRlmEnv._build_system_prompt) — do NOT pre-inject the system
    # prompt here, or it would double-apply vs training.
    # Per-benchmark output-shape framing, env-overridable for prompt A/Bs without code edits:
    # DR_RLM_TASK_FRAMING_JSON='{"researchqa": "..."}' merges over the defaults (same pattern
    # as DR_RLM_PROMPT_FILE for the system prompt — never mutate code under a queued job).
    framing_map = dict(_TASK_FRAMING)
    _env_framing = os.environ.get("DR_RLM_TASK_FRAMING_JSON")
    if _env_framing:
        try:
            framing_map.update(json.loads(_env_framing))
        except (json.JSONDecodeError, TypeError, ValueError):
            print("[a2/new] WARNING: DR_RLM_TASK_FRAMING_JSON is not valid JSON; using defaults")
    framing = framing_map.get(args.benchmark, framing_map["deep_research_bench"])
    user_content = problem + "\n\nTASK FORMAT: " + framing
    input_batch = {
        "prompts": [[{"role": "user", "content": user_content}]],
        "env_classes": ["dr_rlm"],
        # rubrics EMPTY -> in-env reward is 0 and NO judge call (the offline grader scores the report).
        "env_extras": [{"reward_spec": {"rubrics": []}, "extra_info": {}}],
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": getattr(args, "top_p", 0.8),
            "max_generate_length": args.max_completion_tokens,
            # Qwen3.5 non-thinking recommended extras (model card). Passed through api_client's
            # additional_kwargs -> request body; vLLM's OpenAI server reads top_k/min_p/presence_penalty.
            # presence_penalty=1.5 suppresses the untrained policy's bare-search() repetition loop that
            # penalty-free/greedy decoding otherwise locks into (-> empty reports).
            "additional_kwargs": {
                "top_k": getattr(args, "sampling_top_k", 20),
                "min_p": getattr(args, "min_p", 0.0),
                "presence_penalty": getattr(args, "presence_penalty", 1.5),
            },
        },
        "trajectory_ids": [TrajectoryID(instance_id=str(ex["id"]), repetition_id=0)],
        "batch_metadata": None,
    }
    # client.usage is CUMULATIVE across items (never reset) — snapshot before/after for per-item deltas.
    _usage = gen.inference_engine_client.usage
    _u0p, _u0c = _usage.get("prompt_tokens", 0), _usage.get("completion_tokens", 0)
    t0 = time.perf_counter()
    out = await gen.generate(input_batch)
    wall = time.perf_counter() - t0
    prompt_tok = _usage.get("prompt_tokens", 0) - _u0p
    completion_tok = _usage.get("completion_tokens", 0) - _u0c

    root = _root_last_step(out)
    report = root.get("final_answer") or ""
    cited_ids = list(root.get("cited_ids") or [])
    rec = _recursion_summary(out.get("env_metrics") or [], args.max_recursion_depth)
    prov = _provenance_survival(out)
    n_surfaced = sum(int((r or {}).get("n_surfaced_snippets", 0)) for r in (out.get("env_metrics") or []))

    # Evidence-ledger capture is OPT-IN (DR_RLM_CAPTURE_LEDGER): default off keeps trajectory
    # files byte-identical to before (metadata + node lines only) for other --driver new runs.
    # The provenance-credit POC sets it to get the per-tree ledger that ledger_support credit needs.
    _capture_ledger = os.environ.get("DR_RLM_CAPTURE_LEDGER", "") not in ("", "0", "false")
    ledger = _pop_tree_ledger(gen) if _capture_ledger else None
    traj_path = None
    if getattr(args, "save_trajectories", None):
        try:
            traj_path = _save_trajectory(out, ex, args, args.save_trajectories, ledger=ledger)
        except Exception as e:  # never let trajectory capture kill a run
            print(f"[traj] capture failed for id={ex.get('id')}: {e}")

    # sqav2's ASTA converter rebuilds the id->snippet map from <snippet> tags in
    # generated_text (NOT tool_calls); without these blocks citation_precision/recall
    # read 0 no matter how valid the cites are. The converter only scans generated_text
    # when it contains '<think>' (else it falls back to tool_calls[].generated_text — a
    # key our tool_calls don't carry), so the blocks are wrapped in a <think> section.
    # The only eval consumer of generated_text is that converter (others read
    # final_response), so the wrapper leaks nowhere.
    snippet_blocks = _ledger_to_snippet_blocks(ledger, cited_only=set(cited_ids))
    return {
        "final_response": report,
        "full_traces": {
            "generated_text": report + (f"\n\n<think>\n{snippet_blocks}\n</think>" if snippet_blocks else ""),
            "total_tokens": prompt_tok + completion_tok,
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "tool_call_count": n_surfaced,   # surfaced-snippet count (facade is silent; not a raw call count)
            "stopped_reason": "natural",
            # DRB-FACT validates <cite> ids against sources rebuilt from these tool_calls;
            # empty => "no valid citations" (FACT=0) no matter how well-cited the report is.
            "tool_calls": _ledger_to_tool_calls(ledger),
        },
        "additional_output_data": {
            "driver": "new",
            "child_return_mode": getattr(args, "child_return_mode", "prose"),
            "search_backend": args.search_backend,
            "n_surfaced_snippets": n_surfaced,
            "n_cited_ids": len(cited_ids),
            "cited_ids": cited_ids,
            "wall_clock_s": wall,
            "recursion": rec,
            "provenance": prov,   # child->root citation survival (the structured-vs-prose attribution metric)
            "trajectory_path": traj_path,
        },
    }


# ---------------------------------------------------------------------------
# Run loop — reuses generate.py's loaders / sharding / resume; lighter fail-fast
# ---------------------------------------------------------------------------
def _load_items(args) -> List[Dict[str, Any]]:
    from dr_agent.dataset_utils.load_dataset import (
        load_researchqa_data,
        load_sqav2_data,
        load_healthbench_data,
        load_deep_research_bench_data,
    )
    if args.benchmark == "researchqa":
        return load_researchqa_data(num_examples=args.num_examples)
    if args.benchmark == "sqav2":
        return load_sqav2_data(num_examples=args.num_examples)
    if args.benchmark == "healthbench":
        return load_healthbench_data(subset="hard", num_examples=args.num_examples,
                                     local_path=os.environ.get("HEALTHBENCH_LOCAL_PATH"))
    if args.benchmark == "drtulu_rl":
        # DR Tulu RL prompts (the training distribution) from a local subset jsonl:
        # rows {id, problem, rubrics:[{description,title,weight}], ...}. Rubrics ride along
        # in the row (and into original_data) so the provenance-credit POC can judge
        # per-criterion offline without re-fetching HF.
        path = os.environ.get("DRTULU_RL_LOCAL_PATH", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"drtulu_rl needs DRTULU_RL_LOCAL_PATH pointing at a subset jsonl (got {path!r})")
        items: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        if args.num_examples:
            items = items[: args.num_examples]
        return items
    if args.benchmark in ("simpleqa", "2wiki", "webwalker"):
        # Short-form guardrails from a coverage-safe local subset jsonl (built by
        # build_shortform_evalsets.py: rows {id, problem, answer|answers}, all seeded into
        # the frozen corpus). Gold rides along in the row -> original_data so DR-Tulu's
        # evaluate.py grades off-GPU without re-fetching HF.
        env = {"simpleqa": "SIMPLEQA_LOCAL_PATH", "2wiki": "TWOWIKI_LOCAL_PATH",
               "webwalker": "WEBWALKER_LOCAL_PATH"}[args.benchmark]
        path = os.environ.get(env, "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"{args.benchmark} needs {env} pointing at a subset jsonl (got {path!r})")
        items = [json.loads(l) for l in open(path) if l.strip()]
        if args.num_examples:
            items = items[: args.num_examples]
        return items
    return load_deep_research_bench_data(num_examples=args.num_examples)


async def _amain(args) -> None:
    tok_id = args.tokenizer_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    gen, client = _build_generator(args, tokenizer)
    try:
        await client.wake_up()  # no-op for the API client; kept for parity with EvalOnlyEntrypoint
    except Exception:
        pass

    items = _load_items(args)
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
        print(f"[a2/new] shard {args.shard}/{args.num_shards}: {len(items)} items")
    print(f"[a2/new] {args.benchmark}: {len(items)} items; model={args.model} "
          f"backend={args.search_backend} max_recursion_depth={args.max_recursion_depth} "
          f"child_return_mode={getattr(args, 'child_return_mode', 'prose')}")

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["example_id"])
                    except Exception:
                        pass
        print(f"[a2/new] resume: {len(done)} rows already in {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_ok = n_err = 0
    dead_streak = 0   # consecutive items that surfaced ZERO evidence (facade is silent -> proxy for dead search)
    with open(args.output, "a") as fout:
        for i, ex in enumerate(items):
            if ex["id"] in done:
                continue
            tag = f"[{i+1}/{len(items)}] id={ex['id']}"
            try:
                res = await _run_one(gen, ex, args)
                # SILENT ROOT DEATH retry: a provider blip burst at the root's last turn can
                # outlast even the 6-attempt generation backoff, ending the rollout with an
                # empty report and NO root env-metrics row (observed twice on 2026-06-12:
                # checktool pilot id=51, checktool_v3 id=66 — children fine, root vanished,
                # driver printed OK). One fresh re-run of the whole item recovers it; a
                # second empty result is accepted as a real (model) failure and recorded.
                if not (res.get("final_response") or "").strip() and not _get_hard_error().get("hit"):
                    print(f"{tag} EMPTY report (root rollout likely died on a provider blip) — retrying item once")
                    res = await _run_one(gen, ex, args)
                    res["additional_output_data"]["retried_empty"] = True
                # HARD-error abort (credits/auth/quota): the tool facade latches a run-global
                # flag on a non-transient search failure (and stops calling the backend for the
                # rest of the tree). Abort NOW — after the FIRST such item — rather than waiting
                # for the zero-evidence streak, so we never emit ungrounded rows or burn more
                # API calls. Restores legacy generate.py's hard-credit abort.
                he = _get_hard_error()
                if he.get("hit"):
                    print(f"{tag} [FATAL] hard search failure (credits/auth/quota): {he.get('msg')!r} "
                          f"— aborting so the run cannot produce ungrounded answers or burn more calls.")
                    sys.stdout.flush()
                    sys.exit(3)
                aod = res["additional_output_data"]
                if aod["n_surfaced_snippets"] == 0 and args.search_backend != "none":
                    dead_streak += 1
                    print(f"{tag} ZERO-EVIDENCE (no snippets surfaced) — not writing ungrounded row "
                          f"[dead_streak={dead_streak}]")
                    if dead_streak >= args.abort_after_dead_items:
                        print(f"[FATAL] {dead_streak} consecutive zero-evidence items — search ({args.search_backend}) "
                              f"appears unavailable; aborting so the run cannot produce ungrounded answers.")
                        sys.stdout.flush()
                        sys.exit(3)
                    continue
                dead_streak = 0
                row = {
                    "example_id": ex["id"],
                    "problem": ex["problem"],
                    "final_response": res["final_response"],
                    "full_traces": res["full_traces"],
                    "additional_output_data": aod,
                    "original_data": {**ex, "dataset_name": args.benchmark},
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n_ok += 1
                rc = aod["recursion"]
                print(f"{tag} OK {aod['wall_clock_s']:.0f}s tok={res['full_traces']['total_tokens']} "
                      f"snippets={aod['n_surfaced_snippets']} cited={aod['n_cited_ids']} "
                      f"subagents={rc['n_subagents']} depth={rc['max_depth_reached']}/{rc['max_depth_cap']} "
                      f"len={len(res['final_response'])} has_cite={'<cite' in res['final_response']}")
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"{tag} ERROR {e}\n{traceback.format_exc()}")
    print(f"[a2/new] done: {n_ok} ok, {n_err} errors -> {args.output}")


def run(args) -> None:
    """Entry point called by generate.py when --driver new. Owns its own asyncio loop (the
    generator's _setup_env_extras needs a running loop for lm_callback/subcall_fn)."""
    asyncio.run(_amain(args))
