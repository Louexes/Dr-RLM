#!/usr/bin/env python
"""W1 / Arm A2 generator: GPT-5-mini in the `rlm/` recursive REPL harness.

The recursive counterpart to A3 (GPT-5-mini FLAT in the DR-Tulu agent). To keep the
A1/A2/A3 comparison clean, A2 is held identical to A1/A3 on everything except the
generation engine + recursion:

  * SAME items   — loaded via the dr-tulu loaders (env-var subset paths), so ids
                   (RQA md5+orig_id, DRB int id) and prompts match exactly.
  * SAME tools   — the dr_agent MCP search/browse tools (online Serper + Jina) hit
                   the SAME MCP server A1/A3 use; snippets carry `<snippet id=...>`
                   ids and the model cites them with `<cite id="...">` (identical scheme).
  * SAME graders — output rows use the exact schema scripts/evaluate.py consumes
                   (example_id / original_data{orig_id,...} / problem / final_response
                   + full_traces + additional_output_data), so RQA-coverage and
                   DRB-RACE/FACT score A2 identically.
  * DIFFERENCE   — the engine is the `rlm/` recursive REPL driven by GPT-5-mini, run
                   DEEP (max_depth configurable, default deep) so we see what a frontier
                   model does with recursive decomposition; the model = GPT-5-mini, same
                   as A3 (so A2-vs-A3 isolates the recursion scaffold; A1 = DR-Tulu-8B ref).

Faithfulness notes baked in:
  * Input = the raw `problem` (the dr-tulu generate-dataset path drops the example's
    `additional_instructions` via a call-signature filter; dataset framing lives in the
    system prompt). A2 mirrors this: raw problem in, dataset framing in the system
    prompt / user prologue — NOT as an extra instruction the flat arms don't get.
  * GPT-5-mini is a reasoning model: we pass NO `top_p`/`stop` (the rlm OpenAIClient only
    forwards non-None sampling args and maps max_tokens->max_completion_tokens), so the
    API never rejects unsupported params.

This is an API-only arm (no GPU). Run it as a SLURM job on the `genoa` CPU partition —
NEVER a full run on the login node (it gets reaped). See a2_rlm.sh / a2_rlm.job.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List

REPO = "/gpfs/home5/lgehringer/Dr-RLM"
for p in (f"{REPO}/rlm", f"{REPO}/dr-tulu/agent"):
    if p not in sys.path:
        sys.path.insert(0, p)

from rlm import RLM  # noqa: E402
from rlm.environments.local_repl import LocalREPL  # noqa: E402
from rlm.clients.openai import OpenAIClient  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402  (full-trajectory capture: root + nested sub-agents)
from dr_agent.tool_interface import SerperSearchTool  # noqa: E402
from dr_agent.tool_interface.mcp_tools import JinaBrowseTool  # noqa: E402  (not re-exported in __init__)
from dr_agent.dataset_utils.load_dataset import (  # noqa: E402
    load_researchqa_data,
    load_deep_research_bench_data,
    load_sqav2_data,
    load_healthbench_data,
)


# ---------------------------------------------------------------------------
# Recursion / compute instrumentation.
#
# We need to SEE how gpt-5-mini behaves in the scaffold and keep compute comparable
# to A3. Two facts force process-level instrumentation rather than reading the
# returned RLMChatCompletion:
#   1. each recursion node spawns its OWN LMHandler, and the root only accumulates
#      child *cost* (not tokens), so result.usage_summary.total_tokens is ROOT-ONLY
#      (observed = 0 in the first run). We instead count tokens at the source — every
#      OpenAIClient LM call across the whole tree.
#   2. rlm_query / rlm_query_batched are built-in REPL functions (not our custom tools),
#      so they aren't captured by the search/browse wrappers. We count them at the REPL.
# The on_subcall_* callbacks (which DO fire and propagate to every depth) give the
# per-depth sub-agent histogram and the actual max depth reached.
#
# These are MODULE-LEVEL monkeypatches affecting every node in the process; STATS is
# reset per item. NOTE: this assumes items run sequentially (they do); item-level
# concurrency would need a thread-local context instead.
# ---------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.llm_calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.n_rlm_query = 0            # single recursive sub-agent calls
            self.n_rlm_query_batched = 0    # batched (parallel) recursive calls
            self.batch_widths = []          # children per rlm_query_batched call
            self.subcalls_by_depth = {}     # child depth -> count (whole tree)
            self.subcall_durations = []
            self.subcall_errors = 0

    def add_tokens(self, p, c):
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += int(p or 0)
            self.completion_tokens += int(c or 0)

    def inc_rlm_query(self):
        with self._lock:
            self.n_rlm_query += 1

    def add_batch(self, width):
        with self._lock:
            self.n_rlm_query_batched += 1
            self.batch_widths.append(int(width))

    def on_subcall_start(self, depth, model, prompt):
        with self._lock:
            self.subcalls_by_depth[depth] = self.subcalls_by_depth.get(depth, 0) + 1

    def on_subcall_complete(self, depth, model, duration, error):
        with self._lock:
            self.subcall_durations.append(float(duration))
            if error:
                self.subcall_errors += 1

    def recursion_summary(self, max_depth_cap: int) -> Dict[str, Any]:
        with self._lock:
            depths = dict(sorted(self.subcalls_by_depth.items()))
            n_sub = sum(depths.values())
            return {
                "max_depth_cap": max_depth_cap,
                "max_depth_reached": (max(depths) if depths else 0),
                "n_subagents": n_sub,
                "subagents_by_depth": {str(k): v for k, v in depths.items()},
                "n_rlm_query": self.n_rlm_query,
                "n_rlm_query_batched": self.n_rlm_query_batched,
                "batch_widths": list(self.batch_widths),
                "mean_batch_width": (sum(self.batch_widths) / len(self.batch_widths)) if self.batch_widths else 0.0,
                "n_llm_calls": self.llm_calls,
                "n_subcall_errors": self.subcall_errors,
                "mean_subcall_s": (sum(self.subcall_durations) / len(self.subcall_durations))
                if self.subcall_durations else 0.0,
            }


STATS = _Stats()


# Per-node (per-agent) tool budget. Unlike the tree-wide `collector["tool_calls"]` cap, this
# gives EACH node (root + every sub-agent) its OWN search/browse budget — so a recursive run is
# budget-matched PER AGENT to the flat baseline (the test-time-scaling control: each agent is no
# more capable than A3/A2-flat; recursion only adds *more agents*). It's a thread-local: batched
# sub-agents run in pooled worker threads, and `on_subcall_start` fires in the child's own thread
# right before it runs, so we reset there (robust even if the pool reuses a thread); the
# `_rlm_query`/`_rlm_query_batched` wrappers save+restore the caller's count around an in-thread
# (single) subcall so a child's spend isn't charged to its parent.
_NODE_BUDGET = threading.local()


def _node_n() -> int:
    return getattr(_NODE_BUDGET, "n", 0)


def _on_subcall_start(depth, model, prompt):
    _NODE_BUDGET.n = 0  # fresh per-agent tool budget for the child (runs in this thread)
    STATS.on_subcall_start(depth, model, prompt)


# Markers of a HARD, non-transient search failure (dead/empty key, auth, quota) — used by the
# fail-fast guard so a broken backend can never silently produce a whole run of UNGROUNDED,
# uncited answers (the 2026-05-31 "Serper out of credits" incident).
_HARD_ERR_MARKERS = (
    "not enough credits", "insufficient credit", "status 400", "status 401", "status 402",
    "status 403", "unauthorized", "forbidden", "invalid api key", "api key", "quota", "payment",
)


def _looks_like_dead_search(errors) -> bool:
    blob = " ".join(str(e) for e in (errors or [])).lower()
    return any(m in blob for m in _HARD_ERR_MARKERS)


def _install_instrumentation():
    """Wrap OpenAIClient token tracking + the REPL recursion entrypoints (idempotent)."""
    if getattr(_install_instrumentation, "_done", False):
        return
    _orig_track = OpenAIClient._track_cost

    def _track(self, response, model):  # whole-tree token count, at the source
        _orig_track(self, response, model)
        u = getattr(response, "usage", None)
        if u is not None:
            try:
                STATS.add_tokens(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            except Exception:
                pass

    OpenAIClient._track_cost = _track

    _orig_rq = LocalREPL._rlm_query

    def _rq(self, prompt, model=None):
        STATS.inc_rlm_query()
        _saved = _node_n()  # an in-thread (single) subcall must not charge the parent's budget
        try:
            return _orig_rq(self, prompt, model)
        finally:
            _NODE_BUDGET.n = _saved

    LocalREPL._rlm_query = _rq

    _orig_rqb = LocalREPL._rlm_query_batched

    def _rqb(self, prompts, model=None):
        try:
            STATS.add_batch(len(prompts))
        except Exception:
            pass
        _saved = _node_n()
        try:
            return _orig_rqb(self, prompts, model)
        finally:
            _NODE_BUDGET.n = _saved

    LocalREPL._rlm_query_batched = _rqb
    _install_instrumentation._done = True


_install_instrumentation()


# ---------------------------------------------------------------------------
# Background asyncio loop so the (async) MCP tools can be called from the
# synchronous rlm/ REPL (and from its child threads) without a running loop.
# ---------------------------------------------------------------------------
class _BackgroundLoop:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 120.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)


# ---------------------------------------------------------------------------
# LEGACY-driver system prompt (RLM core + orchestrator addendum, baked together) —
# frozen verbatim as used in all v1 runs (gpt-5-mini DRB/RQA/sqav2, HealthBench, probes).
# Passed to RLM(custom_system_prompt=...) with orchestrator=False so the engine does NOT
# re-append its built-in addendum. The prompt to ITERATE on going forward is
# prompts/system_prompt.txt (the unified DrRlmEnv driver, --driver new).
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_FILE = f"{REPO}/dr-rlm/prompts/legacy_system_prompt.txt"


def _load_system_prompt() -> str:
    with open(_SYSTEM_PROMPT_FILE) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Dataset-specific framing — mirrors the canonical task spec (verbatim from the
# dr-tulu loaders' `additional_instructions`) but delivered as system framing, so
# A2 is on equal footing with A1/A3 (which encode question-type guidance in the
# agent's system prompt, not as a user instruction).
# ---------------------------------------------------------------------------
_RQA_FRAMING = (
    "Answer the question completely and precisely in around 240-260 words, in one to "
    "three paragraphs (do not enumerate the facts). Support every statement in the answer "
    "with an in-line citation to a retrieved snippet."
)
_DRB_FRAMING = (
    "Write a well-structured, data-driven research report that thoroughly answers the "
    "research question. Support claims with in-line citations to retrieved snippets."
)
_SQA_FRAMING = (
    "Write a well-structured, data-driven report that thoroughly answers the scientific "
    "research question, grounded in the literature. Support every claim with an in-line "
    "citation to a retrieved snippet."
)
_HB_FRAMING = (
    "Answer the patient's medical question thoroughly, accurately, and safely, grounded in "
    "retrieved evidence with in-line citations. Seek or acknowledge missing context where it "
    "matters, hedge appropriately under uncertainty, flag emergencies and when to seek "
    "in-person care, and communicate clearly for the reader's apparent expertise level."
)

# search/browse return their results but do NOT auto-print, so retrieved text enters context ONLY
# when the model chooses to print it (context decoupling — the default behavior).
_PROLOGUE_BASE = (
    "You are a deep-research agent with NO preloaded context. Gather all evidence yourself:\n"
    "- `search(query)` RETURNS a string of web snippets (each `<snippet id=ID>...</snippet>`) but does "
    "NOT print it. Assign it to a variable, then PRINT ONLY what you need to read or cite — typically the "
    "snippet IDs and the few snippets you will actually use. NEVER print whole result blobs: everything you "
    "print is permanently added to your limited context, so be economical.\n"
    "- `browse(url)` likewise RETURNS a page's content silently; print only the short excerpt you need.\n"
    "- Keep retrieved text in variables across turns and process it with code (slice, str-search, filter, "
    "extract) instead of re-printing it. To cite a snippet you only need to have seen its `id` once.\n"
)
_PROLOGUE_RECURSION = (
    "- For hard, multi-part questions, DECOMPOSE into focused sub-questions and delegate them "
    "IN PARALLEL with `rlm_query_batched([subq1, subq2, ...])` (each spawns a sub-agent with its "
    "own search/browse + REPL and returns a cited mini-report). You may recurse deeply — build a "
    "tree of sub-agents as the question warrants — then synthesize their findings.\n"
)
_PROLOGUE_CITE = (
    "- CITE every non-trivial claim inline as `<cite id=\"ID\">the claim</cite>` (with the "
    "DOUBLE QUOTES around the id — the grader's extractor requires them), using the exact "
    "snippet IDs you saw in search results (preserve sub-agents' citation IDs when you reuse their "
    "evidence). Only cite IDs that appeared in your results.\n"
)


def _user_prologue(dataset_name: str, recursive: bool = True) -> str:
    """Build the system framing. The recursion bullet is included only when the run is actually
    recursive (max_depth > 1), so the FLAT arm (A2-flat) is never told to delegate. search/browse
    are silent by default; the base bullets teach selective inspection (context decoupling)."""
    framing = {"researchqa": _RQA_FRAMING, "sqav2": _SQA_FRAMING, "healthbench": _HB_FRAMING}.get(dataset_name, _DRB_FRAMING)
    prologue = _PROLOGUE_BASE + (_PROLOGUE_RECURSION if recursive else "") + _PROLOGUE_CITE
    return prologue + "\nTASK FORMAT: " + framing


# ---------------------------------------------------------------------------
# Tool descriptions + a tree-wide output-discipline block. These reach EVERY agent
# (root + sub-agents): tool descriptions via custom_sub_tools (propagated to children,
# rlm.py:826-827) and the discipline block via custom_system_prompt (propagated
# verbatim, rlm.py:818). This is what stops sub-agents from re-dumping retrieved text —
# user_prologue alone is root-only (children never receive it).
# NB: no literal { } braces here — the system prompt is later .format()-ed.
# ---------------------------------------------------------------------------
_SEARCH_DESC = (
    "Search the web. RETURNS a string of `<snippet id=ID>...</snippet>` results but does NOT print "
    "them. Assign the return to a variable; print ONLY the snippet IDs and the few snippets you will "
    "actually cite — never the whole result blob. You only need to print a snippet's id once to cite it."
)
_BROWSE_DESC = (
    "Fetch a web page. RETURNS the page content but does NOT print it. Assign to a variable and print "
    "only the short excerpt you need."
)
_OUTPUT_DISCIPLINE = (
    "\n\nTOOL OUTPUT DISCIPLINE: search() and browse() return their results to you as values but do "
    "NOT print them — and this rule applies to you AND to every sub-agent you spawn. Assign results to "
    "variables and process them with code; print ONLY the snippet IDs and the specific short snippets "
    "you will cite. Never print whole search/browse results: everything printed is permanently added to "
    "your (and each sub-agent's) limited context window. To cite a snippet you only need to have printed "
    "its id once."
)


def _system_prompt() -> str:
    """The tracked system prompt + the tree-wide output-discipline block (always on)."""
    return _load_system_prompt() + _OUTPUT_DISCIPLINE


# ---------------------------------------------------------------------------
# REPL tools: thin sync wrappers over the dr_agent MCP search/browse tools.
# ---------------------------------------------------------------------------
def make_tools(mcp_port: int, loop: _BackgroundLoop, collector: Dict[str, Any], top_k: int,
               max_tool_calls: int, budget_scope: str = "tree"):
    search_tool = SerperSearchTool(mcp_port=mcp_port, number_documents_to_search=top_k)
    browse_tool = JinaBrowseTool(mcp_port=mcp_port)
    lock = threading.Lock()
    per_node = (budget_scope == "node")
    # Tool budget. `tree`: ONE counter caps the whole item (the SAME closures are shared across
    # root + every sub-agent, since custom_tools == custom_sub_tools). `node`: each agent gets its
    # own `max_tool_calls` via the `_NODE_BUDGET` thread-local — budget-matched PER AGENT to the
    # flat baseline, so recursion's only added ingredient is *more agents* (the TTS control).
    if per_node:
        budget_msg = (
            f"[TOOL BUDGET EXHAUSTED: this agent has used its {max_tool_calls} search/browse "
            "calls. Do NOT call search() or browse() again. Synthesize your answer from the "
            "evidence already gathered and submit it now via the answer mechanism.]"
        )
    else:
        budget_msg = (
            f"[TOOL BUDGET EXHAUSTED: {max_tool_calls} search/browse calls already used for this "
            "question. Do NOT call search() or browse() again. Synthesize your final answer from the "
            "evidence already gathered and submit it now via the answer mechanism.]"
        )

    def _over_budget() -> bool:
        if per_node:
            return _node_n() >= max_tool_calls
        with lock:
            return collector["tool_calls"] >= max_tool_calls

    def _charge_node():
        if per_node:
            _NODE_BUDGET.n = _node_n() + 1

    def _bump(key, n=1):
        with lock:
            collector[key] = collector.get(key, 0) + n

    def search(query: str, k: int = top_k):
        """Search the web. Returns `<snippet id=...>` blocks (silent — does not print); print only what you cite."""
        if _over_budget():
            _bump("budget_blocked")
            print(budget_msg)
            return budget_msg
        try:
            out = loop.run(search_tool(str(query)))
        except Exception as e:  # never crash the REPL turn on a tool hiccup
            print(f"(search error: {e})")
            _bump("tool_calls"); _bump("failed"); _charge_node()
            with lock:
                collector["errors"].append(f"search: {e}")  # recorded for the fail-fast guard
            return ""
        _bump("tool_calls"); _bump("search"); _charge_node()
        if getattr(out, "error", ""):
            _bump("failed")
            with lock:
                collector["errors"].append(str(out.error))
        for d in (getattr(out, "documents", None) or []):
            if getattr(d, "url", None):
                with lock:
                    collector["searched_links"].add(d.url)
        s = search_tool._format_output(out)
        return s  # silent: results enter context only when the MODEL prints them (context decoupling)

    def browse(url: str):
        """Fetch a web page for closer reading. Returns content (silent — does not print); print only the excerpt you need."""
        if _over_budget():
            _bump("budget_blocked")
            print(budget_msg)
            return budget_msg
        try:
            out = loop.run(browse_tool(str(url)))
        except Exception as e:
            print(f"(browse error: {e})")
            _bump("tool_calls"); _bump("failed"); _charge_node()
            with lock:
                collector["errors"].append(f"browse: {e}")  # recorded for the fail-fast guard
            return ""
        _bump("tool_calls"); _bump("browse"); _charge_node()
        if getattr(out, "error", ""):
            _bump("failed")
        for d in (getattr(out, "documents", None) or []):
            if getattr(d, "url", None):
                with lock:
                    collector["browsed_links"].add(d.url)
        s = browse_tool._format_output(out)
        return s  # silent: model prints only the excerpt it needs (context decoupling)

    # dict form => the description reaches the system prompt's custom-tools section, which propagates
    # to sub-agents (so the WHOLE tree learns search/browse are silent). Plain callables would get no
    # description and the discipline would be root-only.
    return {"search": {"tool": search, "description": _SEARCH_DESC},
            "browse": {"tool": browse, "description": _BROWSE_DESC}}


def _new_collector() -> Dict[str, Any]:
    return {"tool_calls": 0, "search": 0, "browse": 0, "failed": 0, "errors": [],
            "budget_blocked": 0, "searched_links": set(), "browsed_links": set()}


def run_item(problem: str, dataset_name: str, args, mcp_port: int, loop: _BackgroundLoop,
             example_id: Any = None) -> Dict[str, Any]:
    STATS.reset()  # per-item (items run sequentially)
    _NODE_BUDGET.n = 0  # reset the root's per-agent tool budget (the root runs on this thread)
    collector = _new_collector()
    recursive = args.max_depth > 1  # rlm injects subcall_fn (enables recursion) only when max_depth > 1
    tools = make_tools(mcp_port, loop, collector, args.top_k, args.max_tool_calls, args.tool_budget_scope)
    # Full-trajectory capture (opt-in via --log-dir). The ROOT logger writes a per-item JSONL
    # (metadata + every iteration); each iteration carries its code_blocks[].result.rlm_calls[],
    # and every sub-agent gets its OWN in-memory logger whose trajectory is attached to that
    # call's .metadata — so a single root file holds the ENTIRE recursion tree, turn by turn.
    logger = None
    if getattr(args, "log_dir", None):
        os.makedirs(args.log_dir, exist_ok=True)
        logger = RLMLogger(log_dir=args.log_dir, file_name=f"traj_{dataset_name}_{example_id}")
    sampling = {"max_tokens": args.max_completion_tokens}  # -> max_completion_tokens; no top_p/stop
    backend_kwargs = {"model_name": args.model}
    if args.base_url:  # route to a non-OpenAI provider via its OpenAI-compatible endpoint (e.g. Gemini)
        backend_kwargs["base_url"] = args.base_url
        backend_kwargs["api_key"] = os.environ.get(args.api_key_env, "")
    rlm = RLM(
        backend="openai",
        backend_kwargs=backend_kwargs,
        environment="local",
        depth=0,
        max_depth=args.max_depth,                       # 1 = flat REPL (no recursion); >1 = recursive
        max_iterations=args.max_iterations,
        max_timeout=args.item_timeout,                  # per-item wall-clock guard (rlm-enforced, propagated to children)
        max_concurrent_subcalls=args.max_concurrent_subcalls,
        custom_tools=tools,
        custom_sub_tools=tools,                          # children get search/browse too
        custom_system_prompt=_system_prompt(),           # core prompt + output-discipline block (reaches all sub-agents)
        orchestrator=False,                              # addendum is baked into custom_system_prompt; don't re-append (root + children)
        user_prologue=_user_prologue(dataset_name, recursive),
        sampling_args=sampling,
        sub_sampling_args=sampling,
        on_subcall_start=_on_subcall_start,              # resets each child's per-agent budget + telemetry
        on_subcall_complete=STATS.on_subcall_complete,
        logger=logger,                                   # None => no capture (default); set => full-tree JSONL
        verbose=False,
    )
    t0 = time.perf_counter()
    try:
        result = rlm.completion(problem)
        report = result.response or ""
    finally:
        wall = time.perf_counter() - t0
        try:
            rlm.close()
        except Exception:
            pass

    # whole-tree token count (root + all sub-agents), captured at the OpenAIClient level
    tree_tokens = STATS.prompt_tokens + STATS.completion_tokens
    recursion = STATS.recursion_summary(args.max_depth)

    return {
        "final_response": report,
        "full_traces": {
            "generated_text": report,
            "total_tokens": tree_tokens,                 # WHOLE TREE (comparable to A3's single-agent total)
            "prompt_tokens": STATS.prompt_tokens,
            "completion_tokens": STATS.completion_tokens,
            "tool_call_count": collector["tool_calls"],
            "stopped_reason": "natural",
            "tool_calls": [],  # graders score final_response; trace kept lightweight
        },
        "additional_output_data": {
            "browsed_links": sorted(collector["browsed_links"]),
            "searched_links": sorted(collector["searched_links"]),
            "total_tool_calls": collector["tool_calls"],          # search+browse (A1/A3-comparable)
            "n_search": collector["search"],
            "n_browse": collector["browse"],
            "total_failed_tool_calls": collector["failed"],
            "failed_tool_call_errors": collector["errors"],
            "budget_blocked_calls": collector.get("budget_blocked", 0),
            "wall_clock_s": wall,
            "guards": {"item_timeout_s": args.item_timeout, "max_tool_calls": args.max_tool_calls,
                       "tool_budget_scope": args.tool_budget_scope, "orchestrator": args.orchestrator,
                       "hit_time_guard": wall >= args.item_timeout * 0.97},
            "recursion": recursion,                                # rlm_query / rlm_query_batched / depth profile
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True,
                    choices=["researchqa", "deep_research_bench", "sqav2", "healthbench", "drtulu_rl",
                             "simpleqa", "2wiki", "webwalker"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-examples", type=int, default=None, help="cap; None=all of the subset")
    ap.add_argument("--mcp-port", type=int, default=8030)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base_url for a non-OpenAI provider (e.g. Gemini: https://generativelanguage.googleapis.com/v1beta/openai/)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="env var holding the API key for --base-url (e.g. GEMINI_API_KEY)")
    ap.add_argument("--max-depth", type=int, default=5,
                    help="rlm recursion: 1 = flat REPL (no sub-agents); 2 = root + real sub-agents; "
                         ">2 = deeper trees. (rlm injects subcall_fn only when max_depth>1.)")
    ap.add_argument("--max-iterations", type=int, default=12, help="REPL turn budget per node")
    ap.add_argument("--max-concurrent-subcalls", type=int, default=4)
    ap.add_argument("--max-completion-tokens", type=int, default=12000)
    ap.add_argument("--top-k", type=int, default=10, help="search results per search() call")
    ap.add_argument("--log-dir", default=None,
                    help="if set, write a per-item full trajectory JSONL (root + nested sub-agents: "
                         "every iteration's model response, repl code, stdout, locals, child calls) here")
    # ---- per-item runaway guards (a pilot run had a 4.4M-token / 62-min item) ----
    ap.add_argument("--item-timeout", type=float, default=900.0, help="per-item wall-clock cap (s); rlm max_timeout, propagated to children")
    ap.add_argument("--max-tool-calls", type=int, default=80, help="search+browse budget; scope set by --tool-budget-scope; further calls return a 'finalize now' message")
    ap.add_argument("--tool-budget-scope", choices=["tree", "node"], default="tree",
                    help="'tree' = one budget for the whole item (legacy A2); 'node' = max-tool-calls PER agent (the per-agent / TTS control)")
    ap.add_argument("--orchestrator", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-orchestrator drops the decompose/delegate addendum (use for the FLAT arm)")
    # ---- fail-fast guard: never silently emit a whole run of UNGROUNDED answers (dead search key) ----
    ap.add_argument("--fail-fast-after", type=int, default=3,
                    help="an item with 0 SUCCESSFUL tool calls AND >= this many FAILED calls is a 'dead-search' item")
    ap.add_argument("--abort-after-dead-items", type=int, default=2,
                    help="abort the run after this many consecutive dead-search items (or immediately on a hard auth/credit error)")
    # process-level sharding for parallelism WITHOUT cross-item instrumentation bleed:
    # each shard is its own process running its items sequentially (STATS stays isolated).
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    # ---- --driver new: the UNIFIED SkyRL DrRlmEnv driver (shared prompt + tool facade + answer-dict/ledger,
    #      same code RL trains); the default 'legacy' rlm/-engine path below is UNCHANGED. See infer_driver.py ----
    ap.add_argument("--driver", choices=["legacy", "new"], default="legacy",
                    help="legacy = the rlm/ engine (default, unchanged); new = the unified SkyRL DrRlmEnv driver")
    ap.add_argument("--search-backend", default="mcp_http",
                    help="[--driver new] online/offline tool facade: web | mcp_http | bm25 | bm25s | faiss | local_jsonl | none")
    ap.add_argument("--search-endpoint", default="http://localhost:8003/mcp",
                    help="[--driver new, mcp_http] dr_agent FastMCP local_search endpoint")
    ap.add_argument("--search-corpus-path", default="data/corpus.jsonl",
                    help="[--driver new, local_jsonl/bm25/faiss] corpus.jsonl path")
    ap.add_argument("--search-index-path", default="data/bm25",
                    help="[--driver new, bm25s/bm25/faiss] index dir (bm25s: the frozen_corpus/bm25s_index dir)")
    ap.add_argument("--max-recursion-depth", type=int, default=0,
                    help="[--driver new] 0 = flat (worker root); >=1 = recursive sub-agents")
    ap.add_argument("--child-return-mode", choices=["prose", "structured"], default="prose",
                    help="[--driver new] child->parent rlm_query() contract: 'prose' = rendered "
                         "report string (v1); 'structured' = {content, citations} dict the orchestrator "
                         "merges programmatically (the prose-vs-structured A/B)")
    ap.add_argument("--max-children", type=int, default=0,
                    help="[--driver new] per-NODE cap on child sub-agents spawned (0 = unlimited); "
                         "bounds tree width (the tool-call budget bounds Serper calls, not fan-out)")
    ap.add_argument("--citation-verification", action="store_true", default=False,
                    help="[--driver new] pre-submit citation verification: at finalize, batch-judge "
                         "each (claim, snippet) pair via the LM and DROP explicit UNSUPPORTED ones "
                         "before rendering (targets loose citing / low FACT valid_rate). "
                         "NB: architecture-agnostic -> OFF in head-to-heads vs DR-Tulu")
    ap.add_argument("--citations-bounce", action="store_true", default=False,
                    help="[--driver new] one-time empty-citations submission gate: the FIRST "
                         "ready=True with non-empty content, empty citations, and tree evidence "
                         "is rejected with a repair message (resubmit-as-is allowed)")
    ap.add_argument("--check-citations-tool", action="store_true", default=False,
                    help="[--driver new] expose check_citations(citations) in every node's REPL: "
                         "an oracle returning per-entry SUPPORTED/UNSUPPORTED/UNKNOWN verdicts "
                         "judged against the shared ledger (child-inherited entries included). "
                         "Never filters — the agent acts on the verdicts (drop/re-point/rewrite). "
                         "Advertise via a prompt variant (prompts/variants/check_tool.txt)")
    ap.add_argument("--save-trajectories", default=None, metavar="DIR",
                    help="[--driver new] capture each item's full recursion-tree trajectory "
                         "(per-node, per-turn: response/code/stdout + config) as parseable JSONL in DIR; "
                         "render with agent/viz_trajectory_v2.py")
    ap.add_argument("--tokenizer-path", default=None,
                    help="[--driver new] HF tokenizer/model id for apply_chat_template (defaults to --model)")
    ap.add_argument("--max-input-length", type=int, default=32768,
                    help="[--driver new] per-turn input token cap (agent_loop length guard)")
    ap.add_argument("--temperature", type=float, default=0.7, help="[--driver new] sampling temperature")
    # Qwen3.5-4B NON-THINKING (instruct) recommended inference sampling (model card). presence_penalty
    # in particular suppresses the bare-`search()` repetition loop that greedy/penalty-free decoding
    # induces in the untrained policy. These are the eval defaults; pass explicitly to override.
    ap.add_argument("--top-p", type=float, default=0.8, help="[--driver new] nucleus top_p (Qwen3.5 non-thinking rec: 0.8)")
    ap.add_argument("--sampling-top-k", type=int, default=20, help="[--driver new] sampling top_k (Qwen3.5 rec: 20)")
    ap.add_argument("--min-p", type=float, default=0.0, help="[--driver new] min_p (Qwen3.5 rec: 0.0)")
    ap.add_argument("--presence-penalty", type=float, default=1.5,
                    help="[--driver new] presence_penalty (Qwen3.5 non-thinking rec: 1.5; suppresses search-loop repetition)")
    args = ap.parse_args()

    if args.driver == "new":
        # Delegate the whole run to the unified driver; the legacy rlm/ path below is never touched.
        from infer_driver import run as _run_new
        _run_new(args)
        return

    if args.benchmark == "researchqa":
        items = load_researchqa_data(num_examples=args.num_examples)
    elif args.benchmark == "sqav2":
        items = load_sqav2_data(num_examples=args.num_examples)
    elif args.benchmark == "healthbench":
        items = load_healthbench_data(subset="hard", num_examples=args.num_examples,
                                      local_path=os.environ.get("HEALTHBENCH_LOCAL_PATH"))
    else:
        items = load_deep_research_bench_data(num_examples=args.num_examples)
    if args.num_shards > 1:
        # deterministic round-robin slice (loaders return a stable order; no shuffle)
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
        print(f"[a2] shard {args.shard}/{args.num_shards}: {len(items)} items")
    print(f"[a2] {args.benchmark}: {len(items)} items this process; model={args.model} max_depth={args.max_depth} mcp:{args.mcp_port}")

    # resume by example_id (mirrors A3 --use-cache: skip already-generated rows)
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
        print(f"[a2] resume: {len(done)} rows already in {args.output}")

    loop = _BackgroundLoop()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_ok = n_err = 0
    dead_streak = 0  # consecutive items where search produced nothing (fail-fast guard)
    with open(args.output, "a") as fout:
        for i, ex in enumerate(items):
            if ex["id"] in done:
                continue
            tag = f"[{i+1}/{len(items)}] id={ex['id']}"
            try:
                t0 = time.perf_counter()
                res = run_item(ex["problem"], args.benchmark, args, args.mcp_port, loop, example_id=ex["id"])
                aod = res["additional_output_data"]
                rc = aod["recursion"]
                # --- fail-fast: a dead search backend must NOT silently fill a run with ungrounded rows ---
                ok_calls = aod["n_search"] + aod["n_browse"]
                hard = _looks_like_dead_search(aod["failed_tool_call_errors"])
                if hard or (ok_calls == 0 and aod["total_failed_tool_calls"] >= args.fail_fast_after):
                    dead_streak += 1
                    print(f"{tag} SEARCH-DEAD (ok_calls={ok_calls} failed={aod['total_failed_tool_calls']} "
                          f"hard_err={hard}) — NOT writing ungrounded row [dead_streak={dead_streak}]")
                    if hard or dead_streak >= args.abort_after_dead_items:
                        why = "hard auth/credit error" if hard else f"{dead_streak} consecutive dead-search items"
                        print(f"[FATAL] search appears unavailable ({why}). Aborting so this run cannot "
                              f"silently produce ungrounded, uncited answers — fix search (Serper credits/key) and rerun.")
                        sys.stdout.flush()
                        sys.exit(3)
                    continue  # skip this (possibly transient) item; do not write an ungrounded row
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
                print(f"{tag} OK {time.perf_counter()-t0:.0f}s tok={res['full_traces']['total_tokens']} "
                      f"search={aod['n_search']} browse={aod['n_browse']} "
                      f"rlm_query={rc['n_rlm_query']} rlm_query_batched={rc['n_rlm_query_batched']} "
                      f"subagents={rc['n_subagents']} depth_reached={rc['max_depth_reached']}/{rc['max_depth_cap']} "
                      f"by_depth={rc['subagents_by_depth']} blocked={aod['budget_blocked_calls']} "
                      f"timeguard={aod['guards']['hit_time_guard']} len={len(res['final_response'])} "
                      f"cited={'<cite' in res['final_response']}")
            except Exception as e:
                n_err += 1
                print(f"{tag} ERROR {e}\n{traceback.format_exc()}")
    print(f"[a2] done: {n_ok} ok, {n_err} errors -> {args.output}")


if __name__ == "__main__":
    main()
