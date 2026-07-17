"""Recursive cold-start SFT trajectory generator for DR-RLM.

This is the OPTIONAL recursive cold-start (mirroring NovaSky/Sky-T1's mandatory
SFT cold-start before RL): for each training prompt we sample N recursive RLM
rollouts (depth>1) from the *inference* arm (the ``rlm/`` library's ``RLM`` driven
by a local vLLM server), score each root report with the SAME held-constant RER
rubric judge used by the RL reward, and keep the best rollout iff its report score
``R >= threshold`` (rejection sampling). The kept recursion tree is then serialized
to a LLaMA-Factory ``sharegpt`` row (``conversations=[{role, content}, ...]``) whose
assistant turns interleave the REPL think/code, the retrieved search snippets
(wrapped in ``<tool_output>...</tool_output>``), and any delegated sub-agent answers
(wrapped in ``<subagent_output>...</subagent_output>``) so that SFT *span masking*
can mask both spans (loss is computed only on the policy's own reasoning + the final
cited report, not on injected retrieval/sub-agent text).

CONTROLLED-EXPERIMENT DEFAULT
-----------------------------
The DEFAULT for the controlled A/B against DR Tulu is to **warm-start BOTH arms from
DR Tulu-8B** (no recursive SFT at all) and try **RL-only first**. Recursive SFT (this
pipeline) is an OPTIONAL cold-start to run only if RL-from-warm-start fails to induce
genuine depth>1 trees (e.g. the policy collapses to flat single-agent behavior). Keep
the judge, corpus, retriever, and rubric set held constant between the SFT data
generation here and the downstream RL so the comparison stays controlled.

What this matches (verified against the real APIs — do not invent flags):
  * ``rlm.core.rlm.RLM`` — ``RLM(backend='vllm', backend_kwargs={model_name, base_url},
    environment='local', max_depth>=2, max_concurrent_subcalls, custom_system_prompt,
    custom_tools={...}, logger=RLMLogger()).completion(prompt) -> RLMChatCompletion``.
    The recursion tree is captured in ``RLMChatCompletion.metadata`` (run_metadata +
    iterations) when an ``RLMLogger`` is attached, AND the live REPL sub-agent calls are
    on ``REPLResult.rlm_calls`` (each a nested ``RLMChatCompletion`` whose own
    ``.metadata`` holds that sub-agent's trajectory). See rlm/core/types.py.
  * ``judge.score_report_sync(report, question, rubrics, JudgeConfig.from_env_payload(payload))``
    — the same sync rubric scorer the L1 env reward uses (byte-identical to the async path).
  * ``prompts.depth_system_prompt(depth, max_recursion_depth)`` — the depth-banded
    orchestrator/coordinator/worker system prompt.
  * ``corpus_search.make_corpus_tools(payload, node_rid, surfaced_ids)`` — the offline
    ``search()`` / ``get_doc()`` REPL tools with provenance-encoded snippet ids
    (``"{node_rid}-{n}"``), held constant with the RL arm.
  * The ``<cite id=...>`` citation contract is preserved verbatim from the report text.

INPUT
-----
``--prompts`` is a parquet/jsonl produced by the (sibling) ``convert_drtulu_rl`` step,
in the SkyRL RL row schema confirmed in the data-schema map:
  prompt: list[{role, content}]    (the question lives in the last user turn)
  env_class: "dr_rlm"
  reward_spec: {ground_truth, rubrics: [{description, title, weight}]}
  question: str                    (raw question; optional — falls back to prompt)
  extra_info / uid: optional
We read the rubric list from ``reward_spec.rubrics`` (the env's contract) and feed it to
both the judge and (for documentation) the kept row.

OUTPUT
------
``--out_jsonl``: one JSON object per line with a single ``conversations`` key (sharegpt),
ready to register via ``dataset_info_entry.json`` and train with ``dr_rlm_sft.yaml``.

Run (example):
  python -m examples.train.dr_rlm.sft.gen_recursive_sft \
      --prompts ~/data/dr_rlm_rl/train.parquet \
      --out_jsonl ~/data/dr_rlm_sft/dr_rlm_recursive_sft.jsonl \
      --n_samples 4 --max_depth 2 --threshold 0.5 \
      --model Qwen/Qwen3-8B --base_url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Make the dr_rlm package importable as ``examples.train.dr_rlm.*`` and the rlm library
# importable as ``rlm.*`` whether this is run as a module or a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DR_RLM_DIR = os.path.dirname(_THIS_DIR)
_SKYRL_EXAMPLES_ROOT = os.path.abspath(os.path.join(_DR_RLM_DIR, "..", "..", ".."))  # .../SkyRL
_RLM_LIB = "/gpfs/home5/lgehringer/Dr-RLM/rlm"
for _p in (_SKYRL_EXAMPLES_ROOT, _RLM_LIB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loguru import logger

# rlm inference arm
from rlm.core.rlm import RLM
from rlm.core.types import RLMChatCompletion
from rlm.logger import RLMLogger

# held-constant judge + depth-banded prompt + corpus tools (the RL arm's exact pieces)
from examples.train.dr_rlm.judge import JudgeConfig, score_report_sync
from examples.train.dr_rlm.prompts import delegation_section, depth_system_prompt
from examples.train.dr_rlm.corpus_search import make_corpus_tools
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig


# --- span-masking tag vocabulary (kept in sync with mask_span_types in dr_rlm_sft.yaml) ---
TOOL_OPEN, TOOL_CLOSE = "<tool_output>", "</tool_output>"
SUBAGENT_OPEN, SUBAGENT_CLOSE = "<subagent_output>", "</subagent_output>"


# ---------------------------------------------------------------------------
# Prompt loading (SkyRL RL row schema from convert_drtulu_rl)
# ---------------------------------------------------------------------------


def load_prompts(path: str) -> List[Dict[str, Any]]:
    """Load RL rows from parquet or jsonl. Each row is the SkyRL dr_rlm RL schema:
    {prompt:[{role,content}], env_class, reward_spec:{rubrics:[...]}, question, ...}."""
    rows: List[Dict[str, Any]] = []
    if path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        for rec in df.to_dict(orient="records"):
            rows.append(_normalize_row(rec))
    elif path.endswith(".jsonl") or path.endswith(".json"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(_normalize_row(json.loads(line)))
    else:
        raise ValueError(f"Unsupported prompts file (want .parquet/.jsonl): {path}")
    return rows


def _normalize_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce numpy/arrow scalars and nested JSON-strings into plain python.
    reward_spec may arrive as a JSON string (parquet) — parse it; rubrics live under
    reward_spec.rubrics per the env contract."""
    out = dict(rec)
    # reward_spec / rubrics may be a JSON string
    rs = out.get("reward_spec")
    if isinstance(rs, str):
        try:
            rs = json.loads(rs)
        except Exception:
            rs = {}
    out["reward_spec"] = rs or {}
    # prompt may be a JSON string or a numpy array of dicts
    pr = out.get("prompt")
    if isinstance(pr, str):
        try:
            pr = json.loads(pr)
        except Exception:
            pr = [{"role": "user", "content": pr}]
    if hasattr(pr, "tolist"):
        pr = pr.tolist()
    out["prompt"] = [dict(m) for m in (pr or [])]
    return out


def question_of(row: Dict[str, Any]) -> str:
    """The raw question: prefer ``row['question']``, else the last user turn."""
    q = row.get("question")
    if isinstance(q, str) and q.strip():
        return q
    for m in reversed(row.get("prompt", [])):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    # last resort: concatenate everything
    return "\n".join(str(m.get("content", "")) for m in row.get("prompt", []))


def rubrics_of(row: Dict[str, Any]) -> List[dict]:
    """The held-constant rubric list [{description,title,weight}] from reward_spec."""
    rs = row.get("reward_spec") or {}
    rubrics = rs.get("rubrics") or []
    # tolerate a JSON-string rubric list
    if isinstance(rubrics, str):
        try:
            rubrics = json.loads(rubrics)
        except Exception:
            rubrics = []
    return list(rubrics) if rubrics else []


# ---------------------------------------------------------------------------
# Recursion-tree -> sharegpt serialization
# ---------------------------------------------------------------------------


def _stdout_to_tool_output(stdout: str) -> str:
    """Wrap one REPL block's stdout (the printed search hits / prints) so SFT masks it."""
    text = (stdout or "").strip()
    if not text:
        return ""
    return f"{TOOL_OPEN}\n{text}\n{TOOL_CLOSE}"


def _rlm_calls_to_subagent_output(rlm_calls: List[Any]) -> str:
    """Wrap the answer string returned by each delegated sub-agent so SFT masks it.

    ``rlm_calls`` is ``REPLResult.rlm_calls`` — a list of RLMChatCompletion (live tree).
    Each child's ``.response`` is the focused, cited mini-report it returned to the
    orchestrator. We keep the child's ``<cite id=...>`` tags verbatim (provenance flows
    up), but mark the whole returned block as sub-agent output so it is masked at SFT
    (the policy is trained to *synthesize over* sub-agent answers, not to memorize them)."""
    blocks: List[str] = []
    for call in rlm_calls or []:
        resp = _get(call, "response", "")
        if resp and str(resp).strip():
            blocks.append(f"{SUBAGENT_OPEN}\n{str(resp).strip()}\n{SUBAGENT_CLOSE}")
    return "\n\n".join(blocks)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an RLMChatCompletion/REPLResult object or its dict form."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def node_system_prompt(depth: int, env_ceiling: int, child_return_mode: str) -> str:
    """The EXACT system prompt DrRlmEnv serves a node: depth band + env-style fill of
    ``{custom_tools_section}`` via plain .replace (delegation section iff the node can
    delegate, else empty) — mirrors ``DrRlmEnv._build_system_prompt``."""
    band = depth_system_prompt(depth=depth, max_recursion_depth=env_ceiling,
                               child_return_mode=child_return_mode)
    section = delegation_section(child_return_mode) if depth < env_ceiling else ""
    return band.replace("{custom_tools_section}", section)


def _escape_braces(s: str) -> str:
    """The rlm library runs ``.format(custom_tools_section=...)`` over the system prompt
    (rlm/utils/prompts.py:228); our canonical prompt contains literal ``{id, text, url}``
    braces -> KeyError. Escape everything; format() restores single braces and injects
    nothing (we already filled the section env-style)."""
    return s.replace("{", "{{").replace("}", "}}")


def _child_completions(completion: Any) -> List[Any]:
    """All direct child RLMChatCompletions of a node (object or dict form), in call order."""
    kids: List[Any] = []
    meta = _get(completion, "metadata") or {}
    for it in meta.get("iterations", []) or []:
        for _code, _stdout, rlm_calls in _iter_code_blocks(it):
            kids.extend(rlm_calls or [])
    return kids


def _iter_code_blocks(iteration: Dict[str, Any]):
    """Yield (code, stdout, rlm_calls) for each executed REPL block in an iteration dict.

    ``iteration`` is one entry of ``metadata['iterations']`` (RLMIteration.to_dict):
      {prompt, response, code_blocks:[{code, result:{stdout, stderr, rlm_calls, final_answer}}], ...}
    ``rlm_calls`` entries are RLMChatCompletion dicts (to_dict) — the sub-agent answers.
    """
    for cb in iteration.get("code_blocks", []) or []:
        result = cb.get("result", {}) or {}
        yield (
            cb.get("code", "") or "",
            result.get("stdout", "") or "",
            result.get("rlm_calls", []) or [],
        )


def trajectory_to_conversations(
    completion: Any,
    question: str,
    max_recursion_depth: int,  # ENV convention: ceiling on delegation (library max_depth - 1)
    depth: int = 0,
    child_return_mode: str = "structured",
) -> Optional[List[Dict[str, str]]]:
    """Serialize ONE node's trajectory into a sharegpt ``conversations`` list.

    ``depth=0`` serializes the root (orchestrator band); ``depth>=1`` serializes a child
    sub-agent (worker/coordinator band) from its nested RLMChatCompletion (object or dict).
    Child rows exist because root and children share ONE policy: SFT must supervise the
    worker role (focused cited mini-reports — the cites/kid behavior) too, not just the
    orchestrator role.

    Layout (matches the rlm REPL turn shape and the dr-tulu inline-tool convention):
      [ {system: depth-banded ORCHESTRATOR prompt},
        {user:   the research question},
        {assistant: <repl think/code>            # iter 0 response (with ```repl``` block)
                    <tool_output>stdout</tool_output>
                    <subagent_output>child answer</subagent_output>},
        {assistant: <repl think/code> ...},       # iter 1 ...
        ... ,
        {assistant: ... answer["content"]=report; answer["ready"]=True ...} ]

    Returns None if the trajectory has no usable iterations (so the caller can skip it).
    """
    meta = _get(completion, "metadata")
    if not meta or not meta.get("iterations"):
        logger.warning("[gen_recursive_sft] node completion has no logged trajectory; skipping")
        return None

    # System prompt: the node's depth band, env-filled — exactly what the RL env serves
    # that node (and what the teacher effectively saw post-escape/format at generation).
    system_prompt = node_system_prompt(depth, max_recursion_depth, child_return_mode)
    conversations: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Each iteration -> an assistant turn (the model's response) FOLLOWED BY a user turn (the
    # REPL observation: tool_output + subagent_output). The RL env returns each turn's
    # observation as a USER message, so mirroring that here keeps the trajectory's ROLE
    # structure identical to what RL trains AND keeps the sharegpt conversation strictly
    # alternating — LLaMA-Factory's SharegptDatasetConverter rejects consecutive assistant
    # turns. The observation user-turns are masked by LLaMA-Factory (only assistant turns are
    # trained), exactly as the env masks its observation tokens.
    for it in meta["iterations"]:
        if it.get("type") and it["type"] != "iteration":
            continue
        response = it.get("response", "") or ""
        # Thinking-ON serving puts the opening <think> tag in the GENERATION PROMPT
        # (Qwen template: '<|im_start|>assistant\n<think>\n'), so completions arrive as
        # 'reasoning</think>visible'. Restore the opening tag so training rows carry the
        # well-formed '<think>...</think>' pair the student's template expects.
        if "</think>" in response and not response.lstrip().startswith("<think>"):
            response = "<think>\n" + response.lstrip()
        if response.strip():
            conversations.append({"role": "assistant", "content": response.rstrip()})
        obs_parts: List[str] = []
        for _code, stdout, rlm_calls in _iter_code_blocks(it):
            tool_block = _stdout_to_tool_output(stdout)
            if tool_block:
                obs_parts.append(tool_block)
            sub_block = _rlm_calls_to_subagent_output(rlm_calls)
            if sub_block:
                obs_parts.append(sub_block)
        if obs_parts:
            conversations.append({"role": "user", "content": "\n\n".join(obs_parts)})

    # Enforce strict alternation: merge any consecutive same-role turns (an iteration that
    # produced no REPL output would otherwise leave two assistant turns adjacent — the very
    # bug this fixes), then drop a trailing observation so the SFT target ends on the
    # assistant's final answer.
    merged: List[Dict[str, str]] = []
    for m in conversations:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    conversations = merged
    while conversations and conversations[-1]["role"] == "user":
        conversations.pop()
    if not any(m["role"] == "assistant" for m in conversations):
        return None
    return conversations


# ---------------------------------------------------------------------------
# RLM construction (the inference arm)
# ---------------------------------------------------------------------------


def build_rlm(
    model: str,
    base_url: str,
    payload: Dict[str, Any],
    max_depth: int,
    max_concurrent_subcalls: int,
    max_iterations: int,
    api_key: str,
    temperature: float,
    child_return_mode: str = "structured",
) -> RLM:
    """Construct a recursive RLM that mirrors the RL arm:
      * backend='vllm' against the local served policy (model_name + base_url),
      * environment='local' with max_depth>=2 so depth>1 trees can form,
      * the offline corpus search()/get_doc() tools as custom_tools (provenance ids),
      * the depth-banded ORCHESTRATOR system prompt (root band),
      * an RLMLogger so the full trajectory lands on RLMChatCompletion.metadata.

    NOTE on the node_rid baked into snippet ids: at inference we mint a single root rid
    for the corpus tools. This keeps the ``"{rid}-{n}"`` id scheme valid for the citation
    contract; per-node provenance attribution matters for the RL *reward*, not for SFT
    (here we only need a real, well-cited recursive trajectory to imitate)."""
    # one shared surfaced-id sink + a stable root rid for this rollout's tools
    surfaced_ids: List[str] = []
    node_rid = "sftroot"
    tools = make_corpus_tools(payload, node_rid=node_rid, surfaced_ids=surfaced_ids)

    # ENV-PARITY PROMPTS. Library max_depth counts levels (2 = root + children); the env's
    # delegation ceiling is max_depth - 1. Root gets the orchestrator band (delegation
    # section filled env-style), children the worker band (empty section) via the
    # sub_system_prompt patch. Braces are escaped because the library .format()s the
    # prompt (would KeyError on the canonical prompt's literal {id, text, url}).
    env_ceiling = max_depth - 1
    root_prompt = _escape_braces(node_system_prompt(0, env_ceiling, child_return_mode))
    child_prompt = _escape_braces(node_system_prompt(1, env_ceiling, child_return_mode))

    return RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": model,
            "base_url": base_url,
            "api_key": api_key,
        },
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_concurrent_subcalls=max_concurrent_subcalls,
        custom_system_prompt=root_prompt,
        sub_system_prompt=child_prompt,
        # offline corpus tools available in every REPL (root + sub-agents inherit them)
        custom_tools=tools,
        custom_sub_tools=tools,
        logger=RLMLogger(),  # capture the trajectory into .metadata
        verbose=False,
        sampling_args={"temperature": temperature},
        sub_sampling_args={"temperature": temperature},
    )


# ---------------------------------------------------------------------------
# Rejection sampling over N rollouts per prompt
# ---------------------------------------------------------------------------


def best_rollout_for_prompt(
    question: str,
    rubrics: List[dict],
    judge_cfg: JudgeConfig,
    n_samples: int,
    rlm_factory,
) -> Optional[Dict[str, Any]]:
    """Sample N recursive rollouts, score each ROOT report R with the held-constant rubric
    judge (score_report_sync), and return the best by R (with its R and per-criterion).
    Returns None if every rollout errored out."""
    best: Optional[Dict[str, Any]] = None
    for i in range(n_samples):
        rlm = rlm_factory()  # fresh RLM (fresh logger + fresh tool counters) per sample
        try:
            completion: RLMChatCompletion = rlm.completion(question, root_prompt=question)
        except Exception as e:
            logger.warning(f"[gen_recursive_sft] rollout {i} failed: {e}")
            continue
        finally:
            rlm.close()

        report = completion.response or ""
        if not report.strip():
            continue
        try:
            R, per_criterion = score_report_sync(report, question, rubrics, judge_cfg)
        except Exception as e:
            logger.warning(f"[gen_recursive_sft] judge failed on rollout {i}: {e}; R=0")
            R, per_criterion = 0.0, {}

        cand = {
            "completion": completion,
            "R": float(R),
            "per_criterion": per_criterion,
            "sample_index": i,
        }
        if best is None or cand["R"] > best["R"]:
            best = cand
        logger.info(f"[gen_recursive_sft]   rollout {i}: R={R:.3f}")
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True, help="parquet/jsonl of RL rows (from convert_drtulu_rl)")
    p.add_argument("--out_jsonl", required=True, help="output sharegpt jsonl path")
    p.add_argument("--n_samples", type=int, default=4, help="rollouts to sample per prompt (rejection sampling)")
    p.add_argument("--max_depth", type=int, default=2, help="RLM recursion depth ceiling (>=2 for depth>1 trees)")
    p.add_argument("--threshold", type=float, default=0.5, help="keep a rollout iff its report R >= threshold")
    p.add_argument("--model", required=True, help="served policy model id (e.g. Qwen/Qwen3-8B or a DR-Tulu-8B ckpt)")
    p.add_argument("--base_url", required=True, help="OpenAI-compatible base URL of the local vLLM policy server")
    p.add_argument("--api_key", default="EMPTY", help="API key for the policy server (dummy for local vLLM)")
    p.add_argument("--temperature", type=float, default=0.7, help="sampling temperature for the rollouts")
    p.add_argument("--max_iterations", type=int, default=20, help="max REPL iterations per node")
    p.add_argument("--max_concurrent_subcalls", type=int, default=4, help="parallel child sub-agents per node")
    p.add_argument("--limit", type=int, default=None, help="cap number of prompts processed (debug)")
    p.add_argument("--no_children", action="store_true",
                   help="do NOT emit per-child worker rows (default: emit them — shared policy needs the worker role supervised too)")
    p.add_argument("--child_return_mode", default="structured", choices=["prose", "structured"],
                   help="citation contract variant (locked RL/eval config = structured)")
    # judge override knobs (default to the held-constant config so SFT == RL judge)
    p.add_argument("--judge_model", default=DrRlmGeneratorConfig.judge_model)
    p.add_argument("--judge_base_url", default=DrRlmGeneratorConfig.judge_base_url)
    p.add_argument("--judge_api_key_env", default=DrRlmGeneratorConfig.judge_api_key_env)
    p.add_argument("--judge_score_scale", type=float, default=DrRlmGeneratorConfig.judge_score_scale)
    p.add_argument("--judge_max_concurrency", type=int, default=DrRlmGeneratorConfig.judge_max_concurrency)
    # corpus search knobs (held constant with the RL arm; default mcp_http like the config)
    p.add_argument("--search_backend", default=DrRlmGeneratorConfig.search_backend)
    p.add_argument("--search_endpoint", default=DrRlmGeneratorConfig.search_endpoint)
    p.add_argument("--search_corpus_path", default=DrRlmGeneratorConfig.search_corpus_path)
    p.add_argument("--search_index_path", default=DrRlmGeneratorConfig.search_index_path)
    p.add_argument("--search_embed_model", default=DrRlmGeneratorConfig.search_embed_model)
    p.add_argument("--search_top_k", type=int, default=DrRlmGeneratorConfig.search_top_k)
    p.add_argument("--snippet_max_chars", type=int, default=DrRlmGeneratorConfig.snippet_max_chars)
    return p.parse_args()


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """The dr_rlm env-payload sub-dict the judge + corpus tools consume. Built from a
    DrRlmGeneratorConfig so the keys match ``DrRlmGeneratorConfig.env_payload()`` exactly
    (single source of truth) and SFT-time retrieval/judging == RL-time."""
    cfg = DrRlmGeneratorConfig(
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
    return cfg.env_payload()


def main() -> None:
    args = parse_args()
    out_path = os.path.expanduser(args.out_jsonl)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload = build_payload(args)
    judge_cfg = JudgeConfig.from_env_payload(payload)

    rows = load_prompts(os.path.expanduser(args.prompts))
    if args.limit is not None:
        rows = rows[: args.limit]
    logger.info(f"[gen_recursive_sft] loaded {len(rows)} prompts from {args.prompts}")

    kept, attempted = 0, 0
    with open(out_path, "w") as fout:
        for ri, row in enumerate(rows):
            question = question_of(row)
            rubrics = rubrics_of(row)
            attempted += 1
            logger.info(f"[gen_recursive_sft] prompt {ri + 1}/{len(rows)}: {question[:90]!r}")

            def _factory() -> RLM:
                return build_rlm(
                    model=args.model,
                    base_url=args.base_url,
                    payload=payload,
                    max_depth=args.max_depth,
                    max_concurrent_subcalls=args.max_concurrent_subcalls,
                    max_iterations=args.max_iterations,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    child_return_mode=args.child_return_mode,
                )

            best = best_rollout_for_prompt(
                question=question,
                rubrics=rubrics,
                judge_cfg=judge_cfg,
                n_samples=args.n_samples,
                rlm_factory=_factory,
            )
            if best is None:
                logger.warning(f"[gen_recursive_sft] prompt {ri}: no successful rollout; skipping")
                continue

            # Rejection sampling: keep only if the best report clears the threshold.
            if best["R"] < args.threshold:
                logger.info(
                    f"[gen_recursive_sft] prompt {ri}: best R={best['R']:.3f} < threshold "
                    f"{args.threshold}; rejected"
                )
                continue

            conversations = trajectory_to_conversations(
                best["completion"], question, max_recursion_depth=args.max_depth - 1,
                child_return_mode=args.child_return_mode,
            )
            if not conversations:
                logger.warning(f"[gen_recursive_sft] prompt {ri}: kept rollout had no usable trajectory; skipping")
                continue

            uid = str(row.get("uid", ri))
            fout.write(json.dumps({
                "conversations": conversations,
                "_node": "root", "_qid": uid, "_R": round(best["R"], 4),
                "_response": str(best["completion"].response or "")[:4000],
            }, ensure_ascii=False) + "\n")
            kept += 1

            # Per-child worker rows from the SAME kept tree (same R gate as the root: the
            # tree passed rejection sampling as a unit). Empty/unlogged children are skipped.
            n_kids = 0
            if not args.no_children:
                for ci, call in enumerate(_child_completions(best["completion"])):
                    resp = _get(call, "response", "") or ""
                    if not str(resp).strip():
                        continue
                    child_q = _get(call, "prompt", "") or ""
                    child_conv = trajectory_to_conversations(
                        call, str(child_q), max_recursion_depth=args.max_depth - 1, depth=1,
                        child_return_mode=args.child_return_mode,
                    )
                    if not child_conv:
                        continue
                    fout.write(json.dumps({
                        "conversations": child_conv,
                        "_node": "child", "_qid": f"{uid}-c{ci}", "_R": round(best["R"], 4),
                    }, ensure_ascii=False) + "\n")
                    n_kids += 1
            fout.flush()
            logger.info(
                f"[gen_recursive_sft] prompt {ri}: KEPT (R={best['R']:.3f}, "
                f"sample={best['sample_index']}, +{n_kids} child rows)"
            )

    logger.info(f"[gen_recursive_sft] done: kept {kept}/{attempted} prompts -> {out_path}")


if __name__ == "__main__":
    main()
