"""Held-constant rubric / citation judge for DR-RLM.

This is a self-contained, dependency-light re-implementation of DR Tulu's rubric
scoring so the *same* reward logic runs on every arm of the controlled comparison
without importing the whole ``open_instruct`` stack into the SkyRL runtime. The
per-criterion prompt, the 0-2 -> [0,1] normalization, and the ``<cite id=...>``
claim-extraction regex are copied verbatim from:

  * dr-tulu/.../search_rewards/utils/rubric_utils.py::_score_property[_async]
  * dr-tulu/.../search_rewards/utils/citation_utils.py::extract_claims_and_corresponding_citation_ids
  * dr-tulu/.../search_rewards/longform_rubric_rewards.py::_compute_rubric_scores_and_reward

Transport is a plain async OpenAI-compatible ``/chat/completions`` call (httpx) so it
points at a local Qwen served by vLLM (zero external API). To use OpenAI/Azure instead,
point ``base_url``/``model`` at them. A failed/parse-error judge call returns 0.0
exactly like the DR Tulu original (so a judge outage looks like a bad answer, not a
crash) — callers should monitor judge health.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from loguru import logger

# --- per-criterion judge prompt (verbatim from rubric_utils._score_property_async) ---

_CRITERION_SYSTEM_PROMPT = (
    "You will be given a question someone asked (in <question></question> tags) and the "
    "corresponding response (in <response></response> tags) given to them by an assistant.  "
    "You will then be given a specific criterion of the response to evaluate (in "
    "<criterion></criterion> tags).\n"
    "Return a score on a scale of 0 to 2 indicating how appropriate the response is based on "
    "the given criterion. Judge only the specified aspect(s), not any other qualities of the "
    'answer.  Output JSON in the format: {"score": x}.'
)

# DR Tulu's --general_rubric, used as a node-local quality proxy for the RAO (L2) arm
# where no per-node ground-truth rubric exists.
_GENERAL_RUBRIC = (
    "(1) Overall Comprehensiveness: covers the question as comprehensively as possible; "
    "(2) Thoroughness: each point is discussed substantively, not superficially; "
    "(3) Factuality: minimal factual errors; "
    "(4) Coherence: stays focused and relevant to the question."
)

# verbatim from citation_utils.extract_claims_and_corresponding_citation_ids
_CITE_PATTERN = re.compile(r"<cite id=([\"\']?)([^\"\'>\s]+)\1[^>]*>([^<]+)</cite>")
_CITE_TAG_SPLIT = re.compile(r"<cite id=[\"\']?[^\"\'>\s]+[\"\']?[^>]*>[^<]+</cite>")


def extract_claims_and_corresponding_citation_ids(response: str) -> Dict[str, List[str]]:
    """{claim_text: [citation_id, ...]}; uncited fragments map to []. Multi-id cites
    (``id="a,b"``) are comma-split. Verbatim port of the DR Tulu function."""
    claims: Dict[str, List[str]] = {}
    cite_matches = _CITE_PATTERN.findall(response)
    for part in _CITE_TAG_SPLIT.split(response):
        part = part.strip()
        if part:
            claims[part] = []
    for _, citation_ids, cited_text in cite_matches:
        cited_text = cited_text.strip()
        if cited_text:
            claims[cited_text] = [cid for cid in citation_ids.split(",") if cid]
    return claims


def all_cited_ids(response: str) -> List[str]:
    """Flat list of every citation id used in ``response`` (with repeats)."""
    ids: List[str] = []
    for _, citation_ids, _txt in _CITE_PATTERN.findall(response):
        ids.extend(cid for cid in citation_ids.split(",") if cid)
    return ids


@dataclass
class JudgeConfig:
    model: str = "Qwen/Qwen3-8B"
    base_url: str = "http://localhost:8100/v1"
    api_key: str = "EMPTY"
    score_scale: float = 2.0
    max_concurrency: int = 16
    timeout: float = 120.0
    max_retries: int = 4

    @classmethod
    def from_env_payload(cls, p: dict) -> "JudgeConfig":
        import os

        model = p.get("judge_model", "Qwen/Qwen3-8B")
        # strip a litellm-style provider prefix for the raw OpenAI-compatible endpoint
        for prefix in ("hosted_vllm/", "openai/", "vllm/"):
            if model.startswith(prefix):
                model = model[len(prefix):]
        return cls(
            model=model,
            base_url=p.get("judge_base_url", "http://localhost:8100/v1"),
            api_key=os.environ.get(p.get("judge_api_key_env", "JUDGE_API_KEY"), "EMPTY"),
            score_scale=float(p.get("judge_score_scale", 2.0)),
            max_concurrency=int(p.get("judge_max_concurrency", 16)),
        )


class RubricJudge:
    """Async judge over an OpenAI-compatible endpoint. One per reward pass."""

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.max_concurrency)

    async def _chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        async with self._sem:
            async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
                for attempt in range(self.cfg.max_retries):
                    try:
                        r = await client.post(url, json=body, headers=headers)
                        r.raise_for_status()
                        return r.json()["choices"][0]["message"]["content"]
                    except Exception as e:  # transport / 5xx / parse — retry a few times
                        if attempt == self.cfg.max_retries - 1:
                            logger.warning(f"[dr_rlm.judge] giving up after {attempt + 1} tries: {e}")
                            return ""
                        await asyncio.sleep(2 ** attempt)
        return ""

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        if not text:
            return None
        # tolerate ```json fences and trailing prose
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return float(obj["score"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    async def score_criterion(self, response: str, question: str, criterion: str) -> float:
        """s_c in [0,1] for one rubric criterion. Mirrors DR Tulu's normalization
        (clamp(score / score_scale, 0, 1)); returns 0.0 on any failure."""
        user = f"<question>{question}</question>\n<response>{response}</response>\n<criterion>{criterion}</criterion>"
        score = self._parse_score(await self._chat(_CRITERION_SYSTEM_PROMPT, user))
        if score is None:
            return 0.0
        return max(0.0, min(1.0, score / self.cfg.score_scale))

    async def score_quality(self, response: str, question: str) -> float:
        """Node-local quality proxy (general rubric) for the RAO/L2 arm, in [0,1]."""
        return await self.score_criterion(response, question, _GENERAL_RUBRIC)

    async def supports(self, criterion: str, question: str, evidence_text: str) -> float:
        """Soft support strength in [0,1]: does ``evidence_text`` support ``criterion``?
        Used by share_mode='support' to attribute a criterion across nodes."""
        if not evidence_text.strip():
            return 0.0
        user = (
            f"<question>{question}</question>\n"
            f"<criterion>{criterion}</criterion>\n"
            f"<evidence>{evidence_text}</evidence>\n"
            "Does the evidence support this criterion of a good answer? "
            'Output JSON {"score": x} with x in 0 (no), 1 (partial), 2 (fully).'
        )
        score = self._parse_score(await self._chat(_CRITERION_SYSTEM_PROMPT, user))
        if score is None:
            return 0.0
        return max(0.0, min(1.0, score / self.cfg.score_scale))


def weighted_report_reward(scores: List[float], weights: List[float]) -> float:
    """DR Tulu's weighted sum: Σ s·w / max(Σ_{w>0} w, 1); negative-weight criteria
    subtract from the numerator but are excluded from the denominator (R ≤ 1, may be
    < 0). Shared by the sync (L1, in-env) and async (L2-L4, in-generator) paths so the
    report reward R is byte-identical across arms. Verbatim from DR Tulu
    longform_rubric_rewards._compute_rubric_scores_and_reward."""
    num = sum(s * w for s, w in zip(scores, weights))
    den = sum(w for w in weights if w > 0)
    return num / max(den, 1.0)


def score_report_sync(report: str, question: str, rubrics: List[dict], cfg: JudgeConfig) -> "tuple[float, Dict[str, float]]":
    """Synchronous report rubric score R (for the L1 in-env reward path, which runs in
    a SkyRL executor thread). Same per-criterion prompt + normalization as the async
    judge; criteria scored concurrently over a thread pool. Returns (R, per_criterion)."""
    from concurrent.futures import ThreadPoolExecutor

    rubrics = rubrics or []
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}

    def _one(criterion: str) -> float:
        body = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": _CRITERION_SYSTEM_PROMPT},
                {"role": "user", "content": f"<question>{question}</question>\n<response>{report}</response>\n<criterion>{criterion}</criterion>"},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(cfg.max_retries):
            try:
                with httpx.Client(timeout=cfg.timeout) as client:
                    r = client.post(url, json=body, headers=headers)
                    r.raise_for_status()
                    s = RubricJudge._parse_score(r.json()["choices"][0]["message"]["content"])
                    return 0.0 if s is None else max(0.0, min(1.0, s / cfg.score_scale))
            except Exception as e:
                if attempt == cfg.max_retries - 1:
                    logger.warning(f"[dr_rlm.judge.sync] giving up: {e}")
                    return 0.0
                import time

                time.sleep(2 ** attempt)
        return 0.0

    if not rubrics:
        descs = [_GENERAL_RUBRIC]
        weights = [1.0]
    else:
        descs = [r["description"] for r in rubrics]
        weights = [float(r.get("weight", 1.0)) for r in rubrics]

    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as ex:
        scores = list(ex.map(_one, descs))
    R = weighted_report_reward(scores, weights)
    per_criterion = {
        (rubrics[i].get("title", f"c{i}") if rubrics else "general"): float(scores[i]) for i in range(len(scores))
    }
    return R, per_criterion


# ===================== DR Tulu citation reward (grounding term) =====================
# Faithful port of dr-tulu/.../search_rewards/utils/citation_utils.py so the citation_reward
# is byte-identical to the DR Tulu baseline (same prompts, same recall/precision F1, same
# 0.6*avg_f1 + 0.4*valid_id_format blend). Unlike the rubric judge (JSON {"score": x}), the
# citation judges return free text ("Rating: [[...]] Analysis: ..."), so these calls do NOT
# use json_mode and parse with the verbatim regexes. Transport mirrors score_report_sync.
#
# Adaptation for our recursive setting: the `citations` dict {id: snippet_text} is assembled
# from the TREE-WIDE evidence ledger (ledger["snippets"]), not a single agent's <context>.
# The reward is gated ON by the caller (DR_RLM_CITATION_REWARD); the default reward path stays
# rubric-only so existing arms are unaffected (isolation discipline).

_CIT_RECALL_HAS_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical.
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""

_CIT_RECALL_NO_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation.
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".

<question>
{question}
</question>

<response>
{full_response}
</response>

<statement>
{statement}
</statement>"""

_CIT_PRECISION_PROMPT = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almost unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating.
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...".

<question>
{question}
</question>

<statement>
{statement}
</statement>

<snippet>
{concatenated_cited_snippets}
</snippet>"""


def _rate_recall(text: str) -> float:  # Fully/Partially/No support -> 1/0.5/0
    m = re.search(r"Rating: \[\[(.*)\]\]", text or "")
    if not m:
        return 0.0
    t = m.group(1).strip().lower()
    return {"fully supported": 1.0, "partially supported": 0.5, "no support": 0.0}.get(t, 0.0)


def _rate_need_citation(text: str) -> int:  # Need Citation: Yes/No -> 1/0
    m = re.search(r"Need Citation: \[\[(.*)\]\]", text or "")
    if not m:
        return 0
    return 1 if m.group(1).strip().lower() == "yes" else 0


def _rate_relevant(text: str) -> int:  # Relevant/Unrelevant -> 1/0
    m = re.search(r"Rating: \[\[(.*)\]\]", text or "")
    if not m:
        return 0
    return 1 if m.group(1).strip().lower() == "relevant" else 0


def _cite_chat_sync(prompt: str, cfg: JudgeConfig) -> str:
    """Free-text (NON-json) judge call for the citation prompts; returns "" on failure
    (so a dead judge degrades the claim to recall/precision 0, never crashes a rollout)."""
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
    }
    for attempt in range(cfg.max_retries):
        try:
            with httpx.Client(timeout=cfg.timeout) as client:
                r = client.post(url, json=body, headers=headers)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == cfg.max_retries - 1:
                logger.warning(f"[dr_rlm.judge.cite] giving up: {e}")
                return ""
            import time

            time.sleep(2 ** attempt)
    return ""


def score_citation_format(claims: Dict[str, List[str]], citations: Dict[str, str]) -> float:
    """Hallucinated-id check: fraction of cited ids that exist in the ledger. Verbatim port."""
    all_ids = {cid for ids in claims.values() for cid in ids}
    if not all_ids:
        return 0.0
    valid = [cid for cid in all_ids if cid in citations]
    return len(valid) / len(all_ids)


def score_in_context_citations_sync(
    question: str, report: str, citations: Dict[str, str], cfg: JudgeConfig,
    precision_by_id: Optional[Dict[str, Optional[float]]] = None,
    components_out: Optional[Dict[str, float]] = None,
) -> float:
    """citation_reward in [0,1] = 0.6*avg_claim_F1 + 0.4*valid_id_format. Faithful sync port of
    citation_utils.score_in_context_citations_async (recall x precision per claim). `citations`
    is {id: snippet_text} from the tree-wide ledger. Returns 0.0 if no citations were used.

    CHEAP mode: when ``precision_by_id`` ({id: 1.0/0.0/None}) is supplied — our LOCAL
    check_citations_verdicts reused — a cited claim's PRECISION is averaged from those verdicts
    and NO Gemini precision call is made (recall still hits Gemini, the new orphan signal). Falls
    back to a Gemini precision call for any cited claim with no usable local verdict."""
    from concurrent.futures import ThreadPoolExecutor

    if not citations:
        return 0.0
    claims = extract_claims_and_corresponding_citation_ids(report)
    if not claims:
        return 0.0
    fmt = score_citation_format(claims, citations)

    def _precision(claim_text, cite_ids, concat) -> float:
        if precision_by_id is not None:
            vals = [precision_by_id[c] for c in cite_ids if precision_by_id.get(c) is not None]
            if vals:
                return sum(vals) / len(vals)  # reuse local verdicts -> skip Gemini precision call
        return float(_rate_relevant(_cite_chat_sync(
            _CIT_PRECISION_PROMPT.format(question=question, statement=claim_text, concatenated_cited_snippets=concat), cfg)))

    def _f1(item):
        claim_text, cite_ids = item
        concat = "\n\n".join(citations[c] for c in cite_ids if c in citations)
        if not concat:  # uncited fragment: recall = 1 - (needs citation?), precision = 1
            need = _rate_need_citation(_cite_chat_sync(
                _CIT_RECALL_NO_PROMPT.format(question=question, full_response=report, statement=claim_text), cfg))
            recall, precision = float(1 - need), 1.0
        else:            # cited fragment: graded support recall (Gemini) x relevance precision
            recall = _rate_recall(_cite_chat_sync(
                _CIT_RECALL_HAS_PROMPT.format(question=question, statement=claim_text, concatenated_cited_snippets=concat), cfg))
            precision = _precision(claim_text, cite_ids, concat)
        f1 = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)
        return (f1, recall, precision)

    with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as ex:
        _triples = list(ex.map(_f1, claims.items()))
    f1s = [t[0] for t in _triples]
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    if components_out is not None:  # observability only: expose the recall/precision/F1 split for telemetry
        _rs = [t[1] for t in _triples]
        _ps = [t[2] for t in _triples]
        components_out["avg_f1"] = avg_f1
        components_out["valid_id_format"] = fmt
        components_out["avg_recall"] = (sum(_rs) / len(_rs)) if _rs else 0.0
        components_out["avg_precision"] = (sum(_ps) / len(_ps)) if _ps else 0.0
    return 0.6 * avg_f1 + 0.4 * fmt


# DR Tulu's published composite weights (longform_rubric_rewards.REWARD_WEIGHTS). We expose
# them so an arm can be trained exactly like the baseline; the held-constant *outcome* core is
# {rubric, citation} (the architecture-agnostic terms — see docs/RL_TRAINING_LOG.md).
DRTULU_REWARD_WEIGHTS = {"rubric": 0.5, "citation": 0.2, "format": 0.2, "num_search_turns": 0.1}
OUTCOME_REWARD_WEIGHTS = {"rubric": 0.5, "citation": 0.2}  # renormalized at use site


def composite_report_reward(report, question, rubrics, citations, cfg, weights=None, precision_by_id=None):
    """Blend rubric + citation into one report reward R, renormalized by the provided weights so
    R stays in [0,1]. `weights` keys are a subset of {rubric, citation}; process terms
    (format, num_search_turns) are computed by the env/generator where the trace is available and
    can be folded in by the caller. `precision_by_id` (optional) reuses local SUPPORTED verdicts
    for citation precision (cheap path). Returns (R, components_dict)."""
    import os

    weights = dict(weights or OUTCOME_REWARD_WEIGHTS)
    rubric_R, per_criterion = score_report_sync(report, question, rubrics, cfg)
    comp = {"rubric": rubric_R}
    if weights.get("citation", 0.0) > 0.0:
        comp["citation"] = score_in_context_citations_sync(question, report, citations or {}, cfg, precision_by_id)
    if weights.get("format", 0.0) > 0.0:  # Run B: harness-native format (no LLM call)
        comp["format"] = score_format_reward(report)
    if weights.get("search", 0.0) > 0.0:  # Run B: tree-wide retrieval volume (no LLM call)
        comp["search"] = tree_search_reward(len(citations or {}), float(os.environ.get("DR_RLM_SEARCH_CAP", "8") or 8))
    wsum = sum(weights[k] for k in weights if k in comp) or 1.0
    R = sum(weights[k] * comp[k] for k in weights if k in comp) / wsum
    comp["per_criterion"] = per_criterion
    return R, comp


def citation_reward_config() -> "tuple[bool, dict]":
    """Env gate + weights for the grounding-aware composite reward (isolation discipline:
    default OFF = original rubric-only behavior). ``DR_RLM_CITATION_REWARD=1`` turns it on;
    weights via ``DR_RLM_{RUBRIC,CITATION,FORMAT,SEARCH}_WEIGHT`` (default DR Tulu 0.5/0.2, and
    0/0 for the process terms so Run A = rubric+citation only), renormalized at the use site so
    R stays in [0,1]. FORMAT/SEARCH (Run B) are DR Tulu's process terms re-expressed for our
    recursive harness — see score_format_reward / tree_search_reward."""
    import os

    on = str(os.environ.get("DR_RLM_CITATION_REWARD", "0")).lower() not in ("0", "", "false", "no")
    weights = {"rubric": float(os.environ.get("DR_RLM_RUBRIC_WEIGHT", "0.5"))}
    if on:
        weights["citation"] = float(os.environ.get("DR_RLM_CITATION_WEIGHT", "0.2"))
        fw = float(os.environ.get("DR_RLM_FORMAT_WEIGHT", "0") or 0)
        sw = float(os.environ.get("DR_RLM_SEARCH_WEIGHT", "0") or 0)
        if fw > 0:
            weights["format"] = fw
        if sw > 0:
            weights["search"] = sw
    return on, weights


def score_format_reward(report: str) -> float:
    """DR Tulu's format_reward RE-EXPRESSED for DR-RLM. DR Tulu checks for <answer>/<cite>/<search>
    tags; our harness emits plain prose + inline <cite id> (no <answer> wrapper) and searches via
    REPL search() (no <search> tags), so a verbatim port is near-constant/zero. Harness-native
    analog: 0.5*(substantive report) + 0.5*(>=1 citation). The root-<search>-query sub-term is
    dropped — it would penalize the RLM for delegating (see docs/RL_TRAINING_LOG.md)."""
    r = (report or "").strip()
    has_report = 1.0 if len(r) >= 200 else 0.0
    has_cite = 1.0 if _CITE_PATTERN.search(r) else 0.0
    return 0.5 * has_report + 0.5 * has_cite


def tree_search_reward(n_snippets: int, cap: float = 8.0) -> float:
    """DR Tulu's num_search_turns RE-EXPRESSED tree-wide. DR Tulu counts the single agent's own
    <search> turns (min(n/3,1)); for a delegating RLM the root may search little, so we count
    retrieval VOLUME across the WHOLE tree (ledger snippet count) — children's retrievals included,
    so delegation is not penalized. min(n_snippets/cap, 1)."""
    return min(float(n_snippets) / max(cap, 1.0), 1.0)


def ledger_citations(ledger: Optional[dict]) -> Dict[str, str]:
    """Assemble the {citation_id: snippet_text} dict the citation reward needs from our
    TREE-WIDE evidence ledger (``ledger["snippets"] = {sid: {text, node_rid, ...}}``)."""
    snips = (ledger or {}).get("snippets") if isinstance(ledger, dict) else None
    if not snips:
        return {}
    return {sid: str((rec or {}).get("text", "") or "") for sid, rec in snips.items()}
