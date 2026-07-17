"""Recursive Evolving Rubrics (RER) reward pipeline — the DR-RLM contribution core.

Given an assembled recursion tree (the root report plus every sub-agent's answer)
and the held-constant rubric set, this computes a per-node terminal reward ``r_a``
under one of four ladder rungs (proposal §6 / RQ2):

  inherited  (L1) : only the root is scored (report rubric R); children get 0.
  rao        (L2) : r_a = s̃(own answer) + λ·mean(child s̃)   [RAO local-node reward]
  rer        (L3) : r_a = Σ_{c : a supports c} w_c·s_c·share(a,c) − γ·cost(a)
                    — provenance-attributed credit via the citation graph (the core)
  rer_structural (L4) : rer + a structural penalty channel on the root (orphan /
                    redundant / over-fragmented decomposition; RQ3).

Provenance graph (the cheap structural difference-reward estimator, proposal §3):
the corpus ``search()`` tool minted every surfaced snippet id as ``"{node_rid}-{n}"``,
so a ``<cite id=...>`` in the final report decodes to the node that surfaced it. We
therefore read provenance straight off the report's citations — no counterfactual
rollouts. ``share(a,c)`` splits a criterion's mass w_c·s_c across the nodes whose cited
evidence supports it (equal / citation-count / judged-support). Because the shares sum
to 1 per criterion, Σ_a (evidence credit) = R: credit is *conserved* and merely
redistributed by who-earned-it.

This module is async (the judge is async) and is driven from ``DrRlmGenerator.generate``.
It is deliberately framework-free (plain dataclasses in / dict out) so it is unit-testable
without SkyRL or a GPU.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .judge import (
    RubricJudge,
    JudgeConfig,
    extract_claims_and_corresponding_citation_ids,
    weighted_report_reward,
    citation_reward_config,
    ledger_citations,
    score_in_context_citations_sync,
    score_format_reward,
    tree_search_reward,
)
from .corpus_search import owner_rid_of

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class RerNode:
    """One recursion node, as reconstructed from the flattened step-wise tree."""

    rid: str
    depth: int
    parent_rid: Optional[str]
    final_answer: str = ""
    cost: float = 0.0  # normalized subtree cost in [0,1] (for the −γ·cost term)
    cited_ids: List[str] = field(default_factory=list)  # ids this node cited in its OWN answer
    citation_precision: Optional[dict] = None  # {id:1/0/None} local SUPPORTED verdicts (root; cheap citation precision)
    # filled in by the pipeline:
    reward: float = 0.0


@dataclass
class RerResult:
    rewards: Dict[str, float] = field(default_factory=dict)  # rid -> r_a
    report_reward: float = 0.0  # R (grounding-blended when DR_RLM_CITATION_REWARD is on)
    citation_reward: Optional[float] = None  # the citation_reward component, when blended
    cit_recall: Optional[float] = None     # observability: avg per-claim citation recall
    cit_precision: Optional[float] = None  # observability: avg per-claim citation precision
    cit_f1: Optional[float] = None         # observability: avg per-claim citation F1
    per_criterion: Dict[str, float] = field(default_factory=dict)  # title -> s_c
    metrics: Dict[str, float] = field(default_factory=dict)


def _root(nodes: List[RerNode]) -> RerNode:
    for n in nodes:
        if n.depth == 0 or n.parent_rid is None:
            return n
    return nodes[0]


def _children(nodes: List[RerNode], rid: str) -> List[RerNode]:
    return [n for n in nodes if n.parent_rid == rid]


def _tokset(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


async def compute_rer_rewards(
    nodes: List[RerNode],
    rubrics: List[dict],
    question: str,
    payload: dict,
    ledger: Optional[dict] = None,
) -> RerResult:
    """Compute per-node ``r_a`` for one rollout tree. ``rubrics`` is the held-constant
    rubric list ``[{description, title, weight}]``; ``payload`` is the dr_rlm config
    sub-dict (judge + reward_mode + share_mode + gamma_cost + structural weights).

    ``ledger`` (optional) is the per-tree evidence ledger
    (``{"snippets": {sid: {node_rid, text, ...}}}``). When ``share_mode="ledger_support"``
    credit is judged over each node's OWN cited evidence pulled from the ledger — which is
    robust to the orchestrator dropping ``<cite>`` tags on synthesis (the A2 failure)."""
    mode = payload.get("reward_mode", "rer")
    judge = RubricJudge(JudgeConfig.from_env_payload(payload))
    root = _root(nodes)
    res = RerResult()

    # ---- 1. report rubric score R = Σ s_c w_c / Σ_{w>0} w (always computed) ----
    rubrics = rubrics or []
    weights = [float(r.get("weight", 1.0)) for r in rubrics]
    if rubrics:
        s_list = await asyncio.gather(
            *[judge.score_criterion(root.final_answer, question, r["description"]) for r in rubrics]
        )
    else:
        # no rubrics -> fall back to a single general-quality score
        s_list = [await judge.score_quality(root.final_answer, question)]
        weights = [1.0]
    R = weighted_report_reward(list(s_list), weights)
    # grounding-aware composite (opt-in via DR_RLM_CITATION_REWARD): blend DR Tulu's
    # citation_reward (+ optional re-expressed format/search process terms for Run B) into the
    # ROOT R that every mode below uses — and that provenance credit distributes — so children
    # whose cited evidence grounds the report share in the grounding, not just coverage. Off by
    # default => byte-identical to the rubric-only behavior.
    _cit_on, _w = citation_reward_config()
    cit_R = None
    _proc = {}
    if _cit_on:
        _cits = ledger_citations(ledger)
        comp = {"rubric": R}
        if _w.get("citation", 0.0) > 0.0:
            _cit_comp: Dict[str, float] = {}
            cit_R = await asyncio.to_thread(
                score_in_context_citations_sync, question, root.final_answer, _cits,
                JudgeConfig.from_env_payload(payload), root.citation_precision, _cit_comp,
            )
            comp["citation"] = cit_R
            res.cit_recall = _cit_comp.get("avg_recall")
            res.cit_precision = _cit_comp.get("avg_precision")
            res.cit_f1 = _cit_comp.get("avg_f1")
        if _w.get("format", 0.0) > 0.0:  # Run B: harness-native format (no LLM call)
            comp["format"] = score_format_reward(root.final_answer)
        if _w.get("search", 0.0) > 0.0:  # Run B: tree-wide retrieval volume (no LLM call)
            comp["search"] = tree_search_reward(len(_cits), float(os.environ.get("DR_RLM_SEARCH_CAP", "8") or 8))
        _den = sum(_w[k] for k in _w if k in comp) or 1.0
        R = sum(_w[k] * comp[k] for k in _w if k in comp) / _den
        _proc = {k: comp[k] for k in ("format", "search") if k in comp}
    res.citation_reward = cit_R
    res.report_reward = R
    res.per_criterion = {rubrics[i].get("title", f"c{i}"): float(s_list[i]) for i in range(len(rubrics))}

    # ---- 2. inherited (L1) ----
    if mode == "inherited":
        for n in nodes:
            n.reward = R if n is root else 0.0
        res.rewards = {n.rid: n.reward for n in nodes}
        res.metrics = {"report_reward": R, "n_nodes": float(len(nodes))}
        if cit_R is not None:
            res.metrics["citation_reward"] = float(cit_R)
        return res

    # ---- 3. RAO local-node reward (L2) ----
    if mode == "rao":
        lam = float(payload.get("rao_lambda", 0.5))
        # node-local quality proxy s̃ (general rubric on the node's own answer);
        # root uses the real rubric score R as its s̃.
        quality: Dict[str, float] = {}
        non_root = [n for n in nodes if n is not root]
        q_scores = await asyncio.gather(*[judge.score_quality(n.final_answer, question) for n in non_root])
        for n, q in zip(non_root, q_scores):
            quality[n.rid] = float(q)
        quality[root.rid] = R
        for n in nodes:
            kids = _children(nodes, n.rid)
            child_succ = sum(quality.get(c.rid, 0.0) for c in kids) / len(kids) if kids else 0.0
            n.reward = quality[n.rid] + lam * child_succ - float(payload.get("gamma_cost", 0.0)) * n.cost
        res.rewards = {n.rid: n.reward for n in nodes}
        res.metrics = {"report_reward": R, "rao_mean_quality": sum(quality.values()) / max(len(quality), 1)}
        return res

    # ---- 4 & 5. RER provenance-attributed credit (L3 / L4) ----
    gamma = float(payload.get("gamma_cost", 0.0))
    share_mode = payload.get("share_mode", "citation_count")
    rid_set = {n.rid for n in nodes}
    # The judged modes accumulate per-criterion mass w_c·s_c·share — divide by the SAME
    # denominator R uses (Σ_{w>0} w) so credit lands on R's scale and Σ_a credit = R when
    # every satisfied criterion has a contributor. Without this, weights like 3.0 inflate
    # credit ~Σw-fold above R (observed live: r_a=9.0 vs R=0.75 — smoke job 23668426).
    w_den = max(sum(w for w in weights if w > 0), 1.0)

    # report citations -> owner node (decode the snippet-id prefix)
    claims = extract_claims_and_corresponding_citation_ids(root.final_answer)
    node_cite_count: Dict[str, int] = defaultdict(int)
    node_claim_text: Dict[str, List[str]] = defaultdict(list)
    total_cites = 0
    for claim_text, ids in claims.items():
        for cid in ids:
            owner = owner_rid_of(cid)
            if owner in rid_set:
                node_cite_count[owner] += 1
                node_claim_text[owner].append(claim_text)
                total_cites += 1

    evidence_credit: Dict[str, float] = {n.rid: 0.0 for n in nodes}

    if share_mode == "ledger_support" and ledger is not None:
        # Robust path: attribute each criterion across nodes by judged support over each
        # node's OWN retrieved-and-cited evidence (from the harness ledger), NOT the root
        # report's <cite> tags. A worker that surfaced a key snippet gets credit even if the
        # orchestrator paraphrased it without a citation — the A2 FACT≈0 failure no longer
        # zeroes the credit signal. node_cite_count is recomputed to own-cites so the
        # orphan/structural metrics below reflect "node cited no usable evidence".
        snip = ledger.get("snippets", {})
        node_evidence_text: Dict[str, str] = {}
        node_cite_count = defaultdict(int)
        for n in nodes:
            own = [c for c in (n.cited_ids or []) if owner_rid_of(c) == n.rid and c in snip]
            node_cite_count[n.rid] = len(own)
            node_evidence_text[n.rid] = "\n\n".join(snip[c]["text"] for c in own)
        contributors = [n.rid for n in nodes if node_evidence_text[n.rid].strip()]
        for i, r in enumerate(rubrics):
            s_c, w_c = float(s_list[i]), weights[i]
            if w_c <= 0 or s_c <= 0 or not contributors:
                continue
            supports = await asyncio.gather(
                *[judge.supports(r["description"], question, node_evidence_text[rid]) for rid in contributors]
            )
            tot = sum(supports)
            if tot <= 0:
                continue
            for rid, sup in zip(contributors, supports):
                evidence_credit[rid] += (w_c * s_c / w_den) * (sup / tot)
    elif share_mode in ("equal", "citation_count"):
        # criterion-independent share -> r_a^evidence = R · share(a). Conserves Σ = R.
        contributors = [rid for rid in rid_set if node_cite_count[rid] > 0]
        for n in nodes:
            if n.rid not in contributors:
                share = 0.0
            elif share_mode == "equal":
                share = 1.0 / len(contributors)
            else:  # citation_count
                share = node_cite_count[n.rid] / max(total_cites, 1)
            evidence_credit[n.rid] = R * share
    else:  # "support": per-criterion judged attribution
        contributors = [rid for rid in rid_set if node_cite_count[rid] > 0]
        for i, r in enumerate(rubrics):
            s_c, w_c = float(s_list[i]), weights[i]
            if w_c <= 0 or s_c <= 0 or not contributors:
                continue
            supports = await asyncio.gather(
                *[judge.supports(r["description"], question, " ".join(node_claim_text[rid])) for rid in contributors]
            )
            tot = sum(supports)
            if tot <= 0:
                continue
            for rid, sup in zip(contributors, supports):
                evidence_credit[rid] += (w_c * s_c / w_den) * (sup / tot)

    for n in nodes:
        n.reward = evidence_credit[n.rid] - gamma * n.cost

    # HYBRID root reward (root_reward_mode="full_R"): the ROOT earns R itself — direct,
    # undiluted pressure on synthesis quality (what L1/broadcast gets right) — while
    # children keep their provenance credit shares (what conserved L3 gets right).
    # Motivation (2026-06-12 smoke): under pure conserved credit the root's small share
    # diluted the quality gradient — judged report quality drifted DOWN while citing rose;
    # the L1 contrast showed the converse. NB Σ_a r_a ≠ R under this mode, by design.
    if payload.get("root_reward_mode", "credit_share") == "full_R":
        root.reward = R - gamma * root.cost

    # --- child-citing shaping (opt-in: DR_RLM_CHILD_CITE_SHAPING) ------------------
    # Cold-start bootstrap. Provenance credit only rewards a child whose cited evidence
    # ALSO supports a criterion (sparse: ~6% of children at init — probe 24193291). Under a
    # trained root (full_R) the dense root gradient drowns that sparse child signal, so
    # citing never takes off (orphans flat ~94%). This adds a DENSE per-child reward for
    # emitting valid own-evidence citations AT ALL — independent of criterion support — so
    # every citing child gets a gradient that competes with the root's. Bounded, and meant
    # to be ANNEALED toward 0 over training (the launcher sets the schedule); the rubric +
    # provenance credit stay the steady-state objective. The root is never shaped.
    # Default 0.0 => byte-identical to prior behavior (existing tests unaffected).
    _shape = float(os.environ.get("DR_RLM_CHILD_CITE_SHAPING", "0") or 0.0)
    _shape_sum = 0.0
    _n_children = 0
    if _shape > 0.0:
        _cap = max(float(os.environ.get("DR_RLM_CHILD_CITE_CAP", "3") or 3.0), 1.0)
        for n in nodes:
            if n is root:
                continue
            _n_children += 1
            _bonus = _shape * min(float(node_cite_count.get(n.rid, 0)), _cap) / _cap
            n.reward += _bonus
            _shape_sum += _bonus

    metrics = {
        "report_reward": R,
        "n_nodes": float(len(nodes)),
        "n_orphan_children": float(sum(1 for n in nodes if n is not root and node_cite_count[n.rid] == 0)),
        "total_report_cites": float(total_cites),
    }
    if _shape > 0.0:
        metrics["child_cite_shaping"] = float(_shape)
        metrics["child_cite_shaping_mean"] = float(_shape_sum / max(_n_children, 1))
    if cit_R is not None:
        metrics["citation_reward"] = float(cit_R)
    for _k, _v in _proc.items():  # Run B process terms (format / search), when blended
        metrics[f"{_k}_reward"] = float(_v)

    # ---- 5. structural penalty channel (L4) ----
    if mode == "rer_structural":
        penalty, struct_metrics = _structural_penalty(nodes, node_cite_count, payload)
        root.reward -= penalty
        metrics.update(struct_metrics)
        metrics["structural_penalty"] = penalty

    res.rewards = {n.rid: n.reward for n in nodes}
    res.metrics = metrics
    return res


def _structural_penalty(
    nodes: List[RerNode], node_cite_count: Dict[str, int], payload: dict
) -> Tuple[float, Dict[str, float]]:
    """Co-evolving structural-rubric stand-in: penalize orphan / redundant / over-
    fragmented decompositions (proposal §6.4, RQ3). Returns (penalty, metrics)."""
    root = _root(nodes)
    children = _children(nodes, root.rid)
    p_orphan = float(payload.get("structural_orphan_penalty", 0.1))
    p_redund = float(payload.get("structural_redundancy_penalty", 0.1))
    p_frag = float(payload.get("structural_fragmentation_penalty", 0.05))
    soft_cap = int(payload.get("structural_max_children_soft", 6))

    n_orphan = sum(1 for c in children if node_cite_count.get(c.rid, 0) == 0)

    # redundancy: sibling answer-token Jaccard > 0.6 counts as a redundant pair
    n_redundant = 0
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            a, b = _tokset(children[i].final_answer), _tokset(children[j].final_answer)
            if a and b and len(a & b) / len(a | b) > 0.6:
                n_redundant += 1

    n_excess = max(0, len(children) - soft_cap)
    penalty = p_orphan * n_orphan + p_redund * n_redundant + p_frag * n_excess
    return penalty, {
        "struct_orphan": float(n_orphan),
        "struct_redundant_pairs": float(n_redundant),
        "struct_excess_children": float(n_excess),
    }
