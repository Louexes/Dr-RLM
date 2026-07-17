"""Tests for share_mode='ledger_support': credit survives the orchestrator dropping
sub-agents' citations (the A2 DRB FACT≈0 failure).

Setup mirrors that failure: a child does the real research and cites its OWN evidence,
but the root (orchestrator) synthesizes WITHOUT any <cite> tags. Under the prose-parsing
'citation_count' mode the child gets 0 (it looks like an orphan); under 'ledger_support'
the child still gets credit, because provenance is read from the harness evidence ledger
(what the child retrieved+cited), not the root's prose.

Judge is monkeypatched (no HTTP), so this runs anywhere.
"""

import asyncio
import threading

import pytest

from examples.train.dr_rlm.rer_reward import RerNode, compute_rer_rewards


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def patch_judge(monkeypatch):
    from examples.train.dr_rlm.judge import RubricJudge

    async def _crit(self, response, question, criterion):  # every criterion satisfied -> R = 1.0
        return 1.0

    async def _supports(self, criterion, question, evidence_text):  # supported iff there is evidence
        return 1.0 if evidence_text.strip() else 0.0

    monkeypatch.setattr(RubricJudge, "score_criterion", _crit)
    monkeypatch.setattr(RubricJudge, "supports", _supports)


RUBRICS = [{"description": "covers the key finding", "title": "finding", "weight": 1.0}]


def _tree_and_ledger():
    # root synthesized a report with NO citations (orchestrator dropped the child's cites)
    root = RerNode(rid="root", depth=0, parent_rid=None,
                   final_answer="Synthesized report. The finding holds. (no citations)",
                   cited_ids=[])
    # the child did the research and cited ITS OWN evidence id
    child = RerNode(rid="child", depth=1, parent_rid="root",
                    final_answer='The finding holds <cite id="child-0">X is true per source</cite>.',
                    cited_ids=["child-0"])
    ledger = {
        "snippets": {"child-0": {"node_rid": "child", "text": "X is true per source", "url": "u"}},
        "_lock": threading.Lock(),
    }
    return [root, child], ledger


def test_ledger_support_credits_the_evidence_bearing_child(patch_judge):
    nodes, ledger = _tree_and_ledger()
    res = _run(compute_rer_rewards(nodes, RUBRICS, "q?",
                                   {"reward_mode": "rer", "share_mode": "ledger_support"}, ledger=ledger))
    assert res.report_reward == pytest.approx(1.0, abs=1e-6)
    # the child (sole evidence contributor) gets the credit, even though the root never cited it
    assert res.rewards["child"] > 0.0
    assert res.rewards["child"] == pytest.approx(1.0, abs=1e-6)
    assert res.rewards["root"] == pytest.approx(0.0, abs=1e-6)
    # credit conserved: sum == R
    assert sum(res.rewards.values()) == pytest.approx(res.report_reward, abs=1e-6)


def test_citation_count_loses_the_child_credit(patch_judge):
    # contrast: the prose-parsing mode reads the ROOT report's <cite> tags; since the root
    # dropped them, the child looks like an orphan and gets nothing (the A2 failure).
    nodes, _ = _tree_and_ledger()
    res = _run(compute_rer_rewards(nodes, RUBRICS, "q?",
                                   {"reward_mode": "rer", "share_mode": "citation_count"}))
    assert res.rewards["child"] == pytest.approx(0.0, abs=1e-6)
    assert res.rewards["root"] == pytest.approx(0.0, abs=1e-6)


def test_ledger_support_ignores_unledgered_or_foreign_cites(patch_judge):
    # a node only earns credit for ids it OWNS and that exist in the ledger
    root = RerNode(rid="root", depth=0, parent_rid=None, final_answer="report", cited_ids=[])
    # cites someone else's id + a non-existent id -> no valid own evidence -> no credit
    bad = RerNode(rid="bad", depth=1, parent_rid="root",
                  final_answer='claim <cite id="other-9">y</cite><cite id="bad-7">z</cite>',
                  cited_ids=["other-9", "bad-7"])
    ledger = {"snippets": {"other-9": {"node_rid": "other", "text": "y"}}, "_lock": threading.Lock()}
    res = _run(compute_rer_rewards([root, bad], RUBRICS, "q?",
                                   {"reward_mode": "rer", "share_mode": "ledger_support"}, ledger=ledger))
    # "other-9" isn't owned by `bad`; "bad-7" isn't in the ledger -> no contributors -> all 0
    assert res.rewards["bad"] == pytest.approx(0.0, abs=1e-6)
    assert res.rewards["root"] == pytest.approx(0.0, abs=1e-6)


WEIGHTED_RUBRICS = [
    {"description": "covers the key finding", "title": "finding", "weight": 3.0},
    {"description": "explains the mechanism", "title": "mechanism", "weight": 3.0},
    {"description": "gives a counterexample", "title": "counter", "weight": 6.0},
]


def test_ledger_support_conserves_R_under_nonunit_weights(patch_judge):
    """Regression: with raw rubric weights (e.g. 3.0/6.0) credit must still sum to R,
    not to R * Σw (observed live as r_a=9.0 vs R=0.75 — smoke job 23668426)."""
    nodes, ledger = _tree_and_ledger()
    res = _run(compute_rer_rewards(nodes, WEIGHTED_RUBRICS, "q?",
                                   {"reward_mode": "rer", "share_mode": "ledger_support"}, ledger=ledger))
    # patched judge scores every criterion 1.0 -> R = Σw*1/Σw = 1.0
    assert res.report_reward == pytest.approx(1.0, abs=1e-6)
    # ALL credit goes to the sole contributor and must equal R, not R*Σw=12
    assert res.rewards["child"] == pytest.approx(res.report_reward, abs=1e-6)
    assert sum(res.rewards.values()) == pytest.approx(res.report_reward, abs=1e-6)


def test_hybrid_root_reward_full_R(patch_judge):
    """root_reward_mode=full_R: the ROOT earns R itself (undiluted synthesis-quality
    pressure); children keep their provenance credit shares. Σ != R by design."""
    nodes, ledger = _tree_and_ledger()
    res = _run(compute_rer_rewards(nodes, WEIGHTED_RUBRICS, "q?",
                                   {"reward_mode": "rer", "share_mode": "ledger_support",
                                    "root_reward_mode": "full_R"}, ledger=ledger))
    assert res.report_reward == pytest.approx(1.0, abs=1e-6)
    # root gets full R (not its credit share, which is 0 here — root cited nothing)
    assert res.rewards["root"] == pytest.approx(res.report_reward, abs=1e-6)
    # the evidence-bearing child KEEPS its conserved credit share
    assert res.rewards["child"] == pytest.approx(res.report_reward, abs=1e-6)


def test_hybrid_default_is_conserved(patch_judge):
    """Default root_reward_mode=credit_share preserves the original conserved behavior."""
    nodes, ledger = _tree_and_ledger()
    res = _run(compute_rer_rewards(nodes, WEIGHTED_RUBRICS, "q?",
                                   {"reward_mode": "rer", "share_mode": "ledger_support"}, ledger=ledger))
    assert sum(res.rewards.values()) == pytest.approx(res.report_reward, abs=1e-6)
