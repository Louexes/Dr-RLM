"""Tests for the RER reward pipeline: credit conservation + the L1/L2/L4 modes.

The judge is monkeypatched (no HTTP): score_criterion/score_quality/supports return
fixed coroutine values, so these run anywhere.
"""

import asyncio

import pytest

from examples.train.dr_rlm import rer_reward
from examples.train.dr_rlm.rer_reward import RerNode, compute_rer_rewards


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def patch_judge(monkeypatch):
    """Patch RubricJudge so every criterion scores 1.0, quality 0.5, support 1.0."""
    from examples.train.dr_rlm.judge import RubricJudge

    async def _crit(self, response, question, criterion):
        return 1.0

    async def _qual(self, response, question):
        return 0.5

    async def _sup(self, criterion, question, evidence_text):
        return 1.0 if evidence_text.strip() else 0.0

    monkeypatch.setattr(RubricJudge, "score_criterion", _crit)
    monkeypatch.setattr(RubricJudge, "score_quality", _qual)
    monkeypatch.setattr(RubricJudge, "supports", _sup)


RUBRICS = [
    {"description": "covers cause", "title": "cause", "weight": 1.0},
    {"description": "covers effect", "title": "effect", "weight": 1.0},
]


def _tree():
    # root "aaaa" cites one snippet it surfaced (aaaa-0) and one a child surfaced (bbbb-0)
    root = RerNode(
        rid="aaaa",
        depth=0,
        parent_rid=None,
        final_answer='Cause is X <cite id="aaaa-0">x</cite>. Effect is Y <cite id="bbbb-0">y</cite>.',
    )
    child = RerNode(rid="bbbb", depth=1, parent_rid="aaaa", final_answer='Y holds <cite id="bbbb-0">y</cite>.')
    return [root, child]


def test_citation_count_credit_is_conserved(patch_judge):
    payload = {"reward_mode": "rer", "share_mode": "citation_count", "gamma_cost": 0.0}
    res = _run(compute_rer_rewards(_tree(), RUBRICS, "why X->Y?", payload))
    # all weights positive, s_c=1 -> R == 1.0
    assert res.report_reward == pytest.approx(1.0, abs=1e-6)
    # credit conserved: sum of per-node credit == R (no cost term)
    assert sum(res.rewards.values()) == pytest.approx(res.report_reward, abs=1e-6)
    # root and child each cited 1 of 2 -> equal split
    assert res.rewards["aaaa"] == pytest.approx(0.5, abs=1e-6)
    assert res.rewards["bbbb"] == pytest.approx(0.5, abs=1e-6)


def test_inherited_mode_only_root(patch_judge):
    payload = {"reward_mode": "inherited"}
    res = _run(compute_rer_rewards(_tree(), RUBRICS, "q", payload))
    assert res.rewards["aaaa"] == pytest.approx(res.report_reward)
    assert res.rewards["bbbb"] == pytest.approx(0.0)


def test_rao_mode_uses_child_success(patch_judge):
    payload = {"reward_mode": "rao", "rao_lambda": 0.5, "gamma_cost": 0.0}
    res = _run(compute_rer_rewards(_tree(), RUBRICS, "q", payload))
    # root r = s_tilde(root)=R(=1.0) + 0.5*mean(child quality=0.5) = 1.0 + 0.25
    assert res.rewards["aaaa"] == pytest.approx(1.0 + 0.5 * 0.5, abs=1e-6)
    # child has no children -> r = quality(0.5)
    assert res.rewards["bbbb"] == pytest.approx(0.5, abs=1e-6)


def test_structural_penalizes_orphan_child(patch_judge):
    # an orphan child: cites nothing that appears in the root report
    root = RerNode(rid="aaaa", depth=0, parent_rid=None, final_answer='X <cite id="aaaa-0">x</cite>.')
    orphan = RerNode(rid="cccc", depth=1, parent_rid="aaaa", final_answer="I found nothing useful.")
    nodes = [root, orphan]
    base = _run(compute_rer_rewards(nodes, RUBRICS, "q", {"reward_mode": "rer", "share_mode": "citation_count"}))
    struct = _run(
        compute_rer_rewards(
            [RerNode(**{k: getattr(n, k) for k in ("rid", "depth", "parent_rid", "final_answer")}) for n in nodes],
            RUBRICS,
            "q",
            {"reward_mode": "rer_structural", "share_mode": "citation_count", "structural_orphan_penalty": 0.1},
        )
    )
    assert struct.metrics.get("struct_orphan", 0) == 1.0
    # structural mode lowers the root reward relative to plain rer (orphan penalty)
    assert struct.rewards["aaaa"] < base.rewards["aaaa"]


def test_child_cite_shaping_off_by_default(patch_judge, monkeypatch):
    monkeypatch.delenv("DR_RLM_CHILD_CITE_SHAPING", raising=False)
    res = _run(compute_rer_rewards(_tree(), RUBRICS, "q",
                                   {"reward_mode": "rer", "share_mode": "citation_count", "gamma_cost": 0.0}))
    # unchanged from the conserved citation_count baseline
    assert res.rewards["aaaa"] == pytest.approx(0.5, abs=1e-6)
    assert res.rewards["bbbb"] == pytest.approx(0.5, abs=1e-6)
    assert "child_cite_shaping" not in res.metrics


def test_child_cite_shaping_rewards_citing_child_not_root(patch_judge, monkeypatch):
    base = _run(compute_rer_rewards(_tree(), RUBRICS, "q",
                                    {"reward_mode": "rer", "share_mode": "citation_count", "gamma_cost": 0.0}))
    monkeypatch.setenv("DR_RLM_CHILD_CITE_SHAPING", "0.3")
    monkeypatch.setenv("DR_RLM_CHILD_CITE_CAP", "3")
    shaped = _run(compute_rer_rewards(_tree(), RUBRICS, "q",
                                      {"reward_mode": "rer", "share_mode": "citation_count", "gamma_cost": 0.0}))
    # child bbbb cited 1 own snippet -> +0.3 * min(1,3)/3 = +0.1; root never shaped
    assert shaped.rewards["bbbb"] == pytest.approx(base.rewards["bbbb"] + 0.1, abs=1e-6)
    assert shaped.rewards["aaaa"] == pytest.approx(base.rewards["aaaa"], abs=1e-6)
    assert shaped.metrics["child_cite_shaping_mean"] == pytest.approx(0.1, abs=1e-6)


def test_extract_claims_decodes_owner():
    from examples.train.dr_rlm.judge import extract_claims_and_corresponding_citation_ids
    from examples.train.dr_rlm.corpus_search import owner_rid_of

    claims = extract_claims_and_corresponding_citation_ids('A <cite id="ab12cd-7">a</cite> B <cite id="ef34-2,ab12cd-9">b</cite>')
    ids = [cid for v in claims.values() for cid in v]
    assert "ab12cd-7" in ids and "ef34-2" in ids and "ab12cd-9" in ids
    assert owner_rid_of("ab12cd-7") == "ab12cd"
    assert owner_rid_of("ef34-2") == "ef34"
