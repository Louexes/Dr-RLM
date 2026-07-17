"""Offline tests for the DR Tulu citation_reward integration (grounding-aware reward).

No network: ``judge._cite_chat_sync`` and ``judge.score_report_sync`` are monkeypatched so
the recall/precision/format math and the rubric+citation blend are tested deterministically.
The behavioral assertion is the Check-2 sign-flip: stripping citations off factual claims
must DROP citation_reward (it was ~flat under the old rubric-only reward).
"""

import os
import pytest

from examples.train.dr_rlm import judge as J


# --- fake judge: rate by which prompt template fired, so each claim type is deterministic ---
def _fake_cite_chat(prompt, cfg):
    if "Need Citation" in prompt:          # recall-no-citation: treat every claim as factual
        return "Need Citation: [[Yes]] Analysis: factual, needs a citation."
    if "Fully supported" in prompt:        # recall-has-citation rubric
        return "Rating: [[Fully supported]] Analysis: matches snippet."
    if "Relevant" in prompt:               # precision rubric
        return "Rating: [[Relevant]] Analysis: snippet has the key info."
    return ""


@pytest.fixture
def patched_judge(monkeypatch):
    monkeypatch.setattr(J, "_cite_chat_sync", _fake_cite_chat)
    # rubric scorer -> fixed 0.8 so composite math is checkable without network
    monkeypatch.setattr(J, "score_report_sync", lambda report, q, rubrics, cfg: (0.8, {"c": 0.8}))
    return J.JudgeConfig(model="x", base_url="http://x/v1", api_key="EMPTY")


def test_format_reward_is_valid_id_fraction():
    # one valid id (a), one hallucinated (z) -> 1/2
    assert J.score_citation_format({"claim": ["a", "z"]}, {"a": "txt"}) == 0.5
    assert J.score_citation_format({"claim": []}, {"a": "txt"}) == 0.0  # no ids cited


def test_ledger_citations_from_snippets():
    led = {"snippets": {"r1-0": {"text": "alpha", "node_rid": "r1"}, "r2-1": {"text": "beta"}}}
    assert J.ledger_citations(led) == {"r1-0": "alpha", "r2-1": "beta"}
    assert J.ledger_citations(None) == {}
    assert J.ledger_citations({}) == {}


def test_citation_reward_grounded_vs_stripped_signflip(patched_judge):
    cfg = patched_judge
    q = "What does X require?"
    cits = {"s1": "X requires liquid water as a solvent."}
    # pure-claim reports so the math is exact: every fragment IS the cited claim
    grounded = '<cite id="s1">X requires liquid water as a solvent</cite>'
    stripped = "X requires liquid water as a solvent."

    r_grounded = J.score_in_context_citations_sync(q, grounded, cits, cfg)
    r_stripped = J.score_in_context_citations_sync(q, stripped, cits, cfg)

    # cited+supported claim: F1=1, format=1 -> 0.6*1 + 0.4*1 = 1.0
    assert r_grounded == pytest.approx(1.0, abs=1e-6)
    # uncited factual claim: recall=0 -> F1=0, no valid ids -> format=0 -> 0.0
    assert r_stripped == pytest.approx(0.0, abs=1e-6)
    assert r_grounded - r_stripped > 0.5  # the sign-flip we lacked before


def test_uncited_prose_fragments_dilute_recall(patched_judge):
    """DR Tulu treats EVERY text fragment as a claim: uncited prose around a cited span is
    judged 'needs citation?' and (here) penalized. So a report with uncited prose scores
    below a pure cited claim — the mechanism that pushes the policy to cite its factual claims."""
    cfg = patched_judge
    cits = {"s1": "evidence text"}
    pure = '<cite id="s1">a fully cited factual claim</cite>'
    with_prose = 'Some uncited factual lead-in. <cite id="s1">a fully cited factual claim</cite>'
    r_pure = J.score_in_context_citations_sync("q", pure, cits, cfg)
    r_prose = J.score_in_context_citations_sync("q", with_prose, cits, cfg)
    assert r_pure == pytest.approx(1.0)
    assert r_prose < r_pure  # uncited prose fragment drags avg_f1 down


def test_citation_reward_zero_when_no_ledger(patched_judge):
    # nothing retrieved in the tree -> no citations dict -> 0.0 (uncited is legitimate)
    assert J.score_in_context_citations_sync("q", "some prose", {}, patched_judge) == 0.0


def test_composite_blends_and_renormalizes(patched_judge):
    cfg = patched_judge
    q = "What does X require?"
    cits = {"s1": "X requires liquid water."}
    grounded = '<cite id="s1">X requires liquid water</cite>'
    stripped = "X requires liquid water."
    w = {"rubric": 0.5, "citation": 0.2}

    Rg, cg = J.composite_report_reward(grounded, q, [{"description": "d", "weight": 1.0}], cits, cfg, w)
    Rs, cs = J.composite_report_reward(stripped, q, [{"description": "d", "weight": 1.0}], cits, cfg, w)

    # grounded: (0.5*0.8 + 0.2*1.0)/0.7 = 0.857 ; stripped: (0.5*0.8 + 0.2*0.0)/0.7 = 0.571
    assert Rg == pytest.approx((0.5 * 0.8 + 0.2 * 1.0) / 0.7, abs=1e-6)
    assert Rs == pytest.approx((0.5 * 0.8 + 0.2 * 0.0) / 0.7, abs=1e-6)
    assert Rg > Rs
    assert cg["citation"] == pytest.approx(1.0) and cs["citation"] == pytest.approx(0.0)


def test_precision_reuse_skips_gemini_precision_calls(monkeypatch):
    """Cheap path: when precision_by_id is supplied, a cited claim's precision comes from the
    local verdict and NO Gemini precision call is made (recall still fires)."""
    calls = {"precision": 0, "recall_has": 0, "recall_no": 0}

    def counting_chat(prompt, cfg):
        if "Need Citation" in prompt:
            calls["recall_no"] += 1
            return "Need Citation: [[Yes]] ..."
        if "Fully supported" in prompt:
            calls["recall_has"] += 1
            return "Rating: [[Fully supported]] ..."
        if "Relevant" in prompt:
            calls["precision"] += 1
            return "Rating: [[Relevant]] ..."
        return ""

    monkeypatch.setattr(J, "_cite_chat_sync", counting_chat)
    cfg = J.JudgeConfig(model="x", base_url="http://x/v1", api_key="EMPTY")
    cits = {"s1": "evidence"}
    report = '<cite id="s1">a fully cited factual claim</cite>'

    # cheap: precision supplied locally -> 0 Gemini precision calls, recall_has still fires
    r = J.score_in_context_citations_sync("q", report, cits, cfg, precision_by_id={"s1": 1.0})
    assert calls["precision"] == 0 and calls["recall_has"] == 1
    assert r == pytest.approx(1.0)  # recall 1, precision 1 (local) -> F1 1, fmt 1

    # local verdict says unsupported -> precision 0 -> F1 0 -> citation_reward = 0.4*fmt
    calls["recall_has"] = 0
    r0 = J.score_in_context_citations_sync("q", report, cits, cfg, precision_by_id={"s1": 0.0})
    assert calls["precision"] == 0
    assert r0 == pytest.approx(0.4)  # F1=0, fmt=1 -> 0.6*0 + 0.4*1

    # no usable local verdict -> FALLS BACK to a Gemini precision call
    calls["precision"] = 0
    J.score_in_context_citations_sync("q", report, cits, cfg, precision_by_id={"s1": None})
    assert calls["precision"] == 1


def test_format_reward_harness_native():
    long = "x " * 150  # >=200 chars
    assert J.score_format_reward(f'<cite id="a">{long}</cite>') == pytest.approx(1.0)  # substantive + cited
    assert J.score_format_reward(long) == pytest.approx(0.5)                            # substantive, no cite
    assert J.score_format_reward('<cite id="a">short</cite>') == pytest.approx(0.5)     # cited but short
    assert J.score_format_reward("") == pytest.approx(0.0)


def test_tree_search_reward_treewide_volume():
    assert J.tree_search_reward(0, 8) == 0.0
    assert J.tree_search_reward(4, 8) == pytest.approx(0.5)
    assert J.tree_search_reward(8, 8) == pytest.approx(1.0)
    assert J.tree_search_reward(20, 8) == pytest.approx(1.0)  # capped


def test_config_reads_process_weights(monkeypatch):
    monkeypatch.setenv("DR_RLM_CITATION_REWARD", "1")
    monkeypatch.setenv("DR_RLM_FORMAT_WEIGHT", "0.2")
    monkeypatch.setenv("DR_RLM_SEARCH_WEIGHT", "0.1")
    on, w = J.citation_reward_config()
    assert on and w == {"rubric": 0.5, "citation": 0.2, "format": 0.2, "search": 0.1}
    # Run A (no process weights) stays rubric+citation only
    monkeypatch.delenv("DR_RLM_FORMAT_WEIGHT"); monkeypatch.delenv("DR_RLM_SEARCH_WEIGHT")
    on, w = J.citation_reward_config()
    assert w == {"rubric": 0.5, "citation": 0.2}


def test_composite_B_blends_all_four(patched_judge):
    cfg = patched_judge
    long = "word " * 60
    report = f'<cite id="s1">{long}</cite>'   # substantive + cited -> format 1.0
    cits = {"s1": "evidence"}                  # 1 snippet -> search 1/8 = 0.125
    w = {"rubric": 0.5, "citation": 0.2, "format": 0.2, "search": 0.1}
    R, comp = J.composite_report_reward(report, "q", [{"description": "d", "weight": 1.0}], cits, cfg, w)
    assert "format" in comp and "search" in comp
    assert comp["format"] == pytest.approx(1.0)
    assert comp["search"] == pytest.approx(0.125)
    # (0.5*0.8 + 0.2*1.0 + 0.2*1.0 + 0.1*0.125)/1.0
    assert R == pytest.approx((0.5 * 0.8 + 0.2 * 1.0 + 0.2 * 1.0 + 0.1 * 0.125), abs=1e-6)


def test_config_gate_default_off(monkeypatch):
    monkeypatch.delenv("DR_RLM_CITATION_REWARD", raising=False)
    on, w = J.citation_reward_config()
    assert on is False and "citation" not in w  # isolation: default rubric-only
    monkeypatch.setenv("DR_RLM_CITATION_REWARD", "1")
    monkeypatch.setenv("DR_RLM_CITATION_WEIGHT", "0.3")
    on, w = J.citation_reward_config()
    assert on is True and w["citation"] == 0.3
