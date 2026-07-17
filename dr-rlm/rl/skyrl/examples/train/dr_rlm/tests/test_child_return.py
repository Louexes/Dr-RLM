"""Tests for the child->parent return contract (prose vs structured A/B).

Covers:
  * config flag threads into env_payload()
  * DrRlmGenerator._child_return_value returns the structured dict in 'structured' mode,
    falls back to prose when no structured object exists, and returns prose in 'prose' mode
  * DrRlmEnv delegation-prompt section describes the true return shape per mode
"""

import examples.train.dr_rlm  # noqa: F401  (register env id)
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig
from examples.train.dr_rlm.dr_rlm_generator import DrRlmGenerator
from examples.train.dr_rlm.dr_rlm_env import DrRlmEnv


def test_config_threads_child_return_mode():
    cfg = DrRlmGeneratorConfig(child_return_mode="structured")
    assert cfg.env_payload()["child_return_mode"] == "structured"
    # default is prose
    assert DrRlmGeneratorConfig().env_payload()["child_return_mode"] == "prose"


def _gen(mode):
    # _child_return_value only reads self._dr_payload + calls super(); construct without
    # the heavy __init__ (no tokenizer/engine needed for this unit).
    g = object.__new__(DrRlmGenerator)
    g._dr_payload = {"child_return_mode": mode}
    return g


_OBJ = {"content": "Paris is the capital.", "citations": [{"id": "a-1", "claim": "Paris is the capital."}]}
_METRICS = {"answer_obj": _OBJ, "final_answer": "Paris is the capital. <cite id=\"a-1\">...</cite>"}


def test_structured_returns_content_and_citations():
    # structured = REPORT-STYLE contract: child returns {content: full mini-report, citations: [...]}
    out = _gen("structured")._child_return_value(_METRICS)
    assert out == {"content": _OBJ["content"], "citations": _OBJ["citations"]}
    # each citation carries its own id, so provenance travels with it
    assert out["citations"][0] == {"id": "a-1", "claim": "Paris is the capital."}


def test_structured_fallback_is_uniform_dict_not_string():
    # a child with no structured object STILL returns a dict (never a bare string), so a
    # heterogeneous `subs` list can't break the parent's `child["content"]`/`child["citations"]` code
    out = _gen("structured")._child_return_value({"final_answer": "prose only"})
    assert out == {"content": "prose only", "citations": []}
    # empty-content obj is treated as no structured result -> same uniform fallback
    out2 = _gen("structured")._child_return_value(
        {"answer_obj": {"content": "", "citations": []}, "final_answer": "prose only"}
    )
    assert out2 == {"content": "prose only", "citations": []}


def test_prose_mode_returns_string():
    out = _gen("prose")._child_return_value(_METRICS)
    assert isinstance(out, str) and out == _METRICS["final_answer"]


def _env(mode):
    e = DrRlmEnv(extras={"depth": 0, "subcall_fn": lambda *a, **k: None,
                         "dr_rlm": {"max_recursion_depth": 2, "child_return_mode": mode}})
    e._root_prompt = "Q?"
    return e


def test_delegation_section_is_mode_aware():
    structured = _env("structured")._build_system_prompt()
    # structured = REPORT-STYLE contract: read child["content"], write a thorough report,
    # preserve child["citations"] -> no atom-stitching, no FINDINGS
    assert 'child["content"]' in structured and 'child["citations"]' in structured
    # the FINDINGS contract accessors/framing are gone (plain-English "findings" in the band is fine)
    assert 'child["findings"]' not in structured and "SYNTHESIZE FROM FINDINGS" not in structured
    assert "USE THE CHILDREN'S WORK" in structured.upper()   # process framing (use their evidence)
    # DE-COUPLING INVARIANT: output-shape (length/format) lives in per-benchmark TASK_FRAMING,
    # NOT the general system prompt — so it can't conflict with short-answer benchmarks (ResearchQA
    # 240w, HealthBench scoped). Keep these DRB-flavored words OUT of the shared prompt.
    assert "multi-section" not in structured.lower() and "comprehensive" not in structured.lower()
    assert "could not retrieve information" in structured.lower()   # anti-bail
    assert 'answer["citations"]' in structured                       # provenance via the answer dict
    assert "programmatically" in structured.lower()

    prose = _env("prose")._build_system_prompt()
    assert "mini-report (a string with inline" in prose
    assert 'child["citations"]' not in prose and 'child["findings"]' not in prose
    assert 'answer["citations"]' not in prose  # prose arm uses inline tags, not the answer dict


def test_citation_contract_consistent_across_all_bands():
    """The citation contract must match child_return_mode in EVERY band — orchestrator,
    coordinator, AND worker — not just the delegating ones."""
    from examples.train.dr_rlm.prompts import depth_system_prompt
    # (depth, max_depth) producing each band: orchestrator / coordinator / worker
    bands = {"orchestrator": (0, 2), "coordinator": (1, 2), "worker": (2, 2)}
    for name, (d, m) in bands.items():
        structured = depth_system_prompt(d, m, "structured")
        prose = depth_system_prompt(d, m, "prose")
        # structured: programmatic answer["citations"], NOT the prose "wrap inline <cite>" rule
        assert 'answer["citations"]' in structured and "PROGRAMMATICALLY" in structured, name
        assert "WRONG (footnote)" not in structured, f"{name} structured leaked the prose cite rule"
        # prose: inline-tag contract, NOT the structured programmatic one
        assert "WRONG (footnote)" in prose and 'answer["citations"]' not in prose, name


def test_worker_has_no_delegation_section():
    # no subcall_fn => cannot delegate => no delegation block, placeholder filled
    e = DrRlmEnv(extras={"depth": 0, "dr_rlm": {"max_recursion_depth": 0, "child_return_mode": "structured"}})
    e._root_prompt = "Q?"
    sp = e._build_system_prompt()
    assert "Delegation tools available" not in sp and "{custom_tools_section}" not in sp
