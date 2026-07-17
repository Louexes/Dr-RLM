"""Invariants of the FREEZE-CANDIDATE prompt (prompts/variants/final_v4.txt — re-claim
carry rule + generic MEASURE step, NO check_citations / verification), so canonical
adoption via scripts/adopt_capstone.sh is CI-guarded: these assertions hold for the variant
file today and keep holding for canonical system_prompt.txt after the swap (the loader below
follows DR_RLM_PROMPT_FILE, which adopt_capstone.sh leaves unset post-adoption -> canonical).

Fairness ruling (2026-06-16): final_v3's mechanical check_citations FILTER was REMOVED —
it makes a bespoke LLM judge call, a capability DR-Tulu's toolset has no analog of, which
breaks the principle that the RLM's only legitimate advantages over DR-Tulu are recursion
(rlm_query) + the REPL substrate. Citation validity is now left to RL. So the freeze
candidate carries ONLY deterministic, LLM-call-free behaviors:
* the delegation contract says RE-CLAIM, never keep-claim-verbatim (the citation-mass fix);
* a generic MEASURE step (deterministic len()/format check — no judge);
* NO check_citations, NO FILTER recipe, NO verification of any kind;
* benchmark-neutral: no benchmark names or output-shape words in the shared prompt.
"""

import importlib.util
import os
from pathlib import Path

_DRRLM = Path(__file__).resolve().parents[6]  # tests -> dr_rlm -> train -> examples -> skyrl -> rl -> dr-rlm
_CANDIDATE = _DRRLM / "prompts" / "variants" / "final_v4.txt"
_PROMPTS_PY = Path(__file__).resolve().parents[1] / "prompts.py"


def _load(prompt_file):
    old = os.environ.get("DR_RLM_PROMPT_FILE")
    os.environ["DR_RLM_PROMPT_FILE"] = str(prompt_file)
    try:
        spec = importlib.util.spec_from_file_location("prompts_candidate", _PROMPTS_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old is None:
            os.environ.pop("DR_RLM_PROMPT_FILE", None)
        else:
            os.environ["DR_RLM_PROMPT_FILE"] = old


def test_candidate_file_exists():
    assert _CANDIDATE.exists(), f"freeze candidate missing: {_CANDIDATE}"


def test_no_check_citations_or_verification_anywhere():
    """The fairness ruling: NO bespoke citation-judging tool / filter in the harness."""
    p = _load(_CANDIDATE)
    for depth in (0, 1):
        for mode in ("structured", "prose"):
            s = p.depth_system_prompt(depth, 1, mode)
            for banned in ("check_citations", "FINAL FILTER", "2. FILTER", "UNSUPPORTED", "cite-audit"):
                assert banned not in s, f"verification machinery leaked ({banned}) in {mode}@{depth}"


def test_measure_present_all_bands():
    p = _load(_CANDIDATE)
    for depth in (0, 1):
        for mode in ("structured", "prose"):
            s = p.depth_system_prompt(depth, 1, mode)
            assert "MEASURE your" in s, f"MEASURE missing in {mode}@{depth}"
            assert "{CITATIONS}" not in s and "{EVIDENCE}" not in s and "{FINISHING}" not in s


def test_delegation_is_reclaim_not_verbatim():
    p = _load(_CANDIDATE)
    d = p.delegation_section("structured")
    assert "RE-CLAIM" in d
    assert "keep the claim text verbatim" not in d


def test_benchmark_neutrality():
    p = _load(_CANDIDATE)
    s = p.depth_system_prompt(0, 1, "structured").lower()
    for banned in ("240", "260", "researchqa", "healthbench", "deep_research_bench",
                   "sqav2", "multi-section", "comprehensive"):
        assert banned not in s, f"benchmark/output-shape leak: {banned}"
