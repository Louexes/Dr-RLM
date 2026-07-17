"""Tests for the check_citations REPL tool (answer_format.check_citations_verdicts +
DrRlmEnv._get_repl_tools wiring).

The tool is a verdict ORACLE the agent calls on answer["citations"] before submitting —
it judges every {"id","claim"} entry against the shared ledger and returns
SUPPORTED/UNSUPPORTED/UNKNOWN per entry, NEVER filtering (acting on verdicts is the
agent's job; that behavior is what prompts/variants/check_tool.txt elicits and RL can
train). Invariants under test:

* per-entry verdicts align positionally with the input list;
* no claim text / unknown id / judge noise -> UNKNOWN (caller keeps those);
* child-inherited entries are checkable (any sid in the shared ledger judges fine —
  the self_verify root loophole cannot recur);
* one BATCHED lm_callback call; malformed input or callback death never raises;
* env wiring: flag off -> no tool; flag on + lm_callback -> tool present, stats
  accumulate in _check_tool_stats and surface in get_metrics.
"""

import threading

import examples.train.dr_rlm  # noqa: F401
from examples.train.dr_rlm.answer_format import check_citations_verdicts
from examples.train.dr_rlm.dr_rlm_env import DrRlmEnv


def _ledger(**snips):
    return {"snippets": {k: {"node_rid": "n", "text": v, "url": ""} for k, v in snips.items()},
            "_lock": threading.Lock()}


_LED = _ledger(**{"a-0": "Paris has been France's capital for centuries.",
                  "b-1": "The moon is made of rock and dust.",
                  "child-3": "GDP grew 2.1% in 2024 according to the bureau."})

_CITS = [
    {"id": "a-0", "claim": "Paris is the capital of France."},
    {"id": "b-1", "claim": "The moon is made of cheese."},
    {"id": "child-3", "claim": "GDP grew 2.1% in 2024."},   # child-inherited sid: judged like any other
    {"id": "a-0", "claim": ""},                              # no claim -> UNKNOWN, not judged
    {"id": "nope", "claim": "Some claim."},                  # unknown id -> UNKNOWN, not judged
]


def test_verdicts_positional_and_judged():
    calls = []

    def lm(prompts):
        calls.append(len(prompts))
        out = []
        for p in prompts:
            out.append("UNSUPPORTED" if "cheese" in p else "SUPPORTED")
        return out

    v = check_citations_verdicts(_CITS, _LED, lm)
    assert [x["verdict"] for x in v] == ["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "UNKNOWN", "UNKNOWN"]
    assert [x["id"] for x in v] == [c["id"] for c in _CITS]
    assert calls == [3]  # ONE batched call covering exactly the judgeable entries


def test_noise_and_failures_yield_unknown_never_raise():
    v = check_citations_verdicts(_CITS, _LED, lambda ps: ["???" for _ in ps])
    assert [x["verdict"] for x in v][:3] == ["UNKNOWN", "UNKNOWN", "UNKNOWN"]

    def dead(ps):
        raise RuntimeError("judge down")

    v = check_citations_verdicts(_CITS, _LED, dead)
    assert len(v) == len(_CITS)  # rows exist, all UNKNOWN where judged
    v = check_citations_verdicts("not a list", _LED, lambda ps: [])
    assert isinstance(v, list)
    v = check_citations_verdicts([{"weird": 1}, None], _LED, lambda ps: [])
    assert all(x["verdict"] == "UNKNOWN" for x in v)


def test_no_ledger_all_unknown_no_callback_call():
    def lm(prompts):
        raise AssertionError("must not judge without a ledger")

    v = check_citations_verdicts(_CITS, None, lm)
    assert all(x["verdict"] == "UNKNOWN" for x in v)


def test_env_flag_gates_tool_exposure():
    e = DrRlmEnv(extras={"depth": 0, "lm_callback": lambda ps: ["SUPPORTED"] * len(ps),
                         "dr_rlm_ledger": _LED,
                         "dr_rlm": {"max_recursion_depth": 0, "check_citations_tool": False}})
    assert "check_citations" not in e._get_repl_tools()

    # no lm_callback -> no tool even when flagged on (nothing to judge with)
    e2 = DrRlmEnv(extras={"depth": 0, "dr_rlm_ledger": _LED,
                          "dr_rlm": {"max_recursion_depth": 0, "check_citations_tool": True}})
    assert "check_citations" not in e2._get_repl_tools()


def test_env_tool_stats_accumulate_and_surface():
    e = DrRlmEnv(extras={"depth": 0,
                         "lm_callback": lambda ps: ["SUPPORTED", "UNSUPPORTED"][: len(ps)],
                         "dr_rlm_ledger": _LED,
                         "dr_rlm": {"max_recursion_depth": 0, "check_citations_tool": True}})
    tool = e._get_repl_tools()["check_citations"]
    v = tool([{"id": "a-0", "claim": "Paris is the capital."},
              {"id": "b-1", "claim": "The moon is cheese."}])
    assert [x["verdict"] for x in v] == ["SUPPORTED", "UNSUPPORTED"]
    tool([{"id": "a-0", "claim": "Paris is the capital."}])
    assert e._check_tool_stats == {"calls": 2, "checked": 3, "supported": 2, "unsupported": 1}
    e._finalize_answer("done", {"content": "x", "citations": []})
    assert e.get_metrics().get("check_citations_tool") == {"calls": 2, "checked": 3,
                                                           "supported": 2, "unsupported": 1}
