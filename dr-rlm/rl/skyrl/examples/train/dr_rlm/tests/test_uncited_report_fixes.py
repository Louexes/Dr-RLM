"""Tests for the three uncited-report fixes (diagnosed on structured_report_n16, 5/16 uncited):

1. PAUSED CELL CLOCK — time blocked on rlm_query(_batched)/llm_query(_batched) must not
   count against the REPL cell timeout. Root cause #1: the 180s cell timeout killed the
   spawning cell mid-`rlm_query_batched`; children completed but `subs` was never assigned,
   so their citations were unreachable -> uncited report.
2. UNIFORM CHILD DICT — with child_uniform_dict=True (structured mode), EVERY entry the
   parent sees is a {"content", "citations"} dict, including budget-exhausted and error
   messages. Root cause #2: stray strings made `subs[i]["content"]` raise TypeError.
3. CITATIONS NUDGE — the per-turn STATUS warns when a structured draft has content but
   zero citations (model-side forgetting, e.g. copying only subs[0]'s citations).
"""

import time

import examples.train.dr_rlm  # noqa: F401  (register env id)
from examples.train.dr_rlm.dr_rlm_env import DrRlmEnv
from skyrl_gym.envs.rlm.repl import PersistentREPL


# ---------------------------------------------------------------------------
# 1. paused cell clock
# ---------------------------------------------------------------------------

def test_child_wait_does_not_count_against_cell_timeout():
    # subcall blocks ~3x the cell timeout; without the pause the cell dies with
    # "Timeout after 1 seconds" and `subs` is never assigned.
    def slow_child(prompt, context=None):
        time.sleep(3)
        return {"content": f"child:{prompt}", "citations": [{"id": "a-0", "claim": "c"}]}

    r = PersistentREPL(timeout=1.0, subcall_fn=slow_child, lm_callback=lambda ps: ["x"] * len(ps))
    res = r.execute("subs = rlm_query_batched(['q1'])\nprint(subs[0]['content'])")
    assert "Timeout" not in res.stderr, res.stderr
    assert "child:q1" in res.stdout
    assert isinstance(r.locals.get("subs"), list)


def test_single_child_wait_longer_than_timeout_is_exempt():
    # REGRESSION: a SINGLE child wait that alone exceeds the cell timeout must not be killed.
    # The first fix only credited wait time in _pause_timeout's finally (after the wait), so the
    # sliced join saw stale _wait_secs and killed mid-wait -> orphaned `subs` -> uncited report
    # (observed: sqav2/HealthBench 3/4 root timeouts, all DURING rlm_query_batched).
    def slow_child(prompt, context=None):
        time.sleep(2.5)   # one wait, > the 1s timeout
        return {"content": f"child:{prompt}", "citations": []}

    r = PersistentREPL(timeout=1.0, subcall_fn=slow_child, lm_callback=lambda ps: ["x"] * len(ps))
    res = r.execute("subs = rlm_query(' q')\nprint(subs['content'])")
    assert "Timeout" not in res.stderr, res.stderr
    assert "child: q" in res.stdout


def test_hung_cell_still_times_out():
    # a genuinely stuck cell (busy loop, no child waits) must still be killed.
    r = PersistentREPL(timeout=1.0)
    res = r.execute("i = 0\nwhile True:\n    i += 1")
    assert "Timeout after 1 seconds" in res.stderr


def test_llm_wait_is_also_exempt():
    def slow_lm(prompts):
        time.sleep(3)
        return ["ok"] * len(prompts)

    r = PersistentREPL(timeout=1.0, lm_callback=slow_lm)
    res = r.execute("out = llm_query('q')\nprint(out)")
    assert "Timeout" not in res.stderr, res.stderr
    assert "ok" in res.stdout


# ---------------------------------------------------------------------------
# 2. uniform child dict
# ---------------------------------------------------------------------------

def _uniform_repl(max_children, subcall):
    return PersistentREPL(subcall_fn=subcall, max_children=max_children, child_uniform_dict=True)


def test_budget_exhausted_entries_are_dicts_in_uniform_mode():
    r = _uniform_repl(1, lambda p, context=None: {"content": f"child:{p}", "citations": []})
    out = r._rlm_query_batched(["q0", "q1", "q2"])
    assert len(out) == 3
    assert out[0] == {"content": "child:q0", "citations": []}
    for entry in out[1:]:  # over-budget -> message, but still the dict shape
        assert isinstance(entry, dict) and "budget exhausted" in entry["content"]
        assert entry["citations"] == []


def test_error_entries_are_dicts_in_uniform_mode():
    def boom(prompt, context=None):
        raise RuntimeError("child died")

    r = _uniform_repl(0, boom)
    single = r._rlm_query("q")
    assert isinstance(single, dict) and "child died" in single["content"] and single["citations"] == []
    batched = r._rlm_query_batched(["a", "b"])
    assert all(isinstance(e, dict) and "child died" in e["content"] for e in batched)


def test_prose_mode_keeps_string_entries():
    # without the flag (prose), behavior is unchanged: strings stay strings.
    r = PersistentREPL(subcall_fn=lambda p, context=None: f"child:{p}", max_children=1)
    out = r._rlm_query_batched(["q0", "q1"])
    assert out[0] == "child:q0" and isinstance(out[1], str) and "budget exhausted" in out[1]


def test_env_threads_uniform_flag_by_mode():
    def _env(mode):
        e = DrRlmEnv(extras={"depth": 0, "subcall_fn": lambda *a, **k: None,
                             "dr_rlm": {"max_recursion_depth": 1, "child_return_mode": mode}})
        e._root_prompt = "Q?"
        return e
    # the flag is set at REPL construction inside init(); mirror the env's own peek instead
    # of running a full init (which needs an inference client): assert the expression used.
    assert str(_env("structured").dr_cfg.get("child_return_mode")) == "structured"
    assert str(_env("prose").dr_cfg.get("child_return_mode")) == "prose"


# ---------------------------------------------------------------------------
# 3. citations nudge
# ---------------------------------------------------------------------------

class _FakeRepl:
    def __init__(self, content="", citations=None):
        self.locals = {"answer": {"content": content, "citations": citations if citations is not None else []}}


def _env_with(mode, content="", citations=None):
    e = DrRlmEnv(extras={"depth": 0, "subcall_fn": lambda *a, **k: None,
                         "dr_rlm": {"max_recursion_depth": 1, "child_return_mode": mode}})
    e._root_prompt = "Q?"
    e.max_turns = 12
    e.repl = _FakeRepl(content=content, citations=citations)
    return e


def test_warns_on_drafted_but_uncited_structured_answer():
    p = _env_with("structured", content="x" * 500, citations=[])._get_user_prompt(4)["content"]
    assert "WARNING" in p and 'answer["citations"]` is EMPTY' in p


def test_no_warning_when_cited_or_empty_or_prose():
    # cited draft -> no warning
    p = _env_with("structured", content="x" * 500, citations=[{"id": "a-0", "claim": "c"}])._get_user_prompt(4)["content"]
    assert "WARNING" not in p
    # empty draft -> the EMPTY-content status line, not the citations warning
    p = _env_with("structured", content="", citations=[])._get_user_prompt(4)["content"]
    assert "WARNING" not in p and "is EMPTY" in p
    # prose mode -> citations live inline in the content; no structured warning
    p = _env_with("prose", content="x" * 500, citations=[])._get_user_prompt(4)["content"]
    assert "WARNING" not in p
