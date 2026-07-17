"""Tests for the per-node sub-agent fan-out cap (tree-width bound)."""

import examples.train.dr_rlm  # noqa: F401  (register env id)
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig
from skyrl_gym.envs.rlm.repl import PersistentREPL


def _repl(max_children):
    calls = {"n": 0}

    def subcall_fn(prompt, context=None):
        calls["n"] += 1
        return f"child:{prompt}"

    r = PersistentREPL(subcall_fn=subcall_fn, max_children=max_children)
    return r, calls


def test_config_threads_max_children():
    assert DrRlmGeneratorConfig(max_children_per_node=6).env_payload()["max_children_per_node"] == 6
    assert DrRlmGeneratorConfig().env_payload()["max_children_per_node"] == 0


def test_unlimited_by_default():
    r, calls = _repl(0)
    out = r._rlm_query_batched([f"q{i}" for i in range(10)])
    assert len(out) == 10 and calls["n"] == 10
    assert all(o.startswith("child:") for o in out)


def test_batched_caps_and_preserves_length_order():
    r, calls = _repl(3)
    out = r._rlm_query_batched([f"q{i}" for i in range(5)])
    assert len(out) == 5                       # length preserved
    assert calls["n"] == 3                     # only 3 spawned
    assert out[0] == "child:q0" and out[2] == "child:q2"
    assert "budget exhausted" in out[3] and "budget exhausted" in out[4]


def test_budget_is_cumulative_across_calls():
    r, calls = _repl(3)
    a = r._rlm_query_batched(["q0", "q1"])     # 2 spawned, 1 left
    b = r._rlm_query("q2")                      # 3rd spawn, ok
    c = r._rlm_query("q3")                      # over budget -> message
    assert calls["n"] == 3
    assert a[0].startswith("child:") and a[1].startswith("child:")
    assert b.startswith("child:")
    assert "budget exhausted" in c


def test_single_spawn_not_double_counted():
    r, calls = _repl(2)
    r._rlm_query_batched(["q0"])               # 1 via batched single-path
    r._rlm_query("q1")                          # 2 via single
    over = r._rlm_query("q2")                   # 3rd blocked
    assert calls["n"] == 2 and "budget exhausted" in over


def test_no_subcall_fn_ignores_cap():
    # no recursion configured -> falls back to llm_query path, cap irrelevant (no crash)
    r = PersistentREPL(max_children=2)
    assert r.subcall_fn is None
