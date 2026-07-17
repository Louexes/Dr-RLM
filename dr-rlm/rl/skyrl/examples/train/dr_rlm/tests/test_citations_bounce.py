"""Tests for the one-time empty-citations submission gate (DrRlmEnv._submission_bounce +
the BaseRLMEnv hook).

The gate targets the carry-displacement pathology that survived four prompt iterations:
a node writes a full report, never populates answer["citations"] although the tree
surfaced evidence, and submits. Invariants:

* default OFF — flag unset => never bounces (base hook returns None => behavior identical);
* fires ONLY when: flag on, content non-empty, citations empty, tree ledger has snippets;
* fires AT MOST ONCE per node (resubmit-as-is is always accepted);
* no ledger evidence => uncited submission is legitimate => no bounce;
* the message tells the model both repair and submit-as-is paths.
"""

import threading

import examples.train.dr_rlm  # noqa: F401
from examples.train.dr_rlm.dr_rlm_env import DrRlmEnv
from skyrl_gym.envs.rlm.env import BaseRLMEnv


def _ledger(n=3):
    snips = {f"n-{i}": {"node_rid": "n", "text": f"evidence {i}", "url": ""} for i in range(n)}
    return {"snippets": snips, "_lock": threading.Lock()}


def _env(flag=True, ledger=...):
    led = _ledger() if ledger is ... else ledger
    return DrRlmEnv(extras={"depth": 0, "dr_rlm_ledger": led,
                            "dr_rlm": {"max_recursion_depth": 0, "citations_bounce": flag}})


_UNCITED = {"content": "A full report with no citations.", "citations": []}


def test_base_hook_default_none():
    e = BaseRLMEnv(extras={})
    assert e._submission_bounce({"content": "x", "citations": []}) is None


def test_flag_off_never_bounces():
    assert _env(flag=False)._submission_bounce(_UNCITED) is None


def test_bounces_once_then_accepts():
    e = _env()
    msg = e._submission_bounce(_UNCITED)
    assert msg and "SUBMISSION PAUSED" in msg and "3 evidence snippets" in msg
    assert 'set answer["ready"] = True again' in msg  # both repair and as-is paths stated
    assert e._citations_bounced
    assert e._submission_bounce(_UNCITED) is None  # resubmission accepted unconditionally
    assert e.get_metrics().get("citations_bounced") is True


def test_no_bounce_when_cited_or_empty_content():
    e = _env()
    assert e._submission_bounce({"content": "report", "citations": [{"id": "n-0", "claim": "report"}]}) is None
    assert e._submission_bounce({"content": "   ", "citations": []}) is None
    assert not e._citations_bounced  # neither case consumed the one-time gate


def test_no_bounce_without_tree_evidence():
    assert _env(ledger=None)._submission_bounce(_UNCITED) is None
    assert _env(ledger=_ledger(0))._submission_bounce(_UNCITED) is None


def test_malformed_final_obj_is_safe():
    e = _env()
    assert e._submission_bounce(None) is None
    assert e._submission_bounce("not a dict") is None
