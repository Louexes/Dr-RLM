"""Tests for the restored run-safety guards: hard-credit abort + per-tree call budget."""

import threading

import examples.train.dr_rlm  # noqa: F401  (register env id)
from examples.train.dr_rlm import _run_guard
from examples.train.dr_rlm.tools import provider as prov
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig


class _FakeProvider:
    def __init__(self):
        self.calls = 0

    def search(self, q, k):
        self.calls += 1
        return [{"docid": "u", "snippet": "s", "url": "u", "score": 1.0}]

    def get_text(self, d):
        return "text"


def _ledger():
    return {"snippets": {}, "_lock": threading.Lock()}


def test_note_exception_hard_vs_transient():
    _run_guard.reset()
    assert _run_guard.note_exception(Exception("HTTP 402: not enough credits")) is True
    assert _run_guard.hard_error_latched() is True
    _run_guard.reset()
    assert _run_guard.note_exception(Exception("connection reset by peer")) is False
    assert _run_guard.hard_error_latched() is False
    _run_guard.reset()


def test_config_threads_budget():
    assert DrRlmGeneratorConfig(max_search_calls_per_tree=80).env_payload()["max_search_calls_per_tree"] == 80
    assert DrRlmGeneratorConfig().env_payload()["max_search_calls_per_tree"] == 0


def test_search_short_circuits_when_latched(monkeypatch):
    _run_guard.reset()
    fake = _FakeProvider()
    monkeypatch.setattr(prov, "get_provider", lambda payload: fake)
    tools = prov.make_tools({"search_backend": "web"}, "node", [], _ledger())
    assert tools["search"]("q")           # works before the latch
    assert fake.calls == 1
    _run_guard.note_exception(Exception("status 402"))
    assert tools["search"]("q") == []     # short-circuited
    assert fake.calls == 1                 # provider NOT called again
    _run_guard.reset()


def test_per_tree_budget_cap(monkeypatch):
    _run_guard.reset()
    fake = _FakeProvider()
    monkeypatch.setattr(prov, "get_provider", lambda payload: fake)
    led = _ledger()
    tools = prov.make_tools({"search_backend": "web", "max_search_calls_per_tree": 2}, "node", [], led)
    assert tools["search"]("q")           # 1
    assert tools["search"]("q")           # 2
    assert tools["search"]("q") == []     # 3 -> budget exhausted
    assert fake.calls == 2                 # backend hit exactly twice
    _run_guard.reset()


def test_budget_shared_across_nodes(monkeypatch):
    """Two nodes of the same tree share one ledger -> one budget."""
    _run_guard.reset()
    fake = _FakeProvider()
    monkeypatch.setattr(prov, "get_provider", lambda payload: fake)
    led = _ledger()
    payload = {"search_backend": "web", "max_search_calls_per_tree": 3}
    root = prov.make_tools(payload, "root", [], led)
    child = prov.make_tools(payload, "child", [], led)
    assert root["search"]("q")            # 1
    assert child["search"]("q")           # 2
    assert root["search"]("q")            # 3
    assert child["search"]("q") == []     # 4 -> tree budget spent
    assert fake.calls == 3
    _run_guard.reset()
