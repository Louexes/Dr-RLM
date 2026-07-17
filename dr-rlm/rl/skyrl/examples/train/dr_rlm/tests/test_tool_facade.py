"""Locks the RER provenance invariant ACROSS the online/offline switch.

The whole online/offline design hinges on one claim: id-mint + ledger writes live ONLY in
``make_tools`` (provider-agnostic), so whatever backend supplies the raw hits, the snippet
ids and ledger records are byte-identical — hence the RER credit graph is unaffected by
``search_backend``. These tests assert exactly that, with a mock provider standing in for
both corpus- and web-shaped hits (so no network / dr_agent needed).
"""

import threading

import pytest

from examples.train.dr_rlm.tools import provider as P
from examples.train.dr_rlm.tools.provider import make_tools, owner_rid_of


class _MockProvider:
    def __init__(self, hits):
        self._hits = list(hits)

    def search(self, query, k):
        return self._hits

    def get_text(self, doc_id):
        return "FULLTEXT:" + str(doc_id)


# Corpus backends return {docid, snippet, url, score}; the web provider returns the same
# shape (docid=url, snippet=Document.stringify(), score=None). make_tools must treat them
# identically.
CORPUS_HITS = [
    {"docid": "d1", "snippet": "alpha alpha", "url": "u1", "score": 0.9},
    {"docid": "d2", "snippet": "beta beta", "url": "u2", "score": 0.8},
]
WEB_HITS = [
    {"docid": "http://a", "snippet": "Title: A\nURL: http://a\nSnippet: alpha", "url": "http://a", "score": None},
    {"docid": "http://b", "snippet": "Title: B\nURL: http://b\nSnippet: beta", "url": "http://b", "score": None},
]

_LEDGER_KEYS = {"node_rid", "text", "url", "docid", "query"}
_ROW_KEYS = {"id", "text", "url", "score", "docid"}


def _ledger():
    return {"snippets": {}, "_lock": threading.Lock()}


def _run(monkeypatch, hits, node_rid):
    """Drive make_tools with a mock provider (patching get_provider, which make_tools calls)."""
    monkeypatch.setattr(P, "get_provider", lambda payload: _MockProvider(hits))
    surfaced, ledger = [], _ledger()
    tools = make_tools({"search_backend": "x", "snippet_max_chars": 2000}, node_rid, surfaced, ledger)
    rows = tools["search"]("the query")
    return tools, rows, surfaced, ledger


def test_online_and_offline_mint_identical_ids_ledger_and_rows(monkeypatch):
    _, rows_c, surf_c, led_c = _run(monkeypatch, CORPUS_HITS, "ab12cd")
    _, rows_w, surf_w, led_w = _run(monkeypatch, WEB_HITS, "ef34gh")

    # 1. id FORMAT is {node_rid}-{n}, regardless of backend
    assert surf_c == ["ab12cd-0", "ab12cd-1"]
    assert surf_w == ["ef34gh-0", "ef34gh-1"]

    # 2. ledger record KEYS are identical in both modes
    for led in (led_c, led_w):
        assert led["snippets"]
        for rec in led["snippets"].values():
            assert set(rec.keys()) == _LEDGER_KEYS

    # 3. REPL-facing row shape is identical in both modes
    for rows in (rows_c, rows_w):
        for r in rows:
            assert set(r.keys()) == _ROW_KEYS

    # 4. owner decode round-trips both, and no id carries an extra dash (web urls live in
    #    docid/url, NEVER in the id) -> credit attribution is unambiguous
    for sid in surf_c + surf_w:
        assert sid.count("-") == 1
        assert owner_rid_of(sid) == sid.rsplit("-", 1)[0]
    assert owner_rid_of("ef34gh-1") == "ef34gh"


def test_node_rid_must_be_dash_free(monkeypatch):
    # A dashed rid would make owner_rid_of(rsplit('-',1)) decode the WRONG owner -> silent
    # credit corruption. make_tools must fail loud at mint time.
    monkeypatch.setattr(P, "get_provider", lambda payload: _MockProvider(CORPUS_HITS))
    with pytest.raises(AssertionError):
        make_tools({"search_backend": "x"}, "bad-rid", [], _ledger())


def test_get_doc_resolves_surfaced_sid_via_ledger(monkeypatch):
    monkeypatch.setattr(P, "get_provider", lambda payload: _MockProvider(CORPUS_HITS))
    surfaced, ledger = [], _ledger()
    tools = make_tools({"search_backend": "x"}, "ab12cd", surfaced, ledger)
    tools["search"]("q")
    # get_doc on a surfaced snippet id resolves back to its docid and fetches fuller text.
    # Returns a dict {id, text, url} (SAME shape as a search hit) so the model reads it the
    # same way (doc["text"]) — previously a bare string, which the model indexed -> TypeError.
    out = tools["get_doc"]("ab12cd-0")
    assert isinstance(out, dict) and out["id"] == "ab12cd-0"
    assert out["text"].startswith("FULLTEXT:")


def test_get_doc_bad_id_returns_dict_not_crash(monkeypatch):
    # an unknown / wrong-typed id must still return a DICT (never raise, never a bare string), so
    # the model's doc["text"] works instead of crashing the cell on a string index. The real
    # backend yields text="" for a bad id; the mock echoes it, so we only assert shape + str(id).
    monkeypatch.setattr(P, "get_provider", lambda payload: _MockProvider(CORPUS_HITS))
    tools = make_tools({"search_backend": "x"}, "ab12cd", [], _ledger())
    out = tools["get_doc"](7)                     # integer id (a real model mistake)
    assert isinstance(out, dict) and out["id"] == "7" and set(out) == {"id", "text", "url"}


def test_search_disabled_when_backend_none(monkeypatch):
    monkeypatch.setattr(P, "get_provider", lambda payload: None)
    tools = make_tools({"search_backend": "none"}, "ab12cd", [], _ledger())
    assert tools["search"]("q") == []
    assert tools["get_doc"]("ab12cd-0") == {"id": "ab12cd-0", "text": "", "url": ""}


def test_tools_never_auto_print(monkeypatch, capsys):
    """Hard invariant: search()/get_doc() are SILENT — the model accesses everything
    programmatically via the return value and prints only what it chooses (context decoupling)."""
    monkeypatch.setattr(P, "get_provider", lambda payload: _MockProvider(CORPUS_HITS))
    surfaced, ledger = [], _ledger()
    tools = make_tools({"search_backend": "x"}, "ab12cd", surfaced, ledger)
    tools["search"]("q")            # has hits
    tools["get_doc"]("ab12cd-0")    # resolves via ledger
    tools["get_doc"]("missing-id")  # no hit
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == "", f"tools must not print; got stdout={cap.out!r} stderr={cap.err!r}"
    # ... and the no-hits / disabled paths are silent too
    monkeypatch.setattr(P, "get_provider", lambda payload: _MockProvider([]))
    make_tools({"search_backend": "x"}, "ab12cd", [], _ledger())["search"]("q")
    monkeypatch.setattr(P, "get_provider", lambda payload: None)
    make_tools({"search_backend": "none"}, "ab12cd", [], _ledger())["search"]("q")
    cap2 = capsys.readouterr()
    assert cap2.out == "" and cap2.err == "", f"no-hits/disabled must be silent; got {cap2.out!r}/{cap2.err!r}"
