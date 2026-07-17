"""Tests for the corpus search() tool: provenance-encoded ids + local_jsonl backend."""

import json

from examples.train.dr_rlm.corpus_search import make_corpus_tools, owner_rid_of


def test_owner_rid_roundtrip():
    for rid in ("ab12cd", "deadbeef", "00ff99"):
        assert owner_rid_of(f"{rid}-0") == rid
        assert owner_rid_of(f"{rid}-12345") == rid


def test_local_jsonl_search_mints_provenance_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"id": "d1", "contents": "Photosynthesis converts sunlight into chemical energy in plants.", "url": "u1"},
        {"id": "d2", "contents": "The mitochondria is the powerhouse of the cell.", "url": "u2"},
        {"id": "d3", "contents": "Sunlight drives the light-dependent reactions of photosynthesis.", "url": "u3"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows))

    payload = {
        "search_backend": "local_jsonl",
        "search_corpus_path": str(corpus),
        "search_top_k": 5,
        "snippet_max_chars": 1000,
    }
    surfaced = []
    tools = make_corpus_tools(payload, node_rid="node99", surfaced_ids=surfaced)
    hits = tools["search"]("photosynthesis sunlight", k=3)

    assert hits, "expected at least one hit for an in-corpus query"
    # every surfaced id is prefixed with the node rid (the provenance link)
    for h in hits:
        assert h["id"].startswith("node99-")
        assert owner_rid_of(h["id"]) == "node99"
    assert surfaced == [h["id"] for h in hits]
    # the photosynthesis docs should outrank the mitochondria doc
    assert hits[0]["docid"] in ("d1", "d3")


def test_search_backend_none_returns_empty():
    tools = make_corpus_tools({"search_backend": "none"}, node_rid="n", surfaced_ids=[])
    assert tools["search"]("anything") == []
