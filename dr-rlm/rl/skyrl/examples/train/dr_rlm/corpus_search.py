"""Offline corpus ``search()`` / ``get_doc()`` REPL tools for DR-RLM.

Held constant across every arm of the controlled comparison: both the flat (L1)
and recursive (L2-L4) policies retrieve from the SAME frozen corpus with the SAME
retriever. Mirrors DR Tulu's offline wii setup (BM25 / Qwen3-Embed FAISS over a
``corpus.jsonl`` of ``{id, contents, url}`` rows), reusing the dr_agent searchers
in-process so no separate server is required.

The one DR-RLM-specific twist is **provenance-encoded snippet ids**: every hit a
node surfaces is given id ``"{node_rid}-{n}"``. Because node rids are dash-free
uuids, a citation ``<cite id="ab12cd-7">`` in the final report decodes to its owning
node via ``id.rsplit('-', 1)[0]``. That id scheme is exactly the citation/provenance
graph the RER credit assignment walks (see ``rer_reward.py``).

Backends (``search_backend``):
  * ``bm25`` / ``faiss`` — import the dr_agent ``BaseSearcher`` subclass in-process
    (real wii retrieval; bm25 needs Java/pyserini, faiss needs the embed model).
  * ``mcp_http``         — call a running dr_agent FastMCP ``local_search`` over HTTP.
  * ``local_jsonl``      — dependency-free token-overlap fallback over corpus.jsonl
    (for smoke tests / environments without the gated indices).
  * ``none``             — no retrieval (context-only debugging).
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# repo paths so we can import the dr_agent retrievers in-process
_DR_TULU_AGENT = "/gpfs/home5/lgehringer/Dr-RLM/dr-tulu/agent"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _BaseBackend:
    """Returns hits as ``{docid, snippet, url, score}`` and resolves docid->text."""

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_text(self, docid: str) -> Optional[str]:  # pragma: no cover - interface
        return None


class _DrAgentBackend(_BaseBackend):
    """Wraps dr_agent BM25Searcher / FaissSearcher in-process."""

    def __init__(self, payload: dict):
        if _DR_TULU_AGENT not in sys.path:
            sys.path.insert(0, _DR_TULU_AGENT)
        from types import SimpleNamespace

        backend = payload["search_backend"]
        if backend == "bm25":
            from dr_agent.mcp_backend.local.search.bm25_retriever import BM25Searcher as Cls

            args = SimpleNamespace(
                dataset_name=payload["search_corpus_path"],
                index_path=payload["search_index_path"],
            )
        elif backend == "faiss":
            from dr_agent.mcp_backend.local.search.faiss_retriever import FaissSearcher as Cls

            args = SimpleNamespace(
                dataset_name=payload["search_corpus_path"],
                index_path=payload["search_index_path"],
                model_name=payload["search_embed_model"],
                normalize=True,
                pooling="eos",
                torch_dtype="float16",
                task_prefix="Instruct: Given a query, retrieve relevant passages.\nQuery: ",
                max_length=8192,
            )
        else:  # pragma: no cover
            raise ValueError(backend)
        self._searcher = Cls(args)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        resp = self._searcher.search(query, k=k)
        return list(resp.get("results", []))

    def get_text(self, docid: str) -> Optional[str]:
        meta = getattr(self._searcher, "docid_to_metadata", {}).get(str(docid))
        return meta.get("snippet") if meta else None


class _McpHttpBackend(_BaseBackend):
    """Calls a running dr_agent FastMCP ``local_search`` over StreamableHTTP.

    Bridges the async MCP client to the sync REPL via a dedicated background loop.
    """

    def __init__(self, payload: dict):
        import asyncio

        self._endpoint = payload["search_endpoint"]
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro):
        import asyncio

        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=120)

    async def _call(self, tool: str, params: dict):
        from fastmcp import Client  # type: ignore

        async with Client(self._endpoint) as client:
            res = await client.call_tool(tool, params)
        return res.data if hasattr(res, "data") else res

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        data = self._run(self._call("local_search", {"query": query[:2048], "num_results": k}))
        if isinstance(data, dict):
            return list(data.get("results", []))
        return []

    def get_text(self, docid: str) -> Optional[str]:
        return None


class _LocalJsonlBackend(_BaseBackend):
    """Dependency-free TF-IDF-lite retriever over a ``corpus.jsonl`` of {id,contents,url}."""

    def __init__(self, payload: dict):
        import json

        path = payload["search_corpus_path"]
        self._docs: List[Dict[str, Any]] = []
        self._df: Dict[str, int] = {}
        if not os.path.exists(path):
            logger.warning(f"[dr_rlm.search] local_jsonl corpus not found at {path}; search() will return nothing")
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                contents = str(row.get("contents", ""))
                toks = _tok(contents)
                tf: Dict[str, int] = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                for t in tf:
                    self._df[t] = self._df.get(t, 0) + 1
                self._docs.append(
                    {"id": str(row.get("id")), "contents": contents, "url": row.get("url", ""), "tf": tf, "len": len(toks)}
                )
        self._N = max(1, len(self._docs))
        logger.info(f"[dr_rlm.search] local_jsonl loaded {self._N} docs from {path}")

    def _idf(self, t: str) -> float:
        return math.log((self._N + 1) / (self._df.get(t, 0) + 1)) + 1.0

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q = _tok(query)
        if not q or not self._docs:
            return []
        scored = []
        for d in self._docs:
            s = sum(d["tf"].get(t, 0) * self._idf(t) for t in q) / (1.0 + math.log1p(d["len"]))
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, d in scored[:k]:
            out.append({"docid": d["id"], "snippet": d["contents"], "url": d["url"], "score": float(s)})
        return out

    def get_text(self, docid: str) -> Optional[str]:
        for d in self._docs:
            if d["id"] == str(docid):
                return d["contents"]
        return None


class _Bm25sBackend(_BaseBackend):
    """Fast pure-Python BM25 over a frozen ``corpus.jsonl`` via the ``bm25s`` library.

    Built offline by ``corpus_build/index_bm25.py``; ``search_index_path`` points at the saved
    index dir (which also stores the ``{id, contents, url}`` rows, so no separate corpus file is
    needed). Returns ``{docid, snippet, url, score}`` and resolves ``docid -> full contents`` for
    get_doc. The tokenizer (english stopwords + Snowball stemmer) MUST match index_bm25.py.

    Chosen over pyserini for the self-crawled frozen corpus: same BM25 algorithm (controlled-
    comparison invariant holds) with no Java / no gated index format / ms memory-mapped load.
    """

    def __init__(self, payload: dict):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self._stemmer = Stemmer.Stemmer("english")
        self._retriever = bm25s.BM25.load(payload["search_index_path"], load_corpus=True)
        self._id2text: Dict[str, str] = {}
        for d in (getattr(self._retriever, "corpus", None) or []):
            if isinstance(d, dict) and d.get("id") is not None:
                self._id2text[str(d["id"])] = d.get("contents", "")
        self._n = len(self._id2text)
        logger.info(f"[dr_rlm.search] bm25s loaded {self._n} docs from {payload['search_index_path']}")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if self._n == 0:
            return []
        toks = self._bm25s.tokenize(str(query), stopwords="en", stemmer=self._stemmer, show_progress=False)
        docs, scores = self._retriever.retrieve(toks, k=min(int(k), self._n), show_progress=False)
        out: List[Dict[str, Any]] = []
        ncols = docs.shape[1] if hasattr(docs, "shape") else len(docs[0])
        for i in range(ncols):
            d = docs[0, i]
            if not isinstance(d, dict):
                continue
            out.append({
                "docid": str(d.get("id")), "snippet": d.get("contents", ""),
                "url": d.get("url", ""), "score": float(scores[0, i]),
            })
        return out

    def get_text(self, docid: str) -> Optional[str]:
        return self._id2text.get(str(docid))


_BACKEND_CACHE: Dict[str, _BaseBackend] = {}
_CACHE_LOCK = threading.Lock()


def get_backend(payload: dict) -> Optional[_BaseBackend]:
    """Process-wide cached backend (one searcher/index per worker, keyed by config)."""
    backend = payload.get("search_backend", "none")
    if backend == "none":
        return None
    key = f"{backend}|{payload.get('search_corpus_path')}|{payload.get('search_index_path')}|{payload.get('search_endpoint')}"
    with _CACHE_LOCK:
        if key in _BACKEND_CACHE:
            return _BACKEND_CACHE[key]
        try:
            if backend in ("bm25", "faiss"):
                be: _BaseBackend = _DrAgentBackend(payload)
            elif backend == "bm25s":
                be = _Bm25sBackend(payload)
            elif backend == "mcp_http":
                be = _McpHttpBackend(payload)
            elif backend == "local_jsonl":
                be = _LocalJsonlBackend(payload)
            else:
                raise ValueError(f"unknown search_backend={backend!r}")
        except Exception as e:
            logger.warning(f"[dr_rlm.search] backend {backend!r} init failed ({e}); falling back to local_jsonl")
            be = _LocalJsonlBackend(payload)
        _BACKEND_CACHE[key] = be
        return be


# ---------------------------------------------------------------------------
# REPL tool factory
# ---------------------------------------------------------------------------


def make_corpus_tools(
    payload: dict,
    node_rid: str,
    surfaced_ids: List[str],
    ledger: Optional[Dict[str, Any]] = None,
) -> Dict[str, Callable]:
    """Back-compat alias. The tool factory is now provider-agnostic (offline frozen corpus
    OR online live web, switched by ``search_backend``) and lives in
    ``tools/provider.make_tools`` — the SINGLE id/ledger chokepoint. Imported lazily here to
    avoid an import cycle (``tools/provider`` imports ``get_backend``/``owner_rid_of`` from
    this module). Signature and behavior for the offline path are unchanged.
    """
    from .tools.provider import make_tools

    return make_tools(payload, node_rid, surfaced_ids, ledger)


def owner_rid_of(cite_id: str) -> str:
    """Decode the node that surfaced a snippet id: ``"ab12cd-7" -> "ab12cd"``."""
    return cite_id.rsplit("-", 1)[0]
