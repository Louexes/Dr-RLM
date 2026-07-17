"""The single DR-RLM tool chokepoint.

``make_tools`` mints provenance ids (``"{node_rid}-{n}"``) and writes the evidence
ledger ONCE, provider-agnostically, so the offline (frozen corpus) and online (live web)
backends are byte-identical for RER credit. A backend (``ToolProvider``) only supplies raw
hits ``{docid, snippet|contents, url, score}`` and resolves a docid/url to fuller text —
it never sees or mints an id. ``get_provider(payload)`` selects the backend from the single
``search_backend`` flag.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Protocol

# Offline backends + the owner-decode live in corpus_search. NOTE: corpus_search's
# make_corpus_tools imports make_tools LAZILY (at call time), so this module-level import
# does not create a cycle.
from ..corpus_search import get_backend, owner_rid_of  # noqa: F401  (owner_rid_of re-exported)
from .._run_guard import note_exception, hard_error_latched


class ToolProvider(Protocol):
    """A retrieval backend the facade drives. Returns raw hits (no ids minted) and resolves
    a docid/url to fuller text. Hits are ``{docid, snippet|contents, url, score}``."""

    def search(self, query: str, k: int) -> List[Dict[str, Any]]: ...  # pragma: no cover
    def get_text(self, doc_id: str) -> Optional[str]: ...  # pragma: no cover


class CorpusToolProvider:
    """Offline frozen-corpus provider (wraps the corpus_search backends, held constant
    across RL arms)."""

    def __init__(self, payload: dict):
        self._be = get_backend(payload)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        return self._be.search(query, k) if self._be is not None else []

    def get_text(self, doc_id: str) -> Optional[str]:
        return self._be.get_text(doc_id) if self._be is not None else None


def get_provider(payload: dict) -> Optional["ToolProvider"]:
    """Select the retrieval backend from the single ``search_backend`` flag.

    ``"web"`` -> live web (Serper/Jina); ``"none"`` -> no retrieval; anything else -> the
    frozen corpus. This is the ONE switch that flips online/offline in BOTH RL and inference.
    """
    backend = payload.get("search_backend", "mcp_http")
    if backend == "none":
        return None
    if backend == "web":
        from .web_provider import get_web_provider  # lazy: only import web deps when used
        return get_web_provider(payload)
    return CorpusToolProvider(payload)


def make_tools(
    payload: dict,
    node_rid: str,
    surfaced_ids: List[str],
    ledger: Optional[Dict[str, Any]] = None,
) -> Dict[str, Callable]:
    """Build ``{'search': fn, 'get_doc': fn}`` for one node's REPL — the ONLY place that
    mints provenance ids and writes the evidence ledger, so online and offline are
    byte-identical for RER.

    ``node_rid`` (the env's ``rlm_rollout_id``, a dash-free uuid) is baked into every
    surfaced snippet id (``"{node_rid}-{n}"``) so a ``<cite id="...">`` in the report decodes
    to its owning node via ``owner_rid_of``. ``surfaced_ids`` is the env's shared metrics
    list. ``ledger`` (per-tree, shared by reference) records ``ledger["snippets"][sid] =
    {node_rid, text, url, docid, query}`` under ``ledger["_lock"]`` — harness-tracked
    provenance the credit (``rer_reward``) walks rather than re-parsing prose.
    """
    # owner_rid_of uses rsplit('-', 1), so the rid MUST be dash-free. Fail loud at mint time
    # (an online backend that leaked a dashed id/url into node_rid would silently corrupt credit).
    assert "-" not in str(node_rid), f"node_rid must be dash-free (owner decode uses rsplit('-')): {node_rid!r}"
    provider = get_provider(payload)
    default_k = int(payload.get("search_top_k", 10))
    max_chars = int(payload.get("snippet_max_chars", 2000))
    # Per-TREE search-call budget (0 = unlimited). Bounds total Serper/web calls per item
    # across the whole recursion tree (restores legacy --max-tool-calls). Tracked in the
    # shared per-tree ledger so every node counts against the one budget.
    max_calls = int(payload.get("max_search_calls_per_tree", 0) or 0)
    counter = {"n": 0}
    lock = threading.Lock()

    def _budget_exhausted() -> bool:
        """Atomically consume one unit of the per-tree call budget; True if none remain."""
        if max_calls <= 0 or ledger is None:
            return False
        with ledger["_lock"]:
            used = int(ledger.get("_search_calls", 0))
            if used >= max_calls:
                return True
            ledger["_search_calls"] = used + 1
        return False

    def search(query: str, k: int = default_k) -> List[Dict[str, Any]]:
        """Search (corpus or web, per ``search_backend``). RETURNS a list of
        ``{id, text, url, score, docid}`` and is SILENT — it never prints. The model assigns
        the result and prints only the ids/snippets it will read or cite (context decoupling).
        Empty list on no-hits / error / disabled. CITE with ``<cite id="...">`` using a hit's ``id``."""
        if provider is None:
            return []
        # A hard failure (credits/auth/quota) latched anywhere in this tree -> stop calling the
        # backend; the driver will abort after this item. Saves the rest of the tree's API calls.
        if hard_error_latched():
            return []
        # Per-tree call budget: once spent, stop hitting the backend (the model finalizes).
        if _budget_exhausted():
            return []
        try:
            hits = provider.search(str(query)[:2048], int(k))
        except Exception as e:
            # Latch hard (non-transient) failures so the run aborts instead of silently
            # emitting ungrounded reports; transient errors just yield [] for this call.
            note_exception(e)
            return []
        results = []
        for h in hits:
            with lock:
                sid = f"{node_rid}-{counter['n']}"
                counter["n"] += 1
            surfaced_ids.append(sid)
            text = str(h.get("snippet") or h.get("contents") or "")
            if max_chars and len(text) > max_chars:
                text = text[:max_chars] + " …"
            url = h.get("url", "")
            score = h.get("score")
            if ledger is not None:
                with ledger["_lock"]:
                    ledger["snippets"][sid] = {
                        "node_rid": node_rid, "text": text, "url": url,
                        "docid": h.get("docid"), "query": str(query),
                    }
            results.append({"id": sid, "text": text, "url": url, "score": score, "docid": h.get("docid")})
        return results

    def get_doc(doc_id: Any) -> Dict[str, Any]:
        """RETURNS a dict ``{id, text, url}`` (SAME shape as a ``search`` hit, minus ``score``)
        with the fuller text for a surfaced snippet id. Symmetric with ``search`` ON PURPOSE:
        the model treats ``doc["text"]`` / ``doc["id"]`` the same whether the dict came from
        ``search`` or ``get_doc`` — previously ``get_doc`` returned a bare string, and the model
        reflexively indexed it (``doc["text"]``) -> ``TypeError: string indices must be integers``
        -> dead turns -> uncited report. A bad/unknown id yields ``text=""`` (never raises), so
        a wrong id degrades gracefully instead of crashing the cell. SILENT (prints nothing)."""
        sid = str(doc_id)
        empty = {"id": sid, "text": "", "url": ""}
        if provider is None or hard_error_latched():
            return empty
        try:
            if ledger is not None and sid in ledger.get("snippets", {}):
                rec = ledger["snippets"][sid]
                text = provider.get_text(rec.get("docid") or rec.get("url")) or rec.get("text") or ""
                return {"id": sid, "text": text, "url": rec.get("url", "")}
            return {"id": sid, "text": provider.get_text(sid) or "", "url": ""}
        except Exception as e:
            note_exception(e)
            return empty

    return {"search": search, "get_doc": get_doc}
