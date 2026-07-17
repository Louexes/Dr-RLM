"""Online live-web provider: Serper search + Jina browse via the dr_agent MCP tools.

Returns raw hits ONLY — id minting + ledger writes live in ``tools/provider.make_tools``,
so web provenance is byte-identical to the offline corpus. This is exactly what makes the
``search_backend="web"`` switch safe for RER credit. The tool's own random per-call
``call_id`` is DISCARDED; ``make_tools`` assigns the real ``{node_rid}-{n}`` id.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import Any, Dict, List, Optional

_DR_TULU_AGENT = "/gpfs/home5/lgehringer/Dr-RLM/dr-tulu/agent"

# Provider/quota error markers that must NEVER reach the agent's context: a browse backend
# that returns its OWN error text as if it were page content makes the orchestrator read
# "InsufficientBalanceError"/"Timeout" and BAIL on synthesis (observed: a root wrote an apology
# instead of merging its children's findings). We scrub tool output matching these and return
# empty instead, so a dead backend looks like "no hit", not a poison pill.
#
# UNAMBIGUOUS markers — verbatim API error signatures that will not appear in real page/snippet
# text. Used to scrub SEARCH snippets (where a false drop would lose real grounding), so the
# bar is high. (Browse is scrubbed with the same set, but it is disabled when out of quota.)
_POISON_MARKERS = (
    "insufficientbalanceerror", "balance not enough to run this query", "please recharge",
    'code":402', '"code":401', '"code":403', '"code":429',
    "api request failed with status", "mcp tool", "insufficient credit",
)


def _is_poison(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _POISON_MARKERS)


class _BackgroundLoop:
    """Sync->async bridge: a dedicated daemon event loop so the sync REPL can drive the
    async dr_agent MCP tools."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 120.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)


class WebToolProvider:
    """Live-web retrieval. ``search`` -> Serper; ``get_text`` -> Jina browse. Mints NO ids
    and writes NO ledger (``make_tools`` owns provenance)."""

    def __init__(self, payload: dict):
        agent_path = payload.get("dr_agent_path", _DR_TULU_AGENT)
        if agent_path not in sys.path:
            sys.path.insert(0, agent_path)
        from dr_agent.tool_interface import SerperSearchTool
        from dr_agent.tool_interface.mcp_tools import JinaBrowseTool

        port = int(payload.get("web_mcp_port", 8030))
        top_k = int(payload.get("search_top_k", 10))
        self._search_tool = SerperSearchTool(mcp_port=port, number_documents_to_search=top_k)
        self._browse_tool = JinaBrowseTool(mcp_port=port)
        self._loop = _BackgroundLoop()
        # Browse off => never call Jina (set DR_RLM_DISABLE_BROWSE=1 when the browse backend is
        # out of quota). The agent then grounds on search snippets only — which carry the
        # provenance ids credit needs — instead of seeing browse error text and giving up.
        self._browse_disabled = os.environ.get("DR_RLM_DISABLE_BROWSE", "") not in ("", "0", "false")

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        out = self._loop.run(self._search_tool(str(query)[:2048]))
        docs = getattr(out, "documents", None) or []
        # out.call_id is random-per-call and is DISCARDED — make_tools mints the real
        # {node_rid}-{n} id, keeping web provenance byte-identical to the corpus.
        hits = []
        for d in docs:
            snip = d.stringify()
            # Scrub only when a backend is known-dead (browse disabled): a live run keeps the
            # ORIGINAL behavior (no filtering) so this change is inert by default.
            if self._browse_disabled and _is_poison(snip):
                continue
            hits.append({"docid": d.url, "snippet": snip, "url": d.url, "score": getattr(d, "score", None)})
        return hits

    def get_text(self, url_or_docid: str) -> Optional[str]:
        if self._browse_disabled:   # browse backend known-dead -> skip Jina entirely (gated; default off)
            return ""
        out = self._loop.run(self._browse_tool(str(url_or_docid)))
        docs = getattr(out, "documents", None) or []
        return (docs[0].stringify() if docs else "") or ""


_WEB_CACHE: Dict[int, "WebToolProvider"] = {}
_WEB_LOCK = threading.Lock()


def get_web_provider(payload: dict) -> "WebToolProvider":
    """Process-wide cached provider keyed by ``web_mcp_port`` so the root + every sub-agent
    share one event loop + tool client (mirrors the offline backend cache)."""
    port = int(payload.get("web_mcp_port", 8030))
    with _WEB_LOCK:
        if port not in _WEB_CACHE:
            _WEB_CACHE[port] = WebToolProvider(payload)
        return _WEB_CACHE[port]
