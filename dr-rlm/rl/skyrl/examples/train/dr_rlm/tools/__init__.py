"""Unified DR-RLM tool facade.

ONE ``search()`` / ``get_doc()`` surface, backend-switchable between OFFLINE (frozen
corpus) and ONLINE (live web) by the single ``search_backend`` flag. Crucially, the
provenance id mint (``"{node_rid}-{n}"``) and the evidence-ledger write happen ONLY in
``make_tools`` — providers supply raw hits and never touch ids/ledger — so the RER credit
graph is byte-identical across backends, in RL and inference both.
"""

from .provider import ToolProvider, CorpusToolProvider, get_provider, make_tools, owner_rid_of

__all__ = ["ToolProvider", "CorpusToolProvider", "get_provider", "make_tools", "owner_rid_of"]
