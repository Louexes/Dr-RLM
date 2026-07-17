"""Local bm25s search tool for DR-Tulu — retrieves from the DR-RLM frozen corpus.

Why this exists: the controlled comparison (DR-Tulu ReAct vs DR-RLM REPL) must hold
retrieval IDENTICAL — same corpus, same BM25 algorithm, same tokenizer — so any score
gap is the harness, not the index. DR-Tulu's stock search is MCP/Serper (web) or
pyserini/Lucene (Java). The RLM retrieves from the frozen corpus IN-PROCESS via
``rl/skyrl/examples/train/dr_rlm/corpus_search.py`` ``_Bm25sBackend``. This tool reuses
that *exact* backend object (loaded by file path, no package import), so both harnesses
share byte-identical retrieval. No MCP server is launched: we override ``__call__`` and
never touch the MCP transport (MCPSearchTool.__init__ is fully lazy).

The model sees the same Serper-shaped observation it always did (``Title/URL/Snippet``
via ``Document.stringify``); the corpus ``id`` is NOT surfaced — DR-Tulu attributes
citations by its own native (URL) scheme, which is a legitimate harness difference.

See dr-rlm/docs/patches_to_deps.md (2026-06-28).
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Dict, List, Optional, Union

from .data_types import Document, DocumentToolOutput, ToolInput, ToolOutput
from .mcp_tools import MCPBrowseTool, MCPSearchTool

# The single source of truth for DR-RLM/RL retrieval. Loaded by file path so we do not
# import the dr_rlm package (its package-relative imports are lazy / training-only).
_CORPUS_SEARCH_PATH = (
    "/gpfs/home5/lgehringer/Dr-RLM/dr-rlm/rl/skyrl/examples/train/dr_rlm/corpus_search.py"
)
_get_backend = None


def _load_get_backend():
    global _get_backend
    if _get_backend is None:
        spec = importlib.util.spec_from_file_location(
            "dr_rlm_corpus_search", _CORPUS_SEARCH_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _get_backend = mod.get_backend
    return _get_backend


class LocalBm25sSearchTool(MCPSearchTool):
    """Search the frozen corpus via the RLM's in-process ``_Bm25sBackend``.

    Drop-in for ``SerperSearchTool`` in the auto_search workflow: same constructor
    surface (``tool_parser``, ``number_documents_to_search``, ``timeout``, ``name``),
    plus ``search_index_path`` / ``search_corpus_path``. MCP kwargs the workflow may
    pass (``transport_type``/``mcp_executable``/``mcp_port``/``mcp_host``) are accepted
    and ignored — no server is used.
    """

    def __init__(
        self,
        *,
        search_index_path: str,
        search_corpus_path: Optional[str] = None,
        tool_parser=None,
        number_documents_to_search: int = 8,
        timeout: int = 60,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        # We never open an MCP transport; drop server kwargs so they don't confuse init.
        for _k in ("transport_type", "mcp_executable", "mcp_port", "mcp_host"):
            kwargs.pop(_k, None)
        super().__init__(
            tool_parser=tool_parser,
            number_documents_to_search=number_documents_to_search,
            timeout=timeout,
            name=name or "snippet_search",
            description=description
            or "Search the local research corpus; returns the top documents (title, url, text).",
            **kwargs,
        )
        self._search_index_path = search_index_path
        self._search_corpus_path = search_corpus_path
        self._backend = None  # lazy: load the bm25s index on first call

    # -- backend -----------------------------------------------------------------
    def _ensure_backend(self):
        if self._backend is None:
            get_backend = _load_get_backend()
            self._backend = get_backend(
                {
                    "search_backend": "bm25s",
                    "search_corpus_path": self._search_corpus_path,
                    "search_index_path": self._search_index_path,
                }
            )
            if self._backend is None:
                raise RuntimeError(
                    f"bm25s backend failed to init for index {self._search_index_path!r}"
                )
        return self._backend

    # -- abstract-method stubs (required to instantiate; __call__ is overridden so
    #    these are never reached during a real search) ---------------------------
    def get_mcp_tool_name(self) -> str:  # pragma: no cover
        return "local_bm25s_search"

    def get_mcp_params(self, tool_call_info) -> Dict[str, Any]:  # pragma: no cover
        return {"query": tool_call_info.content, "k": self.number_documents_to_search}

    def extract_documents(self, raw_output: Dict[str, Any]) -> List[Document]:  # pragma: no cover
        return list(raw_output.get("documents", []))

    # -- the real entrypoint -----------------------------------------------------
    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        call_id = self._generate_call_id()
        start = time.time()

        tool_call_info = self.preprocess_input(tool_input)
        if not tool_call_info:
            return self._create_error_output(
                "No valid query found in tool call.", call_id, time.time() - start
            )

        # k: tool default, overridable per-call via num_results (mirrors SerperSearchTool).
        k = self.number_documents_to_search
        if tool_call_info.parameters and "num_results" in tool_call_info.parameters:
            try:
                k = int(tool_call_info.parameters["num_results"])
            except (ValueError, TypeError):
                pass

        backend = self._ensure_backend()
        hits = backend.search(tool_call_info.content, k) or []

        documents: List[Document] = []
        for h in hits:
            contents = h.get("snippet") or ""
            # corpus rows have no title field; synthesize a heading from the first
            # sentence (the crawled contents lead with a title-like line) for display only.
            title = (contents.split(". ", 1)[0] or h.get("url") or "Document").strip()[:120]
            documents.append(
                Document(
                    id=str(h.get("docid")),
                    title=title or "Document",
                    url=h.get("url") or "",
                    snippet=contents,
                    score=h.get("score"),
                )
            )

        if not documents:
            return self._create_error_output(
                "No results found for the query.",
                call_id,
                time.time() - start,
                raw_output={"hits": hits},
            )

        content = "\n\n".join(doc.stringify() for doc in documents)
        return DocumentToolOutput(
            tool_name=self.name,
            output=content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start,
            call_id=call_id,
            raw_output={"hits": hits},
            documents=documents,
            query=tool_call_info.content,
            input_params=(
                tool_call_info.parameters if tool_call_info.parameters else None
            ),
        )


class LocalBm25sBrowseTool(MCPBrowseTool):
    """Browse a URL by returning its full text FROM THE FROZEN CORPUS (no live fetch).

    Mirrors the RLM's ``get_doc``: under a frozen passage corpus there is no live page to
    fetch — "browsing" a URL returns the crawled passage(s) for it. Concatenates all corpus
    passages sharing that URL (a URL may be chunked into several rows) and reconstructs the
    page text. Shares the SAME ``_Bm25sBackend`` instance as the search tool (get_backend's
    process-wide cache), so the index is loaded once. Gives DR-Tulu a working browse tool so
    its ReAct policy (trained with browse) is not handicapped vs an inert NoBrowseTool.
    """

    def __init__(
        self,
        *,
        search_index_path: str,
        search_corpus_path: Optional[str] = None,
        tool_parser=None,
        max_pages_to_fetch: int = 5,
        timeout: int = 120,
        context_chars: int = 2000,
        name: Optional[str] = None,
        **kwargs,
    ):
        for _k in ("transport_type", "mcp_executable", "mcp_port", "mcp_host"):
            kwargs.pop(_k, None)
        super().__init__(
            tool_parser=tool_parser,
            max_pages_to_fetch=max_pages_to_fetch,
            timeout=timeout,
            context_chars=context_chars,
            name=name or "browse_webpage",
            **kwargs,
        )
        self._search_index_path = search_index_path
        self._search_corpus_path = search_corpus_path
        self._backend = None
        self._url2docids: Optional[Dict[str, List[str]]] = None

    def _ensure_index(self):
        if self._backend is None:
            get_backend = _load_get_backend()
            self._backend = get_backend(
                {
                    "search_backend": "bm25s",
                    "search_corpus_path": self._search_corpus_path,
                    "search_index_path": self._search_index_path,
                }
            )
            if self._backend is None:
                raise RuntimeError("bm25s backend failed to init for browse")
        if self._url2docids is None:
            u2d: Dict[str, List[str]] = {}
            for d in (getattr(self._backend._retriever, "corpus", None) or []):
                if isinstance(d, dict):
                    u, did = d.get("url"), d.get("id")
                    if u and did is not None:
                        u2d.setdefault(str(u).strip(), []).append(str(did))
            self._url2docids = u2d
        return self._backend

    def _text_for_url(self, url: str) -> Optional[str]:
        docids = (self._url2docids or {}).get(str(url).strip(), [])
        if not docids:
            return None
        parts = [self._backend.get_text(did) or "" for did in docids]
        text = "\n".join(p for p in parts if p)
        return text or None

    # abstract-method stubs (never reached — __call__ is overridden)
    def get_mcp_tool_name(self) -> str:  # pragma: no cover
        return "local_bm25s_browse"

    def get_mcp_params(self, tool_call_info) -> Dict[str, Any]:  # pragma: no cover
        return {"url": tool_call_info.content}

    def _extract_raw_content_from_response(self, raw_output):  # pragma: no cover
        return raw_output.get("text") if isinstance(raw_output, dict) else None

    def _extract_metadata_from_document(self, document, raw_output):  # pragma: no cover
        return (None, None)

    async def __call__(
        self, tool_input: Union[str, Dict[str, Any], ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        call_id = self._generate_call_id()
        start = time.time()
        self._ensure_index()

        input_documents: List[Document] = []
        if isinstance(tool_input, dict):
            url = (tool_input.get("url") or "").strip()
            if url:
                input_documents = [Document(title="", snippet="", url=url)]
        elif isinstance(tool_input, str):
            tci = self.parse_call(tool_input)
            if tci:
                input_documents = [Document(title="", snippet="", url=tci.content.strip())]
        elif isinstance(tool_input, (ToolOutput, DocumentToolOutput)):
            docs = getattr(tool_input, "documents", None) or []
            input_documents = [
                Document(title=d.title, snippet=d.snippet or "", url=d.url)
                for d in docs
                if d.url
            ]

        if not input_documents:
            return self._create_error_output(
                "No valid URL found in tool call.", call_id, time.time() - start
            )

        enriched: List[Document] = []
        for doc in input_documents[: self.max_pages_to_fetch]:
            text = self._text_for_url(doc.url)
            ed = Document(
                id=doc.id,
                title=doc.title,
                snippet=doc.snippet,
                url=doc.url,
                text=text,
                score=doc.score,
            )
            if text is None:
                ed.error = "URL not found in local corpus."
            enriched.append(ed)

        parts = []
        for doc in enriched:
            parts.append(
                doc.stringify(
                    use_localized_snippets=self.use_localized_snippets,
                    context_chars=self.context_chars,
                    fallback_message=(doc.error or None),
                )
            )
        content = "\n\n".join(parts)
        return DocumentToolOutput(
            tool_name=self.name,
            output=content if self.create_string_output else "",
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start,
            call_id=call_id,
            raw_output=None,
            documents=enriched,
            query=getattr(tool_input, "query", None),
        )
