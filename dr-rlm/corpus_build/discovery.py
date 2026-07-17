"""Phase 2 — free web *discovery* via ddgs, with retry/backoff + per-query disk cache.

Discovery answers "which pages are relevant to this query" and returns
``{title, url, snippet}`` rows — the Serper job, done for $0.

Backend choice (validated 2026-06-26 from the Snellius login node): ``backend="auto"``
returns on-topic results (springer / researchgate / wikipedia) where the ddgs *default*
html/lite upstream returned SEO spam ("Best Document Management Software" for a cephalexin
query). ``google`` is a clean fallback; ``duckduckgo``/``mojeek``/``mullvad_brave`` upstreams
were dead/blocked from the cluster IP. So we try ``auto -> google -> bing`` and keep the first
non-empty.

Caching is per ``(query, k)`` so a re-run / a second shard never re-issues a query that already
landed — this is the single most important resumability property (DDG rate limit is the scarce
resource, not CPU).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKENDS = ("auto", "google", "bing")  # in order; keep first non-empty


def _qhash(query: str, k: int) -> str:
    return hashlib.sha1(f"{query}\x00{k}".encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


class Discovery:
    """ddgs discovery with disk cache, polite pacing, and backend fallback.

    Parameters
    ----------
    cache_dir : per-(query,k) JSON cache. Shared across shards is fine (atomic writes).
    max_results : results requested per query (top-k).
    min_interval : seconds between live queries in THIS process (>=2.0 => <=30/min/IP).
    max_retries : retries per backend on transient error.
    """

    def __init__(
        self,
        cache_dir: str,
        max_results: int = 10,
        min_interval: float = 2.0,
        max_retries: int = 2,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_results = int(max_results)
        self.min_interval = float(min_interval)
        self.max_retries = int(max_retries)
        self._last = 0.0
        self._ddgs = None
        self.stats = {"cache_hits": 0, "live": 0, "empty": 0, "errors": 0}

    # -- internals -----------------------------------------------------------
    def _client(self):
        if self._ddgs is None:
            from ddgs import DDGS

            self._ddgs = DDGS()
        return self._ddgs

    def _reset_client(self) -> None:
        try:
            if self._ddgs is not None and hasattr(self._ddgs, "__exit__"):
                self._ddgs.__exit__(None, None, None)
        except Exception:
            pass
        self._ddgs = None

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt + random.uniform(0, 0.4))
        self._last = time.time()

    @staticmethod
    def _normalize(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for r in rows or []:
            url = (r.get("href") or r.get("url") or "").strip()
            if not url:
                continue
            out.append(
                {
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                }
            )
        return out

    def _live(self, query: str) -> List[Dict[str, str]]:
        for be in _BACKENDS:
            for attempt in range(self.max_retries + 1):
                try:
                    self._pace()
                    rows = list(
                        self._client().text(query, max_results=self.max_results, backend=be)
                    )
                    hits = self._normalize(rows)
                    if hits:
                        return hits
                    break  # backend answered but empty -> try next backend
                except Exception:
                    self.stats["errors"] += 1
                    self._reset_client()
                    time.sleep(min(8.0, 1.5 * (2 ** attempt)) + random.uniform(0, 0.6))
        return []

    # -- public --------------------------------------------------------------
    def search(self, query: str) -> List[Dict[str, str]]:
        query = (query or "").strip()
        if not query:
            return []
        cpath = self.cache_dir / f"{_qhash(query, self.max_results)}.json"
        if cpath.exists():
            try:
                obj = json.loads(cpath.read_text())
                self.stats["cache_hits"] += 1
                return obj.get("hits", [])
            except Exception:
                pass  # corrupt cache -> re-fetch
        hits = self._live(query)
        self.stats["live"] += 1
        if not hits:
            self.stats["empty"] += 1
        _atomic_write_json(cpath, {"query": query, "k": self.max_results, "hits": hits})
        return hits

    def close(self) -> None:
        self._reset_client()
