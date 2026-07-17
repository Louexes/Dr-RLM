"""Phase 3 — free page *reading* via httpx + trafilatura, chunked into passages.

Reading answers "the full text of one page" — the Jina-Reader job, done for $0 and, crucially,
*without a browser*. Crawl4AI/Playwright was the plan's first pick, but headless Chromium at
crawl scale on an HPC node is fragile (browser binaries, sandbox, zombie procs). trafilatura's
main-content extractor gets equivalent clean text for the article/wiki/news/scientific pages a
research corpus is made of, as a pure-Python dependency — far more robust to run 16-way sharded.
JS-heavy pages degrade gracefully to their discovery snippet (which we always keep).

We fetch with httpx (explicit timeout, redirect-following, browser-like UA — things
``trafilatura.fetch_url`` does not expose) and extract with trafilatura. Per-URL disk cache so a
URL is fetched **at most once ever**, across questions and across shards.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# Extensions we never try to read as HTML (snippet tier still covers them).
_SKIP_EXT = (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".zip", ".gz",
             ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3", ".csv")
_PARA = re.compile(r"\n\s*\n")


def _uhash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def chunk_passages(text: str, target: int = 1800, max_passages: int = 12, min_chars: int = 200) -> List[str]:
    """Greedy paragraph-pack into ~``target``-char passages (BM25 doc granularity).

    Paragraphs are accumulated until the running passage would exceed ``target``; a single
    over-long paragraph is hard-split. Caps at ``max_passages`` so one giant page can't dominate
    the corpus. Passages shorter than ``min_chars`` (after the cap) are dropped unless it's the
    only one.
    """
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in _PARA.split(text) if p.strip()]
    passages: List[str] = []
    buf = ""
    for p in paras:
        while len(p) > target * 1.5:  # hard-split a monster paragraph
            cut = p[: target]
            if buf:
                passages.append(buf.strip())
                buf = ""
            passages.append(cut.strip())
            p = p[target:]
            if len(passages) >= max_passages:
                break
        if len(passages) >= max_passages:
            break
        if buf and len(buf) + 1 + len(p) > target:
            passages.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n" + p).strip() if buf else p
    if buf and len(passages) < max_passages:
        passages.append(buf.strip())
    passages = passages[:max_passages]
    kept = [p for p in passages if len(p) >= min_chars]
    return kept or (passages[:1] if passages else [])


class Reader:
    """httpx fetch + trafilatura extract + passage chunking, with a per-URL disk cache.

    A cached entry stores ``{url, ok, text, passages, ts}``. ``ok=False`` is cached too (so a
    dead/blocked URL is not retried every run); pass ``retry_failed=True`` to override.
    """

    def __init__(
        self,
        cache_dir: str,
        timeout: float = 20.0,
        max_chars: int = 24000,
        target_passage: int = 1800,
        max_passages: int = 12,
        min_interval: float = 0.0,
        retry_failed: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.max_chars = int(max_chars)
        self.target_passage = int(target_passage)
        self.max_passages = int(max_passages)
        self.min_interval = float(min_interval)
        self.retry_failed = bool(retry_failed)
        self._last = 0.0
        self._client = None
        self.stats = {"cache_hits": 0, "live": 0, "ok": 0, "fail": 0, "skip": 0}

    def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                follow_redirects=True,
                timeout=self.timeout,
                headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            )
        return self._client

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()

    @staticmethod
    def _skippable(url: str) -> bool:
        u = url.split("?", 1)[0].lower()
        return u.endswith(_SKIP_EXT)

    def _extract(self, url: str) -> Optional[str]:
        import trafilatura

        try:
            self._pace()
            resp = self._http().get(url)
            ctype = resp.headers.get("content-type", "")
            if resp.status_code != 200 or "html" not in ctype.lower():
                return None
            html = resp.text
        except Exception:
            return None
        if not html:
            return None
        try:
            text = trafilatura.extract(
                html, include_links=False, include_comments=False,
                include_tables=True, favor_recall=True, url=url,
            )
        except Exception:
            text = None
        if not text:
            return None
        text = text.strip()
        return text[: self.max_chars] if self.max_chars else text

    def read(self, url: str) -> Dict[str, Any]:
        """Return ``{url, ok, text, passages}`` (cached). ``passages`` is the chunked full text."""
        url = (url or "").strip()
        if not url:
            return {"url": url, "ok": False, "text": "", "passages": []}
        cpath = self.cache_dir / f"{_uhash(url)}.json"
        if cpath.exists():
            try:
                obj = json.loads(cpath.read_text())
                if obj.get("ok") or not self.retry_failed:
                    self.stats["cache_hits"] += 1
                    return obj
            except Exception:
                pass
        if self._skippable(url):
            self.stats["skip"] += 1
            obj = {"url": url, "ok": False, "text": "", "passages": [], "reason": "skip_ext"}
            _atomic_write_json(cpath, obj)
            return obj
        self.stats["live"] += 1
        text = self._extract(url)
        if not text:
            self.stats["fail"] += 1
            obj = {"url": url, "ok": False, "text": "", "passages": []}
            _atomic_write_json(cpath, obj)
            return obj
        passages = chunk_passages(text, self.target_passage, self.max_passages)
        self.stats["ok"] += 1
        obj = {"url": url, "ok": True, "text": text, "passages": passages, "ts": int(time.time())}
        _atomic_write_json(cpath, obj)
        return obj

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = None
