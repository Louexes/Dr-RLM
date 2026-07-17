"""Phase 1 — turn one seed question into ~2-3 deliberate discovery queries.

The corpus crawl is *not* the agent: it needs far fewer queries than the agent's ~20 runtime
searches (DATA_PREP_PLAN §5/§7). The single biggest cost/time lever is keeping this small. We use
**heuristic** decomposition (no LLM on the critical path -> no GPU/API dependency, fully
reproducible): the raw question is already a strong DDG query, and for the long research prompts
we add a concise first-clause query plus an entity/keyword query so BM25 has the right *documents*
regardless of the agent's eventual phrasing. The harvest-loop (§7.1) re-crawls any thin question
with more queries later, so under-generating here is cheap and over-generating is not.
"""
from __future__ import annotations

import re
from typing import List, Optional

_WS = re.compile(r"\s+")
_ENTITY = re.compile(r"\b([A-Z][A-Za-z0-9.\-]+(?:\s+[A-Z][A-Za-z0-9.\-]+){0,4})\b")
_QUOTED = re.compile(r"[\"“”']([^\"“”']{4,80})[\"“”']")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_SENT = re.compile(r"(?<=[.?!])\s+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")

_STOP = set(
    """a an the of to in on for and or is are was were be been being this that these those with
    as at by from into about over under between during what which who whom whose how why when where
    does do did can could should would may might will shall i you he she it we they me my your our
    their his her its please tell teach give explain describe provide list summarize compare across
    using based all latest progress such like e.g eg i.e ie according report response include""".split()
)


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def _keyword_query(text: str, max_terms: int = 10) -> str:
    seen, terms = set(), []
    for w in _WORD.findall(text):
        lw = w.lower()
        if lw in _STOP or len(lw) <= 2:
            continue
        if lw in seen:
            continue
        seen.add(lw)
        terms.append(w)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def _entity_query(text: str, max_terms: int = 8) -> str:
    ents: List[str] = []
    ents += [m.strip() for m in _QUOTED.findall(text)]
    ents += [m.strip() for m in _ENTITY.findall(text) if len(m) > 3]
    ents += _YEAR.findall(text)
    seen, out = set(), []
    for e in ents:
        le = e.lower()
        if le in seen or le in _STOP:
            continue
        seen.add(le)
        out.append(e)
    joined = " ".join(out)
    return joined[:200]


def make_queries(question: str, qtype: Optional[str] = None, n: int = 3, long_threshold: int = 180) -> List[str]:
    """Return up to ``n`` deduped discovery queries for one seed question."""
    q = _norm(question)
    if not q:
        return []
    cands: List[str] = []
    if len(q) <= long_threshold:
        # Short factual question: the raw natural-language question is empirically the single
        # best DDG query (validated: it nailed cephalexin/Merton, where a stripped keyword
        # variant returned spam). One query also keeps short-question volume minimal (§7.1).
        cands.append(q[:256])
    else:
        first = _SENT.split(q)[0].strip()
        cands.append((first or q)[:200])
        ent = _entity_query(q)
        if ent:
            cands.append(ent)
        kw = _keyword_query(q)
        if kw:
            cands.append(kw)
    # dedup (case-insensitive), preserve order, cap at n
    seen, out = set(), []
    for c in cands:
        c = _norm(c)
        key = c.lower()
        if not c or key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= n:
            break
    return out
