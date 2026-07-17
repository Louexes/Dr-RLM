"""The DR-RLM structured-answer contract: the ``answer`` dict the model submits, and
how it is normalized + rendered into the graded report.

The model submits upstream RLM's dict-based final answer, from inside the REPL:

    answer["content"] = "<report prose with inline <cite id="..."> tags>"
    answer["ready"]   = True

``normalize_answer`` canonicalizes that into ``{content, citations:[{id, claim}]}``: it
first normalizes the inline cite tags to the double-quoted form the DRB FACT extractor
requires, then recovers the cited (id, claim) pairs by SCRAPING those tags — dropping any id
that is not real retrieved evidence (the evidence-ledger backstop). It also accepts an
optional ``answer["citations"]`` list, but that is consumed only by the off-by-default
``ledger_support`` credit ablation; the default contract is content + ready. ``render_report``
turns the result into the report the FACT grader / user sees.

Why inline and not a reference list: DeepResearchBench FACT scores statement<->citation
*pairs*, so a trailing reference dump earns nothing — claim-aware inline is what moves the
metric. A citation whose claim span can't be located verbatim falls back to a References
block so the evidence is at least present (acknowledged weak for FACT).

Framework-free (str/dict in, str/dict out) so it is unit-testable without SkyRL/GPU.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .judge import all_cited_ids, extract_claims_and_corresponding_citation_ids

# Opening of an inline cite tag with an optional matching quote around the id; used to
# normalize `<cite id=X>` / `<cite id='X'>` -> `<cite id="X">` (the form DRB FACT requires).
_INLINE_CITE_OPEN = re.compile(r'<cite id=(["\']?)([^"\'\s>]+)\1')


def _quote_inline_cites(text: str) -> str:
    """Rewrite inline cite tags to the double-quoted id form, deterministically (idempotent
    for already-quoted tags). This makes the quoted-format guarantee independent of the model
    following the prompt — the DRB FACT extractor's regex requires ``id="..."``."""
    return _INLINE_CITE_OPEN.sub(r'<cite id="\2"', text)


# Self-closing cite marker: `<cite id="x"/>` or `<cite id="x" />` (claim sits BEFORE it).
_SELFCLOSE_CITE = re.compile(r'<cite id=(["\']?)([^"\'\s>]+)\1\s*/>')
# An opening cite tag (any id-quote style), used to walk markers left-to-right.
_OPEN_CITE = re.compile(r'<cite id=(["\']?)([^"\'\s>]+)\1\s*>')
# A well-formed closed cite immediately following an opening tag: claim text then </cite>.
_WELLFORMED_TAIL = re.compile(r'\s*[^<\s][^<]*</cite>')


def _salvage_footnote_cites(text: str) -> str:
    """Convert the two MALFORMED-but-genuine inline cite forms the model often emits into the
    canonical closed form ``<cite id="x">claim</cite>`` — WITHOUT changing which id supports
    which claim (content-preserving):

      * self-closing  ``claim <cite id="x"/>``         (claim before, tag self-closed)
      * footnote/open ``claim <cite id="x">. next``    (claim before, tag never closed)

    Both are footnote-style markers: the claim is the text immediately preceding the tag. We
    wrap the preceding CLAIM SENTENCE (back to the last sentence boundary / previous tag) so the
    grader's extractor (an unchanged verbatim DR Tulu function) can read the citation the model
    genuinely made. Well-formed ``<cite id="x">claim</cite>`` tags are left untouched.

    Conservative on purpose: it never invents a citation and never widens the claim beyond the
    preceding sentence, so a wrong citation stays wrong — it only makes genuine ones legible.
    This is what lets provenance (RER credit) and DRB-FACT both see the citations; the grader
    itself is not modified, so DR Tulu and our agent are scored by the same rule."""
    if "<cite" not in text:
        return text
    text = _SELFCLOSE_CITE.sub(lambda m: f'<cite id="{m.group(2)}">', text)  # self-closing -> bare open
    out: List[str] = []
    pos = 0
    for m in _OPEN_CITE.finditer(text):
        seg = text[pos:m.start()]          # text since the previous marker (contains the claim)
        ids = m.group(2)
        if _WELLFORMED_TAIL.match(text, m.end()):
            out.append(seg)
            out.append(f'<cite id="{ids}">')   # well-formed: leave the opening; claim+</cite> follow
            pos = m.end()
            continue
        # footnote marker: wrap the trailing claim sentence of `seg`
        b = max(seg.rfind("."), seg.rfind("!"), seg.rfind("?"), seg.rfind("\n"), seg.rfind("</cite>"))
        head, claim = seg[: b + 1], seg[b + 1:]
        claim_stripped = claim.strip()
        if claim_stripped:
            lead = claim[: len(claim) - len(claim.lstrip())]
            out.append(head + lead + f'<cite id="{ids}">{claim_stripped}</cite>')
        else:
            out.append(seg)                # nothing to attribute -> drop the empty marker
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def _coerce_citations(raw: Any) -> List[Dict[str, str]]:
    """Accept the model's ``citations`` in either form: a bare id string ``"sid"`` or a
    dict ``{"id": ..., "claim": ...}``. Anything else is skipped (never crash a rollout)."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for c in raw:
        if isinstance(c, str):
            cid, claim = c.strip(), ""
        elif isinstance(c, dict) and c.get("id"):
            cid, claim = str(c["id"]).strip(), str(c.get("claim", "") or "").strip()
        else:
            continue
        if cid:
            out.append({"id": cid, "claim": claim})
    return out


def normalize_answer(
    final_obj: Optional[Dict[str, Any]],
    raw_content: str,
    node_rid: str,
    ledger: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonicalize the model's answer into ``{content, citations:[{id, claim}]}``.

    * ``content``: from the dict, else the raw string.
    * ``citations``: from ``answer["citations"]`` if present; otherwise scraped from inline
      ``<cite>`` tags in the content (the default path, when no structured citations are
      supplied), which also recovers the claim text for each id.
    * **ledger backstop**: an id is kept only if it exists in ``ledger["snippets"]`` (real
      retrieved evidence). With no ledger, ids are accepted as-is.

    Pairs are de-duplicated on ``(id, claim)``. ``node_rid`` is accepted for symmetry;
    attribution-by-owner happens in the reward (the root legitimately carries merged child
    ids in its report), so it is intentionally NOT filtered here.
    """
    content = ""
    raw_cites: Any = None
    if isinstance(final_obj, dict):
        content = str(final_obj.get("content", "") or "")
        raw_cites = final_obj.get("citations")
    if not content:
        content = raw_content or ""
    content = _salvage_footnote_cites(content)  # footnote/self-closing -> closed form (genuine cites only)
    content = _quote_inline_cites(content)      # deterministic quoted-cite format (DRB-FACT)

    pairs = _coerce_citations(raw_cites)
    if not pairs:
        # legacy / unstructured: recover (id, claim) pairs from inline <cite> tags
        for claim, ids in extract_claims_and_corresponding_citation_ids(content).items():
            for cid in ids:
                pairs.append({"id": cid, "claim": claim})

    snips = (ledger or {}).get("snippets") if isinstance(ledger, dict) else None
    seen = set()
    citations: List[Dict[str, str]] = []
    for p in pairs:
        key = (p["id"], p["claim"])
        if key in seen:
            continue
        if snips is not None and p["id"] not in snips:
            continue  # backstop: cannot cite evidence never surfaced into the ledger
        seen.add(key)
        citations.append(p)
    return {"content": content, "citations": citations}


def cited_ids_of(obj: Dict[str, Any]) -> List[str]:
    """Flat, order-preserving, de-duplicated list of the ids in a normalized answer —
    the credit channel consumed as ``RerNode.cited_ids``."""
    return list(dict.fromkeys(c["id"] for c in (obj.get("citations") or [])))


_VERIFY_PROMPT = (
    "You are verifying a citation in a research report. Decide whether the EVIDENCE snippet "
    "genuinely supports the CLAIM (fully or in substantial part — exact numbers may round). "
    "Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
    "CLAIM: {claim}\n\nEVIDENCE: {evidence}\n\nAnswer:"
)


def verify_citations(
    obj: Dict[str, Any],
    ledger: Optional[Dict[str, Any]],
    lm_callback,
    max_evidence_chars: int = 1200,
) -> tuple:
    """Pre-submit citation VERIFICATION (the careful-researcher pass): batch-ask the LM
    whether each cited snippet actually supports its claim, and DROP the failures.

    Why: the harness reliably *makes* citations now, but loosely — measured DRB-FACT
    valid_rate 0.345 on hard items (and sqav2 citation alignment ~0). Each kept citation
    is later judged by the benchmark anyway, so shipping unsupported ones only costs
    score; pruning them is legitimate agent behavior (the grader is untouched).

    Mechanics: one prompt per (claim, evidence) pair, ONE batched ``lm_callback`` call.
    A pair is dropped only on an explicit UNSUPPORTED verdict; parse failures keep the
    citation (conservative — judge noise must not eat valid cites). Citations with no
    claim text or no ledger snippet are kept as-is (nothing to verify against; the
    ledger backstop in normalize_answer already vetted id existence). NEVER raises:
    any failure returns the object unchanged.

    Returns ``(possibly_filtered_obj, stats)`` with stats
    ``{"checked": int, "dropped": int}``.
    """
    stats = {"checked": 0, "dropped": 0}
    try:
        cits = list(obj.get("citations") or [])
        snips = (ledger or {}).get("snippets") if isinstance(ledger, dict) else None
        if not cits or not snips or lm_callback is None:
            return obj, stats
        idx, prompts = [], []
        for i, c in enumerate(cits):
            claim = str(c.get("claim", "") or "").strip()
            rec = snips.get(c.get("id"))
            evidence = str((rec or {}).get("text", "") or "").strip()
            if not claim or not evidence:
                continue  # nothing to judge against -> keep
            idx.append(i)
            prompts.append(_VERIFY_PROMPT.format(claim=claim[:800], evidence=evidence[:max_evidence_chars]))
        if not prompts:
            return obj, stats
        verdicts = lm_callback(prompts)
        drop = set()
        for i, v in zip(idx, verdicts):
            stats["checked"] += 1
            if "UNSUPPORTED" in str(v or "").upper():
                drop.add(i)
        if not drop:
            return obj, stats
        stats["dropped"] = len(drop)
        kept = [c for i, c in enumerate(cits) if i not in drop]
        return {"content": obj.get("content", ""), "citations": kept}, stats
    except Exception:
        return obj, stats  # verification must never break a rollout


def check_citations_verdicts(
    citations: Any,
    ledger: Optional[Dict[str, Any]],
    lm_callback,
    max_evidence_chars: int = 1200,
) -> List[Dict[str, str]]:
    """Citation-checking ORACLE backing the in-REPL ``check_citations`` tool: judge each
    ``{"id", "claim"}`` entry against its ledger snippet and return per-entry VERDICTS —
    it never filters. The agent decides what to drop, re-point, or rewrite (the act-on-
    verdicts behavior is the promptable/RL-trainable part; see prompts/variants/check_tool.txt).

    Same judging path as ``verify_citations`` (one batched ``lm_callback`` call,
    ``_VERIFY_PROMPT``), but verdict-returning so child-inherited entries are checkable at
    any node via the shared ledger. Verdicts: ``"SUPPORTED"`` / ``"UNSUPPORTED"`` /
    ``"UNKNOWN"`` (no claim text, unknown id, or judge noise — caller should keep those).
    NEVER raises: malformed input or a dead callback yields UNKNOWN rows, not an exception.
    """
    out: List[Dict[str, str]] = []
    try:
        cits = list(citations or [])
        snips = (ledger or {}).get("snippets") if isinstance(ledger, dict) else None
        idx, prompts = [], []
        for i, c in enumerate(cits):
            c = c if isinstance(c, dict) else {}
            claim = str(c.get("claim", "") or "").strip()
            rec = (snips or {}).get(c.get("id"))
            evidence = str((rec or {}).get("text", "") or "").strip()
            out.append({"id": str(c.get("id", "") or ""), "claim": claim, "verdict": "UNKNOWN"})
            if not claim or not evidence:
                continue
            idx.append(i)
            prompts.append(_VERIFY_PROMPT.format(claim=claim[:800], evidence=evidence[:max_evidence_chars]))
        if prompts and lm_callback is not None:
            verdicts = lm_callback(prompts)
            for i, v in zip(idx, verdicts):
                v = str(v or "").upper()
                if "UNSUPPORTED" in v:
                    out[i]["verdict"] = "UNSUPPORTED"
                elif "SUPPORTED" in v:
                    out[i]["verdict"] = "SUPPORTED"
        return out
    except Exception:
        return out if out else [{"id": "", "claim": "", "verdict": "UNKNOWN"}]


def _quote(ledger: Optional[Dict[str, Any]], cid: str, max_chars: int = 300) -> str:
    snips = (ledger or {}).get("snippets") if isinstance(ledger, dict) else None
    if not snips or cid not in snips:
        return ""
    text = str(snips[cid].get("text", "") or "").replace("\n", " ").strip()
    return text[:max_chars]


def render_report(
    obj: Dict[str, Any],
    ledger: Optional[Dict[str, Any]] = None,
    mode: str = "inline",
) -> str:
    """Render the canonical answer into the graded report.

    ``mode="inline"`` (default, FACT-friendly): each structured citation NOT already inline
    is attached to the claim it supports by wrapping that claim span with ``<cite id=...>``,
    so FACT can form statement<->citation pairs. Citations whose claim text can't be located
    verbatim fall back to a trailing References block (weak for FACT, evidence preserved).
    ``mode="appended"``: always use the References block.
    """
    content = str(obj.get("content", "") or "")
    citations = obj.get("citations") or []
    if not citations:
        return content

    already_inline = set(all_cited_ids(content))
    pending = [c for c in citations if c["id"] not in already_inline]

    refs: List[tuple] = []  # (id, claim) pairs that could not be inlined
    if mode != "appended":
        # group pending cites by the claim span they support, then inline once per span
        by_claim: "OrderedDict[str, List[str]]" = OrderedDict()
        for c in pending:
            by_claim.setdefault(c["claim"], []).append(c["id"])
        for claim, ids in by_claim.items():
            id_attr = ",".join(dict.fromkeys(ids))  # dedup, keep order
            if claim and claim in content:
                content = content.replace(claim, f'<cite id="{id_attr}">{claim}</cite>', 1)
            else:
                refs.extend((cid, claim) for cid in ids)
    else:
        refs = [(c["id"], c["claim"]) for c in pending]

    if not refs:
        return content
    lines = []
    for cid, claim in refs:
        # The DRB FACT formatter extracts the text INSIDE the <cite> tag as the "fact" it
        # verifies against the source, so put the CLAIM there — NOT the source quote, which
        # would make the support check circular (source supporting itself). Fall back to the
        # ledger quote / id only when there is no claim (keeps the tag non-empty & parseable).
        inner = claim or _quote(ledger, cid) or cid
        lines.append(f'- <cite id="{cid}">{inner}</cite>')
    return content + "\n\nReferences:\n" + "\n".join(lines)
