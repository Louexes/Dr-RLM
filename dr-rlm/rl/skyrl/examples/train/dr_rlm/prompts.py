"""Depth-banded system prompts for DR-RLM, loaded from the canonical prompt file.

ALL model-facing system-prompt text lives in ``dr-rlm/prompts/system_prompt.txt`` (tracked in
git; this vendored skyrl tree is NOT), so there is exactly one file to read and iterate on to
steer the agent — for training and inference alike. This module only parses that file and
assembles the band for a node; it contains no prompt prose.

The file is split into ``<<<NAME>>>``-marked sections: three band templates
(ORCHESTRATOR / COORDINATOR / WORKER) and three shared fragments (EVIDENCE / FINISHING /
REPL_RULES) spliced into the bands via ``{EVIDENCE}``-style placeholders. The
``{custom_tools_section}`` placeholder is left for ``BaseRLMEnv._build_system_prompt`` to fill.

Banding rationale (unchanged from v1): the shipped multi-paper RLM only ever recurses one
level because its child prompt omits the ``rlm_query`` tools. DR-RLM bands the prompt by
*recursion budget remaining* so genuine depth>1 trees form:

  * ROOT / ORCHESTRATOR  (can still delegate)   -> decompose + delegate + synthesize
  * COORDINATOR          (can still delegate)   -> a sub-topic that is itself decomposed
  * WORKER / LEAF        (at the depth ceiling)  -> answer one focused question directly

A node may delegate iff ``depth < max_recursion_depth`` (the generator only injects
``subcall_fn`` there, so the prompt and the capability are kept in sync). All bands share the
same evidence / citation protocol — inline ``<cite id="...">`` tags submitted via the
``answer`` dict — so the citation graph (and thus RER credit) is uniform across depths.
"""

from __future__ import annotations

import os
import re
from typing import Dict

# .../dr-rlm/rl/skyrl/examples/train/dr_rlm -> up 5 -> .../dr-rlm
_DEFAULT_PROMPT_FILE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 5), "prompts", "system_prompt.txt")
)
PROMPT_FILE = os.environ.get("DR_RLM_PROMPT_FILE", _DEFAULT_PROMPT_FILE)

_BANDS = ("ORCHESTRATOR", "COORDINATOR", "WORKER")
_FRAGMENTS = ("EVIDENCE", "FINISHING", "REPL_RULES")   # mode-independent, spliced into every band
_CITATIONS = ("CITATIONS_PROSE", "CITATIONS_STRUCTURED")  # mode-aware, fills {CITATIONS}
_DELEGATION = ("DELEGATION_PROSE", "DELEGATION_STRUCTURED")  # mode-aware, fills {custom_tools_section}

_SECTION_RE = re.compile(r"^<<<([A-Z0-9_]+)>>>[ \t]*$", re.MULTILINE)
_sections_cache: Dict[str, str] | None = None


def _sections() -> Dict[str, str]:
    global _sections_cache
    if _sections_cache is None:
        try:
            with open(PROMPT_FILE) as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"DR-RLM canonical prompt file not found: {PROMPT_FILE} "
                "(set DR_RLM_PROMPT_FILE to override)"
            )
        parts = _SECTION_RE.split(text)  # [preamble, name1, body1, name2, body2, ...]
        # Drop `#`-comment lines (column 0) from every section body, so inter-section
        # documentation comments never leak into the model-facing prompt (and can't smuggle
        # in a stray `{custom_tools_section}` that .replace would then double-fill).
        def _strip_comments(body: str) -> str:
            return "\n".join(l for l in body.splitlines() if not l.startswith("#")).strip("\n")
        sections = {parts[i]: _strip_comments(parts[i + 1]) for i in range(1, len(parts) - 1, 2)}
        # A fragment is satisfied by EITHER its plain section OR both mode-aware variants
        # (e.g. FINISHING, or FINISHING_PROSE + FINISHING_STRUCTURED) — see _assemble.
        def _frag_missing(name: str) -> bool:
            return name not in sections and not (
                f"{name}_PROSE" in sections and f"{name}_STRUCTURED" in sections
            )
        missing = (set(_BANDS) | set(_CITATIONS) | set(_DELEGATION)) - sections.keys()
        missing |= {f for f in _FRAGMENTS if _frag_missing(f)}
        if missing:
            raise ValueError(f"{PROMPT_FILE} is missing sections: {sorted(missing)}")
        _sections_cache = sections
    return _sections_cache


def delegation_section(child_return_mode: str) -> str:
    """The delegation-tools block spliced into a delegating node's prompt (fills
    ``{custom_tools_section}``), selected by ``child_return_mode``. The TEXT lives in
    ``system_prompt.txt`` (single source of truth); this only picks prose vs structured.
    Returned with a leading blank line so it sits cleanly under the band header."""
    key = "DELEGATION_STRUCTURED" if str(child_return_mode) == "structured" else "DELEGATION_PROSE"
    return "\n\n" + _sections()[key]


def _assemble(band: str, child_return_mode: str) -> str:
    s = _sections()
    out = s[band]
    mode = "STRUCTURED" if str(child_return_mode) == "structured" else "PROSE"
    for name in _FRAGMENTS:
        # A fragment may optionally be MODE-AWARE: if the prompt file defines e.g.
        # FINISHING_STRUCTURED / FINISHING_PROSE, the mode's variant wins; otherwise the
        # plain section is used (fully backward compatible — canonical v2 defines only
        # plain sections and assembles byte-identically). Needed because the v3 freeze
        # candidate's FINISHING ritual references check_citations/answer["citations"],
        # which exist only under the structured contract.
        out = out.replace("{" + name + "}", s.get(f"{name}_{mode}", s.get(name, "")))
    # {CITATIONS} is MODE-AWARE: every band gets the prose or structured citation contract, so
    # the way a node records provenance is consistent across orchestrator/coordinator/worker.
    cit = s["CITATIONS_STRUCTURED" if str(child_return_mode) == "structured" else "CITATIONS_PROSE"]
    out = out.replace("{CITATIONS}", cit)
    # NB: plain .replace (not .format) — fragments contain literal braces ({id, text, url}, the
    # answer dict); {custom_tools_section} stays for the env to fill with the DELEGATION section.
    return out


def depth_system_prompt(depth: int, max_recursion_depth: int, child_return_mode: str = "prose") -> str:
    """Pick the band for a node by how much recursion budget it has left, and splice in the
    citation contract for ``child_return_mode`` (so the contract is consistent across all bands).

    Banding keys on *can-delegate* (``depth < max_recursion_depth``), not depth alone, so
    the prompt never tells a node to delegate when the tool is absent:

    * depth 0  AND can delegate    -> orchestrator (root report, decompose + delegate)
    * 0 < depth < ceiling          -> coordinator  (can still delegate)
    * depth >= ceiling             -> worker/leaf   (must answer directly)

    In particular ``max_recursion_depth == 0`` (the flat L0 baseline) makes the root a
    worker: search + answer, no delegation talk — exactly DR Tulu's shape.
    """
    can_delegate = depth < max_recursion_depth
    if depth == 0 and can_delegate:
        band = "ORCHESTRATOR"
    elif can_delegate:
        band = "COORDINATOR"
    else:
        band = "WORKER"
    return _assemble(band, child_return_mode)
