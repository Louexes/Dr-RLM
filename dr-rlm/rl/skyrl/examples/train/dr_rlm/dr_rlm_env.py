"""``DrRlmEnv``: the recursive deep-research RLM environment.

Subclasses ``BaseRLMEnv`` and supplies the three task hooks:

* ``_get_system_prompt`` — a *depth-banded* prompt (orchestrator / coordinator /
  worker) that, unlike the shipped multi-paper env, lets nodes keep delegating while
  recursion budget remains — this is what makes depth>1 trees form.
* ``_get_repl_tools``    — the offline corpus ``search()`` / ``get_doc()`` tools with
  provenance-encoded snippet ids (held constant across arms).
* ``_get_reward``        — for L1 (``per_node_credit=false``) the root scores the report
  with the held-constant rubric judge (sync); children get 0; the trainer broadcasts.
  For L2-L4 (``per_node_credit=true``) the env returns 0 and the generator computes the
  real per-node reward r_a over the whole assembled tree (see ``dr_rlm_generator.py`` /
  ``rer_reward.py``) — the env's reward isn't used because the node rewards are
  overwritten before flattening.

All DR-RLM config arrives via ``extras["dr_rlm"]`` (threaded in by the generator's
``_setup_env_extras`` and propagated to children verbatim), so there is a single config
surface (``cfg.generator.*``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from skyrl_gym.envs.rlm.env import BaseRLMEnv
from skyrl_gym.metrics import default_aggregate_metrics

from .prompts import depth_system_prompt, delegation_section
from .tools.provider import make_tools
from .judge import (
    JudgeConfig,
    score_report_sync,
    all_cited_ids,
    citation_reward_config,
    ledger_citations,
    composite_report_reward,
)
from .answer_format import (
    normalize_answer,
    render_report,
    cited_ids_of,
    verify_citations,
    check_citations_verdicts,
)


class DrRlmEnv(BaseRLMEnv):
    """Recursive deep-research worker/orchestrator over a frozen corpus."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__(env_config=env_config, extras=extras)
        # DR-RLM config: prefer per-rollout extras (threaded by the generator);
        # fall back to env_config (cfg.environment.skyrl_gym.dr_rlm) if provided.
        self.dr_cfg: Dict[str, Any] = dict(self.extras.get("dr_rlm") or {})
        if not self.dr_cfg and isinstance(env_config, dict):
            self.dr_cfg = dict(env_config.get("dr_rlm", env_config))
        self._surfaced_ids: List[str] = []
        self._report_reward: float = 0.0
        self._citation_reward = None  # set when DR_RLM_CITATION_REWARD blends grounding into R
        self._citation_precision = None  # {id: 1/0/None} local SUPPORTED verdicts, reused as citation precision
        self._per_criterion: Dict[str, float] = {}
        self._answer_obj: Dict[str, Any] = None  # normalized {content, citations:[{id,claim}]}
        self._citation_verify_stats: Dict[str, int] = None  # {"checked","dropped"} when verification ran
        self._check_tool_stats: Dict[str, int] = None  # {"calls","checked","supported","unsupported"} when the REPL tool was used
        self._citations_bounced: bool = False  # one-time empty-citations submission bounce used

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        depth = int(self.extras.get("depth", 0))
        max_d = int(self.dr_cfg.get("max_recursion_depth", 2))
        mode = str(self.dr_cfg.get("child_return_mode", "prose"))
        return depth_system_prompt(depth, max_d, mode)

    def _get_repl_tools(self) -> Dict[str, Any]:
        node_rid = str(self.extras.get("rlm_rollout_id", "node"))
        # the per-tree evidence ledger (shared by reference root->children) — search()
        # records each surfaced snippet here so provenance is harness-tracked, not parsed.
        ledger = self.extras.get("dr_rlm_ledger")
        tools = make_tools(self.dr_cfg, node_rid, self._surfaced_ids, ledger)
        if bool(self.dr_cfg.get("check_citations_tool", False)) and self.lm_callback is not None:
            def check_citations(citations: Any) -> List[Dict[str, str]]:
                """Oracle, not effector: per-entry SUPPORTED/UNSUPPORTED/UNKNOWN verdicts
                against the shared ledger — child-inherited entries included. Acting on the
                verdicts (drop/re-point/rewrite) is the agent's job."""
                verdicts = check_citations_verdicts(citations, ledger, self.lm_callback)
                s = self._check_tool_stats or {"calls": 0, "checked": 0, "supported": 0, "unsupported": 0}
                s["calls"] += 1
                s["checked"] += len(verdicts)
                s["supported"] += sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
                s["unsupported"] += sum(1 for v in verdicts if v.get("verdict") == "UNSUPPORTED")
                self._check_tool_stats = s
                return verdicts
            tools["check_citations"] = check_citations
        return tools

    def _build_system_prompt(self) -> str:
        """Fill ``{custom_tools_section}`` with ONLY the recursive delegation tools, and
        only when this node can actually delegate.

        We deliberately do NOT advertise ``llm_query`` / ``llm_query_batched`` (which the
        base would list whenever ``lm_callback`` is set). All information must come from
        ``search()`` (own grounding) or ``rlm_query`` (delegated grounding, which itself
        searches), so the policy cannot route around the citation/provenance graph with
        ungrounded parametric LM calls. This also shrinks the action space and keeps the
        flat baseline (no ``subcall_fn``) at exactly DR Tulu's shape: search + answer.

        ``subcall_fn is not None`` is the ground truth for "can delegate" — the generator
        injects it iff ``depth < max_recursion_depth`` — so the advertised tools stay in
        sync with capability. ``search`` / ``get_doc`` are documented in the band prompt's
        EVIDENCE section (see ``prompts.py``), so they are not re-listed here.
        """
        template = self._get_system_prompt()
        if self.subcall_fn is not None:
            section = self._delegation_tools_section()
        else:
            section = ""
        return template.replace("{custom_tools_section}", section)

    def _delegation_tools_section(self) -> str:
        """The `rlm_query` / `rlm_query_batched` block spliced into a delegating node's prompt.

        The TEXT (prose vs structured contract) lives in ``system_prompt.txt`` — the single source
        of truth — as the DELEGATION_PROSE / DELEGATION_STRUCTURED sections. Here we only SELECT by
        ``child_return_mode``, mirroring how ``depth_system_prompt`` selects the band."""
        return delegation_section(str(self.dr_cfg.get("child_return_mode", "prose")))

    def _get_context_metadata_text(self, context_payload) -> str:
        """The base's "Your context is a str with N total characters, broken up into
        chunks..." implies the answer is sitting in `context`. Here `context` is just the
        research question — say so, and point at search() instead."""
        n = len(context_payload) if isinstance(context_payload, str) else len(str(context_payload))
        return (
            f"Your `context` variable holds the research question itself ({n} characters) and "
            "nothing else — no documents or search results are preloaded. Gather all evidence "
            "with the REPL tools."
        )

    def _answer_content_len(self) -> int:
        """Length of the live ``answer["content"]`` in the REPL, for the per-turn status line.
        Surfacing this stops the model from hallucinating it already submitted (observed:
        a node spinning out 12 turns insisting "the report was submitted in a previous turn"
        while ``answer["content"]`` stayed empty -> a zero-length report). Defensive: any
        access failure (no REPL yet, unexpected shape) reports 0."""
        try:
            ans = self.repl.locals.get("answer") if self.repl is not None else None
            return len(str((ans or {}).get("content", "") or ""))
        except Exception:
            return 0

    def _answer_citations_len(self) -> int:
        """Number of entries in the live ``answer["citations"]`` (0 on any access failure).
        Surfaced in the per-turn status (structured mode) so a drafted-but-uncited answer
        gets flagged BEFORE submission — observed: 5/16 DRB reports shipped with content
        but zero citations (forgotten assignment / copying only one child's citations)."""
        try:
            ans = self.repl.locals.get("answer") if self.repl is not None else None
            cits = (ans or {}).get("citations")
            return len(cits) if isinstance(cits, (list, tuple)) else 0
        except Exception:
            return 0

    def _get_user_prompt(self, iteration: int):
        """Per-turn scaffold without the base's "(which contains the context)" framing.
        Shows the turn budget, the live answer-buffer state (anti false-submit), blocks a
        turn-0 instant answer, and switches to a finalize nudge when <=2 turns remain
        (legacy behavior: submit best inference, don't let the rollout terminate unsubmitted)."""
        head = f'Turn {iteration + 1}/{self.max_turns}. Research question: "{self._root_prompt}"\n\n'
        remaining = self.max_turns - iteration
        # Ground the model in the ACTUAL state of its submission buffer, so it can't think a
        # past turn already wrote/submitted when the buffer is empty.
        n = self._answer_content_len()
        status = (
            "STATUS: `answer[\"content\"]` is EMPTY — nothing has been written or submitted yet. "
            "A report only exists once YOU write it into `answer[\"content\"]` this turn; do not "
            "assume a previous turn did.\n"
            if n == 0 else
            f"STATUS: `answer[\"content\"]` holds a {n}-char draft; it is NOT submitted until you "
            "set `answer[\"ready\"] = True`.\n"
        )
        # Structured mode: a drafted answer with ZERO citations is about to ship ungrounded —
        # flag it while the model can still fix it (uncited answers earn no provenance credit).
        if (
            n > 0
            and str(self.dr_cfg.get("child_return_mode", "prose")) == "structured"
            and self._answer_citations_len() == 0
        ):
            status += (
                "WARNING: `answer[\"citations\"]` is EMPTY — your draft cites nothing, and an "
                "uncited answer scores poorly. Before submitting, copy the supporting "
                "`{\"id\", \"claim\"}` entries (from your `search` hits and every child's "
                "`citations`) into `answer[\"citations\"]`.\n"
            )
        if iteration == 0:
            body = (
                "You have not retrieved any evidence yet, so do not submit a final answer this "
                "turn. Plan briefly, then start researching in ONE ```repl``` block. Your next action:"
            )
        elif remaining <= max(2, getattr(self, "_finalize_nudge_turns", 0)):
            if getattr(self, "_finalize_nudge_turns", 0) > 0:
                body = (
                    status +
                    f"Only {remaining} turn(s) left. STOP searching and STOP reasoning further — do "
                    "NOT open a new line of analysis. In THIS response emit ONE ```repl``` block that "
                    'writes your best-supported report (with its inline `<cite id="...">` tags) into '
                    '`answer["content"]` and sets `answer["ready"] = True`. An unsubmitted rollout '
                    "scores zero; a rough submitted answer always beats nothing. Your next action:"
                )
            else:
                body = (
                    status +
                    f"Only {remaining} turn(s) left. Finalize now: write your best-supported report "
                    '(with its inline `<cite id="...">` tags) into `answer["content"]` and set '
                    '`answer["ready"] = True` — an unsubmitted rollout scores zero. Your next action:'
                )
        else:
            body = (
                status +
                "Continue your research in the REPL (ONE ```repl``` block per response). When your "
                "report is complete, write it into `answer[\"content\"]` and set `answer[\"ready\"] = "
                "True`. Your next action:"
            )
        return {"role": "user", "content": head + body}

    def _submission_bounce(self, final_obj) -> Optional[str]:
        """One-time empty-citations gate (flag ``citations_bounce``, default OFF).

        Targets the carry-displacement pathology that survived four prompt iterations
        (self_verify -> check_tool v4): ~5/16 roots write a full report, never populate
        ``answer["citations"]`` although the tree surfaced evidence, and submit. Bouncing
        the FIRST such submission with a repair message converts lost rows; an answer that
        genuinely needs no citations just resubmits as-is (one extra turn, never a loop —
        the gate fires at most once per node and never on the last turn)."""
        if not bool(self.dr_cfg.get("citations_bounce", False)) or self._citations_bounced:
            return None
        obj = final_obj if isinstance(final_obj, dict) else {}
        content = str(obj.get("content", "") or "")
        if not content.strip() or (obj.get("citations") or []):
            return None
        ledger = self.extras.get("dr_rlm_ledger")
        n_evidence = len(((ledger or {}).get("snippets") or {}) if isinstance(ledger, dict) else {})
        if n_evidence == 0:
            return None  # nothing was ever retrieved in this tree: uncited is legitimate
        self._citations_bounced = True
        return (
            "SUBMISSION PAUSED (one-time check, not an error): your answer has content but "
            f'answer["citations"] is EMPTY, while {n_evidence} evidence snippets were retrieved '
            "in this research tree. If sentences of your answer are supported by retrieved "
            "snippets (your own search hits, or entries in a child's `citations`), populate "
            'answer["citations"] now — keep each `id` unchanged and set its `claim` to the exact '
            'sentence of YOUR answer it supports — then set answer["ready"] = True again. '
            'If your answer genuinely needs no citations, set answer["ready"] = True again '
            "to submit as-is."
        )

    def _finalize_answer(self, raw_content: str, final_obj) -> str:
        """Normalize the submitted ``answer`` dict against this tree's evidence ledger,
        then render the report so each (ledger-validated) citation is attached INLINE to
        the claim it supports — which is what DRB-FACT scores. The normalized object is
        stashed for ``get_metrics`` (→ structured ``cited_ids`` for credit, and the
        child→parent return so an orchestrator can merge a sub-agent's citations)."""
        ledger = self.extras.get("dr_rlm_ledger")
        node_rid = str(self.extras.get("rlm_rollout_id", "node"))
        self._answer_obj = normalize_answer(final_obj, raw_content, node_rid, ledger)
        # Pre-submit citation verification (flag-gated): batch-judge each (claim, snippet)
        # pair and drop explicit failures BEFORE rendering — unsupported citations only
        # cost benchmark score and pollute the provenance signal. Never raises.
        if bool(self.dr_cfg.get("citation_verification", False)) and self.lm_callback is not None:
            self._answer_obj, self._citation_verify_stats = verify_citations(
                self._answer_obj, ledger, self.lm_callback
            )
        # CHEAP citation-reward PRECISION (only when the grounding reward is on, only at the
        # ROOT whose report is scored): judge each (claim, cited-snippet) with the LOCAL model
        # (self.lm_callback — free, batched), so the reward path reuses these verdicts instead
        # of spending Gemini precision calls. Recall stays on Gemini (the new orphan signal).
        on, _w = citation_reward_config()
        if on and int(self.extras.get("depth", 0)) == 0 and self.lm_callback is not None:
            try:
                _verds = check_citations_verdicts(self._answer_obj.get("citations"), ledger, self.lm_callback)
                self._citation_precision = {
                    v["id"]: (1.0 if v["verdict"] == "SUPPORTED" else 0.0 if v["verdict"] == "UNSUPPORTED" else None)
                    for v in _verds if v.get("id")
                }
            except Exception:
                self._citation_precision = None
        mode = str(self.dr_cfg.get("citation_render", "inline"))
        return render_report(self._answer_obj, ledger, mode=mode)

    def _get_reward(self, final_answer: str) -> float:
        depth = int(self.extras.get("depth", 0))
        per_node = bool(self.dr_cfg.get("per_node_credit", False))

        # Per-node credit (L2-L4): the generator overwrites this node's terminal reward
        # with r_a over the whole tree (it needs siblings/parent, unavailable here). The
        # env reward is a placeholder.
        if per_node:
            return 0.0

        # L1 (inherited): only the root (depth 0) is scored, with the held-constant
        # rubric judge; children inherit advantage via the trainer broadcast.
        if depth > 0:
            return 0.0

        rubrics = (self.extras.get("reward_spec") or {}).get("rubrics") or []
        if not rubrics:
            # No rubric -> no in-env score. Without this, score_report_sync falls back to a
            # general rubric and fires a (doomed, ~15s-retrying) judge HTTP call — which the
            # inference driver does NOT want (it passes rubrics=[] and scores the report with
            # the external grader). RL always supplies rubrics, so this never fires there.
            return 0.0
        cfg = JudgeConfig.from_env_payload(self.dr_cfg)
        on, weights = citation_reward_config()
        if on:
            # grounding-aware composite (opt-in): blend DR Tulu's citation_reward into R using
            # this tree's evidence ledger. Default path below stays rubric-only (isolation).
            cits = ledger_citations(self.extras.get("dr_rlm_ledger"))
            R, comp = composite_report_reward(
                final_answer, self._root_prompt, rubrics, cits, cfg, weights,
                precision_by_id=self._citation_precision,
            )
            self._report_reward = R
            self._per_criterion = comp.get("per_criterion", {})
            self._citation_reward = comp.get("citation")
        else:
            R, per_criterion = score_report_sync(final_answer, self._root_prompt, rubrics, cfg)
            self._report_reward = R
            self._per_criterion = per_criterion
        return R

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["depth"] = int(self.extras.get("depth", 0))
        metrics["rlm_rollout_id"] = self.extras.get("rlm_rollout_id")
        metrics["n_surfaced_snippets"] = len(self._surfaced_ids)
        # Structured answer object (normalized, ledger-validated). Exposed so the generator's
        # child→parent subcall can hand the parent a real dict to MERGE (not prose to copy),
        # and so credit reads `cited_ids` from the structured channel rather than re-parsing
        # prose. Falls back to scraping inline <cite> tags when the answer has no structured citations.
        obj = self._answer_obj
        metrics["answer_obj"] = obj
        metrics["cited_ids"] = cited_ids_of(obj) if obj else all_cited_ids(self._final_answer or "")
        if self._citation_verify_stats is not None:
            metrics["citation_verification"] = dict(self._citation_verify_stats)
        if self._check_tool_stats is not None:
            metrics["check_citations_tool"] = dict(self._check_tool_stats)
        if self._citations_bounced:
            metrics["citations_bounced"] = True
        if self._per_criterion:
            metrics["report_reward"] = self._report_reward
        if getattr(self, "_citation_reward", None) is not None:
            metrics["citation_reward"] = self._citation_reward
        if getattr(self, "_citation_precision", None) is not None:
            # local SUPPORTED verdicts {id:1/0/None} — the per-node (L2-L4) reward path reuses
            # these for citation precision instead of spending Gemini precision calls.
            metrics["citation_precision"] = self._citation_precision
        return metrics

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Split rollouts by depth: depth=0 → root/*, depth>=1 → subagent/*."""
        roots = [m for m in metrics if m.get("depth", 0) == 0]
        subs = [m for m in metrics if m.get("depth", 0) > 0]
        out: Dict[str, Any] = {}
        out.update({f"root/{k}": v for k, v in default_aggregate_metrics(roots).items()})
        if subs:
            out.update({f"subagent/{k}": v for k, v in default_aggregate_metrics(subs).items()})
        return out
