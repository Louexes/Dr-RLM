"""Generator-config extensions for DR-RLM (recursive deep-research RLM).

DR-RLM extends the RLM example generator config (``RLMGeneratorConfig``) with the
knobs that select a rung of the ablation ladder (L1..L4), bound recursion depth,
configure the held-constant rubric/citation judge, and wire the offline corpus
search tool.

Everything lives on ``cfg.generator`` (a single source of truth, auto-exposed as
``generator.<field>`` CLI overrides via ``make_config``). The generator threads
the env-relevant subset into ``env_extras["dr_rlm"]`` in ``_setup_env_extras`` so
the env (and, recursively, every child env) sees the same config without a second
config surface. See ``rer_reward.py`` for how these are consumed, and
``ARCHITECTURE.md`` for the mapping onto the thesis proposal's L0-L4 ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from examples.train.rlm.rlm_config import RLMGeneratorConfig


@dataclass
class DrRlmGeneratorConfig(RLMGeneratorConfig):
    # ------------------------------------------------------------------
    # Ablation-ladder selector (see ARCHITECTURE.md / proposal §6)
    # ------------------------------------------------------------------
    reward_mode: str = "inherited"
    """Which credit scheme fills each node's terminal reward:
      * ``"inherited"`` — L1: only the root is scored (report rubric R); children
        get 0. Combine with ``per_node_credit=false`` for stock flatten+broadcast.
      * ``"rao"``       — L2: per-node local reward = own-answer score + lambda *
        mean child success (RAO-style). Requires ``per_node_credit=true``.
      * ``"rer"``       — L3: per-node provenance-attributed credit r_a (the RER
        core). Requires ``per_node_credit=true``.
      * ``"rer_structural"`` — L4: L3 plus the structural-rubric penalty channel.
    """

    per_node_credit: bool = False
    """If True, un-flatten the tree: each recursion node becomes its own step-wise
    trajectory (distinct contiguous TrajectoryID, same instance_id) carrying its own
    terminal reward r_a, so the advantage estimator scores nodes independently.
    If False, fall back to stock SkyRL (flatten the tree into one root trajectory and
    broadcast the root scalar) — this is ladder rung L1.
    Note: ``per_node_credit=true`` implies ``train_child_trajectories=true``."""

    # ------------------------------------------------------------------
    # Recursion structure
    # ------------------------------------------------------------------
    max_recursion_depth: int = 2
    """Depth ceiling. A node at depth ``d`` may spawn children iff ``d <
    max_recursion_depth`` (subcall_fn is withheld at the ceiling). So
    ``max_recursion_depth=1`` is single-level delegation (root + workers);
    ``=2`` permits grandchildren (depth>1, the proposal's target);
    ``=0`` disables recursion entirely (flat single agent)."""

    max_children_per_node: int = 0
    """Per-NODE fan-out cap: the max number of child sub-agents a single node may spawn across
    all its ``rlm_query`` / ``rlm_query_batched`` calls (0 = unlimited). Bounds tree WIDTH — the
    search budget caps Serper calls, NOT the number of sub-agents (one depth-1 item spawned 38).
    Prompts beyond the cap in a batched call get a 'budget exhausted' message (list length
    preserved). The inference driver sets this from ``--max-children``; RL leaves it 0 unless set."""

    max_search_calls_per_tree: int = 0
    """Per-ITEM (whole recursion tree) cap on backend search calls; 0 = unlimited. Bounds total
    Serper/web API calls per item across all nodes (the legacy ``--max-tool-calls`` budget, which
    the unified driver had dropped). Once spent, ``search()`` returns ``[]`` so the model
    finalizes. Counted in the shared per-tree ledger. The inference driver sets this from
    ``--max-tool-calls`` (default 80); RL leaves it 0 unless set."""

    # ------------------------------------------------------------------
    # RER per-node credit parameters (proposal §6.3)
    #   r_a = sum_{c : a supports c} w_c * s_c * share(a,c)  -  gamma * cost(a)
    # ------------------------------------------------------------------
    root_reward_mode: str = "credit_share"
    """What the ROOT node's terminal reward is under rer/rer_structural:
      ``"credit_share"`` — the root earns only its own evidence-credit share of R
        (fully conserved: Σ_a r_a = R), the original L3 scheme;
      ``"full_R"`` — HYBRID: the root earns the report reward R itself (direct,
        undiluted pressure on synthesis quality — what L1 gets right) while children
        keep their provenance credit shares (what L3 gets right). Motivated by the
        2026-06-12 smoke: under pure conserved credit the root's small share diluted
        the quality gradient and judged report quality drifted DOWN while citing rose;
        broadcast (L1) showed the converse. Σ_a r_a = R + Σ_children shares (not
        conserved — by design)."""
    share_mode: str = "citation_count"
    """How a criterion's mass is split across the nodes that support it (all read the
    provenance off the FINAL REPORT's cites unless noted):
      ``"equal"`` (uniform over contributing nodes),
      ``"citation_count"`` (proportional to #cited snippets from the node — DEFAULT),
      ``"support"`` (proportional to judged support strength over report-cited claims),
      ``"ledger_support"`` (ABLATION, off by default: judged support over each node's OWN
        retrieved+cited evidence from the harness ledger — robust to the orchestrator
        dropping a child's cite, but can over-credit evidence that never reached the report.
        Gate on an orphan-rate measurement before defaulting to it; see [[evidence ledger]])."""

    gamma_cost: float = 0.0
    """Per-node cost penalty coefficient. cost(a) is normalized token+call cost of
    node ``a``'s subtree. 0.0 disables the cost term."""

    rao_lambda: float = 0.5
    """L2 only: weight on mean-child-success in the RAO local-node reward."""

    # ------------------------------------------------------------------
    # Citation format (the answer-dict contract)
    # ------------------------------------------------------------------
    # Nodes submit via the dict-based ``answer`` slot ``{content, ready}`` (the upstream RLM /
    # prime-rl format): the report — with inline ``<cite id="...">`` tags — goes in
    # ``answer["content"]`` and ``answer["ready"]`` is flipped when done. Provenance is read
    # from the report's inline cites (``share_mode="ledger_support"`` is an off-by-default
    # ablation that instead reads each node's own retrieved+cited evidence from the ledger).

    child_return_mode: str = "prose"
    """What a sub-agent's ``rlm_query()`` hands back to the PARENT's REPL (the child->parent
    contract; orthogonal to the GRADED report, which is always rendered prose):
      ``"prose"``      — the child's rendered report STRING (with inline ``<cite id="...">``
        tags). The parent must re-read prose and copy ``<cite>`` substrings to keep provenance
        (the v1 behavior). Conflicting free-text mini-reports can confuse the orchestrator's
        synthesis and induce fabrication.
      ``"structured"`` — REPORT-STYLE contract (V1-faithful): the child returns a uniform dict
        ``{content: <full mini-report prose>, citations: [{id, claim}]}``. The parent READS each
        child's ``content`` and writes a thorough report, copying the relevant ``citations`` entries
        into its own ``answer["citations"]`` so provenance is preserved programmatically (no
        re-parsing of prose, nothing dropped). A failed child still returns this dict shape with
        empty ``citations`` (never a bare string), so the parent's consumption code can't break on a
        heterogeneous list. NB: replaced the earlier FINDINGS contract (children returning claim-ATOMS
        the parent stitched), which made the parent write fragile dict-destructuring code that
        spiralled to empty reports and produced terse output (RACE regressed 0.265 -> 0.091).
    This is the prose-vs-structured A/B axis; the GRADED report rendering is unaffected."""

    citation_render: str = "inline"
    """How ``render_report`` emits structured citations into the GRADED report:
      ``"inline"`` — attach each citation to the claim it supports by wrapping that span
        with ``<cite id=...>`` (so DRB-FACT can form statement<->citation pairs; unlocatable
        cites fall back to a References block),
      ``"appended"`` — always emit a trailing References block (weak for FACT)."""

    citation_verification: bool = False
    """Pre-submit citation VERIFICATION (answer_format.verify_citations): at finalize, batch-ask
    the LM whether each cited ledger snippet actually supports its claim and DROP explicit
    failures before rendering. Targets loose citing (measured DRB-FACT valid_rate 0.345 on hard
    items despite discipline prompting). One batched LM call per submitting node. Default OFF:
    inference enables it via --citation-verification; RL training can leave it off (the callback
    would route verification to the policy itself). NB fairness (2026-06-12): this pass is
    architecture-agnostic, so head-to-heads vs DR-Tulu must run it OFF or symmetrically."""

    citations_bounce: bool = False
    """One-time empty-citations submission gate (DrRlmEnv._submission_bounce): when a node
    flips ready=True with non-empty content, EMPTY citations, and evidence in the tree
    ledger, the FIRST submission is rejected with a repair message (the model may also
    resubmit as-is). Targets the carry-displacement pathology no prompt iteration fixed
    (~5/16 roots, persisted from self_verify through check_tool v4). Harness contract
    enforcement, same family as the per-turn answer-state WARNING. Default OFF."""

    check_citations_tool: bool = False
    """Expose ``check_citations(citations)`` in every node's REPL (answer_format.
    check_citations_verdicts): an ORACLE the agent may call on ``answer["citations"]`` that
    returns per-entry SUPPORTED/UNSUPPORTED/UNKNOWN verdicts judged against the shared ledger
    (so child-inherited entries are checkable too). It never filters — acting on the verdicts
    (drop / re-point / rewrite) is the agent's job, which keeps verification inside the agent's
    own budget and trajectory (fair vs DR-Tulu when both get the same tool) and makes it an
    RL-learnable behavior. Default OFF; advertised only by prompt variants."""

    # ------------------------------------------------------------------
    # Structural-rubric channel (L4, proposal §6.4 / RQ3)
    # ------------------------------------------------------------------
    structural_orphan_penalty: float = 0.1
    """Penalty per child whose evidence never reached the final report (orphan)."""
    structural_redundancy_penalty: float = 0.1
    """Penalty per redundant child (evidence overlaps a sibling's)."""
    structural_fragmentation_penalty: float = 0.05
    """Penalty applied when the tree over-fragments (more children than needed)."""
    structural_max_children_soft: int = 6
    """Soft cap; children beyond this incur the fragmentation penalty."""

    # ------------------------------------------------------------------
    # Advantage estimator (proposal §6.5 / RQ4)
    # ------------------------------------------------------------------
    baseline_mode: str = "rloo"
    """Per-node baseline within an instance_id (prompt) group: ``"rloo"``
    (leave-one-out, RAO-style) or ``"grpo"`` (group mean/std)."""
    depth_weight_alpha: float = 1.0
    """RAO depth inverse-frequency weighting: a node at depth d is weighted by
    ``alpha / N_d`` where N_d = #nodes at depth d in the batch. 1.0 with a single
    depth recovers unweighted credit; raise to upweight deep (rare) nodes."""

    # ------------------------------------------------------------------
    # Held-constant rubric/citation judge (proposal §6.1-6.2)
    # ------------------------------------------------------------------
    judge_model: str = "hosted_vllm/Qwen/Qwen3-8B"
    """Judge model id. Defaults to a local Qwen served via an OpenAI-compatible
    vLLM endpoint (zero external calls). The SAME judge is used on every arm."""
    judge_base_url: str = "http://localhost:8100/v1"
    """OpenAI-compatible base URL for the judge endpoint."""
    judge_api_key_env: str = "JUDGE_API_KEY"
    """Env var holding the judge API key (use a dummy like 'EMPTY' for local vLLM)."""
    judge_score_scale: float = 2.0
    """Per-criterion judge scale (DR Tulu uses 0-2 -> normalized to [0,1])."""
    judge_max_concurrency: int = 16
    """Max concurrent judge calls per reward pass."""
    citation_reward_weight: float = 0.0
    """Weight of the in-context citation-support reward folded into the report score
    R (DR Tulu uses 0.2). 0.0 = rubric-only R (recommended while the citation graph
    is used for *credit*, not the reward, to avoid double counting)."""

    # ------------------------------------------------------------------
    # Offline corpus search tool (proposal §6 / held constant across arms)
    # ------------------------------------------------------------------
    search_backend: str = "mcp_http"
    """Retriever backend exposed as the REPL ``search()`` tool. The SAME flag flips
    online/offline in BOTH RL and inference (the tool facade is provider-agnostic):
      ``"mcp_http"``   — call a running dr_agent FastMCP ``local_search`` over HTTP (OFFLINE),
      ``"bm25"``       — import dr_agent BM25Searcher in-process (needs Java/pyserini, OFFLINE),
      ``"bm25s"``      — pure-Python BM25 over the self-crawled frozen corpus (corpus_build/, OFFLINE;
                         needs the bm25s+PyStemmer deps; ``search_index_path`` = the bm25s index dir),
      ``"faiss"``      — import dr_agent FaissSearcher in-process (OFFLINE),
      ``"local_jsonl"``— dependency-free fallback BM25-lite over a local corpus.jsonl (OFFLINE),
      ``"web"``        — live web search (Serper) + browse (Jina) via dr_agent MCP (ONLINE),
      ``"none"``       — no search tool (context-only / debugging)."""
    search_endpoint: str = "http://localhost:8003/mcp"
    """mcp_http: StreamableHTTP endpoint of the dr_agent MCP server."""
    search_corpus_path: str = "data/corpus.jsonl"
    """Path to the wii corpus.jsonl ({id, contents, url} rows)."""
    search_index_path: str = "data/bm25"
    """bm25/faiss: path to the prebuilt index (data/bm25 or data/qwen3-8b/corpus.pkl*)."""
    search_embed_model: str = "Qwen/Qwen3-Embedding-8B"
    """faiss: embedding model id."""
    search_top_k: int = 10
    """Default number of hits returned by search()."""
    snippet_max_chars: int = 2000
    """Per-snippet truncation in the REPL observation."""
    web_mcp_port: int = 8030
    """``search_backend="web"`` (ONLINE): port of the dr_agent FastMCP server exposing the
    Serper search + Jina browse tools."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # per-node credit only makes sense if children are in the training batch
        if self.per_node_credit:
            self.train_child_trajectories = True

    def env_payload(self) -> dict:
        """The subset of config threaded into ``env_extras["dr_rlm"]`` (reaches the
        env as ``self.extras["dr_rlm"]`` and propagates to children verbatim)."""
        return {
            "reward_mode": self.reward_mode,
            "per_node_credit": self.per_node_credit,
            "max_recursion_depth": self.max_recursion_depth,
            "max_children_per_node": self.max_children_per_node,
            "max_search_calls_per_tree": self.max_search_calls_per_tree,
            # RER credit knobs — consumed by compute_rer_rewards (MUST be threaded, else the
            # reward silently falls back to its own defaults: share_mode->citation_count, etc.)
            "share_mode": self.share_mode,
            "root_reward_mode": self.root_reward_mode,
            "gamma_cost": self.gamma_cost,
            "rao_lambda": self.rao_lambda,
            "structural_orphan_penalty": self.structural_orphan_penalty,
            "structural_redundancy_penalty": self.structural_redundancy_penalty,
            "structural_fragmentation_penalty": self.structural_fragmentation_penalty,
            "structural_max_children_soft": self.structural_max_children_soft,
            # citation format (consumed by DrRlmEnv + render)
            "citation_render": self.citation_render,
            "citation_verification": self.citation_verification,
            "check_citations_tool": self.check_citations_tool,
            "citations_bounce": self.citations_bounce,
            # child->parent return contract (consumed by DrRlmGenerator._child_return_value
            # + DrRlmEnv delegation-prompt section)
            "child_return_mode": self.child_return_mode,
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "judge_api_key_env": self.judge_api_key_env,
            "judge_score_scale": self.judge_score_scale,
            "judge_max_concurrency": self.judge_max_concurrency,
            "citation_reward_weight": self.citation_reward_weight,
            "search_backend": self.search_backend,
            "search_endpoint": self.search_endpoint,
            "search_corpus_path": self.search_corpus_path,
            "search_index_path": self.search_index_path,
            "search_embed_model": self.search_embed_model,
            "search_top_k": self.search_top_k,
            "snippet_max_chars": self.snippet_max_chars,
            "web_mcp_port": self.web_mcp_port,
        }
