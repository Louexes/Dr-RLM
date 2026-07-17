# DR-RLM Architecture

*Recursive deep-research RLM training/eval on SkyRL. Design document for an examiner or successor.*

Every claim below is grounded in code. Citations of the form `file:line` point either at this
package (`examples/train/dr_rlm/`) or, where the mechanism lives in the substrate, at SkyRL
(`SkyRL/...`), the RLM inference library (`rlm/...`), or DR Tulu (`dr-tulu/...`). Substrate line
numbers are sourced from the line-level maps in `experiments/_drrlm_maps/` and re-verified against
the package code that calls them.

---

## 1. Thesis in one paragraph

DR-RLM is a **controlled A/B** between *flat* deep research (the DR Tulu recipe: one agent, one
report, one scalar rubric reward) and *recursive* deep research (an orchestrator that delegates
sub-questions to sub-agents over a tree, with **provenance-attributed per-node credit** — RER).
The experiment holds **everything except architecture and credit constant**: the same prompts /
rubrics / judge (`judge.py`, a verbatim port of DR Tulu's rubric scorer), the same frozen offline
corpus and retriever (`corpus_search.py`), the same base policy, the same GRPO trainer. Only two
things vary along the ablation ladder: **architecture** (flat → tree, gated by
`max_recursion_depth`, `dr_rlm_config.py:51`) and **credit** (a single broadcast scalar →
per-node difference reward keyed by the citation graph, gated by `reward_mode` +
`per_node_credit`, `dr_rlm_config.py:29,40`). Because both arms run inside *one* harness with one
reward path (`judge.weighted_report_reward`, `judge.py:189`, shared by the L1 in-env scorer and the
L2–L4 generator-side pipeline), any quality/latency/compute delta is attributable to the
architecture+credit change rather than to incidental differences in data, judge, or corpus.

---

## 2. The L0–L4 ablation ladder

The ladder is the experimental spine (proposal §6). Each rung adds exactly one mechanism. The
selector flags live on `DrRlmGeneratorConfig` and are auto-exposed as `generator.<field>` CLI
overrides by `make_config(generator_cls=DrRlmGeneratorConfig)` (`main_dr_rlm.py:28`).

| Rung | What it isolates | Config flags (`generator.*`) | Implemented in |
|---|---|---|---|
| **L0** | Flat baseline — *external* DR Tulu-8B reference, downloaded, no recursion at all | n/a (run DR Tulu's own stack, or this harness with `max_recursion_depth=0`) | DR Tulu repo / `dr_rlm_config.py:51` |
| **L1** | Recursion alone, **advantage inheritance** (stock SkyRL): only root scored, children broadcast | `reward_mode=inherited`, `per_node_credit=false`, `max_recursion_depth>=1` | env scores root (`dr_rlm_env.py:61-81`); generator falls back to stock flatten+broadcast (`dr_rlm_generator.py:137-139`) |
| **L2** | Coarse per-node scalar — **RAO delegation bonus** + depth weighting | `reward_mode=rao`, `per_node_credit=true`, `rao_lambda` | RAO branch `rer_reward.py:121-137`; optional depth weighting `rer_advantage.py` |
| **L3** | **RER core (RQ2)** — provenance-attributed per-node credit via the citation graph | `reward_mode=rer`, `per_node_credit=true`, `share_mode`, `gamma_cost` | RER branch `rer_reward.py:139-204`; un-flattening `dr_rlm_generator.py:136-236` |
| **L4** | + **structural rubric channel (RQ3)** — orphan/redundant/over-fragmented penalty on the root | `reward_mode=rer_structural`, `per_node_credit=true`, `structural_*` | `rer_reward.py:195-204` + `_structural_penalty` `rer_reward.py:207-235` |

Notes that make the ladder *clean*:

- **L1 vs L2/L3/L4 is the un-flattening switch.** `per_node_credit=true` forces
  `train_child_trajectories=true` in `__post_init__` (`dr_rlm_config.py:140-145`), so children are
  always in the training batch when per-node credit is on; with it off the generator takes the
  stock path verbatim (`dr_rlm_generator.py:137-139`).
- **The report reward `R` is computed the same way on every rung.** L1 uses the synchronous
  in-env scorer `score_report_sync` (`judge.py:200`, runs in a SkyRL executor thread inside
  `_get_reward`, `dr_rlm_env.py:78`); L2–L4 use the async judge inside `compute_rer_rewards`
  (`rer_reward.py:97-110`). Both funnel through the identical `weighted_report_reward`
  (`judge.py:189`), so `R` is byte-identical across arms — the rungs differ only in how `R` (and
  the per-node quality proxies) are *distributed* as credit, not in how `R` is measured.

---

## 3. The un-flattening mechanism (the core systems contribution)

### 3.1 What stock SkyRL does: flatten → zero → broadcast

A recursive RLM rollout is a tree of nodes (orchestrator + sub-agents). Stock SkyRL collapses that
tree into a *single* trajectory and gives the whole tree *one* advantage:

1. **Child rid inherits the parent's shared `trajectory_id`.** `_register_rollout`
   (`rlm/rlm_generator.py:339`) mints a fresh `rid` per node but sets the child's
   `trajectory_id = parent.trajectory_id` (`rlm/rlm_generator.py:369`) — every node in the tree
   carries the *same* `TrajectoryID`.
2. **Tree flatten #1 (generator).** The root branch of `_post_process_agent_loop_output`
   DFS-collects all descendant `step_outputs` and **prepends** them to the root's `step_outputs`
   (`rlm/rlm_generator.py:199`), producing one `StepWiseOutput` for the whole tree.
3. **Tree flatten #2 (base generate).** `SkyRLGymGenerator.generate`'s step-wise branch
   (`skyrl_gym_generator.py:820-841`) flattens every `step_output` into the parallel
   `GeneratorOutput` batch lists, assigning the *same* input `TrajectoryID` to every row
   (`...generate:839`) and marking `is_last_step=True` only on the single final flattened row
   (`...generate:838`).
4. **Reward zeroing #1 (env).** The shipped multi-paper env short-circuits child reward to zero:
   `MultipaperEvidenceRLMEnv._get_reward` does `if depth > 0: return 0.0`
   (`evidence_rlm_env.py:288-290`). Only the root is judged.
5. **Reward zeroing #2 + broadcast (trainer).** `RayPPOTrainer.compute_advantages_and_returns`'s
   step-wise branch selects only `is_last_step` rows, computes the advantage over those, then
   **broadcasts** the single root scalar to every step via `traj_ids = cumsum(shift(is_last_step))`
   (`trainer.py:889-897`). Child token spans receive the root's advantage; their own (zero) rewards
   are discarded.

Net effect: children are *inherited* into the root trajectory and trained on the root's single
scalar. This is exactly **L1**.

### 3.2 What `DrRlmGenerator.generate` does instead

When `per_node_credit=true`, `DrRlmGenerator.generate` (`dr_rlm_generator.py:136-236`) runs the
stock generate first (`super().generate`, `dr_rlm_generator.py:141`) to get the flattened, but
DFS-*contiguous*, step rows, then **re-labels each node as its own step-wise trajectory**:

- **Segment into per-output blocks** at `is_last_step` boundaries (`dr_rlm_generator.py:154-163`),
  then **group each block's rows by node `rid`** (`dr_rlm_generator.py:179-183`). Node `rid`,
  `depth`, and `parent_rid` are read from `env_metrics[r]["rlm_metadata"]`, which the generator
  stamped on every step in `_post_process_agent_loop_output` (`dr_rlm_generator.py:95-105`). Rows of
  one node are contiguous and step-ordered because the DFS prepend (step 2 above) keeps each
  subtree as a contiguous block.
- **Mint a per-node `TrajectoryID`** (`dr_rlm_generator.py`):
  - `instance_id = base_iid` — the **plain prompt `uid`**, unchanged. This is mandatory: the trainer
    builds GRPO groups *and* mini-batch boundaries from `instance_id`
    (`compute_prompt_mini_batch_boundaries`, `preprocess.py:240,248`), asserting each value is
    contiguous and that `#distinct == train_batch_size`. Depth-encoding `instance_id` would make
    same-depth siblings non-contiguous (DFS interleaves grandchildren) and explode the distinct
    count — it crashes training. So all of a prompt's nodes stay in one contiguous group.
  - `repetition_id = base_rep * _REP_STRIDE + seq` where `_REP_STRIDE = 100_000`
    (`dr_rlm_generator.py:44`) and `seq` is the node's index within its block — unique per node so the
    *step-wise validator* (which keys on `to_string()`) treats each node as its own trajectory.
- **Mark `is_last_step=True` on each node's last row** (`dr_rlm_generator.py:218-219`) and write that
  node's terminal reward `r_a` (from `compute_rer_rewards`) as a token vector with the scalar at the
  final position (`dr_rlm_generator.py:220-224`).

### 3.3 Why this satisfies `_validate_step_wise_fields`

`_validate_step_wise_fields` (`trainer_utils.py:678`) enforces a strict step-wise contract on
`GeneratorOutput`, keyed by `TrajectoryID.to_string() = f"{instance_id}_{repetition_id}"`
(`base.py:12`):

- **`is_last_step[-1]` is True** (`trainer_utils.py:709`): the last block's last node ends the batch,
  and the un-flattener marks every node's last row True, so the very last row is True.
- **Contiguity** — all rows of a `to_string()` must be adjacent (`trainer_utils.py:715-730`): each
  node's rows are already a contiguous DFS block, and each gets a *distinct* `to_string()` because
  `repetition_id = base_rep * 100_000 + seq` is unique per node (the stride exceeds any plausible
  nodes-per-tree count, `dr_rlm_generator.py:44-45`), so no `to_string()` is reused after a different
  one intervenes.
- **`is_last_step` True exactly at trajectory boundaries** (`trainer_utils.py:732-750`): setting True
  on each node's last row and nowhere else makes the boundary `tid_cur != tid_next` line up exactly
  with `is_last_step=True`, which is what the trainer's `cumsum`-based trajectory mapping needs.

### 3.4 Why stock GRPO already gives per-node credit (no trainer edits), and the depth-cohort variant

The trainer derives GRPO groups from `instance_id`: `uids = [tid.instance_id for tid in
trajectory_ids]` (`trainer.py:242`), and the GRPO estimator baselines each row against its group
mean/std (`ppo_utils.py:1176-1226`). Because the un-flattener keeps `instance_id = prompt uid` for
every node, **all of a prompt's nodes — root and children, every depth, across the `n` samples —
form one GRPO group**, and each node's *own* terminal `r_a` (not the root's broadcast scalar) drives
its advantage. So stock `grpo` centers each node against the **mean credit of all the prompt's
nodes** — a per-prompt difference-reward baseline. This is the key point: **once the tree is
un-flattened, no trainer edit or custom estimator is needed for per-node credit** — the credit is in
each node's `r_a`, and GRPO does the rest. (The contiguity validator keys on `to_string()`, which is
why `repetition_id` must be unique-and-contiguous; it does not affect grouping.)

That per-prompt baseline mixes depths (a child's small `r_a` is centered against the root's large
`R`). This is a defensible difference reward — nodes that earn above-average credit for the prompt
are reinforced — but the **depth-cohort baseline** (center each node only against same-depth peers,
removing the depth scale bias) and **RAO depth inverse-frequency weighting** (`w_d = alpha / N_d`,
renormalized to unit mean, so rare deep nodes are not drowned out) are the principled RQ4 refinement.
Those need per-row node depth, which the stock dispatch does not forward — so they are wired through
the **opt-in `DrRlmTrainer`** (`dr_rlm_trainer.py`): it copies `node_depth` from
`env_metrics[i]["rlm_metadata"]` into `data.metadata` (`convert_to_training_input`) and threads it
into the estimator in the step-wise branch of `compute_advantages_and_returns`. The `rer_pernode`
estimator (`rer_advantage.py`) then forms `(prompt, depth)` cohorts, baselines within them, and
applies the depth weighting; absent `node_depth` it degrades to the plain per-prompt baseline (so the
name is always safe). It is registered at import via `@register_advantage_estimator("rer_pernode")`,
selected with `trainer.algorithm.advantage_estimator=rer_pernode` (which auto-selects `DrRlmTrainer`
in `main_dr_rlm.py`), and registration happens at package import so the name is in the registry before
`validate_cfg`'s membership check (`utils.py:274`).

---

## 4. The RER reward and the provenance graph

### 4.1 The provenance chain

The credit assignment is a citation-graph difference reward computed **without counterfactual
rollouts** — it reads provenance straight off the report's citations:

1. **`search()` mints provenance-encoded ids.** Each hit a node surfaces is given id
   `f"{node_rid}-{n}"` (`corpus_search.py:260`), where `node_rid` is the env's `rlm_rollout_id`
   (`dr_rlm_env.py:58`, threaded as the node's `rid`). Node rids are dash-free 8-char uuids
   (`rlm/rlm_generator.py:339`), so the suffix is unambiguous.
2. **The report cites those ids.** The depth-banded prompts instruct the policy to cite with
   `<cite id="ab12cd-3">claim</cite>` and tell it that "a passage you surface and cite is
   attributed to you" (`prompts.py:28-38`).
3. **`<cite id>` decodes to the surfacing node.** `extract_claims_and_corresponding_citation_ids`
   (`judge.py:57`, the verbatim DR Tulu regex) extracts `{claim: [cite_id,...]}` from the root
   report, and `owner_rid_of(cite_id)` decodes the node via `cite_id.rsplit("-", 1)[0]`
   (`corpus_search.py:287-289`).
4. **`share(a,c)` splits each criterion's mass `w_c·s_c` across contributing nodes.** Controlled by
   `share_mode` (`dr_rlm_config.py:62`):
   - `equal` — uniform over nodes whose cited evidence appears in the report
     (`rer_reward.py:165`);
   - `citation_count` — proportional to `#cited snippets` from the node (`rer_reward.py:167-168`,
     the default);
   - `support` — per-criterion judged support strength via `RubricJudge.supports`
     (`rer_reward.py:170-183`, `judge.py:171`).
5. **Conservation `Σ_a = R`.** For `equal`/`citation_count` the share is criterion-independent and
   `r_a^evidence = R · share(a)` with `Σ share = 1`, so the evidence credit is **conserved** and
   merely redistributed by who-earned-it (`rer_reward.py:159-169`; for `support`, per-criterion
   shares sum to 1 so `Σ` over contributors `= Σ_c w_c·s_c`, `rer_reward.py:182-183`). The final
   per-node reward subtracts the optional cost term: `n.reward = evidence_credit[rid] - gamma·cost`
   (`rer_reward.py:185-186`).

### 4.2 The `reward_mode` switch

`compute_rer_rewards` (`rer_reward.py:83`) branches on `payload["reward_mode"]`:

- `inherited` (L1): `R` to the root, `0` to children (`rer_reward.py:113-118`).
- `rao` (L2): each node's reward is its own general-rubric quality proxy plus `lambda *
  mean(child quality)` minus the cost term; the root uses `R` as its proxy
  (`rer_reward.py:121-137`).
- `rer` (L3): the provenance-attributed credit of §4.1 (`rer_reward.py:139-186`).
- `rer_structural` (L4): `rer` plus the structural penalty applied to the **root**
  (`rer_reward.py:195-200`).

### 4.3 The structural channel (L4)

`_structural_penalty` (`rer_reward.py:207-235`) is a cheap, deterministic stand-in for the
co-evolving structural rubric (proposal §6.4, RQ3). It penalizes:

- **orphan children** — a child whose evidence never reached the report
  (`node_cite_count[rid] == 0`, `rer_reward.py:219`), weighted by `structural_orphan_penalty`;
- **redundant children** — sibling answer-token Jaccard `> 0.6` (`rer_reward.py:221-227`), weighted
  by `structural_redundancy_penalty`;
- **over-fragmentation** — children beyond `structural_max_children_soft` (`rer_reward.py:229`),
  weighted by `structural_fragmentation_penalty`.

The penalty is subtracted from the root's reward only (`rer_reward.py:198`), and the per-decomposition
counts are logged as metrics (`rer_reward.py:231-235`).

### 4.4 The judge, held constant

`judge.py` is a self-contained re-implementation of DR Tulu's rubric scorer so the *same* reward
logic runs on every arm without dragging the whole `open_instruct` stack into the SkyRL runtime.
What is ported verbatim (header comment `judge.py:1-18`):

- the per-criterion 0–2 → [0,1] scoring prompt + normalization, from
  `dr-tulu/.../rubric_utils.py::_score_property[_async]` → `RubricJudge.score_criterion`
  (`judge.py:158-165`) and the sync `score_report_sync` (`judge.py:200-250`);
- the `<cite id=...>` regex, from `dr-tulu/.../citation_utils.py` →
  `extract_claims_and_corresponding_citation_ids` (`judge.py:53-70`);
- the weighted-sum normalization (negative-weight criteria subtract from the numerator but are
  excluded from the denominator, so `R ≤ 1` and may be `< 0`), from
  `dr-tulu/.../longform_rubric_rewards.py::_compute_rubric_scores_and_reward` →
  `weighted_report_reward` (`judge.py:189-197`).

Transport is a plain async OpenAI-compatible `/chat/completions` call (`judge.py:116-142`) pointed
at a local Qwen served by vLLM (`judge_base_url`, `dr_rlm_config.py:104`), so there are **zero
external API calls** and the same judge model id (`judge_model`, `dr_rlm_config.py:101`) is used on
every arm. A failed/parse-error call returns `0.0` exactly like the DR Tulu original
(`judge.py:138-140,163-164`) so a judge outage looks like a bad answer, not a crash.

---

## 5. depth>1 recursion

Genuine depth>1 trees (grandchildren) are wanted (proposal target). The recursion machinery already
exists structurally in **both** substrates; DR-RLM only adds the depth gate and depth-banded
prompts.

- **The depth gate.** A node at depth `d` may spawn children iff `d < max_recursion_depth`. The
  generator withholds `subcall_fn` at the ceiling (`dr_rlm_generator.py:72-77`): once
  `depth >= max_d`, `rlm_query` degrades to a no-op so capability and prompt stay in sync.
  So `max_recursion_depth=0` is flat, `=1` is single-level delegation (root + workers), `=2`
  permits grandchildren (`dr_rlm_config.py:51-56`).
- **Depth-banded prompts.** `depth_system_prompt(depth, max_d)` (`prompts.py`) picks the band by
  *can-delegate* (`depth < max_d`), not depth alone: ORCHESTRATOR (depth 0 **and** can delegate),
  COORDINATOR (`0 < depth < ceiling`), or WORKER/LEAF (otherwise — including the root when
  `max_recursion_depth=0`, the flat L0 baseline). This is the key difference from the shipped
  multi-paper env, whose child prompt deliberately omits delegation and so only ever recurses one
  level. All bands share the same evidence/citation/`answer`-dict contract so the citation graph (and
  thus RER credit) is uniform across depths.
- **Action space: grounded-only (no `llm_query`).** `DrRlmEnv._build_system_prompt`
  (`dr_rlm_env.py`) advertises only `search`/`get_doc` (in the EVIDENCE block) and, *iff the node can
  delegate* (`subcall_fn is not None`), `rlm_query`/`rlm_query_batched`. The base would also list
  `llm_query`/`llm_query_batched`; DR-RLM deliberately does NOT, so every fact must come from
  `search()` (own grounding) or `rlm_query` (delegated grounding) — the policy cannot inject
  ungrounded parametric text that would be uncited / hallucinate citation ids and is invisible to the
  provenance graph. This also shrinks the action space and pins the flat L0 arm to exactly DR Tulu's
  shape (search + answer). `llm_query` stays *bound* in the REPL only as the silent fallback for a
  leaf's stray `rlm_query` call; it is never advertised, so a warm-started policy won't use it.
- **Recursion already works in SkyRL.** `_make_subcall_fn`'s inner `_run_child` re-enters
  `self.agent_loop(...)` for the child (`rlm/rlm_generator.py:295-327`), so a child env that is
  given `subcall_fn` recurses with no further changes — depth>1 is purely a matter of injecting
  `subcall_fn` (which DR-RLM gates on depth) and prompting for it.
- **Recursion already works in `rlm/` (the eval/untrained arm).** `RLM._subcall` constructs a child
  `RLM` at `depth+1` (`rlm/core/rlm.py:704-868`), and `subcall_fn` is injected only when
  `environment in (local, ipython) and max_depth > 1` (`rlm/core/rlm.py:276-277`). The depth bound is
  `depth >= max_depth` (`rlm/core/rlm.py:348,732`), so **`max_depth >= 3` is required for true
  grandchildren** (root depth0 REPL, child depth1 REPL, grandchild depth2 leaf LM). This is a config
  setting on the `rlm/` inference path, not a code change.

---

## 6. SFT, RL, and eval paths

- **RL training entry point:** `main_dr_rlm.py`. `make_config(generator_cls=DrRlmGeneratorConfig)`
  (`main_dr_rlm.py:28`) wires the config; `DrRlmPPOExp.get_generator` builds `DrRlmGenerator`
  (`main_dr_rlm.py:31-38`); importing the package registers the `dr_rlm` env and the `rer_pernode`
  estimator (`__init__.py:16-20`). Launch configs are intended to live under `configs/` (currently
  empty — see Limitations).
- **Eval entry point:** `main_dr_rlm_eval.py`. Generate-only over the eval dataset via
  `EvalOnlyEntrypoint` with the same `DrRlmGenerator` so the recursive hooks fire
  (`main_dr_rlm_eval.py:28-42`). For the controlled comparison eval set `per_node_credit=false` (eval
  scores the report, not per-node credit). The per-axis latency/compute/quality eval of the
  *untrained* recursive arm is intended for `eval/` (the `rlm/` inference path).
- **SFT path:** intended for `sft/` (warm-start from a DR Tulu-style checkpoint that already emits
  `<cite>` tags; currently empty — see Limitations).

The env, generator, config, reward, judge, search, prompts, and advantage estimator are all present
and import-clean; `configs/`, `sft/`, `eval/`, `tests/`, and `data/` are scaffolded but empty.

---

## 7. Controls and the two-baseline distinction

**Controls held constant across L1–L4 (the in-harness arms):**

- **Prompts** — same depth-banded prompt family (`prompts.py`); only the architecture flag changes
  which bands are reachable.
- **Rubrics** — the same held-constant rubric list, read from
  `reward_spec["rubrics"] = [{description, title, weight}]` (`dr_rlm_env.py:76`,
  `dr_rlm_generator.py:201`).
- **Judge** — one judge config (`JudgeConfig.from_env_payload`, `judge.py:91-106`), one model id,
  one normalization (`weighted_report_reward`) shared by the L1 sync and L2–L4 async paths.
- **Corpus + retriever** — one backend, process-cached per worker (`get_backend`,
  `corpus_search.py:198-220`), with provenance-encoded ids identical across arms.
- **Trainer / estimator** — stock `grpo` for L1–L4 (no trainer edits); only the optional RQ4 run
  swaps in `rer_pernode`.
- **Config surface** — a single source of truth on `cfg.generator`, threaded into
  `env_extras["dr_rlm"]` by `env_payload()` (`dr_rlm_config.py:147-167`) and propagated to every
  child env verbatim, so children and root see identical config.

**Two distinct baselines (do not conflate):**

1. **Controlled flat baseline (L1, in-harness).** Recursion structurally on but credit = stock
   inheritance; same harness, same judge, same corpus. This isolates "recursion + per-node credit"
   from "recursion alone."
2. **External released DR Tulu-8B reference (L0).** The downloadable flat model run on its own stack.
   This is a *reference point* for absolute quality, not a controlled comparison — it differs in
   training data, infra, and possibly judge, so deltas against it are descriptive, not causal.

---

## 8. Known limitations and corrections to the proposal

These are corrections that should be folded back into `Recursive Deep Research Proposal.md`:

1. **`compute_child_rlm_metrics` is dead code (0 callers), not "computed then discarded."** The
   proposal repeatedly cites `compute_child_rlm_metrics` (`evidence_rewards.py:103`) as a per-child
   quality signal that SkyRL "computes and then discards" and frames it as DR-RLM's "wiring hook"
   (proposal §1, §8, risk table, §"Tree-RL substrate"). In fact it has **no Python callers anywhere**
   in the repo (grep-verified; only referenced in `thesis_obsidian/*.md`), its docstring's claim that
   `RLMGymGenerator` uses it is stale, and it returns **aggregate means, not per-child arrays**
   (map 02:64-66,190-192). DR-RLM does **not** route this function into credit; it computes per-node
   credit from scratch in `rer_reward.py`. The proposal should drop the "discarded per-child metric is
   our wiring hook" framing.
2. **SkyRL zeroes+flattens child rewards; it is not mere "inheritance."** The proposal calls the
   SkyRL baseline "advantage inheritance" as if children passively receive the parent scalar. The
   actual mechanism is stronger and two-sided: child rewards are **zeroed** at the env
   (`evidence_rlm_env.py:288-290`) **and** the tree is **flattened** into one trajectory at two sites
   (`rlm/rlm_generator.py:199` then `skyrl_gym_generator.py:839`), and only then is the root scalar
   **broadcast** by the trainer's `cumsum` (`trainer.py:889-897`). "Inheritance" understates that
   children carry zero own-reward and are not independently scored at all. DR-RLM's un-flattening must
   (and does) undo *both* the zeroing and the flatten.
3. **The public `dr-tulu-rl-data` is 4,881 rows (the `os`+`sa` subset), not "~9K."** The proposal's
   §"Corpus/data" says `rl-research/dr-tulu-rl-data (~9K long-form prompts)`. The publicly available
   data is the `os`+`sa` subset at **4,881 rows**; the full ~9K set is not public. Plan the data
   budget around 4,881. (Both the RL prompts `s42chen/wii` and the index repo `s42chen/wii-indexes`
   are gated — map 06:114,317.)
4. **Judge caveats — format compliance, orphan credit, cost term.**
   - *Format compliance:* the judge returns `0.0` on any parse failure (`judge.py:163-164`,
     `score_report_sync` `judge.py:227,231`). A local Qwen that does not reliably emit
     `{"score": x}` JSON silently zeros criteria; monitor judge health (DR Tulu has the same failure
     mode, map 05:188).
   - *Orphan credit:* a node whose surfaced evidence is never cited in the final report gets **zero**
     evidence credit under `rer`/`citation_count`/`equal` (it is not in `contributors`,
     `rer_reward.py:161,171`). Useful work that the orchestrator chose not to cite is uncredited by
     design; this is also what the L4 orphan penalty counts (`rer_reward.py:219`). This is a credit-
     coverage caveat worth stating explicitly, and a place where `share_mode=support` (which can
     credit cited claims even with sparse counts) behaves differently.
   - *Cost term:* `gamma_cost` defaults to `0.0` (disabled, `dr_rlm_config.py:68`) and `cost` is a
     crude normalized **token-share** of the subtree (`dr_rlm_generator.py:195`), not a token+call
     accounting. Treat the cost penalty as a coarse proxy, not the astabench-style accounting the
     proposal's compute axis envisions.
5. **The gated `wii` corpus is the remaining blocker.** Both `s42chen/wii` (RL prompts) and
   `s42chen/wii-indexes` (corpus + prebuilt BM25/Qwen3-Embed indices) require `huggingface-cli login`
   (map 06:114,317). Without them the `bm25`/`faiss` backends cannot initialize; `corpus_search.py`
   falls back to the dependency-free `local_jsonl` backend (`corpus_search.py:216-218`) for smoke
   tests, but the real controlled run needs the gated indices. This matches the standing project
   blocker.

---

## 9. File map

| File | Role |
|---|---|
| `dr_rlm_config.py` | `DrRlmGeneratorConfig`: all ladder/recursion/RER/judge/search knobs; `env_payload()` threads the env-relevant subset into `env_extras["dr_rlm"]`. |
| `prompts.py` | Depth-banded system prompts (orchestrator / coordinator / worker) + the citation/`answer`-dict contract; `depth_system_prompt(depth, max_d)`. |
| `corpus_search.py` | Offline `search()`/`get_doc()` REPL tools; backends bm25/faiss/mcp_http/local_jsonl/none; provenance-encoded `"{rid}-{n}"` ids; `owner_rid_of`. |
| `judge.py` | Held-constant rubric/citation judge (DR Tulu port): `RubricJudge` (async), `score_report_sync` (sync, L1), `weighted_report_reward`, `extract_claims_and_corresponding_citation_ids`, `JudgeConfig.from_env_payload`. |
| `rer_reward.py` | The RER pipeline: `RerNode`, `compute_rer_rewards(nodes, rubrics, question, payload) -> RerResult`; the four `reward_mode` branches + `_structural_penalty`. |
| `dr_rlm_env.py` | `DrRlmEnv(BaseRLMEnv)`: depth-banded prompt hook, corpus-tools hook, `_get_reward` (root scores in L1, returns 0 in L2–L4), depth-split metrics. |
| `dr_rlm_generator.py` | `DrRlmGenerator(RLMGymGenerator)`: depth-ceiling gate + config threading; the un-flattening `generate` that mints per-node `TrajectoryID`s and terminal `r_a`. |
| `rer_advantage.py` | Optional `rer_pernode` advantage estimator (RQ4): depth inverse-frequency weighting on top of the within-group baseline. |
| `main_dr_rlm.py` | RL training entry point (registers env + estimator, wires `DrRlmGenerator`). |
| `main_dr_rlm_eval.py` | Eval-only entry point (generate-only, recursive hooks fire). |
| `__init__.py` | Registers the `dr_rlm` env id and imports `rer_advantage` to register the estimator. |
| `configs/`, `sft/`, `eval/`, `tests/`, `data/` | Scaffolded, currently empty (launch scripts, SFT, untrained-arm eval, unit tests, corpus). |
