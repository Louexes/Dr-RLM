"""``DrRlmGenerator``: the recursive rollout generator with per-node credit.

Extends the RLM example generator (``RLMGymGenerator``) with the two changes that
turn advantage *inheritance* into provenance-attributed per-node credit:

1. **Depth ceiling + config threading** (``_setup_env_extras``): a node may spawn
   children only while ``depth < max_recursion_depth`` (subcall_fn is withheld at the
   ceiling, keeping capability and the depth-banded prompt in sync), and the dr_rlm
   config sub-dict is injected into ``env_extras`` so the env + every child sees it.

2. **Un-flattening with per-node reward** (``generate``, when ``per_node_credit``):
   the stock pipeline flattens the whole tree into ONE root trajectory, zeroes child
   rewards, and broadcasts the root scalar. Here, instead, every recursion node becomes
   its own step-wise trajectory:
     * its rows keep the PROMPT ``uid`` as ``instance_id`` (mandatory — SkyRL builds GRPO
       groups *and* mini-batch boundaries from ``instance_id``, asserting each value is
       contiguous and ``#distinct == train_batch_size``; depth-encoding it would crash
       training), and get a distinct ``repetition_id = base_rep*STRIDE + seq`` so the
       step-wise contiguity validator treats each node as its own trajectory;
     * its terminal step carries its own reward ``r_a`` from the RER pipeline
       (``rer_reward.compute_rer_rewards``) run once over the assembled tree.
   So all of a prompt's nodes form ONE GRPO group and stock GRPO assigns each node an
   independent advantage from its own ``r_a`` (centered on the per-prompt node mean) —
   no trainer edits, no custom estimator needed. The optional depth-cohort/depth-weighted
   ``rer_pernode`` estimator (+ ``DrRlmTrainer``, which threads per-row node depth) is the
   RQ4 variant. When ``per_node_credit`` is off this falls straight back to stock
   behavior (ladder rung L1).

See ``ARCHITECTURE.md`` for the mapping to the proposal's L0-L4 ladder and the
step-wise credit contract (``_validate_step_wise_fields``) this output satisfies.
"""

from __future__ import annotations

import gc
import resource
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from loguru import logger

from examples.train.rlm.rlm_generator import RLMGymGenerator, _RLMRolloutContext  # noqa: F401
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import StepWiseOutput
from skyrl.train.generators.utils import get_rollout_metrics

from .rer_reward import RerNode, compute_rer_rewards


def _credit_telemetry(rows: "List[Dict[str, Any]]") -> None:
    """OPT-IN per-rollout credit telemetry (DR_RLM_CREDIT_METRICS_PATH=<jsonl>).

    The smoke-run evidence channel: appends one JSON line per rollout tree with the
    per-node credit/citation facts (r_a, depth, #cited, orphanhood, RER metrics), on
    BOTH the per_node_credit path and the L1 flatten path — so L3-vs-L1 behavioral
    slopes (orphan fraction, cites/child, advantage spread) are comparable offline.
    Default off => byte-identical behavior; never raises into the training loop."""
    import json as _json
    import os as _os
    import time as _time

    path = _os.environ.get("DR_RLM_CREDIT_METRICS_PATH", "")
    if not path or not rows:
        return
    try:
        with open(path, "a") as f:
            ts = _time.time()
            for r in rows:
                r["ts"] = ts
                f.write(_json.dumps(r) + "\n")
    except Exception as e:  # telemetry must never kill training
        logger.warning(f"[dr_rlm] credit telemetry write failed: {e}")


def _flat_credit_aggregates(nodes: "List[Dict[str, Any]]") -> "Dict[str, float]":
    """Scalar dr_rlm/* aggregates for the tracker (W&B) from flat-path node summaries."""
    roots = [n for n in nodes if n.get("depth", 0) == 0]
    kids = [n for n in nodes if n.get("depth", 0) > 0]
    out = {
        "dr_rlm/children_per_tree": len(kids) / max(len(roots), 1),
        "dr_rlm/n_nodes": float(len(nodes)),
    }
    if kids:
        out["dr_rlm/cites_per_child"] = sum(k.get("n_cited", 0) for k in kids) / len(kids)
        out["dr_rlm/child_cite0_frac"] = sum(1 for k in kids if not k.get("n_cited")) / len(kids)
    return out


def _pernode_credit_aggregates(trees: "List[Dict[str, Any]]") -> "Dict[str, float]":
    """Scalar dr_rlm/* aggregates for the tracker (W&B) from per-node-credit telemetry rows.

    These are the smoke-run evidence metrics, live on the W&B charts every training step:
    report reward health, orphan/credited child fractions, sibling advantage spread
    (L3>0 vs L1's structural 0), credit conservation, citing behavior."""
    out: "Dict[str, float]" = {}
    if not trees:
        return out
    rr = [t["report_reward"] for t in trees if t.get("report_reward") is not None]
    if rr:
        out["dr_rlm/report_reward"] = sum(rr) / len(rr)
        out["dr_rlm/R_positive_frac"] = sum(1 for r in rr if r > 0) / len(rr)
    # citation-quality observability (present only when DR_RLM_CITATION_REWARD is on)
    cr = [t["rer_metrics"]["citation_reward"] for t in trees
          if t.get("rer_metrics") and t["rer_metrics"].get("citation_reward") is not None]
    if cr:
        out["dr_rlm/citation_reward"] = sum(cr) / len(cr)
    for _src, _dst in (("cit_precision", "dr_rlm/citation_precision"),
                       ("cit_recall", "dr_rlm/citation_recall"),
                       ("cit_f1", "dr_rlm/citation_f1")):
        _v = [t[_src] for t in trees if t.get(_src) is not None]
        if _v:
            out[_dst] = sum(_v) / len(_v)
    kids = [n for t in trees for n in t.get("nodes", []) if n.get("depth", 0) > 0]
    out["dr_rlm/children_per_tree"] = len(kids) / max(len(trees), 1)
    out["dr_rlm/recursion_rate"] = sum(
        1 for t in trees if any(n.get("depth", 0) > 0 for n in t.get("nodes", []))
    ) / max(len(trees), 1)
    if kids:
        out["dr_rlm/cites_per_child"] = sum(k.get("n_cited", 0) for k in kids) / len(kids)
        ra = [k.get("r_a") for k in kids if k.get("r_a") is not None]
        if ra:
            out["dr_rlm/orphan_child_frac"] = sum(1 for r in ra if abs(r) < 1e-9) / len(ra)
            out["dr_rlm/credited_child_frac"] = 1.0 - out["dr_rlm/orphan_child_frac"]
    spreads, gaps = [], []
    for t in trees:
        cs = [n["r_a"] for n in t.get("nodes", []) if n.get("depth", 0) > 0 and n.get("r_a") is not None]
        if len(cs) >= 2:
            m = sum(cs) / len(cs)
            spreads.append((sum((c - m) ** 2 for c in cs) / len(cs)) ** 0.5)
        R = t.get("report_reward")
        if R is not None:
            ev = sum(n.get("r_a", 0.0) or 0.0 for n in t.get("nodes", []))
            gaps.append(abs(ev - R))
    if spreads:
        out["dr_rlm/within_tree_ra_std"] = sum(spreads) / len(spreads)
    if gaps:
        out["dr_rlm/conservation_gap"] = sum(gaps) / len(gaps)
    return out


def _nodes_from_env_metrics(env_metrics: "List[Optional[Dict[str, Any]]]") -> "List[Dict[str, Any]]":
    """Best-effort per-node summary off env_metrics rows (works on the flat L1 path)."""
    by_rid: "Dict[str, Dict[str, Any]]" = {}
    for r in env_metrics or []:
        r = r or {}
        meta = r.get("rlm_metadata") or {}
        rid = meta.get("rid") or r.get("rlm_rollout_id")
        if rid is None:
            continue
        rid = str(rid)
        rec = by_rid.setdefault(rid, {"rid": rid, "depth": int(meta.get("depth", r.get("depth", 0)) or 0),
                                      "n_cited": 0, "ans_len": 0})
        cids = r.get("cited_ids") or []
        if cids:
            rec["n_cited"] = max(rec["n_cited"], len(cids))
        fa = r.get("final_answer") or ""
        if fa:
            rec["ans_len"] = max(rec["ans_len"], len(fa))
    return list(by_rid.values())

# repetition_id namespace per root rollout; must exceed max nodes-per-tree.
_REP_STRIDE = 100_000


def _join_prompt(prompt) -> str:
    if isinstance(prompt, list):
        return "\n".join(m.get("content", "") for m in prompt if m.get("content"))
    return str(prompt)


class DrRlmGenerator(RLMGymGenerator):
    RLM_ENV_CLASSES: frozenset = frozenset({"evidence_rlm", "multipaper_evidence_rlm", "dr_rlm"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dr_payload: Dict[str, Any] = (
            self.generator_cfg.env_payload() if hasattr(self.generator_cfg, "env_payload") else {}
        )
        # per-rollout-tree evidence ledgers, keyed by root trajectory_id.to_string().
        # Created at the root in _setup_env_extras, populated by every node's search()
        # (shared by reference down the tree), consumed + cleared in generate().
        self._tree_ledgers: Dict[str, Dict[str, Any]] = {}
        self._ledger_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hook 1: thread dr_rlm config + enforce the depth ceiling
    # ------------------------------------------------------------------

    def _setup_env_extras(self, env_class, env_extras, sampling_params, trajectory_id):
        if env_class in self.RLM_ENV_CLASSES:
            env_extras = dict(env_extras)
            env_extras.setdefault("dr_rlm", self._dr_payload)
            # Root rollout (children carry the _rlm_parent_rid sentinel set by _run_child):
            # mint the per-tree evidence ledger and stash it by root trajectory id. Children
            # inherit the SAME object by reference via _run_child's dict(env_extras) copy, so
            # every node's search() writes into one ledger; generate() reads it back by id.
            if "_rlm_parent_rid" not in env_extras and "dr_rlm_ledger" not in env_extras:
                ledger = {"snippets": {}, "_lock": threading.Lock()}
                env_extras["dr_rlm_ledger"] = ledger
                if trajectory_id is not None:
                    with self._ledger_lock:
                        self._tree_ledgers[trajectory_id.to_string()] = ledger
        env_extras = super()._setup_env_extras(env_class, env_extras, sampling_params, trajectory_id)
        if env_class in self.RLM_ENV_CLASSES:
            depth = int(env_extras.get("depth", 0))
            max_d = int(self._dr_payload.get("max_recursion_depth", 2))
            if depth >= max_d:
                # at the ceiling: withhold recursion so rlm_query degrades to a no-op
                env_extras.pop("subcall_fn", None)
        return env_extras

    # ------------------------------------------------------------------
    # Hook: child->parent return contract (prose vs structured A/B)
    # ------------------------------------------------------------------

    def _child_return_value(self, child_env_metrics: Dict[str, Any]) -> Any:
        """What a sub-agent's ``rlm_query()`` returns to the PARENT's REPL.

        ``child_return_mode="structured"`` (REPORT-STYLE, the V1-faithful contract): hand back
        the child's normalized ``answer_obj`` as ``{content, citations:[{id, claim}]}`` — its
        FULL mini-report plus its citation list — so the parent SYNTHESIZES a thorough report
        from ``content`` and preserves provenance by copying the relevant ``citations`` entries.
        This replaced the FINDINGS contract (children returning claim-ATOMS the parent stitched),
        which made the parent write fragile dict-destructuring code that spiralled into empty
        reports and yielded terse output (RACE 0.265 -> 0.091). The return is ALWAYS a uniform
        dict — even a failed/timed-out child gets ``{content: <prose>, citations: []}``, never a
        bare string — so a heterogeneous ``subs`` list can't break the parent's consumption code.

        ``child_return_mode="prose"`` (control / pre-structured behavior): the base class's
        rendered report string. The GRADED report is rendered prose either way, so only the
        orchestrator's *input* contract changes between arms.
        """
        mode = str(self._dr_payload.get("child_return_mode", "prose"))
        if mode == "structured":
            obj = child_env_metrics.get("answer_obj")
            if isinstance(obj, dict) and obj.get("content"):
                return {"content": obj["content"], "citations": list(obj.get("citations") or [])}
            # Uniform fallback: a child with no structured object still returns a DICT (never a
            # bare string), so `subs` is homogeneous and `child["content"]`/`child["citations"]`
            # always resolve. The base value is the rendered prose report (or empty).
            base = super()._child_return_value(child_env_metrics)
            return {"content": base if isinstance(base, str) else str(base or ""), "citations": []}
        return super()._child_return_value(child_env_metrics)

    # ------------------------------------------------------------------
    # Hook 2: richer per-node metadata + child inlining
    # ------------------------------------------------------------------

    def _post_process_agent_loop_output(self, agent_loop_output, env_extras, trajectory_id):
        rid = env_extras.get("rlm_rollout_id")
        ctx = self.active_rollouts.get(rid) if rid else None
        if ctx is None:
            return agent_loop_output
        assert isinstance(agent_loop_output, StepWiseOutput), (
            f"DrRlmGenerator requires step_wise_trajectories=True, got {type(agent_loop_output).__name__}"
        )

        # Stamp everything needed to reconstruct the tree downstream in generate()
        # (active_rollouts is torn down at the root, so the steps must be self-describing).
        for step_index, step in enumerate(agent_loop_output.step_outputs):
            step.env_metrics["rlm_metadata"] = {
                "rid": ctx.rid,
                "parent_rid": ctx.parent_rid,
                "depth": ctx.depth,
                "child_index": ctx.child_index,
                "step_index": step_index,
            }
        if agent_loop_output.step_outputs:
            agent_loop_output.step_outputs[-1].env_metrics["is_trajectory_boundary"] = True
        ctx.output = agent_loop_output

        if ctx.parent_rid is not None:
            for k in ("lm_callback", "subcall_fn"):
                env_extras.pop(k, None)
            return agent_loop_output

        # root: inline descendants (DFS-contiguous) when training children
        descendants = self._dfs_descendants(rid)
        include_children = bool(self._dr_payload.get("per_node_credit", False)) or getattr(
            self.generator_cfg, "train_child_trajectories", False
        )
        if include_children and descendants:
            children_flat: List = []
            for d in descendants:
                if d.output is None:
                    continue
                children_flat.extend(d.output.step_outputs)
            if children_flat:
                agent_loop_output.step_outputs = children_flat + agent_loop_output.step_outputs

        for r in [rid, *(d.rid for d in descendants)]:
            self.active_rollouts.pop(r, None)
        for k in ("lm_callback", "subcall_fn"):
            env_extras.pop(k, None)
        return agent_loop_output

    # ------------------------------------------------------------------
    # generate: un-flatten + per-node RER reward (per_node_credit only)
    # ------------------------------------------------------------------

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        if not bool(self._dr_payload.get("per_node_credit", False)):
            # L1 / eval: stock flatten + broadcast (root scalar)
            out = await super().generate(input_batch, disable_tqdm)
            try:  # telemetry file is opt-in; tracker aggregates are observability-only
                nodes = _nodes_from_env_metrics(out.get("env_metrics") or [])
                _credit_telemetry([{"path": "flat", "nodes": nodes}])
                rm = dict(out.get("rollout_metrics") or {})
                rm.update(_flat_credit_aggregates(nodes))
                out["rollout_metrics"] = rm
            except Exception:
                pass
            return out

        out = await super().generate(input_batch, disable_tqdm)
        if not out.get("is_last_step"):
            return out  # not step-wise (shouldn't happen for RLM) — leave untouched

        env_metrics = out["env_metrics"]
        responses = out["response_ids"]
        in_tids = out["trajectory_ids"]
        is_last = out["is_last_step"]
        n_rows = len(is_last)
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        env_classes_in = input_batch["env_classes"]

        # segment flat rows into per-output blocks (each block ends at its is_last row)
        blocks: List[List[int]] = []
        cur: List[int] = []
        for r in range(n_rows):
            cur.append(r)
            if is_last[r]:
                blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)

        new_tids: List[TrajectoryID] = list(in_tids)
        new_is_last: List[bool] = [False] * n_rows
        new_rewards = list(out["rewards"])
        new_env_classes: List[str] = [None] * n_rows
        _telem_rows: List[Dict[str, Any]] = []  # per-tree credit rows -> file + tracker aggregates

        for bi, rows in enumerate(blocks):
            root_tid = in_tids[rows[0]]
            base_iid = root_tid.instance_id
            base_rep = root_tid.repetition_id
            ledger = self._tree_ledgers.get(root_tid.to_string())  # this tree's evidence ledger
            ec = env_classes_in[bi] if bi < len(env_classes_in) else env_classes_in[-1]
            for r in rows:
                new_env_classes[r] = ec

            # group rows by node rid (rows of a node are contiguous & step-ordered)
            by_rid: "OrderedDict[str, List[int]]" = OrderedDict()
            for r in rows:
                meta = (env_metrics[r] or {}).get("rlm_metadata", {}) or {}
                node_rid = meta.get("rid", f"root-{bi}")
                by_rid.setdefault(node_rid, []).append(r)

            total_tokens = sum(len(responses[r]) for r in rows) or 1
            node_objs: List[RerNode] = []
            node_last_row: Dict[str, int] = {}
            for node_rid, node_rows in by_rid.items():
                metas = [(r, (env_metrics[r] or {}).get("rlm_metadata", {}) or {}) for r in node_rows]
                depth = int(metas[0][1].get("depth", 0))
                parent_rid = metas[0][1].get("parent_rid")
                last_r = max(metas, key=lambda x: int(x[1].get("step_index", 0)))[0]
                node_last_row[node_rid] = last_r
                final_answer = (env_metrics[last_r] or {}).get("final_answer") or ""
                cited_ids = list((env_metrics[last_r] or {}).get("cited_ids") or [])
                # local SUPPORTED verdicts computed in-rollout (root only) — reused as citation
                # precision so the reward path spends Gemini calls only on recall.
                citation_precision = (env_metrics[last_r] or {}).get("citation_precision")
                cost = sum(len(responses[r]) for r in node_rows) / total_tokens
                node_objs.append(
                    RerNode(rid=node_rid, depth=depth, parent_rid=parent_rid,
                            final_answer=final_answer, cost=cost, cited_ids=cited_ids,
                            citation_precision=citation_precision)
                )

            question = _join_prompt(prompts[bi])
            rubrics = (env_extras[bi].get("reward_spec") or {}).get("rubrics") or []
            try:
                res = await compute_rer_rewards(node_objs, rubrics, question, self._dr_payload, ledger=ledger)
            except Exception as e:  # never let a judge hiccup kill a training step
                logger.warning(f"[dr_rlm] RER reward pass failed for block {bi}: {e}; zeroing block rewards")
                res = None

            for seq, (node_rid, node_rows) in enumerate(by_rid.items()):
                depth = int((env_metrics[node_rows[0]] or {}).get("rlm_metadata", {}).get("depth", 0))
                # instance_id MUST stay the plain prompt uid: SkyRL groups GRPO *and*
                # builds mini-batch boundaries by instance_id, asserting each value is
                # contiguous and that #distinct == train_batch_size
                # (preprocess.compute_prompt_mini_batch_boundaries). All of a prompt's
                # nodes share the prompt uid (and are contiguous), forming ONE GRPO group
                # whose baseline is the mean node credit — a clean per-prompt
                # difference-reward. repetition_id is made unique+contiguous per node so
                # the step-wise validator (_validate_step_wise_fields) treats each node as
                # its own trajectory. Depth is carried in rlm_metadata for the optional
                # depth-cohort/weighted rer_pernode estimator (which needs DrRlmTrainer).
                node_tid = TrajectoryID(
                    instance_id=base_iid,
                    repetition_id=base_rep * _REP_STRIDE + seq,
                )
                for r in node_rows:
                    new_tids[r] = node_tid
                    # carry node depth for the optional rer_pernode estimator
                    (env_metrics[r] or {}).setdefault("rlm_metadata", {})["node_depth"] = depth
                last_r = node_last_row[node_rid]
                new_is_last[last_r] = True
                r_a = float(res.rewards.get(node_rid, 0.0)) if res is not None else 0.0
                vec = [0.0] * len(responses[last_r])
                if vec:
                    vec[-1] = r_a
                new_rewards[last_r] = vec

            # free this tree's evidence ledger (snippet texts can be large)
            if ledger is not None:
                with self._ledger_lock:
                    self._tree_ledgers.pop(root_tid.to_string(), None)

            try:  # telemetry file is opt-in; tracker aggregates are observability-only
                _row = {
                    "path": "per_node",
                    "instance_id": str(base_iid),
                    "repetition_id": int(base_rep),
                    "rer_metrics": dict(res.metrics) if res is not None else None,
                    "report_reward": float(res.report_reward) if res is not None else None,
                    "cit_precision": (float(res.cit_precision) if (res is not None and res.cit_precision is not None) else None),
                    "cit_recall": (float(res.cit_recall) if (res is not None and res.cit_recall is not None) else None),
                    "cit_f1": (float(res.cit_f1) if (res is not None and res.cit_f1 is not None) else None),
                    "nodes": [{
                        "rid": n.rid, "depth": n.depth, "n_cited": len(n.cited_ids or []),
                        "ans_len": len(n.final_answer or ""), "cost": float(n.cost),
                        "r_a": float(res.rewards.get(n.rid, 0.0)) if res is not None else None,
                        # answer text (root: 20k, children: 2k) so report-quality trends can be
                        # RE-JUDGED OFFLINE later — the 06-12 smoke's R-decline could not be
                        # disentangled from judge degradation because no text was stored.
                        "ans_text": (n.final_answer or "")[: (20000 if n.depth == 0 else 2000)],
                    } for n in node_objs],
                }
                _telem_rows.append(_row)
                _credit_telemetry([_row])
            except Exception:
                pass

        # JUDGE-OUTAGE GUARD (opt-in via DR_RLM_JUDGE_OUTAGE_GUARD=<frac>): a dead judge
        # silently scores every report 0.0 (DR-Tulu-inherited semantics), which poisons the
        # policy update with all-zero "real" rewards. If >= <frac> of this call's trees have
        # R==0, treat it as an outage: zero EVERY node reward so GRPO advantages vanish and
        # the step becomes a learning no-op instead of training on judge noise.
        # (Observed: Gemini rate-limit death after ~9h in job 23688527 — 311 silent zeros.)
        try:
            import os as _os
            _thr = float(_os.environ.get("DR_RLM_JUDGE_OUTAGE_GUARD", "0") or 0)
            if _thr > 0 and _telem_rows:
                _rs = [t.get("report_reward") for t in _telem_rows if t.get("report_reward") is not None]
                if _rs and (sum(1 for r in _rs if r == 0.0) / len(_rs)) >= _thr:
                    logger.warning(
                        f"[dr_rlm] JUDGE-OUTAGE GUARD tripped: {sum(1 for r in _rs if r == 0.0)}/{len(_rs)} "
                        "trees scored R==0 — zeroing all rewards (no-op update) for this step."
                    )
                    for _i, _v in enumerate(new_rewards):
                        if isinstance(_v, list) and _v:
                            new_rewards[_i] = [0.0] * len(_v)
                    for _t in _telem_rows:
                        _t["judge_outage_guard"] = True
        except Exception as e:
            logger.warning(f"[dr_rlm] judge-outage guard failed (ignored): {e}")

        out["trajectory_ids"] = new_tids
        out["is_last_step"] = new_is_last
        out["rewards"] = new_rewards
        # recompute logged rollout metrics over the corrected per-node rewards
        try:
            out["rollout_metrics"] = get_rollout_metrics(
                responses, new_rewards, env_metrics, new_env_classes, out["loss_masks"]
            )
        except Exception as e:
            logger.warning(f"[dr_rlm] rollout-metrics recompute failed: {e}")
        # merge the per-step credit evidence aggregates (dr_rlm/*) for the tracker (W&B)
        try:
            rm = dict(out.get("rollout_metrics") or {})
            rm.update(_pernode_credit_aggregates(_telem_rows))
            out["rollout_metrics"] = rm
        except Exception as e:
            logger.warning(f"[dr_rlm] credit-aggregate metrics failed: {e}")

        # ---- end-of-step memory hygiene (fix for host-RAM OOM, job 23784592 OOM'd @ step 17 /
        # 720GB) ----
        # Cleanup of active_rollouts (each ctx pins a full StepWiseOutput trajectory) and
        # _tree_ledgers (each pins snippet/doc TEXT) is per-rid only; a missed pop leaks those
        # objects ACROSS steps -> unbounded host RAM. Nothing here is needed past this return, so
        # defensively drop both + force a collection. The residual counts are logged so the leak
        # is observable (and so we can tell if the real leak is elsewhere — Ray/trainer — in which
        # case these stay ~0 but gc.collect still helps release plasma-pinning python refs).
        try:
            _leaked_ctx = len(self.active_rollouts)
            _leaked_led = len(self._tree_ledgers)
            with self._rollout_lock:
                self.active_rollouts.clear()
            with self._ledger_lock:
                self._tree_ledgers.clear()
            gc.collect()
            _rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # KB->GB (Linux)
            logger.info(
                f"[dr_rlm] end-of-step cleanup: freed {_leaked_ctx} leaked rollout-ctx + "
                f"{_leaked_led} leaked ledgers; generator peak RSS {_rss_gb:.1f} GB"
            )
        except Exception as e:
            logger.warning(f"[dr_rlm] end-of-step cleanup failed (non-fatal): {e}")
        return out
