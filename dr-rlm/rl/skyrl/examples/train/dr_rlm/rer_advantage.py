"""Optional custom advantage estimator for DR-RLM: ``rer_pernode``.

For the core L3 result you do NOT need this. Once ``DrRlmGenerator`` un-flattens the
tree, every node is its own step-wise trajectory carrying its provenance credit r_a, and
all of a prompt's nodes share the prompt's ``instance_id`` (one contiguous GRPO group —
required by SkyRL's mini-batch boundary logic). Stock ``grpo`` then baselines each node
against the *mean node credit for that prompt* — a clean per-prompt difference-reward.

``rer_pernode`` adds two RAO-style refinements (proposal §6.5 / RQ4):
  * a **depth-cohort baseline** — center each node against same-depth peers (not against
    the depth-mixed prompt mean), removing the depth reward-scale bias. This is the
    **per-role baseline separation** that lets a *trained* root (``full_R`` / grounding-aware
    root reward) coexist with the child citing signal: children are centred against children,
    so a large root reward no longer raises the children's baseline. **Always on** when
    depth is available.
  * **depth inverse-frequency weighting** ``w_d = alpha / N_d`` (renormalized to unit mean)
    to stop rare *deep* nodes being drowned out by many shallow ones — designed for DEEP
    recursion. ⚠️ **OFF by default.** Our recursion is shallow (depth-0 root + depth-1
    children), so the *rare* depth is the ROOT and the *common* depth is the CHILDREN; the
    weighting then up-weights the root and down-weights the children — the opposite of what
    we want. Numerically it ~halves the children's collective gradient share (≈83%→50% on a
    4-tree prompt), which starved the child citing signal in runs v3/Run A (orphans stuck
    ~85% vs v1-clean's 43%). Enable (``DR_RLM_DEPTH_WEIGHTING=1``) ONLY for genuinely deep
    trees. See docs/RL_TRAINING_LOG.md (per-role-separation fix, 2026-06-23).

Both need per-row node depth, which the stock dispatch does not forward. ``DrRlmTrainer``
(used automatically when this estimator is selected) threads it in as ``node_depth``.
If ``node_depth`` is absent (e.g. run under the stock trainer), this degrades to a plain
per-``index`` (prompt) baseline — equivalent to GRPO over the node rewards — which, under a
*trained* root (``full_R``), RE-INTRODUCES the shared-baseline crush; so ``DrRlmTrainer``
now warns loudly if depth fails to thread rather than silently degrading. Env vars:
``DR_RLM_DEPTH_WEIGHT_ALPHA`` (default 1.0), ``DR_RLM_DEPTH_WEIGHTING`` (default **0**).

Registration happens at import (the DR-RLM package imports this module), so the name is
in the registry before ``validate_cfg`` and is cloudpickled into the Ray registry actor.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator


def compute_rer_pernode_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
    node_depth=None,
    **kwargs,
):
    """Per-node difference-reward advantage with optional depth-cohort baseline +
    inverse-frequency depth weighting. Returns ``(advantages, returns)`` of shape
    ``(batch, seqlen)`` like every SkyRL estimator. ``node_depth`` (per-row int array,
    threaded by ``DrRlmTrainer``) enables the depth refinements; if None, falls back to
    a per-``index`` baseline (== GRPO over node rewards)."""
    alpha = float(os.environ.get("DR_RLM_DEPTH_WEIGHT_ALPHA", "1.0"))
    # Default OFF: for shallow recursion the inverse-frequency weighting down-weights the
    # (common) children — the citing signal we want — and up-weights the (rare) root. It
    # halves the children's gradient share and starved citing in v3/Run A. Only enable for
    # genuinely deep trees. The depth-cohort BASELINE below is the per-role separation and
    # stays on regardless.
    do_depth_weight = os.environ.get("DR_RLM_DEPTH_WEIGHTING", "0") not in ("0", "false", "False")

    scores = token_level_rewards.sum(dim=-1)  # (batch,)
    bsz = scores.shape[0]
    have_depth = node_depth is not None and len(node_depth) == bsz
    depths = [int(d) for d in node_depth] if have_depth else [0] * bsz

    with torch.no_grad():
        # 1) baseline within each cohort. cohort = (prompt index, depth) when depth is
        #    available, else just the prompt index (-> plain GRPO over node rewards).
        cohort_key = [(index[i], depths[i]) if have_depth else index[i] for i in range(bsz)]
        c2scores = defaultdict(list)
        for i in range(bsz):
            c2scores[cohort_key[i]].append(scores[i])
        c2mean, c2std = {}, {}
        for k, vals in c2scores.items():
            if len(vals) > 1:
                t = torch.stack(vals)
                c2mean[k], c2std[k] = t.mean(), t.std()
            else:
                c2mean[k], c2std[k] = torch.tensor(0.0), torch.tensor(1.0)

        adv = scores.clone()
        for i in range(bsz):
            centered = scores[i] - c2mean[cohort_key[i]]
            adv[i] = centered / (c2std[cohort_key[i]] + epsilon) if grpo_norm_by_std else centered

        # 2) depth inverse-frequency weighting w_d = alpha / N_d, renormalized to mean 1
        if have_depth and do_depth_weight:
            n_d = defaultdict(int)
            for d in depths:
                n_d[d] += 1
            raw = torch.tensor([1.0 / n_d[d] for d in depths], dtype=adv.dtype)
            scale = bsz / float(raw.sum().clamp(min=epsilon))  # mean weight == 1
            adv = adv * (alpha * scale * raw)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


# Idempotent registration: the package __init__ registers this estimator on import, but a
# direct re-import (e.g. under pytest, or a second Ray worker import) must not crash. The
# stock decorator raises ValueError on a duplicate name, so register guardedly instead.
try:
    register_advantage_estimator("rer_pernode")(compute_rer_pernode_advantage)
except ValueError:
    pass  # already registered (benign re-import)
