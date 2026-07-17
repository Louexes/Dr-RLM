"""``DrRlmTrainer``: optional trainer that threads per-node depth to the estimator.

Used only when ``trainer.algorithm.advantage_estimator == "rer_pernode"`` (the
depth-cohort / depth-weighted RQ4 variant). For the core L3 result (``grpo``) the stock
``RayPPOTrainer`` is used unchanged — DR-RLM's un-flattening already makes each node its
own GRPO-scored trajectory, so no trainer edit is needed there.

The stock dispatch forwards only a fixed kwarg set to the advantage estimator, so per-row
node depth (needed for the depth-cohort baseline + inverse-frequency weighting) cannot
reach ``rer_pernode`` on its own. This trainer:
  * ``convert_to_training_input`` — copies per-row node depth from
    ``generator_output["env_metrics"][i]["rlm_metadata"]["node_depth"]`` (stamped by
    ``DrRlmGenerator``) into ``data.metadata["node_depth"]``;
  * ``compute_advantages_and_returns`` — reimplements ONLY the step-wise branch of
    ``RayPPOTrainer`` (verbatim from skyrl/train/trainer.py) with one addition: it passes
    ``node_depth=...`` (subselected to last-step rows) through the dispatch to the
    estimator. Everything else (broadcast via cumsum(is_last_step), non-step-wise path)
    is delegated to ``super()``.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from skyrl.train.trainer import RayPPOTrainer
from skyrl.backends.skyrl_train.utils import ppo_utils


class DrRlmTrainer(RayPPOTrainer):
    def convert_to_training_input(self, generator_output, uids):
        data = super().convert_to_training_input(generator_output, uids)
        env_metrics = generator_output.get("env_metrics")
        if env_metrics is not None:
            depths = [int((m or {}).get("rlm_metadata", {}).get("node_depth", 0)) for m in env_metrics]
            # only attach if it lines up with the (possibly filtered) training rows
            n_rows = len(data.metadata.get("is_last_step", [])) or len(uids)
            if len(depths) == n_rows:
                data.metadata["node_depth"] = depths
            else:
                # CRITICAL: without node_depth the rer_pernode estimator silently degrades to a
                # prompt-only baseline (== GRPO over node rewards). Under a trained root (full_R)
                # that re-introduces the shared-baseline crush that flatlines child citing. Warn
                # loudly rather than degrade silently.
                logger.warning(
                    f"[DrRlmTrainer] node_depth length {len(depths)} != training rows {n_rows}; "
                    "NOT threading depth — rer_pernode will fall back to a prompt-only baseline "
                    "(per-role separation DISABLED). Child citing signal is at risk under full_R."
                )
        else:
            logger.warning(
                "[DrRlmTrainer] generator_output has no 'env_metrics'; cannot thread node_depth — "
                "rer_pernode degrades to a prompt-only baseline (per-role separation DISABLED)."
            )
        return data

    @torch.no_grad()
    def compute_advantages_and_returns(self, data):
        step_wise = self.cfg.generator.step_wise_trajectories
        node_depth = data.metadata.get("node_depth")
        if not step_wise or node_depth is None:
            return super().compute_advantages_and_returns(data)

        # ---- step-wise branch (verbatim from RayPPOTrainer) + node_depth threading ----
        token_level_rewards = data["rewards"]
        is_last_step = torch.tensor(data.metadata["is_last_step"], dtype=torch.bool)
        index = np.array(data.metadata["uids"])
        depths = np.array(node_depth)
        values = data["values"]
        last_idx = is_last_step.cpu().numpy()

        last_step_response_mask = data["response_mask"][is_last_step]
        last_step_advantages, last_step_returns = ppo_utils.compute_advantages_and_returns(
            token_level_rewards=token_level_rewards[is_last_step],
            response_mask=torch.ones_like(last_step_response_mask, dtype=torch.float),
            index=index[last_idx],
            adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
            values=values[is_last_step] if values is not None else None,
            config=self.cfg.trainer.algorithm,
            gamma=self.cfg.trainer.algorithm.gamma,
            lambd=self.cfg.trainer.algorithm.lambd,
            grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            node_depth=depths[last_idx],  # <-- the one addition vs. the base method
        )
        traj_ids = (
            torch.cat([torch.tensor([False], device=is_last_step.device), is_last_step[:-1]]).int().cumsum(dim=0)
        )
        num_traj = traj_ids[-1].item() + 1
        assert num_traj == len(last_step_advantages), (
            f"num_traj {num_traj} != #trajectories from is_last_step {len(last_step_advantages)}; "
            "is_last_step is likely malformed"
        )
        response_mask_float = data["response_mask"].to(last_step_advantages.dtype)
        data["advantages"] = last_step_advantages[traj_ids] * response_mask_float
        data["returns"] = last_step_returns[traj_ids] * response_mask_float
        return data
