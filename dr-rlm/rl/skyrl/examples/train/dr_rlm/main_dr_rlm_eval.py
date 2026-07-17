"""Eval-only entry point for DR-RLM (in-harness rollout eval).

Mirrors ``examples/train/rlm/main_rlm_eval.py``. Runs generate-only over the eval
dataset using ``DrRlmGenerator`` so the recursive hooks fire. For the canonical
controlled comparison set ``generator.per_node_credit=false`` (eval doesn't need
per-node credit — it scores the report); the per-axis latency/compute/quality eval of
the untrained recursive arm lives in ``eval/eval_recursive.py`` (rlm/ inference).
"""

import asyncio
import sys

import ray
from loguru import logger

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg

import examples.train.dr_rlm  # noqa: F401  (registers env + estimator)
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig
from examples.train.dr_rlm.dr_rlm_generator import DrRlmGenerator


DrRlmConfig = make_config(generator_cls=DrRlmGeneratorConfig)


class DrRlmEvalEntrypoint(EvalOnlyEntrypoint):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return DrRlmGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    async def run(self, inference_engine_client) -> dict:
        """Same flow as ``EvalOnlyEntrypoint.run``, with two W&B fixes:

        1. The tracker is created BEFORE eval runs (the stock entrypoint creates it
           after, so the run is invisible on W&B until completion and then holds a
           single step-0 point that charts render as empty).
        2. The per-batch ``dr_rlm/*`` credit aggregates are logged as step-indexed
           curves, derived from the credit-telemetry file — ``evaluate_step_wise``
           drops ``rollout_metrics``, so without this nothing but the final eval
           scalars ever reaches W&B in eval-only mode.
        """
        import json as _json
        import os as _os

        from skyrl.train.evaluate import evaluate, evaluate_step_wise
        from skyrl.train.utils.trainer_utils import build_dataloader

        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"
        tracker = self.get_tracker()  # EARLY: the W&B run exists while eval is running

        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        eval_fn = evaluate_step_wise if self.cfg.generator.step_wise_trajectories else evaluate
        results = await eval_fn(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        # step-indexed dr_rlm/* curves from the credit telemetry (one step per rollout batch)
        step = 0
        tpath = _os.environ.get("DR_RLM_CREDIT_METRICS_PATH", "")
        if tpath and _os.path.exists(tpath):
            try:
                from examples.train.dr_rlm.dr_rlm_generator import (
                    _flat_credit_aggregates,
                    _pernode_credit_aggregates,
                )

                with open(tpath) as f:
                    rows = [_json.loads(line) for line in f if line.strip()]
                for step, row in enumerate(rows):
                    if row.get("path") == "flat":
                        m = _flat_credit_aggregates(row.get("nodes") or [])
                    else:
                        m = _pernode_credit_aggregates([row])
                    if m:
                        tracker.log(m, step=step, commit=True)
                step = len(rows)
            except Exception as e:  # observability must never fail the eval
                logger.warning(f"[dr_rlm eval] telemetry->tracker logging failed: {e}")

        tracker.log(results, step=step, commit=True)
        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg) -> dict:
    exp = DrRlmEvalEntrypoint(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = DrRlmConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"DR-RLM eval metrics: {metrics}")


if __name__ == "__main__":
    main()
