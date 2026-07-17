"""Training entry point for DR-RLM (recursive deep-research RLM).

Mirrors ``examples/train/rlm/main_rlm.py`` but wires the DR-RLM generator config and
the un-flattening per-node-credit generator. Importing this module registers the
``dr_rlm`` env and the ``rer_pernode`` advantage estimator (so both are present in the
Ray driver and, via the package import, in workers).

Run (see configs/run_dr_rlm_L*.sh):
  uv run --isolated --extra vllm -m examples.train.dr_rlm.main_dr_rlm \
      environment.env_class=dr_rlm \
      generator.per_node_credit=true generator.reward_mode=rer \
      trainer.algorithm.advantage_estimator=grpo ...
"""

import sys

import ray

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg

import examples.train.dr_rlm  # noqa: F401  (registers env + estimator)
from examples.train.dr_rlm.dr_rlm_config import DrRlmGeneratorConfig
from examples.train.dr_rlm.dr_rlm_generator import DrRlmGenerator
from examples.train.dr_rlm.dr_rlm_trainer import DrRlmTrainer


DrRlmConfig = make_config(generator_cls=DrRlmGeneratorConfig)


class DrRlmPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return DrRlmGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        # The depth-weighted per-node estimator needs per-row node depth threaded to it;
        # use DrRlmTrainer only then. For grpo/rloo (incl. the core L3 path) the stock
        # trainer is correct as-is (un-flattening already gives per-node GRPO scoring).
        trainer_cls = DrRlmTrainer if cfg.trainer.algorithm.advantage_estimator == "rer_pernode" else RayPPOTrainer
        return trainer_cls(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    DrRlmPPOExp(cfg).run()


def main() -> None:
    cfg = DrRlmConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
