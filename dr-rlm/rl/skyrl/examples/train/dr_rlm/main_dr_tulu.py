"""Training entry point for the flat DR-Tulu baseline arm.

Unlike ``main_dr_rlm.py`` this needs NO custom generator or trainer: the DR-Tulu ReAct
episode is an ordinary tag-based multi-turn text env (``dr_tulu_env.DrTuluEnv``) driven
by the stock ``SkyRLGymGenerator`` (``use_conversation_multi_turn`` + ``sampling_params.
stop=['</call_tool>','</call>']``) and scored with a scalar terminal reward under stock
GRPO — exactly the original Open-Instruct optimization shape (see
docs/DRTULU_RL_FAITHFULNESS_AUDIT.md). Importing the package registers the env.

Run: configs/run_dr_tulu_flat.sh
"""

import sys

import ray

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

import examples.train.dr_rlm  # noqa: F401  (registers the dr_tulu env)


DrTuluConfig = make_config()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = DrTuluConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
