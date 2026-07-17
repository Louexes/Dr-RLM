"""Registers the internal gym envs."""

from skyrl_gym.envs.registration import deregister, register

register(
    id="aime",
    entry_point="skyrl_gym.envs.aime.env:AIMEEnv",
)

register(
    id="gsm8k",
    entry_point="skyrl_gym.envs.gsm8k.env:GSM8kEnv",
)

register(
    id="gsm8k_multi_turn",
    entry_point="skyrl_gym.envs.gsm8k.multi_turn_env:GSM8kMultiTurnEnv",
)

register(
    id="text2sql",
    entry_point="skyrl_gym.envs.sql.env:SQLEnv",
)

register(
    id="search",
    entry_point="skyrl_gym.envs.search.env:SearchEnv",
)

register(
    id="lcb",
    entry_point="skyrl_gym.envs.lcb.env:LCBEnv",
)

register(
    id="searchcode",
    entry_point="skyrl_gym.envs.searchcode.env:SearchCodeEnv",
)

# DR-RLM project envs. Registered here (lazily, by entry-point string) so the ids are known in
# EVERY process that imports skyrl_gym -- including the stock SkyRLGymGenerator's Ray rollout
# actor, which never imports examples.train.dr_rlm. The recursive arm's custom generator pulls
# the registration in via its own import chain; the flat (dr_tulu) arm uses the stock generator,
# so without this its rollout worker raised "No registered env with id: dr_tulu". Guarded so it
# no-ops if examples.train.dr_rlm's __init__ already registered them in-process. The DrTuluEnv /
# DrRlmEnv classes are imported only at make() time, when the working dir is on the actor's path.
from skyrl_gym.envs.registration import registry as _registry  # noqa: E402

if "dr_rlm" not in _registry:
    register(id="dr_rlm", entry_point="examples.train.dr_rlm.dr_rlm_env:DrRlmEnv")
if "dr_tulu" not in _registry:
    register(id="dr_tulu", entry_point="examples.train.dr_rlm.dr_tulu_env:DrTuluEnv")

__all__ = [
    "deregister",
    "register",
]
