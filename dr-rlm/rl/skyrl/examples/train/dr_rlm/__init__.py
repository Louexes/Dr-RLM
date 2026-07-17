"""DR-RLM: recursive deep-research RLM training/eval on SkyRL.

Importing this package has two registration side effects (mirrors how
``examples/train/rlm/multi_paper_env`` registers its envs):

  * registers the ``dr_rlm`` gym env id  -> ``DrRlmEnv``
  * registers the ``rer_pernode`` advantage estimator (optional, depth-weighted)

Both are guarded so the package still imports in light contexts (data conversion, SFT
generation, offline reward unit tests) where the full SkyRL/torch RL stack is absent —
the registrations simply no-op there, and fire when the training entry points import the
package inside the SkyRL runtime (and its Ray workers).
"""

try:
    from skyrl_gym.envs.registration import register, registry

    if "dr_rlm" not in registry:
        register(id="dr_rlm", entry_point="examples.train.dr_rlm.dr_rlm_env:DrRlmEnv")
    # flat DR-Tulu baseline arm (ReAct, composite reward) — see dr_tulu_env.py
    if "dr_tulu" not in registry:
        register(id="dr_tulu", entry_point="examples.train.dr_rlm.dr_tulu_env:DrTuluEnv")
except Exception:  # skyrl_gym not installed (e.g. data-prep / unit-test context)
    pass

try:
    from . import rer_advantage  # noqa: F401  (registers the rer_pernode estimator)
except Exception:  # torch / skyrl_train not installed
    pass
