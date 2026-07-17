"""The load-bearing test: the per-node un-flattened layout DrRlmGenerator emits must
satisfy SkyRL's real step-wise validator (contiguity + is_last_step at boundaries),
because the trainer's cumsum(is_last_step) trajectory mapping silently corrupts credit
otherwise. We construct the exact layout DrRlmGenerator.generate produces for a
2-rollout, depth-1 tree and assert the real validator accepts it (and rejects a broken
variant)."""

import pytest

# skip cleanly if SkyRL isn't importable in this environment
tu = pytest.importorskip("skyrl.train.utils.trainer_utils")
base = pytest.importorskip("skyrl.train.generators.base")

TrajectoryID = base.TrajectoryID
_validate = tu._validate_step_wise_fields

_STRIDE = 100_000


def _tid(uid, base_rep, seq):
    return TrajectoryID(instance_id=uid, repetition_id=base_rep * _STRIDE + seq)


def _good_layout():
    """Two rollouts of prompt 'q'. Rollout r0: child c0 (2 steps) then root (2 steps).
    Rollout r1: child c1 (1 step) then root (2 steps). Children are DFS-prepended before
    their root (as _post_process does), each node a contiguous block."""
    uid = "q"
    rows = []
    # rollout 0 (base_rep=0): child seq0 (2 steps), root seq1 (2 steps)
    c0, r0 = _tid(uid, 0, 0), _tid(uid, 0, 1)
    rows += [(c0, False), (c0, True), (r0, False), (r0, True)]
    # rollout 1 (base_rep=1): child seq0 (1 step), root seq1 (2 steps)
    c1, r1 = _tid(uid, 1, 0), _tid(uid, 1, 1)
    rows += [(c1, True), (r1, False), (r1, True)]
    tids = [t for t, _ in rows]
    is_last = [b for _, b in rows]
    return tids, is_last


def _gen_output(tids, is_last):
    n = len(tids)
    return {
        "trajectory_ids": tids,
        "is_last_step": is_last,
        "response_ids": [[1, 2] for _ in range(n)],
        "rewards": [[0.0, 0.0] for _ in range(n)],
        "loss_masks": [[1, 1] for _ in range(n)],
        "prompt_token_ids": [[0] for _ in range(n)],
        "stop_reasons": ["stop"] * n,
        "rollout_logprobs": None,
        "rollout_metrics": {},
        "env_metrics": [{} for _ in range(n)],
        "rollout_expert_indices": None,
    }


def test_per_node_layout_passes_validator():
    tids, is_last = _good_layout()
    # must not raise
    _validate(_gen_output(tids, is_last), num_responses=len(tids))


def test_broken_is_last_step_is_rejected():
    tids, _ = _good_layout()
    # only the global last step marked -> boundary between nodes without is_last -> invalid
    broken = [False] * len(tids)
    broken[-1] = True
    with pytest.raises(AssertionError):
        _validate(_gen_output(tids, broken), num_responses=len(tids))


def test_noncontiguous_trajectory_is_rejected():
    uid = "q"
    a, b = _tid(uid, 0, 0), _tid(uid, 0, 1)
    # a, b, a -> 'a' reappears non-contiguously
    tids = [a, b, a]
    is_last = [True, True, True]
    with pytest.raises(AssertionError):
        _validate(_gen_output(tids, is_last), num_responses=3)
