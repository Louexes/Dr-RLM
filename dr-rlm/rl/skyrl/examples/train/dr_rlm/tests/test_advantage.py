"""Tests for the optional rer_pernode advantage estimator: shapes, cohort baseline,
and depth weighting. Skips if torch/skyrl are unavailable."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skyrl.backends.skyrl_train.utils.ppo_utils")

from examples.train.dr_rlm.rer_advantage import compute_rer_pernode_advantage


def _rewards(vals):
    # one nonzero reward per row at the last token (mimics per-node terminal reward)
    t = torch.zeros(len(vals), 4)
    for i, v in enumerate(vals):
        t[i, -1] = v
    return t


def test_shapes_and_per_prompt_baseline_no_depth():
    rew = _rewards([1.0, 0.0, 0.5, 0.5])
    mask = torch.ones(4, 4)
    index = np.array(["q", "q", "q", "q"])
    adv, ret = compute_rer_pernode_advantage(rew, mask, index, grpo_norm_by_std=False, node_depth=None)
    assert adv.shape == rew.shape and ret.shape == rew.shape
    # without depth, cohort == prompt; centered advantages sum to ~0 over the group
    per_row = adv.sum(dim=-1)
    assert float(per_row.sum()) == pytest.approx(0.0, abs=1e-5)


def test_depth_cohort_baseline(monkeypatch):
    monkeypatch.setenv("DR_RLM_DEPTH_WEIGHTING", "0")  # isolate the cohort baseline
    # depth0 rows: {2.0, 0.0} -> mean 1.0 ; depth1 rows: {0.4,0.6,0.5,0.5} -> mean 0.5
    rew = _rewards([2.0, 0.0, 0.4, 0.6, 0.5, 0.5])
    mask = torch.ones(6, 4)
    index = np.array(["q"] * 6)
    depth = [0, 0, 1, 1, 1, 1]
    adv, _ = compute_rer_pernode_advantage(rew, mask, index, grpo_norm_by_std=False, node_depth=depth)
    s = adv.sum(dim=-1) / mask.sum(dim=-1)  # per-token advantage (estimator broadcasts over seqlen)
    # centered within depth cohort: depth0 -> {+1,-1}, depth1 -> {-0.1,+0.1,0,0}
    assert float(s[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(s[1]) == pytest.approx(-1.0, abs=1e-5)
    assert float(s[2]) == pytest.approx(-0.1, abs=1e-5)
    assert float(s[3]) == pytest.approx(0.1, abs=1e-5)


def test_depth_weighting_changes_magnitude(monkeypatch):
    monkeypatch.setenv("DR_RLM_DEPTH_WEIGHTING", "1")
    rew = _rewards([2.0, 0.0, 0.4, 0.6, 0.5, 0.5])
    mask = torch.ones(6, 4)
    index = np.array(["q"] * 6)
    depth = [0, 0, 1, 1, 1, 1]  # N_0=2, N_1=4 -> deep nodes upweighted
    adv, _ = compute_rer_pernode_advantage(rew, mask, index, grpo_norm_by_std=False, node_depth=depth)
    s = adv.sum(dim=-1) / mask.sum(dim=-1)  # per-token advantage (estimator broadcasts over seqlen)
    # depth-0 weight = scale/N_0, depth-1 weight = scale/N_1, scale=6/(2*(1/2)+4*(1/4))=3
    # w0 = 3/2 = 1.5 -> row0 advantage = 1.0*1.5 = 1.5
    assert float(s[0]) == pytest.approx(1.5, abs=1e-5)


# --- per-role-separation fix (2026-06-23): keep the depth-cohort BASELINE, default the
# inverse-frequency WEIGHTING off, so a trained root (full_R) doesn't starve child citing ---

def _full_r_batch():
    """A realistic 1-prompt batch: 4 trees x (root + 4 children). Roots carry a large full_R
    report reward; children carry small ledger_support credit (a few citing, most orphan)."""
    roots = [0.8, 0.6, 0.5, 0.3]
    kids = [[0.2, 0, 0, 0], [0.15, 0.1, 0, 0], [0, 0, 0, 0], [0.1, 0, 0, 0]]
    vals, depth = [], []
    for r, ks in zip(roots, kids):
        vals.append(r); depth.append(0)
        vals.extend(ks); depth.extend([1, 1, 1, 1])
    return _rewards(vals), torch.ones(len(vals), 4), np.array(["q"] * len(vals)), depth, vals


def _child_grad_share(adv, depth):
    s = adv.sum(dim=-1).abs()
    d = np.array(depth)
    child = float(s[torch.tensor(d == 1)].sum())
    root = float(s[torch.tensor(d == 0)].sum())
    return child / (child + root)


def test_default_weighting_is_off(monkeypatch):
    """Default (no env var) must NOT apply depth weighting — the fix. With a large full_R root,
    leaving weighting on would halve the children's gradient share."""
    monkeypatch.delenv("DR_RLM_DEPTH_WEIGHTING", raising=False)
    rew, mask, index, depth, _ = _full_r_batch()
    adv_default, _ = compute_rer_pernode_advantage(rew, mask, index, node_depth=depth)
    monkeypatch.setenv("DR_RLM_DEPTH_WEIGHTING", "0")
    adv_off, _ = compute_rer_pernode_advantage(rew, mask, index, node_depth=depth)
    assert torch.allclose(adv_default, adv_off)


def test_cohort_baseline_preserves_child_citing_under_full_R(monkeypatch):
    """Under a trained root (full_R), the depth-cohort baseline must keep citing children with a
    clearly positive advantage AND keep the children's collective gradient share high (>0.7).
    Turning the weighting on ~halves that share (the v3/Run A failure)."""
    rew, mask, index, depth, vals = _full_r_batch()
    is_cite = torch.tensor([d == 1 and v > 0 for d, v in zip(depth, vals)])

    monkeypatch.setenv("DR_RLM_DEPTH_WEIGHTING", "0")
    adv_off, _ = compute_rer_pernode_advantage(rew, mask, index, node_depth=depth)
    monkeypatch.setenv("DR_RLM_DEPTH_WEIGHTING", "1")
    adv_on, _ = compute_rer_pernode_advantage(rew, mask, index, node_depth=depth)

    # citing children get a positive mean advantage either way (cohort baseline rescues direction)
    assert float(adv_off.sum(dim=-1)[is_cite].mean()) > 0.5
    # but the children's gradient share collapses toward 50% when weighting is on
    share_off = _child_grad_share(adv_off, depth)
    share_on = _child_grad_share(adv_on, depth)
    assert share_off > 0.7, f"weighting-off child share {share_off:.2f} should stay high"
    assert share_on < 0.6, f"weighting-on child share {share_on:.2f} should collapse toward 0.5"
    assert share_off - share_on > 0.15
