#!/usr/bin/env python3
"""Minimal runnable self-check for the dedup + Mann-Whitney logic in
plot_training_curves.py. Run: python analysis/test_plot_training_curves.py
"""
import json
import tempfile
from pathlib import Path

from plot_training_curves import load_metrics, mann_whitney_p


def test_dedup_keeps_last():
    # 17 rows: 16 unique (instance_id, repetition_id) pairs + 1 crash-duplicate
    # re-roll of the first row with a later ts and a different reward -- the
    # deduped set must keep the re-roll's reward and land on exactly 16 rows.
    rows = [{"ts": i, "instance_id": str(i), "repetition_id": 0, "report_reward": 0.0}
            for i in range(16)]
    rows.append({"ts": 100, "instance_id": "0", "repetition_id": 0, "report_reward": 0.9})
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    deduped = load_metrics(path)
    Path(path).unlink()
    assert len(deduped) == 16, f"expected 16 deduped rows, got {len(deduped)}"
    kept = next(r for r in deduped if r["instance_id"] == "0")
    assert kept["report_reward"] == 0.9, "dedup must keep the LAST occurrence"


def test_mann_whitney_identical_distributions_high_p():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = mann_whitney_p(x, y)
    assert p > 0.9, f"identical samples should give p near 1, got {p}"


def test_mann_whitney_clearly_separated_low_p():
    x = [0.0] * 20
    y = [1.0] * 20
    p = mann_whitney_p(x, y)
    assert p < 0.001, f"fully separated samples should give tiny p, got {p}"


if __name__ == "__main__":
    test_dedup_keeps_last()
    test_mann_whitney_identical_distributions_high_p()
    test_mann_whitney_clearly_separated_low_p()
    print("[ok] all self-checks passed")
