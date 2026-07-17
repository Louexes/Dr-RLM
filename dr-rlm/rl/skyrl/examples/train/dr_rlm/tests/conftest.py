"""Pytest config for DR-RLM unit tests.

Tests run WITHOUT a GPU or network: the rubric judge is monkeypatched, retrieval uses
the dependency-free ``local_jsonl`` backend, and torch/skyrl-dependent tests
``importorskip`` so they are skipped (not failed) where those heavy deps are absent.

Run from the SkyRL repo root:  pytest examples/train/dr_rlm/tests
"""

import os
import sys

# Make ``examples.train.dr_rlm`` importable (SkyRL repo root = 5 levels up from this file:
# .../SkyRL/examples/train/dr_rlm/tests/conftest.py -> .../SkyRL)
_SKYRL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _SKYRL_ROOT not in sys.path:
    sys.path.insert(0, _SKYRL_ROOT)

# deterministic estimator behavior in tests
os.environ.setdefault("DR_RLM_DEPTH_WEIGHT_ALPHA", "1.0")
