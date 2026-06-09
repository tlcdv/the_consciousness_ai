"""Value-based tests for the perception-decodability probe helper.

These test only the linear-decoding contract (no env, no model weights):
  - linearly separable classes decode well above chance
  - random labels decode at ~chance
  - the return contract and the skip path for degenerate inputs
"""
from __future__ import annotations

import numpy as np

from scripts.analysis.probe_perception_decodability import linear_decode


def _separable(n_per_class=120, dim=16, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(n_classes, dim))
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.normal(scale=1.0, size=(n_per_class, dim)))
        y.extend([f"class{c}"] * n_per_class)
    return np.vstack(X), np.array(y)


def test_separable_classes_decode_well_above_chance():
    X, y = _separable(seed=1)
    res = linear_decode(X, y, seed=1)
    # 4 well-separated clusters: a linear probe should be near-perfect and far
    # above the 0.25 uniform chance.
    assert res["n_classes"] == 4
    assert res["uniform_chance"] == 0.25
    assert res["test_acc"] > 0.85


def test_random_labels_decode_near_chance():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 16))
    y = rng.integers(0, 4, size=400)  # labels independent of X
    res = linear_decode(X, y, seed=2)
    # No real structure: held-out accuracy should sit near uniform chance (0.25).
    # Allow slack for finite-sample noise, but it must not look "decodable".
    assert res["test_acc"] < 0.45


def test_return_contract_and_skip_path():
    X, y = _separable(seed=3)
    res = linear_decode(X, y, seed=3)
    for key in ("test_acc", "uniform_chance", "majority", "n", "n_classes", "method"):
        assert key in res
    assert res["n"] == len(y)
    assert 0.0 <= res["majority"] <= 1.0

    # Single class -> skip path returns NaN test_acc without raising.
    Xs = np.random.default_rng(4).normal(size=(20, 8))
    ys = np.array(["only"] * 20)
    skip = linear_decode(Xs, ys, seed=4)
    assert skip["n_classes"] == 1
    assert skip["method"] == "skip"
    assert np.isnan(skip["test_acc"])
