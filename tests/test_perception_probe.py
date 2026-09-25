"""Value-based tests for the perception-decodability probe helper.

These test only the linear-decoding contract (no env, no model weights):
  - linearly separable classes decode well above chance
  - random labels decode at ~chance
  - the return contract and the skip path for degenerate inputs
  - NaN input raises
and the error contract of `_compute_broadcast`, with stubs in place of the models.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from scripts.analysis.probe_perception_decodability import _compute_broadcast, linear_decode


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


def test_nan_input_raises_instead_of_falling_back_to_torch():
    # The sklearn path used to sit in `except Exception`, so the ValueError sklearn
    # raises on NaN sent the data to a torch probe that trained on it and reported
    # an accuracy. The torch path is now reserved for a missing sklearn.
    X, y = _separable(seed=5)
    X[3, 2] = np.nan
    with pytest.raises(ValueError):
        linear_decode(X, y, seed=5)


# --- _compute_broadcast, the workspace forward shared by six probes ---------------

class _Tectum:
    def get_capsule_payload(self):
        return None


class _Reentrant:
    def __init__(self, broadcast=None, error=None):
        self.broadcast, self.error = broadcast, error

    def settle(self, **kwargs):
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(broadcast_content=self.broadcast)


def _broadcast(reentrant, ws_dim=8):
    config = {"device": "cpu", "workspace_dim": ws_dim}
    return _compute_broadcast(config, _Tectum(), None, reentrant, None, None, None,
                              torch.zeros(1, ws_dim), 0.5, None)


def test_a_failing_forward_raises_instead_of_returning_none():
    # It returned None on ANY exception, and three probes (PCI, workspace ordering,
    # gate attenuation) turned None into a zero broadcast and kept measuring.
    with pytest.raises(RuntimeError, match="settle failed"):
        _broadcast(_Reentrant(error=RuntimeError("settle failed")))


def test_an_empty_broadcast_is_zeros_as_in_training():
    # A step that does not ignite returns {}; the training loop feeds zeros then.
    out = _broadcast(_Reentrant(broadcast={}))
    assert out.shape == (8,)
    assert not out.any()


def test_a_tensor_broadcast_is_returned_flat():
    tensor = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    out = _broadcast(_Reentrant(broadcast={"tensor": tensor, "source": "tectum"}))
    assert out.tolist() == list(range(8))
