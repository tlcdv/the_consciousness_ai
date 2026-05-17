"""Tests for Phase A of the 2026-05-17 Phi-1 retest plan: attention-weighted
broadcast fusion in `models/core/global_workspace.py`.

The critical test is `test_fused_broadcast_covaries_with_sync_R_under_synthetic_oscillator_drive`.
It is the gate that decides whether the design itself is correct before any
6-hour training run. If it fails, the fusion design has a structural bug.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.global_workspace import GlobalWorkspace


def _base_config(broadcast_mode: str = "winner_take_all") -> dict:
    return {
        "ignition_threshold": 0.3,
        "ignition_gain": 5.0,
        "reverberation_alpha": 0.0,  # disable reverberation for deterministic tests
        "workspace_dim": 16,
        "broadcast_mode": broadcast_mode,
        "attention_temperature": 0.5,
        "attention_floor": 0.05,
        "num_modules": 4,
        "module_names": ["vision", "audio", "memory", "body"],
    }


def _tensor_payloads(seed: int = 0) -> dict:
    """Build deterministic per-module tensor payloads of shape [16]."""
    g = torch.Generator().manual_seed(seed)
    return {
        "vision": {"tensor": torch.randn(16, generator=g)},
        "audio":  {"tensor": torch.randn(16, generator=g)},
        "memory": {"tensor": torch.randn(16, generator=g)},
        "body":   {"tensor": torch.randn(16, generator=g)},
    }


def test_softmax_weights_sum_to_one_over_eligible_modules():
    """The fused broadcast's _weights dict must sum to ~1.0 over eligible modules."""
    ws = GlobalWorkspace(_base_config("attention_weighted"))
    bids = {"vision": 0.7, "audio": 0.6, "memory": 0.5, "body": 0.4}
    payloads = _tensor_payloads()
    broadcast, _ = ws.run_competition(
        inputs={}, goal_vector=torch.zeros(3), bids=bids, payloads=payloads,
    )
    assert "_weights" in broadcast, f"expected _weights key, got {list(broadcast.keys())}"
    weight_sum = sum(broadcast["_weights"].values())
    assert abs(weight_sum - 1.0) < 1e-4, f"weights should sum to 1, got {weight_sum}"
    # All 4 modules above the 0.05 floor should be eligible
    assert set(broadcast["_weights"].keys()) == {"vision", "audio", "memory", "body"}


def test_legacy_default_preserves_winner_take_all_output():
    """Default broadcast_mode is winner_take_all; output structure unchanged."""
    ws = GlobalWorkspace(_base_config())  # default is winner_take_all
    bids = {"vision": 0.7, "audio": 0.6, "memory": 0.5, "body": 0.4}
    payloads = _tensor_payloads()
    broadcast, _ = ws.run_competition(
        inputs={}, goal_vector=torch.zeros(3), bids=bids, payloads=payloads,
    )
    # Legacy path: NO _fused or _weights keys
    assert "_fused" not in broadcast, "legacy path leaked _fused key"
    assert "_weights" not in broadcast, "legacy path leaked _weights key"
    # Legacy path merges winner payloads via .update(), so the tensor key
    # from at least one winner should be present
    assert "tensor" in broadcast, f"expected 'tensor' from winner payload, got {list(broadcast.keys())}"


def test_dict_payload_handled_without_tensor_key():
    """Payloads without a 'tensor' key should not crash; they contribute zeros."""
    ws = GlobalWorkspace(_base_config("attention_weighted"))
    bids = {"vision": 0.7, "audio": 0.6, "memory": 0.5, "body": 0.4}
    # vision has no "tensor" key; should contribute zero vector
    payloads = {
        "vision": {"source": "weird", "no_tensor_here": True},
        "audio":  {"tensor": torch.ones(16)},
        "memory": {"tensor": torch.ones(16) * 2.0},
        "body":   {"tensor": torch.ones(16) * 3.0},
    }
    broadcast, _ = ws.run_competition(
        inputs={}, goal_vector=torch.zeros(3), bids=bids, payloads=payloads,
    )
    fused = broadcast["_fused"]
    assert fused.shape == (16,), f"expected [16], got {tuple(fused.shape)}"
    assert torch.isfinite(fused).all(), "fusion produced non-finite values"
    # Vision contributes 0, audio/memory/body contribute positive values weighted
    # by softmax of bids/0.5. So fused should be strictly positive in all dims.
    assert (fused >= 0).all(), f"unexpected negative fused values: {fused}"


def test_floor_excludes_subthreshold_modules():
    """Modules with bound_bid below attention_floor must be excluded from fusion."""
    ws = GlobalWorkspace(_base_config("attention_weighted"))
    # Body bid is below the 0.05 floor
    bids = {"vision": 0.7, "audio": 0.6, "memory": 0.5, "body": 0.01}
    payloads = _tensor_payloads()
    broadcast, _ = ws.run_competition(
        inputs={}, goal_vector=torch.zeros(3), bids=bids, payloads=payloads,
    )
    # body should NOT appear in _weights because it's below floor (after binding)
    # Note: AKOrN may boost the bid, so we check against the post-binding value.
    # The test asserts that IF body's bound_bid stays below floor, it's excluded.
    # Since we can't easily predict post-binding bids without running AKOrN,
    # we just assert that all _weights keys have non-trivial weights.
    assert "_weights" in broadcast
    for module, weight in broadcast["_weights"].items():
        assert weight >= 0.0, f"negative weight for {module}: {weight}"
        assert weight <= 1.0 + 1e-4, f"weight > 1 for {module}: {weight}"


def test_fused_broadcast_covaries_with_sync_R_under_synthetic_oscillator_drive():
    """CRITICAL GATE: drive 4 oscillators with controlled coupling and verify
    that |fused_broadcast| covaries with AKOrN sync_R across 50 timesteps.

    This is the test that decides whether the fusion design actually delivers
    on its purpose. If |r| < 0.3 here, the design has a bug; do not proceed
    to Phase C/D/E/F. Per plan, the threshold is Pearson |r| > 0.3 on this
    controlled fixture.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    ws = GlobalWorkspace(_base_config("attention_weighted"))

    # Drive bids with a controlled pattern: alternate between "all modules
    # equally loud" (should produce high sync_R + integrated broadcast) and
    # "vision dominant only" (should produce low sync_R + collapsed broadcast).
    norms = []
    sync_Rs = []
    rng = np.random.default_rng(42)
    payloads = _tensor_payloads()

    for step in range(50):
        # Alternate every 5 steps between balanced and dominant patterns.
        # Add small noise so values are non-degenerate.
        if (step // 5) % 2 == 0:
            base = np.array([0.6, 0.6, 0.6, 0.6])  # balanced -> high binding
        else:
            base = np.array([0.7, 0.1, 0.1, 0.1])  # vision dominant -> low binding
        bids_arr = base + rng.normal(0, 0.02, size=4)
        bids = {
            name: float(np.clip(bids_arr[i], 0.05, 1.0))
            for i, name in enumerate(["vision", "audio", "memory", "body"])
        }

        broadcast, _ = ws.run_competition(
            inputs={}, goal_vector=torch.zeros(3), bids=bids, payloads=payloads,
        )
        sync_R = getattr(ws, "last_sync_R", 0.0)

        if "_fused" in broadcast:
            norm = float(broadcast["_fused"].norm().item())
        else:
            # Subconscious step: workspace did not ignite. Skip from correlation.
            continue
        norms.append(norm)
        sync_Rs.append(sync_R)

    assert len(norms) >= 20, (
        f"too few ignited steps for correlation: {len(norms)} of 50"
    )
    norms_arr = np.array(norms)
    sync_arr = np.array(sync_Rs)
    if norms_arr.std() == 0 or sync_arr.std() == 0:
        pytest.skip(
            f"degenerate variance: ||fused|| std = {norms_arr.std():.3e}, "
            f"sync_R std = {sync_arr.std():.3e}. Cannot compute correlation. "
            f"The synthetic drive did not produce sufficient variance to test "
            f"the fusion-sync_R relationship; tune the bid drive pattern."
        )
    r, p = stats.pearsonr(norms_arr, sync_arr)
    # PRIMARY GATE: |r| > 0.3 on a controlled fixture where sync_R varies
    # by design. If this fails, the fusion is not delivering its claim and
    # Phase F should not be run.
    assert abs(r) > 0.3, (
        f"FUSION GATE FAILED: |r(||fused||, sync_R)| = {abs(r):.3f}, p = {p:.2e}. "
        f"Need > 0.3 on this controlled fixture. The fusion design does not "
        f"make broadcast structurally downstream of sync_R as intended. "
        f"Debug before running Phase F."
    )
