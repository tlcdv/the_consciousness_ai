"""Tests for the RIIU phi pathway (`models/evaluation/phi_riiu.py`)."""
from __future__ import annotations

import pytest
import torch

from models.evaluation.phi_riiu import AutoPhiSurrogate, RIIUPhi


def _make_riiu(dim: int = 32, rank: int = 4, window: int = 16) -> RIIUPhi:
    """Default RIIUPhi for tests. Small dims to keep SVD cheap."""
    return RIIUPhi(dim=dim, rank=rank, window=window, device="cpu")


def test_output_shape_scalar():
    riiu = _make_riiu()
    for _ in range(riiu.rank + 2):
        riiu.push(torch.randn(riiu.dim))
    phi = riiu.compute()
    assert phi.ndim == 0, f"expected 0-D scalar, got shape {tuple(phi.shape)}"


def test_output_range_nonneg():
    riiu = _make_riiu()
    torch.manual_seed(0)
    for _ in range(riiu.rank + 2):
        riiu.push(torch.randn(riiu.dim))
    for _ in range(50):
        riiu.push(torch.randn(riiu.dim))
        v = riiu.compute_value()
        assert v >= 0.0, f"phi must be >= 0, got {v}"


def test_cold_buffer_returns_zero():
    riiu = _make_riiu()
    assert not riiu.is_warm
    assert riiu.compute_value() == 0.0
    # one push: still cold (need rank+1)
    riiu.push(torch.randn(riiu.dim))
    assert not riiu.is_warm
    assert riiu.compute_value() == 0.0


def test_warmup_threshold():
    riiu = _make_riiu()
    for i in range(riiu.rank):
        riiu.push(torch.randn(riiu.dim))
        assert not riiu.is_warm, f"expected cold after {i + 1} pushes (rank={riiu.rank})"
    riiu.push(torch.randn(riiu.dim))
    assert riiu.is_warm, "expected warm after rank+1 pushes"


def test_gradient_flows_through_surrogate():
    """AutoPhiSurrogate produces finite gradients on well-conditioned input.

    SVD backprop is numerically fragile near degenerate singular values
    (input where samples < dims, or low-rank covariance). The training loop
    uses RIIU phi as a detached float reward signal, so this fragility does
    not affect the project; the test asserts the differentiable path works
    for the well-conditioned case (more samples than dims, full rank).
    """
    surrogate = AutoPhiSurrogate(rank=4)
    torch.manual_seed(0)
    z = torch.randn(64, 8, requires_grad=True)
    phi = surrogate(z)
    phi.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all(), (
        "gradients should be finite for well-conditioned (n_samples > dim) input"
    )


def test_svd_stability_identical_input():
    riiu = _make_riiu()
    fixed = torch.ones(riiu.dim)
    for _ in range(riiu.window):
        riiu.push(fixed)
    v = riiu.compute_value()
    # Identical samples produce zero centered covariance; the surrogate should
    # gracefully return 0.0 via the NaN-guard or the SVD-residual on an
    # all-zero covariance.
    assert v == 0.0 or v < 1e-3


def test_svd_stability_orthogonal_input():
    """Cycle through one-hot vectors. SVD must produce finite phi."""
    riiu = _make_riiu(dim=8, rank=2, window=16)
    for step in range(riiu.window):
        z = torch.zeros(riiu.dim)
        z[step % riiu.dim] = 1.0
        riiu.push(z)
    v = riiu.compute_value()
    assert v == v  # not NaN
    assert v >= 0.0
    assert v <= 2.0  # generously bounded; theoretical max is 1.0 for the ratio


def test_low_rank_signal_has_low_phi():
    """Data living entirely in a 2-D subspace should give phi near 0."""
    riiu = _make_riiu(dim=16, rank=4, window=32)
    torch.manual_seed(0)
    basis = torch.randn(2, 16)  # 2 fixed directions
    for _ in range(riiu.window):
        coeffs = torch.randn(2)
        sample = coeffs[0] * basis[0] + coeffs[1] * basis[1]
        riiu.push(sample)
    v = riiu.compute_value()
    assert v < 0.1, f"low-rank signal should have low phi, got {v}"


def test_full_rank_noise_has_higher_phi_than_low_rank():
    """Gaussian noise (full rank) should produce phi above a structured input."""
    torch.manual_seed(0)

    # Low-rank input
    low = _make_riiu(dim=16, rank=2, window=32)
    basis = torch.randn(2, 16)
    for _ in range(low.window):
        coeffs = torch.randn(2)
        low.push(coeffs[0] * basis[0] + coeffs[1] * basis[1])
    phi_low = low.compute_value()

    # Full-rank Gaussian noise on the same dim/rank/window
    high = _make_riiu(dim=16, rank=2, window=32)
    for _ in range(high.window):
        high.push(torch.randn(16))
    phi_high = high.compute_value()

    assert phi_high > phi_low, (
        f"expected full-rank phi > low-rank phi, got {phi_high} vs {phi_low}"
    )


def test_sliding_window_eviction():
    riiu = _make_riiu(dim=8, rank=2, window=16)
    for _ in range(riiu.window + 10):
        riiu.push(torch.randn(riiu.dim))
    assert len(riiu) == riiu.window, (
        f"buffer should be capped at window={riiu.window}, got {len(riiu)}"
    )


def test_reset_clears_buffer():
    riiu = _make_riiu()
    for _ in range(riiu.rank + 2):
        riiu.push(torch.randn(riiu.dim))
    assert riiu.is_warm
    riiu.reset()
    assert not riiu.is_warm
    assert len(riiu) == 0
    assert riiu.compute_value() == 0.0


def test_batch_input_averaged():
    """Pushing [B, dim] averages rows before appending (one sample per push)."""
    riiu = _make_riiu(dim=4, rank=2, window=8)
    batch = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    riiu.push(batch)
    assert len(riiu) == 1
    stored = list(riiu._buffer)[0]
    assert torch.allclose(stored, torch.full((4,), 2.0))


def test_dim_mismatch_raises():
    riiu = _make_riiu(dim=8)
    with pytest.raises(ValueError, match="dim=8"):
        riiu.push(torch.randn(10))


def test_invalid_construction_rejected():
    with pytest.raises(ValueError, match="rank must be positive"):
        RIIUPhi(dim=16, rank=0, window=8)
    with pytest.raises(ValueError, match="window"):
        RIIUPhi(dim=16, rank=8, window=4)  # window <= rank
    with pytest.raises(ValueError, match="dim must be positive"):
        RIIUPhi(dim=0, rank=2, window=8)
