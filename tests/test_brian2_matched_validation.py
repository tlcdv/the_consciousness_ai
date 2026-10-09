"""Tests for the matched AKOrN versus Brian2 comparison.

The older validate_binding compares AKOrN against a different system (nonzero
omega against a zero legacy rotation, sender-only amplitude, a 1/N factor and
an unrelated time axis). The matched comparison integrates the continuous limit
of the layer's own update, so phase trajectories can be compared directly.
"""
import math

import numpy as np
import pytest
import torch

from models.core.oscillatory_binding import KuramotoLayer
from models.validation import brian2_binding_validation as bv

needs_brian2 = pytest.mark.skipif(not bv.BRIAN2_AVAILABLE, reason="Brian2 not installed")


def _layer(natural_frequency: bool, seed: int = 0) -> KuramotoLayer:
    torch.manual_seed(seed)
    return KuramotoLayer(num_oscillators=4, dimensions=2, coupling_strength=1.0,
                         natural_frequency_std=1.0, dt=0.05,
                         natural_frequency=natural_frequency)


def test_effective_frequencies_are_zero_for_legacy_layer():
    # Arrange
    layer = _layer(natural_frequency=False)

    # Act
    freqs = bv.effective_natural_frequencies(layer)

    # Assert: the legacy update applies no rotation at all.
    assert np.array_equal(freqs, np.zeros(4))


def test_effective_frequencies_read_the_generator_forward_uses():
    # Arrange: forward() uses omega = P - P^T, so w = P[1,0] - P[0,1].
    layer = _layer(natural_frequency=True)
    P = layer.natural_frequencies.detach().numpy()

    # Act
    freqs = bv.effective_natural_frequencies(layer)

    # Assert
    assert np.allclose(freqs, P[:, 1, 0] - P[:, 0, 1])


def test_effective_frequencies_refuse_higher_dimensions():
    # Arrange
    layer = KuramotoLayer(num_oscillators=3, dimensions=3, natural_frequency=True)

    # Act and Assert: no scalar frequency exists for D > 2, so no proxy is returned.
    with pytest.raises(ValueError):
        bv.effective_natural_frequencies(layer)


@needs_brian2
@pytest.mark.parametrize("natural_frequency", [False, True])
def test_matched_phase_error_shrinks_with_step_size(natural_frequency):
    # Arrange: the same 2-unit time span at dt and dt/2. The Euler-plus-normalize
    # step has a global error of order dt, so a matched system must converge.
    amplitudes = np.array([1.0, 0.8, 0.6, 0.4])
    errors = []

    # Act
    for dt in (0.025, 0.0125):
        torch.manual_seed(0)
        layer = KuramotoLayer(num_oscillators=4, dimensions=2, coupling_strength=1.0,
                              natural_frequency_std=1.0, dt=dt,
                              natural_frequency=natural_frequency)
        comparison = bv.validate_binding_matched(layer, amplitudes=amplitudes,
                                                 total_steps=int(round(2.0 / dt)),
                                                 brian2_dt_ms=0.5)
        errors.append(comparison.max_phase_error)

    # Assert
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 0.01


@needs_brian2
def test_matched_comparison_detects_a_frequency_mismatch():
    # Arrange: integrate the fixed layer but compare against the legacy one.
    fixed = _layer(natural_frequency=True)
    legacy = _layer(natural_frequency=False)
    amplitudes = np.ones(4)

    # Act
    ok = bv.validate_binding_matched(fixed, amplitudes=amplitudes, total_steps=40)
    wrong = bv.validate_binding_matched(legacy, amplitudes=amplitudes, total_steps=40,
                                        brian2_frequencies=bv.effective_natural_frequencies(fixed))

    # Assert: the check must fail when the two systems differ.
    assert wrong.max_phase_error > 10 * ok.max_phase_error


# ---------------------------------------------------------------------------
# LIF bridge layer against Brian2. Brian2 stays optional.
#
# Closed form for a constant current I above threshold, v_rest = v_reset = 0:
#   first spike at t* = tau ln(I / (I - v_th)). Euler steps lag this by at most one dt.
# Count tolerance against the continuous time reference: 1 spike plus 5 percent.
# ---------------------------------------------------------------------------

LIF = dict(tau=0.02, v_th=1.0, v_rest=0.0, v_reset=0.0, dt=0.001)


def _constant_currents(steps: int = 400, seed: int = 0) -> np.ndarray:
    rates = np.random.default_rng(seed).uniform(1.2, 3.0, (1, 8))
    return np.tile(rates, (steps, 1))


def _within_tolerance(bridge: np.ndarray, reference: np.ndarray) -> bool:
    diff = np.abs(bridge.sum(0) - reference.sum(0))
    return bool(np.all(diff <= 1 + 0.05 * reference.sum(0)))


def test_bridge_first_spike_matches_closed_form_without_brian2():
    # Arrange
    currents = _constant_currents()

    # Act
    spikes = bv._bridge_spike_train(currents, **LIF)

    # Assert: step k is tested after k + 1 updates, so the spike time is (k + 1) dt,
    # and it lies within one dt of tau ln(I / (I - v_th)).
    first = (spikes.argmax(0) + 1) * LIF["dt"]
    expected = LIF["tau"] * np.log(currents[0] / (currents[0] - LIF["v_th"]))
    assert np.all(np.abs(first - expected) <= LIF["dt"] + 1e-12)


@needs_brian2
def test_bridge_matches_brian2_euler_at_the_same_dt():
    # Arrange
    currents = np.random.default_rng(1).uniform(0.5, 4.0, (400, 8))

    # Act
    result = bv.validate_lif_bridge(currents, **LIF)

    # Assert
    assert result.matched_mismatch_fraction == 0.0


@needs_brian2
def test_bridge_counts_match_continuous_time_reference():
    # Arrange
    currents = _constant_currents()

    # Act
    result = bv.validate_lif_bridge(currents, **LIF)

    # Assert
    assert _within_tolerance(result.bridge_spikes, result.reference_spikes)


@needs_brian2
def test_wrong_drive_fails_the_tolerance():
    # Arrange: Brian2 reference driven at 1.5 times the current the bridge sees.
    currents = _constant_currents()
    bridge = bv._bridge_spike_train(currents, **LIF)

    # Act
    wrong = bv._brian2_spike_train(1.5 * currents, **LIF, method="rk4", refine=10)

    # Assert: the check can fail.
    assert not _within_tolerance(bridge, wrong)


@needs_brian2
def test_validate_lif_bridge_raises_without_brian2(monkeypatch):
    monkeypatch.setattr(bv, "BRIAN2_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        bv.validate_lif_bridge(_constant_currents(10))
