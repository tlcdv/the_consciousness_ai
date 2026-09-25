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
