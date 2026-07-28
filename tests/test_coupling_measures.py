"""
Tests for models/evaluation/coupling_measures.py.

Every expectation is derived from the definition of the estimator on a constructed
signal, never from a run. The constructions are the standard validation cases:

  PLV   two sinusoids with a fixed lag -> ~1; independent noise -> ~0
  PTE   a signal that literally copies another's past -> asymmetric in the right
        direction; two independent signals -> symmetric within the surrogate null
  PAC   a fast carrier whose envelope is driven by a slow phase -> high modulation
        index; the same carrier at constant envelope -> ~0

The units are cycles per step throughout. These tests deliberately assert nothing
about Hz, because the module maps to no sampling rate.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evaluation.coupling_measures import (  # noqa: E402
    PTEResult,
    amplitude_envelope,
    analytic_phase,
    bandpass,
    phase_amplitude_coupling,
    phase_locking_value,
    phase_transfer_entropy,
)

N = 2048
SLOW = (0.01, 0.05)
FAST = (0.15, 0.35)


def _sine(freq: float, phase: float = 0.0, n: int = N) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2.0 * np.pi * freq * t + phase)


class TestBandpass:
    def test_passes_an_in_band_tone_and_removes_an_out_of_band_one(self):
        in_band = _sine(0.03)
        out_of_band = _sine(0.30)
        assert bandpass(in_band, *SLOW).std() > 0.5 * in_band.std()
        assert bandpass(out_of_band, *SLOW).std() < 0.05 * out_of_band.std()

    def test_rejects_invalid_band_edges(self):
        x = _sine(0.03)
        with pytest.raises(ValueError, match="band edges"):
            bandpass(x, 0.3, 0.1)
        with pytest.raises(ValueError, match="band edges"):
            bandpass(x, 0.1, 0.9)  # above the once-per-step Nyquist limit

    def test_rejects_signals_that_are_too_short(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            bandpass(np.array([1.0, 2.0]), 0.1, 0.2)


class TestAnalyticSignal:
    def test_envelope_of_a_pure_tone_is_flat(self):
        env = amplitude_envelope(_sine(0.05))
        interior = env[100:-100]  # edges ring, as any Hilbert transform does
        assert np.allclose(interior, 1.0, atol=0.05)

    def test_phase_advances_at_the_signal_frequency(self):
        freq = 0.05
        phase = np.unwrap(analytic_phase(_sine(freq)))
        interior = phase[100:-100]
        measured = np.mean(np.diff(interior)) / (2.0 * np.pi)
        assert measured == pytest.approx(freq, abs=0.005)

    def test_handles_odd_and_even_lengths(self):
        for n in (255, 256):
            env = amplitude_envelope(_sine(0.05, n=n))
            assert env.shape == (n,)
            assert np.all(np.isfinite(env))


class TestPhaseLockingValue:
    def test_constant_lag_gives_plv_near_one(self):
        a = _sine(0.03)
        b = _sine(0.03, phase=np.pi / 3)
        assert phase_locking_value(a, b, SLOW) > 0.99

    def test_identical_signals_give_plv_of_one(self):
        a = _sine(0.03)
        assert phase_locking_value(a, a, SLOW) == pytest.approx(1.0)

    def test_independent_noise_gives_low_plv(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=N)
        b = rng.normal(size=N)
        assert phase_locking_value(a, b, SLOW) < 0.35

    def test_is_symmetric(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=N)
        b = rng.normal(size=N)
        assert phase_locking_value(a, b, SLOW) == pytest.approx(
            phase_locking_value(b, a, SLOW)
        )

    def test_is_insensitive_to_the_size_of_the_lag(self):
        # PLV measures lag CONSTANCY, not lag size.
        base = _sine(0.03)
        small = phase_locking_value(base, _sine(0.03, phase=0.1), SLOW)
        large = phase_locking_value(base, _sine(0.03, phase=2.5), SLOW)
        assert small == pytest.approx(large, abs=0.02)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            phase_locking_value(_sine(0.03, n=256), _sine(0.03, n=512), SLOW)


class TestPhaseTransferEntropy:
    def test_returns_a_result_with_the_surrogate_null_exposed(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=N)
        b = rng.normal(size=N)
        result = phase_transfer_entropy(a, b, SLOW)
        assert isinstance(result, PTEResult)
        assert result.n_surrogates == 20
        assert result.surrogate_std >= 0.0

    def test_a_driven_signal_shows_the_asymmetry_in_the_right_direction(self):
        # target is a delayed copy of source plus noise, so source's past predicts
        # target's future but not the reverse.
        rng = np.random.default_rng(3)
        source = rng.normal(size=N)
        delay = 4
        target = np.roll(source, delay) + 0.3 * rng.normal(size=N)
        forward = phase_transfer_entropy(source, target, SLOW, delay=delay)
        backward = phase_transfer_entropy(target, source, SLOW, delay=delay)
        assert forward.corrected > backward.corrected

    def test_independent_signals_are_symmetric_within_the_null(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=N)
        b = rng.normal(size=N)
        forward = phase_transfer_entropy(a, b, SLOW)
        backward = phase_transfer_entropy(b, a, SLOW)
        gap = abs(forward.corrected - backward.corrected)
        pooled = max(forward.surrogate_std, backward.surrogate_std, 1e-9)
        assert gap < 5.0 * pooled

    def test_correction_reduces_the_raw_estimate(self):
        # The raw estimate is positively biased on finite records; that is exactly
        # why the corrected value is the one to compare across conditions.
        rng = np.random.default_rng(5)
        a = rng.normal(size=N)
        b = rng.normal(size=N)
        result = phase_transfer_entropy(a, b, SLOW)
        assert result.surrogate_mean > 0.0
        assert result.corrected < result.pte

    def test_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=N)
        b = rng.normal(size=N)
        first = phase_transfer_entropy(a, b, SLOW, seed=99)
        second = phase_transfer_entropy(a, b, SLOW, seed=99)
        assert first.corrected == second.corrected

    def test_invalid_delay_raises(self):
        a, b = _sine(0.03), _sine(0.04)
        with pytest.raises(ValueError, match="delay must be >= 1"):
            phase_transfer_entropy(a, b, SLOW, delay=0)

    def test_signal_shorter_than_the_delay_raises(self):
        short = _sine(0.1, n=8)
        with pytest.raises(ValueError, match="too short"):
            phase_transfer_entropy(short, short, delay=20)


class TestPhaseAmplitudeCoupling:
    def _coupled_pair(self, seed: int = 7):
        """A fast carrier whose envelope is driven by a slow oscillation."""
        rng = np.random.default_rng(seed)
        t = np.arange(N)
        slow = np.sin(2.0 * np.pi * 0.02 * t)
        envelope = 1.0 + 0.9 * slow
        fast = envelope * np.sin(2.0 * np.pi * 0.25 * t)
        return slow + 0.01 * rng.normal(size=N), fast

    def test_coupled_signals_give_a_positive_modulation_index(self):
        slow, fast = self._coupled_pair()
        assert phase_amplitude_coupling(slow, fast, SLOW, FAST) > 0.05

    def test_uncoupled_carrier_gives_a_near_zero_modulation_index(self):
        rng = np.random.default_rng(8)
        t = np.arange(N)
        slow = np.sin(2.0 * np.pi * 0.02 * t) + 0.01 * rng.normal(size=N)
        flat_carrier = np.sin(2.0 * np.pi * 0.25 * t)  # constant envelope
        coupled = phase_amplitude_coupling(slow, *self._coupled_pair()[1:], SLOW, FAST)
        uncoupled = phase_amplitude_coupling(slow, flat_carrier, SLOW, FAST)
        assert uncoupled < coupled

    def test_overlapping_bands_raise_rather_than_return_a_number(self):
        slow, fast = self._coupled_pair()
        with pytest.raises(ValueError, match="strictly slower"):
            phase_amplitude_coupling(slow, fast, (0.1, 0.3), (0.2, 0.4))

    def test_is_non_negative(self):
        rng = np.random.default_rng(9)
        for _ in range(5):
            a = rng.normal(size=N)
            b = rng.normal(size=N)
            assert phase_amplitude_coupling(a, b, SLOW, FAST) >= 0.0

    def test_length_mismatch_raises(self):
        slow, _ = self._coupled_pair()
        with pytest.raises(ValueError, match="length mismatch"):
            phase_amplitude_coupling(slow, _sine(0.25, n=512), SLOW, FAST)


class TestUnitsDiscipline:
    """The module must not silently accept a band outside the once-per-step Nyquist."""

    def test_band_above_nyquist_is_rejected_everywhere(self):
        x = _sine(0.05)
        with pytest.raises(ValueError):
            analytic_phase(x, (0.4, 0.8))
        with pytest.raises(ValueError):
            amplitude_envelope(x, (0.6, 0.9))

    def test_module_docstring_carries_the_units_warning(self):
        # Load-bearing: the no-Hz-claims rule is enforced by this docstring, so a
        # refactor that drops it should fail here rather than silently.
        import models.evaluation.coupling_measures as cm

        # Normalize wrapping so the assertion pins the wording, not the line breaks.
        doc = " ".join(cm.__doc__.split())
        assert "CYCLES PER STEP" in doc
        assert "NO Hz interpretation" in doc
        assert "no Hz interpretation" not in doc.replace("NO Hz interpretation", "")
