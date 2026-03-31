"""
Tests for Brian2 biological validation of AKOrN oscillatory binding.

Tests are structured in two groups:
1. Translation logic and AKOrN simulation (always runs, no Brian2 needed)
2. Brian2 comparison (skipped when Brian2 is not installed)
"""

import unittest
import math
import numpy as np
import torch
from unittest.mock import patch

from models.core.oscillatory_binding import KuramotoLayer, WorkspaceBindingSystem
from models.validation.brian2_binding_validation import (
    BRIAN2_AVAILABLE,
    SyncCurve,
    ValidationResult,
    run_akorn_simulation,
    validate_binding,
    translate_akorn_params,
    _extract_akorn_natural_frequencies,
    _extract_akorn_coupling_matrix,
    _interpolate_to_common_times,
)


class TestParameterTranslation(unittest.TestCase):
    """Test that AKOrN parameters are correctly extracted for Brian2."""

    def setUp(self):
        torch.manual_seed(42)
        self.kuramoto = KuramotoLayer(
            num_oscillators=4,
            dimensions=2,
            coupling_strength=2.0,
            natural_frequency_std=0.3,
            dt=0.1,
        )

    def test_natural_frequency_extraction_2d(self):
        """2D oscillators: natural freq = off-diagonal of skew-symmetric matrix."""
        freqs = _extract_akorn_natural_frequencies(self.kuramoto)
        self.assertEqual(freqs.shape, (4,))
        # Skew-symmetric: omega[i, 0, 1] = -omega[i, 1, 0]
        omega = self.kuramoto.natural_frequencies.detach().cpu().numpy()
        for i in range(4):
            expected = omega[i, 1, 0]
            self.assertAlmostEqual(freqs[i], expected, places=5)

    def test_natural_frequency_extraction_higher_d(self):
        """Higher dimension: use Frobenius norm proxy."""
        torch.manual_seed(7)
        k = KuramotoLayer(num_oscillators=3, dimensions=4, natural_frequency_std=0.5)
        freqs = _extract_akorn_natural_frequencies(k)
        self.assertEqual(freqs.shape, (3,))
        # All frequencies should be non-negative (Frobenius norm)
        for f in freqs:
            self.assertGreaterEqual(f, 0.0)

    def test_coupling_matrix_extraction(self):
        """Coupling matrix shape and values match KuramotoLayer."""
        matrix = _extract_akorn_coupling_matrix(self.kuramoto)
        self.assertEqual(matrix.shape, (4, 4))
        expected = self.kuramoto.coupling_weights.detach().cpu().numpy()
        np.testing.assert_allclose(matrix, expected, atol=1e-6)

    def test_translate_akorn_params_keys(self):
        """translate_akorn_params returns all required keys."""
        params = translate_akorn_params(self.kuramoto)
        required_keys = {
            "num_oscillators", "dimensions", "coupling_strength",
            "natural_frequencies", "coupling_matrix", "dt",
        }
        self.assertEqual(set(params.keys()), required_keys)
        self.assertEqual(params["num_oscillators"], 4)
        self.assertEqual(params["dimensions"], 2)
        self.assertAlmostEqual(params["coupling_strength"], 2.0)
        self.assertAlmostEqual(params["dt"], 0.1)

    def test_coupling_strength_scalar(self):
        """Coupling strength extracted as float."""
        params = translate_akorn_params(self.kuramoto)
        self.assertIsInstance(params["coupling_strength"], float)


class TestAKOrNSimulation(unittest.TestCase):
    """Test the AKOrN simulation wrapper produces valid SyncCurves."""

    def setUp(self):
        torch.manual_seed(42)
        self.kuramoto = KuramotoLayer(
            num_oscillators=5,
            dimensions=2,
            coupling_strength=2.0,
            natural_frequency_std=0.1,
            dt=0.1,
        )

    def test_sync_curve_shape(self):
        """SyncCurve has correct shapes for times, R, and phases."""
        N = 5
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * math.pi, N)
        init = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        amps = np.ones(N)

        curve = run_akorn_simulation(self.kuramoto, init, amps, total_steps=50)

        self.assertEqual(curve.times.shape, (50,))
        self.assertEqual(curve.order_parameter.shape, (50,))
        self.assertEqual(curve.phases.shape, (50, N))
        self.assertEqual(curve.source, "akorn")

    def test_sync_curve_r_bounded(self):
        """Order parameter R stays in [0, 1]."""
        N = 5
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * math.pi, N)
        init = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        amps = np.ones(N)

        curve = run_akorn_simulation(self.kuramoto, init, amps, total_steps=100)

        self.assertTrue(np.all(curve.order_parameter >= 0.0))
        self.assertTrue(np.all(curve.order_parameter <= 1.0 + 1e-6))

    def test_high_coupling_synchronizes(self):
        """High coupling strength should increase R over time."""
        torch.manual_seed(42)
        k = KuramotoLayer(
            num_oscillators=5,
            dimensions=2,
            coupling_strength=5.0,
            natural_frequency_std=0.05,
            dt=0.1,
        )
        N = 5
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * math.pi, N)
        init = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        amps = np.ones(N)

        curve = run_akorn_simulation(k, init, amps, total_steps=100)

        # Final R should be higher than initial R
        self.assertGreater(curve.order_parameter[-1], curve.order_parameter[0])

    def test_zero_coupling_no_sync(self):
        """Zero coupling: oscillators rotate independently, R stays low."""
        torch.manual_seed(42)
        k = KuramotoLayer(
            num_oscillators=8,
            dimensions=2,
            coupling_strength=0.0,
            natural_frequency_std=0.5,
            dt=0.1,
        )
        N = 8
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * math.pi, N)
        init = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        amps = np.ones(N)

        curve = run_akorn_simulation(k, init, amps, total_steps=100)

        # With many oscillators and no coupling, R should stay well below 1
        self.assertLess(np.mean(curve.order_parameter), 0.8)

    def test_amplitude_modulation(self):
        """Non-uniform amplitudes change the synchronization dynamics."""
        torch.manual_seed(42)
        k = KuramotoLayer(
            num_oscillators=4,
            dimensions=2,
            coupling_strength=3.0,
            natural_frequency_std=0.1,
            dt=0.1,
        )
        N = 4
        rng = np.random.RandomState(0)
        angles = rng.uniform(0, 2 * math.pi, N)
        init = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

        # Uniform amps
        curve_uniform = run_akorn_simulation(k, init, np.ones(N), total_steps=50)
        # Sparse amps (only first two active)
        sparse_amps = np.array([1.0, 1.0, 0.0, 0.0])
        curve_sparse = run_akorn_simulation(k, init, sparse_amps, total_steps=50)

        # Different amplitude patterns should produce different R curves
        diff = np.abs(curve_uniform.order_parameter - curve_sparse.order_parameter)
        self.assertGreater(np.max(diff), 0.01)

    def test_times_monotonic(self):
        """Time values are strictly increasing."""
        N = 5
        init = np.stack([np.ones(N), np.zeros(N)], axis=-1)  # All at phase 0
        amps = np.ones(N)
        curve = run_akorn_simulation(self.kuramoto, init, amps, total_steps=20)
        diffs = np.diff(curve.times)
        self.assertTrue(np.all(diffs > 0))


class TestInterpolation(unittest.TestCase):
    """Test the time-grid interpolation utility."""

    def test_same_length_curves(self):
        """Two curves with same time points: no interpolation needed."""
        t = np.linspace(0, 1, 50)
        r_a = np.sin(t * math.pi)
        r_b = np.cos(t * math.pi)
        curve_a = SyncCurve(times=t, order_parameter=r_a, phases=np.zeros((50, 4)), source="a")
        curve_b = SyncCurve(times=t, order_parameter=r_b, phases=np.zeros((50, 4)), source="b")

        common_t, interp_a, interp_b = _interpolate_to_common_times(curve_a, curve_b)
        np.testing.assert_allclose(interp_a, r_a, atol=1e-10)
        np.testing.assert_allclose(interp_b, r_b, atol=1e-10)

    def test_different_length_curves(self):
        """Shorter curve used as reference grid; longer curve interpolated."""
        t_short = np.linspace(0, 1, 20)
        t_long = np.linspace(0, 1, 100)
        r_short = np.sin(t_short * math.pi)
        r_long = np.sin(t_long * math.pi)

        curve_a = SyncCurve(times=t_short, order_parameter=r_short, phases=np.zeros((20, 4)), source="a")
        curve_b = SyncCurve(times=t_long, order_parameter=r_long, phases=np.zeros((100, 4)), source="b")

        common_t, interp_a, interp_b = _interpolate_to_common_times(curve_a, curve_b)
        self.assertEqual(len(common_t), 20)
        # Interpolated values should be close to sin curve
        np.testing.assert_allclose(interp_b, np.sin(common_t * math.pi), atol=0.05)


class TestValidateBindingWithoutBrian2(unittest.TestCase):
    """Test validate_binding when Brian2 is not available."""

    def setUp(self):
        torch.manual_seed(42)
        self.kuramoto = KuramotoLayer(
            num_oscillators=4,
            dimensions=2,
            coupling_strength=2.0,
            natural_frequency_std=0.1,
            dt=0.1,
        )

    @unittest.skipIf(BRIAN2_AVAILABLE, "Test only runs when Brian2 is NOT installed")
    def test_graceful_skip(self):
        """Without Brian2, validation returns method='skipped' with a warning."""
        with patch("models.validation.brian2_binding_validation.warnings") as mock_warnings:
            result = validate_binding(self.kuramoto, total_steps=20)
        mock_warnings.warn.assert_called_once()
        call_args = mock_warnings.warn.call_args
        self.assertIs(call_args[0][1], UserWarning)

        self.assertEqual(result.method, "skipped")
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.akorn_curve)
        self.assertIsNone(result.brian2_curve)
        self.assertTrue(math.isnan(result.correlation))
        self.assertIn("not installed", result.message)

    def test_akorn_curve_always_present(self):
        """Even without Brian2, the AKOrN curve is computed."""
        result = validate_binding(self.kuramoto, total_steps=20)
        self.assertIsNotNone(result.akorn_curve)
        self.assertEqual(result.akorn_curve.source, "akorn")
        self.assertEqual(len(result.akorn_curve.times), 20)
        self.assertIsInstance(result.final_r_akorn, float)


class TestValidationResultStructure(unittest.TestCase):
    """Test the ValidationResult dataclass."""

    def test_dataclass_fields(self):
        """ValidationResult contains all expected fields."""
        curve = SyncCurve(
            times=np.array([0.0, 0.1]),
            order_parameter=np.array([0.5, 0.8]),
            phases=np.zeros((2, 4)),
            source="akorn",
        )
        result = ValidationResult(
            akorn_curve=curve,
            brian2_curve=None,
            correlation=0.95,
            max_deviation=0.05,
            mean_deviation=0.02,
            final_r_akorn=0.8,
            final_r_brian2=None,
            passed=True,
            method="brian2",
            message="ok",
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.method, "brian2")
        self.assertAlmostEqual(result.correlation, 0.95)


class TestSyncCurve(unittest.TestCase):
    """Test the SyncCurve dataclass."""

    def test_fields(self):
        curve = SyncCurve(
            times=np.linspace(0, 1, 10),
            order_parameter=np.random.rand(10),
            phases=np.random.rand(10, 4),
            source="test",
        )
        self.assertEqual(curve.source, "test")
        self.assertEqual(curve.times.shape, (10,))
        self.assertEqual(curve.phases.shape, (10, 4))


@unittest.skipUnless(BRIAN2_AVAILABLE, "Brian2 not installed")
class TestBrian2Integration(unittest.TestCase):
    """
    Integration tests that require Brian2.
    Skipped in CI where Brian2 is not installed.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.kuramoto = KuramotoLayer(
            num_oscillators=4,
            dimensions=2,
            coupling_strength=3.0,
            natural_frequency_std=0.1,
            dt=0.1,
        )

    def test_brian2_simulation_runs(self):
        """Brian2 simulation completes and returns valid SyncCurve."""
        from models.validation.brian2_binding_validation import run_brian2_simulation

        freqs = _extract_akorn_natural_frequencies(self.kuramoto)
        matrix = _extract_akorn_coupling_matrix(self.kuramoto)
        N = 4
        rng = np.random.RandomState(42)
        phases = rng.uniform(0, 2 * math.pi, N)
        amps = np.ones(N)

        curve = run_brian2_simulation(
            num_oscillators=N,
            natural_frequencies=freqs,
            coupling_strength=3.0,
            coupling_matrix=matrix,
            amplitudes=amps,
            initial_phases=phases,
            duration_seconds=0.5,
            dt_ms=1.0,
        )

        self.assertEqual(curve.source, "brian2")
        self.assertGreater(len(curve.times), 0)
        self.assertTrue(np.all(curve.order_parameter >= 0))
        self.assertTrue(np.all(curve.order_parameter <= 1.0 + 1e-6))

    def test_full_validation_passes(self):
        """Full AKOrN vs Brian2 comparison with high coupling."""
        result = validate_binding(
            self.kuramoto,
            total_steps=50,
            brian2_duration=0.5,
            brian2_dt_ms=1.0,
            correlation_threshold=0.7,
        )

        self.assertEqual(result.method, "brian2")
        self.assertIsNotNone(result.brian2_curve)
        self.assertFalse(math.isnan(result.correlation))
        # With matched parameters, curves should correlate
        self.assertGreater(result.correlation, 0.5)

    def test_validation_result_has_both_curves(self):
        """Both AKOrN and Brian2 curves are present in result."""
        result = validate_binding(self.kuramoto, total_steps=30, brian2_duration=0.3)
        self.assertIsNotNone(result.akorn_curve)
        self.assertIsNotNone(result.brian2_curve)
        self.assertEqual(result.akorn_curve.source, "akorn")
        self.assertEqual(result.brian2_curve.source, "brian2")


class TestWorkspaceBindingTranslation(unittest.TestCase):
    """Test translation from WorkspaceBindingSystem level."""

    def setUp(self):
        torch.manual_seed(42)
        self.binding = WorkspaceBindingSystem(num_modules=4, iterations=10)
        self.binding.register_modules(["vision", "audio", "memory", "body"])

    def test_extract_from_binding_system(self):
        """Can extract parameters from the WorkspaceBindingSystem's inner KuramotoLayer."""
        params = translate_akorn_params(self.binding.kuramoto)
        self.assertEqual(params["num_oscillators"], 4)
        self.assertEqual(params["dimensions"], 2)

    def test_binding_system_phases_match(self):
        """Running bind_bids then extracting phases gives consistent state."""
        bids = {"vision": 0.8, "audio": 0.6, "memory": 0.4, "body": 0.3}
        self.binding.bind_bids(bids)

        # After one bind call, current_phases should be set
        self.assertIsNotNone(self.binding.current_phases)
        self.assertEqual(self.binding.current_phases.shape, (1, 4, 2))


if __name__ == "__main__":
    unittest.main()
