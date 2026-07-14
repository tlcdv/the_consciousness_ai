"""
Tests for Effective Information (EI) computation.

Validates:
1. Deterministic TPM produces maximal EI
2. Identity/uniform TPM produces zero or near-zero EI
3. Random trajectories produce intermediate EI
4. Gate vs workspace comparison returns valid structure
5. Discretization helper works correctly
"""

import unittest
import numpy as np

from models.evaluation.effective_information import (
    compute_effective_information,
    constant_trajectory_floor,
    corrected_effective_information,
    compare_ei_levels,
    discretize_continuous,
    _build_tpm,
)


class TestEffectiveInformation(unittest.TestCase):

    def test_deterministic_tpm_maximal_ei(self):
        """A fully deterministic system should have EI = log2(num_states)."""
        # Trajectory: 0 -> 1 -> 2 -> 3 -> 0 -> 1 -> 2 -> 3 -> ...
        # Each state always transitions to the same next state.
        cycle = list(range(4)) * 50  # 200 steps cycling deterministically
        trajectories = [np.array(cycle)]

        ei = compute_effective_information(trajectories, num_states=4)

        # With Laplace smoothing the EI won't be exactly log2(4)=2.0,
        # but should be close to the maximum
        self.assertGreater(ei, 1.5, "Deterministic TPM should have high EI")

    def test_uniform_tpm_low_ei(self):
        """A system with uniform random transitions should have EI near 0."""
        rng = np.random.RandomState(42)
        # Generate random transitions (each state equally likely to go anywhere)
        trajectories = [rng.randint(0, 4, size=1000)]

        ei = compute_effective_information(trajectories, num_states=4)

        # Random transitions = high noise = low EI
        self.assertLess(ei, 0.5, "Random TPM should have low EI")

    def test_single_state_returns_zero(self):
        """A single-state system has no information to integrate."""
        ei = compute_effective_information([np.array([0, 0, 0, 0])], num_states=1)
        self.assertEqual(ei, 0.0)

    def test_moderate_structure(self):
        """A system with partial structure should have intermediate EI."""
        # State 0 always goes to 1, state 1 always goes to 0,
        # but states 2 and 3 are random
        rng = np.random.RandomState(42)
        traj = []
        state = 0
        for _ in range(500):
            traj.append(state)
            if state == 0:
                state = 1
            elif state == 1:
                state = 0
            else:
                state = rng.randint(0, 4)
        trajectories = [np.array(traj)]

        ei = compute_effective_information(trajectories, num_states=4)

        # Should be between 0 and 2.0 (log2(4))
        self.assertGreater(ei, 0.3)
        self.assertLess(ei, 2.0)

    def test_compare_ei_levels_valid_output(self):
        """compare_ei_levels should return all expected fields."""
        rng = np.random.RandomState(42)

        # Gates: random transitions (low EI)
        gate_trajs = [rng.randint(0, 4, size=200)]

        # Workspace: deterministic transitions (high EI)
        cycle = list(range(4)) * 50
        ws_trajs = [np.array(cycle)]

        result = compare_ei_levels(
            gate_trajectories=gate_trajs,
            gate_num_states=4,
            workspace_trajectories=ws_trajs,
            workspace_num_states=4,
        )

        self.assertIn("ei_gates", result)
        self.assertIn("ei_workspace", result)
        self.assertIn("ratio", result)
        self.assertIn("emergent", result)

        # Workspace is deterministic, gates are random -> should be emergent
        self.assertTrue(result["emergent"])
        self.assertGreater(result["ratio"], 1.0)

    def test_compare_no_emergence(self):
        """When gates are more structured, emergence should be False."""
        rng = np.random.RandomState(42)

        # Gates: deterministic
        gate_trajs = [np.array(list(range(4)) * 50)]

        # Workspace: random
        ws_trajs = [rng.randint(0, 4, size=200)]

        result = compare_ei_levels(
            gate_trajectories=gate_trajs,
            gate_num_states=4,
            workspace_trajectories=ws_trajs,
            workspace_num_states=4,
        )

        self.assertFalse(result["emergent"])
        self.assertLess(result["ratio"], 1.0)


class TestFloorCorrection(unittest.TestCase):
    """The constant-trajectory Laplace floor and its subtraction.

    The 2026-07 signature assessment showed the raw EI of a FROZEN trajectory is a
    positive constant depending only on (num_states, window length), bit-identical
    across five runs (0.031178 for 243 states at window 10000), and that the ~12x
    'emergence ratio' was the ratio of two such floors. These tests pin that value,
    verify the analytic floor matches the computed EI exactly, and verify the
    corrected EI kills the artifact without killing real transition structure.
    """

    def test_analytic_floor_matches_computed_ei(self):
        """The closed-form floor must equal compute_effective_information on a
        constant trajectory, for a range of state counts and window lengths."""
        for num_states in (2, 8, 243):
            for length in (10, 200, 10000):
                simulated = compute_effective_information(
                    [np.zeros(length, dtype=int)], num_states)
                analytic = constant_trajectory_floor(num_states, length)
                self.assertAlmostEqual(analytic, simulated, places=12,
                                       msg=f"N={num_states} L={length}")

    def test_floor_reproduces_the_observed_frozen_values(self):
        """Pin the exact frozen values from the 2026-07 assessment: 243 states and
        8 states at window 10000 (the values seen in every run's episodes.csv)."""
        self.assertAlmostEqual(constant_trajectory_floor(243, 10000),
                               0.031178, places=6)
        self.assertAlmostEqual(constant_trajectory_floor(8, 10000),
                               0.373712, places=6)

    def test_constant_trajectory_corrected_to_zero(self):
        """A frozen trajectory must report corrected EI exactly 0."""
        for num_states in (8, 243):
            corrected = corrected_effective_information(
                [np.zeros(10000, dtype=int)], num_states)
            self.assertEqual(corrected, 0.0)

    def test_transitioning_trajectory_stays_positive(self):
        """Real deterministic transition structure must survive the correction."""
        cycle = np.array(list(range(4)) * 50)
        raw = compute_effective_information([cycle], 4)
        corrected = corrected_effective_information([cycle], 4)
        self.assertGreater(corrected, 0.5)
        self.assertLess(corrected, raw)  # correction only subtracts

    def test_corrected_never_exceeds_raw(self):
        rng = np.random.RandomState(0)
        traj = [rng.randint(0, 8, size=500)]
        raw = compute_effective_information(traj, 8)
        corrected = corrected_effective_information(traj, 8)
        self.assertLessEqual(corrected, raw)
        self.assertGreaterEqual(corrected, 0.0)

    def test_frozen_both_levels_not_emergent_after_correction(self):
        """The core artifact: two frozen levels gave raw ratio ~12x (floor ratio).
        After correction both are 0, the ratio is 0, and emergent_corr is False."""
        frozen_micro = [np.zeros(10000, dtype=int)]
        frozen_macro = [np.zeros(10000, dtype=int)]
        result = compare_ei_levels(frozen_micro, 243, frozen_macro, 8)
        # raw reproduces the artifact
        self.assertAlmostEqual(result["ei_gates"], 0.031178, places=6)
        self.assertAlmostEqual(result["ei_workspace"], 0.373712, places=6)
        self.assertGreater(result["ratio"], 10.0)
        self.assertTrue(result["emergent"])  # the documented bias
        # corrected kills it
        self.assertEqual(result["ei_gates_corr"], 0.0)
        self.assertEqual(result["ei_workspace_corr"], 0.0)
        self.assertEqual(result["ratio_corr"], 0.0)
        self.assertFalse(result["emergent_corr"])

    def test_real_macro_structure_still_emergent_after_correction(self):
        """Correction must not erase genuine emergence: random micro, deterministic
        macro stays emergent on the corrected reading."""
        rng = np.random.RandomState(42)
        gate_trajs = [rng.randint(0, 4, size=200)]
        ws_trajs = [np.array(list(range(4)) * 50)]
        result = compare_ei_levels(gate_trajs, 4, ws_trajs, 4)
        self.assertTrue(result["emergent_corr"])
        self.assertGreater(result["ratio_corr"], 1.0)

    def test_short_trajectory_floor_is_zero(self):
        self.assertEqual(constant_trajectory_floor(8, 1), 0.0)
        self.assertEqual(constant_trajectory_floor(1, 100), 0.0)


class TestDiscretize(unittest.TestCase):

    def test_basic_binning(self):
        """Values at 0 and 1 should map to first and last bins."""
        vals = np.array([0.0, 0.5, 1.0])
        bins = discretize_continuous(vals, num_bins=8)

        self.assertEqual(bins[0], 0)
        self.assertEqual(bins[-1], 7)

    def test_out_of_range_clamped(self):
        """Values outside [0,1] should be clamped."""
        vals = np.array([-0.5, 1.5, 2.0])
        bins = discretize_continuous(vals, num_bins=4)

        self.assertEqual(bins[0], 0)
        self.assertEqual(bins[1], 3)
        self.assertEqual(bins[2], 3)

    def test_output_shape_preserved(self):
        """Output should have the same shape as input."""
        vals = np.random.rand(100)
        bins = discretize_continuous(vals, num_bins=16)
        self.assertEqual(bins.shape, vals.shape)


class TestBuildTPM(unittest.TestCase):

    def test_tpm_rows_sum_to_one(self):
        """Each row of the TPM should sum to 1."""
        traj = [np.array([0, 1, 2, 3, 0, 1, 2, 3])]
        tpm = _build_tpm(traj, num_states=4)

        for i in range(4):
            self.assertAlmostEqual(tpm[i].sum(), 1.0, places=10)

    def test_tpm_shape(self):
        """TPM should be [num_states x num_states]."""
        traj = [np.array([0, 1, 0, 1])]
        tpm = _build_tpm(traj, num_states=3)
        self.assertEqual(tpm.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
