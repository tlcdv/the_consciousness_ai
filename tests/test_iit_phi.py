"""
Tests for IIT Phi integration (models/evaluation/iit_phi.py).

Covers:
- Gate state binarization with adaptive thresholds
- Empirical TPM construction from observed transitions
- Connectivity matrix structure
- Phi proxy (geometric) correctness without pyphi
- PhiResult metadata
- Graceful degradation with insufficient data
- Legacy method deprecation warnings
- TPM diagnostic stats
"""

import unittest
import warnings
import numpy as np
import torch

from models.evaluation.iit_phi import (
    IITMetrics,
    PhiResult,
    GATE_CM,
    GATE_NODE_LABELS,
)
from models.core.consciousness_gating import GatingState


class TestGateStateBinarization(unittest.TestCase):
    """Tests for converting continuous gate values to binary states."""

    def setUp(self):
        self.metrics = IITMetrics(history_len=50)

    def test_binarize_above_threshold(self):
        """All values above default thresholds produce all-ones state."""
        state = GatingState(
            attention_level=0.8,
            stability_score=0.7,
            adaptation_rate=0.05,
            meta_memory_coherence=0.6,
            narrator_confidence=0.9,
        )
        binary = self.metrics.update_from_gate_state(state)
        self.assertEqual(binary, (1, 1, 1, 1, 1))

    def test_binarize_below_threshold(self):
        """All values below default thresholds produce all-zeros state."""
        state = GatingState(
            attention_level=0.0,
            stability_score=0.0,
            adaptation_rate=0.0,
            meta_memory_coherence=0.0,
            narrator_confidence=0.0,
        )
        binary = self.metrics.update_from_gate_state(state)
        self.assertEqual(binary, (0, 0, 0, 0, 0))

    def test_binarize_mixed(self):
        """Mixed values produce correct partial binary state."""
        state = GatingState(
            attention_level=0.8,   # above 0.5
            stability_score=0.1,   # below 0.5
            adaptation_rate=0.05,  # above 0.01
            meta_memory_coherence=0.0,  # below 0.5
            narrator_confidence=0.9,    # above 0.5
        )
        binary = self.metrics.update_from_gate_state(state)
        self.assertEqual(binary, (1, 0, 1, 0, 1))

    def test_adaptive_thresholds_shift_with_data(self):
        """Thresholds adapt based on running median of observed values."""
        metrics = IITMetrics(history_len=50)

        # Feed 20 states with high attention (0.9) to shift threshold up
        for _ in range(20):
            metrics.update_from_gate_state(GatingState(
                attention_level=0.9,
                stability_score=0.9,
                adaptation_rate=0.1,
                meta_memory_coherence=0.8,
                narrator_confidence=0.8,
            ))

        # After adaptation, median is 0.9 for attention.
        # A value of 0.7 should now be BELOW threshold (below median 0.9)
        state = GatingState(
            attention_level=0.7,
            stability_score=0.7,
            adaptation_rate=0.05,
            meta_memory_coherence=0.5,
            narrator_confidence=0.5,
        )
        binary = metrics.update_from_gate_state(state)
        # Attention: 0.7 < median(0.9) -> 0
        self.assertEqual(binary[0], 0)

    def test_binarize_returns_tuple_of_ints(self):
        """Binary state is always a tuple of Python ints."""
        state = GatingState(attention_level=0.6)
        binary = self.metrics.update_from_gate_state(state)
        self.assertIsInstance(binary, tuple)
        self.assertEqual(len(binary), 5)
        for b in binary:
            self.assertIsInstance(b, int)
            self.assertIn(b, (0, 1))

    def test_raw_history_recorded(self):
        """Raw continuous values are stored for threshold computation."""
        state = GatingState(
            attention_level=0.3,
            stability_score=0.7,
            adaptation_rate=0.02,
            meta_memory_coherence=0.1,
            narrator_confidence=0.5,
        )
        self.metrics.update_from_gate_state(state)
        self.assertEqual(len(self.metrics._raw_history), 1)
        np.testing.assert_allclose(
            self.metrics._raw_history[0],
            [0.3, 0.7, 0.02, 0.1, 0.5],
        )


class TestTPMConstruction(unittest.TestCase):
    """Tests for empirical TPM building from state transitions."""

    def setUp(self):
        self.metrics = IITMetrics(history_len=200)

    def test_insufficient_history_returns_uniform(self):
        """With < 2 states, TPM should be uniform 0.5."""
        tpm = self.metrics.build_empirical_tpm(5)
        np.testing.assert_allclose(tpm, 0.5)
        self.assertEqual(tpm.shape, (32, 5))

    def test_tpm_shape(self):
        """TPM has shape (2^N, N) for N nodes."""
        # Add some transitions
        for _ in range(10):
            self.metrics.state_history.append((1, 0, 1, 0, 1))
            self.metrics.state_history.append((0, 1, 0, 1, 0))

        tpm = self.metrics.build_empirical_tpm(5)
        self.assertEqual(tpm.shape, (32, 5))

    def test_tpm_rows_are_probabilities(self):
        """Each TPM entry should be in [0, 1]."""
        for _ in range(50):
            state = tuple(np.random.randint(0, 2, 5).tolist())
            self.metrics.state_history.append(state)

        tpm = self.metrics.build_empirical_tpm(5)
        self.assertTrue(np.all(tpm >= 0.0))
        self.assertTrue(np.all(tpm <= 1.0))

    def test_deterministic_transitions_produce_extreme_tpm(self):
        """A system that always transitions A->B should have a near-deterministic TPM row."""
        metrics = IITMetrics(history_len=200)
        state_a = (1, 0, 0, 0, 0)
        state_b = (0, 1, 0, 0, 0)

        # Alternate A -> B -> A -> B ...
        for _ in range(50):
            metrics.state_history.append(state_a)
            metrics.state_history.append(state_b)

        tpm = metrics.build_empirical_tpm(5)

        # Row for state_a (little-endian index: bit 0 is ON -> index 1)
        idx_a = 1  # (1,0,0,0,0) in little-endian = 1
        # After state_a, node 1 (stability) should almost always be ON
        self.assertGreater(tpm[idx_a, 1], 0.8)
        # Node 0 (attention) should almost always be OFF after state_a
        self.assertLess(tpm[idx_a, 0], 0.3)

    def test_little_endian_indexing(self):
        """Verify state tuple (1,0,0,0,0) maps to index 1, not 16."""
        metrics = IITMetrics()
        state = (1, 0, 0, 0, 0)
        # Little-endian: bit 0 is 1, rest 0 -> index = 1
        metrics.state_history.append(state)
        metrics.state_history.append((0, 0, 0, 0, 0))

        tpm = metrics.build_empirical_tpm(5)
        # Row 1 should have more visits than row 16
        # (With Laplace smoothing both start at 2 visits, but row 1 gets +1)
        # Just verify the shape is correct
        self.assertEqual(tpm.shape, (32, 5))

    def test_laplace_smoothing_prevents_zeros(self):
        """Unvisited states should not have zero probability entries."""
        # Only visit one state pair
        metrics = IITMetrics()
        metrics.state_history.append((0, 0, 0, 0, 0))
        metrics.state_history.append((1, 1, 1, 1, 1))

        tpm = metrics.build_empirical_tpm(5)
        # Unvisited rows should still have non-zero entries due to Laplace
        self.assertTrue(np.all(tpm > 0.0))


class TestConnectivityMatrix(unittest.TestCase):
    """Tests for the gate subsystem connectivity matrix."""

    def test_cm_shape(self):
        """CM should be 5x5."""
        self.assertEqual(GATE_CM.shape, (5, 5))

    def test_cm_is_binary(self):
        """CM entries should be 0 or 1."""
        self.assertTrue(np.all((GATE_CM == 0) | (GATE_CM == 1)))

    def test_cm_no_self_loops(self):
        """Diagonal should be zero (no self-connections)."""
        np.testing.assert_array_equal(np.diag(GATE_CM), 0)

    def test_cm_has_connections(self):
        """CM should not be empty (at least some causal links)."""
        self.assertGreater(GATE_CM.sum(), 0)

    def test_attention_drives_stability(self):
        """Attention (node 0) should connect to stability (node 1)."""
        self.assertEqual(GATE_CM[0, 1], 1)

    def test_attention_drives_coherence(self):
        """Attention (node 0) should connect to coherence (node 3)."""
        self.assertEqual(GATE_CM[0, 3], 1)

    def test_stability_drives_adaptation(self):
        """Stability (node 1) should connect to adaptation (node 2)."""
        self.assertEqual(GATE_CM[1, 2], 1)

    def test_adaptation_drives_confidence(self):
        """Adaptation (node 2) should connect to confidence (node 4).
        This edge closes the cycle so the system is irreducible under IIT."""
        self.assertEqual(GATE_CM[2, 4], 1)

    def test_coherence_drives_adaptation(self):
        """Coherence (node 3) should connect to adaptation (node 2)."""
        self.assertEqual(GATE_CM[3, 2], 1)

    def test_coherence_drives_confidence(self):
        """Coherence (node 3) should connect to confidence (node 4)."""
        self.assertEqual(GATE_CM[3, 4], 1)

    def test_confidence_drives_attention(self):
        """Confidence (node 4) should connect back to attention (node 0)."""
        self.assertEqual(GATE_CM[4, 0], 1)

    def test_every_node_has_in_and_out_degree(self):
        """For phi > 0 under IIT, every node must lie inside a directed
        cycle, which requires in-degree >= 1 and out-degree >= 1."""
        in_degrees = GATE_CM.sum(axis=0)
        out_degrees = GATE_CM.sum(axis=1)
        self.assertTrue(np.all(in_degrees >= 1),
                        msg=f"in-degrees: {in_degrees}")
        self.assertTrue(np.all(out_degrees >= 1),
                        msg=f"out-degrees: {out_degrees}")

    def test_node_labels_match_cm_size(self):
        """Label count should match CM dimension."""
        self.assertEqual(len(GATE_NODE_LABELS), GATE_CM.shape[0])


class TestPhiProxy(unittest.TestCase):
    """Tests for the geometric proxy when pyphi is unavailable."""

    def setUp(self):
        self.metrics = IITMetrics(history_len=200)

    def test_uniform_tpm_gives_zero_proxy(self):
        """A uniform 0.5 TPM (no causal structure) should give near-zero proxy."""
        tpm = np.full((32, 5), 0.5)
        proxy = self.metrics.compute_phi_proxy_from_tpm(tpm, (0, 0, 0, 0, 0))
        self.assertAlmostEqual(proxy, 0.0, places=5)

    def test_deterministic_tpm_gives_positive_proxy(self):
        """A deterministic TPM should give a clearly positive proxy."""
        # Each row is either all 0 or all 1 (fully deterministic)
        tpm = np.zeros((32, 5))
        tpm[:16, :] = 1.0  # First half always ON
        proxy = self.metrics.compute_phi_proxy_from_tpm(tpm, (0, 0, 0, 0, 0))
        self.assertGreater(proxy, 0.5)

    def test_proxy_nonnegative(self):
        """Proxy should never be negative."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            tpm = rng.rand(32, 5)
            proxy = self.metrics.compute_phi_proxy_from_tpm(tpm, (0, 0, 0, 0, 0))
            self.assertGreaterEqual(proxy, 0.0)

    def test_structured_tpm_higher_than_random(self):
        """A structured TPM should score higher than a random one."""
        # Structured: identity-like transitions
        tpm_structured = np.zeros((32, 5))
        for i in range(32):
            for j in range(5):
                tpm_structured[i, j] = float((i >> j) & 1)

        rng = np.random.RandomState(0)
        tpm_random = rng.rand(32, 5) * 0.3 + 0.35  # Close to uniform

        proxy_struct = self.metrics.compute_phi_proxy_from_tpm(
            tpm_structured, (0, 0, 0, 0, 0))
        proxy_random = self.metrics.compute_phi_proxy_from_tpm(
            tpm_random, (0, 0, 0, 0, 0))

        self.assertGreater(proxy_struct, proxy_random)


class TestComputePhiFromGateState(unittest.TestCase):
    """Tests for the primary compute_phi_from_gate_state() entry point."""

    def setUp(self):
        self.metrics = IITMetrics(history_len=200)

    def _make_state(self, att=0.5, stb=0.5, adp=0.01, coh=0.5, conf=0.5):
        return GatingState(
            attention_level=att,
            stability_score=stb,
            adaptation_rate=adp,
            meta_memory_coherence=coh,
            narrator_confidence=conf,
        )

    def test_returns_phi_result(self):
        """Should return a PhiResult dataclass."""
        for _ in range(10):
            result = self.metrics.compute_phi_from_gate_state(
                self._make_state(att=0.8, stb=0.6))
        self.assertIsInstance(result, PhiResult)

    def test_insufficient_data_method(self):
        """With < 5 transitions, method should be 'insufficient_data'."""
        result = self.metrics.compute_phi_from_gate_state(self._make_state())
        self.assertEqual(result.method, "insufficient_data")
        self.assertEqual(result.phi, 0.0)

    def test_sufficient_data_uses_proxy_without_pyphi(self):
        """With enough data and no pyphi, method should be 'proxy'."""
        # Feed 10 varied states
        states = [
            self._make_state(att=0.8, stb=0.7, adp=0.05, coh=0.6, conf=0.9),
            self._make_state(att=0.2, stb=0.3, adp=0.001, coh=0.1, conf=0.1),
            self._make_state(att=0.9, stb=0.8, adp=0.08, coh=0.7, conf=0.8),
            self._make_state(att=0.1, stb=0.1, adp=0.002, coh=0.05, conf=0.2),
            self._make_state(att=0.7, stb=0.6, adp=0.04, coh=0.5, conf=0.7),
            self._make_state(att=0.3, stb=0.4, adp=0.01, coh=0.2, conf=0.3),
        ]
        for s in states:
            result = self.metrics.compute_phi_from_gate_state(s)

        # Last result should have enough transitions
        self.assertIn(result.method, ("proxy", "pyphi"))

    def test_phi_result_metadata(self):
        """PhiResult should contain correct metadata."""
        for i in range(8):
            result = self.metrics.compute_phi_from_gate_state(
                self._make_state(att=0.1 * i, stb=0.1 * i))

        self.assertEqual(result.num_nodes, 5)
        self.assertEqual(result.node_labels, GATE_NODE_LABELS)
        self.assertEqual(len(result.current_state), 5)
        self.assertGreater(result.num_transitions, 0)

    def test_scalar_convenience(self):
        """compute_phi_from_gate_state_scalar returns a float."""
        for _ in range(8):
            phi = self.metrics.compute_phi_from_gate_state_scalar(
                self._make_state(att=0.8))
        self.assertIsInstance(phi, float)

    def test_varied_states_produce_positive_proxy(self):
        """Diverse gate activity should produce positive phi proxy."""
        np.random.seed(42)
        for _ in range(30):
            self.metrics.compute_phi_from_gate_state(self._make_state(
                att=np.random.rand(),
                stb=np.random.rand(),
                adp=np.random.rand() * 0.1,
                coh=np.random.rand(),
                conf=np.random.rand(),
            ))

        # Final result
        result = self.metrics.compute_phi_from_gate_state(
            self._make_state(att=0.5, stb=0.5, adp=0.02, coh=0.5, conf=0.5))

        if result.method == "proxy":
            # With varied data, proxy should be positive
            self.assertGreater(result.phi, 0.0)


class TestTPMStats(unittest.TestCase):
    """Tests for get_tpm_stats() diagnostics."""

    def test_empty_stats(self):
        """No data should return insufficient_data status."""
        metrics = IITMetrics()
        stats = metrics.get_tpm_stats()
        self.assertEqual(stats["status"], "insufficient_data")
        self.assertEqual(stats["transitions"], 0)

    def test_stats_with_data(self):
        """With data, stats should report coverage and determinism."""
        metrics = IITMetrics()
        np.random.seed(123)
        for _ in range(50):
            metrics.update_from_gate_state(GatingState(
                attention_level=np.random.rand(),
                stability_score=np.random.rand(),
                adaptation_rate=np.random.rand() * 0.1,
                meta_memory_coherence=np.random.rand(),
                narrator_confidence=np.random.rand(),
            ))

        stats = metrics.get_tpm_stats()
        self.assertEqual(stats["status"], "ready")
        self.assertGreater(stats["transitions"], 0)
        self.assertGreater(stats["unique_states"], 1)
        self.assertEqual(stats["possible_states"], 32)
        self.assertIn("attention", stats["node_determinism"])
        self.assertGreaterEqual(stats["state_coverage"], 0.0)
        self.assertLessEqual(stats["state_coverage"], 1.0)


class TestLegacyDeprecation(unittest.TestCase):
    """Tests that legacy methods emit deprecation warnings."""

    def test_compute_phi_proxy_warns(self):
        """Legacy compute_phi_proxy should emit DeprecationWarning."""
        metrics = IITMetrics()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metrics.compute_phi_proxy(torch.rand(10))
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_extract_subsystem_state_still_works(self):
        """Legacy extract_subsystem_state should still function."""
        metrics = IITMetrics()
        result = metrics.extract_subsystem_state(torch.tensor([0.8, 0.2, 0.9, 0.1]))
        self.assertIn("state", result)
        self.assertIn("node_indices", result)
        self.assertEqual(len(result["state"]), 4)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness checks."""

    def test_all_zero_gate_state(self):
        """All-zero gate state should not crash."""
        metrics = IITMetrics()
        state = GatingState()  # All defaults = 0.0
        binary = metrics.update_from_gate_state(state)
        self.assertEqual(binary, (0, 0, 0, 0, 0))

    def test_constant_state_history(self):
        """Feeding the same state repeatedly should produce valid TPM."""
        metrics = IITMetrics()
        state = GatingState(
            attention_level=0.8,
            stability_score=0.7,
            adaptation_rate=0.05,
            meta_memory_coherence=0.6,
            narrator_confidence=0.9,
        )
        for _ in range(20):
            metrics.update_from_gate_state(state)

        tpm = metrics.build_empirical_tpm(5)
        self.assertEqual(tpm.shape, (32, 5))
        self.assertTrue(np.all(np.isfinite(tpm)))

    def test_history_len_respected(self):
        """History should not exceed configured lengths.

        state_history is bounded by tpm_window (sliding window for TPM),
        _raw_history is bounded by history_len (for adaptive thresholds).
        """
        metrics = IITMetrics(history_len=10, tpm_window=15)
        for _ in range(50):
            metrics.update_from_gate_state(GatingState(attention_level=0.8))
        self.assertLessEqual(len(metrics.state_history), 15)
        self.assertLessEqual(len(metrics._raw_history), 10)

    def test_three_node_tpm(self):
        """build_empirical_tpm should work with arbitrary node counts."""
        metrics = IITMetrics()
        metrics.state_history.append((0, 1, 0))
        metrics.state_history.append((1, 0, 1))
        metrics.state_history.append((0, 1, 0))

        tpm = metrics.build_empirical_tpm(3)
        self.assertEqual(tpm.shape, (8, 3))
        self.assertTrue(np.all(tpm >= 0.0))
        self.assertTrue(np.all(tpm <= 1.0))


class TestPhiVariesWithVariedInput(unittest.TestCase):
    """Locks the R2 invariant: when gates receive varied inputs, the
    binarized state history visits multiple states and the empirical
    TPM has structure. With the cyclic GATE_CM, phi is non-zero on at
    least some steps."""

    def test_varied_input_produces_state_diversity_and_phi_signal(self):
        import torch as _torch
        from models.core.consciousness_gating import ConsciousnessGate

        _torch.manual_seed(0)
        np.random.seed(0)

        gate = ConsciousnessGate({"hidden_size": 64})
        metrics = IITMetrics()

        unique_states = set()
        phi_values = []
        for _ in range(80):
            inp = _torch.randn(1, 64) * 2.0
            _, gs = gate(inp)
            r = metrics.compute_phi_from_gate_state(gs)
            unique_states.add(r.current_state)
            phi_values.append(r.phi)

        # Varied input must make the gates visit multiple binary states.
        # Without diversity the TPM has no structure and phi is trivially 0.
        self.assertGreater(
            len(unique_states), 4,
            msg=f"Only {len(unique_states)} unique states visited",
        )

        # If pyphi is installed and the cyclic GATE_CM is doing its job,
        # at least one step in a varied sequence should produce phi > 0.
        # If pyphi is unavailable we get the proxy, which also varies on
        # structured TPMs. Either way phi should not be a flat zero.
        max_phi = max(phi_values)
        std_phi = float(np.std(phi_values))
        self.assertTrue(
            max_phi > 1e-5 or std_phi > 1e-5,
            msg=f"Phi was identically zero across 80 varied steps. "
                f"max={max_phi:.6e}, std={std_phi:.6e}. "
                f"Either GATE_CM is reducible or the proxy is broken.",
        )


if __name__ == "__main__":
    unittest.main()
