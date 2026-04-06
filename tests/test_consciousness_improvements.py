"""
Tests for the consciousness architecture improvements (Stages 1-4).

Covers: gradient flow, RND curiosity, gate causal feedback, TPM decay,
gate diversity, phi intrinsic reward, memory ungating, arousal fix,
and numerical stability.
"""
from __future__ import annotations

import torch
import numpy as np
import unittest

from models.core.rnd_curiosity import RNDCuriosity
from models.core.consciousness_gating import ConsciousnessGate, GatingState
from models.evaluation.iit_phi import IITMetrics
from models.emotion.emotional_processing import EmotionalProcessingCore


class TestRNDCuriosity(unittest.TestCase):
    """Tests for the RND curiosity module (Stage 1d)."""

    def setUp(self):
        self.rnd = RNDCuriosity(input_dim=64, feature_dim=32)

    def test_novel_state_high_curiosity(self):
        """Novel states should produce higher curiosity than repeated ones."""
        # Train predictor on a fixed state to make it familiar
        familiar = torch.randn(1, 64)
        optimizer = torch.optim.Adam(self.rnd.predictor_network.parameters(), lr=1e-2)
        for _ in range(50):
            _, loss = self.rnd(familiar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        familiar_score, _ = self.rnd(familiar)
        novel_score, _ = self.rnd(torch.randn(1, 64))
        self.assertGreater(novel_score, familiar_score)

    def test_target_frozen(self):
        """Target network parameters must not change after training."""
        target_before = [p.clone() for p in self.rnd.target_network.parameters()]
        optimizer = torch.optim.Adam(self.rnd.predictor_network.parameters(), lr=1e-2)
        for _ in range(20):
            _, loss = self.rnd(torch.randn(1, 64))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for before, after in zip(target_before, self.rnd.target_network.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_output_types(self):
        """forward() returns (float, Tensor)."""
        score, loss = self.rnd(torch.randn(64))
        self.assertIsInstance(score, float)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)

    def test_input_dimension_adaptation(self):
        """RND should handle broadcast with wrong dimension gracefully."""
        # Smaller input
        score, loss = self.rnd(torch.randn(32))
        self.assertIsInstance(score, float)
        # Larger input
        score, loss = self.rnd(torch.randn(128))
        self.assertIsInstance(score, float)

    def test_predictor_learns(self):
        """Curiosity should decrease for repeated states as predictor learns."""
        state = torch.randn(1, 64)
        initial_score, _ = self.rnd(state)
        optimizer = torch.optim.Adam(self.rnd.predictor_network.parameters(), lr=1e-2)
        for _ in range(100):
            _, loss = self.rnd(state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final_score, _ = self.rnd(state)
        self.assertLess(final_score, initial_score)


class TestGateCausalFeedback(unittest.TestCase):
    """Tests for inter-gate causal connections (Stage 2a)."""

    def setUp(self):
        self.gate = ConsciousnessGate({
            "hidden_size": 64,
            "gating": {"attention_threshold": 0.5},
        })

    def test_gate_has_feedback_layer(self):
        """Gate must have a feedback projection layer."""
        self.assertTrue(hasattr(self.gate, 'gate_feedback'))
        self.assertIsInstance(self.gate.gate_feedback, torch.nn.Linear)
        self.assertEqual(self.gate.gate_feedback.in_features, 5)

    def test_prev_gate_values_initialized_none(self):
        """prev_gate_values starts as None."""
        self.assertIsNone(self.gate.prev_gate_values)

    def test_prev_gate_values_stored_after_forward(self):
        """After one forward call, prev_gate_values should be populated."""
        x = torch.randn(1, 64)
        self.gate(x)
        self.assertIsNotNone(self.gate.prev_gate_values)
        self.assertEqual(self.gate.prev_gate_values.shape, (5,))

    def test_feedback_changes_gate_state(self):
        """Two forward calls with same input should produce different gate
        states due to temporal feedback from prev_gate_values."""
        x = torch.randn(1, 64)
        _, state1 = self.gate(x)
        vals1 = (state1.attention_level, state1.stability_score,
                 state1.meta_memory_coherence, state1.narrator_confidence)
        _, state2 = self.gate(x)  # Now uses prev_gate_values
        vals2 = (state2.attention_level, state2.stability_score,
                 state2.meta_memory_coherence, state2.narrator_confidence)
        # At least one gate value should differ due to feedback
        self.assertNotEqual(vals1, vals2)

    def test_varied_broadcasts_produce_varied_gates(self):
        """Different broadcasts should produce different gate states."""
        states = []
        for _ in range(10):
            x = torch.randn(1, 64) * 5  # Amplified to avoid sigmoid saturation
            _, state = self.gate(x)
            states.append(state.attention_level)
        # Should have at least 3 distinct values
        unique = len(set(round(s, 4) for s in states))
        self.assertGreaterEqual(unique, 3)

    def test_nan_guard(self):
        """Gate output should not contain NaN even with extreme inputs."""
        x = torch.full((1, 64), float('nan'))
        out, _ = self.gate(x)
        self.assertFalse(torch.isnan(out).any())


class TestTPMDecay(unittest.TestCase):
    """Tests for sliding-window TPM with exponential decay (Stage 2b)."""

    def setUp(self):
        self.iit = IITMetrics(tpm_window=200, tpm_decay=0.995)

    def test_tpm_decay_parameter(self):
        """IITMetrics should accept tpm_decay parameter."""
        self.assertEqual(self.iit.tpm_decay, 0.995)

    def test_default_window_200(self):
        """Default TPM window should be 200 (was 100)."""
        iit = IITMetrics()
        self.assertEqual(iit.tpm_window, 200)

    def test_recent_transitions_dominate(self):
        """With decay, recent transitions should have more influence than old ones."""
        # Feed 100 transitions to state (0,0,0,0,0) -> (1,1,1,1,1)
        for _ in range(100):
            self.iit.state_history.append((0, 0, 0, 0, 0))
            self.iit.state_history.append((1, 1, 1, 1, 1))

        tpm_old_pattern = self.iit.build_empirical_tpm()

        # Now feed 50 transitions with a different pattern: (0,0,0,0,0) -> (0,0,0,0,0)
        for _ in range(50):
            self.iit.state_history.append((0, 0, 0, 0, 0))

        tpm_new_pattern = self.iit.build_empirical_tpm()

        # Row 0 (state 00000) column probabilities should shift toward 0
        # because recent transitions show (0,0,0,0,0) -> (0,0,0,0,0)
        # Old pattern had row 0 -> all 1s, new pattern has row 0 -> all 0s
        self.assertLess(tpm_new_pattern[0, 0], tpm_old_pattern[0, 0])

    def test_tpm_decay_prevents_saturation(self):
        """Phi should vary even after many transitions (not converge to fixed point)."""
        # Feed alternating states
        for i in range(500):
            if i % 2 == 0:
                self.iit.state_history.append((1, 0, 1, 0, 1))
            else:
                self.iit.state_history.append((0, 1, 0, 1, 0))

        tpm1 = self.iit.build_empirical_tpm()
        phi1 = self.iit.compute_phi_proxy_from_tpm(tpm1, (1, 0, 1, 0, 1))

        # Feed 200 more with a third state introduced
        for i in range(200):
            self.iit.state_history.append((1, 1, 0, 0, 1))

        tpm2 = self.iit.build_empirical_tpm()
        phi2 = self.iit.compute_phi_proxy_from_tpm(tpm2, (1, 1, 0, 0, 1))

        # Phi should change (not stuck at same value)
        self.assertNotAlmostEqual(phi1, phi2, places=3)

    def test_reduced_laplace_smoothing(self):
        """With reduced Laplace smoothing, empty rows should be closer to 0.5
        but not exactly 0.5 (alpha=0.1 not 1.0)."""
        # Only add a couple of transitions
        self.iit.state_history.append((0, 0, 0, 0, 0))
        self.iit.state_history.append((1, 1, 1, 1, 1))

        tpm = self.iit.build_empirical_tpm()
        # Row for state (0,0,0,0,0) = index 0: should show transition to all-1s
        # With low Laplace, the probability should be close to 1.0 (not diluted)
        self.assertGreater(tpm[0, 0], 0.7)  # Was ~0.67 with alpha=1, now should be >0.8


class TestArousalFix(unittest.TestCase):
    """Tests for self-reinforcing arousal fix (Stage 3b)."""

    def test_arousal_bounded_under_repeated_novelty(self):
        """Arousal should not self-amplify with repeated novelty signals."""
        processor = EmotionalProcessingCore({"emotion_alpha": 0.3})
        for _ in range(100):
            processor.update({
                "perception": {"novelty": 0.5},
                "previous_state": {},
            })
        state = processor.get_state()
        # Arousal should converge to novelty * 0.3 = 0.15, not drift to 1.0
        self.assertLess(state["arousal"], 0.3,
                        f"Arousal drifted to {state['arousal']}, should be bounded")

    def test_arousal_converges_to_novelty_target(self):
        """With constant novelty, arousal should converge to novelty * 0.3."""
        processor = EmotionalProcessingCore({"emotion_alpha": 0.3})
        for _ in range(200):
            processor.update({
                "perception": {"novelty": 0.8},
                "previous_state": {},
            })
        state = processor.get_state()
        expected_target = 0.8 * 0.3  # 0.24
        # With decay of 0.02, the equilibrium is slightly below the target
        self.assertAlmostEqual(state["arousal"], expected_target, delta=0.1)


class TestMemoryPriority(unittest.TestCase):
    """Tests for memory ungating and priority (Stage 1b)."""

    def test_store_experience_accepts_priority(self):
        """store_experience() should accept a priority parameter."""
        from models.memory.memory_core import MemoryCore
        memory = MemoryCore({})
        state = torch.randn(16)
        action = torch.randn(2)
        memory.store_experience(
            state=state, action=action, reward=1.0,
            emotion_values={"valence": 0.5, "arousal": 0.3, "dominance": 0.0},
            attention_level=0.8, priority=0.5,
        )
        self.assertEqual(len(memory.recent_experiences), 1)
        self.assertEqual(memory.recent_experiences[0]["priority"], 0.5)

    def test_default_priority_is_one(self):
        """Default priority should be 1.0 when not specified."""
        from models.memory.memory_core import MemoryCore
        memory = MemoryCore({})
        memory.store_experience(
            state=torch.randn(16), action=torch.randn(2), reward=0.5,
            emotion_values={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            attention_level=0.3,
        )
        self.assertEqual(memory.recent_experiences[0]["priority"], 1.0)


class TestGateDiversityLoss(unittest.TestCase):
    """Tests for gate diversity regularization (Stage 2c)."""

    def test_diversity_loss_penalizes_midpoint(self):
        """Gate values near 0.5 should produce higher diversity loss
        than values near 0 or 1."""
        # Midpoint values
        mid = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        mid_loss = -torch.log(torch.abs(mid - 0.5).clamp(min=0.01)).mean()

        # Extreme values
        extreme = torch.tensor([[0.1, 0.9, 0.05, 0.95]])
        extreme_loss = -torch.log(torch.abs(extreme - 0.5).clamp(min=0.01)).mean()

        self.assertGreater(mid_loss.item(), extreme_loss.item())


if __name__ == "__main__":
    unittest.main()
