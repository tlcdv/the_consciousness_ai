"""
Tests for Phase 5 deliverable 1: the dynamic self-vector loop.

Covers:
  - SelfVectorModule encode/predict shapes and finiteness.
  - SelfRepresentationCore.first_order_features returns the fixed-length vector.
  - VALUE TEST: the SPR-style one-step self-prediction objective learns
    structure beyond persistence on a learnable trajectory (skill > 0). This is
    the deliverable's core claim; a self-model that cannot beat "next == current"
    has learned nothing meta-representational.
  - Integration: the run_episode-style loop exposes a finite self_vector on the
    self-model state and a computable skill.
"""
import math
import unittest

import numpy as np
import torch

from models.self_model.self_representation_core import (
    SelfRepresentationCore,
    SelfVectorModule,
    SELF_VECTOR_FEATURE_DIM,
)
from models.core.consciousness_gating import ConsciousnessGate


class TestSelfVectorModuleShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = SelfVectorModule(self_dim=32)

    def test_encode_shape(self):
        sv = self.m.encode(torch.randn(1, SELF_VECTOR_FEATURE_DIM))
        self.assertEqual(tuple(sv.shape), (1, 32))
        self.assertTrue(torch.isfinite(sv).all())

    def test_predict_shape(self):
        pred = self.m.predict(torch.randn(1, 32))
        self.assertEqual(tuple(pred.shape), (1, SELF_VECTOR_FEATURE_DIM))
        self.assertTrue(torch.isfinite(pred).all())

    def test_predict_next_is_residual(self):
        feats = torch.randn(1, SELF_VECTOR_FEATURE_DIM)
        nxt = self.m.predict_next(feats)
        self.assertEqual(tuple(nxt.shape), (1, SELF_VECTOR_FEATURE_DIM))
        # predict_next must be features + predicted delta (a residual).
        delta = self.m.predict(self.m.encode(feats))
        self.assertTrue(torch.allclose(nxt, feats + delta))


class TestFirstOrderFeatures(unittest.TestCase):
    def setUp(self):
        self.core = SelfRepresentationCore({})

    def test_length_and_finite(self):
        feats = self.core.first_order_features(
            {"valence": 0.2, "arousal": 0.1, "dominance": -0.1},
            (1.5, 0.01, 0.3),
        )
        self.assertEqual(len(feats), SELF_VECTOR_FEATURE_DIM)
        for v in feats:
            self.assertTrue(math.isfinite(v))

    def test_uses_state_defaults_when_emotion_none(self):
        feats = self.core.first_order_features(None, (0.0, 0.0, 0.0))
        self.assertEqual(len(feats), SELF_VECTOR_FEATURE_DIM)

    def test_capability_summary_reflects_model(self):
        self.core.state.capability_model = {"a_valence": 0.4, "b_valence": -0.2}
        feats = self.core.first_order_features({}, (0.0, 0.0, 0.0))
        # index 9 = capability mean, index 10 = capability count norm
        self.assertAlmostEqual(feats[9], 0.1, places=5)
        self.assertAlmostEqual(feats[10], 0.2, places=5)

    def test_performance_features_move_with_reward(self):
        # Phase B: the reward EMAs must make the self-state move on tasks where
        # PAD/interoception are static (index 14 = recent_reward_ema,
        # index 15 = fast-slow trend).
        base = self.core.first_order_features({}, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(base[14], 0.0, places=6)
        for _ in range(10):
            self.core.update_performance(1.0)
        after = self.core.first_order_features({}, (0.0, 0.0, 0.0))
        self.assertGreater(after[14], base[14])
        self.assertGreater(after[14], 0.1)
        self.assertGreater(after[15], 0.0)  # fast EMA leads slow -> positive trend
        self.core.reset_performance()
        cleared = self.core.first_order_features({}, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(cleared[14], 0.0, places=6)


class TestSelfPredictionBeatsPersistence(unittest.TestCase):
    """The deliverable's core claim: the self-model learns predictive structure
    beyond a persistence baseline on a learnable trajectory."""

    def test_skill_positive_on_learnable_dynamics(self):
        torch.manual_seed(0)
        d = SELF_VECTOR_FEATURE_DIM
        m = SelfVectorModule(self_dim=32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)

        # Learnable trajectory: next = 0.7 * tanh(W x) + 0.3 * noise. The
        # deterministic part is a one-step map predictable from the current
        # state; the noise is irreducible. next differs from current, so
        # persistence (predict next == current) is a beatable baseline.
        W = torch.randn(d, d) * 0.5
        x = torch.randn(1, d)
        skills = []
        prev = None
        for _ in range(500):
            if prev is not None:
                pred = m.predict_next(prev)
                loss = torch.nn.functional.mse_loss(pred, x.detach())
                persistence = torch.nn.functional.mse_loss(prev, x).item()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if persistence > 1e-8:
                    skills.append(1.0 - loss.item() / persistence)
            prev = x.detach()
            x = 0.7 * torch.tanh(x @ W.t()) + 0.3 * torch.randn(1, d)

        recent = float(np.mean(skills[-50:]))
        self.assertGreater(
            recent, 0.05,
            f"self-prediction did not beat persistence (skill={recent:.3f})",
        )


class TestSelfVectorLoopIntegration(unittest.TestCase):
    """Mirrors the run_episode loop: self_vector exposed on state, skill computable."""

    def test_loop_exposes_self_vector(self):
        torch.manual_seed(0)
        core = SelfRepresentationCore({})
        m = SelfVectorModule(self_dim=32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        prev = None
        last_skill = None
        for i in range(6):
            core.state.emotional_state = {
                "valence": 0.1 * i, "arousal": 0.05 * i, "dominance": 0.0,
            }
            feats = torch.tensor(
                core.first_order_features(core.state.emotional_state,
                                          (float(i), 0.01 * i, 0.1)),
                dtype=torch.float32,
            ).unsqueeze(0)
            if prev is not None:
                pred = m.predict_next(prev)
                loss = torch.nn.functional.mse_loss(pred, feats.detach())
                persistence = torch.nn.functional.mse_loss(prev, feats).item()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if persistence > 1e-8:
                    last_skill = 1.0 - loss.item() / persistence
            with torch.no_grad():
                core.state.self_vector = m.encode(feats).detach()
            prev = feats.detach()

        self.assertIsNotNone(core.state.self_vector)
        self.assertEqual(tuple(core.state.self_vector.shape), (1, 32))
        self.assertTrue(torch.isfinite(core.state.self_vector).all())
        self.assertIsNotNone(last_skill)
        self.assertTrue(math.isfinite(last_skill))


class TestSelfVectorGating(unittest.TestCase):
    """Phase 5 deliverable 3: gate conditioning on the self_vector.

    Default off must leave the gate path bit-identical (so the WCST ablation is
    clean); enabled, a different self_vector must change the gate outputs.
    """

    def _gate(self, use_self_vector):
        torch.manual_seed(0)
        return ConsciousnessGate({
            "hidden_size": 32,
            "use_self_vector": use_self_vector,
            "self_vector_dim": 8,
            "gating": {"attention_threshold": 0.5, "stability_threshold": 0.6,
                       "base_adaptation_rate": 0.01},
        })

    def test_projection_exists_only_when_enabled(self):
        self.assertIsNone(self._gate(False).self_projection)
        self.assertIsNotNone(self._gate(True).self_projection)

    def test_disabled_ignores_self_vector(self):
        gate = self._gate(False)
        x = torch.randn(1, 32)
        gate.reset_episode()
        gate(x, self_vector=None)
        out1 = gate.last_gate_values_tensor.detach().clone()
        gate.reset_episode()
        gate(x, self_vector=torch.randn(1, 8))
        out2 = gate.last_gate_values_tensor.detach().clone()
        self.assertTrue(torch.allclose(out1, out2),
                        "disabled gate must ignore the self_vector")

    def test_enabled_self_vector_changes_output(self):
        gate = self._gate(True)
        x = torch.randn(1, 32)
        gate.reset_episode()
        gate(x, self_vector=torch.zeros(1, 8))
        out_a = gate.last_gate_values_tensor.detach().clone()
        gate.reset_episode()
        gate(x, self_vector=torch.ones(1, 8) * 3.0)
        out_b = gate.last_gate_values_tensor.detach().clone()
        self.assertFalse(torch.allclose(out_a, out_b),
                         "enabled gate output must depend on the self_vector")


class TestSelfVectorAction(unittest.TestCase):
    """P3: the self-vector concatenated onto the PFC input (causally central).

    Default off keeps PFC input dim = workspace_dim (baseline bit-identical);
    enabled grows the PFC input and routes the self-vector into the policy.
    """

    def _core(self, use_self_vector):
        from models.self_model.action_selection_core import ActionSelectionCore
        from models.emotion.reward_shaping import EmotionalRewardShaper
        from models.memory.memory_core import MemoryCore
        torch.manual_seed(0)
        cfg = {
            "workspace_dim": 32, "context_dim": 16, "action_dim": 4,
            "device": "cpu",
            "use_self_vector": use_self_vector, "self_vector_dim": 8,
        }
        return ActionSelectionCore(cfg, EmotionalRewardShaper({}), MemoryCore({}))

    def test_pfc_input_dim_grows_when_enabled(self):
        self.assertEqual(self._core(False).pfc.working_memory.input_size, 32)
        self.assertEqual(self._core(True).pfc.working_memory.input_size, 40)  # 32 + 8

    def test_augment_noop_when_disabled(self):
        core = self._core(False)
        b = torch.randn(1, 32)
        out = core._augment(b, torch.randn(1, 8))
        self.assertEqual(tuple(out.shape), (1, 32))
        self.assertTrue(torch.allclose(out, b))

    def test_augment_concats_when_enabled(self):
        core = self._core(True)
        b = torch.randn(1, 32)
        sv = torch.randn(1, 8)
        out = core._augment(b, sv)
        self.assertEqual(tuple(out.shape), (1, 40))
        self.assertTrue(torch.allclose(out[:, :32], b))
        self.assertTrue(torch.allclose(out[:, 32:], sv))

    def test_augment_zero_fills_when_enabled_and_no_self_vector(self):
        core = self._core(True)
        out = core._augment(torch.randn(1, 32), None)
        self.assertEqual(tuple(out.shape), (1, 40))
        self.assertTrue(torch.allclose(out[:, 32:], torch.zeros(1, 8)))

    def test_select_action_runs_with_self_vector(self):
        core = self._core(True)
        core.reset_state(1)
        action, value = core.select_action(
            torch.randn(1, 32), self_vector=torch.randn(1, 8))
        self.assertEqual(action.shape[0], 4)
        self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
