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
                pred = m.predict(m.encode(prev))
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
                pred = m.predict(m.encode(prev))
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


if __name__ == "__main__":
    unittest.main()
