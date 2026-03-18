"""Tests for PhenomenologicalMapper (formerly QualiaMapper)."""
from __future__ import annotations

import unittest
import torch
import numpy as np
from models.core.qualia_mapper import (
    PhenomenologicalState,
    PhenomenologicalMapper,
    QualiaState,
    QualiaMapper,
)


class TestPhenomenologicalState(unittest.TestCase):
    def test_to_vector_shape(self):
        state = PhenomenologicalState(intensity=0.5, valence=0.2, complexity=0.8)
        vec = state.to_vector()
        self.assertEqual(vec.shape, (3,))
        self.assertEqual(vec.dtype, np.float32)

    def test_to_vector_values(self):
        state = PhenomenologicalState(intensity=0.3, valence=-0.7, complexity=0.9)
        vec = state.to_vector()
        np.testing.assert_allclose(vec, [0.3, -0.7, 0.9], atol=1e-6)

    def test_backward_compat_alias(self):
        """QualiaState should be an alias for PhenomenologicalState."""
        self.assertIs(QualiaState, PhenomenologicalState)
        state = QualiaState(intensity=1.0, valence=0.0, complexity=0.5)
        self.assertIsInstance(state, PhenomenologicalState)


class TestPhenomenologicalMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = PhenomenologicalMapper()

    def test_backward_compat_alias(self):
        self.assertIs(QualiaMapper, PhenomenologicalMapper)

    def test_map_state_returns_state(self):
        ws = torch.randn(64)
        goal = torch.randn(64)
        result = self.mapper.map_state(ws, goal)
        self.assertIsInstance(result, PhenomenologicalState)

    def test_intensity_bounded(self):
        ws = torch.randn(128) * 10
        goal = torch.randn(128)
        result = self.mapper.map_state(ws, goal)
        self.assertGreaterEqual(result.intensity, 0.0)
        self.assertLessEqual(result.intensity, 1.0)

    def test_valence_bounded(self):
        ws = torch.randn(64)
        goal = torch.randn(64)
        result = self.mapper.map_state(ws, goal)
        self.assertGreaterEqual(result.valence, -1.0)
        self.assertLessEqual(result.valence, 1.0)

    def test_complexity_bounded(self):
        ws = torch.randn(64)
        goal = torch.randn(64)
        result = self.mapper.map_state(ws, goal)
        self.assertGreaterEqual(result.complexity, 0.0)
        self.assertLessEqual(result.complexity, 1.0)

    def test_zero_workspace_low_intensity(self):
        ws = torch.zeros(32)
        goal = torch.randn(32)
        result = self.mapper.map_state(ws, goal)
        self.assertAlmostEqual(result.intensity, 0.0, places=5)

    def test_aligned_vectors_positive_valence(self):
        direction = torch.randn(64)
        ws = direction * 2.0
        goal = direction * 3.0
        result = self.mapper.map_state(ws, goal)
        self.assertGreater(result.valence, 0.8)

    def test_opposed_vectors_negative_valence(self):
        direction = torch.randn(64)
        ws = direction
        goal = -direction
        result = self.mapper.map_state(ws, goal)
        self.assertLess(result.valence, -0.8)

    def test_shape_mismatch_handled(self):
        ws = torch.randn(128)
        goal = torch.randn(64)
        result = self.mapper.map_state(ws, goal)
        self.assertIsInstance(result, PhenomenologicalState)

    def test_numpy_input_accepted(self):
        ws = np.random.randn(32).astype(np.float32)
        goal = np.random.randn(32).astype(np.float32)
        result = self.mapper.map_state(ws, goal)
        self.assertIsInstance(result, PhenomenologicalState)

    def test_uniform_distribution_high_complexity(self):
        """A uniform-ish workspace should have high normalized entropy."""
        ws = torch.ones(100)
        goal = torch.randn(100)
        result = self.mapper.map_state(ws, goal)
        self.assertGreater(result.complexity, 0.9)


if __name__ == "__main__":
    unittest.main()
