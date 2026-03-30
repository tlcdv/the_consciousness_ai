"""Tests for audio affect extraction (acoustic features -> PAD)."""
from __future__ import annotations

import unittest
import torch
import numpy as np

from models.audio.audio_affect_extractor import AudioAffectExtractor, PARALINGUISTIC_CLASSES


class TestAudioAffectExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = AudioAffectExtractor(num_bands=64)

    def _make_inputs(self, B=1, T=100):
        envelope = torch.rand(B, 64, T)
        tfs = torch.randn(B, 64, T) * 0.1
        energy = torch.rand(B, 64, T)
        return envelope, tfs, energy

    def test_pad_output_shape(self):
        env, tfs, energy = self._make_inputs()
        result = self.extractor(env, tfs, energy)
        self.assertEqual(result["pad_delta"].shape, (1, 3))

    def test_acoustic_features_shape(self):
        env, tfs, energy = self._make_inputs()
        result = self.extractor(env, tfs, energy)
        self.assertEqual(result["acoustic_features"].shape, (1, 6))

    def test_pad_in_range(self):
        """PAD deltas should be in [-1, 1] due to tanh."""
        env, tfs, energy = self._make_inputs()
        result = self.extractor(env, tfs, energy)
        pad = result["pad_delta"]
        self.assertTrue(torch.all(pad >= -1.0).item())
        self.assertTrue(torch.all(pad <= 1.0).item())

    def test_paralinguistic_output(self):
        env, tfs, energy = self._make_inputs()
        result = self.extractor(env, tfs, energy)
        self.assertIn(result["paralinguistic"], PARALINGUISTIC_CLASSES)

    def test_paralinguistic_logits_shape(self):
        env, tfs, energy = self._make_inputs()
        result = self.extractor(env, tfs, energy)
        self.assertEqual(result["paralinguistic_logits"].shape, (1, 7))

    def test_silence_near_neutral(self):
        """Silence should produce near-zero features and neutral PAD."""
        env = torch.zeros(1, 64, 100)
        tfs = torch.zeros(1, 64, 100)
        energy = torch.zeros(1, 64, 100)
        result = self.extractor(env, tfs, energy)
        features = result["acoustic_features"]
        self.assertLess(features.abs().mean().item(), 1.0)

    def test_loud_noise_different_from_silence(self):
        """Loud noise should produce different features than silence."""
        env_loud = torch.rand(1, 64, 100) * 5.0
        tfs_loud = torch.randn(1, 64, 100)
        energy_loud = torch.rand(1, 64, 100) * 5.0

        env_quiet = torch.zeros(1, 64, 100)
        tfs_quiet = torch.zeros(1, 64, 100)
        energy_quiet = torch.zeros(1, 64, 100)

        result_loud = self.extractor(env_loud, tfs_loud, energy_loud)
        result_quiet = self.extractor(env_quiet, tfs_quiet, energy_quiet)

        diff = (result_loud["acoustic_features"] - result_quiet["acoustic_features"]).abs().sum().item()
        self.assertGreater(diff, 0.01)

    def test_gradient_flow(self):
        env = torch.rand(1, 64, 100, requires_grad=True)
        tfs = torch.randn(1, 64, 100, requires_grad=True)
        energy = torch.rand(1, 64, 100, requires_grad=True)
        result = self.extractor(env, tfs, energy)
        loss = result["pad_delta"].sum()
        loss.backward()
        self.assertIsNotNone(env.grad)


if __name__ == "__main__":
    unittest.main()
