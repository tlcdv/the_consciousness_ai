"""Tests for spatial audio computation (ITD/ILD localization)."""
from __future__ import annotations

import unittest
import torch

from models.audio.spatial_audio import SpatialAudioComputer


class TestSpatialAudioComputer(unittest.TestCase):
    def setUp(self):
        self.spatial = SpatialAudioComputer(sample_rate=16000)

    def test_mono_default_center(self):
        """Mono audio should localize to center (0, 0)."""
        waveform = torch.randn(1, 1, 1056)
        coords = self.spatial(waveform)
        self.assertEqual(coords.shape, (1, 2))
        self.assertAlmostEqual(coords[0, 0].item(), 0.0)
        self.assertAlmostEqual(coords[0, 1].item(), 0.0)

    def test_metadata_passthrough(self):
        """Environment-provided coordinates take priority."""
        waveform = torch.randn(1, 1, 1056)
        meta = {"audio_azimuth": -0.7, "audio_elevation": 0.3}
        coords = self.spatial(waveform, metadata=meta)
        self.assertAlmostEqual(coords[0, 0].item(), -0.7)
        self.assertAlmostEqual(coords[0, 1].item(), 0.3)

    def test_output_range(self):
        """Coordinates must be in [-1, 1]."""
        waveform = torch.randn(2, 2, 4000)
        coords = self.spatial(waveform)
        self.assertTrue(torch.all(coords >= -1.0).item())
        self.assertTrue(torch.all(coords <= 1.0).item())

    def test_expand_for_tectum(self):
        coords = torch.tensor([[0.5, -0.3]])
        expanded = SpatialAudioComputer.expand_for_tectum(coords, feature_dim=64)
        self.assertEqual(expanded.shape, (1, 64, 2))
        self.assertAlmostEqual(expanded[0, 0, 0].item(), 0.5)
        self.assertAlmostEqual(expanded[0, 63, 1].item(), -0.3)


if __name__ == "__main__":
    unittest.main()
