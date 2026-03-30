"""Tests for the auditory specialist (workspace-competing module)."""
from __future__ import annotations

import unittest
import torch
import numpy as np

from tests.utils.audio_generators import (
    generate_pure_tone, generate_white_noise, generate_silence,
    generate_harmonic_stack,
)


class TestAuditorySpecialist(unittest.TestCase):
    def setUp(self):
        from models.audio.auditory_specialist import AuditorySpecialist
        self.config = {
            "audio_sample_rate": 16000,
            "audio_num_bands": 64,
            "tectum_feature_dim": 64,
            "tectum_grid_size": 16,
            "workspace_dim": 256,
        }
        self.specialist = AuditorySpecialist(self.config)

    def _waveform_tensor(self, signal: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)

    def test_forward_with_waveform(self):
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        content, bid = self.specialist(waveform)
        self.assertEqual(content.shape, (1, 256))
        self.assertIsInstance(bid, float)

    def test_forward_none_returns_zeros(self):
        content, bid = self.specialist(None)
        self.assertEqual(content.shape, (1, 256))
        self.assertEqual(bid, 0.0)
        self.assertTrue(torch.all(content == 0).item())

    def test_forward_silence_returns_zero_bid(self):
        silence = generate_silence(0.066, 16000)
        waveform = self._waveform_tensor(silence)
        content, bid = self.specialist(waveform)
        self.assertEqual(bid, 0.0)

    def test_workspace_content_shape(self):
        tone = generate_harmonic_stack(220, 5, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        content, _ = self.specialist(waveform)
        self.assertEqual(content.shape, (1, 256))

    def test_bid_range(self):
        noise = generate_white_noise(0.066, 16000, 0.5)
        waveform = self._waveform_tensor(noise)
        _, bid = self.specialist(waveform)
        self.assertGreaterEqual(bid, 0.0)
        self.assertLessEqual(bid, 1.0)

    def test_receive_broadcast(self):
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        self.specialist(waveform)
        broadcast = torch.randn(1, 256)
        updated_bid = self.specialist.receive_broadcast(broadcast, 0.3)
        self.assertGreaterEqual(updated_bid, 0.0)
        self.assertLessEqual(updated_bid, 1.0)

    def test_spatial_output_shape(self):
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        self.specialist(waveform)
        spatial = self.specialist.get_spatial_for_tectum()
        self.assertEqual(spatial.shape, (1, 64, 2))

    def test_affect_output_keys(self):
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        self.specialist(waveform)
        affect = self.specialist.get_affect_output()
        self.assertIsNotNone(affect)
        self.assertIn("pad_delta", affect)
        self.assertIn("acoustic_features", affect)
        self.assertIn("paralinguistic", affect)

    def test_tonotopic_for_tectum(self):
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        self.specialist(waveform)
        tono = self.specialist.get_tonotopic_for_tectum(grid_size=16)
        self.assertIsNotNone(tono)
        self.assertEqual(tono.shape, (1, 64, 16, 16))

    def test_different_sounds_different_content(self):
        """Different audio inputs should produce different workspace content."""
        tone = generate_pure_tone(440, 0.066, 16000, 0.5)
        noise = generate_white_noise(0.066, 16000, 0.5)
        content_tone, _ = self.specialist(self._waveform_tensor(tone))
        content_noise, _ = self.specialist(self._waveform_tensor(noise))
        diff = (content_tone - content_noise).abs().sum().item()
        self.assertGreater(diff, 0.1)


if __name__ == "__main__":
    unittest.main()
