"""Integration tests for the full auditory pipeline."""
from __future__ import annotations

import unittest
import torch
import numpy as np

from tests.utils.audio_generators import generate_pure_tone, generate_white_noise


class TestAuditoryIntegration(unittest.TestCase):
    """End-to-end integration tests for the auditory system."""

    def setUp(self):
        from models.audio.auditory_specialist import AuditorySpecialist
        from models.core.sensory_tectum import SensoryTectum
        from models.core.global_workspace import GlobalWorkspace

        self.config = {
            "audio_sample_rate": 16000,
            "audio_num_bands": 64,
            "tectum_feature_dim": 64,
            "tectum_grid_size": 16,
            "workspace_dim": 256,
        }
        self.specialist = AuditorySpecialist(self.config)
        self.tectum = SensoryTectum(self.config)
        self.workspace = GlobalWorkspace(self.config)

    def _waveform_tensor(self, signal: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)

    def test_full_pipeline(self):
        """Audio waveform -> specialist -> tectum spatial -> workspace bid."""
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)

        # Process through specialist
        audio_content, audio_bid = self.specialist(waveform)
        audio_spatial = self.specialist.get_spatial_for_tectum()

        # Process through tectum (with dummy visual frame)
        frame = torch.randn(1, 3, 224, 224)
        tectum_content, vision_bid = self.tectum(frame, audio_spatial)

        # Both should produce valid tensors
        self.assertEqual(audio_content.shape, (1, 256))
        self.assertEqual(tectum_content.shape, (1, 256))
        self.assertIsInstance(audio_bid, float)

    def test_audio_bid_in_competition(self):
        """Audio bid should be accepted by workspace competition."""
        tone = generate_pure_tone(440, 0.066, 16000, amplitude=0.8)
        waveform = self._waveform_tensor(tone)
        audio_content, audio_bid = self.specialist(waveform)

        bids = {
            "vision": 0.3,
            "audio": max(0.0, min(1.0, audio_bid)),
            "memory": 0.1,
            "body": 0.05,
            "semantic": 0.0,
        }
        payloads = {
            "vision": {"tensor": torch.randn(1, 256), "source": "tectum"},
            "audio": {"tensor": audio_content, "source": "audio"},
        }
        goal_vector = torch.tensor([1.0, -1.0, 1.0])

        state = self.workspace.run_competition(payloads, goal_vector, bids=bids)
        # Competition should complete without error
        self.assertIsNotNone(state)

    def test_backward_compat_no_audio(self):
        """System works exactly as before when audio is None."""
        # Simulate current training loop behavior
        audio_spatial = torch.zeros(1, 64, 2)
        frame = torch.randn(1, 3, 224, 224)
        tectum_content, vision_bid = self.tectum(frame, audio_spatial)
        self.assertEqual(tectum_content.shape, (1, 256))

        bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": 0.0,
            "memory": 0.1,
            "body": 0.05,
            "semantic": 0.0,
        }
        payloads = {"vision": {"tensor": tectum_content, "source": "tectum"}}
        goal_vector = torch.tensor([1.0, -1.0, 1.0])
        state = self.workspace.run_competition(payloads, goal_vector, bids=bids)
        self.assertIsNotNone(state)

    def test_affect_feeds_emotion(self):
        """Affect extraction produces valid PAD deltas."""
        noise = generate_white_noise(0.066, 16000, 0.5)
        waveform = self._waveform_tensor(noise)
        self.specialist(waveform)
        affect = self.specialist.get_affect_output()
        self.assertIsNotNone(affect)
        pad = affect["pad_delta"]
        self.assertEqual(pad.shape, (1, 3))
        self.assertTrue(torch.all(pad >= -1.0).item())
        self.assertTrue(torch.all(pad <= 1.0).item())

    def test_reentrant_with_audio(self):
        """Audio specialist participates in reentrant feedback."""
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        self.specialist(waveform)

        broadcast = torch.randn(1, 256)
        new_bid = self.specialist.receive_broadcast(broadcast, 0.3)
        self.assertGreaterEqual(new_bid, 0.0)
        self.assertLessEqual(new_bid, 1.0)
        # Bid should change from the broadcast
        self.assertNotAlmostEqual(new_bid, 0.3, places=3)

    def test_tonotopic_to_tectum_grid(self):
        """Tonotopic features reshape correctly for tectum grid."""
        tone = generate_pure_tone(440, 0.066, 16000)
        waveform = self._waveform_tensor(tone)
        self.specialist(waveform)
        grid = self.specialist.get_tonotopic_for_tectum(16)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.shape, (1, 64, 16, 16))


if __name__ == "__main__":
    unittest.main()
