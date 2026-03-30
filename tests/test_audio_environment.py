"""Tests for environment audio synthesis mixin."""
from __future__ import annotations

import unittest
import numpy as np

from simulations.environments.audio_mixin import (
    AudioMixin, DarkRoomAudioMixin,
    adsr_envelope, fm_tone, noise_burst, apply_reverb, ascending_sequence,
)


class TestSynthesisPrimitives(unittest.TestCase):
    def test_adsr_shape(self):
        env = adsr_envelope(0.066, 16000)
        self.assertEqual(len(env), int(0.066 * 16000))

    def test_adsr_starts_zero_ends_zero(self):
        env = adsr_envelope(0.1, 16000, attack=0.01, decay=0.02,
                            sustain_level=0.5, release=0.02)
        self.assertAlmostEqual(env[0], 0.0, places=2)
        self.assertAlmostEqual(env[-1], 0.0, places=1)

    def test_adsr_range(self):
        env = adsr_envelope(0.1, 16000)
        self.assertTrue(np.all(env >= 0))
        self.assertTrue(np.all(env <= 1.01))

    def test_fm_tone_shape(self):
        tone = fm_tone(440, 5, 50, 0.066, 16000)
        self.assertEqual(len(tone), int(0.066 * 16000))

    def test_fm_tone_not_silence(self):
        tone = fm_tone(440, 5, 50, 0.066, 16000)
        self.assertGreater(np.abs(tone).max(), 0.1)

    def test_noise_burst_short(self):
        burst = noise_burst(0.015, 16000)
        self.assertEqual(len(burst), int(0.015 * 16000))

    def test_reverb_preserves_length(self):
        signal = np.random.randn(1056).astype(np.float32)
        reverbed = apply_reverb(signal, room_size=0.3, sample_rate=16000)
        self.assertEqual(len(reverbed), len(signal))

    def test_ascending_sequence_longer_than_single_note(self):
        seq = ascending_sequence(400, 3, 0.02, 16000)
        single = int(0.02 * 16000)
        self.assertGreater(len(seq), single)


class TestAudioMixin(unittest.TestCase):
    def setUp(self):
        class DummyEnv(AudioMixin):
            pass
        self.env = DummyEnv()

    def test_generate_audio_returns_array(self):
        audio = self.env._generate_audio({})
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)

    def test_audio_length_matches_duration(self):
        audio = self.env._generate_audio({})
        expected = int(self.env._audio_frame_duration * self.env._audio_sample_rate)
        self.assertEqual(len(audio), expected)

    def test_audio_in_range(self):
        """Output should be soft-clipped to [-1, 1]."""
        audio = self.env._generate_audio({"reward": 1.0})
        self.assertTrue(np.all(audio >= -1.0))
        self.assertTrue(np.all(audio <= 1.0))

    def test_reward_positive_produces_event(self):
        """Positive reward should produce different audio than no event."""
        audio_reward = self.env._generate_audio({"reward": 1.0})
        audio_none = self.env._generate_audio({})
        # They should differ (reward triggers ascending sequence)
        diff = np.abs(audio_reward - audio_none).sum()
        self.assertGreater(diff, 0.1)

    def test_collision_produces_event(self):
        audio = self.env._generate_audio({"collision": True})
        energy = np.sum(audio ** 2)
        audio_quiet = self.env._generate_audio({})
        energy_quiet = np.sum(audio_quiet ** 2)
        self.assertGreater(energy, energy_quiet)


class TestDarkRoomAudioMixin(unittest.TestCase):
    def setUp(self):
        class DummyDarkRoom(DarkRoomAudioMixin):
            pass
        self.env = DummyDarkRoom()

    def test_proximity_changes_audio(self):
        audio_close = self.env._generate_audio({"distance_to_light": 1.0})
        audio_far = self.env._generate_audio({"distance_to_light": 9.0})
        close_energy = np.sum(audio_close ** 2)
        far_energy = np.sum(audio_far ** 2)
        self.assertGreater(close_energy, far_energy)

    def test_in_light_chord(self):
        """In-light chord event should differ from no-event ambient noise."""
        self.env._audio_rng = np.random.default_rng(42)
        audio_light = self.env._generate_audio({"in_light": True})
        self.env._audio_rng = np.random.default_rng(42)
        audio_none = self.env._generate_audio({})
        diff = np.abs(audio_light - audio_none).sum()
        self.assertGreater(diff, 0.1)


if __name__ == "__main__":
    unittest.main()
