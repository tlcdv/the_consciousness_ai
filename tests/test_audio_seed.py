"""The environment audio noise follows the run seed.

Before 2026-09-15 the audio generator was built with no seed, so two runs of the
same command with --enable-audio gave different metrics from step 1 (measured:
bid_audio 0.476660013 against 0.476288438 at the same step and seed).
"""

import unittest

import numpy as np

from simulations.environments.simple_visual_env import SimpleVisualEnv

STEPS = 5


def _waveforms(seed):
    env = SimpleVisualEnv(width=64, height=64)
    if seed is not None:
        env.seed_audio(seed)
    env.reset()
    waves = []
    for _ in range(STEPS):
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        waves.append(info["audio_waveform"].copy())
    return np.stack(waves)


class TestAudioSeed(unittest.TestCase):

    def test_same_seed_gives_identical_audio(self):
        self.assertTrue(np.array_equal(_waveforms(42), _waveforms(42)))

    def test_different_seeds_give_different_audio(self):
        self.assertFalse(np.array_equal(_waveforms(42), _waveforms(43)))

    def test_seeding_and_audio_use_no_global_random_state(self):
        # reset() itself draws positions from the global generator
        # (simple_visual_env.py:55-56), so only the audio calls are measured.
        env = SimpleVisualEnv(width=64, height=64)
        np.random.seed(7)
        before = np.random.get_state()[1].copy()
        env.seed_audio(42)
        env._generate_audio({"distance_to_light": 10.0, "in_light": False})
        self.assertTrue(np.array_equal(np.random.get_state()[1], before))


if __name__ == "__main__":
    unittest.main()
