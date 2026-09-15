"""Closed-form checks of models/core/precision_weighting.py.

Vision frames come from the real agent-centered renderer, so the mask over the
agent's own disc is checked against the disc the environment draws.
"""
import unittest

import numpy as np
import torch

from models.core import precision_weighting as pw
from simulations.environments.simple_visual_env import SimpleVisualEnv

LIGHT_GRAY = (255 + 255 + 200) / 3.0 / 255.0
WALL_GRAY = 60 / 255.0


def _frame_tensor(agent, light):
    env = SimpleVisualEnv(width=224, height=224, view="agent_centered")
    env.reset()
    env.agent_pos = np.array(agent, dtype=np.float64)
    env.light_pos = np.array(light, dtype=np.float64)
    frame = env._get_obs()
    return torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0)


def _tone(amplitude, samples=16000):
    t = np.arange(samples) / 16000.0
    wave = amplitude * np.sin(2 * np.pi * 440.0 * t)
    return torch.from_numpy(np.stack([wave] * 4)).float().unsqueeze(0)


class TestVisionGain(unittest.TestCase):
    def test_agent_alone_in_a_dark_room_gives_zero(self):
        self.assertEqual(pw.vision_gain(_frame_tensor((112, 112), (10, 10))), 0.0)

    def test_light_filling_a_cell_gives_the_light_gray(self):
        self.assertAlmostEqual(pw.vision_gain(_frame_tensor((112, 112), (172, 112))),
                               LIGHT_GRAY, places=5)

    def test_wall_filling_a_cell_gives_the_wall_gray(self):
        self.assertAlmostEqual(pw.vision_gain(_frame_tensor((20, 112), (200, 200))),
                               WALL_GRAY, places=5)


class TestAudioGain(unittest.TestCase):
    def test_tone_at_the_source_level_gives_one(self):
        self.assertAlmostEqual(pw.audio_gain(_tone(0.5)), 1.0, places=3)

    def test_half_amplitude_gives_half(self):
        self.assertAlmostEqual(pw.audio_gain(_tone(0.25)), 0.5, places=3)

    def test_silence_and_absent_audio_give_zero(self):
        self.assertEqual(pw.audio_gain(torch.zeros(1, 4, 100)), 0.0)
        self.assertEqual(pw.audio_gain(None), 0.0)


class TestWeightedSenseBids(unittest.TestCase):
    def test_equal_gains_leave_both_bids_unchanged(self):
        vision, audio = pw.weighted_sense_bids(0.8, 0.3, 0.4, 0.4)
        self.assertAlmostEqual(vision, 0.8)
        self.assertAlmostEqual(audio, 0.3)

    def test_no_gain_at_all_leaves_both_bids_unchanged(self):
        self.assertEqual(pw.weighted_sense_bids(0.8, 0.3, 0.0, 0.0), (0.8, 0.3))

    def test_the_louder_sense_keeps_more_of_its_bid(self):
        # shares 0.1 and 0.9; factors 0.2 and 1.8
        vision, audio = pw.weighted_sense_bids(0.5, 0.5, 0.1, 0.9)
        self.assertAlmostEqual(vision, 0.1)
        self.assertAlmostEqual(audio, 0.9)

    def test_bids_stay_inside_zero_and_one(self):
        vision, audio = pw.weighted_sense_bids(1.0, 1.0, 0.9, 0.1)
        self.assertEqual(vision, 1.0)
        self.assertAlmostEqual(audio, 0.2)

    def test_shares_sum_to_one(self):
        share_v, share_a = pw.precision_shares(0.3, 0.1)
        self.assertAlmostEqual(share_v + share_a, 1.0)
        self.assertAlmostEqual(share_v, 0.75)


if __name__ == "__main__":
    unittest.main()
