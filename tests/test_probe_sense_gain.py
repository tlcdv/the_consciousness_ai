"""Closed-form checks of the gain measures in scripts/analysis/probe_sense_gain.py.

The frames come from the real agent-centered renderer, so the self mask is checked against
the disc the environment draws, not against an assumed size.
"""
import unittest

import numpy as np

from scripts.analysis.probe_sense_gain import (
    audio_gain, audio_share, habituation_ratios, in_view_runs, vision_gain,
)
from simulations.environments.simple_visual_env import SimpleVisualEnv

LIGHT_GRAY = (255 + 255 + 200) / 3.0 / 255.0
WALL_GRAY = 60 / 255.0


def _frame(agent, light):
    env = SimpleVisualEnv(width=224, height=224, view="agent_centered")
    env.reset()
    env.agent_pos = np.array(agent, dtype=np.float64)
    env.light_pos = np.array(light, dtype=np.float64)
    return env._get_obs()


class TestVisionGain(unittest.TestCase):
    def test_agent_alone_in_a_dark_room_gives_zero(self):
        self.assertEqual(vision_gain(_frame((112, 112), (10, 10))), 0.0)

    def test_light_filling_a_cell_gives_the_light_gray(self):
        self.assertAlmostEqual(vision_gain(_frame((112, 112), (172, 112))), LIGHT_GRAY, places=6)

    def test_wall_filling_a_cell_gives_the_wall_gray(self):
        self.assertAlmostEqual(vision_gain(_frame((20, 112), (200, 200))), WALL_GRAY, places=6)


class TestAudioGain(unittest.TestCase):
    def test_tone_at_the_source_level_gives_one(self):
        t = np.arange(16000) / 16000.0
        tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        self.assertAlmostEqual(audio_gain(np.stack([tone] * 4)), 1.0, places=3)

    def test_half_amplitude_gives_half(self):
        t = np.arange(16000) / 16000.0
        tone = 0.25 * np.sin(2 * np.pi * 440.0 * t)
        self.assertAlmostEqual(audio_gain(np.stack([tone] * 4)), 0.5, places=3)

    def test_silence_gives_zero(self):
        self.assertEqual(audio_gain(np.zeros((4, 100))), 0.0)


class TestAudioShare(unittest.TestCase):
    def test_share_is_the_audio_fraction(self):
        share = audio_share(np.array([0.3, 0.0]), np.array([0.1, 0.0]))
        self.assertAlmostEqual(share[0], 0.75)
        self.assertEqual(share[1], 0.0)


class TestInViewRuns(unittest.TestCase):
    def test_a_gap_in_steps_splits_a_run(self):
        steps = np.array(list(range(21, 46)) + list(range(50, 75)))
        in_view = np.ones(len(steps), dtype=bool)
        runs = in_view_runs(steps, in_view)
        self.assertEqual([len(run) for run in runs], [25, 25])

    def test_out_of_view_step_ends_a_run_and_short_runs_are_dropped(self):
        steps = np.arange(21, 71)
        in_view = np.ones(50, dtype=bool)
        in_view[19] = False
        runs = in_view_runs(steps, in_view)
        self.assertEqual([len(run) for run in runs], [30])
        self.assertEqual(runs[0][0], 20)


class TestHabituationRatios(unittest.TestCase):
    def test_constant_measure_gives_one(self):
        ratios, excluded = habituation_ratios(np.full(30, 0.9), [list(range(30))])
        self.assertEqual((ratios, excluded), ([1.0], 0))

    def test_decay_to_half_gives_half(self):
        measure = np.array([0.8] * 10 + [0.4] * 10)
        self.assertEqual(habituation_ratios(measure, [list(range(20))]), ([0.5], 0))

    def test_zero_early_median_is_excluded(self):
        self.assertEqual(habituation_ratios(np.zeros(20), [list(range(20))]), ([], 1))


if __name__ == "__main__":
    unittest.main()
