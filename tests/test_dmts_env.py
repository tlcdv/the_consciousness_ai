"""Tests for the Delayed Match-to-Sample environment."""
from __future__ import annotations

import unittest
import numpy as np
from simulations.environments.dmts_env import DMTSEnv
from simulations.environments._stimulus_renderer import SHAPE_NAMES, COLOR_NAMES


class TestDMTSEnv(unittest.TestCase):

    def setUp(self):
        self.env = DMTSEnv(render_mode="rgb_array", width=112, height=112,
                           num_trials=3, min_delay=5, max_delay=10)

    def tearDown(self):
        self.env.close()

    def test_reset_returns_correct_shapes(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (112, 112, 3))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertIn("phase", info)

    def test_initial_phase_is_fixation(self):
        _, info = self.env.reset(seed=42)
        self.assertEqual(info["phase"], "fixation")

    def test_fixation_to_sample_transition(self):
        self.env.reset(seed=42)
        for _ in range(self.env.fixation_steps):
            obs, reward, term, trunc, info = self.env.step(0)  # wait
        self.assertEqual(info["phase"], "sample")

    def test_sample_to_delay_transition(self):
        self.env.reset(seed=42)
        # Skip fixation
        for _ in range(self.env.fixation_steps):
            self.env.step(0)
        # Skip sample
        for _ in range(self.env.sample_steps):
            self.env.step(0)
        _, _, _, _, info = self.env.step(0)
        self.assertEqual(info["phase"], "delay")

    def test_full_phase_sequence(self):
        self.env.reset(seed=42)
        phases = set()
        # Run through enough steps to see all phases
        for _ in range(100):
            _, _, term, _, info = self.env.step(0)
            phases.add(info["phase"])
            if "choice" in phases:
                break
        self.assertIn("fixation", phases)
        self.assertIn("sample", phases)
        self.assertIn("delay", phases)
        self.assertIn("choice", phases)

    def test_correct_match_gives_positive_reward(self):
        self.env.reset(seed=42)
        # Advance to choice phase
        for _ in range(200):
            _, _, _, _, info = self.env.step(0)
            if info["phase"] == "choice":
                break
        # Select the target position
        target = info["target_position"]
        _, reward, _, _, _ = self.env.step(target)
        self.assertEqual(reward, 1.0)

    def test_incorrect_match_gives_negative_reward(self):
        self.env.reset(seed=42)
        for _ in range(200):
            _, _, _, _, info = self.env.step(0)
            if info["phase"] == "choice":
                break
        target = info["target_position"]
        wrong = 1 if target != 1 else 2
        _, reward, _, _, _ = self.env.step(wrong)
        self.assertEqual(reward, -0.5)

    def test_premature_response_penalized(self):
        self.env.reset(seed=42)
        # Act during fixation
        _, reward, _, _, info = self.env.step(1)
        self.assertEqual(reward, -0.2)
        self.assertEqual(info["phase"], "fixation")

    def test_fixation_wait_gives_shaping_reward(self):
        self.env.reset(seed=42)
        _, reward, _, _, _ = self.env.step(0)
        self.assertAlmostEqual(reward, 0.01)

    def test_timeout_penalty(self):
        env = DMTSEnv(render_mode="rgb_array", width=64, height=64,
                      num_trials=1, min_delay=2, max_delay=2,
                      choice_timeout=3)
        env.reset(seed=42)
        # Advance to choice
        for _ in range(100):
            _, _, _, _, info = env.step(0)
            if info["phase"] == "choice":
                break
        # Wait through timeout
        last_reward = 0.0
        for _ in range(5):
            _, r, term, _, info = env.step(0)
            last_reward = r
            if term:
                break
        self.assertEqual(last_reward, -0.3)
        env.close()

    def test_episode_terminates_after_num_trials(self):
        env = DMTSEnv(render_mode="rgb_array", width=64, height=64,
                      num_trials=1, min_delay=2, max_delay=2)
        env.reset(seed=42)
        terminated = False
        for _ in range(200):
            _, _, term, _, info = env.step(0)
            if info["phase"] == "choice":
                target = info["target_position"]
                _, _, term, _, _ = env.step(target)
                if term:
                    terminated = True
                    break
        self.assertTrue(terminated)
        env.close()

    def test_sample_stimulus_is_valid(self):
        _, info = self.env.reset(seed=42)
        self.assertIn(info["sample_shape"], SHAPE_NAMES)
        self.assertIn(info["sample_color"], COLOR_NAMES)
        self.assertIn(info["sample_size"], ["small", "large"])

    def test_delay_length_varies(self):
        delays = set()
        for seed in range(20):
            self.env.reset(seed=seed)
            delays.add(self.env._current_delay)
        self.assertGreater(len(delays), 1)

    def test_info_dict_completeness(self):
        _, info = self.env.reset(seed=42)
        expected_keys = {"phase", "trial", "sample_shape", "sample_color",
                         "sample_size", "target_position", "distractor_overlap",
                         "delay_length", "correct", "trials_correct", "trials_total"}
        self.assertTrue(expected_keys.issubset(set(info.keys())))

    def test_observation_changes_between_phases(self):
        self.env.reset(seed=42)
        fixation_obs, _, _, _, _ = self.env.step(0)
        # Advance to sample
        for _ in range(self.env.fixation_steps):
            sample_obs, _, _, _, info = self.env.step(0)
        if info["phase"] == "sample":
            self.assertFalse(np.array_equal(fixation_obs, sample_obs))


class TestDMTSDistractors(unittest.TestCase):

    def test_overlap_0_no_shared_features(self):
        env = DMTSEnv(distractor_overlap=0, num_choices=2)
        env.reset(seed=42)
        for stim in env._choice_stimuli:
            if stim["position"] != env._target_position:
                shared = sum([
                    stim["shape"] == env._sample_shape,
                    stim["color"] == env._sample_color,
                    stim["size"] == env._sample_size,
                ])
                self.assertEqual(shared, 0)
        env.close()

    def test_overlap_2_shares_two_features(self):
        env = DMTSEnv(distractor_overlap=2, num_choices=2)
        env.reset(seed=42)
        for stim in env._choice_stimuli:
            if stim["position"] != env._target_position:
                shared = sum([
                    stim["shape"] == env._sample_shape,
                    stim["color"] == env._sample_color,
                    stim["size"] == env._sample_size,
                ])
                # Should share exactly 2 (overlap capped at len(features)-1=2)
                self.assertEqual(shared, 2)
        env.close()


if __name__ == "__main__":
    unittest.main()
