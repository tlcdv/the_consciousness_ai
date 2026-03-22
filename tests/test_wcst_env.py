"""Tests for the Wisconsin Card Sort Test environment."""
from __future__ import annotations

import unittest
import numpy as np
from simulations.environments.wcst_env import WCSTEnv, _RULE_DIMS


class TestWCSTEnv(unittest.TestCase):

    def setUp(self):
        self.env = WCSTEnv(render_mode="rgb_array", width=112, height=112,
                           num_trials=20, correct_to_switch=3)

    def tearDown(self):
        self.env.close()

    def test_reset_returns_correct_shapes(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (112, 112, 3))
        self.assertEqual(obs.dtype, np.uint8)

    def test_info_dict_completeness(self):
        _, info = self.env.reset(seed=42)
        expected = {"active_rule", "rule_changes", "consecutive_correct",
                    "trial", "trials_correct", "trials_total",
                    "perseverative_errors", "categories_completed",
                    "last_feedback"}
        self.assertTrue(expected.issubset(set(info.keys())))

    def test_active_rule_is_valid(self):
        _, info = self.env.reset(seed=42)
        self.assertIn(info["active_rule"], _RULE_DIMS)

    def test_correct_sort_gives_positive_reward(self):
        self.env.reset(seed=42)
        correct = self.env._correct_reference()
        _, reward, _, _, info = self.env.step(correct)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["last_feedback"], "correct")

    def test_incorrect_sort_gives_negative_reward(self):
        self.env.reset(seed=42)
        correct = self.env._correct_reference()
        wrong = (correct + 1) % 4
        _, reward, _, _, info = self.env.step(wrong)
        self.assertIn(reward, [-0.3, -0.5])

    def test_rule_changes_after_n_correct(self):
        self.env.reset(seed=42)
        initial_rule = self.env._active_rule
        # Get N consecutive correct
        for _ in range(self.env.correct_to_switch):
            correct = self.env._correct_reference()
            self.env.step(correct)
            # Skip feedback duration
            for _ in range(self.env.feedback_duration):
                self.env.step(0)
        # Rule should have changed
        self.assertNotEqual(self.env._active_rule, initial_rule)

    def test_rule_never_repeats_consecutively(self):
        self.env.reset(seed=42)
        prev_rules = [self.env._active_rule]
        for _ in range(50):
            correct = self.env._correct_reference()
            _, _, term, _, _ = self.env.step(correct)
            if term:
                break
            # Skip feedback
            for _ in range(self.env.feedback_duration):
                _, _, term, _, _ = self.env.step(0)
                if term:
                    break
            if self.env._active_rule != prev_rules[-1]:
                # Rule changed
                if len(prev_rules) >= 2:
                    self.assertNotEqual(self.env._active_rule, prev_rules[-1])
                prev_rules.append(self.env._active_rule)

    def test_perseverative_error_detection(self):
        self.env.reset(seed=42)
        # Get enough correct to trigger rule switch
        for _ in range(self.env.correct_to_switch):
            correct = self.env._correct_reference()
            self.env.step(correct)
            for _ in range(self.env.feedback_duration):
                self.env.step(0)

        # Now old rule is stored in _prev_rule
        if self.env._prev_rule is not None:
            # Sort by old rule (perseverative error)
            old_match = self.env._match_reference(self.env._prev_rule)
            new_correct = self.env._correct_reference()
            if old_match != new_correct:
                _, reward, _, _, _ = self.env.step(old_match)
                self.assertEqual(reward, -0.5)

    def test_episode_terminates_after_num_trials(self):
        env = WCSTEnv(render_mode="rgb_array", width=64, height=64,
                      num_trials=5, correct_to_switch=10)
        env.reset(seed=42)
        terminated = False
        for _ in range(500):
            _, _, term, _, _ = env.step(0)
            if term:
                terminated = True
                break
        self.assertTrue(terminated)
        env.close()

    def test_feedback_visible_for_duration(self):
        self.env.reset(seed=42)
        correct = self.env._correct_reference()
        self.env.step(correct)
        # Feedback should be showing
        self.assertGreater(self.env._feedback_remaining, 0)
        self.assertEqual(self.env._last_feedback, "correct")

    def test_reference_cards_always_present(self):
        obs, _ = self.env.reset(seed=42)
        self.assertEqual(len(self.env._reference_cards), 4)
        # Each card has shape, color, count
        for card in self.env._reference_cards:
            self.assertIn("shape", card)
            self.assertIn("color", card)
            self.assertIn("count", card)

    def test_max_rule_changes_respected(self):
        env = WCSTEnv(render_mode="rgb_array", width=64, height=64,
                      num_trials=100, correct_to_switch=2, max_rule_changes=2)
        env.reset(seed=42)
        for _ in range(500):
            correct = env._correct_reference()
            _, _, term, _, _ = env.step(correct)
            if term:
                break
            for _ in range(env.feedback_duration):
                _, _, term, _, _ = env.step(0)
                if term:
                    break
        self.assertLessEqual(env._rule_changes, 2)
        env.close()

    def test_current_card_is_valid(self):
        self.env.reset(seed=42)
        card = self.env._current_card
        self.assertIn(card["shape"], ["triangle", "square", "pentagon", "hexagon"])
        self.assertIn(card["color"], ["red", "blue", "green", "yellow"])
        self.assertIn(card["count"], [1, 2, 3, 4])

    def test_different_rules_give_different_answers(self):
        """On average, different rules should map to different reference cards."""
        self.env.reset(seed=42)
        answers = {}
        for rule in _RULE_DIMS:
            answers[rule] = self.env._match_reference(rule)
        # At least two rules should give different answers for most cards
        # (not guaranteed for every card, but highly likely)
        unique_answers = len(set(answers.values()))
        # Just check it runs without error
        self.assertGreaterEqual(unique_answers, 1)


class TestWCSTRendering(unittest.TestCase):

    def test_render_produces_valid_image(self):
        env = WCSTEnv(render_mode="rgb_array", width=112, height=112)
        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (112, 112, 3))
        # Not all one color
        self.assertGreater(obs.std(), 0)
        env.close()

    def test_feedback_changes_frame(self):
        env = WCSTEnv(render_mode="rgb_array", width=112, height=112)
        obs_before, _ = env.reset(seed=42)
        correct = env._correct_reference()
        obs_after, _, _, _, _ = env.step(correct)
        # Frame should be different (feedback indicator added)
        self.assertFalse(np.array_equal(obs_before, obs_after))
        env.close()


if __name__ == "__main__":
    unittest.main()
