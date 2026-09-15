"""Learned valence: modules that predict task reward gain salience.

Replaces, when enabled, the fixed approach/threat module sets of AffectiveModulator,
which are this project's own invention (Feinberg & Mallatt: valence is assigned
separately by limbic structures). Learning follows temporal difference error (Bennett,
breakthrough 2). The reward is the external task reward only (ethics rule E2).
Expected values are closed forms.
"""

import pathlib
import unittest

from models.emotion.affective_modulator import AffectiveModulator
from models.emotion.learned_valence import (
    TD_DISCOUNT,
    TD_LEARNING_RATE,
    LearnedValence,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BIDS = {"vision": 0.5, "audio": 0.2, "memory": 0.1}
NEUTRAL_PAD = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}


class TestTemporalDifference(unittest.TestCase):

    def test_first_observation_only_stores(self):
        valence = LearnedValence()
        valence.observe(BIDS, reward=1.0)
        self.assertEqual(valence.values, {})
        self.assertIsNone(valence.last_td_error)

    def test_one_transition_closed_form(self):
        valence = LearnedValence()
        valence.observe({"audio": 0.4}, reward=1.0)
        valence.observe({"audio": 0.8}, reward=0.0)
        # values start at 0: delta = r + gamma * 0 - 0 = 1.0; v = alpha * delta * bid.
        self.assertEqual(valence.last_td_error, 1.0)
        self.assertAlmostEqual(valence.values["audio"], TD_LEARNING_RATE * 1.0 * 0.4, places=15)

    def test_second_transition_uses_the_learned_values(self):
        valence = LearnedValence()
        valence.observe({"audio": 0.4}, reward=1.0)
        valence.observe({"audio": 0.8}, reward=0.0)
        v = valence.values["audio"]
        valence.observe({"audio": 0.5}, reward=0.0)
        expected_delta = 0.0 + TD_DISCOUNT * v * 0.5 - v * 0.8
        self.assertAlmostEqual(valence.last_td_error, expected_delta, places=15)
        self.assertAlmostEqual(valence.values["audio"], v + TD_LEARNING_RATE * expected_delta * 0.8,
                               places=15)

    def test_episode_end_is_terminal(self):
        valence = LearnedValence()
        valence.observe({"audio": 0.5}, reward=2.0)
        valence.end_episode()
        # terminal: delta = r - V(last) = 2.0 - 0; the next episode starts without history.
        self.assertEqual(valence.last_td_error, 2.0)
        self.assertAlmostEqual(valence.values["audio"], TD_LEARNING_RATE * 2.0 * 0.5, places=15)
        valence.observe({"audio": 0.5}, reward=0.0)
        self.assertEqual(valence.last_td_error, 2.0)

    def test_a_silent_module_learns_nothing(self):
        valence = LearnedValence()
        valence.observe({"audio": 0.0, "vision": 0.6}, reward=1.0)
        valence.observe({"audio": 0.0, "vision": 0.6}, reward=0.0)
        self.assertEqual(valence.values["audio"], 0.0)
        self.assertGreater(valence.values["vision"], 0.0)


class TestModulation(unittest.TestCase):

    def test_default_modulator_has_no_learned_valence(self):
        self.assertIsNone(AffectiveModulator().learned_valence)

    def test_untrained_learned_valence_changes_no_bid(self):
        modulator = AffectiveModulator()
        modulator.learned_valence = LearnedValence()
        bids, _ = modulator.modulate(dict(BIDS), NEUTRAL_PAD)
        self.assertEqual(bids, BIDS)

    def test_boost_is_gain_times_absolute_value(self):
        modulator = AffectiveModulator()
        modulator.learned_valence = LearnedValence()
        modulator.learned_valence.values = {"audio": -0.4, "vision": 0.1}
        bids, _ = modulator.modulate(dict(BIDS), {"valence": 0.9, "arousal": 0.0, "dominance": 0.0})
        self.assertAlmostEqual(bids["audio"], 0.2 + modulator.valence_gain * 0.4, places=12)
        self.assertAlmostEqual(bids["vision"], 0.5 + modulator.valence_gain * 0.1, places=12)
        self.assertEqual(bids["memory"], 0.1)

    def test_training_loop_feeds_the_external_reward_only(self):
        source = (REPO_ROOT / "scripts" / "training" / "train_rlhf.py").read_text(encoding="utf-8")
        self.assertIn("learned_valence.observe(raw_bids, reward=env_reward)", source)


if __name__ == "__main__":
    unittest.main()
