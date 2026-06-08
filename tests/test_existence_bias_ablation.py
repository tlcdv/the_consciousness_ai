"""
Tests for the existence-bias ablation (Metzinger ethics, Phase 5 gated item A).

The `--ablate-existence-bias` flag (config key `ablate_existence_bias`) runs a
"no existence-bias" configuration: the affective modulator stops turning
interoceptive drives (energy/fatigue/damage) into affect, and the reward shaper
drops the homeostatic arousal penalty and the dominance/agency term.

These tests verify two things the plan requires:
1. The flag zeroes the targeted terms when ON.
2. The default (flag OFF) is bit-identical to the prior behavior.

This is an ablation experiment, not a claim about suffering. See
docs/ethics_framework.md and docs/metzinger_phenomenal_self_model.md.
"""

import unittest

from models.emotion.affective_modulator import AffectiveModulator
from models.emotion.reward_shaping import EmotionalRewardShaper


# A depleted, damaged, fatigued interoceptive state: the case where the
# existence drive generates the strongest affect.
DEPLETED_STATE = {"energy": 0.1, "fatigue": 0.8, "damage": 0.6}


class TestModulatorExistenceBiasAblation(unittest.TestCase):

    def test_interoceptive_pad_zeroed_when_ablated(self):
        m = AffectiveModulator({"ablate_existence_bias": True})
        pad = m.interoceptive_to_pad(DEPLETED_STATE)
        self.assertEqual(pad["valence"], 0.0)
        self.assertEqual(pad["arousal"], 0.0)
        self.assertEqual(pad["dominance"], 0.0)

    def test_interoceptive_pad_nonzero_by_default(self):
        # Default (flag off): a depleted, damaged state must generate negative
        # valence. This is the existence drive the ablation removes.
        m = AffectiveModulator()
        pad = m.interoceptive_to_pad(DEPLETED_STATE)
        self.assertLess(pad["valence"], 0.0)

    def test_default_is_bit_identical_to_explicit_off(self):
        m_default = AffectiveModulator()
        m_off = AffectiveModulator({"ablate_existence_bias": False})
        self.assertEqual(
            m_default.interoceptive_to_pad(DEPLETED_STATE),
            m_off.interoceptive_to_pad(DEPLETED_STATE),
        )

    def test_modulate_ablated_ignores_interoception(self):
        # With the flag on, passing an interoceptive state must produce the same
        # bids/threshold as passing none, because interoception adds zero affect.
        bids = {"vision": 0.5, "audio": 0.5, "memory": 0.5, "body": 0.5}
        pad = {"valence": 0.2, "arousal": 0.1, "dominance": 0.0}
        m = AffectiveModulator({"ablate_existence_bias": True})
        with_intero = m.modulate(bids, pad, interoceptive_state=DEPLETED_STATE)
        without_intero = m.modulate(bids, pad, interoceptive_state=None)
        self.assertEqual(with_intero, without_intero)

    def test_modulate_default_uses_interoception(self):
        # Sanity check the opposite direction: by default, interoception changes
        # the modulation outcome (otherwise the ablation would be meaningless).
        bids = {"vision": 0.5, "audio": 0.5, "memory": 0.5, "body": 0.5}
        pad = {"valence": 0.2, "arousal": 0.1, "dominance": 0.0}
        m = AffectiveModulator()
        with_intero = m.modulate(bids, pad, interoceptive_state=DEPLETED_STATE)
        without_intero = m.modulate(bids, pad, interoceptive_state=None)
        self.assertNotEqual(with_intero, without_intero)


class TestRewardShaperExistenceBiasAblation(unittest.TestCase):

    BASE_CFG = {"emotional_dims": 3, "hidden_size": 16}
    EMOTION = {"valence": 0.5, "arousal": 0.9, "dominance": 0.8}

    def test_ablated_drops_arousal_and_dominance_terms(self):
        # With the flag on, only the external reward and the valence term remain.
        shaper = EmotionalRewardShaper({**self.BASE_CFG, "ablate_existence_bias": True})
        reward = shaper.compute_emotional_reward(self.EMOTION, base_reward=1.0)
        expected = 1.0 + self.EMOTION["valence"] * shaper.valence_weight
        self.assertAlmostEqual(reward, expected, places=6)

    def test_default_includes_survival_terms(self):
        # Default (flag off): reward must include the dominance bonus and the
        # homeostatic arousal penalty, so it differs from the ablated value.
        shaper = EmotionalRewardShaper({**self.BASE_CFG})
        reward = shaper.compute_emotional_reward(self.EMOTION, base_reward=1.0)
        ablated = 1.0 + self.EMOTION["valence"] * shaper.valence_weight
        self.assertNotAlmostEqual(reward, ablated, places=6)
        # Verify each survival term is present with the correct sign.
        dominance_term = self.EMOTION["dominance"] * shaper.dominance_weight
        deviation = self.EMOTION["arousal"] - shaper.arousal_target
        arousal_penalty = shaper.arousal_lambda * (deviation ** 2)
        expected = ablated + dominance_term - arousal_penalty
        self.assertAlmostEqual(reward, expected, places=6)

    def test_default_is_bit_identical_to_explicit_off(self):
        default = EmotionalRewardShaper({**self.BASE_CFG})
        explicit_off = EmotionalRewardShaper({**self.BASE_CFG, "ablate_existence_bias": False})
        r_default = default.compute_emotional_reward(self.EMOTION, base_reward=1.0)
        r_off = explicit_off.compute_emotional_reward(self.EMOTION, base_reward=1.0)
        self.assertAlmostEqual(r_default, r_off, places=6)


if __name__ == "__main__":
    unittest.main()
