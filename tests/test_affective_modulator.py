"""
Tests for the Affective Modulator (Tier 2 Architecture).

Validates:
1. Valence field: positive valence boosts approach bids, negative boosts threat bids
2. Arousal-threshold coupling: high arousal lowers ignition threshold
3. Neutral PAD causes no modulation
4. Integration with GlobalWorkspace
"""

import unittest
from models.emotion.affective_modulator import AffectiveModulator


class TestAffectiveModulator(unittest.TestCase):

    def setUp(self):
        self.modulator = AffectiveModulator({
            "valence_gain": 0.15,
            "arousal_gain": 0.2,
            "baseline_threshold": 0.6,
            "dominance_gain": 0.05,
        })
        self.base_bids = {
            "vision": 0.5,
            "audio": 0.5,
            "memory": 0.5,
            "body": 0.5,
        }

    def test_neutral_pad_no_modulation(self):
        """Neutral PAD state should produce negligible modulation."""
        pad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        modulated, threshold = self.modulator.modulate(dict(self.base_bids), pad)

        # Bids should be unchanged (no valence, no dominance)
        for name in self.base_bids:
            self.assertAlmostEqual(modulated[name], self.base_bids[name], places=5)

        # Threshold should be at baseline
        self.assertAlmostEqual(threshold, 0.6, places=5)

    def test_high_arousal_lowers_threshold(self):
        """High arousal should lower ignition threshold (easier ignition)."""
        pad = {"valence": 0.0, "arousal": 0.8, "dominance": 0.0}
        _, threshold = self.modulator.modulate(dict(self.base_bids), pad)

        # 0.6 - (0.8 * 0.2) = 0.6 - 0.16 = 0.44
        self.assertLess(threshold, 0.6)
        self.assertAlmostEqual(threshold, 0.44, places=2)

    def test_low_arousal_raises_threshold(self):
        """Negative arousal should raise ignition threshold (harder ignition)."""
        pad = {"valence": 0.0, "arousal": -0.5, "dominance": 0.0}
        _, threshold = self.modulator.modulate(dict(self.base_bids), pad)

        # 0.6 - (-0.5 * 0.2) = 0.6 + 0.10 = 0.70
        self.assertGreater(threshold, 0.6)

    def test_positive_valence_boosts_approach_bids(self):
        """Positive valence should boost approach-relevant module bids."""
        pad = {"valence": 0.8, "arousal": 0.0, "dominance": 0.0}
        modulated, _ = self.modulator.modulate(dict(self.base_bids), pad)

        # Vision, audio, memory, body are approach modules
        for name in ("vision", "audio", "memory", "body"):
            self.assertGreater(
                modulated[name], self.base_bids[name],
                f"{name} bid should be boosted by positive valence"
            )

    def test_negative_valence_boosts_threat_bids(self):
        """Negative valence should boost threat-relevant module bids."""
        pad = {"valence": -0.8, "arousal": 0.0, "dominance": 0.0}
        modulated, _ = self.modulator.modulate(dict(self.base_bids), pad)

        # Body, vision, and audio are threat modules (auditory startle reflex)
        for name in ("body", "vision", "audio"):
            self.assertGreater(
                modulated[name], self.base_bids[name],
                f"{name} bid should be boosted by negative valence"
            )

        # Memory is NOT a threat module, should be unchanged
        self.assertAlmostEqual(
            modulated["memory"], self.base_bids["memory"], places=5,
            msg="memory bid should not change with negative valence"
        )

    def test_high_dominance_boosts_all_bids(self):
        """Positive dominance should slightly boost all bids."""
        pad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.8}
        modulated, _ = self.modulator.modulate(dict(self.base_bids), pad)

        for name in self.base_bids:
            self.assertGreater(
                modulated[name], self.base_bids[name],
                f"{name} bid should be boosted by positive dominance"
            )

    def test_bids_clamped_to_valid_range(self):
        """Modulated bids should never exceed [0.0, 1.0]."""
        # Very high bids + strong modulation
        high_bids = {"vision": 0.98, "audio": 0.99, "memory": 0.95, "body": 0.97}
        pad = {"valence": 1.0, "arousal": 1.0, "dominance": 1.0}
        modulated, _ = self.modulator.modulate(high_bids, pad)

        for name, bid in modulated.items():
            self.assertLessEqual(bid, 1.0, f"{name} bid exceeds 1.0")
            self.assertGreaterEqual(bid, 0.0, f"{name} bid below 0.0")

    def test_threshold_clamped(self):
        """Threshold should stay within [0.2, 0.9]."""
        # Extreme arousal
        pad = {"valence": 0.0, "arousal": 1.0, "dominance": 0.0}
        _, threshold_high = self.modulator.modulate(dict(self.base_bids), pad)
        self.assertGreaterEqual(threshold_high, 0.2)

        pad = {"valence": 0.0, "arousal": -1.0, "dominance": 0.0}
        _, threshold_low = self.modulator.modulate(dict(self.base_bids), pad)
        self.assertLessEqual(threshold_low, 0.9)


class TestInteroceptiveAffect(unittest.TestCase):
    """Tests for the interoceptive-to-affect loop.

    Validates that homeostatic imbalance (low energy, high fatigue, damage)
    generates PAD signals that modulate sensory bids, closing the
    embodiment-affect loop (Feinberg & Mallatt gap #2).
    """

    def setUp(self):
        self.modulator = AffectiveModulator({
            "valence_gain": 0.15,
            "arousal_gain": 0.2,
            "baseline_threshold": 0.6,
            "dominance_gain": 0.05,
            "intero_gain": 0.4,
        })
        self.base_bids = {
            "vision": 0.5,
            "audio": 0.5,
            "memory": 0.5,
            "body": 0.5,
        }
        self.neutral_pad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        self.healthy_intero = {"energy": 1.0, "fatigue": 0.0, "damage": 0.0}

    def test_healthy_intero_no_effect(self):
        """Full energy, no fatigue, no damage should produce near-zero PAD delta."""
        delta = self.modulator.interoceptive_to_pad(self.healthy_intero)
        for key in ("valence", "arousal", "dominance"):
            self.assertAlmostEqual(delta[key], 0.0, places=3,
                                   msg=f"Healthy state should produce zero {key} delta")

    def test_low_energy_negative_valence(self):
        """Depleted energy should generate negative valence."""
        intero = {"energy": 0.1, "fatigue": 0.0, "damage": 0.0}
        delta = self.modulator.interoceptive_to_pad(intero)
        self.assertLess(delta["valence"], -0.05)

    def test_high_fatigue_suppresses_arousal(self):
        """Fatigue should reduce arousal (sluggishness)."""
        intero = {"energy": 1.0, "fatigue": 0.8, "damage": 0.0}
        delta = self.modulator.interoceptive_to_pad(intero)
        self.assertLess(delta["arousal"], -0.1)

    def test_high_fatigue_negative_valence(self):
        """Fatigue should also generate negative valence."""
        intero = {"energy": 1.0, "fatigue": 0.9, "damage": 0.0}
        delta = self.modulator.interoceptive_to_pad(intero)
        self.assertLess(delta["valence"], -0.05)

    def test_damage_negative_valence_and_high_arousal(self):
        """Damage (pain) should generate strong negative valence and arousal spike."""
        intero = {"energy": 1.0, "fatigue": 0.0, "damage": 0.8}
        delta = self.modulator.interoceptive_to_pad(intero)
        self.assertLess(delta["valence"], -0.3, "Damage should cause strong negative valence")
        self.assertGreater(delta["arousal"], 0.2, "Damage should cause arousal spike")

    def test_damage_reduces_dominance(self):
        """Damage should reduce dominance (feeling vulnerable)."""
        intero = {"energy": 1.0, "fatigue": 0.0, "damage": 0.7}
        delta = self.modulator.interoceptive_to_pad(intero)
        self.assertLess(delta["dominance"], -0.1)

    def test_intero_modulates_bids(self):
        """Low energy should shift bids via the valence field."""
        depleted = {"energy": 0.1, "fatigue": 0.5, "damage": 0.0}
        bids_with, thresh_with = self.modulator.modulate(
            dict(self.base_bids), self.neutral_pad, interoceptive_state=depleted
        )
        bids_without, thresh_without = self.modulator.modulate(
            dict(self.base_bids), self.neutral_pad, interoceptive_state=None
        )
        # Threat modules should get boosted (negative valence from depletion)
        self.assertGreater(bids_with["body"], bids_without["body"])

    def test_intero_shifts_threshold(self):
        """Damage-induced arousal should lower the ignition threshold."""
        damaged = {"energy": 1.0, "fatigue": 0.0, "damage": 0.8}
        _, thresh_damaged = self.modulator.modulate(
            dict(self.base_bids), self.neutral_pad, interoceptive_state=damaged
        )
        _, thresh_healthy = self.modulator.modulate(
            dict(self.base_bids), self.neutral_pad, interoceptive_state=self.healthy_intero
        )
        self.assertLess(thresh_damaged, thresh_healthy,
                        "Damage arousal should lower ignition threshold")

    def test_pad_deltas_clamped(self):
        """PAD deltas should stay within [-1, 1]."""
        extreme = {"energy": 0.0, "fatigue": 1.0, "damage": 1.0}
        delta = self.modulator.interoceptive_to_pad(extreme)
        for key in ("valence", "arousal", "dominance"):
            self.assertGreaterEqual(delta[key], -1.0)
            self.assertLessEqual(delta[key], 1.0)

    def test_combined_external_and_intero(self):
        """External positive valence should partially offset interoceptive negativity."""
        positive_pad = {"valence": 0.5, "arousal": 0.0, "dominance": 0.0}
        depleted = {"energy": 0.2, "fatigue": 0.3, "damage": 0.0}
        bids, _ = self.modulator.modulate(
            dict(self.base_bids), positive_pad, interoceptive_state=depleted
        )
        # Vision should still be boosted (positive external valence partially offsets)
        # but less than pure positive valence
        bids_pure, _ = self.modulator.modulate(
            dict(self.base_bids), positive_pad, interoceptive_state=None
        )
        # The interoceptive negative valence should reduce the boost
        self.assertLessEqual(bids["vision"], bids_pure["vision"])


class TestAffectiveModulatorIntegration(unittest.TestCase):
    """Test integration with GlobalWorkspace."""

    def test_workspace_accepts_modulator(self):
        """GlobalWorkspace should accept an affective modulator."""
        from models.core.global_workspace import GlobalWorkspace
        import torch

        config = {"ignition_threshold": 0.6, "ignition_gain": 10.0}
        gnw = GlobalWorkspace(config)

        modulator = AffectiveModulator()
        gnw.affective_modulator = modulator

        bids = {"vision": 0.7, "audio": 0.4, "memory": 0.5, "body": 0.3}
        payloads = {k: {"data": k} for k in bids}
        goal = torch.zeros(3)

        # Pass PAD state explicitly via the new run_competition kwarg.
        # The previous _current_pad_state magic attribute was removed because
        # production code never set it: the modulator pathway was silently
        # inert for the entire project history.
        broadcast, result_bids = gnw.run_competition(
            inputs={}, goal_vector=goal, bids=bids, payloads=payloads,
            pad_state={"valence": 0.5, "arousal": 0.6, "dominance": 0.3},
        )

        # Should complete without error
        self.assertIsInstance(result_bids, dict)


class TestValenceBidModulationValues(unittest.TestCase):
    """Numeric guarantees for the valence field modulation.

    Distinct from the shape tests in TestAffectiveModulator: these assert
    threshold-based percentage shifts, locking the contract that the
    modulator actually moves bids in the documented direction by a
    documented amount. Brought forward from peaceful-conjuring-castle
    Task 3 test #5.
    """

    def setUp(self):
        # Use a memory module: it is in APPROACH_MODULES but not THREAT_MODULES,
        # giving a clean read on each direction without overlap.
        self.modulator = AffectiveModulator({"valence_gain": 0.15,
                                             "dominance_gain": 0.0})
        self.bids = {"vision": 0.5, "audio": 0.5, "memory": 0.5, "body": 0.5}

    def test_negative_valence_boosts_threat_modules(self):
        neutral, _ = self.modulator.modulate(
            dict(self.bids),
            pad_state={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        )
        negative, _ = self.modulator.modulate(
            dict(self.bids),
            pad_state={"valence": -0.8, "arousal": 0.0, "dominance": 0.0},
        )
        # body is in THREAT_MODULES; assert > 5% relative boost
        rel_change = (negative["body"] - neutral["body"]) / neutral["body"]
        self.assertGreater(
            rel_change, 0.05,
            msg=f"negative valence only changed body bid by {rel_change*100:.2f}%",
        )

    def test_positive_valence_boosts_approach_modules(self):
        neutral, _ = self.modulator.modulate(
            dict(self.bids),
            pad_state={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        )
        positive, _ = self.modulator.modulate(
            dict(self.bids),
            pad_state={"valence": 0.8, "arousal": 0.0, "dominance": 0.0},
        )
        # memory is in APPROACH_MODULES but NOT THREAT_MODULES: clean signal
        rel_change = (positive["memory"] - neutral["memory"]) / neutral["memory"]
        self.assertGreater(
            rel_change, 0.05,
            msg=f"positive valence only changed memory bid by {rel_change*100:.2f}%",
        )

    def test_neutral_valence_does_not_modulate_memory(self):
        neutral, _ = self.modulator.modulate(
            dict(self.bids),
            pad_state={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        )
        self.assertAlmostEqual(neutral["memory"], 0.5, places=6)

    def test_explicit_None_pad_is_noop_at_workspace_level(self):
        """When run_competition gets pad_state=None it must not modulate.
        Locks the explicit-arg contract that replaced the magic attribute."""
        from models.core.global_workspace import GlobalWorkspace
        import torch

        gnw = GlobalWorkspace({"ignition_threshold": 0.6, "ignition_gain": 10.0})
        gnw.affective_modulator = AffectiveModulator()
        threshold_before = gnw.ignition_threshold

        bids = dict(self.bids)
        gnw.run_competition(
            inputs={}, goal_vector=torch.zeros(3),
            bids=bids, payloads={k: {"data": k} for k in bids},
            pad_state=None,
        )
        # No pad_state -> no threshold shift
        self.assertEqual(gnw.ignition_threshold, threshold_before)


if __name__ == "__main__":
    unittest.main()
