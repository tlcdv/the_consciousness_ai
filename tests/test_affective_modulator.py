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

        # Body and vision are threat modules
        for name in ("body", "vision"):
            self.assertGreater(
                modulated[name], self.base_bids[name],
                f"{name} bid should be boosted by negative valence"
            )

        # Audio and memory are NOT threat modules, should be unchanged
        for name in ("audio", "memory"):
            self.assertAlmostEqual(
                modulated[name], self.base_bids[name], places=5,
                msg=f"{name} bid should not change with negative valence"
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

        # Run competition with PAD state set
        gnw._current_pad_state = {"valence": 0.5, "arousal": 0.6, "dominance": 0.3}

        bids = {"vision": 0.7, "audio": 0.4, "memory": 0.5, "body": 0.3}
        payloads = {k: {"data": k} for k in bids}
        goal = torch.zeros(3)

        broadcast, result_bids = gnw.run_competition(
            inputs={}, goal_vector=goal, bids=bids, payloads=payloads
        )

        # Should complete without error
        self.assertIsInstance(result_bids, dict)


if __name__ == "__main__":
    unittest.main()
