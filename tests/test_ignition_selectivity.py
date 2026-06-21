"""Selective GNW ignition: is_conscious must discriminate, not saturate to always-on.

The 2026-06-21 self-review found consciousness_ratio == 1.000 on the trained agent:
the old `is_conscious = current_strength >= ignition_threshold` saturated by
construction (AKOrN-boosted bids sit above the fixed 0.6 threshold every step and the
sticky reverberation keeps current_strength above it forever). The fix makes ignition
SELECTIVE: a moment is conscious when its salience (max bound bid) rises above the
running baseline, so above-baseline moments ignite and below-baseline ones do not.

Honest scope (locked by test_constant_salience_is_the_low_variation_limit): with
near-CONSTANT input the salience is ~0 and the signal still does not discriminate. That
residual degeneracy is the deeper low-variation-dynamics issue (the same root as the
perception collapse), NOT something this fix claims to solve.
"""
import unittest

import torch

from models.core.global_workspace import GlobalWorkspace


def _run(gnw, energy):
    bids = {"vision": energy, "audio": 0.1, "memory": 0.1, "body": 0.1}
    gnw.run_competition(
        inputs={}, goal_vector=torch.zeros(3), bids=bids,
        payloads={k: {"data": k} for k in bids},
    )
    return bool(gnw.state.is_conscious)


class TestSelectiveIgnition(unittest.TestCase):
    def test_varying_salience_discriminates(self):
        """With oscillating input energy, is_conscious takes BOTH values: the
        below-baseline troughs are subconscious. The old absolute-threshold check
        would report conscious on every step; this locks that regression."""
        gnw = GlobalWorkspace({"ignition_threshold": 0.6, "ignition_gain": 10.0})
        states = [_run(gnw, e) for e in [0.95, 0.1, 0.95, 0.1, 0.95, 0.1, 0.95, 0.1]]
        self.assertTrue(any(states), "never conscious (over-suppressed)")
        self.assertFalse(all(states), "always conscious (saturated, the old bug)")

    def test_constant_salience_is_the_low_variation_limit(self):
        """Honest scope: near-constant input gives salience ~ 0, so the signal does
        not discriminate. This documents the residual degeneracy (low-variation
        dynamics) the selective fix does NOT claim to solve, so a future change that
        appears to 'fix' constant-input discrimination is flagged as suspicious."""
        gnw = GlobalWorkspace({"ignition_threshold": 0.6, "ignition_gain": 10.0})
        states = [_run(gnw, 0.8) for _ in range(12)]
        # After the baseline converges to the constant input, salience ~ 0; the
        # signal is uniform (degenerate) for constant input, by construction.
        self.assertTrue(all(states) or not any(states))


if __name__ == "__main__":
    unittest.main()
