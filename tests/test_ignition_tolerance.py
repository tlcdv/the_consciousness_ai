"""Ignition with a tolerance band, measured once per environment step.

Gate B (docs/results/dark_room_senses_2026_09.md) found the workspace silent on 0.62 to
0.69 of steps with change-based bids. In an offline replay with 0.97 to 1.00 fidelity,
only this rule (V3) removed the silence at all 3 seeds: the running mean and sd of the
input energy move once per environment step (read before update), and a step ignites
when its energy is at least mean - k * sd.

The selectivity protection of tests/test_ignition_selectivity.py must still hold.
"""

import math
import unittest

import torch

from models.core.global_workspace import GlobalWorkspace
from models.core.reentrant_processor import ReentrantProcessor

FACTOR = 0.95


def _tolerance_workspace(k=1.0):
    return GlobalWorkspace({"ignition_gain": 10.0, "ignition_rule": "tolerance",
                            "ignition_tolerance_sd": k})


def _compete(gnw, energy):
    bids = {"vision": energy, "audio": 0.1, "memory": 0.1, "body": 0.1}
    gnw.run_competition(inputs={}, goal_vector=torch.zeros(3), bids=bids,
                        payloads={k: {"data": k} for k in bids})
    return bool(gnw.state.is_conscious)


class TestDefaultRuleUnchanged(unittest.TestCase):

    def test_default_rule_and_end_step_changes_nothing(self):
        gnw = GlobalWorkspace({"ignition_gain": 10.0})
        self.assertEqual(gnw.ignition_rule, "running_average")
        _compete(gnw, 0.8)
        baseline = gnw._energy_baseline
        gnw.end_step()
        self.assertEqual(gnw._energy_baseline, baseline)
        self.assertIsNone(gnw._tolerance_mean)

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            GlobalWorkspace({"ignition_rule": "always"})


class TestToleranceStatistics(unittest.TestCase):

    def test_first_update_sets_mean_and_zero_variance(self):
        gnw = _tolerance_workspace()
        gnw._update_tolerance(0.7)
        self.assertEqual((gnw._tolerance_mean, gnw._tolerance_var), (0.7, 0.0))

    def test_second_update_closed_form(self):
        gnw = _tolerance_workspace()
        gnw._update_tolerance(0.5)
        gnw._update_tolerance(0.9)
        delta = 0.9 - 0.5
        self.assertAlmostEqual(gnw._tolerance_mean, 0.5 + (1 - FACTOR) * delta, places=12)
        self.assertAlmostEqual(gnw._tolerance_var, FACTOR * (0.0 + (1 - FACTOR) * delta ** 2),
                               places=12)

    def test_ignition_level_is_energy_before_any_step_then_mean_minus_k_sd(self):
        gnw = _tolerance_workspace(k=2.0)
        self.assertEqual(gnw._ignition_level(0.42), 0.42)
        gnw._tolerance_mean, gnw._tolerance_var = 0.6, 0.01
        self.assertAlmostEqual(gnw._ignition_level(0.42), 0.6 - 2.0 * 0.1, places=12)


class TestToleranceBehaviour(unittest.TestCase):

    def test_statistics_do_not_move_inside_a_step(self):
        gnw = _tolerance_workspace()
        _compete(gnw, 0.7)
        gnw.end_step()
        before = (gnw._tolerance_mean, gnw._tolerance_var)
        _compete(gnw, 0.2)
        _compete(gnw, 0.9)
        self.assertEqual((gnw._tolerance_mean, gnw._tolerance_var), before)

    def test_end_step_uses_the_last_competition_energy(self):
        gnw = _tolerance_workspace()
        _compete(gnw, 0.7)
        energy = max(gnw.state.competition_results.values())
        gnw.end_step()
        self.assertEqual(gnw._tolerance_mean, energy)

    def test_small_dip_ignites_and_deep_drop_does_not(self):
        gnw = _tolerance_workspace(k=1.0)
        gnw._tolerance_mean, gnw._tolerance_var = 0.6, 0.04  # sd 0.2
        self.assertTrue(gnw._ignites(0.6 - 0.5 * 0.2))
        self.assertFalse(gnw._ignites(0.6 - 3.0 * 0.2))

    def test_varying_input_is_not_always_conscious(self):
        gnw = _tolerance_workspace(k=1.0)
        states = []
        for energy in [0.95, 0.05] * 15:
            states.append(_compete(gnw, energy))
            gnw.end_step()
        self.assertTrue(any(states))
        self.assertFalse(all(states))


class TestSettleEndsTheStep(unittest.TestCase):

    def test_settle_calls_end_step_once(self):
        gnw = _tolerance_workspace()
        calls = []
        original = gnw.end_step
        gnw.end_step = lambda: (calls.append(1), original())
        bids = {"vision": 0.7, "audio": 0.1}
        ReentrantProcessor({"max_cycles": 5, "convergence_threshold": 0.01}).settle(
            workspace=gnw, specialists={}, initial_bids=bids,
            payloads={k: {"data": k} for k in bids}, goal_vector=torch.zeros(3))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
