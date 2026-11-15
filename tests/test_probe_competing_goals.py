"""The probe of the competing goals gate (scripts/analysis/probe_competing_goals.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`, Gate section: C1 to C3 at 3 of 3 seeds, the
denominator of 15 episodes, VOID before any other verdict, and the cross-checks that have no effect on
the verdict. Expected values are counted in each test.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_competing_goals as probe


def firsts(early, late):
    """60 episodes: `early` fills episodes 15 to 29 and `late` fills episodes 45 to 59."""
    assert len(early) == 15 and len(late) == 15
    return ["light"] * 15 + list(early) + ["light"] * 15 + list(late)


def mix(light=0, decoy=0, neither=0):
    assert light + decoy + neither == 15
    return ["light"] * light + ["decoy"] * decoy + [None] * neither


GOOD_BENEFIT = firsts(mix(light=13, decoy=2), mix(light=2, decoy=13))
FLAT_BINARY = firsts(mix(light=12, decoy=3), mix(light=6, decoy=9))


class TestWindows:
    def test_the_count_reads_the_15_episodes_of_a_window(self):
        values = ["decoy"] * 15 + ["light"] * 15 + ["decoy"] * 30
        assert probe.window_count(values, probe.C1_WINDOW, "light") == 15
        assert probe.window_count(values, probe.C2_WINDOW, "decoy") == 15
        assert probe.window_count(values, probe.C2_WINDOW, "light") == 0

    def test_the_windows_are_episodes_15_to_29_and_45_to_59(self):
        assert (probe.C1_WINDOW, probe.C2_WINDOW) == ((15, 30), (45, 60))


class TestCriteria:
    def test_c1_needs_the_light_first_in_12_of_15(self):
        twelve = probe.seed_criteria(firsts(mix(light=12, decoy=3), mix(decoy=15)), FLAT_BINARY)
        eleven = probe.seed_criteria(firsts(mix(light=11, decoy=4), mix(decoy=15)), FLAT_BINARY)
        assert twelve["C1"] is True and eleven["C1"] is False
        assert twelve["light_first_15_29"] == 12

    def test_neither_stays_in_the_denominator(self):
        result = probe.seed_criteria(firsts(mix(light=11, neither=4), mix(decoy=11, neither=4)), FLAT_BINARY)
        assert result["C1"] is False and result["C2"] is False

    def test_c2_needs_the_decoy_first_in_12_of_15(self):
        twelve = probe.seed_criteria(firsts(mix(light=15), mix(decoy=12, light=3)), FLAT_BINARY)
        eleven = probe.seed_criteria(firsts(mix(light=15), mix(decoy=11, light=4)), FLAT_BINARY)
        assert twelve["C2"] is True and eleven["C2"] is False

    def test_c3_compares_the_change_of_the_two_arms_of_one_seed(self):
        # Benefit: decoy first 2 then 13, a change of 11. Binary: 3 then 9, a change of 6.
        # The difference is 5 of 15, 0.333, which is 0.3 or more.
        result = probe.seed_criteria(GOOD_BENEFIT, FLAT_BINARY)
        assert (result["change_benefit"], result["change_binary"]) == (11, 6)
        assert result["C3"] is True

    def test_a_difference_of_4_of_15_is_below_the_margin(self):
        # Binary: 3 then 10, a change of 7. The difference is 4 of 15, 0.267.
        binary = firsts(mix(light=12, decoy=3), mix(light=5, decoy=10))
        assert probe.seed_criteria(GOOD_BENEFIT, binary)["C3"] is False

    def test_c3_fails_when_the_binary_arm_changes_as_much(self):
        assert probe.seed_criteria(GOOD_BENEFIT, GOOD_BENEFIT)["C3"] is False


class TestVoid:
    def run(self, rows=12000, folders=60, first=None, second=None):
        first = GOOD_BENEFIT if first is None else first
        return {"run": "runs/x", "metrics_rows": rows, "episode_folders": folders, "first": first,
                "second": first if second is None else second}

    def test_a_complete_run_with_equal_counts_has_no_reason(self):
        assert probe.void_reasons(self.run()) == []

    def test_a_run_without_12000_rows_is_void(self):
        reasons = probe.void_reasons(self.run(rows=11999))
        assert len(reasons) == 1 and "11999" in reasons[0]

    def test_a_run_without_60_episode_folders_is_void(self):
        assert len(probe.void_reasons(self.run(folders=59))) == 1

    def test_a_difference_between_the_two_counts_is_void_and_names_the_episodes(self):
        other = list(GOOD_BENEFIT)
        other[17], other[50] = "decoy", None
        reasons = probe.void_reasons(self.run(second=other))
        assert len(reasons) == 1 and "17" in reasons[0] and "50" in reasons[0]


class TestVerdict:
    HOLD = {"C1": True, "C2": True, "C3": True}

    def test_holds_when_each_criterion_holds_at_3_of_3_seeds(self):
        assert probe.verdict([], {1301: self.HOLD, 1302: self.HOLD, 1303: self.HOLD}) == "HOLDS"

    def test_failed_when_one_criterion_fails_at_one_seed(self):
        fail = dict(self.HOLD, C2=False)
        assert probe.verdict([], {1301: self.HOLD, 1302: fail, 1303: self.HOLD}) == "FAILED"

    def test_void_comes_before_any_other_verdict(self):
        assert probe.verdict(["runs/x has 11999 rows"], {}) == "VOID"

    def test_the_verdict_needs_the_three_seeds_of_the_gate(self):
        with pytest.raises(ValueError):
            probe.verdict([], {1301: self.HOLD, 1302: self.HOLD})
        with pytest.raises(ValueError):
            probe.verdict([], {999: self.HOLD, 1302: self.HOLD, 1303: self.HOLD})


class TestCrossChecks:
    def test_the_chosen_object_is_the_object_of_the_chosen_tone_band(self):
        assert [probe.chosen_object(band) for band in (18, 23, 7)] == ["light", "decoy", None]

    def test_the_fed_object_is_the_first_food_of_the_episode(self):
        def step(fed):
            return {"internals": {"info_after_step": {"_truth_food_object": fed}}}
        assert probe.fed_object([step(None), step("decoy"), step("light")]) == "decoy"
        assert probe.fed_object([step(None), step(None)]) is None

    def test_the_counts_of_episodes_that_differ(self):
        first = ["light", "decoy", None, "light"]
        chosen = ["light", "light", "decoy", "light"]
        fed = ["light", "decoy", None, "decoy"]
        assert probe.cross_check(first, chosen, fed) == {
            "episodes": 4, "first_differs_from_choice": 2, "first_differs_from_first_food": 1}


class TestFlags:
    BASE = ["--seed", "1302", "--episodes", "60", "--max-steps", "200", "--value-salience-floor", "0.15",
            "--stop-on-non-finite"]
    BENEFIT = BASE + ["--value-outcome", "hyperbolic", "--value-delay-rate", "0.1"]

    def test_the_seed_of_a_gate_run_is_returned(self):
        assert probe.check_flags(self.BENEFIT, "benefit") == 1302
        assert probe.check_flags(self.BASE, "binary") == 1302

    def test_a_seed_outside_the_gate_is_refused(self):
        with pytest.raises(ValueError):
            probe.check_flags(["--seed", "999"] + self.BENEFIT[2:], "benefit")

    def test_an_arm_with_the_flags_of_the_other_arm_is_refused(self):
        with pytest.raises(ValueError):
            probe.check_flags(self.BASE, "benefit")
        with pytest.raises(ValueError):
            probe.check_flags(self.BENEFIT, "binary")

    def test_a_run_without_the_floor_or_the_stop_is_refused(self):
        with pytest.raises(ValueError):
            probe.check_flags([flag for flag in self.BENEFIT if flag != "--stop-on-non-finite"], "benefit")
        with pytest.raises(ValueError):
            probe.check_flags(self.BENEFIT[:6] + ["--value-salience-floor", "0.2"] + self.BENEFIT[8:], "benefit")


class TestSelfTest:
    def run(self, first, second, chosen):
        return {"run": "runs/x", "first": first, "second": second, "chosen": chosen}

    def test_it_holds_when_both_counts_give_the_chosen_object_in_each_episode(self):
        values = ["light", "decoy"] * 4
        result = probe.self_test([self.run(values, values, values), self.run(values, values, values)])
        assert result == {"episodes": 16, "first_count_equal": 16, "second_count_equal": 16, "holds": True}

    def test_it_fails_when_one_count_differs_in_one_episode(self):
        values = ["light", "decoy"] * 4
        other = ["decoy"] + values[1:]
        result = probe.self_test([self.run(values, other, values)])
        assert result == {"episodes": 8, "first_count_equal": 8, "second_count_equal": 7, "holds": False}
