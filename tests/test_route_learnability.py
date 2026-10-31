"""The inputs and the rule of the route learnability check."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import check_route_learnability as check
from scripts.analysis import diagnose_binder_motion as motions


class TestPlaceMask:
    def test_marks_the_cells_on_the_object_with_unit_norm(self):
        mask = check.place_mask(np.array([24.0, -24.0]), 12.0)
        assert (mask > 0).sum() > 0
        assert np.linalg.norm(mask) == pytest.approx(1.0)

    def test_is_all_zeros_out_of_view(self):
        assert not check.place_mask(np.array([500.0, 500.0]), 12.0).any()

    def test_the_mask_moves_with_the_object(self):
        right, left = check.place_mask(np.array([24.0, 0.0]), 12.0), check.place_mask(np.array([-24.0, 0.0]), 12.0)
        assert right @ left == 0.0


class TestRule:
    def test_needs_the_score_and_the_margin_in_episodes_21_to_40(self):
        ones, half = [0.0] * 20 + [1.0] * 20, [0.0] * 20 + [1.0, 0.0] * 10
        assert check.holds_for(ones, half) is True
        assert check.holds_for(half, half) is False
        assert check.holds_for([1.0] * 20 + [0.75] * 20, [0.0] * 40) is False
        assert check.holds_for(ones, [0.0] * 20 + [0.9] * 20) is False


class FakeRoom:
    def __init__(self, light, decoy):
        self.agent_pos, self.light_pos, self.decoy_pos = np.zeros(2), np.array(light, float), np.array(decoy, float)


class TestFirstContact:
    def test_names_the_object_in_reach_and_the_nearer_on_a_tie(self):
        assert check.first_contact(FakeRoom([30, 0], [100, 0])) == "light"
        assert check.first_contact(FakeRoom([100, 0], [30, 0])) == "decoy"
        assert check.first_contact(FakeRoom([30, 0], [20, 0])) == "decoy"
        assert check.first_contact(FakeRoom([100, 0], [100, 0])) == ""


class TestFollowUp:
    def test_the_cut_changes_the_input_only_after_episode_30(self):
        assert check.effective_arm("route_cut", 29) == "route"
        assert check.effective_arm("route_cut", 30) == "blank"
        assert check.effective_arm("route", 39) == "route"

    def test_f1_and_f2(self):
        learned = [0.0] * 20 + [1.0] * 20
        chance = [0.0] * 20 + [1.0, 0.0] * 10
        cut = [0.0] * 20 + [1.0] * 10 + [1.0, 0.0] * 5
        cell = {"route": {"light_first": learned}, "blank": {"light_first": chance}, "route_cut": {"light_first": cut}}
        assert check.follow_up_holds(cell) == {"F1": True, "F2": True}
        cell["route_cut"] = {"light_first": learned}
        assert check.follow_up_holds(cell)["F2"] is False


class TestBinderInput:
    def test_the_mask_of_a_group_and_of_no_group(self):
        mask = check.group_mask([1, 2, 3, 4])
        assert np.linalg.norm(mask) == pytest.approx(1.0)
        assert mask[1] == pytest.approx(0.5)
        assert not check.group_mask([]).any()

    def test_the_cut_arm_gets_the_constant_input_after_episode_30(self):
        basis = np.eye(check.DIM)
        before = check.binder_input("route_cut", 29, [5], basis).numpy().ravel()
        after = check.binder_input("route_cut", 30, [5], basis).numpy().ravel()
        assert before[5] == pytest.approx(1.0) and before[0] == 0.0
        assert after[0] == pytest.approx(1.0) and after[5] == 0.0


class TestMotionDiagnosis:
    def test_all_motions_hold_still_at_the_start_and_still_stays_still(self):
        rng = np.random.default_rng(0)
        offset = np.array([30.0, 40.0])
        for motion in motions.MOTIONS:
            assert not motions.motion_action(motion, motions.HOLD - 1, offset, rng).any()
        assert not motions.motion_action("still", 40, offset, rng).any()

    def test_smooth_moves_toward_the_target(self):
        action = motions.motion_action("smooth", 40, np.array([30.0, 40.0]), np.random.default_rng(0))
        assert action @ np.array([0.6, 0.8]) > 0.2

    def test_phase_shares_count_only_steps_with_a_group(self):
        grouped = [[1] * motions.STEPS, [0] * motions.STEPS]
        on_light = [[1] * motions.STEPS, [0] * motions.STEPS]
        shares = motions.phase_shares(on_light, grouped)
        assert shares["steps 31 to 60"] == {"group": 0.5, "on_light": 1.0}
