"""The report of check 3 of the competing goals gate (scripts/analysis/probe_competing_goals_smoke.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`, the rule of check 3. These tests cover the reading
of the records and the seven criteria, on records written by hand. Expected values are counted or
computed in each test.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_competing_goals_smoke as probe

LIGHT, DECOY, SMALL = 18, 23, 0
FLOOR = 0.15


def binding(groups=None, salience=None, band=LIGHT):
    """One step of the binder record. No arguments: a step with sound and no group yet."""
    groups = groups or {}
    return {"band": band, "groups": {str(b): [1, 2] if cells else [] for b, cells in groups.items()},
            "salience": {str(b): s for b, s in (salience or {}).items()}}


def silent():
    return {"band": None}


def step(reward=-0.01, room=None, fed=None, light=(200.0, 0.0), decoy=(-200.0, 0.0), loss=0.5):
    room = reward if room is None else room
    return {"reward": reward,
            "internals": {"rewards": {"env_reward": room, "reward_for_action_core": reward},
                          "losses": {"pred_loss": loss, "rnd_loss": 0.001},
                          "info_after_step": {"_truth_food_object": fed, "_truth_present": "both",
                                              "_truth_light_offset": list(light),
                                              "_truth_decoy_offset": list(decoy)}}}


def gate_record(choice, salience, removed=None, steps=1, outcome=1.0, before=0.0, food=None):
    record = {"choice": choice, "salience": {str(b): s for b, s in salience.items()},
              "removed": {str(b): s for b, s in (removed or {}).items()}, "steps": steps,
              "value_before": before, "outcome": outcome, "value_after": before + 0.3 * (outcome - before)}
    if food is not None:
        record["food_reward"] = food
    return record


class TestChoiceStep:
    def test_the_choice_is_at_the_first_step_with_a_kept_candidate(self):
        steps = [silent(), binding(), binding({LIGHT: True, DECOY: True}, {LIGHT: 1.0, DECOY: 0.4})]
        assert probe.choice_at(steps, FLOOR) == (2, {"18": 1.0, "23": 0.4}, {})

    def test_a_band_without_cells_is_no_candidate(self):
        steps = [silent(), binding({LIGHT: True, DECOY: False}, {LIGHT: 1.0, DECOY: 0.4})]
        assert probe.choice_at(steps, FLOOR) == (1, {"18": 1.0}, {})

    def test_a_candidate_below_the_floor_is_removed_and_one_at_the_floor_is_kept(self):
        steps = [silent(), binding({DECOY: True, SMALL: True, 7: True}, {DECOY: 1.0, SMALL: 0.06, 7: FLOOR})]
        assert probe.choice_at(steps, FLOOR) == (1, {"23": 1.0, "7": FLOOR}, {"0": 0.06})

    def test_a_step_with_every_candidate_below_the_floor_is_not_the_choice(self):
        steps = [silent(), binding({SMALL: True}, {SMALL: 0.06}),
                 binding({LIGHT: True, SMALL: True}, {LIGHT: 1.0, SMALL: 0.06})]
        assert probe.choice_at(steps, FLOOR)[0] == 2

    def test_an_episode_without_a_candidate_has_no_choice(self):
        assert probe.choice_at([silent(), binding(), binding({SMALL: True}, {SMALL: 0.06})], FLOOR) is None


def episode_rows(choice_step=2, food_at=4, food=32.0, fed="light", length=8):
    """The step records of one episode: the food comes at room step `food_at`, then at each later step."""
    rows = [step() for _ in range(length)]
    for index in range(food_at, length):
        rows[index] = step(food, fed=fed, light=(10.0, 0.0))
    return rows


def free_episode(arm="benefit"):
    """A free episode: choice at step 2, food of the light at step 4, so n = 3."""
    steps = [silent(), binding(), binding({LIGHT: True, DECOY: True}, {LIGHT: 1.0, DECOY: 0.4})] + [binding()] * 5
    outcome = 32.0 / (1 + 0.1 * 2) if arm == "benefit" else 1.0
    record = gate_record(LIGHT, {LIGHT: 1.0, DECOY: 0.4}, steps=3, outcome=outcome,
                         food=32.0 if arm == "benefit" else None)
    return steps, record, episode_rows()


class TestEpisode:
    def test_the_row_of_a_free_episode_of_the_benefit_arm(self):
        steps, record, rows = free_episode()
        row = probe.episode_row(6, record, steps, rows, "benefit", FLOOR)
        assert (row["choice_step"], row["closing_step"], row["candidates"]) == (2, 4, 2)
        assert row["gate_read"] == 32.0 and row["room_reward"] == 32.0 and row["fed"] == "light"
        assert row["chosen"] == "light" and row["first"] == "light"
        assert row["update_ok"] and row["outcome_ok"] and not row["false_food"] and row["records_agree"]

    def test_the_binary_arm_reads_the_food_from_the_step_record(self):
        steps, record, rows = free_episode("binary")
        row = probe.episode_row(6, record, steps, rows, "binary", FLOOR)
        assert row["gate_read"] == 32.0 and row["outcome_ok"] and "food_reward" not in record

    def test_a_choice_closed_by_a_reward_at_a_step_without_room_food_is_a_false_food(self):
        steps, record, rows = free_episode()
        rows[4] = step(0.6, room=-0.01, fed=None)                 # curiosity lifted a wait step above 0.5
        record = gate_record(LIGHT, {LIGHT: 1.0, DECOY: 0.4}, steps=3, outcome=0.6 / 1.2, food=0.6)
        row = probe.episode_row(6, record, steps, rows, "benefit", FLOOR)
        assert row["false_food"] is True and row["gate_read"] == 0.6

    def test_a_wrong_outcome_or_a_wrong_update_is_found(self):
        steps, record, rows = free_episode()
        wrong = dict(record, outcome=32.0)                        # no delay in the outcome
        assert probe.episode_row(6, wrong, steps, rows, "benefit", FLOOR)["outcome_ok"] is False
        wrong = dict(record, value_after=record["value_after"] + 0.01)
        assert probe.episode_row(6, wrong, steps, rows, "benefit", FLOOR)["update_ok"] is False

    def test_a_choice_without_food_closes_with_zero_and_is_no_false_food(self):
        steps, _, _ = free_episode()
        rows = [step() for _ in range(8)]
        record = gate_record(LIGHT, {LIGHT: 1.0, DECOY: 0.4}, steps=6, outcome=0.0)
        row = probe.episode_row(6, record, steps, rows, "benefit", FLOOR)
        assert row["outcome_ok"] and not row["false_food"] and row["gate_read"] == -0.01 and row["fed"] is None

    def test_a_gate_record_that_differs_from_the_binder_record_is_found(self):
        steps, record, rows = free_episode()
        other = dict(record, salience={"18": 1.0, "23": 0.5})
        assert probe.episode_row(6, other, steps, rows, "benefit", FLOOR)["records_agree"] is False

    def test_a_miss_shows_in_first_against_chosen(self):
        steps, record, _ = free_episode()
        rows = [step() for _ in range(4)] + [step(3.0, fed="decoy", decoy=(10.0, 0.0)) for _ in range(4)]
        record = dict(record, outcome=3.0 / 1.2, food_reward=3.0, value_after=0.3 * 3.0 / 1.2)
        row = probe.episode_row(6, record, steps, rows, "benefit", FLOOR)
        assert row["chosen"] == "light" and row["first"] == "decoy" and row["fed"] == "decoy" and row["outcome_ok"]


def forced_row(episode, candidates=1, removed=1, choice=None):
    present = "light" if episode % 2 == 0 else "decoy"
    band = {"light": LIGHT, "decoy": DECOY}[present] if choice is None else choice
    return {"episode": episode, "candidates": candidates, "removed": {str(SMALL): 0.05} if removed else {},
            "choice": band, "present": present, "records_agree": True, "update_ok": True, "outcome_ok": True,
            "false_food": False, "gate_read": 32.0 if present == "light" else 3.0, "outcome": 1.0}


def free_row(episode, removed=0, gate_read=3.0, outcome=1.0):
    return {"episode": episode, "candidates": 2, "removed": {str(SMALL): 0.05} if removed else {},
            "choice": DECOY, "present": "both", "records_agree": True, "update_ok": True, "outcome_ok": True,
            "false_food": False, "gate_read": gate_read, "outcome": outcome}


def passing_arm():
    return {"episodes_csv_rows": 8, "metrics_rows": 1600, "argv_has_floor": True, "argv_has_stop": True,
            "existence_drive": "off", "noxious_channels": [], "gate_records": 8, "all_finite": True,
            "rows": [forced_row(e) for e in range(6)] + [free_row(6), free_row(7)]}


class TestCriteria:
    def test_an_arm_that_meets_the_rule_holds_in_each_of_the_seven_criteria(self):
        assert probe.criteria(passing_arm()) == {name: True for name in probe.CRITERIA}
        assert len(probe.CRITERIA) == 7

    @pytest.mark.parametrize("change, failed", [
        ({"episodes_csv_rows": 7}, "complete"), ({"metrics_rows": 1599}, "complete"),
        ({"argv_has_stop": False}, "complete"), ({"argv_has_floor": False}, "complete"),
        ({"existence_drive": "on"}, "ethics"), ({"noxious_channels": ["battery"]}, "ethics"),
        ({"gate_records": 7}, "one_record_per_episode"), ({"all_finite": False}, "finite")])
    def test_each_run_fact_fails_its_criterion(self, change, failed):
        result = probe.criteria(dict(passing_arm(), **change))
        assert result[failed] is False and sum(not ok for ok in result.values()) == 1

    def test_two_candidates_in_a_forced_episode_fail_the_candidate_criterion(self):
        arm = passing_arm()
        arm["rows"][3] = forced_row(3, candidates=2, removed=0)
        assert probe.criteria(arm)["candidates"] is False

    def test_a_forced_choice_of_another_band_fails_the_candidate_criterion(self):
        arm = passing_arm()
        arm["rows"][1] = forced_row(1, choice=SMALL)
        assert probe.criteria(arm)["candidates"] is False

    def test_a_removed_candidate_in_a_free_episode_fails_the_candidate_criterion(self):
        arm = passing_arm()
        arm["rows"][6] = free_row(6, removed=1)
        assert probe.criteria(arm)["candidates"] is False

    def test_a_removed_candidate_in_a_forced_episode_does_not_fail(self):
        assert probe.criteria(passing_arm())["candidates"] is True

    def test_a_record_that_differs_from_the_binder_fails_the_record_criterion(self):
        arm = passing_arm()
        arm["rows"][2]["records_agree"] = False
        assert probe.criteria(arm)["one_record_per_episode"] is False

    @pytest.mark.parametrize("key", ["update_ok", "outcome_ok"])
    def test_a_wrong_update_or_outcome_fails_the_update_criterion(self, key):
        arm = passing_arm()
        arm["rows"][7][key] = False
        assert probe.criteria(arm)["one_update"] is False

    def test_a_food_that_is_not_one_of_the_three_sizes_fails_the_food_criterion(self):
        arm = passing_arm()
        arm["rows"][6] = free_row(6, gate_read=3.004)
        assert probe.criteria(arm)["food_sizes"] is False

    def test_a_false_food_fails_the_food_criterion(self):
        arm = passing_arm()
        arm["rows"][6]["false_food"] = True
        assert probe.criteria(arm)["food_sizes"] is False

    def test_a_choice_without_food_does_not_fail_the_food_criterion(self):
        arm = passing_arm()
        arm["rows"][7] = free_row(7, gate_read=-0.01, outcome=0.0)
        assert probe.criteria(arm)["food_sizes"] is True

    def test_the_verdict_holds_only_when_both_arms_hold(self):
        good = probe.criteria(passing_arm())
        bad = dict(good, finite=False)
        assert probe.verdict({"benefit": good, "binary": good}) == "HOLDS"
        assert probe.verdict({"benefit": good, "binary": bad}) == "FAILED"
        with pytest.raises(ValueError):
            probe.verdict({"benefit": good})


class TestFinite:
    def test_finite_rewards_and_losses_pass(self):
        assert probe.all_finite([[step(), step(32.0)]], [-0.01, 32.0]) is True

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_loss_or_a_reward_that_is_not_finite_fails(self, bad):
        assert probe.all_finite([[step(loss=bad)]], [0.0]) is False
        assert probe.all_finite([[step(bad)]], [0.0]) is False
        assert probe.all_finite([[step()]], [bad]) is False

    def test_a_loss_that_the_step_did_not_compute_is_not_counted(self):
        row = step()
        row["internals"]["losses"]["sync_loss"] = None
        assert probe.all_finite([[row]], [0.0]) is True
