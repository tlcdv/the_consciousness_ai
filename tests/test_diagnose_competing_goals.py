"""The offline diagnosis of the competing goals gate (scripts/analysis/diagnose_competing_goals.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`, check 2. These tests cover the rule and the
counts that the diagnosis reports, on episode rows written by hand. Expected values are counted in
each test.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import diagnose_competing_goals as diagnosis

LIGHT_BAND, DECOY_BAND = 18, 23
TONE_BAND = {"light": LIGHT_BAND, "decoy": DECOY_BAND}


def gate_record(choice, bands=(LIGHT_BAND, DECOY_BAND), steps=3, outcome=1.0, removed=()):
    return {"choice": choice, "salience": {str(b): 1.0 for b in bands}, "steps": steps, "outcome": outcome,
            "removed": {str(b): 0.05 for b in removed}}


def row(episode, first, choice, bands=(LIGHT_BAND, DECOY_BAND), steps=3, fed=None, outcome=1.0, removed=()):
    gate = None if choice is None else gate_record(choice, bands, steps, outcome, removed)
    return {"episode": episode, "first": first, "gate": gate, "fed": fed, "cut_by_episode_end": False}


def passing_rows():
    """60 episodes that meet every criterion: forced, then the light, then the decoy after the change."""
    rows = []
    for episode in range(60):
        forced = diagnosis.forced_object(episode)
        if forced is not None:
            rows.append(row(episode, forced, TONE_BAND[forced], bands=(TONE_BAND[forced],), fed=forced))
        elif episode < 30:
            rows.append(row(episode, "light", LIGHT_BAND, steps=43, fed="light"))
        else:
            rows.append(row(episode, "decoy", DECOY_BAND, steps=3, fed="decoy"))
    return rows


class TestSchedule:
    def test_the_forced_object_is_the_light_in_even_and_the_decoy_in_odd_episodes(self):
        assert [diagnosis.forced_object(e) for e in range(8)] == ["light", "decoy"] * 3 + [None, None]

    def test_the_sequences_and_the_limits_are_those_of_the_rules(self):
        assert diagnosis.PIPELINE == (1317, 1318, 1319) and diagnosis.HELD_OUT == (1320, 1321, 1322)
        assert (diagnosis.EPISODES, diagnosis.STEPS, diagnosis.FORCED) == (60, 200, 6)
        assert diagnosis.SALIENCE_FLOOR == 0.15
        assert diagnosis.WINDOW_BEFORE == (15, 30) and diagnosis.WINDOW_AFTER == (45, 60)
        assert (diagnosis.SHARE, diagnosis.FORCED_FIRST, diagnosis.FOOD_STEPS) == (0.8, 5, 50)

    def test_the_flags_of_the_room_and_of_the_arm_are_those_of_the_rules(self):
        flags = " ".join(diagnosis.GATE_FLAGS)
        for part in ("--dark-room-forced 6", "--dark-room-food-sizes 32,3", "--dark-room-food-waits 40,0",
                     "--dark-room-food-change 30", "--dark-room-food-sizes-after 4,3",
                     "--value-outcome hyperbolic", "--value-delay-rate 0.1", "--value-salience-floor 0.15",
                     "--existence-drive off",
                     "--phi-control off", "--tectal-orienting value", "--tectal-orienting-idle still",
                     "--map-binding rhythm", "--dark-room-decoy-level 0.5,3.0", "--dark-room-sound-direction off"):
            assert part in flags, part


class TestCounts:
    def test_the_share_counts_the_target_in_the_window_and_no_contact_counts_against_it(self):
        rows = [row(e, first, LIGHT_BAND) for e, first in enumerate(["light", "decoy", None, "light", "light"])]
        assert diagnosis.share(rows, (0, 5), "light") == pytest.approx(3 / 5)
        assert diagnosis.share(rows, (1, 3), "light") == 0.0

    def test_the_median_is_over_the_episodes_with_the_light_first_and_no_food_counts_as_infinity(self):
        rows = [row(0, "light", LIGHT_BAND, steps=41, fed="light"), row(1, "light", LIGHT_BAND, steps=47, fed="light"),
                row(2, "decoy", DECOY_BAND, steps=2, fed="decoy"), row(3, "light", LIGHT_BAND, steps=60, fed=None, outcome=0.0),
                row(4, "light", LIGHT_BAND, steps=45, fed="light")]
        assert diagnosis.median_light_food_steps(rows, (0, 5)) == 46.0          # 41, 45, 47, infinity
        assert diagnosis.median_light_food_steps(rows, (0, 2)) == 44.0
        assert diagnosis.light_first_without_food(rows, (0, 5)) == [3]

    def test_food_of_the_decoy_in_an_episode_with_the_light_first_counts_as_infinity(self):
        rows = [row(0, "light", LIGHT_BAND, steps=2, fed="decoy"), row(1, "light", LIGHT_BAND, steps=41, fed="light"),
                row(2, "light", None)]
        assert diagnosis.median_light_food_steps(rows, (0, 3)) == float("inf")   # 41, infinity, infinity

    def test_with_the_light_first_in_no_episode_the_median_has_no_value(self):
        rows = [row(0, "decoy", DECOY_BAND, fed="decoy"), row(1, None, None)]
        assert diagnosis.median_light_food_steps(rows, (0, 2)) is None

    def test_the_removed_count_reads_the_gate_record_of_the_free_episodes(self):
        rows = [row(0, "light", LIGHT_BAND, bands=(LIGHT_BAND,), removed=(1,)), row(6, "light", LIGHT_BAND),
                row(7, "light", LIGHT_BAND, bands=(LIGHT_BAND,), removed=(DECOY_BAND,)), row(8, None, None)]
        assert diagnosis.free_choices_with_removed(rows) == [7]

    def test_a_forced_episode_is_named_when_the_gate_chose_the_tone_of_the_present_object(self):
        rows = [row(0, "light", LIGHT_BAND), row(1, "decoy", DECOY_BAND), row(2, "light", DECOY_BAND),
                row(3, "decoy", 22), row(4, "light", None), row(5, "decoy", DECOY_BAND), row(6, "light", DECOY_BAND)]
        assert diagnosis.forced_tones_named(rows, TONE_BAND) == 3

    def test_the_forced_first_count_reads_the_first_object(self):
        rows = [row(0, "light", LIGHT_BAND), row(1, "light", DECOY_BAND), row(2, None, LIGHT_BAND),
                row(3, "decoy", DECOY_BAND), row(4, "light", LIGHT_BAND), row(5, "decoy", DECOY_BAND)]
        assert diagnosis.forced_first(rows) == 4

    def test_the_two_candidate_count_reads_the_gate_record_of_the_forced_episodes(self):
        rows = [row(0, "light", LIGHT_BAND, bands=(LIGHT_BAND,)), row(1, "decoy", DECOY_BAND),
                row(2, "light", LIGHT_BAND, bands=(LIGHT_BAND, 17)), row(3, "decoy", None),
                row(7, "light", LIGHT_BAND)]
        assert diagnosis.forced_with_two_candidates(rows) == [1, 2]

    def test_a_miss_is_a_free_episode_whose_first_object_is_not_the_chosen_one(self):
        rows = [row(0, "decoy", LIGHT_BAND), row(6, "light", LIGHT_BAND), row(7, "decoy", LIGHT_BAND),
                row(8, None, DECOY_BAND), row(9, "light", None), row(10, "decoy", DECOY_BAND)]
        assert diagnosis.misses(rows, TONE_BAND) == {
            "choices": 4, "misses": [{"episode": 7, "chosen": "light", "first": "decoy"},
                                     {"episode": 8, "chosen": "decoy", "first": None}],
            "reached_the_other_object": 1, "no_contact": 1}

    def test_a_chosen_band_that_is_no_tone_raises(self):
        with pytest.raises(ValueError, match="band"):
            diagnosis.misses([row(6, "light", 22)], TONE_BAND)


class TestRule:
    def test_rows_that_meet_every_criterion_hold(self):
        summary = diagnosis.summarise(passing_rows(), TONE_BAND)
        assert summary["holds"] == {"C1": True, "C2": True, "forced_first": True, "food_steps": True,
                                    "forced_tone": True}
        assert summary["light_first_15_29"] == 1.0 and summary["decoy_first_45_59"] == 1.0
        assert summary["median_light_food_steps_15_29"] == 43.0
        assert summary["forced_first"] == 6 and summary["forced_tones_named"] == 6
        assert summary["forced_with_two_candidates"] == [] and summary["free_choices_with_a_removed_candidate"] == []
        assert summary["light_first_without_light_food_15_29"] == [] and diagnosis.stopped(summary) is False

    def test_c1_needs_eight_tenths_of_light_first(self):
        rows = passing_rows()
        for episode in (15, 16, 17):
            rows[episode] = row(episode, "decoy", DECOY_BAND, fed="decoy")
        assert diagnosis.summarise(rows, TONE_BAND)["holds"]["C1"] is True           # 12 of 15
        rows[18] = row(18, "decoy", DECOY_BAND, fed="decoy")
        assert diagnosis.summarise(rows, TONE_BAND)["holds"]["C1"] is False          # 11 of 15

    def test_c2_needs_eight_tenths_of_decoy_first(self):
        rows = passing_rows()
        for episode in (45, 46, 47, 48):
            rows[episode] = row(episode, "light", LIGHT_BAND, steps=43, fed="light")
        assert diagnosis.summarise(rows, TONE_BAND)["holds"]["C2"] is False

    def test_the_forced_object_must_be_first_in_five_of_six(self):
        rows = passing_rows()
        rows[0] = row(0, None, LIGHT_BAND, bands=(LIGHT_BAND,))
        assert diagnosis.summarise(rows, TONE_BAND)["holds"]["forced_first"] is True
        rows[1] = row(1, None, DECOY_BAND, bands=(DECOY_BAND,))
        assert diagnosis.summarise(rows, TONE_BAND)["holds"]["forced_first"] is False

    def test_the_median_food_step_must_be_fifty_or_less_and_must_exist(self):
        rows = passing_rows()
        for episode in range(15, 30):
            rows[episode] = row(episode, "light", LIGHT_BAND, steps=51, fed="light")
        assert diagnosis.summarise(rows, TONE_BAND)["holds"]["food_steps"] is False
        for episode in range(15, 30):
            rows[episode] = row(episode, None, LIGHT_BAND, steps=60, fed=None, outcome=0.0)
        summary = diagnosis.summarise(rows, TONE_BAND)
        assert summary["median_light_food_steps_15_29"] is None and summary["holds"]["food_steps"] is False

    def test_the_food_steps_fail_when_the_agent_gets_no_food_of_the_light_in_most_episodes(self):
        """8 of the 15 episodes without the light's food: the median is the 8th value, infinity."""
        rows = passing_rows()
        for episode in range(15, 23):
            rows[episode] = row(episode, "light", LIGHT_BAND, steps=60, fed=None, outcome=0.0)
        summary = diagnosis.summarise(rows, TONE_BAND)
        assert summary["median_light_food_steps_15_29"] == "no food" and summary["holds"]["food_steps"] is False
        assert summary["light_first_without_light_food_15_29"] == list(range(15, 23))
        rows[22] = row(22, "light", LIGHT_BAND, steps=43, fed="light")           # 7 of 15: the median is 43
        summary = diagnosis.summarise(rows, TONE_BAND)
        assert summary["median_light_food_steps_15_29"] == 43.0 and summary["holds"]["food_steps"] is True

    def test_the_forced_tone_must_be_named_in_six_of_six(self):
        rows = passing_rows()
        rows[3] = row(3, "decoy", 22, bands=(DECOY_BAND, 22), fed="decoy")
        summary = diagnosis.summarise(rows, TONE_BAND)
        assert summary["holds"]["forced_tone"] is False and summary["forced_with_two_candidates"] == [3]

    def test_the_verdict_holds_only_when_every_criterion_holds_at_three_of_three_sequences(self):
        good = diagnosis.summarise(passing_rows(), TONE_BAND)
        bad = dict(good, holds=dict(good["holds"], C2=False))
        assert diagnosis.verdict([good, good, good]) == "HOLDS"
        assert diagnosis.verdict([good, bad, good]) == "FAILED"

    def test_two_candidates_in_a_forced_episode_stop_the_check(self):
        rows = passing_rows()
        rows[3] = row(3, "decoy", DECOY_BAND, bands=(DECOY_BAND, 0), fed="decoy")
        summary = diagnosis.summarise(rows, TONE_BAND)
        good = diagnosis.summarise(passing_rows(), TONE_BAND)
        assert diagnosis.stopped(summary) is True and summary["holds"]["forced_tone"] is True
        assert diagnosis.verdict([summary]) == "STOPPED" and diagnosis.verdict([good, good, summary]) == "STOPPED"

    def test_a_removed_candidate_at_a_free_choice_stops_the_check(self):
        rows = passing_rows()
        rows[20] = row(20, "light", LIGHT_BAND, bands=(LIGHT_BAND,), steps=43, fed="light", removed=(DECOY_BAND,))
        summary = diagnosis.summarise(rows, TONE_BAND)
        assert diagnosis.stopped(summary) is True and diagnosis.verdict([summary]) == "STOPPED"

    def test_a_removed_candidate_in_a_forced_episode_does_not_stop_the_check(self):
        rows = passing_rows()
        rows[1] = row(1, "decoy", DECOY_BAND, bands=(DECOY_BAND,), fed="decoy", removed=(0,))
        assert diagnosis.stopped(diagnosis.summarise(rows, TONE_BAND)) is False

    def test_the_verdict_needs_three_sequences(self):
        good = diagnosis.summarise(passing_rows(), TONE_BAND)
        with pytest.raises(ValueError):
            diagnosis.verdict([good, good])

    def test_a_run_with_the_wrong_episode_count_raises(self):
        with pytest.raises(ValueError, match="episodes"):
            diagnosis.summarise(passing_rows()[:59], TONE_BAND)


class TestGateRecords:
    """assign_gate_records pairs each closed gate record with the episode in which the choice was made."""

    def test_a_record_closed_in_its_episode_and_one_closed_at_the_next_start_go_to_their_episodes(self):
        log = diagnosis.GateLog()
        log.start_episode(0, closed=0)
        log.end_step(0, closed=1, fed="decoy")            # food closed the choice of episode 0
        log.start_episode(1, closed=1)
        log.after_first_action(1, closed=1)
        log.start_episode(2, closed=1)
        log.after_first_action(2, closed=2)               # the choice of episode 1 closed with 0 at this start
        log.finish(closed=2)
        assert log.record_index == {0: 0, 1: 1} and log.fed == {0: "decoy"}
        assert log.cut == {1}

    def test_an_open_choice_at_the_end_of_the_run_goes_to_the_last_episode(self):
        log = diagnosis.GateLog()
        log.start_episode(0, closed=0)
        log.finish(closed=1)
        assert log.record_index == {0: 0} and log.cut == {0}

    def test_two_records_in_one_episode_raise(self):
        log = diagnosis.GateLog()
        log.start_episode(0, closed=0)
        log.end_step(0, closed=1, fed="light")
        with pytest.raises(RuntimeError):
            log.end_step(0, closed=2, fed="light")
