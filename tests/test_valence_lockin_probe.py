"""The valence lock-in probe reads recorded steps. These pin its arithmetic on
constructed steps, with expected values worked out by hand."""
from __future__ import annotations

import math

import pytest

from scripts.analysis.probe_valence_lockin import (
    episode_row, learned_values, rho_share_against_gap, run_summary, vision_share)


def _step(winner, in_light=False, vision=0.0, audio=0.0):
    return {"winner": winner,
            "internals": {"info_after_step": {"in_light": in_light},
                          "learned_valence": {"values": {"vision": vision, "audio": audio}}}}


def test_vision_share_counts_ignited_steps_only():
    steps = [_step("vision"), _step("vision"), _step("audio"), _step(""), _step("")]

    assert vision_share(steps) == pytest.approx(2 / 3)


def test_vision_share_without_an_ignited_step_is_not_a_number():
    assert math.isnan(vision_share([_step(""), _step("")]))


def test_episode_row_takes_the_values_after_the_last_step():
    steps = [_step("vision", in_light=True, vision=1.0, audio=0.5),
             _step("audio", in_light=True, vision=2.0, audio=0.75),
             _step("vision", in_light=False, vision=4.0, audio=0.25)]

    row = episode_row(steps)

    assert row == {"vision_share": pytest.approx(2 / 3), "in_light": 2,
                   "value_vision": 4.0, "value_audio": 0.25}


def test_run_summary_means_are_over_all_steps_of_all_episodes():
    episodes = {0: [_step("vision", vision=1.0, audio=0.0), _step("vision", vision=3.0, audio=1.0)],
                1: [_step("audio", in_light=True, vision=5.0, audio=2.0)]}

    summary = run_summary(episodes)

    assert summary["vision_share"] == pytest.approx(2 / 3)
    assert summary["in_light"] == 1
    assert summary["mean_value_vision"] == pytest.approx(3.0)
    assert summary["mean_value_audio"] == pytest.approx(1.0)


def test_rho_is_one_when_the_share_rises_with_the_gap():
    summaries = [{"vision_share": share, "mean_value_vision": gap, "mean_value_audio": 0.0}
                 for share, gap in ((0.5, 0.1), (0.7, 0.9), (0.99, 4.0))]

    assert rho_share_against_gap(summaries) == pytest.approx(1.0)


def test_rho_needs_three_runs():
    summaries = [{"vision_share": 0.5, "mean_value_vision": 1.0, "mean_value_audio": 0.0}] * 2

    assert math.isnan(rho_share_against_gap(summaries))


def test_a_run_without_learned_valence_is_refused():
    step = {"winner": "vision", "internals": {"info_after_step": {}}}

    with pytest.raises(ValueError):
        learned_values(step)
