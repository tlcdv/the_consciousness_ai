"""Pins the helper functions of scripts/analysis/probe_workspace_action.py.

Expected values are derived by hand from the definitions, never copied from a run.
"""
import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from models.self_model.action_selection_core import BasalGanglia, PrefrontalCortex
from scripts.analysis.probe_workspace_action import (
    pathway_signals, relay_alone_reading, replay_mean,
    action_path, gain_reading, partial_link, reach_label, reach_ratio, residual_on_index,
    rotation_test, state_difference, unweighted_bid)


def table(f, a):
    return [(index, f[index], a[index], 50) for index in range(len(f))]


def test_action_path_defaults_to_the_policy_action():
    assert action_path(["--env", "dark_room"]) == (
        "dark_room", "off", "policy", "overt", "gonogo", "broadcast", "legacy")


def test_action_path_reads_the_value_path_and_covert():
    argv = ["--env", "dark_room", "--tectal-orienting", "value", "--tectal-orienting-idle", "still",
            "--tectal-orienting-covert", "--broadcast-merge", "top_winner"]
    assert action_path(argv)[1:4] == ("value", "still", "covert")
    assert action_path(argv)[6] == "top_winner"


def test_reach_ratio_is_zero_for_a_constant_action_mean():
    mean = np.tile([0.3, -0.2], (50, 1))
    noise = np.random.default_rng(0).normal(0.0, 0.1, (50, 2))
    assert reach_ratio(mean, noise) == pytest.approx(0.0, abs=1e-12)


def test_reach_ratio_uses_the_largest_mean_spread_and_the_smallest_noise_spread():
    mean = np.array([[0.0, 0.0], [2.0, 0.0]])
    noise = np.array([[0.0, 0.0], [4.0, 8.0]])
    assert reach_ratio(mean, noise) == pytest.approx(1.0 / 2.0)


@pytest.mark.parametrize("ratio, gap, label", [
    (0.5, 0.0, "SUPPORTED"), (0.1, 0.0, "EXCLUDED"), (0.2, 0.0, "UNDECIDED"), (0.9, 1e-6, "VOID")])
def test_reach_label_follows_the_stated_limits(ratio, gap, label):
    assert reach_label(ratio, gap) == label


def test_residual_on_index_is_zero_for_a_series_that_only_follows_the_index():
    assert np.allclose(residual_on_index([0.1, 0.4, 0.5, 0.9], [0, 1, 2, 3]), 0.0)


def test_partial_link_is_not_measurable_when_the_link_is_only_the_index():
    """The input that fools the raw criterion: f and a both rise with the episode number."""
    rising = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert partial_link([table(rising, rising)] * 3, shuffles=20) == {"measurable": False, "seeds": 3}


def test_partial_link_is_one_when_f_and_a_share_what_the_index_does_not_explain():
    shared = [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]
    link = partial_link([table(shared, shared)] * 3, shuffles=200)
    assert link["r"] == pytest.approx(1.0)
    assert link["positive_seeds"] == 3


def test_partial_link_is_negative_when_only_the_index_makes_the_raw_link_positive():
    """Both shares rise with the episode number, and they move against each other beside it."""
    f, a = [0.2, 0.1, 0.4, 0.3, 0.6, 0.5], [0.1, 0.3, 0.2, 0.5, 0.4, 0.6]
    assert spearmanr(f, a).correlation > 0
    link = partial_link([table(f, a)] * 3, shuffles=200)
    assert link["r"] < 0
    assert link["positive_seeds"] == 0
    assert link["p"] > 0.5


def test_state_difference_is_out_of_view_minus_in_view():
    audio = np.array([1.0, 1.0, 0.0, 1.0])
    in_view = np.array([False, False, True, True])
    assert state_difference(audio, in_view) == pytest.approx(1.0 - 0.5)


def test_rotation_test_leaves_out_an_episode_with_too_few_steps_in_one_state():
    audio, in_view = np.ones(20), np.array([True] * 15 + [False] * 5)
    assert rotation_test([(audio, in_view)], shuffles=10) == {"episodes": 0}


def test_rotation_test_gives_the_smallest_p_for_a_perfect_link_inside_the_episode():
    in_view = np.array([True] * 10 + [False] * 10)
    outcome = rotation_test([((~in_view).astype(float), in_view)], shuffles=100)
    assert outcome["mean_d"] == pytest.approx(1.0)
    assert outcome["positive_episodes"] == 1
    assert outcome["p"] == pytest.approx(1.0 / 101.0)


def test_rotation_test_is_not_fooled_by_an_audio_share_that_ignores_the_view():
    in_view = np.array([True, False] * 15)
    outcome = rotation_test([(np.ones(30), in_view)], shuffles=100)
    assert outcome["mean_d"] == 0.0
    assert outcome["p"] == 1.0


@pytest.mark.parametrize("recorded, share, expected", [
    (0.4, 0.25, 0.8), (0.9, 0.3, 1.0), (1.0, 0.5, 1.0), (1.0, 0.6, None), (0.3, 0.0, None), (0.0, 0.5, 0.0)])
def test_unweighted_bid_inverts_twice_the_share_or_returns_none(recorded, share, expected):
    assert unweighted_bid(recorded, share) == (None if expected is None else pytest.approx(expected))


def passing(mean_d, p, episodes=10):
    return {"episodes": episodes, "mean_d": mean_d, "p": p}


@pytest.mark.parametrize("weighted, plain, recoverable, agreement, reading", [
    (passing(0.3, 0.01), passing(-0.2, 1.0), 0.85, 0.95, "SUPPORTED"),
    (passing(0.3, 0.01), passing(0.2, 0.01), 0.85, 0.95, "EXCLUDED"),
    (passing(0.3, 0.01), passing(0.1, 0.01), 0.85, 0.95, "UNDECIDED"),
    (passing(0.3, 0.20), passing(0.2, 0.01), 0.85, 0.95, "NOT MEASURABLE"),
    (passing(0.3, 0.01, episodes=4), passing(0.2, 0.01), 0.85, 0.95, "NOT MEASURABLE"),
    (passing(0.3, 0.01), passing(-0.2, 1.0), 0.85, 0.89, "VOID"),
    (passing(0.3, 0.01), passing(-0.2, 1.0), 0.79, 0.95, "VOID")])
def test_gain_reading_applies_the_controls_before_the_rule(weighted, plain, recoverable, agreement, reading):
    assert gain_reading(weighted, plain, recoverable, agreement) == reading


def small_policy():
    torch.manual_seed(0)
    return PrefrontalCortex(8, 6).eval(), BasalGanglia(6, 2).eval()


def test_pathway_signals_give_the_input_of_the_relay_layer():
    """Closed form: the relay layer applied to the value read must give the action mean of the policy."""
    pfc, bg = small_policy()
    inputs = torch.randn(5, 8)
    signals = pathway_signals(pfc, bg, 6, inputs)
    with torch.no_grad():
        through_relay = bg.thalamic_relay(torch.tensor(signals["before_relay"])).clamp(-1.0, 1.0).numpy()
    assert np.allclose(through_relay, replay_mean(pfc, bg, 6, inputs), atol=1e-6)
    expected = signals["go"] * (1.0 - signals["no_go"]) * (1.0 - signals["brake"])
    assert np.allclose(signals["before_relay"], expected, atol=1e-6)


def test_pathway_signals_do_not_move_when_the_input_does_not_move():
    pfc, bg = small_policy()
    signals = pathway_signals(pfc, bg, 6, torch.zeros(1, 8).repeat(1, 1))
    again = pathway_signals(pfc, bg, 6, torch.zeros(1, 8))
    assert np.array_equal(signals["before_relay"], again["before_relay"])


@pytest.mark.parametrize("labels, reading", [
    (["SUPPORTED"] * 9 + ["EXCLUDED"] * 3, "SUPPORTED"),
    (["EXCLUDED"] * 9 + ["UNDECIDED"] * 3, "EXCLUDED"),
    (["EXCLUDED"] * 8 + ["UNDECIDED"] * 4, "UNDECIDED"),
    (["SUPPORTED"] * 6 + ["EXCLUDED"] * 6, "UNDECIDED")])
def test_relay_alone_reading_needs_three_quarters_of_the_runs(labels, reading):
    assert relay_alone_reading(labels) == reading
