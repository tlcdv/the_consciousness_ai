"""The boost rule of the learned valence: `linear` (default) or `saturating`.

The linear boost is gain * |value| and the value has no upper limit, so one module can
receive a boost as large as the largest raw bid (docs/results/vision_lockin_2026_10.md).
The saturating rule is gain * tanh(|value|). It is at most the gain, and it equals the
linear rule to first order for small values. The rule changes the boost only. What the
critic learns is the same under both rules.

Expected values are closed forms.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.emotion.affective_modulator import AffectiveModulator
from models.emotion.learned_valence import BOOST_RULES, LearnedValence
from scripts.training import train_rlhf

GAIN = 0.15
NEUTRAL_PAD = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
# The highest values of vision in the recorded runs that fired the kill rule.
RECORDED_HIGH_VALUES = (5.87, 6.29, 7.02, 8.03, 10.85)


# --- the rule itself -----------------------------------------------------------------

def test_the_default_rule_is_linear():
    assert LearnedValence().boost_rule == "linear"
    assert BOOST_RULES == ("linear", "saturating")


def test_linear_boost_is_gain_times_size_and_has_no_upper_limit():
    learned = LearnedValence()
    learned.values = {"vision": 8.03}
    assert learned.boost("vision", GAIN) == pytest.approx(GAIN * 8.03, abs=1e-12)
    assert learned.boost("vision", GAIN) > 1.0


def test_saturating_boost_is_gain_times_tanh_of_the_size():
    learned = LearnedValence(boost_rule="saturating")
    learned.values = {"vision": 8.03, "audio": -0.4}
    assert learned.boost("vision", GAIN) == pytest.approx(GAIN * math.tanh(8.03), abs=1e-12)
    assert learned.boost("audio", GAIN) == pytest.approx(GAIN * math.tanh(0.4), abs=1e-12)


@pytest.mark.parametrize("size", (0.0, 0.5, 1.0) + RECORDED_HIGH_VALUES + (1e6,))
def test_saturating_boost_is_never_above_the_gain(size):
    learned = LearnedValence(boost_rule="saturating")
    learned.values = {"vision": size, "audio": -size}
    assert 0.0 <= learned.boost("vision", GAIN) <= GAIN
    assert learned.boost("audio", GAIN) == learned.boost("vision", GAIN)


@pytest.mark.parametrize("size", (0.0, 0.01, 0.05, 0.1))
def test_the_two_rules_agree_to_first_order_for_small_values(size):
    linear = LearnedValence()
    saturating = LearnedValence(boost_rule="saturating")
    linear.values = saturating.values = {"audio": size}
    # tanh(x) = x - x**3 / 3 + ..., an alternating series, so the gap is at most x**3 / 3.
    gap = linear.boost("audio", GAIN) - saturating.boost("audio", GAIN)
    assert 0.0 <= gap <= GAIN * size ** 3 / 3 + 1e-15


def test_a_module_without_a_value_gets_no_boost_under_either_rule():
    for rule in BOOST_RULES:
        assert LearnedValence(boost_rule=rule).boost("memory", GAIN) == 0.0


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="boost_rule"):
        LearnedValence(boost_rule="share")


def test_the_rule_does_not_change_what_the_critic_learns():
    linear = LearnedValence()
    saturating = LearnedValence(boost_rule="saturating")
    transitions = [({"vision": 0.9, "audio": 0.2}, 1.0), ({"vision": 0.7, "audio": 0.6}, 1.0),
                   ({"vision": 0.1, "audio": 0.8}, 0.0)]
    for learned in (linear, saturating):
        for bids, reward in transitions:
            learned.observe(bids, reward=reward)
        learned.end_episode()
    assert linear.values == saturating.values
    assert linear.last_td_error == saturating.last_td_error


# --- the rule inside the modulator ---------------------------------------------------

def _modulated(rule: str, values: dict, bids: dict) -> dict:
    modulator = AffectiveModulator()
    modulator.learned_valence = LearnedValence(boost_rule=rule)
    modulator.learned_valence.values = dict(values)
    modulated, _ = modulator.modulate(dict(bids), NEUTRAL_PAD)
    return modulated


def test_linear_rule_fills_the_bid_of_a_module_with_a_high_value():
    """The recorded defect: the boost alone puts the bid at its upper limit of 1.0."""
    modulated = _modulated("linear", {"vision": 8.03, "audio": 0.451},
                           {"vision": 0.0, "audio": 0.9})
    assert modulated["vision"] == 1.0
    assert modulated["vision"] > modulated["audio"]


def test_saturating_rule_keeps_the_raw_bid_order_at_the_same_values():
    modulated = _modulated("saturating", {"vision": 8.03, "audio": 0.451},
                           {"vision": 0.0, "audio": 0.9})
    assert modulated["vision"] == pytest.approx(GAIN * math.tanh(8.03), abs=1e-12)
    assert modulated["audio"] == pytest.approx(0.9 + GAIN * math.tanh(0.451), abs=1e-12)
    assert modulated["audio"] > modulated["vision"]


@pytest.mark.parametrize("vision_value", RECORDED_HIGH_VALUES)
def test_saturating_boost_moves_no_bid_by_more_than_the_gain(vision_value):
    bids = {"vision": 0.3, "audio": 0.5, "memory": 0.1}
    modulated = _modulated("saturating", {"vision": vision_value, "audio": 1.2}, bids)
    modulator_gain = AffectiveModulator().valence_gain
    for name in bids:
        assert 0.0 <= modulated[name] - bids[name] <= modulator_gain + 1e-12


# --- the command line ----------------------------------------------------------------

class _ParserCaptured(Exception):
    pass


def _parser() -> argparse.ArgumentParser:
    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _ParserCaptured

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(_ParserCaptured):
            train_rlhf.main()
    return captured["parser"]


def test_flag_defaults_to_linear_and_reaches_the_config():
    parser = _parser()
    default_args = parser.parse_args([])
    assert default_args.valence_boost == "linear"
    assert train_rlhf.build_config(default_args)["valence_boost"] == "linear"
    flagged = parser.parse_args(["--learned-valence", "--valence-boost", "saturating"])
    assert train_rlhf.build_config(flagged)["valence_boost"] == "saturating"


def test_build_config_without_the_attribute_keeps_linear():
    args = argparse.Namespace(action_dim=2, lr=1e-3, episodes=1, max_steps=10)
    assert train_rlhf.build_config(args)["valence_boost"] == "linear"


def test_saturating_without_learned_valence_is_rejected(capsys):
    """Without --learned-valence nothing reads the rule, so the flag would do nothing."""
    parser = _parser()
    args = parser.parse_args(["--valence-boost", "saturating"])
    with pytest.raises(SystemExit) as exit_info:
        train_rlhf._reject_incompatible_flags(parser, args)
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "--valence-boost" in message and "--learned-valence" in message


@pytest.mark.parametrize("argv", [
    [],
    ["--learned-valence"],
    ["--learned-valence", "--valence-boost", "saturating"],
    ["--valence-boost", "linear"],
])
def test_supported_combinations_are_accepted(argv):
    parser = _parser()
    train_rlhf._reject_incompatible_flags(parser, parser.parse_args(argv))


def test_help_states_the_limit_and_the_default():
    text = next(action.help for action in _parser()._actions
                if "--valence-boost" in action.option_strings)
    assert "tanh" in text
    assert "linear" in text and "default" in text
    assert "--learned-valence" in text


def test_the_training_loop_passes_the_rule_to_the_critic():
    source_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "training",
                               "train_rlhf.py")
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    assert 'LearnedValence(boost_rule=config.get("valence_boost", "linear"))' in source


# --- the session record --------------------------------------------------------------

def test_the_session_record_names_the_rule_the_run_used():
    """A gate reads this field at every step to confirm the rule was in use."""
    from scripts.training.session_capture import _learned_valence

    modulator = AffectiveModulator()
    assert _learned_valence(modulator) is None
    modulator.learned_valence = LearnedValence()
    assert _learned_valence(modulator)["boost_rule"] == "linear"
    modulator.learned_valence = LearnedValence(boost_rule="saturating")
    recorded = _learned_valence(modulator)
    assert recorded["boost_rule"] == "saturating"
    assert set(recorded) == {"values", "td_error", "boost_rule"}
