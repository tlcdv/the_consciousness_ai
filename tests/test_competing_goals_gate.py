"""The hyperbolic outcome of the value gate and the flags of the competing goals gate.

`docs/decisions/2026_11_08_competing_goals_gate.md`, check 1. The outcome is the reward of the food
step divided by (1 + k (n - 1)), where n counts the steps from the choice to the food. Expected values
are closed forms computed in each test. The binary outcome is the default and must stay as it is.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import value_gate as vg
from scripts.analysis import model_competing_goals as model

K = 0.1
BIG, SMALL, DECOY_FOOD = 32.0, 4.0, 3.0
STEP_COST = -0.01
BINARY_RECORD_KEYS = ["choice", "salience", "priorities", "value_before", "steps", "outcome", "error", "value_after"]


def hyperbolic(size: float, n: int, k: float = K) -> float:
    return size / (1.0 + k * (n - 1))


def feed(gate: vg.ValueGate, size: float, n: int, band: int = 18) -> dict:
    """One episode in which the food of `size` comes at step n after the choice. Returns its record."""
    gate.choose({band: 1.0})
    for _ in range(n - 1):
        gate.observe_reward(STEP_COST)
    gate.observe_reward(size)
    gate.new_episode()
    return gate.episodes[-1]


class TestHyperbolicOutcome:
    @pytest.mark.parametrize("size, n", [(DECOY_FOOD, 1), (DECOY_FOOD, 3), (BIG, 43), (SMALL, 43), (BIG, 60)])
    def test_the_outcome_is_the_food_size_divided_by_one_plus_k_times_n_minus_one(self, size, n):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        record = feed(gate, size, n)
        assert record["steps"] == n
        assert record["outcome"] == pytest.approx(hyperbolic(size, n), rel=1e-12)

    def test_food_at_the_first_step_gives_the_size_exactly(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        assert feed(gate, DECOY_FOOD, 1)["outcome"] == DECOY_FOOD

    def test_a_delay_rate_of_zero_gives_the_size_at_any_delay(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=0.0)
        assert feed(gate, BIG, 43)["outcome"] == BIG

    def test_the_value_after_t_episodes_is_the_outcome_times_one_minus_seven_tenths_to_the_t(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        o = hyperbolic(BIG, 43)
        for t in range(1, 21):
            feed(gate, BIG, 43)
            assert gate.values[18] == pytest.approx(o * (1.0 - 0.7 ** t), rel=1e-12)

    def test_the_design_of_the_rules_converges_to_its_stated_values(self):
        """Median travel 3: decoy 3 / 1.2, light 32 / 5.2 before the change and 4 / 5.2 after it."""
        assert hyperbolic(DECOY_FOOD, 3) == pytest.approx(2.500, abs=5e-4)
        assert hyperbolic(BIG, 43) == pytest.approx(6.154, abs=5e-4)
        assert hyperbolic(SMALL, 43) == pytest.approx(0.769, abs=5e-4)

    def test_no_food_within_sixty_steps_gives_zero(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        gate.choose({18: 1.0})
        for _ in range(vg.OUTCOME_STEPS):
            gate.observe_reward(STEP_COST)
        gate.observe_reward(BIG)                          # step 61: the choice is closed already
        assert len(gate.episodes) == 1 and gate.episodes[0]["outcome"] == 0.0

    def test_a_reward_of_one_half_or_less_is_not_food(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        gate.choose({18: 1.0})
        gate.observe_reward(vg.FOOD)
        assert gate.episodes == []

    def test_the_food_updates_once_and_later_food_of_the_episode_is_ignored(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        gate.choose({18: 1.0})
        gate.observe_reward(BIG)
        gate.observe_reward(BIG)
        assert len(gate.episodes) == 1 and gate.values[18] == pytest.approx(0.3 * BIG)

    def test_the_record_holds_the_food_reward(self):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        assert feed(gate, BIG, 43)["food_reward"] == BIG

    @pytest.mark.parametrize("n", [1, 3, 21, 43, 50, 60, 61])
    def test_the_gate_and_the_abstract_model_give_the_same_outcome(self, n):
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        gate.choose({18: 1.0})
        for step in range(1, vg.OUTCOME_STEPS + 2):
            gate.observe_reward(BIG if step == n else STEP_COST)
        want = model.outcome("hyperbolic", np.array([BIG]), np.array([n]), np.array([True]), K)[0]
        assert gate.episodes[0]["outcome"] == pytest.approx(want, rel=1e-12)


class TestBinaryIsUnchanged:
    def test_binary_is_the_default(self):
        gate = vg.ValueGate(seed=0)
        assert gate.outcome_rule == "binary"

    @pytest.mark.parametrize("size", [1.0, DECOY_FOOD, BIG])
    def test_the_binary_outcome_is_one_for_a_food_of_any_size_at_any_delay(self, size):
        gate = vg.ValueGate(seed=0)
        assert feed(gate, size, 43)["outcome"] == 1.0

    def test_the_binary_record_has_the_keys_it_had_in_the_same_order(self):
        gate = vg.ValueGate(seed=0)
        assert list(feed(gate, 1.0, 4)) == BINARY_RECORD_KEYS

    def test_a_delay_rate_does_not_reach_the_binary_outcome(self):
        gate = vg.ValueGate(seed=0, outcome_rule="binary", delay_rate=K)
        assert feed(gate, BIG, 43)["outcome"] == 1.0


class TestGateRules:
    def test_an_unknown_outcome_rule_is_refused(self):
        with pytest.raises(ValueError, match="outcome_rule"):
            vg.ValueGate(seed=0, outcome_rule="exponential")

    def test_a_delay_rate_below_zero_is_refused(self):
        with pytest.raises(ValueError, match="delay_rate"):
            vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=-0.1)


def wired(**gate_options):
    """The gate wired as train_rlhf wires it, on the two pulsing blocks of the rhythm binding tests."""
    import torch
    from models.core.global_workspace import GlobalWorkspace
    from models.core.sensory_tectum import SensoryTectum
    from scripts.training import train_rlhf
    from tests.test_rhythm_binding import FakeSpecialist
    tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
    spec = FakeSpecialist()
    train_rlhf._wire_map_binding(tectum, workspace, None, "rhythm", auditory_specialist=spec, seed=0,
                                 bands_to_bind=2)

    class Policy:
        def select_action(self, state, emotion_arousal=0.5, rpe_cache=0.0, self_vector=None):
            return np.array([0.6, -0.4], dtype=np.float32), 0.1

        def step(self, **kwargs):
            return {}

    policy = Policy()
    train_rlhf._wire_value_gate(policy, tectum, seed=0, log_dir=None, idle="still", **gate_options)
    return tectum, spec, policy, torch


def run_until_food(tectum, spec, policy, torch, food: float, steps_after_choice: int) -> None:
    """Steps of the training loop's calls; the food comes `steps_after_choice` steps after the choice."""
    from tests.test_rhythm_binding import pulse_bands, pulse_frames
    since = 0
    for t in range(80):
        if t > 0:
            spec.hear(pulse_bands(t))
        tectum(pulse_frames(t), torch.zeros(1, tectum.feature_dim, 2))
        policy.select_action("broadcast")
        since += 1 if tectum.value_gate.choice is not None else 0
        policy.step(raw_reward=food if since == steps_after_choice else STEP_COST)
        if since == steps_after_choice:
            return
    raise AssertionError("the gate made no choice in 80 steps")


class TestWiring:
    @pytest.mark.parametrize("n", [1, 5])
    def test_the_wired_gate_gives_the_hyperbolic_outcome_of_the_food_that_the_loop_passes(self, n):
        tectum, spec, policy, torch = wired(outcome_rule="hyperbolic", delay_rate=K)
        run_until_food(tectum, spec, policy, torch, BIG, n)
        gate = tectum.value_gate
        assert gate.episodes[0]["steps"] == n
        assert gate.episodes[0]["outcome"] == pytest.approx(hyperbolic(BIG, n), rel=1e-12)
        assert gate.values[gate.choice] == pytest.approx(0.3 * hyperbolic(BIG, n), rel=1e-12)

    def test_the_wired_gate_is_binary_without_the_option(self):
        tectum, spec, policy, torch = wired()
        run_until_food(tectum, spec, policy, torch, BIG, 5)
        assert tectum.value_gate.episodes[0]["outcome"] == 1.0

    def test_the_binary_run_record_has_the_keys_it_had(self):
        from scripts.training import train_rlhf
        gate = vg.ValueGate(seed=0)
        feed(gate, 1.0, 4)
        record = train_rlhf._value_gate_record(gate, learning_rate=0.3)
        assert list(record) == ["learning_rate", "episodes", "values"]
        assert record["values"] == {"18": pytest.approx(0.3)}

    def test_the_hyperbolic_run_record_names_the_rule_and_the_rate(self):
        from scripts.training import train_rlhf
        gate = vg.ValueGate(seed=0, outcome_rule="hyperbolic", delay_rate=K)
        feed(gate, BIG, 43)
        record = train_rlhf._value_gate_record(gate, learning_rate=0.3)
        assert record["outcome_rule"] == "hyperbolic" and record["delay_rate"] == K


def parser() -> argparse.ArgumentParser:
    from tests.test_value_gate import TestWiring as ValueGateWiring
    return ValueGateWiring.parser()


GATE_ROOM = ["--dark-room-audio", "binaural", "--dark-room-audio-channels", "4", "--dark-room-decoy-sound",
             "--dark-room-decoy-visible"]
ROOM_FLAGS = ["--dark-room-forced", "6", "--dark-room-food-sizes", "32,3", "--dark-room-food-waits", "40,0",
              "--dark-room-food-change", "30", "--dark-room-food-sizes-after", "4,3"]


def checked(argv):
    """The arguments after the check that train_rlhf makes on the new flags."""
    from scripts.training import train_rlhf
    p = parser()
    args = p.parse_args(argv)
    train_rlhf._check_competing_goals_flags(p, args)
    return args


class TestFlags:
    def test_every_flag_is_off_by_default(self):
        args = parser().parse_args([])
        assert args.value_outcome == "binary" and args.value_delay_rate is None
        assert args.dark_room_food_sizes is None and args.dark_room_food_waits is None
        assert args.dark_room_food_change is None and args.dark_room_food_sizes_after is None
        assert args.dark_room_forced is None

    def test_with_every_flag_off_the_room_gets_no_new_option(self):
        from scripts.training import train_rlhf
        assert train_rlhf._room_food_options(checked([])) == {
            "food_sizes": None, "food_waits": None, "food_change": None, "food_sizes_after": None, "forced": None}

    def test_the_flags_of_the_gate_reach_the_room_as_numbers(self):
        from scripts.training import train_rlhf
        assert train_rlhf._room_food_options(checked(GATE_ROOM + ROOM_FLAGS)) == {
            "food_sizes": (32.0, 3.0), "food_waits": (40, 0), "food_change": 30, "food_sizes_after": (4.0, 3.0),
            "forced": 6}

    def test_the_room_that_the_training_script_builds_has_the_options_of_the_flags(self):
        from scripts.training import train_rlhf
        np.random.seed(5)
        room = train_rlhf._build_dark_room(checked(GATE_ROOM + ROOM_FLAGS), None)
        assert room.food_sizes == (32.0, 3.0) and room.food_waits == (40, 0)
        assert room.food_change == 30 and room.food_sizes_after == (4.0, 3.0) and room.forced == 6
        room.seed_decoy(5)
        room.reset()
        room.agent_pos = room.light_pos.astype(np.float32).copy()
        facts = room.step(np.zeros(2, dtype=np.float32))[4]
        assert facts["_truth_present"] == "light" and facts["_truth_food_sizes"] == [32.0, 3.0]

    def test_the_room_that_the_training_script_builds_with_every_flag_off_has_no_option(self):
        from scripts.training import train_rlhf
        np.random.seed(5)
        room = train_rlhf._build_dark_room(checked(GATE_ROOM), None)
        assert room._food_options is False and room.absent_object is None
        assert (room.food_sizes, room.food_waits, room.food_change, room.food_sizes_after, room.forced) == (None,) * 5

    def test_with_every_flag_off_the_run_facts_get_no_new_key(self):
        from scripts.training import train_rlhf
        assert train_rlhf._competing_goals_facts(checked([])) == {}

    def test_the_run_facts_hold_each_flag_that_is_given(self):
        from scripts.training import train_rlhf
        argv = GATE_ROOM + ROOM_FLAGS + ["--tectal-orienting", "value", "--value-outcome", "hyperbolic",
                                         "--value-delay-rate", "0.1"]
        assert train_rlhf._competing_goals_facts(checked(argv)) == {
            "dark_room_food_sizes": (32.0, 3.0), "dark_room_food_waits": (40, 0), "dark_room_food_change": 30,
            "dark_room_food_sizes_after": (4.0, 3.0), "dark_room_forced": 6, "value_outcome": "hyperbolic",
            "value_delay_rate": 0.1}

    def test_the_help_of_each_flag_names_the_rules_document(self):
        actions = parser()._actions
        for flag in ("--dark-room-food-sizes", "--dark-room-food-waits", "--dark-room-food-change",
                     "--dark-room-forced", "--value-outcome"):
            text = next(a.help for a in actions if flag in a.option_strings)
            assert "2026_11_08_competing_goals_gate" in text, flag

    @pytest.mark.parametrize("text", ["32", "32,3,1", "a,b", ""])
    def test_a_pair_is_two_numbers(self, text):
        from scripts.training import train_rlhf
        with pytest.raises(ValueError):
            train_rlhf.parse_pair(text, float)

    def test_a_pair_of_waits_is_two_whole_numbers(self):
        from scripts.training import train_rlhf
        assert train_rlhf.parse_pair("40,0", int) == (40, 0)
        with pytest.raises(ValueError):
            train_rlhf.parse_pair("1.5,0", int)

    @pytest.mark.parametrize("flags", [["--dark-room-food-sizes", "32,3"], ["--dark-room-food-waits", "40,0"],
                                       ["--dark-room-forced", "6"],
                                       ["--dark-room-food-change", "30", "--dark-room-food-sizes-after", "4,3"]])
    def test_each_room_flag_needs_the_visible_decoy_and_refuses_the_food_switch(self, flags):
        with pytest.raises(SystemExit):
            checked(flags)
        with pytest.raises(SystemExit):
            checked(GATE_ROOM + flags + ["--dark-room-food-switch", "29"])
        checked(GATE_ROOM + flags)

    def test_the_change_and_the_sizes_after_it_need_each_other(self):
        with pytest.raises(SystemExit):
            checked(GATE_ROOM + ["--dark-room-food-change", "30"])
        with pytest.raises(SystemExit):
            checked(GATE_ROOM + ["--dark-room-food-sizes-after", "4,3"])

    def test_the_hyperbolic_outcome_needs_the_value_gate_and_a_delay_rate(self):
        with pytest.raises(SystemExit):
            checked(["--value-outcome", "hyperbolic", "--value-delay-rate", "0.1"])
        with pytest.raises(SystemExit):
            checked(["--tectal-orienting", "value", "--value-outcome", "hyperbolic"])
        args = checked(["--tectal-orienting", "value", "--value-outcome", "hyperbolic", "--value-delay-rate", "0.1"])
        assert args.value_delay_rate == pytest.approx(0.1)

    def test_a_delay_rate_needs_the_hyperbolic_outcome(self):
        with pytest.raises(SystemExit):
            checked(["--tectal-orienting", "value", "--value-delay-rate", "0.1"])
