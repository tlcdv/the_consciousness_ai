"""The salience floor of the value gate and the stop on a value that is not finite.

`docs/decisions/2026_11_08_competing_goals_gate.md`, owner decisions of 2026-11-14. With the floor,
the gate ignores a candidate whose salience at the choice is below it. With `--stop-on-non-finite`
a run stops when the reward, a loss of the policy update or a gradient at an optimizer step is not
finite. Both are off by default, and expected values are computed in each test.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import value_gate as vg
from tests.test_competing_goals_gate import BINARY_RECORD_KEYS, STEP_COST, checked, feed, parser, wired

FLOOR = 0.15
TONE, SMALL = 23, 0


class TestFloor:
    def test_no_floor_is_the_default_and_the_record_has_the_keys_it_had(self):
        gate = vg.ValueGate(seed=0)
        assert gate.salience_floor is None
        assert list(feed(gate, 1.0, 4)) == BINARY_RECORD_KEYS

    @pytest.mark.parametrize("seed", range(40))
    def test_a_candidate_below_the_floor_is_never_chosen(self, seed):
        """Without the floor the small band of check 2 was chosen with a chance of about 0.02."""
        gate = vg.ValueGate(seed=seed, salience_floor=FLOOR)
        gate.values = {SMALL: 5.0}                              # even a high value does not bring it back
        assert gate.choose({TONE: 1.0, SMALL: 0.0677}) == TONE

    def test_the_record_holds_the_kept_candidates_and_the_removed_ones(self):
        gate = vg.ValueGate(seed=0, salience_floor=FLOOR)
        gate.choose({TONE: 1.0, SMALL: 0.0677})
        gate.observe_reward(1.0)
        record = gate.episodes[0]
        assert record["salience"] == {str(TONE): 1.0} and record["priorities"] == {str(TONE): 1.0}
        assert record["removed"] == {str(SMALL): 0.0677}

    def test_two_tones_above_the_floor_are_both_kept_and_nothing_is_removed(self):
        gate = vg.ValueGate(seed=0, salience_floor=FLOOR)
        gate.choose({18: 0.2832, 23: 1.0})
        gate.observe_reward(1.0)
        assert gate.episodes[0]["salience"] == {"18": 0.2832, "23": 1.0} and gate.episodes[0]["removed"] == {}

    @pytest.mark.parametrize("seed", range(25))
    def test_with_two_tones_above_the_floor_the_choices_are_those_of_the_gate_without_it(self, seed):
        plain, floored = vg.ValueGate(seed=seed), vg.ValueGate(seed=seed, salience_floor=FLOOR)
        generator = np.random.default_rng(seed)
        for _ in range(30):
            salience = {18: float(generator.uniform(FLOOR, 1.0)), 23: 1.0}
            food = 1.0 if generator.random() < 0.5 else STEP_COST
            for gate in (plain, floored):
                gate.choose(salience)
                gate.observe_reward(food)
                gate.new_episode()
        assert [e["choice"] for e in plain.episodes] == [e["choice"] for e in floored.episodes]
        assert plain.values == floored.values

    def test_a_candidate_at_the_floor_is_kept(self):
        gate = vg.ValueGate(seed=0, salience_floor=FLOOR)
        gate.choose({18: FLOOR, 23: 1.0})
        gate.observe_reward(1.0)
        assert set(gate.episodes[0]["salience"]) == {"18", "23"}

    def test_with_every_candidate_below_the_floor_the_gate_makes_no_choice_and_draws_nothing(self):
        gate = vg.ValueGate(seed=0, salience_floor=FLOOR)
        state = gate.rng.bit_generator.state
        assert gate.choose({SMALL: 0.05}) is None
        assert gate.choice is None and gate.rng.bit_generator.state == state
        gate.observe_reward(1.0)                                # no open choice: the reward is ignored
        assert gate.episodes == [] and gate.values == {}
        assert gate.choose({TONE: 1.0, SMALL: 0.05}) == TONE   # a later step of the episode can choose

    def test_a_floor_below_zero_is_refused(self):
        with pytest.raises(ValueError, match="salience_floor"):
            vg.ValueGate(seed=0, salience_floor=-0.1)


class TestFloorWiring:
    def test_the_wired_gate_has_the_floor_and_the_run_record_names_it(self):
        from scripts.training import train_rlhf
        tectum, _, _, _ = wired(salience_floor=FLOOR)
        gate = tectum.value_gate
        assert gate.salience_floor == FLOOR
        assert train_rlhf._value_gate_record(gate, learning_rate=0.3)["salience_floor"] == FLOOR

    def test_without_the_floor_the_run_record_has_the_keys_it_had(self):
        from scripts.training import train_rlhf
        tectum, _, _, _ = wired()
        assert list(train_rlhf._value_gate_record(tectum.value_gate, learning_rate=0.3)) == [
            "learning_rate", "episodes", "values"]

    def test_with_every_candidate_below_the_floor_the_wired_agent_stays_still(self):
        """A floor above every salience: the real binder gives candidates and the gate takes none."""
        from tests.test_rhythm_binding import pulse_bands, pulse_frames
        tectum, spec, policy, torch_module = wired(salience_floor=2.0)
        actions = []
        for t in range(20):
            if t > 0:
                spec.hear(pulse_bands(t))
            tectum(pulse_frames(t), torch_module.zeros(1, tectum.feature_dim, 2))
            actions.append(policy.select_action("broadcast")[0])
            policy.step(raw_reward=1.0)
        assert tectum.value_gate.choice is None and tectum.value_gate.episodes == []
        assert all(np.allclose(action, 0.0) for action in actions)

    def test_the_flag_is_off_by_default_and_needs_the_value_gate(self):
        assert parser().parse_args([]).value_salience_floor is None
        with pytest.raises(SystemExit):
            checked(["--value-salience-floor", "0.15"])
        assert checked(["--tectal-orienting", "value", "--value-salience-floor", "0.15"]).value_salience_floor == FLOOR

    def test_the_run_facts_hold_the_floor_only_when_it_is_given(self):
        from scripts.training import train_rlhf
        assert train_rlhf._competing_goals_facts(checked(["--tectal-orienting", "value"])) == {}
        facts = train_rlhf._competing_goals_facts(checked(["--tectal-orienting", "value", "--value-salience-floor", "0.15"]))
        assert facts == {"value_salience_floor": FLOOR}

    def test_the_help_names_the_rules_document(self):
        text = next(a.help for a in parser()._actions if "--value-salience-floor" in a.option_strings)
        assert "2026_11_08_competing_goals_gate" in text and "below" in text


class Core:
    """A stand-in with the two calls of the action core that the stop wraps."""

    def __init__(self, losses):
        self.losses, self.rewards = losses, []

    def update_policy(self):
        return dict(self.losses)

    def step(self, *args, **kwargs):
        self.rewards.append(kwargs["raw_reward"] if "raw_reward" in kwargs else args[2])
        return {"stepped": True}


def linear_with_optimizer():
    torch.manual_seed(0)
    layer = torch.nn.Linear(3, 1)
    return layer, torch.optim.Adam(layer.parameters(), lr=0.1)


class TestStopOnNonFinite:
    def test_finite_values_pass_and_the_calls_return_what_they_returned(self):
        from scripts.training import train_rlhf
        core = Core({"total_loss": 0.25, "value_loss": 3.0})
        train_rlhf._wire_finite_stop(core, [])
        assert core.update_policy() == {"total_loss": 0.25, "value_loss": 3.0}
        assert core.step("broadcast", "action", 32.0) == {"stepped": True}
        assert core.step(raw_reward=-0.01) == {"stepped": True} and core.rewards == [32.0, -0.01]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_a_loss_of_the_policy_update_that_is_not_finite_stops_the_run(self, bad):
        from scripts.training import train_rlhf
        core = Core({"total_loss": 0.25, "value_loss": bad})
        train_rlhf._wire_finite_stop(core, [])
        with pytest.raises(FloatingPointError, match="value_loss"):
            core.update_policy()

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_reward_that_is_not_finite_stops_the_run_before_the_core_reads_it(self, bad):
        from scripts.training import train_rlhf
        core = Core({})
        train_rlhf._wire_finite_stop(core, [])
        with pytest.raises(FloatingPointError, match="reward"):
            core.step(raw_reward=bad)
        with pytest.raises(FloatingPointError, match="reward"):
            core.step("broadcast", "action", torch.tensor(bad))
        assert core.rewards == []

    def test_a_gradient_that_is_not_finite_stops_the_run_before_the_optimizer_changes_a_weight(self):
        from scripts.training import train_rlhf
        layer, optimizer = linear_with_optimizer()
        train_rlhf._wire_finite_stop(Core({}), [optimizer])
        before = [p.detach().clone() for p in layer.parameters()]
        loss = layer(torch.ones(1, 3)).sum() * float("nan")
        loss.backward()
        with pytest.raises(FloatingPointError, match="gradient"):
            optimizer.step()
        assert all(torch.equal(a, b) for a, b in zip(before, layer.parameters()))

    def test_with_finite_gradients_the_weights_are_those_of_an_optimizer_without_the_stop(self):
        from scripts.training import train_rlhf
        plain_layer, plain = linear_with_optimizer()
        layer, optimizer = linear_with_optimizer()
        train_rlhf._wire_finite_stop(Core({}), [optimizer, None])
        for model, opt in ((plain_layer, plain), (layer, optimizer)):
            for _ in range(3):
                opt.zero_grad()
                (model(torch.ones(1, 3)).sum() - 32.0).pow(2).backward()
                opt.step()
        assert all(torch.equal(a, b) for a, b in zip(plain_layer.parameters(), layer.parameters()))

    def test_the_closed_form_of_the_check(self):
        from scripts.training import train_rlhf
        train_rlhf._require_finite({"a": 0.0, "b": -32.0, "c": torch.tensor(1e30)}, "here")
        for bad in (math.nan, math.inf, torch.tensor(float("nan"))):
            with pytest.raises(FloatingPointError, match="b .*here"):
                train_rlhf._require_finite({"a": 0.0, "b": bad}, "here")

    def test_the_flag_is_off_by_default_and_its_help_names_what_it_reads(self):
        assert parser().parse_args([]).stop_on_non_finite is False
        text = next(a.help for a in parser()._actions if "--stop-on-non-finite" in a.option_strings)
        assert "reward" in text and "policy update" in text and "gradient" in text
