"""The unit tests of groups A and B of step P2 of design P: the choice head and check 2 on built data
(group A), and the wiring of the choice mode in the training script (group B, at the end).

Rules: docs/decisions/2026_11_22_workspace_choice_mode.md, with its sections "Changes after the
independent review" and "Owner decision on the scaling". The numbers A1 to A26 are those of the rules.
These tests use built data and rows made for the test only. They read no saved run.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import choice_head
from models.core.choice_head import ChoiceHead, exploring_choices, normalize_rows
from scripts.analysis import check_choice_head as check
from scripts.analysis.check_choice_read import ControlMismatch, standardize


def a_head(seed: int = 0, exploration=(0, 0, 0, 2), share: float = choice_head.EXPLORATION_SHARE) -> ChoiceHead:
    return ChoiceHead(seed, list(exploration), share=share)


def random_rows(count: int, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, choice_head.INPUT_DIM))


class UpdateSpy:
    """Records the rows and the step count of each call of ChoiceHead.update, then calls it."""

    def __init__(self, monkeypatch):
        self.calls, original = [], ChoiceHead.update

        def spy(head, rows, chosen, outcome, steps=1):
            self.calls.append((np.array(rows, dtype=np.float64, copy=True).reshape(-1, choice_head.INPUT_DIM),
                               steps))
            return original(head, rows, chosen, outcome, steps)
        monkeypatch.setattr(ChoiceHead, "update", spy)


# The head.

def test_a1_the_head_has_the_layers_and_the_initial_weights_of_build_head():
    for seed in (0, 1, 102, 905):
        body = a_head(seed).body
        reference = check.build_head("plain", 256, seed)["net"]
        assert [type(layer).__name__ for layer in body] == ["Linear", "Tanh", "Linear", "Tanh", "Linear"]
        assert [tuple(p.shape) for p in body.parameters()] == [(128, 256), (128,), (128, 128), (128,), (2, 128), (2,)]
        for mine, theirs in zip(body.state_dict().values(), reference.state_dict().values()):
            assert torch.equal(mine, theirs)


def test_a2_to_build_the_head_leaves_each_random_state_and_seeds_no_generator(monkeypatch):
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    def forbidden(*args, **kwargs):
        raise AssertionError("the live head must not seed a global generator")
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(check, "build_head", forbidden)
    a_head(7)
    assert torch.equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    if torch.cuda.is_available():
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_states))


def test_a3_with_no_exploration_the_choice_is_the_larger_output_and_slot_0_on_a_tie():
    outputs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-2.0, -3.0]], dtype=np.float32)
    chosen, explored = exploring_choices(outputs, np.random.default_rng(3), share=0.0)
    assert chosen.tolist() == [0, 1, 0, 0]
    assert not explored.any()


def test_a4_the_share_of_exploration_draws_and_their_slots():
    outputs = np.random.default_rng(5).normal(size=(10000, 2)).astype(np.float32)
    chosen, explored = exploring_choices(outputs, np.random.default_rng(6), share=0.25)
    assert 0.235 <= explored.mean() <= 0.265
    assert 0.47 <= (chosen[explored] == 0).mean() <= 0.53


def test_a5_each_choice_takes_random_then_integers_whatever_the_outputs():
    first, second = np.random.default_rng(21), np.random.default_rng(21)
    reference = np.random.default_rng(21)
    chosen, explored = exploring_choices(np.zeros((7, 2), dtype=np.float32), first, share=0.25)
    exploring_choices(np.random.default_rng(8).normal(size=(7, 2)).astype(np.float32), second, share=0.25)
    drawn = [(reference.random(), int(reference.integers(0, 2))) for _ in range(7)]
    assert first.bit_generator.state == second.bit_generator.state == reference.bit_generator.state
    assert explored.tolist() == [uniform < 0.25 for uniform, _ in drawn]
    assert chosen.tolist() == [slot if uniform < 0.25 else 0 for uniform, slot in drawn]


def test_a6_the_choices_of_a_set_do_not_depend_on_its_targets():
    rows = random_rows(24)
    picks = []
    for target in (np.zeros(24, dtype=int), np.arange(24) % 2):
        head, window = a_head(3), check.ChoiceWindow()
        picks.append(check.learn_from_set(head, rows, target, False, window, steps=1))
    assert np.array_equal(picks[0], picks[1])


def test_a7_the_loss_reads_the_output_of_the_chosen_slot_only():
    head = a_head(4)
    last = head.body[4]
    weight, bias = last.weight.detach().clone(), last.bias.detach().clone()
    head.update(random_rows(1), np.array([0]), np.array([1.0]), steps=1)
    assert torch.equal(last.weight[1], weight[1]) and torch.equal(last.bias[1], bias[1])
    assert not torch.equal(last.weight[0], weight[0])


def test_a8_the_live_rule_makes_5_updates_on_the_row_of_each_choice_only(monkeypatch):
    spy = UpdateSpy(monkeypatch)
    case = {"part": "B", "pass": "first", "rows": 40, "data_seed": 2}
    check.part_b_training(case, 0, 1, episodes=3, checkpoints=(0, 3))
    episode_rows = check.part_b_rows(2, 1)["rows"]
    assert [steps for _, steps in spy.calls] == [5, 5, 5]
    for episode, (rows, _) in enumerate(spy.calls):
        assert rows.shape == (1, 256) and np.allclose(rows[0], episode_rows[episode])


# The rows and the seeds of check 2.

def test_a9_a_new_row_set_of_part_a_is_the_set_of_condition_c2():
    built = check.built_case(40, 1.0, "random", 3)
    split = check.fold_split(built, 2)
    rows, target = check.part_a_set(split, 3, 2, 7, check.built_noise(40, 3))
    reference = check.new_row_set(32, "random", 3, 2, 7)
    _, expected = standardize(split["raw_train_rows"], reference["rows"])
    assert np.array_equal(rows, expected) and np.array_equal(target, reference["target"])


def first_draws(seed) -> np.ndarray:
    return np.random.default_rng(seed).random(4)


def test_a10_the_streams_of_part_b_and_of_the_exploration_are_their_own():
    for data_seed, fold, initial_seed in itertools.product(range(10), range(5), range(3)):
        rows_stream = first_draws(check.part_b_row_seed(data_seed, fold))
        explore_stream = first_draws(check.exploration_seed(initial_seed, data_seed, fold))
        others = [first_draws(d) for d in range(10)] + [first_draws([data_seed, fold, u]) for u in (1, 2, 200)]
        assert not any(np.array_equal(rows_stream, other) for other in others + [explore_stream])
        assert not any(np.array_equal(explore_stream, other) for other in others)
    for data_seed, repeat in itertools.product(range(10), range(5)):
        check.part_b_rows(data_seed, repeat)            # K3: raises on a row of the data set
    for data_seed, fold, update in itertools.product(range(10), range(5), (1, 100, 200)):
        split = check.fold_split(check.built_case(30, 1.0, "random", data_seed), fold)
        check.part_a_set(split, data_seed, fold, update, check.built_noise(30, data_seed))


def test_a11_the_outcomes_of_the_two_passes():
    chosen, target = np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0])
    assert check.mode_outcomes(chosen, target, False).tolist() == [1.0, 0.0, 0.0, 1.0]
    assert check.mode_outcomes(chosen, target, True).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_a12_the_rows_of_each_training_and_their_means():
    assert check.mode_expected_counts() == (760, 11400)
    for case, length, checkpoints in ((check.mode_cases()[0], 2, (0, 1, 2)),
                                      ({"part": "O2", "pass": "second", "rows": 40, "data_seed": 1}, 2, (0, 2))):
        training_rows, mean_rows = check.mode_case_rows(case, length, checkpoints)
        assert len(training_rows) == 15 * len(checkpoints) and len(mean_rows) == len(checkpoints)
        keys = {(r["initial_seed"], r["fold"], r["checkpoint"]) for r in training_rows}
        assert keys == set(itertools.product(range(3), range(5), checkpoints))
        for mean_row in mean_rows:
            members = [r["score"] for r in training_rows if r["checkpoint"] == mean_row["checkpoint"]]
            assert mean_row["score"] == pytest.approx(np.mean(members))


# The label code on rows made for the test.

def made_mean_rows(score_of):
    """Each mean row of a complete job. score_of(part, pass, rows, data_seed, checkpoint) gives the score."""
    made = []
    for case in check.mode_cases():
        checkpoints = check.MODE_A_CHECKPOINTS if case["part"] == "A" else check.MODE_B_CHECKPOINTS
        for checkpoint in checkpoints:
            row = dict(case, checkpoint=checkpoint, **{name: 0.25 for name in check.MODE_MEASURES})
            row["score"] = score_of(case["part"], case["pass"], case["rows"], case["data_seed"], checkpoint)
            made.append(row)
    return made


def part_a_scores(first_30, first_40, second=0.5):
    def score_of(part, pass_, rows, data_seed, checkpoint):
        if part != "A":
            return 0.5
        return second if pass_ == "second" else {30: first_30, 40: first_40}[rows]
    return made_mean_rows(score_of)


def part_b_scores(first_40, first_60, second=0.5):
    def score_of(part, pass_, rows, data_seed, checkpoint):
        if part == "A" or pass_ == "second":
            return second if part != "A" else 0.5
        return {40: first_40, 60: first_60}.get(checkpoint, 0.5)
    return made_mean_rows(score_of)


def labels(mean_rows):
    return check.mode_labels_of(mean_rows)


def test_a13_part_a_holds():
    made = labels(part_a_scores(0.9, 0.9))
    assert made["A"] == "HOLDS"


def test_a14_part_a_fails():
    assert labels(part_a_scores(0.7, 0.6))["A"] == "FAILS"


def test_a15_part_a_not_decided_from_a_split_of_the_row_counts():
    assert labels(part_a_scores(0.9, 0.7))["A"] == "NOT DECIDED"


def test_a16_part_a_not_decided_from_a_high_score_with_d_below_the_limit():
    assert labels(part_a_scores(0.85, 0.85, second=0.84))["A"] == "NOT DECIDED"


def test_a17_part_b_holds():
    assert labels(part_b_scores(0.9, 0.9))["B"] == "HOLDS"


def test_a18_part_b_partly():
    assert labels(part_b_scores(0.7, 0.9))["B"] == "PARTLY"


def test_a19_part_b_fails():
    assert labels(part_b_scores(0.7, 0.7))["B"] == "FAILS"


def test_a20_part_b_not_decided_from_a_high_score_after_40_and_a_low_one_after_60():
    assert labels(part_b_scores(0.9, 0.7))["B"] == "NOT DECIDED"


def test_a21_void_by_k2_when_a_recorded_count_differs_from_the_count_drawn_again():
    seed = check.exploration_seed(1, 4, 2)
    counts = check.redrawn_explored(seed, [0, 32, 608])
    made = [{"part": "A", "pass": "first", "rows": 40, "data_seed": 4, "initial_seed": 1, "fold": 2,
             "checkpoint": checkpoint, "count_choices": choices, "count_explored": explored}
            for checkpoint, choices, explored in zip((0, 1, 20), (0, 32, 608), counts)]
    assert check.check_exploration_counts(made)["mismatches"] == 0
    made[2] = dict(made[2], count_explored=made[2]["count_explored"] + 1)
    assert check.check_exploration_counts(made)["mismatches"] == 1
    assert check.mode_report(part_a_scores(0.9, 0.9), made)["verdict"] == "VOID"


# Part B, the arm of O2 and the normalization.

def test_a22_part_b_scores_on_the_40_rows_of_the_data_set_of_40_rows_with_no_standardization():
    for data_seed in range(10):
        score_set = check.mode_score_set(data_seed)
        reference = check.built_case(40, 1.0, "random", data_seed)
        assert np.array_equal(score_set["rows"], reference["rows"])
        assert np.array_equal(score_set["target"], reference["target"])
        for repeat in range(5):
            episode_rows = check.part_b_rows(data_seed, repeat)["rows"]
            assert check.rows_in_common(episode_rows, score_set["rows"]) == 0


def test_a23_the_arm_of_o2_makes_one_update_on_all_rows_so_far(monkeypatch):
    spy = UpdateSpy(monkeypatch)
    case = {"part": "O2", "pass": "first", "rows": 40, "data_seed": 5}
    check.part_b_training(case, 2, 0, episodes=4, checkpoints=(0, 4))
    episode_rows = check.part_b_rows(5, 0)["rows"]
    assert [steps for _, steps in spy.calls] == [1, 1, 1, 1]
    for episode, (rows, _) in enumerate(spy.calls):
        assert rows.shape == (episode + 1, 256) and np.allclose(rows, episode_rows[:episode + 1])


def test_a24_the_output_is_the_body_applied_to_the_normalized_row():
    head, x = a_head(9), torch.tensor(random_rows(3) * 0.03 + 0.01, dtype=torch.float32)
    mean = x.mean(dim=1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=1, keepdim=True)
    by_hand = (x - mean) / torch.sqrt(variance + 1e-5)
    with torch.no_grad():
        assert torch.allclose(normalize_rows(x), by_hand, atol=1e-6)
        assert torch.allclose(head.forward_rows(x), head.body(by_hand), atol=1e-6)
        assert torch.equal(head.forward_rows(x), head.body(F.layer_norm(x, (256,), eps=1e-5)))


def test_a25_a_row_of_zeros_stays_zero_and_gives_a_finite_output():
    zeros = torch.zeros(1, 256)
    assert torch.equal(normalize_rows(zeros), zeros)
    outputs = a_head(10).outputs(np.zeros((1, 256)))
    assert np.isfinite(outputs).all()


def test_a26_check_2_passes_its_rows_through_the_same_normalization(monkeypatch):
    seen, original = [], choice_head.normalize_rows

    def spy(rows):
        seen.append(tuple(rows.shape))
        return original(rows)
    monkeypatch.setattr(choice_head, "normalize_rows", spy)
    check.part_a_training(check.mode_cases()[0], 0, 0, updates=1, checkpoints=(0, 1))
    check.part_b_training({"part": "B", "pass": "first", "rows": 40, "data_seed": 0}, 0, 0, episodes=1,
                          checkpoints=(0, 1))
    assert (24, 256) in seen and (6, 256) in seen        # a set of part A at 30 rows, and its test rows
    assert (1, 256) in seen and (40, 256) in seen        # a row of part B, and its 40 score rows


# Group B: the wiring of the choice mode in scripts/training/train_rlhf.py (work 4). The numbers B1 to
# B12 are those of the rules, with O5 in B1 and N4 in B10. B13 is not written (owner decision before work
# 4). These tests wire a tectum and a policy made for the test, as tests/test_value_gate.py does. They
# run no training and read no saved run.
# After the independent review of the wiring, B14 is O1 and B12 also has O2 (the rules, section "Changes
# after the independent review of the wiring").

import argparse
import ast
import json
import random



def train_rlhf_parser():
    """The parser of train_rlhf.main, captured before it parses."""
    from unittest import mock
    from scripts.training import train_rlhf
    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise RuntimeError("captured")
    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(RuntimeError):
            train_rlhf.main()
    return captured["parser"]


VALID_PLAIN = ["--choice-head", "plain", "--tectal-orienting", "value", "--map-binding", "rhythm"]


def wired_mode(mode: str, seed: int = 0):
    """A tectum with the rhythm binding of two bands, a policy made for the test, and the wiring of
    `mode` ("plain" for the choice head, "value" for the value gate)."""
    from models.core.global_workspace import GlobalWorkspace
    from models.core.sensory_tectum import SensoryTectum
    from scripts.training import train_rlhf
    from tests.test_rhythm_binding import FakeSpecialist
    tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
    spec = FakeSpecialist()
    train_rlhf._wire_map_binding(tectum, workspace, None, "rhythm", auditory_specialist=spec, seed=0,
                                 bands_to_bind=2)
    calls = []

    class Policy:
        def select_action(self, state, emotion_arousal=0.5, rpe_cache=0.0, self_vector=None):
            calls.append("select")
            return np.array([0.6, -0.4], dtype=np.float32), 0.1

        def step(self, **kwargs):
            calls.append(("step", kwargs["raw_reward"]))
            return {}

    policy = Policy()
    if mode == "plain":
        live = train_rlhf._wire_choice_head(policy, tectum, seed, None, idle="still")
    else:
        train_rlhf._wire_value_gate(policy, tectum, seed=seed, log_dir=None, idle="still")
        live = None
    return tectum, spec, policy, calls, live


def run_steps(tectum, spec, policy, count, rewards=None, states=None):
    """`count` steps of the room made for the test. Step 0 of a call hears no new sound."""
    from tests.test_rhythm_binding import pulse_bands, pulse_frames
    actions = []
    for t in range(count):
        if t > 0:
            spec.hear(pulse_bands(t))
        tectum(pulse_frames(t), torch.zeros(1, tectum.feature_dim, 2))
        state = states[t] if states is not None else torch.zeros(1, choice_head.INPUT_DIM)
        action, _ = policy.select_action(state)
        policy.step(raw_reward=(rewards or {}).get(t, -0.01))
        actions.append(action)
    return actions


def step_states(count, seed=31):
    rows = np.random.default_rng(seed).normal(size=(count, choice_head.INPUT_DIM)).astype(np.float32)
    return [torch.as_tensor(row).unsqueeze(0) for row in rows]


def first_step_with_candidates(map_history, start=0):
    return next(t for t, record in enumerate(map_history[start:])
                if any(cells for cells in record.get("groups", {}).values()))


def test_b1_the_flag_its_needs_and_its_refusals():
    from scripts.training import train_rlhf
    parser = train_rlhf_parser()
    assert parser.parse_args([]).choice_head == "off"
    for value in ("both", "hopeful"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--choice-head", value])
    train_rlhf._check_choice_head_flags(parser, parser.parse_args(VALID_PLAIN))
    train_rlhf._check_choice_head_flags(parser, parser.parse_args(VALID_PLAIN[:4] + ["--map-binding",
                                                                                     "rhythm-scrambled"]))
    train_rlhf._check_choice_head_flags(parser, parser.parse_args(VALID_PLAIN + ["--value-learning-rate", "0.3"]))
    train_rlhf._check_choice_head_flags(parser, parser.parse_args(["--value-learning-rate", "0.5"]))
    refused = (["--choice-head", "plain"],
               ["--choice-head", "plain", "--tectal-orienting", "group", "--map-binding", "rhythm"],
               ["--choice-head", "plain", "--tectal-orienting", "value", "--map-binding", "limit"],
               VALID_PLAIN + ["--policy-input", "tectum"],
               VALID_PLAIN + ["--value-outcome", "hyperbolic", "--value-delay-rate", "0.1"],
               VALID_PLAIN + ["--value-salience-floor", "0.1"],
               VALID_PLAIN + ["--value-learning-rate", "0.5"])          # O5
    for argv in refused:
        with pytest.raises(SystemExit):
            train_rlhf._check_choice_head_flags(parser, parser.parse_args(argv))


def test_b2_with_the_flag_off_main_wires_the_value_gate_as_before(monkeypatch):
    from scripts.training import train_rlhf
    parser = train_rlhf_parser()
    seen = []
    monkeypatch.setattr(train_rlhf, "_wire_value_gate", lambda *a, **k: seen.append(("gate", a, k)))
    monkeypatch.setattr(train_rlhf, "_wire_choice_head", lambda *a, **k: seen.append(("head", a, k)))
    core, tectum = object(), object()
    argv = ["--tectal-orienting", "value", "--map-binding", "rhythm", "--seed", "73", "--log-dir", "x",
            "--tectal-orienting-idle", "still"]
    assert train_rlhf._wire_orienting(parser.parse_args(argv), core, tectum) is None
    assert seen == [("gate", (core, tectum, 73, "x"),
                     {"idle": "still", "learning_rate": 0.3, "outcome_rule": "binary", "delay_rate": 0.0,
                      "salience_floor": None})]
    seen.clear()
    train_rlhf._wire_orienting(parser.parse_args(argv + ["--choice-head", "plain"]), core, tectum)
    assert [name for name, _, _ in seen] == ["head"]
    assert seen[0][1] == (core, tectum, 73, "x") and seen[0][2] == {"idle": "still"}
    off = train_rlhf.build_config(parser.parse_args([]))
    assert "choice_head" not in off and off == train_rlhf.build_config(parser.parse_args(["--choice-head", "off"]))
    assert train_rlhf._competing_goals_facts(parser.parse_args(["--choice-head", "off"])) == \
        train_rlhf._competing_goals_facts(parser.parse_args([]))


def test_b3_the_wrapper_calls_the_policy_then_the_head_then_the_motor_map():
    tectum, spec, policy, calls, live = wired_mode("plain")
    order = []
    choose, motor_step = live.choose, tectum.tectal_motor_map.step

    def spy_choose(*args, **kwargs):
        order.append(("head", len(calls)))
        return choose(*args, **kwargs)

    def spy_motor(*args, **kwargs):
        order.append(("motor", len(calls)))
        return motor_step(*args, **kwargs)
    live.choose, tectum.tectal_motor_map.step = spy_choose, spy_motor
    run_steps(tectum, spec, policy, 20)
    heads = [i for i, (name, _) in enumerate(order) if name == "head"]
    assert len(heads) == 1
    calls_before = order[heads[0]][1]
    assert calls[calls_before - 1] == "select"                         # the policy came first
    assert order[heads[0] + 1] == ("motor", calls_before)              # then the motor map, in the same step


def test_b4_the_head_waits_chooses_once_and_the_motor_map_follows_the_chosen_band():
    from tests.test_rhythm_binding import BLOCK_A, BLOCK_B
    from models.core.tectal_motor_map import cell_offsets
    tectum, spec, policy, calls, live = wired_mode("plain")
    actions = run_steps(tectum, spec, policy, 20)
    first = first_step_with_candidates(tectum.map_history)
    assert first > 0 and all(np.allclose(a, 0.0) for a in actions[:first])
    assert len(live.entries) == 1 and live.entries[0]["step"] == first
    assert live.choice in (18, 23) and live.entries[0]["choice"] == live.choice
    block = BLOCK_A if live.choice == 18 else BLOCK_B
    dx, dy = cell_offsets()
    centre = np.mean([[dx.flat[i], dy.flat[i]] for i in block], axis=0)
    target = tectum.tectal_motor_map.history[-1]["target"]
    assert target is not None and np.dot(target, centre) > 0


def test_b5_slot_0_is_the_louder_candidate_and_a_tie_puts_the_lower_band_first():
    assert choice_head.candidate_slots({18: 0.5, 23: 0.9}) == ([23, 18], False)
    assert choice_head.candidate_slots({23: 0.7, 18: 0.7}) == ([18, 23], True)
    assert choice_head.candidate_slots({23: 0.4}) == ([23], False)
    with pytest.raises(ValueError):
        choice_head.candidate_slots({1: 0.1, 2: 0.2, 3: 0.3})
    live = choice_head.LiveChoiceHead(5)
    live.choose({23: 0.7, 18: 0.7}, np.zeros(choice_head.INPUT_DIM, dtype=np.float32), step=3, episode=0)
    assert live.entries[0]["slots"] == [18, 23] and live.entries[0]["tie"] is True
    assert live.entries[0]["salience"] == {"23": 0.7, "18": 0.7}


def test_b6_with_one_candidate_there_is_no_draw_and_no_update():
    live = choice_head.LiveChoiceHead(5)
    state = live.head.rng.bit_generator.state
    weights = [p.detach().clone() for p in live.head.body.parameters()]
    assert live.choose({23: 0.4}, np.ones(choice_head.INPUT_DIM, dtype=np.float32), step=2, episode=0) == 23
    for _ in range(3):
        live.observe_reward(1.0)
    entry = live.entries[0]
    assert live.head.rng.bit_generator.state == state
    assert all(torch.equal(a, b) for a, b in zip(weights, live.head.body.parameters()))
    assert entry["one_candidate"] is True and entry["updates"] == 0 and entry["choice"] == 23
    assert entry["outputs"] is None and entry["explored"] is False and entry["outcome"] == 1.0


def test_b7_each_choice_closes_at_the_step_and_with_the_outcome_of_the_value_gate():
    from models.core.value_gate import ValueGate
    cases = {"food at once": [1.0], "no food in 60 steps": [-0.01] * 70,
             "a step without sound before the end": [-0.01] * 10 + ["silent"]}
    for name, rewards in cases.items():
        gate, live = ValueGate(1), choice_head.LiveChoiceHead(1)
        gate.choose({18: 0.9, 23: 0.4})
        live.choose({18: 0.9, 23: 0.4}, np.zeros(choice_head.INPUT_DIM, dtype=np.float32), step=0, episode=0)
        for reward in rewards:
            if reward == "silent":
                gate.new_episode()
                live.new_episode()
            else:
                gate.observe_reward(reward)
                live.observe_reward(reward)
        assert len(gate.episodes) == 1, name
        assert live.entries[0]["outcome"] == gate.episodes[0]["outcome"], name
        assert live.entries[0]["steps"] == gate.episodes[0]["steps"], name
        assert (live.choice is None) == (rewards[-1] == "silent"), name


def test_b8_after_a_closed_choice_with_two_candidates_the_head_makes_5_updates_on_that_row(monkeypatch):
    spy = UpdateSpy(monkeypatch)
    live = choice_head.LiveChoiceHead(5)
    row = random_rows(1, seed=9)[0].astype(np.float32)
    live.choose({18: 0.9, 23: 0.4}, row, step=0, episode=0)
    live.observe_reward(1.0)
    live.observe_reward(1.0)                                           # closed: no second update
    assert [steps for _, steps in spy.calls] == [5]
    assert spy.calls[0][0].shape == (1, 256) and np.allclose(spy.calls[0][0][0], row)
    step_counts = [state["step"] for state in live.head.optimizer.state.values()]
    assert step_counts and all(int(count) == 5 for count in step_counts)
    assert live.entries[0]["updates"] == 5 and live.entries[0]["loss_after"] is not None


def test_b9_the_head_reads_the_first_argument_of_select_action():
    tectum, spec, policy, calls, live = wired_mode("plain")
    states = step_states(20)
    run_steps(tectum, spec, policy, 20, states=states)
    step = live.entries[0]["step"]
    assert np.array_equal(live.rows[0], states[step].numpy().reshape(-1))
    assert live.rows[0].dtype == np.float32


class FakeRoom:
    def __init__(self):
        self.resets = []

    def reset(self, *args, **kwargs):
        self.resets.append(kwargs.get("seed"))
        return None, {}


def test_b10_the_record_files_their_keys_and_the_episode_of_each_choice(tmp_path):
    from scripts.training import train_rlhf
    for seed in (7, None):
        tectum, spec, policy, calls, live = wired_mode("plain")
        room = FakeRoom()
        train_rlhf.seed_environment(room, seed)                       # before the wrap, as in main
        train_rlhf._wire_choice_head_episodes(live, room, tectum)
        starts = []
        for episode in range(3):                                       # run_episode resets each episode
            room.reset()
            starts.append(len(tectum.map_history))
            run_steps(tectum, spec, policy, 20)
        assert [entry["episode"] for entry in live.entries] == [0, 1, 2], seed
        assert [entry["step"] for entry in live.entries] == \
            [first_step_with_candidates(tectum.map_history, start) for start in starts], seed
    out = tmp_path / "run"
    train_rlhf._write_choice_head_record(live, tectum.tectal_motor_map, str(out))
    names = sorted(p.name for p in out.iterdir())
    assert names == ["choice_head.json", "choice_head.pt", "choice_head_inputs.npy", "tectal_orienting.json"]
    record = json.loads((out / "choice_head.json").read_text(encoding="utf-8"))
    assert record["settings"] == {"head": "plain", "rule": "chosen", "share": 0.25, "updates_per_choice": 5,
                                  "learning_rate": 3e-4, "head_seed": 2000003, "exploration_seed": 3000005}
    keys = {"episode", "step", "salience", "slots", "tie", "one_candidate", "outputs", "greedy_slot", "explored",
            "chosen_slot", "choice", "outcome", "steps", "loss_before", "loss_after", "updates"}
    assert len(record["choices"]) == 3 and all(set(entry) == keys for entry in record["choices"])
    rows = np.load(out / "choice_head_inputs.npy")
    assert rows.shape == (3, 256) and rows.dtype == np.float32
    saved = torch.load(out / "choice_head.pt")
    assert set(saved) == {"body", "optimizer"} and "state" in saved["optimizer"]
    orienting = json.loads((out / "tectal_orienting.json").read_text(encoding="utf-8"))
    assert orienting["mode"] == "head" and orienting["steps"] == len(orienting["history"])
    assert not (out / "value_gate.json").exists()


def test_b11_a_loss_that_is_not_finite_raises():
    live = choice_head.LiveChoiceHead(5)
    live.choose({18: 0.9, 23: 0.4}, np.ones(choice_head.INPUT_DIM, dtype=np.float32), step=0, episode=0)
    with torch.no_grad():
        live.head.body[4].bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        live.observe_reward(1.0)


def test_b12_the_streams_of_the_agent_are_those_of_the_wiring_with_the_flag_off():
    states = {}
    cuda_states = {}                                                   # O2
    for mode in ("value", "plain"):
        torch.manual_seed(123)                                         # the test sets the agent streams
        np.random.seed(123)
        random.seed(123)
        tectum, spec, policy, calls, live = wired_mode(mode)
        run_steps(tectum, spec, policy, 20, rewards={15: 1.0}, states=step_states(20))
        states[mode] = (torch.get_rng_state(), np.random.get_state(), random.getstate(), live)
        cuda_states[mode] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    (t_off, n_off, p_off, _), (t_on, n_on, p_on, live) = states["value"], states["plain"]
    assert torch.equal(t_off, t_on) and p_off == p_on
    assert n_off[0] == n_on[0] and np.array_equal(n_off[1], n_on[1]) and n_off[2:] == n_on[2:]
    reference = np.random.default_rng(3000005)
    reference.random(), reference.integers(0, 2)
    assert len(live.entries) == 1 and live.head.rng.bit_generator.state == reference.bit_generator.state
    assert len(cuda_states["value"]) == len(cuda_states["plain"]) == torch.cuda.device_count()
    assert all(torch.equal(off, on) for off, on in zip(cuda_states["value"], cuda_states["plain"]))


def calls_in(tree, name: str) -> list:
    """Each call of the function `name` inside `tree`, in the order of the source."""
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]
    return sorted(found, key=lambda node: (node.lineno, node.col_offset))


def test_b14_main_wires_the_orienting_once_and_checks_the_flags_in_order():
    """O1 of the review of the wiring. The text of main is read with ast, with no run."""
    from scripts.training import train_rlhf
    with open(train_rlhf.__file__, encoding="utf-8") as handle:
        module = ast.parse(handle.read())
    main = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    assert len(main) == 1
    orienting = calls_in(main[0], "_wire_orienting")
    assert len(orienting) == 1 and not orienting[0].keywords
    assert [ast.unparse(argument) for argument in orienting[0].args] == ["args", "action_core", "tectum"]
    choice, competing = calls_in(main[0], "_check_choice_head_flags"), calls_in(main[0], "_check_competing_goals_flags")
    assert len(choice) == 1 and len(competing) == 1
    assert (choice[0].lineno, choice[0].col_offset) > (competing[0].lineno, competing[0].col_offset)
    episodes = calls_in(main[0], "_wire_choice_head_episodes")
    assert len(episodes) == 1
    guards = [node for node in ast.walk(main[0]) if isinstance(node, ast.If)
              and any(child is episodes[0] for statement in node.body for child in ast.walk(statement))]
    assert "live_choice_head is not None" in [ast.unparse(guard.test) for guard in guards]
