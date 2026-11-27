"""The unit tests of group C of step P2 of design P: check 2b, three learning rules of the choice head.

Rules: docs/decisions/2026_11_22_workspace_choice_mode.md, sections "Owner decision before work 4, and
the rules of check 2b" and "Changes after the independent review of check 2b". The numbers C1 to C8 are
those of the rules, with R1 in C7, R4 in C3 and R5 in C2. These tests use built data, rows made for the
test, and the committed rows of check 2. They read no saved run.
"""
from __future__ import annotations

import itertools
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import choice_head
from models.core.choice_head import ChoiceHead
from scripts.analysis import check_choice_head as check
from scripts.analysis.check_choice_read import ControlMismatch

CHECK_2_ROWS = os.path.join(os.path.dirname(__file__), "..", "docs", "results", "data",
                            "workspace_choice_mode_2026_11", "training_rows.csv")


def random_rows(count: int, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, choice_head.INPUT_DIM))


def rule_case(rule: str, run_pass: str = "first", data_seed: int = 0) -> dict:
    return {"rule": rule, "part": "B", "pass": run_pass, "rows": check.MODE_B_SCORE_ROWS, "data_seed": data_seed}


def rule_training(rule: str, run_pass: str = "first", data_seed: int = 0, seed: int = 0, repeat: int = 0,
                  episodes: int = check.MODE_B_EPISODES, checkpoints=check.MODE_B_CHECKPOINTS):
    return check.part_b_training(rule_case(rule, run_pass, data_seed), seed, repeat, episodes, checkpoints,
                                 checkpoint_row=check.rule_checkpoint_row)


# C1. The loss of `both`.

def test_c1_both_trains_the_chosen_output_toward_the_outcome_and_the_other_toward_1_minus_it():
    outputs = torch.tensor([[0.3, -0.2], [1.5, 0.4], [0.0, 2.0]])
    chosen, outcome = torch.tensor([0, 1, 1]), torch.tensor([[1.0], [0.0], [1.0]])
    targets = choice_head.both_slots_targets(chosen, outcome)
    assert targets.tolist() == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    closed_form = ((0.3 - 1) ** 2 + (-0.2 - 0) ** 2 + (1.5 - 1) ** 2 + (0.4 - 0) ** 2 + (0.0 - 0) ** 2
                   + (2.0 - 1) ** 2) / 6
    assert choice_head.both_slots_loss(outputs, chosen, outcome).item() == pytest.approx(closed_form, abs=1e-7)


def test_c1_the_update_of_both_moves_the_two_outputs_and_chosen_moves_one():
    for rule, moves_other in (("both", True), ("chosen", False), ("hopeful", False)):
        head = ChoiceHead(4, [0, 0, 0, 2], rule=rule)
        last = head.body[4]
        weight, bias = last.weight.detach().clone(), last.bias.detach().clone()
        head.update(random_rows(1), np.array([0]), np.array([1.0]), steps=1)
        assert not torch.equal(last.weight[0], weight[0])
        assert (not torch.equal(last.bias[1], bias[1])) == moves_other


def test_c1_an_unknown_rule_is_refused_and_chosen_is_the_default():
    assert ChoiceHead(0, [0, 0, 0, 2]).rule == "chosen"
    assert choice_head.RULES == ("chosen", "both", "hopeful")
    with pytest.raises(ValueError):
        ChoiceHead(0, [0, 0, 0, 2], rule="other")


# C2. `hopeful` at the start (R5).

def test_c2_the_outputs_of_hopeful_at_the_start_are_those_of_chosen_plus_1():
    row_sets = [check.mode_score_set(data_seed)["rows"] for data_seed in range(10)]
    row_sets += [random_rows(50, seed=3) * 20.0, np.zeros((1, choice_head.INPUT_DIM))]
    for data_seed, initial_seed in itertools.product(range(10), range(3)):
        seed = check.mode_head_seed(data_seed, initial_seed)
        chosen = ChoiceHead(seed, check.exploration_seed(initial_seed, data_seed, 0))
        hopeful = ChoiceHead(seed, check.exploration_seed(initial_seed, data_seed, 0), rule="hopeful")
        for rows in row_sets:
            assert np.max(np.abs(hopeful.outputs(rows) - (chosen.outputs(rows) + 1.0))) <= 1e-6
            assert np.max(np.abs(hopeful.preferences(rows) - chosen.preferences(rows))) <= 2e-6


def test_c2_hopeful_adds_1_to_each_bias_of_the_last_layer_and_changes_no_other_weight():
    chosen, hopeful = ChoiceHead(205, [0, 2, 0, 2]), ChoiceHead(205, [0, 2, 0, 2], rule="hopeful")
    for (name, mine), theirs in zip(hopeful.body.state_dict().items(), chosen.body.state_dict().values()):
        expected = theirs + 1.0 if name == "4.bias" else theirs
        assert torch.equal(mine, expected), name


def test_c2_to_build_a_head_of_each_rule_draws_no_number_from_a_stream_of_the_agent(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the head must not seed a global generator")
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    for rule in choice_head.RULES:
        torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        ChoiceHead(7, [0, 0, 0, 2], rule=rule)
        assert torch.equal(torch.get_rng_state(), torch_state)
        after = np.random.get_state()
        assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
        assert random.getstate() == python_state


# C3. K6 with R4: `chosen` gives the committed rows of part B of check 2.

def test_c3_chosen_gives_the_committed_rows_of_part_b_of_check_2_in_each_pass():
    committed = check.read_mode_rows(CHECK_2_ROWS)
    made = []
    for run_pass in check.MODE_PASSES:
        made += rule_training("chosen", run_pass, data_seed=3, seed=1, repeat=2)
    assert check.K6_MEASURES == ("score", "count_choices", "count_explored", "share_explored", "share_food",
                                 "share_slot_0", "mean_outcome", "informative")
    reading = check.rule_k6(made, committed)
    assert reading == {"compared": 14, "mismatches": 0, "missing": 0}


def test_c3_k6_finds_a_change_of_one_measure_and_a_missing_row():
    committed = check.read_mode_rows(CHECK_2_ROWS)
    made = rule_training("chosen", "second", data_seed=0, seed=0, repeat=0)
    made[3] = dict(made[3], share_slot_0=made[3]["share_slot_0"] + 1e-12)
    assert check.rule_k6(made, committed)["mismatches"] == 1
    other = [row for row in committed if not (row["part"] == "B" and row["pass"] == "second"
                                              and row["data_seed"] == 0 and row["initial_seed"] == 0
                                              and row["fold"] == 0 and row["checkpoint"] == 60)]
    assert check.rule_k6(made, other)["missing"] == 1


def test_c3_k6_reads_only_the_rows_of_chosen():
    committed = check.read_mode_rows(CHECK_2_ROWS)
    made = rule_training("hopeful", "first", episodes=10, checkpoints=(0, 10))
    assert check.rule_k6(made, committed) == {"compared": 0, "mismatches": 0, "missing": 0}


# C4. greedy_food.

def test_c4_greedy_food_on_made_outputs_with_slot_0_on_a_tie():
    outputs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5], [-1.0, -2.0]], dtype=np.float32)
    target = np.array([1, 0, 1, 0, 0])          # target 1: the food is in slot 0
    # greedy slots 0, 1, 0, 0, 0; food slots 0, 1, 0, 1, 1
    assert check.greedy_food(outputs, target) == pytest.approx(3 / 5)
    assert check.greedy_food(outputs, 1 - target) == pytest.approx(2 / 5)


def test_c4_each_checkpoint_row_of_check_2b_holds_greedy_food_of_the_head_on_the_score_rows():
    rows = rule_training("both", episodes=10, checkpoints=(0, 10))
    head = ChoiceHead(check.mode_head_seed(0, 0), check.exploration_seed(0, 0, 0), rule="both")
    scoring = check.mode_score_set(0)
    assert rows[0]["greedy_food"] == check.greedy_food(head.outputs(scoring["rows"]), scoring["target"])
    assert all(0.0 <= row["greedy_food"] <= 1.0 for row in rows)


# C5. The draws of the three rules.

def test_c5_the_three_rules_take_the_same_two_numbers_for_each_choice():
    states, counts = [], []
    for rule in choice_head.RULES:
        head = ChoiceHead(check.mode_head_seed(5, 2), check.exploration_seed(2, 5, 4), rule=rule)
        window = check.ChoiceWindow()
        episode_set = check.part_b_rows(5, 4)
        for episode in range(30):
            check.choose_one_episode(head, episode_set["rows"][episode:episode + 1],
                                     episode_set["target"][episode:episode + 1], rule_case(rule, "first", 5),
                                     window, [])
        states.append(head.rng.bit_generator.state)
        counts.append(window.close()["count_explored"])
    reference = np.random.default_rng(check.exploration_seed(2, 5, 4))
    for _ in range(30):
        reference.random(), reference.integers(0, 2)
    assert states[0] == states[1] == states[2] == reference.bit_generator.state
    assert counts[0] == counts[1] == counts[2] == check.redrawn_explored(check.exploration_seed(2, 5, 4), [30])[0]


def test_c5_k2_holds_for_each_rule_and_finds_a_changed_count():
    made = []
    for rule in choice_head.RULES:
        made += rule_training(rule, "first", data_seed=1, seed=2, repeat=3, episodes=20, checkpoints=(0, 10, 20))
    reading = check.rule_k2(made)
    assert reading["mismatches"] == 0 and reading["trainings"] == 3
    made[4] = dict(made[4], count_explored=made[4]["count_explored"] + 1)
    assert check.rule_k2(made)["mismatches"] == 1


# C6. The second pass of `both`.

def test_c6_the_second_pass_of_both_gives_the_other_output_the_target_0():
    chosen = np.array([0, 1, 1, 0])
    outcome = check.mode_outcomes(chosen, np.array([1, 1, 0, 0]), True)
    targets = choice_head.both_slots_targets(torch.as_tensor(chosen), torch.as_tensor(outcome).float().unsqueeze(1))
    assert targets.tolist() == [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]


def test_c6_in_the_second_pass_both_moves_the_outputs_toward_1_for_its_choice_and_0_for_the_other():
    head = ChoiceHead(9, [0, 0, 0, 2], rule="both")
    row = random_rows(1, seed=4)
    target = np.array([1.0, 0.0])
    before = float(np.mean((head.outputs(row)[0] - target) ** 2))
    first_loss = head.update(row, np.array([0]), np.array([1.0]), steps=choice_head.UPDATES_PER_CHOICE)
    after = float(np.mean((head.outputs(row)[0] - target) ** 2))
    assert first_loss == pytest.approx(before, abs=1e-6) and after < before


# C7. The choosing rule on made mean rows (R1).

def made_rule_rows(greedy, score_first=0.9, score_second=0.5, greedy_second=0.5):
    """Each mean row of a complete check 2b. greedy[rule] is the first pass greedy_food of each data set."""
    made = []
    for case in check.rule_cases():
        for checkpoint in check.MODE_B_CHECKPOINTS:
            row = dict(case, checkpoint=checkpoint, **{name: 0.25 for name in check.RULE_MEASURES})
            first = case["pass"] == "first"
            row["score"] = score_first if first else score_second
            row["greedy_food"] = greedy[case["rule"]] if first else greedy_second
            made.append(row)
    return made


def choice_of(greedy, mismatches=0, **kwargs):
    return check.choose_rule(check.RuleTable(made_rule_rows(greedy, **kwargs)), mismatches)


def test_c7_no_rule_passes():
    made = choice_of({"chosen": 0.77, "both": 0.80, "hopeful": 0.77})
    assert (made["result"], made["rule"]) == ("no rule passes", "chosen")


def test_c7_no_rule_passes_with_a_low_score():
    made = choice_of({"chosen": 0.77, "both": 0.95, "hopeful": 0.95}, score_first=0.79)
    assert (made["result"], made["rule"]) == ("no rule passes", "chosen")


def test_c7_one_rule_passes():
    assert choice_of({"chosen": 0.77, "both": 0.77, "hopeful": 0.90})["rule"] == "hopeful"
    made = choice_of({"chosen": 0.77, "both": 0.90, "hopeful": 0.80})
    assert (made["result"], made["rule"]) == ("one rule passes", "both")


def test_c7_both_pass_with_a_difference_within_the_limit_gives_hopeful():
    for both, hopeful in ((0.92, 0.90), (0.90, 0.92), (0.90, 0.90)):
        made = choice_of({"chosen": 0.77, "both": both, "hopeful": hopeful})
        assert (made["result"], made["rule"]) == ("both rules pass", "hopeful")
        assert made["tie"]["d"] == pytest.approx(both - hopeful) and made["tie"]["limit"] == pytest.approx(0.05)


def test_c7_both_pass_with_a_difference_above_the_limit_each_way():
    assert choice_of({"chosen": 0.77, "both": 0.98, "hopeful": 0.90})["rule"] == "both"
    assert choice_of({"chosen": 0.77, "both": 0.90, "hopeful": 0.98})["rule"] == "hopeful"


def test_c7_void_gives_no_rule():
    made = choice_of({"chosen": 0.77, "both": 0.98, "hopeful": 0.90}, mismatches=1)
    assert (made["result"], made["rule"]) == ("VOID", None)


def test_c7_the_conditions_use_paired_differences_and_the_limit_of_change_19():
    greedy = {"chosen": 0.60, "both": 0.73, "hopeful": 0.60}
    made = made_rule_rows(greedy)
    spread = np.array([0.3, -0.3] * 5)           # `chosen` alternates, so the paired differences spread
    for row in made:
        if row["rule"] == "chosen" and row["pass"] == "first":
            row["greedy_food"] = 0.60 + spread[row["data_seed"]]
    conditions = check.rule_conditions(check.RuleTable(made), "both")
    limit = check.limit_of_change_19(0.13 - spread)["limit"]
    assert conditions["greedy_food_40"]["d"] == pytest.approx(0.13)
    assert conditions["greedy_food_40"]["limit"] == pytest.approx(limit) and limit > 0.13
    assert conditions["score_40"]["holds"] and conditions["greedy_food_d_60"]["holds"]
    assert not conditions["greedy_food_40"]["holds"] and not conditions["passes"]


# C8. The rows of each training, their means and the counts.

def test_c8_the_counts_420_and_6300_and_the_60_cases():
    assert check.rule_expected_counts() == (420, 6300)
    cases = check.rule_cases()
    assert len(cases) == 60
    assert {(c["rule"], c["pass"], c["data_seed"]) for c in cases} == set(
        itertools.product(choice_head.RULES, check.MODE_PASSES, range(10)))


def test_c8_the_rows_of_each_training_and_their_means():
    case = rule_case("hopeful", "second", 2)
    training_rows, mean_rows = check.rule_case_rows(case, 2, (0, 1, 2))
    assert len(training_rows) == 15 * 3 and len(mean_rows) == 3
    keys = {(r["initial_seed"], r["fold"], r["checkpoint"]) for r in training_rows}
    assert keys == set(itertools.product(range(3), range(5), (0, 1, 2)))
    for mean_row in mean_rows:
        for name in ("score", "greedy_food"):
            members = [r[name] for r in training_rows if r["checkpoint"] == mean_row["checkpoint"]]
            assert mean_row[name] == pytest.approx(np.mean(members))
        assert mean_row["rule"] == "hopeful" and mean_row["pass"] == "second"


def test_c8_an_incomplete_check_is_refused():
    made = made_rule_rows({"chosen": 0.77, "both": 0.77, "hopeful": 0.77})
    with pytest.raises(ControlMismatch):
        check.check_rule_complete(made[:-1], [{}] * 6300)
    with pytest.raises(ControlMismatch):
        check.check_rule_complete(made, [{}] * 6299)
    check.check_rule_complete(made, [{}] * 6300)
