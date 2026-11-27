"""The calculations of the offline choice head (scripts/analysis/check_choice_head.py).

Rules: docs/decisions/2026_11_16_workspace_choice_head.md. These tests use built data only.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.self_model.action_selection_core import ActionSelectionCore
from scripts.analysis import check_choice_head as check
from scripts.analysis.check_choice_read import ControlMismatch


def as_column(numbers):
    return torch.tensor(numbers, dtype=torch.float32).unsqueeze(1)


class TestVariantLossByHand:
    """Three rows: an advantage above 0, one below 0 and one of exactly 0."""

    value = as_column([0.25, 0.5, 1.0])
    outcome = as_column([1.0, 0.0, 1.0])            # advantage +0.75, -0.5, 0
    action = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    go = torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    no_go = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    brake = as_column([0.0, 0.0, 0.0])

    def terms(self):
        return check.variant_loss_terms(self.value, self.go, self.no_go, self.brake, self.action, self.outcome)

    def test_value_term(self):
        # Squared errors 0.75^2, 0.5^2 and 0, mean over 3 rows.
        assert self.terms()["value"].item() == pytest.approx((0.5625 + 0.25 + 0.0) / 3)

    def test_go_term(self):
        # Targets: row 1 (+1, 0), row 2 (0, -1), row 3 (0, 0). Errors: 0.5, 0 | 0, 1.5 | 0.5, 0.5.
        assert self.terms()["go"].item() == pytest.approx((0.25 + 0.0 + 0.0 + 2.25 + 0.25 + 0.25) / 6)

    def test_no_go_term(self):
        # Each no-go signal is 0.5, so each of the 6 terms is ln 2 for a target of 0 or of 1.
        assert self.terms()["no_go"].item() == pytest.approx(math.log(2.0))

    def test_brake_term(self):
        assert self.terms()["brake"].item() == pytest.approx((math.tanh(0.75) ** 2 + math.tanh(0.5) ** 2) / 3)

    def test_total_has_the_weights_of_the_code(self):
        terms = self.terms()
        expected = terms["go"] + terms["no_go"] + 0.3 * terms["brake"] + 0.5 * terms["value"]
        assert terms["total"].item() == pytest.approx(expected.item())


class TestVariantLossAgainstTheCode:
    def test_same_total_as_update_policy_on_the_same_rows(self):
        # The policy class of the agent, with 2 channels, the working memory at zero, one row for each
        # rollout (done is true, so the return is the reward), and the action vector of the rules.
        torch.manual_seed(0)
        core = ActionSelectionCore({"workspace_dim": 256, "action_dim": 2}, emotion_shaper=None, memory=None)
        rows = torch.randn(12, 256)
        chosen = torch.tensor([0, 1] * 6)
        outcome = torch.tensor([1.0, 0.0, 0.0, 1.0] * 3)
        action = check.action_vectors(chosen)
        expected = check.variant_loss(core.pfc, core.bg, rows, action, outcome.unsqueeze(1))["total"].item()
        for index in range(12):
            core.rollout_buffer.append({"state": rows[index:index + 1], "hidden": torch.zeros(1, 256),
                                        "action": action[index], "reward": float(outcome[index]),
                                        "next_state": rows[index:index + 1], "done": True, "rpe": 0.0})
        assert core.update_policy()["total_loss"] == pytest.approx(expected, rel=1e-5)


class TestSlotNumbers:
    def test_the_number_is_go_times_one_minus_no_go(self):
        torch.manual_seed(1)
        head = check.build_head("variant", 256, seed=0)
        rows = torch.randn(5, 256)
        state, _ = head["pfc"](rows, torch.zeros(5, 256))
        expected = head["bg"].direct_pathway(state) * (1.0 - head["bg"].indirect_pathway(state))
        assert torch.allclose(check.slot_numbers(head, rows), expected)

    def test_the_plain_value_head_has_the_body_of_the_plain_actor_critic(self):
        head = check.build_head("plain", 256, seed=0)
        layers = [layer for layer in head["net"] if isinstance(layer, nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in layers] == [(256, 128), (128, 128), (128, 2)]

    def test_the_counts_of_weights_of_the_rules(self):
        variant, plain = check.build_head("variant", 256, seed=0), check.build_head("plain", 256, seed=0)
        assert sum(p.numel() for part in ("pfc", "bg") for p in variant[part].parameters()) == 576396
        assert sum(p.numel() for p in plain["net"].parameters()) == 49666

    def test_the_same_seed_gives_the_same_initial_weights(self):
        first, second = check.build_head("plain", 256, seed=2), check.build_head("plain", 256, seed=2)
        assert all(torch.equal(a, b) for a, b in zip(first["net"].parameters(), second["net"].parameters()))


class TestFeedback:
    def test_action_vectors_have_1_for_the_chosen_slot(self):
        assert check.action_vectors(torch.tensor([0, 1])).tolist() == [[1.0, 0.0], [0.0, 1.0]]

    def test_the_outcome_is_1_when_the_chosen_slot_is_the_slot_of_the_food(self):
        target = np.array([1, 1, 0, 0])      # 1 means that the food is in slot 0
        chosen = np.array([0, 1, 0, 1])
        assert check.built_outcomes(chosen, target).tolist() == [1.0, 0.0, 0.0, 1.0]

    @pytest.mark.parametrize("kind", ["variant", "plain"])
    def test_a_head_never_receives_the_target(self, kind):
        rng = np.random.default_rng(0)
        rows, chosen = rng.normal(size=(20, 16)), rng.integers(0, 2, 20)
        outcome = rng.integers(0, 2, 20).astype(float)
        first = check.train_head(kind, 0, rows, chosen, outcome, updates=5)
        second = check.train_head(kind, 0, rows, chosen, outcome, updates=5)
        for a, b in zip(check.head_parameters(first), check.head_parameters(second)):
            assert torch.equal(a, b)
        assert "target" not in check.train_head.__code__.co_varnames

    @pytest.mark.parametrize("kind", ["variant", "plain"])
    def test_training_changes_the_weights(self, kind):
        rng = np.random.default_rng(0)
        rows, chosen = rng.normal(size=(20, 16)), rng.integers(0, 2, 20)
        outcome = check.built_outcomes(chosen, rng.integers(0, 2, 20))
        before = [p.clone() for p in check.head_parameters(check.build_head(kind, 16, seed=0))]
        after = check.head_parameters(check.train_head(kind, 0, rows, chosen, outcome, updates=5))
        assert any(not torch.equal(a, b) for a, b in zip(before, after))


class TestBlankControl:
    @pytest.mark.parametrize("kind", ["variant", "plain"])
    def test_an_input_of_zeros_scores_one_half_exactly(self, kind):
        target = np.arange(30) % 2
        chosen = np.random.default_rng(0).integers(0, 2, 30)
        outcome = check.built_outcomes(chosen, target)
        assert check.score_of_one_seed(kind, 0, np.zeros((30, 8)), chosen, outcome, target, updates=5) == 0.5


class TestBuiltData:
    def test_the_targets_are_in_equal_parts(self):
        case = check.built_case(rows=30, signal_share=1.0, chooser="random", data_seed=0)
        assert case["target"].sum() == 15 and case["rows"].shape == (30, 256)

    def test_the_signal_is_in_64_dimensions_of_the_rows_with_target_1(self):
        noise = check.built_case(rows=40, signal_share=0.0, chooser="random", data_seed=3)
        full = check.built_case(rows=40, signal_share=1.0, chooser="random", data_seed=3)
        shift = full["rows"] - noise["rows"]
        assert np.allclose(shift[full["target"] == 1][:, :64], 1.0)
        assert np.all(shift[full["target"] == 0] == 0.0) and np.all(shift[:, 64:] == 0.0)

    def test_a_share_of_60_percent_gives_the_signal_to_60_percent_of_the_rows(self):
        case = check.built_case(rows=40, signal_share=0.6, chooser="random", data_seed=1)
        assert case["with_signal"].sum() == 24

    def test_no_signal_gives_noise_only(self):
        case = check.built_case(rows=30, signal_share=0.0, chooser="random", data_seed=1)
        assert case["with_signal"].sum() == 0

    def test_the_four_choosers(self):
        cases = {name: check.built_case(rows=40, signal_share=1.0, chooser=name, data_seed=2)
                 for name in check.CHOOSERS}
        food_slot = 1 - cases["random"]["target"]
        assert np.all(cases["slot_0"]["chosen"] == 0)
        assert np.array_equal(cases["food"]["chosen"], food_slot) and np.all(cases["food"]["outcome"] == 1.0)
        assert (cases["food_80"]["chosen"] == food_slot).sum() == 32
        assert 0 < cases["random"]["chosen"].sum() < 40

    def test_the_second_pass_sets_each_outcome_to_1(self):
        case = check.built_case(rows=30, signal_share=1.0, chooser="random", data_seed=0, outcomes_at_1=True)
        assert np.all(case["outcome"] == 1.0)

    def test_the_count_of_result_rows_of_the_test(self):
        assert len(check.built_test_cases()) * len(check.HEADS) == 960


def gate_episode(salience_18, salience_23, choice, outcome=1.0):
    """One episode of a built value_gate.json, with the keys that the gate writes."""
    return {"choice": choice, "salience": {"18": salience_18, "23": salience_23}, "outcome": outcome}


def built_records(chooser, rows=40, data_seed=0):
    case = check.built_case(rows=rows, signal_share=1.0, chooser=chooser, data_seed=data_seed)
    return [{"chosen": int(c), "outcome": int(o), "target": int(t)}
            for c, o, t in zip(case["chosen"], case["outcome"], case["target"])]


class TestSlotOrderAndTarget:
    def test_the_food_band_in_slot_0_gives_target_1(self):
        record = check.slot_record(gate_episode(0.9, 0.4, choice=18), food_band=18, where="built")
        assert record == {"chosen": 0, "outcome": 1, "target": 1}

    def test_the_food_band_in_slot_1_gives_target_0(self):
        record = check.slot_record(gate_episode(0.4, 0.9, choice=18), food_band=18, where="built")
        assert record == {"chosen": 1, "outcome": 1, "target": 0}

    def test_the_target_follows_the_band_of_the_food_after_the_switch(self):
        record = check.slot_record(gate_episode(0.4, 0.9, choice=18, outcome=0.0), food_band=23, where="built")
        assert record == {"chosen": 1, "outcome": 0, "target": 1}

    def test_equal_salience_raises(self):
        with pytest.raises(ControlMismatch):
            check.slot_record(gate_episode(0.5, 0.5, choice=18), food_band=18, where="built")

    def test_another_set_of_candidates_raises(self):
        episode = {"choice": 18, "salience": {"18": 0.9, "23": 0.4, "30": 0.1}, "outcome": 1.0}
        with pytest.raises(ControlMismatch):
            check.slot_record(episode, food_band=18, where="built")

    def test_an_outcome_that_is_not_0_or_1_raises(self):
        with pytest.raises(ControlMismatch):
            check.slot_record(gate_episode(0.9, 0.4, choice=18, outcome=0.5), food_band=18, where="built")


class TestCountRule:
    def test_the_count_table_has_8_cells_that_sum_to_the_episodes(self):
        records = built_records("random")
        table = check.count_table(records)
        assert len(table) == 8 and sum(table.values()) == 40
        assert table["chosen_0_outcome_1_target_1"] == sum(
            r == {"chosen": 0, "outcome": 1, "target": 1} for r in records)

    def test_2_of_each_outcome_for_one_slot_is_informative(self):
        records = [{"chosen": 0, "outcome": o, "target": 0} for o in (1, 1, 0, 0)]
        records += [{"chosen": 1, "outcome": 1, "target": 0}] * 5
        counts = check.training_set_counts(records)
        assert counts["slot_0"] == {"outcome_1": 2, "outcome_0": 2}
        assert counts["slot_1"] == {"outcome_1": 5, "outcome_0": 0}
        assert counts["rule_count"] == 2 and counts["informative"]

    def test_1_episode_of_an_outcome_is_not_informative(self):
        records = [{"chosen": 0, "outcome": o, "target": 0} for o in (1, 1, 1, 0)]
        records += [{"chosen": 1, "outcome": o, "target": 0} for o in (0, 0, 0, 1)]
        counts = check.training_set_counts(records)
        assert counts["rule_count"] == 1 and not counts["informative"]

    def test_a_chooser_that_takes_the_food_in_all_episodes_but_2_is_void(self):
        records = built_records("food")
        records[0]["outcome"], records[1]["outcome"] = 0, 0
        report = check.unit_report(records, "built")
        assert report["void_for_the_recorded_feedback"] and report["smallest_rule_count"] < 2

    def test_a_random_chooser_is_not_void(self):
        report = check.unit_report(built_records("random"), "built")
        assert not report["void_for_the_recorded_feedback"] and len(report["training_sets"]) == 5
        assert [one_set["episodes"] for one_set in report["training_sets"]] == [32] * 5

    def test_too_few_episodes_of_a_target_raises(self):
        records = [{"chosen": 0, "outcome": 1, "target": int(index < 9)} for index in range(30)]
        with pytest.raises(ControlMismatch):
            check.unit_report(records, "built")


class TestSharesAndNearestChooser:
    @pytest.mark.parametrize("chooser,food_share,slot_0_share", [("food", 1.0, 0.5), ("slot_0", 0.5, 1.0),
                                                                  ("food_80", 0.8, None)])
    def test_the_shares_of_a_built_chooser(self, chooser, food_share, slot_0_share):
        report = check.unit_report(built_records(chooser), "built")
        assert report["share_food_slot_chosen"] == pytest.approx(food_share)
        # With food_80 the share of slot 0 depends on which rows are changed, so it has no exact value.
        assert slot_0_share is None or report["share_slot_0_chosen"] == pytest.approx(slot_0_share)
        assert report["nearest_chooser"] == chooser

    def test_the_exact_point_of_each_chooser_is_nearest_to_itself(self):
        for name, (food_share, slot_0_share) in check.CHOOSER_POINTS.items():
            assert check.nearest_chooser(food_share, slot_0_share) == name


class TestReadOfABuiltRun:
    def write_run(self, folder, name, count):
        episodes = [gate_episode(0.9 if index % 2 else 0.4, 0.6, choice=18 if index % 3 else 23,
                                 outcome=float(index % 3 != 0)) for index in range(count)]
        (folder / name).mkdir()
        (folder / name / "value_gate.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")

    def test_a_unit_holds_its_own_episodes_only(self, tmp_path):
        self.write_run(tmp_path, "rv_switch_s701", 60)
        before = check.read_unit_records(str(tmp_path), "rv_switch_s701", 0, 30, 18)
        after = check.read_unit_records(str(tmp_path), "rv_switch_s701", 30, 60, 23)
        assert len(before) == 30 and len(after) == 30
        assert before[1] == {"chosen": 0, "outcome": 1, "target": 1}       # band 18 leads and is chosen
        assert after[0] == {"chosen": 0, "outcome": 0, "target": 1}        # band 23 leads and is chosen

    def test_another_count_of_episodes_raises(self, tmp_path):
        self.write_run(tmp_path, "vg_learn_s601", 39)
        with pytest.raises(ControlMismatch):
            check.read_unit_records(str(tmp_path), "vg_learn_s601", 0, 40, 18)

    def test_the_nine_units_of_the_rules(self):
        assert len(check.UNITS) == 9 and sum(unit[5] for unit in check.UNITS) == 6

    def test_the_food_switch_units_divide_at_episode_29(self):
        # The saved runs moved the food at recorded episode 29 (the correction of 2026-11-16 in the rules).
        spans = {name: (first, after_last, food_band) for name, _, first, after_last, food_band, _ in check.UNITS}
        for seed in (701, 702, 703):
            assert spans["rv_switch_s%d before" % seed] == (0, 29, 18)
            assert spans["rv_switch_s%d after" % seed] == (29, 60, 23)
        for seed in (601, 602, 603):
            assert spans["vg_learn_s%d" % seed] == (0, 40, 18)

    def test_episode_29_of_a_built_food_switch_run_is_in_the_unit_after_the_switch(self, tmp_path):
        self.write_run(tmp_path, "rv_switch_s701", 60)
        units = {name: (run, first, after_last, band) for name, run, first, after_last, band, _ in check.UNITS}
        before = check.read_unit_records(str(tmp_path), *units["rv_switch_s701 before"])
        after = check.read_unit_records(str(tmp_path), *units["rv_switch_s701 after"])
        assert len(before) == 29 and len(after) == 31
        # Episode 29 of the built run: band 18 leads (salience 0.9) and is chosen, with outcome 1.
        # After the switch the food band is 23, in slot 1, so the target is 0.
        assert after[0] == {"chosen": 0, "outcome": 1, "target": 0}


# The check on the variant. Rules: docs/decisions/2026_11_16_choice_head_variant_sign.md.
# The 13 unit tests of the rules, in their order. Built data only.

def small_split(rows=12, dimensions=16, seed=0):
    """A hand-made fold with each outcome at 1, as the food chooser gives."""
    rng = np.random.default_rng(seed)
    return {"train_rows": rng.normal(size=(rows, dimensions)), "chosen": rng.integers(0, 2, rows),
            "outcome": np.ones(rows), "test_rows": rng.normal(size=(6, dimensions)),
            "test_target": np.array([0, 1, 0, 1, 0, 1])}


class TwoCritics:
    """One row with outcome 1. The learning critic gives 0.5 (advantage +0.5). The untrained critic
    gives 2.0 (advantage -1.0). So each target shows which critic it uses."""

    signals = {"value": as_column([0.5]), "go": torch.tensor([[0.5, 0.0]]),
               "no_go": torch.tensor([[0.25, 0.25]]), "brake": as_column([0.0])}
    action = torch.tensor([[1.0, 0.0]])
    outcome = as_column([1.0])
    untrained_value = as_column([2.0])
    go_term = {"learning": (0.5 ** 2 + 0.0) / 2, "untrained": (1.5 ** 2 + 0.0) / 2}   # target +1 or -1
    no_go_term = {"learning": -math.log(0.75), "untrained": -math.log(0.25)}          # target 0 or 1
    brake_term = {"learning": math.tanh(0.5) ** 2, "untrained": math.tanh(1.0) ** 2}

    @classmethod
    def terms(cls, condition):
        return check.sourced_loss_terms(cls.signals, cls.action, cls.outcome, cls.untrained_value,
                                        check.UNTRAINED_TARGETS[condition])

    @classmethod
    def assert_sources(cls, condition, go, no_go, brake):
        terms = cls.terms(condition)
        assert terms["go"].item() == pytest.approx(cls.go_term[go])
        assert terms["no_go"].item() == pytest.approx(cls.no_go_term[no_go])
        assert terms["brake"].item() == pytest.approx(cls.brake_term[brake])
        assert terms["value"].item() == pytest.approx(0.25)     # the value term has the learning critic
        expected = terms["go"] + terms["no_go"] + 0.3 * terms["brake"] + 0.5 * terms["value"]
        assert terms["total"].item() == pytest.approx(expected.item())


class TestVariantSign:
    def test_01_share_of_negative_advantage_and_mean_absolute_advantage(self):
        measures = check.advantage_measures(as_column([0.5, -0.25, 0.0, -1.0]))
        assert measures["share_negative"] == pytest.approx(0.5)
        assert measures["share_zero"] == pytest.approx(0.25)
        assert measures["mean_abs_advantage"] == pytest.approx(1.75 / 4)

    def test_02_go_target_of_the_chosen_slot_is_plus_1_minus_1_and_0(self):
        action = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        targets = check.go_targets(action, as_column([0.75, -0.5, 0.0]))
        assert targets.tolist() == [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]

    def test_03_score_with_the_go_signals_only(self):
        # Go of slot 0 minus go of slot 1: 0.8, -0.6, 0.0, -0.2. Target 1 has 0.8 and -0.2, target 0
        # has -0.6 and 0.0. Three of the four pairs are in the right order.
        go = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.1, 0.3]])
        assert check.go_only_score(go, np.array([1, 0, 0, 1])) == pytest.approx(0.75)

    def test_04_new_rows_of_c2_differ_between_updates_and_repeat_for_the_same_seed_fold_and_update(self):
        first = check.new_row_set(24, "food", data_seed=3, fold=1, update=7)
        again = check.new_row_set(24, "food", data_seed=3, fold=1, update=7)
        other = check.new_row_set(24, "food", data_seed=3, fold=1, update=8)
        assert first["rows"].shape == (24, 256)
        assert np.array_equal(first["rows"], again["rows"]) and np.array_equal(first["chosen"], again["chosen"])
        assert check.rows_in_common(first["rows"], other["rows"]) == 0

    def test_05_the_advantage_of_c3_does_not_change_when_the_critic_learns(self):
        split = small_split()
        record = check.train_measured("C3b", 0, split, updates=5, checkpoints=(0, 5))
        rows = torch.as_tensor(split["train_rows"], dtype=torch.float32)
        with torch.no_grad():
            learned = check.variant_signals(record["head"], rows)["value"]
        assert torch.equal(record["untrained_value_first"], record["untrained_value_last"])
        assert not torch.equal(learned, record["untrained_value_first"])
        shares = [update["untrained_share_at_or_below_0"] for update in record["updates"]]
        assert shares == [0.0] * 5

    def test_06_a_checkpoint_at_200_updates_gives_the_weights_of_train_head(self):
        split = small_split()
        record = check.train_measured("C1", 1, split, updates=200, checkpoints=(0, 200))
        plain_training = check.train_head("variant", 1, split["train_rows"], split["chosen"], split["outcome"], 200)
        for measured, expected in zip(check.head_parameters(record["head"]), check.head_parameters(plain_training)):
            assert torch.equal(measured, expected)
        assert [checkpoint["checkpoint"] for checkpoint in record["checkpoints"]] == [0, 200]

    def test_07_the_counts_of_result_rows(self):
        cases = check.variant_sign_cases()
        by_condition = {name: sum(case["condition"] == name for case in cases) for name in check.SIGN_CONDITIONS}
        trainings = len(cases) * len(check.INITIAL_SEEDS) * check.OUTER_FOLDS
        assert by_condition == {"C1": 80, "C2": 80, "C3a": 20, "C3b": 20, "C3c": 20, "C4": 80}
        assert len(check.SIGN_CHECKPOINTS) == 12 and trainings == 4500
        assert len(cases) * len(check.SIGN_CHECKPOINTS) == 3600
        assert trainings * len(check.SIGN_CHECKPOINTS) == 54000

    def test_08_in_c3a_only_the_sign_of_the_go_target_uses_the_untrained_critic(self):
        TwoCritics.assert_sources("C3a", go="untrained", no_go="learning", brake="learning")

    def test_09_in_c3b_each_of_the_three_targets_uses_the_untrained_critic(self):
        TwoCritics.assert_sources("C3b", go="untrained", no_go="untrained", brake="untrained")

    def test_10_in_c3c_only_the_no_go_target_uses_the_untrained_critic(self):
        TwoCritics.assert_sources("C3c", go="learning", no_go="untrained", brake="learning")
        # With no target from the untrained critic, the terms are those of the loss of step P1.
        sourced = check.sourced_loss_terms(TwoCritics.signals, TwoCritics.action, TwoCritics.outcome,
                                           TwoCritics.untrained_value, ())
        signals = TwoCritics.signals
        step_p1 = check.variant_loss_terms(signals["value"], signals["go"], signals["no_go"], signals["brake"],
                                           TwoCritics.action, TwoCritics.outcome)
        assert all(torch.equal(sourced[name], step_p1[name]) for name in step_p1)

    @pytest.mark.parametrize("condition", ["C1", "C2", "C3a", "C4"])
    def test_11_rows_for_each_training_and_their_mean(self, condition):
        case = {"condition": condition, "rows": 30, "chooser": "food", "data_seed": 0}
        training_rows, mean_rows = check.run_case(case, updates=2, checkpoints=(0, 1, 2))
        keys = {(row["condition"], row["rows"], row["chooser"], row["data_seed"], row["initial_seed"],
                 row["fold"], row["checkpoint"]) for row in training_rows}
        assert len(training_rows) == 3 * 5 * 3 and len(keys) == len(training_rows)
        assert [row["checkpoint"] for row in mean_rows] == [0, 1, 2]
        measured = ("score",) if condition == "C4" else ("score", "score_go_only", "share_negative", "mean_no_go")
        for mean_row in mean_rows[1:]:
            members = [row for row in training_rows if row["checkpoint"] == mean_row["checkpoint"]]
            for name in measured:
                assert mean_row[name] == pytest.approx(np.mean([row[name] for row in members]))
        assert mean_rows[0]["share_negative"] == "" and mean_rows[0]["score"] != ""
        assert (mean_rows[2]["share_negative"] == "") == (condition == "C4")

    def test_12_no_new_row_of_c2_is_a_row_of_the_data_set_before_the_signal_is_added(self):
        for rows, data_seed in [(30, seed) for seed in range(10)] + [(40, seed) for seed in range(10)]:
            data_noise = check.built_noise(rows, data_seed)
            new_noise = check.new_row_noise(rows * 4 // 5, data_seed, fold=0, update=1)
            assert check.rows_in_common(new_noise, data_noise) == 0
            check.check_new_rows(new_noise, data_noise, "test")
        # The noise is the rows of the case before the signal is added: equal outside the 64 signal
        # dimensions, and the new row set is made from the noise of its seed, fold and update.
        case = check.built_case(rows=30, signal_share=1.0, chooser="food", data_seed=5)
        assert np.array_equal(check.built_noise(30, 5)[:, 64:], case["rows"][:, 64:])
        first = check.new_row_set(24, "food", data_seed=5, fold=0, update=1)
        assert np.array_equal(check.new_row_noise(24, 5, fold=0, update=1)[:, 64:], first["rows"][:, 64:])
        assert np.array_equal(first["rows"], check.new_row_set(24, "food", 5, 0, 1)["rows"])
        # The case that fails: an update count from 0 gives the rows of the data set again.
        repeated = check.new_row_noise(24, 5, fold=0, update=0)
        assert check.rows_in_common(repeated, check.built_noise(30, 5)) == 24
        with pytest.raises(ControlMismatch, match="V8"):
            check.check_new_rows(repeated, check.built_noise(30, 5), "test")

    @pytest.mark.parametrize("control", ["V6", "V7", "V9"])
    def test_13_a_mismatch_of_v6_v7_or_v9_stops_the_job_before_the_first_training(self, control, monkeypatch, tmp_path):
        cases = [{"condition": "C3a", "rows": 30, "chooser": "food", "data_seed": 0},
                 {"condition": "C2", "rows": 30, "chooser": "food", "data_seed": 0}]
        assert check.first_step(cases)["V6_values_C3a"] == 3 * 5 * 24      # no mismatch on the built data

        def no_training(*args, **kwargs):
            raise AssertionError("a training started")

        monkeypatch.setattr(check, "train_measured", no_training)
        monkeypatch.setattr(check, "train_head", no_training)
        if control == "V6":
            monkeypatch.setattr(check, "untrained_values", lambda seed, rows: torch.full((len(rows), 1), 1.5))
        if control == "V7":
            monkeypatch.setattr(check, "built_outcomes", lambda chosen, target: np.zeros(len(chosen)))
        if control == "V9":
            c2_split = check.c2_split
            monkeypatch.setattr(check, "c2_split", lambda built, case, fold: dict(
                c2_split(built, case, fold), test_rows=c2_split(built, case, fold)["test_rows"] + 1.0))
        with pytest.raises(ControlMismatch, match=control):
            check.run_variant_sign(str(tmp_path / "rows"), str(tmp_path / "no_file.csv"), cases=cases)
        assert list(tmp_path.iterdir()) == []


# The labels of the check on the variant, from the rows of the job. Each row below is hand-made.
# No row comes from a run.

def series(steps):
    """A number for each checkpoint: the number of the largest listed checkpoint at or below it."""
    return lambda checkpoint, row_count, data_seed: steps[max(key for key in steps if key <= checkpoint)]


def by_rows(at_30, at_40):
    return lambda checkpoint, row_count, data_seed: (at_30 if row_count == 30 else at_40)(checkpoint, row_count, data_seed)


def early_spread(base, size):
    """Adds +size or -size by the data set at checkpoints 40 to 100. The 10 differences of d2 then have
    a sample standard deviation of size x sqrt(10 / 9) and a standard error of size / 3."""
    def number(checkpoint, row_count, data_seed):
        shift = (size if data_seed % 2 == 0 else -size) if 40 <= checkpoint <= 100 else 0.0
        return base(checkpoint, row_count, data_seed) + shift
    return number


C1_FOOD_SHARE = series({0: 0.0, 60: 0.9})                               # the sign turns at checkpoint 60
C1_FOOD_SCORE = series({0: 0.5, 20: 0.9, 40: 0.85, 60: 0.6, 80: 0.3})
# The base case: a HOLDS, b FAILS, reading 2 HOLDS, c NOT DECIDED, d HOLDS.
LABEL_BASE = {
    ("C1", "food"): {"share_negative": C1_FOOD_SHARE, "score": C1_FOOD_SCORE, "score_go_only": series({0: 0.6})},
    ("C1", "random"): {"mean_abs_advantage": series({0: 0.1}), "score": series({0: 0.5, 40: 0.7, 120: 0.55})},
    ("C2", "food"): {"share_negative": series({0: 0.9}), "score": series({0: 0.3})},
    ("C2", "random"): {"score": series({0: 0.55})},
    ("C3a", "food"): {"share_negative": series({0: 0.9}), "score": series({0: 0.9})},
    ("C3b", "food"): {"score": series({0: 0.95})},
    ("C3c", "food"): {"score": series({0: 0.4})},
}
B_HOLDS = {("C2", "food"): {"share_negative": series({0: 0.1}), "score": series({0: 0.9})},
           ("C2", "random"): {"score": series({0: 0.9})}}
C_HOLDS = {("C1", "food"): {"score_go_only": series({0: 0.9})}}


def label_mean_rows(changes=None):
    spec = {key: dict(measures) for key, measures in LABEL_BASE.items()}
    for key, measures in (changes or {}).items():
        spec.setdefault(key, {}).update(measures)
    rows = []
    for (condition, chooser), measures in spec.items():
        for row_count, data_seed, checkpoint in itertools.product((30, 40), range(10), check.SIGN_CHECKPOINTS):
            row = dict({name: None for name in check.MEAN_MEASURES}, condition=condition, rows=row_count,
                       chooser=chooser, data_seed=data_seed, checkpoint=checkpoint)
            row.update({name: number(checkpoint, row_count, data_seed) for name, number in measures.items()})
            rows.append(row)
    return rows


def label_report(changes=None, training_rows=()):
    return check.label_report_of(label_mean_rows(changes), list(training_rows))


def labels_of(changes=None):
    return label_report(changes)["labels"]


def one_training(data_seed, first_update, low_from, row_count=30):
    """The rows of one training of C1 with the food chooser. The score is 0.3 from checkpoint low_from."""
    return [{"condition": "C1", "rows": row_count, "chooser": "food", "data_seed": data_seed, "initial_seed": 0,
             "fold": 0, "checkpoint": checkpoint, "first_update_share_above_half": first_update,
             "score": 0.3 if low_from is not None and checkpoint >= low_from else 0.9}
            for checkpoint in check.SIGN_CHECKPOINTS]


class TestVariantSignLabels:
    def test_the_base_case(self):
        report = label_report()
        assert report["labels"] == {"a": "HOLDS", "b": "FAILS", "reading_2": "HOLDS", "c": "NOT DECIDED",
                                    "d": "HOLDS"}
        assert report["predictions"] == {
            "a1_part_1": True, "a1_part_2": True, "a2": True, "a3": True, "C3b": True, "C3c": False,
            "b1": False, "b2": False, "b3": False, "b4": True, "c1": False, "d1": True, "d2": True}
        facts = report["row_counts"][30]
        assert facts["a1_part_2_share"] == pytest.approx(0.9) and facts["a2_score_at_200"] == pytest.approx(0.3)
        assert facts["reading_2_turn_checkpoint"] == 60 and facts["reading_2_checkpoints_before_the_turn"] == [20, 40]
        assert facts["measurement_7"] == pytest.approx(0.9) and facts["c3a_says_nothing_about_c"] is False
        assert facts["d2_difference"] == pytest.approx(0.7 - 0.55)

    # Hypothesis a.

    def test_a_partly_when_c3a_is_below_0_80_at_both_row_counts(self):
        assert labels_of({("C3a", "food"): {"score": series({0: 0.4})}})["a"] == "PARTLY"

    def test_a_fails_by_the_first_way_no_turn_in_c1(self):
        report = label_report({("C1", "food"): {"share_negative": series({0: 0.0, 20: 0.1})}})
        assert report["labels"]["a"] == "FAILS" and report["labels"]["reading_2"] == "NOT DECIDED"

    def test_a_fails_by_the_second_way_c3b_below_0_80(self):
        assert labels_of({("C3b", "food"): {"score": series({0: 0.5})}})["a"] == "FAILS"

    def test_a_not_decided_with_the_two_ways_at_different_row_counts(self):
        no_turn = series({0: 0.0, 20: 0.1})
        changes = {("C1", "food"): {"share_negative": by_rows(no_turn, C1_FOOD_SHARE)},
                   ("C3b", "food"): {"score": by_rows(series({0: 0.95}), series({0: 0.5}))}}
        assert labels_of(changes)["a"] == "NOT DECIDED"

    def test_a_not_decided_with_a_way_at_one_row_count_only(self):
        changes = {("C1", "food"): {"share_negative": by_rows(series({0: 0.0, 20: 0.1}), C1_FOOD_SHARE)}}
        assert labels_of(changes)["a"] == "NOT DECIDED"
        assert labels_of({("C3b", "food"): {"score": by_rows(series({0: 0.95}), series({0: 0.5}))}})["a"] == "NOT DECIDED"

    def test_a_not_decided_with_c3a_at_0_80_or_more_at_one_row_count_only(self):
        changes = {("C3a", "food"): {"score": by_rows(series({0: 0.9}), series({0: 0.4}))}}
        assert labels_of(changes)["a"] == "NOT DECIDED"

    def test_a_not_decided_when_a2_fails_or_part_1_of_a1_fails(self):
        no_high_early_score = {("C1", "food"): {"score": series({0: 0.5, 20: 0.7, 80: 0.3})}}
        assert labels_of(no_high_early_score)["a"] == "NOT DECIDED"
        share_at_update_1 = {("C1", "food"): {"share_negative": series({0: 0.2, 60: 0.9})}}
        report = label_report(share_at_update_1)
        assert report["labels"]["a"] == "NOT DECIDED" and report["predictions"]["a1_part_1"] is False

    def test_a_not_decided_when_a_fails_way_is_met_and_part_1_of_a1_fails(self):
        # Owner decision 1 on the label code. The share is 0.1 at update 1 (above the limit of 0.05) and
        # 0.1 over updates 181 to 200 (the first way of FAILS, at both row counts).
        report = label_report({("C1", "food"): {"share_negative": series({0: 0.1})}})
        assert report["predictions"]["a1_part_1"] is False and report["predictions"]["a1_part_2"] is False
        assert report["labels"]["a"] == "NOT DECIDED"
        assert report["step_p2"]["rows"] == ["Each other case"]

    # Hypothesis b and reading 2.

    def test_b_holds_partly_and_fails(self):
        assert labels_of(B_HOLDS)["b"] == "HOLDS"
        assert labels_of({("C2", "food"): B_HOLDS[("C2", "food")]})["b"] == "PARTLY"
        b2_fails = {("C2", "food"): {"share_negative": series({0: 0.1}), "score": series({0: 0.5})}}
        assert labels_of(b2_fails)["b"] == "FAILS"
        one_row_count = {("C2", "food"): B_HOLDS[("C2", "food")],
                         ("C2", "random"): {"score": by_rows(series({0: 0.9}), series({0: 0.55}))}}
        assert labels_of(one_row_count)["b"] == "PARTLY"       # b3 holds only at both row counts

    def test_reading_2_fails_with_a_score_of_0_50_or_less_before_the_turn(self):
        low_before_the_turn = series({0: 0.5, 20: 0.9, 40: 0.45, 60: 0.6, 80: 0.3})
        report = label_report({("C1", "food"): {"score": low_before_the_turn}})
        assert report["labels"]["reading_2"] == "FAILS" and report["predictions"]["b4"] is False
        one_row_count = {("C1", "food"): {"score": by_rows(low_before_the_turn, C1_FOOD_SCORE)}}
        assert labels_of(one_row_count)["reading_2"] == "NOT DECIDED"

    def test_reading_2_not_decided_with_no_checkpoint_before_the_turn(self):
        report = label_report({("C1", "food"): {"share_negative": series({0: 0.0, 20: 0.9})}})
        assert report["labels"]["reading_2"] == "NOT DECIDED"
        assert report["row_counts"][30]["reading_2_checkpoints_before_the_turn"] == []
        assert report["predictions"]["b4"] is None

    def test_reading_2_not_decided_when_part_2_of_a1_fails_after_a_turn(self):
        turn_and_back = series({0: 0.0, 60: 0.9, 180: 0.2})
        assert labels_of({("C1", "food"): {"share_negative": turn_and_back}})["reading_2"] == "NOT DECIDED"

    # Hypothesis c.

    def test_c_holds_fails_and_not_decided(self):
        assert labels_of(C_HOLDS)["c"] == "HOLDS"
        near = series({0: 0.32})                      # 0.02 above the score of 0.3 at 200
        assert labels_of({("C1", "food"): {"score_go_only": near}})["c"] == "FAILS"
        one_row_count = {("C1", "food"): {"score_go_only": by_rows(near, series({0: 0.6}))}}
        assert labels_of(one_row_count)["c"] == "NOT DECIDED"

    def test_c_fails_when_c1_holds_and_the_two_scores_are_near(self):
        # Owner decision 2 on the label code. The go-only score 0.90 holds c1, and it is 0.02 above the
        # score with the preference, 0.88, which is within the limit of 0.05.
        changes = {("C1", "food"): {"score": series({0: 0.88}), "score_go_only": series({0: 0.9})}}
        report = label_report(changes)
        assert report["predictions"]["c1"] is True
        assert report["row_counts"][30]["c_scores_within_the_limit"] is True
        assert report["labels"]["c"] == "FAILS"

    # Hypothesis d and the limit of change 19.

    def test_d_fails_and_not_decided(self):
        assert labels_of({("C1", "random"): {"mean_abs_advantage": series({0: 0.4})}})["d"] == "FAILS"
        assert labels_of({("C1", "random"): {"score": series({0: 0.55})}})["d"] == "NOT DECIDED"
        one_row_count = {("C1", "random"): {"mean_abs_advantage": by_rows(series({0: 0.1}), series({0: 0.4}))}}
        assert labels_of(one_row_count)["d"] == "FAILS"        # d1 holds only at both row counts

    def test_the_limit_is_0_05_when_2_standard_errors_are_below_it(self):
        facts = label_report()["row_counts"][40]
        for name in ("a1_part_1_limit", "c_limit", "d2_limit"):
            assert facts[name]["standard_error"] == pytest.approx(0.0, abs=1e-12) and facts[name]["limit"] == 0.05

    def test_the_limit_is_2_standard_errors_when_they_are_above_0_05(self):
        # The 10 differences of d2 are 0.10 + 0.2 or 0.10 - 0.2. Their mean is 0.10. The standard error
        # is 0.2 / 3, so the limit is 0.4 / 3 = 0.133. d2 fails. With a limit of 0.05 it would hold.
        score = early_spread(series({0: 0.5, 40: 0.65, 120: 0.55}), 0.2)
        report = label_report({("C1", "random"): {"score": score}})
        facts = report["row_counts"][30]
        assert facts["d2_difference"] == pytest.approx(0.10)
        assert facts["d2_limit"]["standard_error"] == pytest.approx(0.2 / 3)
        assert facts["d2_limit"]["limit"] == pytest.approx(0.4 / 3)
        assert report["predictions"]["d2"] is False and report["labels"]["d"] == "NOT DECIDED"
        wider = early_spread(series({0: 0.5, 40: 0.7, 120: 0.55}), 0.2)        # a mean of 0.15, above 0.133
        assert labels_of({("C1", "random"): {"score": wider}})["d"] == "HOLDS"

    def test_the_standard_error_has_the_divisor_9_and_the_root_of_10(self):
        numbers = np.array([0.0, 0.3] * 5)       # sum of squares 10 x 0.15^2, deviation sqrt(0.225 / 9)
        limit = check.limit_of_change_19(numbers)
        assert limit["standard_error"] == pytest.approx(math.sqrt(0.225 / 9) / math.sqrt(10))
        assert limit["limit"] == pytest.approx(2 * math.sqrt(0.225 / 9) / math.sqrt(10))
        with pytest.raises(ControlMismatch):
            check.limit_of_change_19(np.zeros(9))

    # The table for step P2.

    def test_the_rows_of_the_base_case(self):
        step_p2 = label_report()["step_p2"]
        assert step_p2["rows"] == ["a HOLDS and b FAILS", "d HOLDS and a HOLDS"]
        assert step_p2["place"] == 2 and step_p2["advice"] == "Do not build with the variant"

    def test_two_rows_apply_together_and_the_first_place_holds(self):
        both = {**B_HOLDS, ("C1", "food"): {"score_go_only": series({0: 0.9})},
                ("C3a", "food"): {"score": by_rows(series({0: 0.9}), series({0: 0.4}))}}
        step_p2 = label_report(both)["step_p2"]            # a NOT DECIDED, b HOLDS, c HOLDS
        assert step_p2["rows"] == ["b HOLDS", "c HOLDS"]
        assert step_p2["place"] == 3 and step_p2["advice"] == "A build needs a check at the amount of data of a run"
        fails_and_holds = {**B_HOLDS, ("C3b", "food"): {"score": series({0: 0.5})}}
        step_p2 = label_report(fails_and_holds)["step_p2"]  # a FAILS, b HOLDS
        assert step_p2["rows"] == ["b HOLDS", "a FAILS"] and step_p2["place"] == 2

    def test_each_row_of_the_table_has_its_place(self):
        def place(a, b, c="NOT DECIDED", d="NOT DECIDED"):
            step_p2 = check.step_p2_rows({"a": a, "b": b, "reading_2": "HOLDS", "c": c, "d": d})
            return step_p2["rows"], step_p2["place"]

        assert place("HOLDS", "FAILS") == (["a HOLDS and b FAILS"], 2)
        assert place("PARTLY", "FAILS") == (["a PARTLY and b FAILS"], 2)
        assert place("NOT DECIDED", "PARTLY") == (["b PARTLY"], 2)
        assert place("NOT DECIDED", "HOLDS") == (["b HOLDS"], 3)
        assert place("FAILS", "FAILS") == (["a FAILS"], 2)
        assert place("NOT DECIDED", "FAILS", c="HOLDS") == (["c HOLDS"], 4)
        assert place("NOT DECIDED", "FAILS") == (["Each other case"], 5)
        assert place("NOT DECIDED", "FAILS", d="HOLDS") == (["Each other case"], 5)    # d gives no advice
        assert check.step_p2_rows({"a": "NOT DECIDED", "b": "FAILS", "reading_2": "FAILS", "c": "FAILS",
                                   "d": "FAILS"})["advice"] == "The author reports the table and the owner decides"

    # What is reported with no label.

    def test_the_count_of_trainings_of_a2(self):
        trainings = (one_training(0, 30, 40) + one_training(1, "none", 40) + one_training(2, 100, 60)
                     + one_training(3, 60, 60) + one_training(4, 30, None) + one_training(5, 30, 40, row_count=40))
        counts = label_report(training_rows=trainings)["a2_trainings"]
        assert counts[30] == {"trainings": 5, "turn_before_the_first_low_score": 2, "no_turn": 1, "no_low_score": 1}
        assert counts[40] == {"trainings": 1, "turn_before_the_first_low_score": 1, "no_turn": 0, "no_low_score": 0}

    def test_measurement_8_is_the_score_at_checkpoint_1_minus_the_score_at_checkpoint_0(self):
        report = label_report({("C3c", "food"): {"score": series({0: 0.5, 1: 0.52, 20: 0.4})}})
        differences = {(case["condition"], case["rows"], case["chooser"], case["data_seed"]): case["difference"]
                       for case in report["measurement_8"]}
        assert len(differences) == 7 * 2 * 10
        assert differences[("C3c", 30, "food", 4)] == pytest.approx(0.02)
        assert differences[("C1", 40, "food", 0)] == pytest.approx(0.0)

    def test_slot_0_and_food_80_are_reported_with_no_label(self):
        changes = {("C1", "slot_0"): {"score": series({0: 0.6})}, ("C2", "food_80"): {"score": series({0: 0.52})}}
        no_label = label_report(changes)["no_label"]
        assert [(entry["condition"], entry["chooser"], entry["rows"]) for entry in no_label] == [
            ("C1", "slot_0", 30), ("C1", "slot_0", 40), ("C2", "food_80", 30), ("C2", "food_80", 40)]
        assert all(entry["label"] == "none" for entry in no_label)
        assert no_label[0]["score_at_200"] == pytest.approx(0.6) and no_label[3]["score_at_200"] == pytest.approx(0.52)

    # The read of the two files.

    def test_the_rows_of_a_file_are_read_with_their_blanks(self, tmp_path):
        path = tmp_path / "training_rows.csv"
        header = ",".join(check.TRAINING_COLUMNS)
        blanks = ",".join([""] * (len(check.TRAINING_COLUMNS) - 10))      # the 7 measurements of an update
        path.write_text(header + "\nC1,30,food,4,2,3,20,0.75,0.5," + blanks + ",none\n"
                        + "C4,40,random,0,0,0,0,0.5,," + blanks + ",\n", encoding="utf-8")
        first, second = check.read_rows(str(path))
        assert (first["condition"], first["rows"], first["data_seed"], first["checkpoint"]) == ("C1", 30, 4, 20)
        assert first["score"] == 0.75 and first["share_negative"] is None
        assert first["first_update_share_above_half"] == "none"
        assert second["score_go_only"] is None and second["first_update_share_above_half"] is None

    def test_a_job_that_is_not_complete_gives_no_label(self):
        with pytest.raises(ControlMismatch, match="V4"):
            check.check_complete(label_mean_rows(), [])

    def test_a_missing_data_set_gives_no_label(self):
        rows = [row for row in label_mean_rows() if not (row["condition"] == "C3b" and row["data_seed"] == 9)]
        with pytest.raises(ControlMismatch):
            check.label_report_of(rows, [])
