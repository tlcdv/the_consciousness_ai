"""The calculations of the read of the broadcast at the choice step (scripts/analysis/check_choice_read.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import check_choice_read as check
from scripts.analysis import check_workspace_priority as priority


def binding_step(band, groups):
    return {"band": band, "groups": groups, "salience": {}}


def balanced_labels(count):
    return np.arange(count) % 2


def one_row_each(count):
    return np.arange(count)


class TestChoiceStep:
    def test_same_step_as_the_priority_check_on_a_built_record(self):
        steps = [binding_step(None, {}), binding_step(23, {"23": [], "18": []}),
                 binding_step(23, {"23": [], "18": [4]}), binding_step(23, {"23": [1], "18": [4]})]
        assert check.find_choice_step(steps) == priority.find_choice_step(steps) == 2

    def test_an_episode_without_a_group_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.find_choice_step([binding_step(None, {}), binding_step(23, {"23": []})])


class TestLabel:
    def test_a_quiet_decoy_gives_1_and_a_loud_decoy_gives_0(self):
        assert check.label_of_level(0.5) == 1
        assert check.label_of_level(3.0) == 0

    def test_a_level_of_1_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.label_of_level(1.0)


class TestFolds:
    @pytest.mark.parametrize("count,folds", [(40, 5), (60, 5), (32, 4), (48, 4)])
    def test_each_fold_has_both_labels_in_equal_parts(self, count, folds):
        labels = balanced_labels(count)
        fold_of = check.make_folds(labels, folds, seed=0)
        for fold in range(folds):
            held = labels[fold_of == fold]
            assert (held == 1).sum() == (held == 0).sum() == count // (2 * folds)

    def test_the_same_seed_gives_the_same_folds(self):
        labels = balanced_labels(40)
        assert np.array_equal(check.make_folds(labels, 5, seed=0), check.make_folds(labels, 5, seed=0))

    def test_too_few_episodes_of_a_label_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.make_folds(np.array([1, 1, 1, 1, 1, 0, 0]), 5, seed=0)


class TestAuc:
    def test_a_full_separation_is_1_and_its_reverse_is_0(self):
        labels = np.array([0, 0, 1, 1])
        assert check.auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
        assert check.auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0

    def test_equal_scores_are_one_half(self):
        assert check.auc(np.zeros(6), np.array([0, 1, 0, 1, 0, 1])) == 0.5

    def test_a_known_count_of_ordered_pairs(self):
        # Pairs (positive, negative): (0.4, 0.1) and (0.4, 0.35) are ordered, (0.3, 0.1) is ordered,
        # (0.3, 0.35) is not. So 3 of 4.
        assert check.auc(np.array([0.1, 0.35, 0.4, 0.3]), np.array([0, 0, 1, 1])) == 0.75

    def test_one_label_only_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.auc(np.array([0.1, 0.2]), np.array([1, 1]))


class TestStandardize:
    def test_training_rows_get_mean_0_and_deviation_1(self):
        rng = np.random.default_rng(3)
        train, test = rng.normal(5.0, 2.0, (30, 4)), rng.normal(5.0, 2.0, (6, 4))
        scaled_train, _ = check.standardize(train, test)
        assert np.allclose(scaled_train.mean(axis=0), 0.0)
        assert np.allclose(scaled_train.std(axis=0), 1.0)

    def test_a_dimension_with_no_spread_is_set_to_0(self):
        train = np.column_stack([np.arange(5.0), np.full(5, 7.0)])
        test = np.array([[1.0, 9.0]])
        scaled_train, scaled_test = check.standardize(train, test)
        assert np.all(scaled_train[:, 1] == 0.0)
        assert scaled_test[0, 1] == 0.0


class TestRidge:
    def test_same_weights_as_the_normal_equations(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(20, 6)), rng.normal(size=20)
        x_new = rng.normal(size=(5, 6))
        for column, penalty in enumerate(check.PENALTIES):
            weights = np.linalg.solve(x.T @ x + penalty * np.eye(6), x.T @ (y - y.mean()))
            assert np.allclose(check.ridge_predictions(x, y, x_new, check.PENALTIES)[:, column], x_new @ weights)

    def test_more_dimensions_than_rows(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(8, 30)), rng.normal(size=8)
        weights = np.linalg.solve(x.T @ x + 10.0 * np.eye(30), x.T @ (y - y.mean()))
        column = check.PENALTIES.index(10.0)
        assert np.allclose(check.ridge_predictions(x, y, x, check.PENALTIES)[:, column], x @ weights)


class TestRead:
    # The three tests of a planted signal and the test of noise follow the correction of 2026-11-16
    # in docs/decisions/2026_11_15_workspace_choice_read.md, written before the first run.

    def test_a_planted_signal_in_16_of_256_dimensions_scores_high(self):
        rng = np.random.default_rng(6)
        labels = balanced_labels(40)
        rows = rng.normal(size=(40, 256))
        rows[:, 17:33] += 6.0 * labels[:, None]
        reading = check.read_score(rows, one_row_each(40), labels)
        assert reading["score"] >= 0.95
        assert len(reading["fold_aucs"]) == check.OUTER_FOLDS == len(reading["penalties"])
        assert all(penalty in check.PENALTIES for penalty in reading["penalties"])

    def test_a_planted_signal_in_one_of_12_dimensions_scores_high(self):
        rng = np.random.default_rng(6)
        labels = balanced_labels(40)
        rows = rng.normal(size=(40, 12))
        rows[:, 5] += 6.0 * labels
        assert check.read_score(rows, one_row_each(40), labels)["score"] >= 0.95

    def test_the_known_limit_of_one_dimension_among_256(self):
        # The read does not find a label that only one of 256 dimensions holds, at any strength.
        # This test pins the limit. If it fails, the read changed and the rules need a new look.
        scores = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            labels = balanced_labels(40)
            rows = rng.normal(size=(40, 256))
            rows[:, 0] += 6.0 * labels
            scores.append(check.read_score(rows, one_row_each(40), labels)["score"])
        assert np.mean(scores) < 0.90

    def test_pure_noise_with_60_rows_is_above_its_null_in_3_of_20_cases_at_most(self):
        # The null is a 95th percentile, so about 1 case of 20 is above it by chance.
        labels, above = balanced_labels(60), 0
        for seed in range(20):
            rows = np.random.default_rng(100 + seed).normal(size=(60, 256))
            score = check.read_score(rows, one_row_each(60), labels)["score"]
            null = check.null_scores(rows, one_row_each(60), labels, count=40, seed=1)
            above += score > np.percentile(null, check.NULL_PERCENTILE)
        assert above <= 3

    def test_the_read_is_the_same_on_a_second_call(self):
        rng = np.random.default_rng(8)
        labels, rows = balanced_labels(40), rng.normal(size=(40, 12))
        first = check.read_score(rows, one_row_each(40), labels)
        assert first == check.read_score(rows, one_row_each(40), labels)

    def test_all_rows_of_an_episode_are_held_out_together(self):
        # Each episode has 3 rows that carry the number of the episode and nothing of the label.
        # A read that divides rows of one episode between training and test would find the label
        # through the episode. With whole episodes held out the score stays near one half.
        rng = np.random.default_rng(9)
        labels = rng.permutation(balanced_labels(40))
        episode_of_row = np.repeat(np.arange(40), 3)
        rows = np.zeros((120, 40))
        rows[np.arange(120), episode_of_row] = 1.0
        assert check.read_score(rows, episode_of_row, labels)["score"] <= 0.75

    def test_the_null_keeps_the_count_of_each_label(self):
        labels = balanced_labels(40)
        orders = check.label_orders(labels, count=5, seed=1)
        assert all(order.sum() == 20 for order in orders)
        assert any(not np.array_equal(order, labels) for order in orders)


def passing_reading(**changes):
    reading = {"broadcast": 0.9, "broadcast_null": 0.7, "audio": 0.95, "audio_null": 0.7, "episode_number": 0.5}
    reading.update(changes)
    return reading


class TestRunLabel:
    def test_holds(self):
        assert check.run_label(passing_reading()) == "HOLDS"

    def test_the_threshold_of_0_80_is_inside_holds(self):
        assert check.run_label(passing_reading(broadcast=0.80)) == "HOLDS"

    def test_at_null_when_the_score_is_not_above_the_null(self):
        assert check.run_label(passing_reading(broadcast=0.7)) == "AT NULL"

    def test_between_when_above_the_null_and_below_0_80(self):
        assert check.run_label(passing_reading(broadcast=0.75)) == "BETWEEN"

    def test_between_when_not_above_the_episode_number(self):
        assert check.run_label(passing_reading(episode_number=0.92)) == "BETWEEN"

    def test_void_when_the_reference_is_below_0_90(self):
        assert check.run_label(passing_reading(audio=0.89)) == "VOID"

    def test_void_when_the_reference_is_not_above_its_null(self):
        assert check.run_label(passing_reading(audio=0.95, audio_null=0.96)) == "VOID"


class TestVerdict:
    def test_holds_needs_3_of_3_in_each_group(self):
        assert check.verdict({"a": ["HOLDS"] * 3, "b": ["HOLDS"] * 3}) == "HOLDS"
        assert check.verdict({"a": ["HOLDS"] * 3, "b": ["HOLDS", "HOLDS", "BETWEEN"]}) == "UNDECIDED"

    def test_failed_needs_3_of_3_at_null_in_each_group(self):
        assert check.verdict({"a": ["AT NULL"] * 3, "b": ["AT NULL"] * 3}) == "FAILED"
        assert check.verdict({"a": ["AT NULL"] * 3, "b": ["HOLDS"] * 3}) == "UNDECIDED"

    def test_one_void_run_makes_the_check_void(self):
        assert check.verdict({"a": ["HOLDS"] * 3, "b": ["HOLDS", "HOLDS", "VOID"]}) == "VOID"
