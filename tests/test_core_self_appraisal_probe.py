"""The calculations of the Stage 2 appraisal probe (scripts/analysis/probe_core_self_appraisal.py)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_core_self_appraisal as probe


class TestFoodAhead:
    def test_a_step_is_positive_when_food_comes_inside_its_window(self):
        rewards = np.array([-0.01, -0.01, 1.0, -0.01, -0.01])
        labels, keep = probe.food_ahead(rewards, window=2)
        assert keep.tolist() == [True, True, True, False, False]
        assert labels[keep].tolist() == [1, 1, 0]

    def test_the_food_step_itself_does_not_count(self):
        labels, keep = probe.food_ahead(np.array([1.0, -0.01, -0.01]), window=1)
        assert labels[keep].tolist() == [0, 0]


class TestAuc:
    def test_closed_forms(self):
        labels = np.array([0, 0, 1, 1])
        assert probe.auc(np.array([1.0, 1.0, 1.0, 1.0]), labels) == 0.5
        assert probe.auc(np.array([0.0, 1.0, 2.0, 3.0]), labels) == 1.0
        assert probe.auc(np.array([3.0, 2.0, 1.0, 0.0]), labels) == 0.0

    def test_a_tie_counts_as_one_half(self):
        """Pairs (1 > 0), (1 = 1), (2 > 0), (2 > 1) give 3.5 of 4."""
        assert probe.auc(np.array([0.0, 1.0, 1.0, 2.0]), np.array([0, 0, 1, 1])) == pytest.approx(0.875)

    def test_one_class_raises(self):
        with pytest.raises(ValueError, match="both labels"):
            probe.auc(np.array([0.1, 0.2]), np.array([1, 1]))


class TestDesignA:
    def test_only_the_first_three_values_set_the_valence(self):
        """map_state cuts the 256 values to the 3 of the goal (models/core/qualia_mapper.py:72)."""
        row = np.zeros(256, dtype=np.float32)
        row[:3] = [1.0, -1.0, 1.0]
        other = row.copy()
        other[3:] = 5.0
        streams = probe.design_a_streams(np.stack([row, other]), np.array([0.5, 0.5]))
        assert streams["phen_valence"] == pytest.approx([1.0, 1.0], abs=1e-6)
        assert streams["a_valence"] == pytest.approx([0.6 * 0.5 + 0.4, 0.6 * 0.5 + 0.4], abs=1e-6)

    def test_dominance_is_three_tenths_of_tanh_of_the_length(self):
        row = np.zeros(256, dtype=np.float32)
        row[:3] = [-1.0, 1.0, -1.0]
        streams = probe.design_a_streams(row[None, :], np.array([0.0]))
        assert streams["phen_valence"][0] == pytest.approx(-1.0, abs=1e-6)
        assert streams["a_dominance"][0] == pytest.approx(0.3 * math.tanh(math.sqrt(3.0)), abs=1e-6)


class TestHeldOutRead:
    def test_a_feature_equal_to_the_label_is_read_perfectly_on_held_out_episodes(self):
        groups = np.repeat(np.arange(10), 20)
        labels = np.tile(np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)], 10)
        x = labels[:, None].astype(float)
        assert probe.auc(probe.held_out_scores(x, labels, groups), labels) == 1.0


def _write_episode(folder, steps=70, food_at=65):
    os.makedirs(folder)
    with open(os.path.join(folder, "steps.jsonl"), "w", encoding="utf-8") as handle:
        for t in range(steps):
            handle.write(json.dumps({"reward": 1.0 if t == food_at else -0.01,
                                     "internals": {"emotion": {"valence": 0.02 * t},
                                                   "value_estimate": None if t == 0 else float(t)}}) + "\n")
    broadcast = np.arange(steps * 256, dtype=np.float32).reshape(steps, 1, 256)
    np.savez(os.path.join(folder, "vectors.npz"), broadcast=broadcast,
             broadcast__steps=np.arange(steps, dtype=np.int32))


class TestReadEpisode:
    def test_rows_labels_and_shapes_follow_the_recorded_layout(self, tmp_path):
        """The recorder stores [steps, 1, 256] vectors with a `__steps` index (session_recorder.py)."""
        folder = str(tmp_path / "ep_0000")
        _write_episode(folder)
        rec = probe.read_episode(folder)
        assert rec["step"].tolist() == list(range(10))  # t + 60 < 70
        assert rec["label"].tolist() == [0] * 5 + [1] * 5  # food at 65 is inside t + 1 .. t + 60 from t = 5
        assert rec["broadcast"].shape == (10, 256)
        assert rec["broadcast"][3, 0] == 3 * 256
        assert np.isnan(rec["value_estimate"][0]) and rec["value_estimate"][4] == 4.0
        assert rec["reflex"][2] == pytest.approx(0.04)


class TestFollowUp:
    def test_near_food_marks_the_last_k_steps_before_the_first_food(self):
        rewards = np.full(20, -0.01)
        rewards[[10, 11, 15]] = 1.0
        steps, labels = probe.near_food(rewards, k=3)
        assert steps.tolist() == list(range(10))
        assert labels.tolist() == [0] * 7 + [1] * 3  # the first food at 10 is within 3 steps from t = 7

    def test_an_episode_without_food_gives_no_steps(self):
        steps, labels = probe.near_food(np.full(20, -0.01))
        assert len(steps) == 0 and len(labels) == 0

    def test_fold_aucs_do_not_mix_the_offsets_of_folds(self):
        """Each episode is ordered perfectly, at its own offset. Pooled, the offsets give 15 of 25 pairs."""
        groups = np.repeat(np.arange(5), 2)
        labels = np.tile([0, 1], 5)
        scores = labels + 10.0 * groups
        assert probe.fold_aucs(scores, labels, groups) == [1.0] * 5
        assert probe.auc(scores, labels) == pytest.approx(0.6)

    def test_read_approach_keeps_only_the_steps_before_the_first_food(self, tmp_path):
        _write_episode(str(tmp_path / "episodes" / "ep_0000"), steps=30, food_at=10)
        _write_episode(str(tmp_path / "episodes" / "ep_0001"), steps=30, food_at=-1)
        rec = probe.read_approach(str(tmp_path))
        assert rec["step"].tolist() == list(range(10))
        assert rec["label"].tolist() == [0] * 3 + [1] * 7  # APPROACH_K = 7
        assert set(rec["group"].tolist()) == {0}
        assert rec["broadcast"].shape == (10, 256)


class TestSeedRule:
    def test_pass_fail_and_undecided(self):
        assert probe.seed_rule(3, 3, 3) == "PASS"
        assert probe.seed_rule(1, 3, 3) == "FAIL"
        assert probe.seed_rule(2, 3, 3) == "UNDECIDED"
        assert probe.seed_rule(2, 3, 2) == "UNDECIDED"  # one VOID seed blocks a PASS


class TestVerdicts:
    @staticmethod
    def _run(phen_holds, dominance_holds, read_holds):
        return {"void": False,
                "streams": {"phen_valence": {"reading": 0.7 if phen_holds else 0.5, "null_p95": 0.6, "distinct": 9},
                            "a_dominance": {"reading": 0.2 if dominance_holds else 0.0, "null_p95": 0.1,
                                            "distinct": 9}},
                "broadcast_read": {"auc": 0.8 if read_holds else 0.55, "null_p95": 0.6, "step_index_auc": 0.7}}

    def test_three_holds_pass_and_two_failures_fail(self):
        held = {s: self._run(True, True, True) for s in "abc"}
        assert probe.verdicts(held)["design_A_CC1"] == "PASS"
        failed = {"a": self._run(False, True, False), "b": self._run(False, True, False), "c": self._run(True, True, True)}
        assert probe.verdicts(failed)["design_A_CC1"] == "FAIL"
        assert probe.verdicts(failed)["design_B_CC2"] == "FAIL"

    def test_a_read_below_the_step_index_does_not_hold(self):
        run = self._run(True, True, True)
        run["broadcast_read"]["step_index_auc"] = 0.9
        assert probe.verdicts({s: run for s in "abc"})["B2_seeds"] == 0

    def test_a_void_seed_is_neither_a_hold_nor_a_failure(self):
        runs = {"a": {"void": True}, "b": self._run(True, True, True), "c": self._run(False, True, True)}
        assert probe.verdicts(runs)["design_A_CC1"] == "UNDECIDED"
        assert probe.verdicts(runs)["void_seeds"] == 1
