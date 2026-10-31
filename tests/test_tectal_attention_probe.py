"""The gate probe of the tectal attention gain (scripts/analysis/probe_tectal_attention.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.tectal_attention import attend, attention_field
from models.core.tectal_motor_map import cell_offsets
from scripts.analysis import diagnose_attention_modes as modes
from scripts.analysis import probe_tectal_attention as probe


def info(light, decoy):
    return {"_truth_light_offset": list(light), "_truth_decoy_offset": list(decoy)}


def cells_near(offset, count=5):
    """The `count` map cells whose room offset is nearest to `offset`."""
    dx, dy = cell_offsets()
    distance = np.hypot(dx - offset[0], dy - offset[1]).reshape(-1)
    return [int(i) for i in np.argsort(distance)[:count]]


class TestSelectedObject:
    def test_the_group_on_the_light_selects_the_light(self):
        assert probe.selected_object(cells_near((30, 0)), info((30, 0), (-30, 0))) == "light"

    def test_the_group_on_the_decoy_selects_the_decoy(self):
        assert probe.selected_object(cells_near((-30, 5)), info((30, 0), (-30, 5))) == "decoy"

    def test_no_group_selects_nothing(self):
        assert probe.selected_object([], info((30, 0), (-30, 0))) is None

    def test_the_direction_is_the_unit_offset_of_the_object(self):
        assert probe.direction(info((3, 4), (-1, 0)), "light").tolist() == pytest.approx([0.6, 0.8])


class TestScore:
    @staticmethod
    def data(signal, episodes=30, steps=20, seed=0):
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, episodes)
        y = np.repeat(np.stack([np.cos(angles), np.sin(angles)], 1), steps, axis=0)
        groups = np.repeat(np.arange(episodes), steps)
        x = rng.normal(size=(episodes * steps, 64))
        x[:, :2] += signal * y
        return x, y, groups

    def test_a_vector_that_carries_the_direction_scores_near_one(self):
        x, y, groups = self.data(signal=5.0)
        assert probe.direction_score(x, y, groups) > 0.9

    def test_a_vector_without_the_direction_scores_near_zero(self):
        x, y, groups = self.data(signal=0.0)
        assert abs(probe.direction_score(x, y, groups)) < 0.2

    def test_the_folds_hold_whole_episodes_out(self):
        groups = np.repeat(np.arange(10), 3)
        for train, test in probe.episode_folds(groups, folds=5):
            assert not set(groups[train]) & set(groups[test])


class TestChecks:
    def test_pass_fail_and_the_stated_thresholds(self):
        assert probe.SCORE == 0.5 and probe.NULL == 0.2
        link = {"content": [0.8] * 3, "broadcast": [0.7] * 3, "p": [0.001] * 3}
        nolink = {"content": [0.0] * 3, "broadcast": [0.1] * 3}
        holds = probe.checks(link, nolink)
        assert holds == {"A1": [True] * 3, "A2": [True] * 3, "A3": [True] * 3}
        nolink_high = {"content": [0.3] * 3, "broadcast": [0.1] * 3}
        assert probe.checks(link, nolink_high)["A3"] == [False] * 3


class TestDiagnosis:
    def test_the_map_ratio_reads_the_gain_of_an_attended_map(self):
        torch.manual_seed(0)
        before = torch.randn(1, 64, 16, 16)
        cells = [0, 17, 34, 200]
        attended = attend(before, attention_field(cells, 16, 1.5))
        assert modes.map_gain_ratio(attended.numpy(), before.numpy(), cells) == pytest.approx(1.5)

    def test_the_relative_change_is_the_ratio_of_the_norms(self):
        assert modes.relative(np.array([3.0, 4.0]), np.array([0.0, 10.0])) == pytest.approx(0.5)
