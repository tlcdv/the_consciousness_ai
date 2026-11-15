"""The report of the long smoke run of the competing goals gate (scripts/analysis/probe_training_values.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`, the rule of the long smoke run. The run holds when
`metrics.csv` has 12000 rows and every recorded reward, loss and gradient norm is finite. Expected
values are counted in each test.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_training_values as probe


def step(reward=1.0, pred_loss=0.5, topo_loss=-0.2, tectum=2.0, gate=None, value=0.1):
    return {"reward": reward,
            "internals": {"rewards": {"env_reward": reward, "reward_for_action_core": reward},
                          "losses": {"pred_loss": pred_loss, "topo_loss": topo_loss},
                          "gradient_norms": {"tectum": tectum, "gate": gate},
                          "value_estimate": value}}


class TestKinds:
    def test_one_step_gives_the_reward_each_loss_each_gradient_norm_and_the_value_estimate(self):
        assert probe.step_values(step()) == {"reward": 1.0, "loss pred_loss": 0.5, "loss topo_loss": -0.2,
                                             "gradient tectum": 2.0, "value_estimate": 0.1}

    def test_a_value_that_the_step_did_not_compute_is_left_out(self):
        assert "gradient gate" not in probe.step_values(step(gate=None))
        assert "loss pred_loss" not in probe.step_values(step(pred_loss=None))


class TestBlocks:
    def test_the_largest_absolute_value_of_each_kind_in_each_block(self):
        episodes = [[step(reward=3.0, topo_loss=-0.9), step(reward=32.0, tectum=7.0)],
                    [step(reward=-0.01, pred_loss=1500.0)],
                    [step(reward=4.0, value=-150.0)]]
        blocks = probe.block_maxima(episodes, block=2)
        assert [b["episodes"] for b in blocks] == [[0, 1], [2, 2]]
        assert blocks[0]["largest"] == {"reward": 32.0, "loss pred_loss": 1500.0, "loss topo_loss": 0.9,
                                        "gradient tectum": 7.0, "value_estimate": 0.1}
        assert blocks[1]["largest"]["value_estimate"] == 150.0 and blocks[1]["largest"]["reward"] == 4.0

    def test_sixty_episodes_give_six_blocks_of_ten(self):
        blocks = probe.block_maxima([[step()] for _ in range(60)])
        assert [b["episodes"] for b in blocks] == [[0, 9], [10, 19], [20, 29], [30, 39], [40, 49], [50, 59]]

    def test_a_kind_that_no_step_of_a_block_computed_is_absent_from_that_block(self):
        blocks = probe.block_maxima([[step(pred_loss=None)], [step()]], block=1)
        assert "loss pred_loss" not in blocks[0]["largest"] and blocks[1]["largest"]["loss pred_loss"] == 0.5


class TestFinite:
    def test_finite_values_give_no_entry(self):
        assert probe.not_finite([[step(), step(reward=32.0)]], [1.0, 32.0]) == []

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_each_value_that_is_not_finite_is_named_with_its_episode_and_step(self, bad):
        found = probe.not_finite([[step()], [step(), step(pred_loss=bad)]], [0.0])
        assert found == [{"episode": 1, "step": 1, "kind": "loss pred_loss"}]
        assert probe.not_finite([[step(tectum=bad)]], [0.0]) == [{"episode": 0, "step": 0, "kind": "gradient tectum"}]
        assert probe.not_finite([[step(reward=bad)]], [0.0])[0]["kind"] == "reward"

    def test_a_reward_of_the_metrics_file_that_is_not_finite_is_named_with_its_row(self):
        assert probe.not_finite([[step()]], [0.0, float("nan")]) == [{"metrics_row": 1, "kind": "reward"}]


class TestRule:
    def test_the_run_holds_with_12000_rows_and_only_finite_values(self):
        assert probe.holds(12000, []) is True

    def test_fewer_rows_or_a_value_that_is_not_finite_fails(self):
        assert probe.holds(11999, []) is False
        assert probe.holds(12000, [{"episode": 3, "step": 7, "kind": "loss pred_loss"}]) is False

    def test_the_limits_are_those_of_the_rule(self):
        assert (probe.EPISODES, probe.STEPS, probe.BLOCK) == (60, 200, 10)
