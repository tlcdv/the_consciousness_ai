"""The calculations of the workspace priority check (scripts/analysis/check_workspace_priority.py)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.value_gate import ValueGate
from scripts.analysis import check_workspace_priority as check


def binding_step(band, groups, salience):
    return {"band": band, "groups": groups, "salience": salience}


class TestChoiceStep:
    def test_first_step_with_a_band_and_a_group_with_cells(self):
        steps = [binding_step(None, {}, {}), binding_step(23, {"23": [], "18": []}, {}),
                 binding_step(23, {"23": [], "18": [4]}, {}), binding_step(23, {"23": [1], "18": [4]}, {})]
        assert check.find_choice_step(steps) == 2

    def test_an_episode_without_a_group_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.find_choice_step([binding_step(None, {}, {}), binding_step(23, {"23": []}, {})])


class TestWorkspaceTerm:
    BANDS = ["23", "18"]

    def test_audio_winner_marks_the_loudest_band_only(self):
        term = check.workspace_term({"ignited": True, "winner": "audio"}, 23, self.BANDS)
        assert term == {"23": 1.0, "18": 0.0}

    def test_vision_winner_gives_no_term(self):
        assert check.workspace_term({"ignited": True, "winner": "vision"}, 23, self.BANDS) == {"23": 0.0, "18": 0.0}

    def test_no_ignition_gives_no_term(self):
        assert check.workspace_term({"ignited": False, "winner": "audio"}, 23, self.BANDS) == {"23": 0.0, "18": 0.0}

    def test_a_loudest_band_that_is_no_candidate_gets_nothing(self):
        assert check.workspace_term({"ignited": True, "winner": "audio"}, 30, self.BANDS) == {"23": 0.0, "18": 0.0}

    def test_a_record_without_the_fields_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.workspace_term({"winner": "audio"}, 23, self.BANDS)


class TestChoice:
    def test_equal_scores_give_equal_probabilities(self):
        assert check.choice_probabilities([1.0, 1.0]).tolist() == [0.5, 0.5]

    def test_a_score_gap_of_one_temperature_gives_the_logistic_of_one(self):
        p = check.choice_probabilities([1.0, 0.75])
        assert p[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), abs=1e-12)

    def test_one_candidate_has_probability_one(self):
        assert check.choice_probabilities([0.3]).tolist() == [1.0]

    def test_the_draw_selects_by_the_cumulative_sum(self):
        p = np.array([0.3, 0.7])
        assert check.pick_index(p, 0.29) == 0
        assert check.pick_index(p, 0.3) == 1
        assert check.pick_index(p, 0.99) == 1

    def test_the_term_moves_a_draw_between_the_two_probabilities(self):
        # loud 1.0 against quiet 0.9: P(loud) = logistic(0.4) = 0.5987. With the term it is logistic(4.4) = 0.9879.
        bands, priorities, term = ["23", "18"], [1.0, 0.9], {"23": 1.0, "18": 0.0}
        assert check.predict_choice(bands, priorities, term, weight=0.0, draw=0.7) == "18"
        assert check.predict_choice(bands, priorities, term, weight=1.0, draw=0.7) == "23"
        assert check.predict_choice(bands, priorities, term, weight=1.0, draw=0.99) == "18"


class TestFit:
    def test_a_linear_function_of_the_salience_gives_one(self):
        salience = np.array([1.0, 0.4, 1.0, 0.2, 1.0, 0.33])
        assert check.r_squared(salience, 2.0 * salience + 1.0) == pytest.approx(1.0, abs=1e-12)

    def test_a_constant_term_raises_and_gives_no_number(self):
        with pytest.raises(ValueError):
            check.r_squared(np.array([1.0, 0.4, 1.0, 0.2]), np.zeros(4))

    def test_a_constant_salience_raises(self):
        with pytest.raises(ValueError):
            check.r_squared(np.ones(4), np.array([1.0, 0.0, 1.0, 0.0]))

    def test_closed_form_with_a_fixed_quiet_salience(self):
        # 10 episodes, audio wins in 4, so q = 0.4. The identity of the rules file gives q / (2 - q) = 0.25.
        salience, term = [], []
        for episode in range(10):
            salience += [1.0, 0.4]
            term += [1.0 if episode < 4 else 0.0, 0.0]
        assert check.r_squared(np.array(salience), np.array(term)) == pytest.approx(0.25, abs=1e-12)


class TestValues:
    def test_values_follow_the_update_rule_of_the_gate(self):
        episodes = [{"choice": 23, "outcome": 1.0}, {"choice": 23, "outcome": 0.0}, {"choice": 18, "outcome": 1.0}]
        before = check.replay_values(episodes, learning_rate=0.3)
        assert before[0] == {}
        assert before[1] == {"23": pytest.approx(0.3)}
        assert before[2] == {"23": pytest.approx(0.21)}


class TestVerdict:
    @staticmethod
    def run(group, rule_1, rule_2):
        return {"group": group, "rule_1": rule_1, "rule_2": rule_2}

    def test_supported_needs_both_rules_at_every_seed_of_each_group(self):
        runs = [self.run(g, True, True) for g in ("a", "a", "a", "b", "b", "b")]
        assert check.verdict(runs) == "SUPPORTED"

    def test_one_failed_rule_at_one_seed_excludes(self):
        runs = [self.run(g, True, True) for g in ("a", "a", "a", "b", "b")] + [self.run("b", False, True)]
        assert check.verdict(runs) == "EXCLUDED"

    def test_a_group_with_fewer_than_three_seeds_raises(self):
        with pytest.raises(check.ControlMismatch):
            check.verdict([self.run("a", True, True)] * 2)


def write_run(folder, seed, episodes=6, audio_wins=(), steps=4):
    """A saved run in the format of the training loop, with choices made by the real ValueGate."""
    gate, history = ValueGate(seed + 1000003), []
    for episode in range(episodes):
        loud, quiet = ("23", "18") if episode % 2 == 0 else ("18", "23")
        salience = {loud: 1.0, quiet: 0.9}
        history += [binding_step(None, {}, {}), binding_step(int(loud), {loud: [], quiet: []}, salience)]
        history += [binding_step(int(loud), {loud: [1], quiet: [2]}, salience)] * (steps - 2)
        gate.choose({int(b): s for b, s in salience.items()})
        gate.observe_reward(1.0 if episode % 3 == 0 else 0.0)
        gate.new_episode()
        rows = [{"step": t, "ignited": True, "winner": "audio" if episode in audio_wins else "vision"}
                for t in range(steps)]
        os.makedirs(folder / "episodes" / f"ep_{episode:04d}")
        (folder / "episodes" / f"ep_{episode:04d}" / "steps.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (folder / "value_gate.json").write_text(json.dumps({"learning_rate": 0.3, "episodes": gate.episodes}))
    (folder / "map_binding.json").write_text(json.dumps({"steps": len(history), "history": history}))
    (folder / "session.json").write_text(json.dumps({"run": {"seed": seed, "max_steps": steps}}))


class TestSavedRun:
    def test_the_replay_matches_the_real_gate_and_counts_the_audio_winners(self, tmp_path):
        write_run(tmp_path, seed=7, episodes=6, audio_wins=(0, 1, 2))
        report = check.check_run(str(tmp_path), expected_episodes=6)
        assert report["episodes"] == 6
        assert report["control_a_matches"] == 6
        assert report["audio_share"] == pytest.approx(0.5)
        # The quiet salience is fixed at 0.9, so the closed form holds: q / (2 - q) with q = 0.5.
        assert report["r_squared"] == pytest.approx(1.0 / 3.0, abs=1e-12)
        assert report["rule_2"] is True

    def test_no_audio_winner_gives_an_undefined_fit_and_no_change(self, tmp_path):
        write_run(tmp_path, seed=7, episodes=6, audio_wins=())
        report = check.check_run(str(tmp_path), expected_episodes=6)
        assert report["r_squared"] is None
        assert report["rule_2"] is False
        assert report["changed"][1.0] == 0
        assert report["rule_1"] is False

    def test_a_changed_recorded_choice_fails_control_a(self, tmp_path):
        write_run(tmp_path, seed=7)
        path = tmp_path / "value_gate.json"
        record = json.loads(path.read_text())
        record["episodes"][0]["choice"] = 18 if record["episodes"][0]["choice"] == 23 else 23
        path.write_text(json.dumps(record))
        with pytest.raises(check.ControlMismatch):
            check.check_run(str(tmp_path), expected_episodes=6)

    def test_a_wrong_episode_count_is_void(self, tmp_path):
        write_run(tmp_path, seed=7)
        with pytest.raises(check.ControlMismatch):
            check.check_run(str(tmp_path), expected_episodes=40)

    def test_a_changed_salience_fails_control_b(self, tmp_path):
        write_run(tmp_path, seed=7)
        path = tmp_path / "map_binding.json"
        record = json.loads(path.read_text())
        record["history"][2] = dict(record["history"][2], salience={"23": 1.0, "18": 0.5})
        path.write_text(json.dumps(record))
        with pytest.raises(check.ControlMismatch):
            check.check_run(str(tmp_path), expected_episodes=6)
