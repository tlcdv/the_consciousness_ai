"""The abstract model of the competing goals gate (scripts/analysis/model_competing_goals.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`. Expected values are closed forms computed in each
test. The miss rule follows the saved runs: an agent that misses its chosen object reaches the other
object, and no episode ends with no contact.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.value_gate import LEARNING_RATE
from scripts.analysis import model_competing_goals as model

DESIGN = model.Design(k=0.2, decoy=3.0, size=4.0, big=24.0, wait=20)
TRAVEL = 3
MEASURED = model.Measured(travel=np.array([TRAVEL]), quiet=np.array([0.4]))


def hyperbolic(size: float, n: int, k: float = DESIGN.k) -> float:
    return size / (1.0 + k * (n - 1))


class TestContact:
    def test_a_miss_in_a_free_episode_reaches_the_other_object(self):
        chosen, reached = np.array([model.LIGHT, model.DECOY]), np.array([False, False])
        contact = model.contact_object(chosen, reached, episode=model.FORCED)
        assert contact.tolist() == [model.DECOY, model.LIGHT]

    def test_a_miss_in_a_forced_episode_is_no_contact(self):
        chosen, reached = np.array([model.LIGHT, model.DECOY]), np.array([False, True])
        contact = model.contact_object(chosen, reached, episode=0)
        assert contact.tolist() == [-1, model.DECOY]

    def test_a_reached_object_is_the_chosen_one(self):
        chosen, reached = np.array([model.LIGHT, model.DECOY]), np.array([True, True])
        assert model.contact_object(chosen, reached, episode=model.FORCED).tolist() == chosen.tolist()


class TestEpisode:
    def test_a_miss_credits_the_food_of_the_other_object_to_the_chosen_value(self):
        v = np.zeros((200, 2))
        first = model.play_episode("benefit", DESIGN, MEASURED, model.FORCED, v, np.random.default_rng(0), reach=0.0)
        at_decoy, at_light = first == model.DECOY, first == model.LIGHT
        assert at_decoy.any() and at_light.any() and (at_decoy | at_light).all()
        # The light was chosen and the decoy was reached: the decoy's food at once goes to the light's value.
        assert v[at_decoy, model.LIGHT] == pytest.approx(LEARNING_RATE * hyperbolic(DESIGN.decoy, TRAVEL))
        assert (v[at_decoy, model.DECOY] == 0.0).all()
        # The decoy was chosen and the light was reached: the light's food after the wait goes to the decoy's value.
        assert v[at_light, model.DECOY] == pytest.approx(LEARNING_RATE * hyperbolic(DESIGN.big, TRAVEL + DESIGN.wait))
        assert (v[at_light, model.LIGHT] == 0.0).all()

    def test_a_miss_in_a_forced_episode_gives_no_food(self):
        v = np.zeros((50, 2))
        first = model.play_episode("benefit", DESIGN, MEASURED, 0, v, np.random.default_rng(0), reach=0.0)
        assert (first == -1).all() and (v == 0.0).all()

    def test_a_reached_light_gives_its_food_after_the_wait(self):
        v = np.zeros((50, 2))
        first = model.play_episode("benefit", DESIGN, MEASURED, 0, v, np.random.default_rng(0), reach=1.0)
        assert (first == model.LIGHT).all()
        assert v[:, model.LIGHT] == pytest.approx(LEARNING_RATE * hyperbolic(DESIGN.big, TRAVEL + DESIGN.wait))

    def test_the_binary_arm_gives_one_at_either_object(self):
        v = np.zeros((200, 2))
        model.play_episode("binary", DESIGN, MEASURED, model.FORCED, v, np.random.default_rng(0), reach=0.0)
        assert v.sum(axis=1) == pytest.approx(LEARNING_RATE)


class TestSalience:
    def test_the_quieter_salience_is_drawn_from_the_measured_values(self):
        measured = model.Measured(travel=np.array([TRAVEL]), quiet=np.array([0.3, 0.55]))
        salience = model.draw_salience(1, 400, np.random.default_rng(0), measured.quiet)
        assert (salience[:, model.LIGHT] == 1.0).all()
        assert set(salience[:, model.DECOY].tolist()) == {0.3, 0.55}

    def test_the_decoy_is_louder_in_even_episodes(self):
        salience = model.draw_salience(0, 10, np.random.default_rng(0), MEASURED.quiet)
        assert (salience[:, model.DECOY] == 1.0).all() and (salience[:, model.LIGHT] == 0.4).all()


class TestRecords:
    def test_travel_and_quieter_salience_come_from_the_gate_records(self, tmp_path):
        episodes = [{"salience": {"18": 1.0, "23": 0.31}, "steps": 4, "outcome": 1.0},
                    {"salience": {"18": 0.52, "23": 1.0}, "steps": 60, "outcome": 0.0}]
        (tmp_path / "value_gate.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
        measured, md5 = model.load_measured([str(tmp_path)])
        assert measured.travel.tolist() == [4]
        assert measured.quiet.tolist() == [0.31, 0.52]
        assert len(md5) == 1

    def test_a_record_with_one_candidate_raises(self, tmp_path):
        episodes = [{"salience": {"18": 1.0}, "steps": 4, "outcome": 1.0}]
        (tmp_path / "value_gate.json").write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
        with pytest.raises(ValueError):
            model.load_measured([str(tmp_path)])
