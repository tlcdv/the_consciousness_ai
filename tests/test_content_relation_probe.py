"""The calculations of the content relation probe (scripts/analysis/probe_content_relation.py)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_content_relation as probe


class TestLabels:
    def test_seen_object_needs_exactly_one_object_in_view(self):
        assert probe.seen_object({"_truth_light_in_view": True, "_truth_decoy_in_view": False}) == "light"
        assert probe.seen_object({"_truth_light_in_view": False, "_truth_decoy_in_view": True}) == "decoy"
        assert probe.seen_object({"_truth_light_in_view": True, "_truth_decoy_in_view": True}) is None
        assert probe.seen_object({"_truth_light_in_view": False, "_truth_decoy_in_view": False}) is None

    def test_louder_source_follows_distance_and_level(self):
        near, far = [10.0, 0.0], [150.0, 0.0]
        assert probe.louder_source({"_truth_light_offset": near, "_truth_decoy_offset": far,
                                    "_truth_decoy_level": 1.0}) == "light"
        assert probe.louder_source({"_truth_light_offset": near, "_truth_decoy_offset": near,
                                    "_truth_decoy_level": 3.0}) == "decoy"
        assert probe.louder_source({"_truth_light_offset": near, "_truth_decoy_offset": near,
                                    "_truth_decoy_level": 1.0}) is None
        assert probe.louder_source({"_truth_light_offset": near}) is None

    def test_only_steps_whose_seen_object_is_louder_are_counted(self):
        record = {"seen": np.array([0, 0, 1, 1]), "louder": np.array([0, 1, 1, 0]), "v": np.arange(4)}
        assert probe.matched(record)["v"].tolist() == [0, 2]


def _record(episodes=10, per_episode=40, seed=0):
    """One identity direction in each sense, v[0] and a[0] = +-1 for the seen object, plus noise."""
    rng = np.random.default_rng(seed)
    seen = rng.integers(0, 2, episodes * per_episode)
    v = 0.05 * rng.standard_normal((len(seen), 8))
    a = 0.05 * rng.standard_normal((len(seen), 8))
    v[:, 0] += 2 * seen - 1
    a[:, 0] += 2 * seen - 1
    return {"v": v, "a": a, "seen": seen, "louder": seen.copy(), "group": np.repeat(np.arange(episodes), per_episode)}


class TestIdentityCheck:
    def test_a_vector_that_carries_the_object_holds(self):
        record = _record()
        check = probe.identity_check(record["v"], record["seen"], record["group"], np.random.default_rng(0))
        assert check["auc"] == pytest.approx(1.0) and check["holds"] is True

    def test_a_vector_that_carries_only_time_does_not_hold(self):
        """Each episode shows one object, and the first half shows the light. The episode number alone reads
        as well as a vector that is the episode number, so the time control stops it."""
        groups = np.repeat(np.arange(10), 20)
        seen = (groups >= 5).astype(int)
        x = groups.astype(float)[:, None]
        check = probe.identity_check(x, seen, groups, np.random.default_rng(0))
        assert check["auc"] == pytest.approx(check["episode_number_auc"])
        assert check["holds"] is False


class TestTwoChoice:
    def test_the_items_are_balanced_and_the_partners_are_valid(self):
        seen = np.array([0, 0, 0, 0, 1, 1, 1])
        items = probe.two_choice_items(seen, np.random.default_rng(0))
        chosen = [i for i, _, _ in items]
        assert sum(seen[chosen] == 0) == sum(seen[chosen] == 1) == 3
        for i, same, other in items:
            assert same != i and seen[same] == seen[i] and seen[other] != seen[i]

    def test_the_closed_form_of_r(self):
        v_train = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        a_train = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        scores = probe.relation_scores(v_train, a_train, np.array([[1.0], [1.0]]), np.array([[1.0], [-1.0]]))
        assert scores.tolist() == [1.0, -1.0]  # W = 1, r = v a

    def test_co_occurrence_picks_the_same_object_partner(self):
        assert probe.two_choice_reading(_record(), np.random.default_rng(1)) == pytest.approx(1.0)

    def test_breaking_the_pairing_removes_the_choice(self):
        record = _record()
        broken = [probe.two_choice_reading(record, np.random.default_rng(seed), broken=True) for seed in range(10)]
        assert np.mean(broken) < 0.75


class TestSmokeStop:
    def test_the_smaller_count_is_scaled_from_2_to_20_episodes(self):
        assert probe.smoke_stop({"light_seen": 4, "decoy_seen": 30}) is True  # 40 < 50
        assert probe.smoke_stop({"light_seen": 5, "decoy_seen": 30}) is False  # 50


def _write_episode(folder, infos):
    """steps.jsonl and vectors.npz as the session recorder writes them, [steps, 1, 256] with __steps."""
    os.makedirs(folder)
    with open(os.path.join(folder, "steps.jsonl"), "w", encoding="utf-8") as handle:
        for info in infos:
            handle.write(json.dumps({"internals": {"info_after_step": info}}) + "\n")
    count = len(infos)
    for_steps = np.arange(count, dtype=np.int32)
    np.savez(os.path.join(folder, "vectors.npz"),
             tectum_content=np.arange(count * 256, dtype=np.float32).reshape(count, 1, 256),
             tectum_content__steps=for_steps,
             audio_content=-np.arange(count * 256, dtype=np.float32).reshape(count, 1, 256),
             audio_content__steps=for_steps)


class TestReadEpisode:
    def test_step_t_takes_the_ground_truth_of_step_t_minus_1(self, tmp_path):
        light_seen = {"_truth_light_in_view": True, "_truth_decoy_in_view": False,
                      "_truth_light_offset": [10.0, 0.0], "_truth_decoy_offset": [150.0, 0.0],
                      "_truth_decoy_level": 1.0}
        both_seen = dict(light_seen, _truth_decoy_in_view=True)
        folder = str(tmp_path / "ep_0000")
        _write_episode(folder, [light_seen, both_seen, light_seen])
        record = probe.read_episode(folder)
        assert len(record["seen"]) == 1  # step 1 counts (truth of step 0); step 2 has both in view
        assert record["v"][0][0] == 256.0  # the tectum vector of step 1
        assert record["seen"].tolist() == [0] and record["louder"].tolist() == [0]


class TestVerdict:
    def test_v1_and_a1_must_both_hold(self):
        def run(v1, a1):
            return {"void": None, "V1": {"holds": v1}, "A1": {"holds": a1}}
        assert probe.verdict({"a": run(True, True), "b": run(True, True), "c": run(True, True)})[
            "identity_condition_of_B2"] == "PASS"
        assert probe.verdict({"a": run(False, True), "b": run(True, False), "c": run(True, True)})[
            "identity_condition_of_B2"] == "FAIL"
        assert probe.verdict({"a": {"void": "V0"}, "b": run(True, True), "c": run(True, True)})[
            "identity_condition_of_B2"] == "UNDECIDED"
