"""The second count of the first object (scripts/analysis/count_first_object.py).

`docs/decisions/2026_11_08_competing_goals_gate.md`, Gate section. The second count has the definition
of the first count (the first object closer than 60 room pixels in the first 60 steps, the nearer one
on a tie) in code that imports neither the reversal probe nor the gate probe. Expected values are
counted in each test.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import count_first_object as second
from scripts.analysis.probe_reversal import first_object

FAR = ([300.0, 0.0], [0.0, 300.0])


def record(light, decoy):
    return {"internals": {"info_after_step": {"_truth_light_offset": light, "_truth_decoy_offset": decoy}}}


class TestDefinition:
    def test_the_first_object_closer_than_60_pixels(self):
        assert second.first_contact([FAR, FAR, ([30.0, 40.0], [200.0, 0.0])]) == "light"
        assert second.first_contact([FAR, ([200.0, 0.0], [0.0, -59.0]), ([10.0, 0.0], [200.0, 0.0])]) == "decoy"

    def test_no_contact_gives_none(self):
        assert second.first_contact([FAR] * 60) is None

    def test_a_distance_of_60_is_not_a_contact(self):
        assert second.first_contact([([60.0, 0.0], [300.0, 0.0])]) is None
        assert second.first_contact([([59.99, 0.0], [300.0, 0.0])]) == "light"

    def test_a_contact_after_step_60_does_not_count(self):
        assert second.first_contact([FAR] * 60 + [([1.0, 0.0], [300.0, 0.0])]) is None
        assert second.first_contact([FAR] * 59 + [([1.0, 0.0], [300.0, 0.0])]) == "light"

    def test_two_objects_in_contact_give_the_nearer_one(self):
        assert second.first_contact([([50.0, 0.0], [0.0, 20.0])]) == "decoy"
        assert second.first_contact([([0.0, 20.0], [50.0, 0.0])]) == "light"

    def test_the_limits_are_those_of_the_rule(self):
        assert (second.CONTACT, second.LIMIT) == (60.0, 60)


class TestSeparateCode:
    def test_the_module_imports_no_probe_and_no_numpy(self):
        with open(second.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for name in ("probe_reversal", "probe_competing_goals", "numpy"):
            assert name not in source

    def test_it_agrees_with_the_first_count_on_500_random_episodes(self):
        rng = random.Random(7)
        for _ in range(500):
            offsets = [([rng.uniform(-150, 150), rng.uniform(-150, 150)],
                        [rng.uniform(-150, 150), rng.uniform(-150, 150)]) for _ in range(70)]
            assert second.first_contact(offsets) == first_object([record(l, d) for l, d in offsets])


class TestRunFolder:
    def test_one_entry_for_each_episode_folder_in_order(self, tmp_path):
        episodes = [[FAR, ([10.0, 0.0], [300.0, 0.0])], [FAR, FAR], [([300.0, 0.0], [0.0, 5.0])]]
        for index, offsets in enumerate(episodes):
            folder = tmp_path / "episodes" / ("ep_%05d" % index)
            folder.mkdir(parents=True)
            with open(folder / "steps.jsonl", "w", encoding="utf-8") as handle:
                for light, decoy in offsets:
                    handle.write(json.dumps(record(light, decoy)) + "\n")
        assert second.count_run(str(tmp_path)) == ["light", None, "decoy"]
