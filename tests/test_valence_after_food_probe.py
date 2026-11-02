"""The calculations of the valence after food probe (scripts/analysis/probe_valence_after_food.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.emotion.affective_modulator import AffectiveModulator
from scripts.analysis import probe_valence_after_food as probe

NEUTRAL_BODY = {"energy": 1.0, "fatigue": 0.0, "damage": 0.0}  # no interoceptive PAD at all


def step(final_bids, valence=-1.0):
    """A step record shaped as the recorder writes it. The last settle cycle starts from final_bids."""
    modulator = AffectiveModulator({"ablate_existence_bias": False})
    pad = {"valence": valence, "arousal": 0.5, "dominance": 0.0}
    logged, _ = modulator.modulate(final_bids, pad, NEUTRAL_BODY)
    raw = dict(final_bids, audio=0.0)  # the reentrant cycles changed audio, so raw differs
    return {"raw_bids": raw, "interoception": NEUTRAL_BODY,
            "internals": {"emotion": pad, "modulated_bids": logged, "settle": {"final_bids": final_bids}}}


class TestAfterFood:
    def test_the_steps_whose_previous_step_had_food(self):
        steps = [{"reward": r} for r in (-0.01, 1.0, 1.0, -0.01, -0.01)]
        assert probe.after_food_steps(steps) == [2, 3]


class TestLeader:
    def test_a_unique_largest_bid_and_a_tie(self):
        assert probe.leader({"vision": 1.0, "audio": 0.2}) == "vision"
        assert probe.leader({"vision": 1.0, "audio": 1.0, "body": 0.1}) == "tie"


class TestCompareStep:
    def test_the_threat_set_rises_by_the_valence_gain_and_vision_stays_at_its_limit(self):
        """THREAT_MODULES is body, vision and audio. valence_gain is 0.15. Bids stop at 1.0."""
        raw = {"vision": 1.0, "audio": 0.5, "memory": 0.1, "body": 0.05, "semantic": 0.0}
        compared = probe.compare_step(AffectiveModulator({"ablate_existence_bias": False}), step(raw))
        assert compared["reproduced"] is True
        assert compared["changes"]["audio"] == pytest.approx(0.15)
        assert compared["changes"]["body"] == pytest.approx(0.15)
        assert compared["changes"]["vision"] == pytest.approx(0.0)
        assert compared["changes"]["memory"] == compared["changes"]["semantic"] == 0.0
        assert compared["leader_changed"] is False

    def test_a_tie_made_by_the_valence_is_a_change_of_leader(self):
        raw = {"vision": 1.0, "audio": 0.9, "memory": 0.1, "body": 0.05, "semantic": 0.0}
        compared = probe.compare_step(AffectiveModulator({"ablate_existence_bias": False}), step(raw))
        assert compared["leader_changed"] is True

    def test_a_logged_bid_from_another_cycle_is_not_reproduced(self):
        raw = {"vision": 1.0, "audio": 0.5, "memory": 0.1, "body": 0.05, "semantic": 0.0}
        record = step(raw)
        record["internals"]["modulated_bids"] = dict(record["internals"]["modulated_bids"], audio=0.3)
        compared = probe.compare_step(AffectiveModulator({"ablate_existence_bias": False}), record)
        assert compared["reproduced"] is False


class TestDecision:
    def test_no_change_at_every_seed_and_a_change_at_one(self):
        quiet = {"void": False, "leader_changed_share": 0.0}
        moved = {"void": False, "leader_changed_share": 0.01}
        assert probe.decision({"a": quiet, "b": quiet, "c": quiet}).startswith("NO LEADER CHANGE")
        assert probe.decision({"a": quiet, "b": moved, "c": quiet}).startswith("LEADER CHANGES")
        assert probe.decision({"a": dict(quiet, void=True), "b": quiet, "c": quiet}).startswith("VOID")
