"""
The Stage 2 emotion appraisal in scripts/training/train_rlhf.py has never run.

evaluate_emotion calls `qualia_mapper.map_state(broadcast)`, but map_state requires
a goal vector as well, so the call raises TypeError on every step and a bare
`except Exception: pass` hides it. Every training step so far used the reflex
valence and a dominance of 0.0, so the reward's agency term and the modulator's
dominance boost never contributed.

Passing a goal vector would change every training number, so it waits for an owner
decision. These tests pin the behavior as it is, and that the failure is now
reported once per process instead of never.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.qualia_mapper import PhenomenologicalMapper
from models.emotion.reward_shaping import EmotionalRewardShaper
from scripts.training import train_rlhf


def test_map_state_needs_a_goal_vector():
    """The root cause. The one argument call cannot succeed."""
    with pytest.raises(TypeError, match="goal_vector"):
        PhenomenologicalMapper().map_state(torch.randn(256))


def test_appraisal_returns_the_reflex_values_and_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(train_rlhf, "_APPRAISAL_FAILURE_REPORTED", False)
    reflex = train_rlhf.evaluate_emotion(0.8, 1.0, 0.0)
    broadcast = torch.randn(256)

    with caplog.at_level(logging.WARNING, logger=train_rlhf.logger.name):
        first = train_rlhf.evaluate_emotion(0.8, 1.0, 0.0, broadcast=broadcast,
                                            qualia_mapper=PhenomenologicalMapper())
        second = train_rlhf.evaluate_emotion(0.8, 1.0, 0.0, broadcast=broadcast,
                                             qualia_mapper=PhenomenologicalMapper())

    assert first == reflex and second == reflex
    assert first["dominance"] == 0.0
    reports = [r for r in caplog.records if "appraisal" in r.getMessage()]
    assert len(reports) == 1
    assert "goal_vector" in reports[0].getMessage()


# --- the emotion config ------------------------------------------------------------

class _ReadRecorder(dict):
    def __init__(self, data):
        super().__init__(data)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def _emotion_config() -> dict:
    args = argparse.Namespace(action_dim=2, lr=1e-3, episodes=1, max_steps=10)
    return train_rlhf.build_config(args)["emotion"]


def test_every_emotion_config_key_is_read_by_the_reward_shaper():
    """build_config used to set "arousal_penalty" to 1.0, a key nothing reads."""
    recorder = _ReadRecorder(_emotion_config())
    EmotionalRewardShaper(recorder)
    assert set(recorder) - recorder.read == set()


def test_the_arousal_penalty_in_force_is_unchanged():
    shaper = EmotionalRewardShaper(dict(_emotion_config()))
    assert shaper.arousal_lambda == 0.1
    assert shaper.valence_weight == 0.5
