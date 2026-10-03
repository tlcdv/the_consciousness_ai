"""Probe of the vector the policy received (scripts/analysis/probe_broadcast_carrier.py).

Every expected number is counted by hand on constructed steps. The episode written here
has 5 steps: 4 ignite, 1 does not.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis.probe_broadcast_carrier import (
    admitted_count,
    carried_module,
    episode_rows,
    pair_counts,
    run_rows,
    run_summary,
)

VISION = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
AUDIO = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
ZEROS = np.zeros((1, 3), dtype=np.float32)


def _step(index, winner, ignited, vision_bid, audio_bid, threshold=0.5):
    return {"step": index, "winner": winner, "ignited": ignited,
            "bound_bids": {"vision": vision_bid, "audio": audio_bid, "memory": 0.1},
            "internals": {"ignition": {"threshold": threshold}}}


def _write_episode(run_dir, name="ep_0000"):
    """Steps 0 to 4. The soft limit is 0.8 * 0.5 = 0.4.

    0  vision wins, both admitted, broadcast is the audio vector   (weakest admitted)
    1  audio wins, both admitted, broadcast is the vision vector   (weakest admitted)
    2  vision wins, only vision admitted, broadcast is the vision vector
    3  not ignited
    4  audio wins, only audio admitted, broadcast is the audio vector
    """
    episode_dir = os.path.join(run_dir, "episodes", name)
    os.makedirs(episode_dir)
    steps = [_step(0, "vision", True, 0.9, 0.6), _step(1, "audio", True, 0.5, 0.7),
             _step(2, "vision", True, 0.9, 0.3), _step(3, "", False, 0.2, 0.1),
             _step(4, "audio", True, 0.3, 0.8)]
    with open(os.path.join(episode_dir, "steps.jsonl"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(step) for step in steps) + "\n")
    indices = np.arange(5)
    broadcast = np.stack([AUDIO, VISION, VISION, ZEROS, AUDIO])
    np.savez(os.path.join(episode_dir, "vectors.npz"),
             broadcast=broadcast, broadcast__steps=indices,
             tectum_content=np.stack([VISION] * 5), tectum_content__steps=indices,
             audio_content=np.stack([AUDIO] * 5), audio_content__steps=indices)
    return episode_dir


# --- one vector ----------------------------------------------------------------------

def test_carried_module_names_the_equal_vector():
    assert carried_module(VISION, VISION, AUDIO) == "vision"
    assert carried_module(AUDIO, VISION, AUDIO) == "audio"


def test_a_vector_equal_to_neither_is_other_and_an_all_zero_vector_is_zeros():
    assert carried_module(VISION + 1e-6, VISION, AUDIO) == "other"
    assert carried_module(ZEROS, VISION, AUDIO) == "zeros"


def test_two_equal_content_vectors_are_reported_as_both_and_not_as_one_of_them():
    assert carried_module(VISION, VISION, VISION.copy()) == "both"


def test_admitted_count_uses_the_soft_threshold_of_the_workspace():
    assert admitted_count(_step(0, "vision", True, 0.9, 0.6)) == 2
    assert admitted_count(_step(0, "vision", True, 0.9, 0.39)) == 1
    # Exactly at 0.8 * threshold counts, as in _resolve_competition (>=).
    assert admitted_count(_step(0, "vision", True, 0.9, 0.4, threshold=0.5)) == 2


# --- one episode and one run ---------------------------------------------------------

def test_episode_rows_cover_the_ignited_steps_only(tmp_path):
    rows = episode_rows(_write_episode(str(tmp_path)))
    assert rows == [("vision", "audio", 2), ("audio", "vision", 2),
                    ("vision", "vision", 1), ("audio", "audio", 1)]


def test_run_summary_counted_by_hand(tmp_path):
    _write_episode(str(tmp_path))
    summary = run_summary(run_rows(str(tmp_path)))
    assert summary["ignited"] == 4
    assert summary["share_winner_vector"] == 0.5   # steps 2 and 4
    assert summary["share_vision_vector"] == 0.5   # steps 1 and 2
    assert summary["share_audio_vector"] == 0.5    # steps 0 and 4
    assert summary["share_two_admitted"] == 0.5    # steps 0 and 1


def test_two_episodes_are_pooled(tmp_path):
    _write_episode(str(tmp_path), "ep_0000")
    _write_episode(str(tmp_path), "ep_0001")
    rows = run_rows(str(tmp_path))
    assert len(rows) == 8
    assert pair_counts(rows) == {"audio/audio": 2, "audio/vision": 2,
                                 "vision/audio": 2, "vision/vision": 2}


def test_a_run_without_an_ignited_step_gives_none_and_never_zero():
    summary = run_summary([])
    assert summary["ignited"] == 0
    for name in ("share_winner_vector", "share_vision_vector", "share_audio_vector",
                 "share_two_admitted"):
        assert summary[name] is None


def test_a_merge_that_keeps_the_winner_scores_one(tmp_path):
    """The control: when the broadcast is the winner's vector at every step, the share is 1."""
    rows = [("vision", "vision", 2), ("audio", "audio", 2), ("vision", "vision", 1)]
    assert run_summary(rows)["share_winner_vector"] == 1.0


def test_a_run_folder_without_episodes_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_rows(str(tmp_path))


def test_an_ignited_step_without_a_recorded_vector_is_refused(tmp_path):
    episode_dir = _write_episode(str(tmp_path))
    indices = np.arange(4)
    np.savez(os.path.join(episode_dir, "vectors.npz"),
             broadcast=np.stack([AUDIO] * 4), broadcast__steps=indices,
             tectum_content=np.stack([VISION] * 4), tectum_content__steps=indices,
             audio_content=np.stack([AUDIO] * 4), audio_content__steps=indices)
    with pytest.raises(ValueError, match="no recorded vector"):
        episode_rows(episode_dir)
