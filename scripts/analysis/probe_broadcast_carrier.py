"""Which module's vector did the policy receive at each ignited step. Read-only.

The training loop records the winner of a step as the module with the highest bound bid.
The workspace admits every module whose bound bid is at least 0.8 of the ignition
threshold, and with the default `--broadcast-merge legacy` the broadcast then holds the
vector of the WEAKEST admitted module. This probe compares the recorded broadcast vector
with the recorded vision and audio vectors, element by element, so it reads what the
policy received and does not rebuild it from the bids.

It reports, per run:
  share_winner_vector   ignited steps where the broadcast is the recorded winner's vector
  share_vision_vector   ignited steps where the broadcast is the vision vector
  share_audio_vector    ignited steps where the broadcast is the audio vector
  share_two_admitted    ignited steps where two or more modules passed the threshold

A run without an ignited step gives None for every share, never 0.

Usage:
    python -m scripts.analysis.probe_broadcast_carrier --runs runs/gate_b6_s63 runs/gate_b6_s64
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

SOFT_THRESHOLD_FACTOR = 0.8  # GlobalWorkspace._resolve_competition
CARRIED_KINDS = ("vision", "audio", "both", "zeros", "other")


def vectors_by_step(archive, name: str) -> Dict[int, np.ndarray]:
    """Recorded vectors of one name, keyed by the step index they belong to."""
    return dict(zip(archive[name + "__steps"].tolist(), archive[name]))


def carried_module(sent: np.ndarray, vision: np.ndarray, audio: np.ndarray) -> str:
    """Name the recorded vector the broadcast is equal to, element by element."""
    is_vision = np.array_equal(sent.ravel(), vision.ravel())
    is_audio = np.array_equal(sent.ravel(), audio.ravel())
    if is_vision and is_audio:
        return "both"
    if is_vision:
        return "vision"
    if is_audio:
        return "audio"
    return "other" if sent.any() else "zeros"


def admitted_count(step: dict) -> int:
    """Modules whose bound bid passed the soft threshold of the workspace."""
    limit = step["internals"]["ignition"]["threshold"] * SOFT_THRESHOLD_FACTOR
    return sum(1 for bid in step["bound_bids"].values() if bid >= limit)


def ignited_steps(episode_dir: str) -> List[dict]:
    with open(os.path.join(episode_dir, "steps.jsonl"), encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle if line.strip()]
    return [step for step in steps if step.get("ignited") and step.get("winner")]


def episode_rows(episode_dir: str) -> List[Tuple[str, str, int]]:
    """(recorded winner, carried module, admitted modules) per ignited step."""
    archive = np.load(os.path.join(episode_dir, "vectors.npz"))
    broadcast = vectors_by_step(archive, "broadcast")
    vision = vectors_by_step(archive, "tectum_content")
    audio = vectors_by_step(archive, "audio_content")
    rows = []
    for step in ignited_steps(episode_dir):
        index = step["step"]
        if index not in broadcast or index not in vision or index not in audio:
            raise ValueError("%s: ignited step %d has no recorded vector" % (episode_dir, index))
        carried = carried_module(broadcast[index], vision[index], audio[index])
        rows.append((step["winner"], carried, admitted_count(step)))
    return rows


def run_rows(run_dir: str) -> List[Tuple[str, str, int]]:
    episode_dirs = sorted(glob.glob(os.path.join(run_dir, "episodes", "ep_*")))
    if not episode_dirs:
        raise FileNotFoundError("%s holds no recorded episode" % run_dir)
    return [row for episode_dir in episode_dirs for row in episode_rows(episode_dir)]


def run_summary(rows: List[Tuple[str, str, int]]) -> Dict[str, Optional[float]]:
    """Shares over ignited steps. Every share is None when no step ignited."""
    total = len(rows)
    names = ("share_winner_vector", "share_vision_vector", "share_audio_vector",
             "share_two_admitted")
    if total == 0:
        return dict({name: None for name in names}, ignited=0)
    return {
        "ignited": total,
        "share_winner_vector": sum(1 for won, carried, _ in rows if won == carried) / total,
        "share_vision_vector": sum(1 for _, carried, _ in rows if carried == "vision") / total,
        "share_audio_vector": sum(1 for _, carried, _ in rows if carried == "audio") / total,
        "share_two_admitted": sum(1 for _, _, admitted in rows if admitted >= 2) / total,
    }


def pair_counts(rows: List[Tuple[str, str, int]]) -> Dict[str, int]:
    """Counts of 'recorded winner/carried module' pairs."""
    counts = collections.Counter("%s/%s" % (won, carried) for won, carried, _ in rows)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="Run folders with episodes/.")
    args = parser.parse_args()
    for run_dir in args.runs:
        rows = run_rows(run_dir)
        print("=== %s" % run_dir)
        print(json.dumps({"summary": run_summary(rows), "pairs": pair_counts(rows)}, indent=2))


if __name__ == "__main__":
    main()
