"""Does the learned value of vision explain the seeds where vision takes nearly every step?

Read-only. Works on the session records of dark room runs made with --learned-valence.

WHY. Gate B6 (docs/results/dark_room_senses_2026_09.md) FAILED on competition at 3 of 10
seeds, where vision took 0.968 to 0.996 of ignited steps. The same happened at seed 55 in
Gate B3. The learned valence (models/emotion/learned_valence.py) adds
valence_gain * |value| to a module's bid, and a raw bid is at most 1.0. The value has no
upper limit. An agent that stays in the light collects reward 1.0 per step while the
vision bid is high, so the value of vision can grow far above the value of hearing.

WHAT IS REPORTED, per run and per episode.
    vision share   share of ignited steps that vision won
    in light       steps with the agent in the light (environment truth)
    value          the learned value of vision and of hearing after the last step
Per run: the mean value of vision and of hearing over all steps, and the Spearman rho
across runs between the vision share and the mean value gap (vision minus hearing).

This is a description of recorded runs. A rho across seeds is not a cause. The cause is
tested by an ablation that is stated in the verdict document before it is run.

Run:
    python -m scripts.analysis.probe_valence_lockin --runs runs/gate_b6_s63 ... runs/gate_b6_s72
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr


def load_episodes(run: str) -> dict:
    """Episode index to the list of recorded steps, read from the session record."""
    episodes = {}
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            episodes[int(folder[-4:])] = [json.loads(line) for line in handle]
    if not episodes:
        raise FileNotFoundError("no recorded episodes under %s" % run)
    return episodes


def learned_values(step: dict) -> dict:
    recorded = step["internals"].get("learned_valence")
    if recorded is None:
        raise ValueError("the run was not made with --learned-valence")
    return recorded["values"]


def vision_share(steps: list) -> float:
    """Share of ignited steps won by vision. NaN when no step ignited."""
    ignited = [step for step in steps if step["winner"]]
    if not ignited:
        return float("nan")
    return float(np.mean([step["winner"] == "vision" for step in ignited]))


def episode_row(steps: list) -> dict:
    values = learned_values(steps[-1])
    in_light = sum(bool(step["internals"]["info_after_step"].get("in_light")) for step in steps)
    return {"vision_share": vision_share(steps), "in_light": in_light,
            "value_vision": float(values.get("vision", 0.0)),
            "value_audio": float(values.get("audio", 0.0))}


def run_summary(episodes: dict) -> dict:
    steps = [step for index in sorted(episodes) for step in episodes[index]]
    vision = [learned_values(step).get("vision", 0.0) for step in steps]
    audio = [learned_values(step).get("audio", 0.0) for step in steps]
    return {"vision_share": vision_share(steps),
            "in_light": sum(episode_row(episodes[i])["in_light"] for i in episodes),
            "mean_value_vision": float(np.mean(vision)),
            "mean_value_audio": float(np.mean(audio)),
            "episodes": {index: episode_row(episodes[index]) for index in sorted(episodes)}}


def rho_share_against_gap(summaries: list) -> float:
    """Spearman rho across runs of the vision share against the mean value gap."""
    if len(summaries) < 3:
        return float("nan")
    shares = [s["vision_share"] for s in summaries]
    gaps = [s["mean_value_vision"] - s["mean_value_audio"] for s in summaries]
    return float(spearmanr(shares, gaps).correlation)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    summaries = []
    for run in args.runs:
        summary = run_summary(load_episodes(run))
        summaries.append(summary)
        print("\n=== %s" % run)
        print("vision share %.3f, in light %d steps, mean value vision %.3f, hearing %.3f"
              % (summary["vision_share"], summary["in_light"],
                 summary["mean_value_vision"], summary["mean_value_audio"]))
        for index, row in summary["episodes"].items():
            print("  episode %d  vision share %.2f  in light %3d  value vision %+.2f  hearing %+.2f"
                  % (index, row["vision_share"], row["in_light"],
                     row["value_vision"], row["value_audio"]))
    print("\nrho across runs, vision share against mean value gap: %.3f (%d runs)"
          % (rho_share_against_gap(summaries), len(summaries)))


if __name__ == "__main__":
    main()
