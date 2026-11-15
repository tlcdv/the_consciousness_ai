"""Report of the long smoke run of the competing goals gate, with the rule written before the run.

`docs/decisions/2026_11_08_competing_goals_gate.md`, section "the long smoke run". Read-only. It reads
one run folder of the training script: seed 999, 60 episodes of 200 steps, the benefit arm, the flags
of the gate with the salience floor and `--stop-on-non-finite`.

RULE. The run HOLDS when `metrics.csv` has EPISODES x STEPS rows and every recorded reward, loss and
gradient norm is finite. The recorded values are those of the step records (`steps.jsonl`): the reward
that the loop passed to the action core, each loss and each gradient norm that the step computed, and
the value estimate of the policy. The `reward` column of `metrics.csv` is read too.

Reported: the largest absolute value of each kind for each block of BLOCK episodes. A loss can be
below 0, so the size is the absolute value.

No number of the trade-off is read from this run. The script does not open the gate record.

Usage:
    python -m scripts.analysis.probe_training_values --run runs/<run> --output <json>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, List

EPISODES, STEPS, BLOCK = 60, 200, 10


def step_values(record: dict) -> Dict[str, float]:
    """The reward, each loss, each gradient norm and the value estimate that one step recorded."""
    internals = record["internals"]
    found = {"reward": internals["rewards"]["reward_for_action_core"]}
    found.update({"loss %s" % name: number for name, number in internals["losses"].items()})
    found.update({"gradient %s" % name: number for name, number in internals["gradient_norms"].items()})
    found["value_estimate"] = internals.get("value_estimate")
    return {kind: float(number) for kind, number in found.items() if number is not None}


def block_maxima(episode_steps: List[List[dict]], block: int = BLOCK) -> List[dict]:
    """For each block of `block` episodes, the largest absolute value of each kind."""
    blocks = []
    for start in range(0, len(episode_steps), block):
        part = episode_steps[start:start + block]
        largest: Dict[str, float] = {}
        for steps in part:
            for record in steps:
                for kind, number in step_values(record).items():
                    largest[kind] = max(largest.get(kind, 0.0), abs(number))
        blocks.append({"episodes": [start, start + len(part) - 1], "largest": largest})
    return blocks


def not_finite(episode_steps: List[List[dict]], metric_rewards: List[float]) -> List[dict]:
    """Each recorded value that is not finite, with its place."""
    found = []
    for episode, steps in enumerate(episode_steps):
        for index, record in enumerate(steps):
            found += [{"episode": episode, "step": index, "kind": kind}
                      for kind, number in step_values(record).items() if not math.isfinite(number)]
    found += [{"metrics_row": row, "kind": "reward"}
              for row, number in enumerate(metric_rewards) if not math.isfinite(float(number))]
    return found


def holds(metrics_rows: int, found: List[dict]) -> bool:
    """The rule: the full number of rows and no value that is not finite."""
    return metrics_rows == EPISODES * STEPS and not found


def _load_steps(run: str) -> List[List[dict]]:
    episodes = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            episodes.append([json.loads(line) for line in handle if line.strip()])
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="The run folder.")
    parser.add_argument("--output", required=True, help="Write the report as JSON here.")
    options = parser.parse_args()
    with open(os.path.join(options.run, "metrics.csv"), encoding="utf-8", newline="") as handle:
        rewards = [float(row["reward"]) for row in csv.DictReader(handle)]
    episode_steps = _load_steps(options.run)
    found = not_finite(episode_steps, rewards)
    blocks = block_maxima(episode_steps)
    recorded = sum(len(step_values(record)) for steps in episode_steps for record in steps)
    result = {"run": options.run, "metrics_rows": len(rewards), "episode_folders": len(episode_steps),
              "step_records": sum(len(steps) for steps in episode_steps), "recorded_values": recorded,
              "not_finite": found, "blocks": blocks, "holds": holds(len(rewards), found)}
    print("metrics rows %d, episode folders %d, step records %d, recorded values %d, not finite %d" % (
        result["metrics_rows"], result["episode_folders"], result["step_records"], recorded, len(found)))
    for kind in sorted({kind for b in blocks for kind in b["largest"]}):
        print("  %-36s %s" % (kind, "  ".join("%12.4g" % b["largest"].get(kind, float("nan")) for b in blocks)))
    print("VERDICT of the long smoke run:", "HOLDS" if result["holds"] else "FAILED")
    with open(options.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
