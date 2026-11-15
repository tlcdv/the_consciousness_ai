"""Report of check 3 of the competing goals gate, the smoke run, with the rule written before the run.

`docs/decisions/2026_11_08_competing_goals_gate.md`, section "the rule of check 3". Read-only. It reads
two run folders of the training script, one for each arm, made at seed 999 with 8 episodes of 200
steps, the flags of the gate, `--value-salience-floor 0.15` and `--stop-on-non-finite`.

RULE. Check 3 HOLDS when each criterion holds in both arms:
  complete                 8 rows in episodes.csv, 1600 rows in metrics.csv, and the recorded command
                           has the floor and the stop.
  ethics                   the ethics manifest records the drive off and no noxious channel.
  one_record_per_episode   the gate record has 8 episodes, and for each one the candidates and the
                           removed bands are those of the binder record at the step of the choice.
  candidates               in each forced episode (0 to 5) the gate had one candidate, the band of the
                           present object's tone (18 light, 23 decoy). In each free episode (6 and 7)
                           the floor removed no candidate.
  one_update               each value after a choice is the value before plus LEARNING_RATE x (outcome
                           minus the value before). The outcome is the food reward divided by
                           (1 + DELAY_RATE (n - 1)) in the benefit arm and 1 in the binary arm, and 0
                           without food.
  food_sizes               each food that closed a choice is 32, 3 or 4 in the reward that the gate
                           read, and no choice was closed by a reward above 0.5 at a step where the
                           room gave no food (a false food).
  finite                   every reward and every recorded loss of every step is finite.

The step of a choice. The gate chooses at the first step of an episode at which the binder record has
a band and one or more bands with cells at a salience of the floor or more. This is the rule of the
wiring in the training script, repeated here from the binder record alone.

The reward that the gate read at a step is `reward_for_action_core` of the step record. The room
reward of the step is `env_reward`. The food of a choice is read at the step that closed it, the step
of the choice plus the gate's step count minus 1.

Reported with no criterion: the step of each choice, each removed band with its salience, the food
reward of each gate episode, the largest reward of each arm, the largest reward at a step where the
room gave no food, the largest reward prediction loss, and the first object against the chosen one.

Usage:
    python -m scripts.analysis.probe_competing_goals_smoke --benefit runs/<benefit run> \
        --binary runs/<binary run> --output <json>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from models.core.value_gate import FOOD, LEARNING_RATE
from scripts.analysis.probe_reversal import first_object

EPISODES, STEPS, FORCED = 8, 200, 6
FLOOR, DELAY_RATE = 0.15, 0.1
FOOD_SIZES = (32.0, 3.0, 4.0)
TONE_BAND = {"light": 18, "decoy": 23}
ARMS = ("benefit", "binary")
CRITERIA = ("complete", "ethics", "one_record_per_episode", "candidates", "one_update", "food_sizes", "finite")
TOLERANCE = 1e-9


def choice_at(binding_steps: List[dict], floor: float) -> Optional[Tuple[int, dict, dict]]:
    """(step, kept candidates, removed candidates) of the choice of one episode, or None without one."""
    for index, record in enumerate(binding_steps):
        if record.get("band") is None:
            continue
        candidates = {band: record["salience"][band] for band, cells in record.get("groups", {}).items() if cells}
        kept = {band: s for band, s in candidates.items() if s >= floor}
        if kept:
            return index, kept, {band: s for band, s in candidates.items() if s < floor}
    return None


def expected_outcome(arm: str, gate_read: float, steps: int) -> float:
    """The outcome that the gate must give for the reward that closed the choice."""
    if gate_read <= FOOD:
        return 0.0
    return gate_read / (1.0 + DELAY_RATE * (steps - 1)) if arm == "benefit" else 1.0


def chosen_object(choice) -> Optional[str]:
    """The object whose tone has the chosen band, or None for a band that is no tone."""
    for name, band in TONE_BAND.items():
        if int(choice) == band:
            return name
    return None


def episode_row(episode: int, gate: dict, binding_steps: List[dict], steps: List[dict], arm: str,
                floor: float) -> dict:
    """What the report holds for one episode, from its gate record, binder record and step records."""
    found = choice_at(binding_steps, floor)
    if found is None:
        raise ValueError("episode %d has a gate record and no step with a kept candidate" % episode)
    choice_step, kept, removed = found
    closing_step = choice_step + int(gate["steps"]) - 1
    closing = steps[closing_step]["internals"]
    gate_read = float(closing["rewards"]["reward_for_action_core"])
    room_reward = float(closing["rewards"]["env_reward"])
    update = gate["value_before"] + LEARNING_RATE * (gate["outcome"] - gate["value_before"])
    return {"episode": episode, "choice": int(gate["choice"]), "chosen": chosen_object(gate["choice"]),
            "candidates": len(gate["salience"]), "removed": gate.get("removed", {}),
            "choice_step": choice_step, "closing_step": closing_step, "gate_steps": int(gate["steps"]),
            "outcome": float(gate["outcome"]), "gate_read": gate_read, "room_reward": room_reward,
            "food_reward": gate.get("food_reward"), "fed": closing["info_after_step"].get("_truth_food_object"),
            "present": closing["info_after_step"].get("_truth_present"), "first": first_object(steps),
            "records_agree": gate["salience"] == kept and gate.get("removed", {}) == removed,
            "update_ok": abs(gate["value_after"] - update) <= TOLERANCE,
            "outcome_ok": abs(gate["outcome"] - expected_outcome(arm, gate_read, int(gate["steps"]))) <= TOLERANCE
            and gate.get("food_reward", gate_read if gate_read > FOOD else None) in (None, gate_read),
            "false_food": gate_read > FOOD and room_reward <= FOOD}


def all_finite(episode_steps: List[List[dict]], metric_rewards: List[float]) -> bool:
    """True when every reward and every loss that a step computed is finite."""
    numbers = list(metric_rewards)
    for steps in episode_steps:
        for record in steps:
            numbers.append(record["reward"])
            numbers += list(record["internals"]["rewards"].values())
            numbers += list(record["internals"]["losses"].values())
    return all(math.isfinite(float(number)) for number in numbers if number is not None)


def _forced_ok(row: dict) -> bool:
    present = "light" if row["episode"] % 2 == 0 else "decoy"
    return row["candidates"] == 1 and row["choice"] == TONE_BAND[present] and row["present"] == present


def criteria(arm: dict) -> Dict[str, bool]:
    """Which of the seven criteria hold for one arm."""
    rows = arm["rows"]
    fed_rows = [row for row in rows if row["outcome"] > 0.0]
    return {
        "complete": arm["episodes_csv_rows"] == EPISODES and arm["metrics_rows"] == EPISODES * STEPS
        and arm["argv_has_floor"] and arm["argv_has_stop"],
        "ethics": arm["existence_drive"] == "off" and arm["noxious_channels"] == [],
        "one_record_per_episode": arm["gate_records"] == EPISODES and len(rows) == EPISODES
        and all(row["records_agree"] for row in rows),
        "candidates": all(_forced_ok(row) for row in rows if row["episode"] < FORCED)
        and all(not row["removed"] and row["present"] == "both" for row in rows if row["episode"] >= FORCED),
        "one_update": all(row["update_ok"] and row["outcome_ok"] for row in rows),
        "food_sizes": all(row["gate_read"] in FOOD_SIZES for row in fed_rows)
        and not any(row["false_food"] for row in rows),
        "finite": bool(arm["all_finite"])}


def verdict(holds: Dict[str, Dict[str, bool]]) -> str:
    """HOLDS when each criterion holds in both arms, else FAILED."""
    if sorted(holds) != sorted(ARMS):
        raise ValueError("the rule needs the arms %s, got %s" % (ARMS, sorted(holds)))
    return "HOLDS" if all(all(arm.values()) for arm in holds.values()) else "FAILED"


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_steps(run: str) -> List[List[dict]]:
    episodes = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            episodes.append([json.loads(line) for line in handle if line.strip()])
    return episodes


def _csv_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extremes(episode_steps: List[List[dict]]) -> dict:
    """The largest reward, the largest reward at a step without room food, and the largest reward loss."""
    records = [record["internals"] for steps in episode_steps for record in steps]
    no_food = [r["rewards"]["reward_for_action_core"] for r in records if r["rewards"]["env_reward"] <= FOOD]
    losses = [r["losses"]["pred_loss"] for r in records if r["losses"].get("pred_loss") is not None]
    return {"largest_reward": max(r["rewards"]["reward_for_action_core"] for r in records),
            "largest_reward_without_room_food": max(no_food), "largest_pred_loss": max(losses)}


def read_arm(run: str, arm: str) -> dict:
    """The run facts and the episode rows of one arm."""
    gate = _load_json(os.path.join(run, "value_gate.json"))
    history = _load_json(os.path.join(run, "map_binding.json"))["history"]
    manifest = _load_json(os.path.join(run, "ethics_manifest.json"))
    argv = _load_json(os.path.join(run, "session.json"))["run"]["argv"]
    episode_steps = _load_steps(run)
    metrics = _csv_rows(os.path.join(run, "metrics.csv"))
    if len(history) != EPISODES * STEPS or len(episode_steps) != EPISODES or len(gate["episodes"]) != EPISODES:
        raise ValueError("%s: %d binder steps, %d episode folders, %d gate records; the run needs %d, %d, %d" % (
            run, len(history), len(episode_steps), len(gate["episodes"]), EPISODES * STEPS, EPISODES, EPISODES))
    rows = [episode_row(e, gate["episodes"][e], history[e * STEPS:(e + 1) * STEPS], episode_steps[e], arm, FLOOR)
            for e in range(EPISODES)]
    report = {"run": run, "arm": arm, "rows": rows, "gate_records": len(gate["episodes"]),
              "episodes_csv_rows": len(_csv_rows(os.path.join(run, "episodes.csv"))), "metrics_rows": len(metrics),
              "argv_has_floor": "--value-salience-floor" in argv
              and float(argv[argv.index("--value-salience-floor") + 1]) == FLOOR,
              "argv_has_stop": "--stop-on-non-finite" in argv, "existence_drive": manifest["existence_drive"],
              "noxious_channels": manifest["noxious_channels_present"], "git_commit": manifest.get("git_commit"),
              "all_finite": all_finite(episode_steps, [float(row["reward"]) for row in metrics]),
              "values": gate["values"]}
    report.update(_extremes(episode_steps))
    return report


def print_arm(report: dict, holds: Dict[str, bool]) -> None:
    print("%s arm, %s" % (report["arm"], report["run"]))
    for row in report["rows"]:
        print("  episode %d %-5s choice %s at step %d, %d candidate(s), removed %s, closed at step %d, gate read "
              "%.4f, room %.4f, outcome %.4f, fed %s, first %s" % (
                  row["episode"], row["present"], row["choice"], row["choice_step"], row["candidates"],
                  {band: round(s, 4) for band, s in row["removed"].items()}, row["closing_step"],
                  row["gate_read"], row["room_reward"], row["outcome"], row["fed"], row["first"]))
    for key in ("largest_reward", "largest_reward_without_room_food", "largest_pred_loss", "values"):
        print("  %-34s %s" % (key, report[key]))
    print("  holds %s" % holds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for arm in ARMS:
        parser.add_argument("--" + arm, required=True, help="The run folder of the %s arm." % arm)
    parser.add_argument("--output", required=True, help="Write the report as JSON here.")
    options = parser.parse_args()
    reports = {arm: read_arm(getattr(options, arm), arm) for arm in ARMS}
    holds = {arm: criteria(report) for arm, report in reports.items()}
    for arm in ARMS:
        print_arm(reports[arm], holds[arm])
    result = {"arms": reports, "holds": holds, "verdict": verdict(holds)}
    print("VERDICT of check 3:", result["verdict"])
    with open(options.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
