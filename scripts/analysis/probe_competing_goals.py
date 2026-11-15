"""Probe of the competing goals gate, with the rule written before the gate runs.

`docs/decisions/2026_11_08_competing_goals_gate.md`, Gate section. Read-only. It reads six run folders
of the training script: seeds 1301, 1302 and 1303, the benefit arm and the binary arm, 60 episodes of
200 steps. The gate tests AE-1 only. The workspace does not take part.

FIRST OBJECT. Within the first 60 steps of an episode, the first object closer than 60 room pixels,
the nearer one on a tie. No contact counts as neither. The first count is `first_object` of
`scripts/analysis/probe_reversal.py`. The second count is `scripts/analysis/count_first_object.py`,
which has the same definition in separate code.

RULE, at 3 of 3 seeds. Each window has 15 episodes and neither stays in the denominator, so 0.8 needs
12 of 15.
  C1  benefit arm, episodes 15 to 29: light first in 12 or more of 15.
  C2  benefit arm, episodes 45 to 59: decoy first in 12 or more of 15.
  C3  for each seed, with the two arms of that seed: (decoy first in 45 to 59) minus (decoy first in
      15 to 29) is larger in the benefit arm than in the binary arm by 0.3 or more. With 15 episodes
      this is a difference of 5 episodes or more.

VERDICT. VOID comes first. The gate is VOID when a run does not have 12000 rows in `metrics.csv` or 60
episode folders, whatever the cause, or when the two counts differ in one or more episodes. A
difference stays VOID until it is explained. No criterion is computed for a VOID gate. If the gate is
not VOID, it HOLDS when C1, C2 and C3 hold at each of the 3 seeds, and it is FAILED if not.

CROSS-CHECKS, in the report only, with no effect on the verdict. For each run, the count of episodes
in which the first object differs from the object of the gate's choice (tone band 18 is the light, 23
is the decoy), and the count in which it differs from the first object that gave food
(`_truth_food_object`).

SELF-TEST, before the gate. On the two smoke runs of check 3 (seed 999, 8 episodes), each count must
give the first object equal to the object of the gate's choice in 16 of 16 episodes.

Usage:
    python -m scripts.analysis.probe_competing_goals --benefit <3 runs> --binary <3 runs> --output <json>
    python -m scripts.analysis.probe_competing_goals --self-test <2 runs> --output <json>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.analysis import count_first_object
from scripts.analysis.probe_reversal import LIMIT, first_object

SEEDS = (1301, 1302, 1303)
EPISODES, STEPS = 60, 200
C1_WINDOW, C2_WINDOW = (15, 30), (45, 60)
NEED = 12
MARGIN_TENTHS = 3
FLOOR, DELAY_RATE = 0.15, 0.1
TONE_BAND = {"light": 18, "decoy": 23}
CRITERIA = ("C1", "C2", "C3")

First = Optional[str]


def window_count(firsts: Sequence[First], window: Tuple[int, int], target: str) -> int:
    """The number of episodes of [start, stop) whose first object is `target`."""
    return sum(1 for first in firsts[window[0]:window[1]] if first == target)


def seed_criteria(benefit: Sequence[First], binary: Sequence[First]) -> dict:
    """C1 to C3 of one seed, from the first objects of its benefit run and its binary run."""
    width = C1_WINDOW[1] - C1_WINDOW[0]
    light_early = window_count(benefit, C1_WINDOW, "light")
    decoy_late = window_count(benefit, C2_WINDOW, "decoy")
    change = {arm: window_count(values, C2_WINDOW, "decoy") - window_count(values, C1_WINDOW, "decoy")
              for arm, values in (("benefit", benefit), ("binary", binary))}
    return {"light_first_15_29": light_early, "decoy_first_45_59": decoy_late,
            "change_benefit": change["benefit"], "change_binary": change["binary"], "episodes_of_a_window": width,
            "C1": light_early >= NEED, "C2": decoy_late >= NEED,
            "C3": 10 * (change["benefit"] - change["binary"]) >= MARGIN_TENTHS * width}


def void_reasons(run: dict) -> List[str]:
    """Each reason for which one run makes the gate VOID. An empty list when there is none."""
    reasons = []
    if run["metrics_rows"] != EPISODES * STEPS:
        reasons.append("%s has %d rows in metrics.csv, not %d" % (run["run"], run["metrics_rows"], EPISODES * STEPS))
    if run["episode_folders"] != EPISODES:
        reasons.append("%s has %d episode folders, not %d" % (run["run"], run["episode_folders"], EPISODES))
    differ = [e for e, (a, b) in enumerate(zip(run["first"], run["second"])) if a != b]
    if differ or len(run["first"]) != len(run["second"]):
        reasons.append("%s: the two counts differ, episodes %s (lengths %d and %d)" % (
            run["run"], differ, len(run["first"]), len(run["second"])))
    return reasons


def verdict(void: List[str], criteria: Dict[int, dict]) -> str:
    """VOID with one or more reasons. If not, HOLDS when C1 to C3 hold at each seed of the gate, else FAILED."""
    if void:
        return "VOID"
    if tuple(sorted(criteria)) != SEEDS:
        raise ValueError("the rule needs the seeds %s, got %s" % (SEEDS, sorted(criteria)))
    return "HOLDS" if all(criteria[seed][name] for seed in SEEDS for name in CRITERIA) else "FAILED"


def chosen_object(band) -> First:
    """The object whose tone has the chosen band, or None for a band that is no tone."""
    for name, tone in TONE_BAND.items():
        if int(band) == tone:
            return name
    return None


def fed_object(steps: List[dict]) -> First:
    """The first object that gave food in the episode, or None without food."""
    for record in steps:
        fed = record["internals"]["info_after_step"].get("_truth_food_object")
        if fed is not None:
            return fed
    return None


def cross_check(first: Sequence[First], chosen: Sequence[First], fed: Sequence[First]) -> dict:
    """Report only. The counts of episodes in which the first object differs from the choice and the food."""
    return {"episodes": len(first),
            "first_differs_from_choice": sum(1 for a, b in zip(first, chosen) if a != b),
            "first_differs_from_first_food": sum(1 for a, b in zip(first, fed) if a != b)}


def _flag_value(argv: Sequence[str], flag: str) -> Optional[str]:
    return argv[argv.index(flag) + 1] if flag in argv else None


def check_flags(argv: Sequence[str], arm: str) -> int:
    """The seed of a gate run. Raises when the recorded command is not that of the arm."""
    seed = int(_flag_value(argv, "--seed") or -1)
    if seed not in SEEDS:
        raise ValueError("seed %d is not a seed of the gate %s" % (seed, SEEDS))
    if float(_flag_value(argv, "--value-salience-floor") or "nan") != FLOOR or "--stop-on-non-finite" not in argv:
        raise ValueError("a gate run needs --value-salience-floor %s and --stop-on-non-finite" % FLOOR)
    hyperbolic = _flag_value(argv, "--value-outcome") == "hyperbolic"
    if arm == "benefit" and not (hyperbolic and float(_flag_value(argv, "--value-delay-rate") or "nan") == DELAY_RATE):
        raise ValueError("the benefit arm needs --value-outcome hyperbolic --value-delay-rate %s" % DELAY_RATE)
    if arm == "binary" and hyperbolic:
        raise ValueError("the binary arm has no --value-outcome hyperbolic")
    return seed


def self_test(runs: List[dict]) -> dict:
    """The self-test: each count gives the object of the gate's choice in each episode of each run."""
    episodes = sum(len(run["chosen"]) for run in runs)
    equal = {key: sum(1 for run in runs for a, b in zip(run[key], run["chosen"]) if a == b)
             for key in ("first", "second")}
    same_length = all(len(run["first"]) == len(run["second"]) == len(run["chosen"]) for run in runs)
    return {"episodes": episodes, "first_count_equal": equal["first"], "second_count_equal": equal["second"],
            "holds": same_length and equal["first"] == episodes and equal["second"] == episodes}


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_run(run: str) -> dict:
    """The facts of one run folder: rows, folders, both counts, the gate's choices and the first foods."""
    first, fed = [], []
    folders = sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))
    for folder in folders:
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle if line.strip()]
        first.append(first_object(steps, LIMIT))
        fed.append(fed_object(steps))
    with open(os.path.join(run, "metrics.csv"), encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    gate_path = os.path.join(run, "value_gate.json")
    gate = _load_json(gate_path)["episodes"] if os.path.exists(gate_path) else []
    return {"run": run, "metrics_rows": rows, "episode_folders": len(folders), "first": first,
            "second": count_first_object.count_run(run), "fed": fed,
            "chosen": [chosen_object(record["choice"]) for record in gate],
            "argv": _load_json(os.path.join(run, "session.json"))["run"]["argv"]}


def gate(benefit: List[str], binary: List[str]) -> dict:
    """The report and the verdict of the gate from three run folders of each arm."""
    runs: Dict[str, Dict[int, dict]] = {"benefit": {}, "binary": {}}
    for arm, folders in (("benefit", benefit), ("binary", binary)):
        for folder in folders:
            facts = read_run(folder)
            runs[arm][check_flags(facts["argv"], arm)] = facts
    if tuple(sorted(runs["benefit"])) != SEEDS or tuple(sorted(runs["binary"])) != SEEDS:
        raise ValueError("each arm needs one run for each of the seeds %s" % (SEEDS,))
    void = [reason for arm in runs.values() for facts in arm.values() for reason in void_reasons(facts)]
    criteria = {} if void else {seed: seed_criteria(runs["benefit"][seed]["first"], runs["binary"][seed]["first"])
                                for seed in SEEDS}
    checks = {arm: {seed: cross_check(facts["first"], facts["chosen"], facts["fed"]) for seed, facts in by_seed.items()}
              for arm, by_seed in runs.items()}
    rows = {arm: {seed: facts["metrics_rows"] for seed, facts in by_seed.items()} for arm, by_seed in runs.items()}
    return {"runs": runs, "metrics_rows": rows, "void": void, "criteria": criteria, "cross_checks": checks,
            "verdict": verdict(void, criteria)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benefit", nargs=3, help="The three run folders of the benefit arm.")
    parser.add_argument("--binary", nargs=3, help="The three run folders of the binary arm.")
    parser.add_argument("--self-test", nargs=2, help="The two smoke runs of check 3.")
    parser.add_argument("--output", required=True, help="Write the report as JSON here.")
    options = parser.parse_args()
    if options.self_test:
        runs = [read_run(folder) for folder in options.self_test]
        result = {"runs": runs, "self_test": self_test(runs)}
        print("SELF-TEST:", result["self_test"], "HOLDS" if result["self_test"]["holds"] else "FAILED")
    elif options.benefit and options.binary:
        result = gate(options.benefit, options.binary)
        for reason in result["void"]:
            print("VOID:", reason)
        print("metrics rows:", result["metrics_rows"])
        for seed, numbers in result["criteria"].items():
            print("seed %d: %s" % (seed, numbers))
        print("cross-checks, report only:", result["cross_checks"])
        print("VERDICT of the competing goals gate:", result["verdict"])
    else:
        parser.error("give --benefit and --binary, or --self-test")
    with open(options.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
