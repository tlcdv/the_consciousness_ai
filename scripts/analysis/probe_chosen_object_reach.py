"""How often the agent reaches the object that the value gate chose, and what a miss is.

Read-only probe on saved runs made with `--tectal-orienting value`. It checks three assumptions of the
abstract model of the competing goals gate (`scripts/analysis/model_competing_goals.py`,
`docs/decisions/2026_11_08_competing_goals_gate.md`).

- Reach: the share of episodes whose first object is the chosen one. First object as in
  `scripts/analysis/probe_reversal.py`: within the first 60 steps, the first object closer than 60 room
  pixels.
- Miss: in an episode whose first object is not the chosen one, whether the agent reached the other
  object or no object.
- Stay: after the first contact with the chosen object, the steps outside its radius in the next
  `--wait` steps. A food that comes after a wait needs the agent to stay.

The light's tone is the band that is louder in the episodes with a decoy level below 1. The probe
raises when that band is not the same in every such episode, or when an episode has no gate record.

Usage:
    python -m scripts.analysis.probe_chosen_object_reach --runs runs/vg_learn_s601 runs/rv_stay_s701
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.analysis.probe_reversal import CONTACT, LIMIT, first_object

OBJECTS = ("light", "decoy")


def read_steps(folder: str) -> List[dict]:
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def distances(steps: List[dict], name: str) -> np.ndarray:
    """The distance to object `name` after each step, in room pixels."""
    return np.array([np.hypot(*step["internals"]["info_after_step"]["_truth_%s_offset" % name]) for step in steps])


def light_band(gate_episodes: List[dict], levels: List[float], run: str) -> str:
    """The band of the light's tone: the louder band when the decoy level is below 1."""
    bands = {max(episode["salience"], key=episode["salience"].get)
             for episode, level in zip(gate_episodes, levels) if level < 1.0}
    if len(bands) != 1:
        raise ValueError("%s: the louder band at a quiet decoy is %s, expected one band" % (run, sorted(bands)))
    return bands.pop()


def steps_outside(steps: List[dict], name: str, wait: int) -> Optional[int]:
    """Steps outside the radius of `name` in the `wait` steps after its first contact; None if too few."""
    inside = distances(steps, name) < CONTACT
    contact = int(np.argmax(inside))
    after = inside[contact + 1:contact + 1 + wait]
    return None if len(after) < wait else int((~after).sum())


def episode_facts(steps: List[dict], gate_episode: dict, band: str, wait: int) -> Dict[str, object]:
    """The chosen object, the first object, and the steps outside after the contact with the chosen one."""
    chosen = "light" if str(gate_episode["choice"]) == band else "decoy"
    first = first_object(steps, LIMIT)
    return {"chosen": chosen, "first": first, "candidates": len(gate_episode["salience"]),
            "quieter_chosen": min(gate_episode["salience"], key=gate_episode["salience"].get) == str(gate_episode["choice"]),
            "outside": steps_outside(steps, chosen, wait) if first == chosen else None}


def run_facts(run: str, wait: int) -> List[Dict[str, object]]:
    """The facts of every episode of one run, in order."""
    with open(os.path.join(run, "value_gate.json"), encoding="utf-8") as handle:
        gate_episodes = json.load(handle)["episodes"]
    folders = sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))
    if len(folders) != len(gate_episodes):
        raise ValueError("%s: %d episodes and %d gate records" % (run, len(folders), len(gate_episodes)))
    episodes = [read_steps(folder) for folder in folders]
    levels = [steps[0]["internals"]["info_after_step"]["_truth_decoy_level"] for steps in episodes]
    band = light_band(gate_episodes, levels, run)
    return [dict(episode_facts(steps, gate, band, wait), run=os.path.basename(os.path.normpath(run)), episode=index)
            for index, (steps, gate) in enumerate(zip(episodes, gate_episodes))]


def count(facts: List[Dict[str, object]]) -> str:
    """"hits of n = share" of the episodes whose first object is the chosen one."""
    hits = sum(fact["first"] == fact["chosen"] for fact in facts)
    return "%d of %d = %.3f" % (hits, len(facts), hits / len(facts)) if facts else "0 of 0"


def report(facts: List[Dict[str, object]], wait: int) -> None:
    misses = [fact for fact in facts if fact["first"] != fact["chosen"]]
    print("episodes %d; candidates at the choice: %s" % (len(facts), dict(collections.Counter(f["candidates"] for f in facts))))
    print("reach, chosen object first: %s" % count(facts))
    for name in OBJECTS:
        print("  %s chosen: %s" % (name, count([fact for fact in facts if fact["chosen"] == name])))
    print("  quieter tone chosen: %s; louder tone chosen: %s" % (
        count([fact for fact in facts if fact["quieter_chosen"]]), count([fact for fact in facts if not fact["quieter_chosen"]])))
    for run in sorted({fact["run"] for fact in facts}):
        mine = [fact for fact in facts if fact["run"] == run]
        print("  %s: %s, miss episodes %s" % (run, count(mine), [f["episode"] for f in mine if f["first"] != f["chosen"]]))
    print("misses %d: other object first %d, no contact %d" % (
        len(misses), sum(fact["first"] is not None for fact in misses), sum(fact["first"] is None for fact in misses)))
    outside = [fact["outside"] for fact in facts if fact["first"] == fact["chosen"]]
    print("steps outside the chosen object in the %d steps after the first contact: %s (None: the episode ended first)" % (
        wait, dict(collections.Counter(outside))))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="run folders with value_gate.json and episodes/")
    parser.add_argument("--wait", type=int, default=20, help="steps after the first contact that the stay check reads")
    args = parser.parse_args(argv)
    facts = [fact for run in args.runs for fact in run_facts(run, args.wait)]
    report(facts, args.wait)


if __name__ == "__main__":
    main()
