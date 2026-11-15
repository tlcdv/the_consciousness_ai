"""The second count of the first object of each episode, for the competing goals gate.

`docs/decisions/2026_11_08_competing_goals_gate.md`, Gate section. Read-only. The definition is that
of the first count: within the first LIMIT steps of an episode, the first object closer than CONTACT
room pixels, the nearer one when both are in contact at that step, and None without a contact.

This file is separate code. It imports no other analysis script of this repository and only the
standard library. It reads `_truth_light_offset` and `_truth_decoy_offset` of each step record.

LIMIT OF THE SECOND COUNT. It reads the fields that the first count reads. So it finds an error of the
code of a count. It does not find an error of the recorded offsets, because no second field of a step
record has the place of the decoy.

Usage:
    python -m scripts.analysis.count_first_object --run runs/<run>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple

CONTACT, LIMIT = 60.0, 60

Offsets = Tuple[Sequence[float], Sequence[float]]


def first_contact(offsets: Iterable[Offsets]) -> Optional[str]:
    """The first object in contact over (light offset, decoy offset) pairs, one pair for each step."""
    for index, (light, decoy) in enumerate(offsets):
        if index >= LIMIT:
            break
        to_light = math.hypot(light[0], light[1])
        to_decoy = math.hypot(decoy[0], decoy[1])
        if to_light < CONTACT and to_light <= to_decoy:
            return "light"
        if to_decoy < CONTACT:
            return "decoy"
    return None


def episode_offsets(path: str) -> List[Offsets]:
    """The offset pairs of the first LIMIT step records of one `steps.jsonl`."""
    pairs: List[Offsets] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if len(pairs) == LIMIT:
                break
            if not line.strip():
                continue
            info = json.loads(line)["internals"]["info_after_step"]
            pairs.append((info["_truth_light_offset"], info["_truth_decoy_offset"]))
    return pairs


def count_run(run: str) -> List[Optional[str]]:
    """The first object of each episode of a run folder, in the order of the episode folders."""
    root = os.path.join(run, "episodes")
    names = sorted(name for name in os.listdir(root) if name.startswith("ep_"))
    return [first_contact(episode_offsets(os.path.join(root, name, "steps.jsonl"))) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="The run folder.")
    options = parser.parse_args()
    print(json.dumps(count_run(options.run)))


if __name__ == "__main__":
    main()
