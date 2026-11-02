"""Does the Stage 1 valence of -1.0 after a food step change the bids or the leader? Read-only.

docs/results/core_self_appraisal_2026_11.md found that the valence computed before the action is
-1.0 at every step after a food step, because both calls before the action pass an environment
reward of 0.0 (scripts/training/train_rlhf.py:1146 and :1532). The valence of 1146 is the PAD state
the settle loop gives to the modulator (:1315). The owner asked on 2026-11-02 for this check before
any decision about a fix. Written and pushed before its run.

DATA. runs/vg_learn_s601, s602 and s603, existence drive on, `learned_valence` off, so the modulator
uses its fixed sets (THREAT_MODULES is body, vision and audio). The training script builds the
modulator with its defaults and `ablate_existence_bias` only (train_rlhf.py:444). AFTER_FOOD steps are
the steps t >= 1 whose step t - 1 had a reward of at least FOOD_REWARD.

ARMS, at every AFTER_FOOD step, with the real AffectiveModulator on the recorded interoception and on
the bids that enter the last settle cycle (`settle.final_bids`), whose modulated bids are the ones
recorded. The first version used the raw bids. Its smoke pass reproduced the recorded bids at only
4.3 percent of the steps (R0), because the reentrant cycles change the bids before the last
competition. That change came before any counted run.
  recorded  the recorded PAD state (valence -1.0).
  neutral   the same PAD state with valence 0.0, as if the reward change were not counted.

  R0  VOID rule. With the recorded arm, the modulator reproduces the recorded modulated bids within
      1e-6 at REPRODUCE_SHARE or more of the AFTER_FOOD steps of a run. Else that seed is VOID,
      because the recorded bids then come from a later reentrant cycle or another PAD state.
  V1  Reported. The mean and the largest absolute bid change between the arms, per module.
  V2  The share of AFTER_FOOD steps at which the leader, the module with the largest modulated
      bid, differs between the arms. A tie for the largest bid is its own leader value, so a tie
      in one arm only counts as a change.

DECISION RULE, stated before the run. If V2 is 0.0 at 3 of 3 seeds, the inverted valence does not
change the leader in these runs, so the defect is recorded and no change of the default is
proposed. If V2 is above 0.0 at one seed or more, a default-off fix is proposed for the owner.

SMOKE. runs/rv_stay_s701 first, not counted, to show the check reaches every step it reads.

LIMITS. Open loop, one settle cycle. The leader of the modulated bids is not the winner, which comes
after binding and ignition. The other readers of the valence (the memory query at :1183, the
self-model at :1265 and :1426, the logged metrics at :2148) are not measured here.

Usage:
    python -m scripts.analysis.probe_valence_after_food --runs runs --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from models.emotion.affective_modulator import AffectiveModulator
from scripts.analysis.probe_content_map_sensitivity import episode_folder

FOOD_REWARD = 0.5
REPRODUCE_SHARE = 0.9
TOLERANCE = 1e-6
COUNTED = tuple("vg_learn_s%d" % seed for seed in (601, 602, 603))
SMOKE_RUN = "rv_stay_s701"


def after_food_steps(steps: List[dict]) -> List[int]:
    """The steps whose previous step had food."""
    return [t for t in range(1, len(steps)) if float(steps[t - 1]["reward"]) >= FOOD_REWARD]


def leader(bids: Dict[str, float]) -> str:
    """The module with the largest bid, or "tie" when two or more share it."""
    top = max(bids.values())
    names = [name for name, bid in bids.items() if bid == top]
    return names[0] if len(names) == 1 else "tie"


def compare_step(modulator: AffectiveModulator, step: dict) -> dict:
    """Both arms at one step, and whether the recorded arm reproduces the recorded bids."""
    pad, bids = dict(step["internals"]["emotion"]), step["internals"]["settle"]["final_bids"]
    recorded, _ = modulator.modulate(bids, pad, step["interoception"])
    neutral, _ = modulator.modulate(bids, dict(pad, valence=0.0), step["interoception"])
    logged = step["internals"]["modulated_bids"]
    return {"reproduced": all(abs(recorded[name] - logged[name]) <= TOLERANCE for name in recorded),
            "changes": {name: recorded[name] - neutral[name] for name in recorded},
            "leader_changed": leader(recorded) != leader(neutral)}


def check_run(run: str) -> dict:
    """R0, V1 and V2 for one run."""
    modulator = AffectiveModulator({"ablate_existence_bias": False})
    compared = []
    for number in range(len(os.listdir(os.path.join(run, "episodes")))):
        with open(os.path.join(episode_folder(run, number), "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle]
        compared += [compare_step(modulator, steps[t]) for t in after_food_steps(steps)]
    share = float(np.mean([c["reproduced"] for c in compared]))
    names = sorted(compared[0]["changes"])
    changes = {name: np.array([abs(c["changes"][name]) for c in compared]) for name in names}
    return {"after_food_steps": len(compared), "reproduced_share": share, "void": share < REPRODUCE_SHARE,
            "change_mean": {name: float(changes[name].mean()) for name in names},
            "change_max": {name: float(changes[name].max()) for name in names},
            "leader_changed_share": float(np.mean([c["leader_changed"] for c in compared]))}


def decision(per_run: Dict[str, dict]) -> str:
    """The pre-stated decision rule over the counted seeds."""
    if any(r["void"] for r in per_run.values()):
        return "VOID at %d seed(s)" % sum(r["void"] for r in per_run.values())
    if all(r["leader_changed_share"] == 0.0 for r in per_run.values()):
        return "NO LEADER CHANGE at 3 of 3 seeds. Record the defect, and propose no change of the default"
    return "LEADER CHANGES. Propose a default-off fix to the owner"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--smoke", action="store_true", help="The smoke run only, not counted.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.smoke:
        report = {"smoke": {SMOKE_RUN: check_run(os.path.join(args.runs, SMOKE_RUN))}}
    else:
        per_run = {name: check_run(os.path.join(args.runs, name)) for name in COUNTED}
        report = {"runs": per_run, "decision": decision(per_run)}
    report["settings"] = {"food_reward": FOOD_REWARD, "reproduce_share": REPRODUCE_SHARE, "tolerance": TOLERANCE}
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
