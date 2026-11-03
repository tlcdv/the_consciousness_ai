"""Do vision and hearing carry what a relation of content needs, the identity of each object? Read-only.

docs/decisions/2026_11_02_content_relation_design.md, design B2. In the superior colliculus, which senses
go together is learned from signals that occur together, with no reward (Yu, Rowland and Stein 2010), and
a learned coupling encodes that two features go together (Chen and Spence 2017). This probe asks the
necessary condition of B2 on recorded vectors. It builds no mechanism in the agent.

HISTORY. Version 1 (commit 60f32aa) asked if a learned relation tells related from unrelated steps. Its
smoke run stopped it by its own rule: of the four cells of (object seen, louder source), two were empty
(102 and 198 steps against 0 and 0). In this room the object in view is near and the other is 120 to 176
pixels away, so the seen object is always the louder one. Version 2 below was written after that smoke
run and pushed before any counted run. The owner approved it on 2026-11-02.

WHY THE IDENTITY READS DECIDE. Because the seen object is always the louder one, any learner of
co-occurrence pairs the colour of an object with its tone as soon as vision carries the object and
hearing carries the louder tone. So a two-choice test of the learned relation passes almost by
construction when V1 and A1 hold. The checks that can fail are V1 and A1. The two-choice test is reported
as a confirmation and is not evidence for B2.

RUNS. The Gate B7 command (the recorded argv of runs/f2_py312_s73) with these changes, on Python 3.12.10,
from a checkout that has the room options of the content relation test bed:
  --episodes 20 --existence-drive off
  --dark-room-decoy-sound --dark-room-decoy-visible --dark-room-decoy-colour 200,255,255
  --dark-room-decoy-level 0.5,3.0 --dark-room-sound-direction off
Counted seeds 1101, 1102 and 1103, one run at a time, in runs/cr_s1101 to runs/cr_s1103. The smoke run is
runs/cr_smoke_s1199, seed 1199, 2 episodes, made for version 1 with the same command and discarded. The
drive is off because the movement records were the same with the drive on and off
(docs/results/reversal_drive_off_2026_11.md), and off adds no noxious channel. The decoy colour is a pale
cyan of almost the light's luminance (243 against 251), so the two objects differ in hue. The sound has
no direction, so place cannot tell which object a tone comes from.

STEPS. A step t counts when exactly one object was in view in the scene the agent received, which is the
ground truth of step t - 1, as in probe_binding_selectivity, and when that object was the louder source,
from binaural_amplitude of each distance times the decoy level, as in probe_binding_selectivity.condition.
SEEN names that object (0 light, 1 decoy). v = tectum_content and a = audio_content of step t, recorded in
live runs, which train.

CHECKS. Whole episodes are held out in FOLDS folds, and each AUC is computed inside its fold and averaged.
  V0  VOID rule. Each seed has at least MIN_CELL counted steps with each object seen. Else that seed is
      VOID. A seed is also VOID when no held-out fold shows both objects, so no read can be computed.
  V1  Vision carries the object. The ridge read of v gives SEEN at a mean fold AUC of IDENTITY_AUC or
      more, above the p95 of SHUFFLES reads with the labels of other episodes, and above the read of the
      episode number alone. The episode number is the control for time, because the modules train during
      the run and each episode tends to show one object.
  A1  Hearing carries the louder tone. The same three conditions for the ridge read of a.
  Reported, no verdict. The two-choice test. On a held-out fold, W is the cross-covariance of the
  centred v and a of the training episodes, learned with no label, and r = (v - mean_v) W (a - mean_a).
  Each item is a held-out step. It is correct when r with the audio of another step of the same object
  beats r with the audio of a step of the other object, both partners from the same fold. Items are
  balanced between the two objects, so a preference for one sound scores 0.5. Its null learns W from the
  audio of other training steps. Also reported, the cell counts of every seed.

RULE. The identity condition of B2 is met when V1 and A1 hold at 3 of 3 seeds, and it FAILS when V1 or A1
fails at 2 or more live seeds. Otherwise it is UNDECIDED. A VOID seed counts as neither.

STOP RULE OF THE SMOKE. The counted runs start only if the smoke run, with its counts scaled from 2 to 20
episodes, has MIN_CELL or more steps with each object seen.

WHAT EACH OUTCOME POINTS TO.
  V1 fails. The tectum content does not carry the difference of hue at almost equal luminance, so no
  relation of content can use vision in this agent.
  A1 fails. The hearing vector does not tell which tone is louder.
  Both hold. The identities a relation needs are there. A real test of the relation needs steps where
  the seen object and the louder sound conflict, which this room does not make. That needs another change
  of the room, the owner's decision.

LIMITS. Open loop and offline. The settings above are the only ones tried.

Usage:
    python -m scripts.analysis.probe_content_relation --runs runs --smoke --output-json smoke.json
    python -m scripts.analysis.probe_content_relation --runs runs --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np

from scripts.analysis.probe_content_map_sensitivity import FOLDS, episode_folder, swapped_labels
from scripts.analysis.probe_core_self_appraisal import fold_mean_auc, seed_rule
from simulations.environments.audio_mixin import binaural_amplitude

MIN_CELL = 50
IDENTITY_AUC = 0.7
SHUFFLES = 20
COUNTED = tuple("cr_s%d" % seed for seed in (1101, 1102, 1103))
SMOKE_RUN, SMOKE_EPISODES, TARGET_EPISODES = "cr_smoke_s1199", 2, 20
OBJECTS = ("light", "decoy")


def seen_object(info: dict) -> Optional[str]:
    """The object in view, or None when both or neither are in view."""
    light, decoy = bool(info.get("_truth_light_in_view")), bool(info.get("_truth_decoy_in_view"))
    if light == decoy:
        return None
    return "light" if light else "decoy"


def louder_source(info: dict) -> Optional[str]:
    """The source that was louder at the agent, or None at equal loudness or without a decoy."""
    if info.get("_truth_decoy_offset") is None:
        return None
    light = binaural_amplitude(float(np.hypot(*info["_truth_light_offset"])))
    decoy = float(info.get("_truth_decoy_level", 1.0)) * binaural_amplitude(float(np.hypot(*info["_truth_decoy_offset"])))
    if light == decoy:
        return None
    return "light" if light > decoy else "decoy"


def read_episode(folder: str) -> Dict[str, np.ndarray]:
    """Steps with one object in view, with v, a, SEEN and LOUDER (0 light, 1 decoy)."""
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle]
    tables = np.load(os.path.join(folder, "vectors.npz"))
    rows = {name: {int(s): i for i, s in enumerate(tables[name + "__steps"])}
            for name in ("tectum_content", "audio_content")}
    kept = {"v": [], "a": [], "seen": [], "louder": []}
    for t in range(1, len(steps)):
        info = steps[t - 1]["internals"]["info_after_step"]
        seen, louder = seen_object(info), louder_source(info)
        if seen is None or louder is None or any(t not in rows[name] for name in rows):
            continue
        kept["v"].append(np.asarray(tables["tectum_content"][rows["tectum_content"][t]], dtype=float).ravel())
        kept["a"].append(np.asarray(tables["audio_content"][rows["audio_content"][t]], dtype=float).ravel())
        kept["seen"].append(OBJECTS.index(seen))
        kept["louder"].append(OBJECTS.index(louder))
    return {name: np.array(column) for name, column in kept.items()}


def read_run(run: str, episodes: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Every step of a run with one object in view, joined, with the episode number in `group`."""
    count = episodes if episodes is not None else len(os.listdir(os.path.join(run, "episodes")))
    parts = []
    for number in range(count):
        part = read_episode(episode_folder(run, number))
        if len(part["seen"]):
            part["group"] = np.full(len(part["seen"]), number)
            parts.append(part)
    if not parts:
        raise ValueError("no step in %s had exactly one object in view" % run)
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def matched(record: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The counted steps, where the seen object is also the louder source."""
    keep = record["seen"] == record["louder"]
    return {name: column[keep] for name, column in record.items()}


def cell_counts(seen: np.ndarray, louder: np.ndarray) -> Dict[str, int]:
    return {"%s_seen_%s_louder" % (OBJECTS[s], OBJECTS[l]): int(((seen == s) & (louder == l)).sum())
            for s in (0, 1) for l in (0, 1)}


def identity_check(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict:
    """V1 or A1. The read, its null with the labels of other episodes, and the episode-number control."""
    read = fold_mean_auc(x, labels, groups)
    null = [fold_mean_auc(x, swapped_labels(labels, groups, rng), groups) for _ in range(SHUFFLES)]
    episode = fold_mean_auc(groups.astype(float), labels, groups)
    null_p95 = float(np.nanpercentile(null, 95))
    holds = read >= IDENTITY_AUC and read > null_p95 and read > episode
    return {"auc": read, "null_p95": null_p95, "episode_number_auc": episode, "holds": bool(holds)}


def relation_scores(v_train: np.ndarray, a_train: np.ndarray, v_test: np.ndarray, a_test: np.ndarray) -> np.ndarray:
    """r = (v - mean_v) W (a - mean_a), with W the cross-covariance of the training pairs."""
    mean_v, mean_a = v_train.mean(0), a_train.mean(0)
    w = (v_train - mean_v).T @ (a_train - mean_a) / len(v_train)
    return np.einsum("ij,jk,ik->i", v_test - mean_v, w, a_test - mean_a)


def two_choice_items(seen: np.ndarray, rng: np.random.Generator):
    """Balanced items, each with a partner of the same object and a partner of the other object."""
    by_object = [np.flatnonzero(seen == s) for s in (0, 1)]
    size = min(len(group) for group in by_object)
    if size < 2:
        return []
    items = []
    for s in (0, 1):
        for i in rng.choice(by_object[s], size, replace=False):
            same = rng.choice(by_object[s][by_object[s] != i])
            items.append((int(i), int(same), int(rng.choice(by_object[1 - s]))))
    return items


def two_choice_reading(record: Dict[str, np.ndarray], rng: np.random.Generator, broken: bool = False) -> Optional[float]:
    """Reported only. Mean over folds of the share of items whose same-object partner wins (ties 0.5)."""
    episodes, readings = np.unique(record["group"]), []
    for k in range(FOLDS):
        test = np.isin(record["group"], episodes[k::FOLDS])
        items = two_choice_items(record["seen"][test], np.random.default_rng((0, k)))
        if test.all() or not items:
            continue
        a_train = record["a"][~test][rng.permutation(int((~test).sum()))] if broken else record["a"][~test]
        v_test, a_test = record["v"][test], record["a"][test]
        chosen, same, other = (np.array(column) for column in zip(*items))
        win_same = relation_scores(record["v"][~test], a_train, v_test[chosen], a_test[same])
        win_other = relation_scores(record["v"][~test], a_train, v_test[chosen], a_test[other])
        readings.append(float(np.mean((win_same > win_other) + 0.5 * (win_same == win_other))))
    return float(np.mean(readings)) if readings else None


def check_run(run: str, episodes: Optional[int] = None) -> dict:
    """V0, V1, A1 and the reported two-choice test for one run."""
    rng = np.random.default_rng(0)
    every = read_run(run, episodes)
    record = matched(every)
    counts = {"light_seen": int((record["seen"] == 0).sum()), "decoy_seen": int((record["seen"] == 1).sum())}
    report = {"cells": cell_counts(every["seen"], every["louder"]), "counted": counts,
              "episodes_with_each_object": {OBJECTS[s]: int(len(np.unique(record["group"][record["seen"] == s])))
                                            for s in (0, 1)}}
    if min(counts.values()) < MIN_CELL:
        return dict(report, void="V0")
    report["V1"] = identity_check(record["v"], record["seen"], record["group"], rng)
    report["A1"] = identity_check(record["a"], record["seen"], record["group"], rng)
    if any(np.isnan(report[name][key]) for name in ("V1", "A1") for key in ("auc", "episode_number_auc")):
        return dict(report, void="unreadable: no held-out fold shows both objects")
    null = [two_choice_reading(record, rng, broken=True) for _ in range(SHUFFLES)]
    report["two_choice"] = {"reading": two_choice_reading(record, rng),
                            "null_p95": float(np.percentile([n for n in null if n is not None], 95))}
    return dict(report, void=None)


def verdict(per_run: Dict[str, dict]) -> dict:
    """The pre-stated rule over the counted seeds."""
    live = [r for r in per_run.values() if r["void"] is None]
    held = sum(r["V1"]["holds"] and r["A1"]["holds"] for r in live)
    return {"void_seeds": len(per_run) - len(live), "V1_and_A1_seeds": held,
            "identity_condition_of_B2": seed_rule(held, len(per_run), len(live))}


def smoke_stop(counts: Dict[str, int]) -> bool:
    """True when the smoke counts, scaled to TARGET_EPISODES, fall below MIN_CELL for an object."""
    return min(counts.values()) * TARGET_EPISODES / SMOKE_EPISODES < MIN_CELL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--smoke", action="store_true", help="The smoke run only, not counted.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.smoke:
        record = matched(read_run(os.path.join(args.runs, SMOKE_RUN), SMOKE_EPISODES))
        counts = {"light_seen": int((record["seen"] == 0).sum()), "decoy_seen": int((record["seen"] == 1).sum())}
        report = {"smoke": {"counted": counts, "stop": smoke_stop(counts)}}
    else:
        per_run = {name: check_run(os.path.join(args.runs, name)) for name in COUNTED}
        report = {"runs": per_run, "verdict": verdict(per_run)}
    report["settings"] = {"min_cell": MIN_CELL, "identity_auc": IDENTITY_AUC, "shuffles": SHUFFLES, "folds": FOLDS}
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
