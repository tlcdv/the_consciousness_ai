"""Cheap checks of the Stage 2 appraisal designs, on recorded dark room sessions. Read-only.

docs/decisions/2026_11_02_core_self_appraisal_design.md compares the designs. This probe only reads
recorded sessions. Written and pushed before its first run.

DATA. runs/vg_learn_s601, s602 and s603, 40 episodes of 200 steps each, existence drive on. The
vectors were recorded in live runs, which train, so the module mode is the mode of the runs. A smoke
pass comes first, on all 20 episodes of runs/s2c_s125, a binding run with the decoy sound. Its
numbers are not counted. The first smoke choice, episodes 0 and 1 of runs/ta_nolink_s801, was VOID
at its run (0 food steps), and that run has no food step in any of its 40 episodes. The second,
episodes 0 to 4 of runs/s2c_s125, was VOID too (39 positive steps), because its food came after
step 140. Neither reached the checks. Only the smoke data changed. Nothing counted changed.

LABEL. FOOD_AHEAD at step t is 1 when a recorded reward of at least FOOD_REWARD comes at one of the
steps t + 1 to t + WINDOW, else 0. A food step gives a reward of about 1.0 and other steps about
-0.01. WINDOW is OUTCOME_STEPS of models/core/value_gate.py (60), the window in which the value gate
reads the outcome of a choice. Steps whose window passes the end of the episode are dropped.

STREAMS at step t, from the step record and the recorded broadcast, which is the tensor the step gave
to the appraisal (scripts/training/train_rlhf.py:1534 and :2114).
  reflex          the recorded Stage 1 valence. Stage 2 never ran, so this is what the agent used.
  phen_valence    the valence of map_state(broadcast, GOAL) of the real PhenomenologicalMapper.
  a_valence       0.6 * reflex + 0.4 * phen_valence, the blend of evaluate_emotion (design A).
  a_dominance     0.3 * the intensity of the same map_state (design A).
  value_estimate  the recorded value of the action core, a stand-in for design C.

CC1. Each stream is a fixed function of the record, so no fit is needed. Its AUC for FOOD_AHEAD over
the counted steps of a run is compared with the p95 of NULL_SHUFFLES AUCs in which every episode
gets the label of another episode (swapped_labels of probe_content_map_sensitivity).
  V0  VOID rule. Each run has at least MIN_STEPS counted steps of each label, else that seed is VOID.
  A1  Sanity, expected to hold, not evidence. a_dominance takes more than one value in each run.
  A2  phen_valence is higher before food. Its AUC is above the null p95, at 3 of 3 seeds.
  A3  a_dominance carries the label in either direction. |AUC - 0.5| is above the null p95 of the
      same quantity, at 3 of 3 seeds.
  Design A PASSES CC1 when A2 holds at 3 of 3 seeds and FAILS when A2 fails at 2 or more seeds.
  reflex, a_valence and value_estimate are read the same way and reported without a verdict.

CC2. A ridge read of the 256 broadcast values (RIDGE_ALPHA, FOLDS, whole episodes held out) gives
held-out scores for FOOD_AHEAD, and their AUC.
  B2  The AUC is above the p95 of RIDGE_SHUFFLES reads with swapped labels, and above the AUC of
      the step index read the same way, at 3 of 3 seeds. The step index control is there because
      food can come at typical times of an episode.
  Design B meets its necessary condition when B2 holds at 3 of 3 seeds and FAILS it when B2 fails
  at 2 or more seeds.

WHAT EACH OUTCOME POINTS TO.
  A2 fails. Design A carries nothing about the coming food, and it is rejected on evidence.
  A2 holds. Three coordinates happen to carry the signal, and design A still has no basis.
  B2 fails. The broadcast does not predict the coming food, so design B has nothing to learn from.
  B2 holds. The necessary condition of design B is met, which is not enough to build it.

LIMITS. Both checks are open loop. A Stage 2 that runs would change the emotion, the bids and so the
broadcast, which these records cannot show. The settings above are the only ones. No other window,
threshold or read is tried after the results.

FOLLOW-UP CC2b, 2026-11-02, approved by the owner and written and pushed before its run. CC1 and
CC2 FAILED their own design (docs/results/core_self_appraisal_2026_11.md). Their label never changed
inside an episode. Before these rules I read only when the first food comes in each episode of the
counted runs. It comes at steps 12 to 21 in all 112 food episodes, with a median of 14.
  DATA. The same three runs. Only the steps before the first food of each food episode count.
       Episodes without food are left out, because they are the early episodes.
  LABEL. NEAR_FOOD at step t is 1 when the first food comes within APPROACH_K steps after t, else 0.
       APPROACH_K = 7 is half of the median of 14, so a typical approach holds both labels. It is an
       assumed value set from the timing of the food alone.
  READ. The ridge read of the broadcast above, whole episodes held out (FOLDS). The AUC is computed
       inside each fold, where one model scores every step, and averaged over the folds. This
       removes the mixing of fold intercepts that made the pooled AUC of CC2 unreadable.
  B2b  holds at a seed when the mean fold AUC of the broadcast read is above the p95 of
       RIDGE_SHUFFLES reads with swapped labels and above the mean fold AUC of the step index read
       the same way. Design B meets its necessary condition when B2b holds at 3 of 3 seeds and FAILS
       it when B2b fails at 2 or more seeds. VOID rule V0 with MIN_STEPS applies to the new label.
  The step index is a strong control here, because the first food comes at nearly the same step in
  every episode. A pass shows that the broadcast carries how near the food is beyond the time since
  the start. That is a necessary condition of design B only. A fail removes design B for this
  broadcast. This is the only follow-up label. Smoke on runs/rv_stay_s701 first, not counted. The
  first smoke choice, runs/s2c_s125, was VOID at its run (4 food episodes, 50 approach steps).

Usage:
    python -m scripts.analysis.probe_core_self_appraisal --runs runs --output-json out.json
    python -m scripts.analysis.probe_core_self_appraisal --runs runs --smoke --output-json smoke.json
    python -m scripts.analysis.probe_core_self_appraisal --runs runs --follow-up --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from models.core.qualia_mapper import PhenomenologicalMapper
from models.core.value_gate import OUTCOME_STEPS
from scripts.analysis.probe_content_map_sensitivity import FOLDS, RIDGE_ALPHA, episode_folder, swapped_labels
from scripts.analysis.probe_content_map_sensitivity import NULL_SHUFFLES as RIDGE_SHUFFLES

WINDOW = OUTCOME_STEPS
FOOD_REWARD = 0.5
GOAL = (1.0, -1.0, 1.0)  # the literal goal of scripts/training/train_rlhf.py:1314
NULL_SHUFFLES = 200
MIN_STEPS = 100
COUNTED = tuple("vg_learn_s%d" % seed for seed in (601, 602, 603))
SMOKE_RUN, SMOKE_EPISODES = "s2c_s125", None  # None reads every episode
STREAMS = ("reflex", "phen_valence", "a_valence", "a_dominance", "value_estimate")
APPROACH_K = 7
FOLLOW_UP_SMOKE_RUN = "rv_stay_s701"


def food_ahead(rewards: np.ndarray, window: int = WINDOW, threshold: float = FOOD_REWARD):
    """Labels and the mask of steps whose whole window lies inside the episode."""
    count = len(rewards)
    labels = np.zeros(count, dtype=int)
    keep = np.arange(count) + window < count
    for t in np.flatnonzero(keep):
        labels[t] = int((rewards[t + 1:t + 1 + window] >= threshold).any())
    return labels, keep


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a positive step scores above a negative one, ties counted as one half."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both labels, got %d positive and %d negative" % (positives, negatives))
    unique, first, counts = np.unique(np.sort(scores), return_index=True, return_counts=True)
    ranks = (first + (counts + 1) / 2.0)[np.searchsorted(unique, scores)]  # mean rank of each tie
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def design_a_streams(broadcasts: np.ndarray, reflex: np.ndarray, mapper=None) -> Dict[str, np.ndarray]:
    """phen_valence, a_valence and a_dominance from the real map_state, one broadcast per row."""
    mapper = mapper or PhenomenologicalMapper()
    goal = torch.tensor(GOAL, dtype=torch.float32)
    states = [mapper.map_state(torch.as_tensor(row, dtype=torch.float32), goal) for row in broadcasts]
    phen = np.array([state.valence for state in states])
    intensity = np.array([state.intensity for state in states])
    return {"phen_valence": phen, "a_valence": 0.6 * reflex + 0.4 * phen, "a_dominance": 0.3 * intensity}


def read_episode(folder: str) -> Dict[str, np.ndarray]:
    """The recorded rewards, reflex valence, value estimate and broadcast, at the steps with a broadcast."""
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle]
    tables = np.load(os.path.join(folder, "vectors.npz"))
    where = {int(s): i for i, s in enumerate(tables["broadcast__steps"])}
    rewards = np.array([float(step["reward"]) for step in steps])
    labels, keep = food_ahead(rewards)
    rows = [t for t in range(len(steps)) if keep[t] and t in where]
    internals = [steps[t]["internals"] for t in rows]
    return {"label": labels[rows], "step": np.array(rows),
            "reflex": np.array([float(i["emotion"]["valence"]) for i in internals]),
            "value_estimate": np.array([np.nan if i["value_estimate"] is None else i["value_estimate"]
                                        for i in internals], dtype=float),
            "broadcast": np.asarray(tables["broadcast"], dtype=np.float32)[[where[t] for t in rows]].reshape(len(rows), -1)}


def read_run(run: str, episodes: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """All episodes of a run, joined, with the episode number of each step in `group`."""
    numbers = episodes if episodes is not None else range(len(os.listdir(os.path.join(run, "episodes"))))
    parts = []
    for number in numbers:
        part = read_episode(episode_folder(run, number))
        part["group"] = np.full(len(part["label"]), number)
        parts.append(part)
    joined = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    joined.update(design_a_streams(joined["broadcast"], joined["reflex"]))
    return joined


def stream_check(stream: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                 rng: np.random.Generator, two_sided: bool = False) -> dict:
    """AUC of a fixed stream and the p95 of the same reading with the labels of other episodes."""
    def reading(y):
        score = auc(stream, y)
        return abs(score - 0.5) if two_sided else score
    null = [reading(swapped_labels(labels, groups, rng)) for _ in range(NULL_SHUFFLES)]
    return {"auc": auc(stream, labels), "reading": reading(labels), "null_p95": float(np.percentile(null, 95)),
            "distinct": int(len(np.unique(stream)))}


def held_out_scores(x: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Ridge read with whole episodes held out, FOLDS folds, the score of every step."""
    x = np.asarray(x, dtype=float).reshape(len(labels), -1)
    target = np.where(labels > 0, 1.0, -1.0)
    episodes, scores = np.unique(groups), np.zeros(len(labels))
    for k in range(FOLDS):
        test = np.isin(groups, episodes[k::FOLDS])
        mean, scale = x[~test].mean(0), x[~test].std(0)
        scale = np.where(scale == 0, 1.0, scale)
        a, b = (x[~test] - mean) / scale, (x[test] - mean) / scale
        centre = target[~test].mean()
        weights = np.linalg.solve(a.T @ a + RIDGE_ALPHA * np.eye(a.shape[1]), a.T @ (target[~test] - centre))
        scores[test] = b @ weights + centre
    return scores


def ridge_check(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, steps: np.ndarray,
                rng: np.random.Generator) -> dict:
    """B2. Held-out AUC of the broadcast read, its swapped-label null, and the step index control."""
    null = [auc(held_out_scores(x, y, groups), y) for y in
            (swapped_labels(labels, groups, rng) for _ in range(RIDGE_SHUFFLES))]
    return {"auc": auc(held_out_scores(x, labels, groups), labels), "null_p95": float(np.percentile(null, 95)),
            "step_index_auc": auc(held_out_scores(steps, labels, groups), labels)}


def check_run(run: str, episodes: Optional[List[int]] = None) -> dict:
    """CC1 and CC2 for one run."""
    rng = np.random.default_rng(0)
    record = read_run(run, episodes)
    labels, groups = record["label"], record["group"]
    counts = {"positive": int(labels.sum()), "negative": int((labels == 0).sum())}
    if min(counts.values()) < MIN_STEPS:
        return {"counts": counts, "void": True}
    streams = {}
    for name in STREAMS:
        finite = np.isfinite(record[name])
        streams[name] = stream_check(record[name][finite], labels[finite], groups[finite], rng,
                                     two_sided=(name == "a_dominance"))
        streams[name]["steps_without_value"] = int((~finite).sum())
    return {"counts": counts, "void": False, "streams": streams,
            "broadcast_read": ridge_check(record["broadcast"], labels, groups, record["step"], rng)}


def verdicts(per_run: Dict[str, dict]) -> dict:
    """The pre-stated rules over the counted seeds."""
    live = [r for r in per_run.values() if not r["void"]]
    a1 = all(r["streams"]["a_dominance"]["distinct"] > 1 for r in live)
    a2 = sum(r["streams"]["phen_valence"]["reading"] > r["streams"]["phen_valence"]["null_p95"] for r in live)
    a3 = sum(r["streams"]["a_dominance"]["reading"] > r["streams"]["a_dominance"]["null_p95"] for r in live)
    b2 = sum(r["broadcast_read"]["auc"] > max(r["broadcast_read"]["null_p95"], r["broadcast_read"]["step_index_auc"])
             for r in live)
    seeds, alive = len(per_run), len(live)
    return {"void_seeds": seeds - alive, "A1": a1, "A2_seeds": a2, "A3_seeds": a3, "B2_seeds": b2,
            "design_A_CC1": seed_rule(a2, seeds, alive), "design_B_CC2": seed_rule(b2, seeds, alive)}


def seed_rule(held: int, seeds: int, alive: int) -> str:
    """PASS at every seed, FAIL at 2 or more live seeds, else UNDECIDED. A VOID seed is neither."""
    if held == seeds == alive:
        return "PASS"
    return "FAIL" if alive - held >= 2 else "UNDECIDED"


def near_food(rewards: np.ndarray, k: int = APPROACH_K, threshold: float = FOOD_REWARD):
    """The steps before the first food and their NEAR_FOOD labels, both empty when no step has food."""
    hits = np.flatnonzero(np.asarray(rewards) >= threshold)
    if len(hits) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    steps = np.arange(hits[0])
    return steps, (hits[0] - steps <= k).astype(int)


def read_approach(run: str) -> Dict[str, np.ndarray]:
    """The approach steps of every food episode of a run, with broadcast, label, step and episode."""
    parts = []
    for number in range(len(os.listdir(os.path.join(run, "episodes")))):
        folder = episode_folder(run, number)
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            steps, labels = near_food(np.array([float(json.loads(line)["reward"]) for line in handle]))
        tables = np.load(os.path.join(folder, "vectors.npz"))
        where = {int(s): i for i, s in enumerate(tables["broadcast__steps"])}
        keep = [i for i, t in enumerate(steps) if int(t) in where]
        if keep:
            rows = [where[int(steps[i])] for i in keep]
            parts.append({"label": labels[keep], "step": steps[keep], "group": np.full(len(keep), number),
                          "broadcast": np.asarray(tables["broadcast"], dtype=np.float32)[rows].reshape(len(keep), -1)})
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def fold_aucs(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> List[float]:
    """AUC inside each held-out fold of held_out_scores, for the folds that hold both labels."""
    episodes, aucs = np.unique(groups), []
    for k in range(FOLDS):
        test = np.isin(groups, episodes[k::FOLDS])
        if len(np.unique(labels[test])) == 2:
            aucs.append(auc(scores[test], labels[test]))
    return aucs


def fold_mean_auc(x: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """B2b's reading. One model scores every step of a fold, so fold intercepts never mix."""
    return float(np.mean(fold_aucs(held_out_scores(x, labels, groups), labels, groups)))


def follow_up_run(run: str) -> dict:
    """CC2b for one run."""
    rng = np.random.default_rng(0)
    record = read_approach(run)
    labels, groups = record["label"], record["group"]
    counts = {"positive": int(labels.sum()), "negative": int((labels == 0).sum()),
              "episodes": int(len(np.unique(groups)))}
    if min(counts["positive"], counts["negative"]) < MIN_STEPS:
        return {"counts": counts, "void": True}
    null = [fold_mean_auc(record["broadcast"], y, groups)
            for y in (swapped_labels(labels, groups, rng) for _ in range(RIDGE_SHUFFLES))]
    return {"counts": counts, "void": False, "auc": fold_mean_auc(record["broadcast"], labels, groups),
            "null_p95": float(np.percentile(null, 95)), "step_index_auc": fold_mean_auc(record["step"], labels, groups)}


def follow_up_verdict(per_run: Dict[str, dict]) -> dict:
    """The pre-stated rule of B2b over the counted seeds."""
    live = [r for r in per_run.values() if not r["void"]]
    held = sum(r["auc"] > max(r["null_p95"], r["step_index_auc"]) for r in live)
    return {"void_seeds": len(per_run) - len(live), "B2b_seeds": held,
            "design_B_CC2b": seed_rule(held, len(per_run), len(live))}


def build_report(args) -> dict:
    """The checks the arguments ask for."""
    if args.follow_up:
        names = [FOLLOW_UP_SMOKE_RUN] if args.smoke else list(COUNTED)
        per_run = {name: follow_up_run(os.path.join(args.runs, name)) for name in names}
        return {"smoke": per_run} if args.smoke else {"runs": per_run, "verdicts": follow_up_verdict(per_run)}
    if args.smoke:
        return {"smoke": check_run(os.path.join(args.runs, SMOKE_RUN), SMOKE_EPISODES)}
    per_run = {name: check_run(os.path.join(args.runs, name)) for name in COUNTED}
    return {"runs": per_run, "verdicts": verdicts(per_run)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--smoke", action="store_true", help="The smoke run only, not counted.")
    parser.add_argument("--follow-up", action="store_true", help="CC2b, the follow-up with a label inside episodes.")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_report(args)
    report["settings"] = {"window": WINDOW, "food_reward": FOOD_REWARD, "goal": GOAL, "null_shuffles": NULL_SHUFFLES,
                          "ridge_shuffles": RIDGE_SHUFFLES, "ridge_alpha": RIDGE_ALPHA, "folds": FOLDS,
                          "approach_k": APPROACH_K}
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report.get("verdicts", report), indent=2)[:4000])


if __name__ == "__main__":
    main()
