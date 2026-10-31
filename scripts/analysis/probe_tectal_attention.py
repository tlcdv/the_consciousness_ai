"""Gate of the tectal attention gain (docs/decisions/2026_10_17_tectal_attention.md): A1 to A3.

Read-only probe on covert runs (`--tectal-orienting group --tectal-orienting-covert`), one arm with
`--tectal-attention gain` (link) and one without (no link), paired by seed. A step counts when the
binding record of the previous step of the same episode holds a group (the group the gain used).
The selected object is the object nearer to the centroid of that group (the motor map's cell
offsets, the `_truth_*_offset` keys), and the label is its direction from the agent (a unit
vector). A ridge decoder (alpha 100, standardized vectors, 5 folds that each hold whole episodes
out) predicts the direction from `tectum_content` (every counted step) and from `broadcast` (the
counted steps that ignited). Score: the mean cosine between the predicted and the true direction
(chance 0).

  A1: link arm, content score >= 0.5.
  A2: link arm, broadcast score >= 0.5.
  A3: link above no link for both vectors, both no-link scores <= 0.2, and a permutation test
      (directions shuffled across episodes, 200 shuffles) gives p < 0.05 for both link scores.

Usage:
    python -m scripts.analysis.probe_tectal_attention --link runs/ta_link_s801 ... \
        --nolink runs/ta_nolink_s801 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.core.tectal_motor_map import cell_offsets
from scripts.analysis.probe_value_gate import seed_label, verdict

ALPHA, FOLDS, SHUFFLES = 100.0, 5, 200
SCORE, NULL = 0.5, 0.2


def selected_object(cells: List[int], info: dict) -> Optional[str]:
    """The object nearer to the centroid of the group's cells, or None without cells."""
    if not cells:
        return None
    dx, dy = cell_offsets()
    centre = np.array([dx.reshape(-1)[cells].mean(), dy.reshape(-1)[cells].mean()])
    distance = {name: np.linalg.norm(centre - np.asarray(info["_truth_%s_offset" % name], dtype=float))
                for name in ("light", "decoy")}
    return min(distance, key=distance.get)


def direction(info: dict, name: str) -> np.ndarray:
    """Unit vector from the agent to the object."""
    offset = np.asarray(info["_truth_%s_offset" % name], dtype=float)
    return offset / np.linalg.norm(offset)


def episode_folds(groups: np.ndarray, folds: int = FOLDS) -> list:
    """(train, test) index pairs; each fold holds whole episodes out."""
    episodes = np.unique(groups)
    pairs = []
    for k in range(folds):
        held = episodes[k::folds]
        test = np.isin(groups, held)
        pairs.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return pairs


def direction_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = FOLDS,
                    alpha: float = ALPHA) -> float:
    """Mean cosine between the cross-validated prediction and the true direction."""
    cosines = []
    for train, test in episode_folds(groups, folds):
        if len(train) == 0 or len(test) == 0:
            continue
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        p = model.predict(scaler.transform(x[test]))
        p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
        cosines.extend(np.sum(p * y[test], axis=1).tolist())
    return float(np.mean(cosines))


def permutation_p(x: np.ndarray, y: np.ndarray, groups: np.ndarray, observed: float,
                  shuffles: int = SHUFFLES, seed: int = 0) -> float:
    """One-sided p of `observed` against scores with the directions shuffled across episodes."""
    rng = np.random.default_rng(seed)
    episodes = np.unique(groups)
    first = {e: y[np.flatnonzero(groups == e)[0]] for e in episodes}
    at_least = 0
    for _ in range(shuffles):
        mapping = dict(zip(episodes, rng.permutation(episodes)))
        shuffled = np.array([first[mapping[g]] for g in groups])
        at_least += direction_score(x, shuffled, groups) >= observed
    return (1 + at_least) / (1 + shuffles)


def run_samples(run: str) -> dict:
    """{"content": (x, y, groups), "broadcast": (x, y, groups)} of the counted steps of a run."""
    with open(os.path.join(run, "map_binding.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    rows = {"content": ([], [], []), "broadcast": ([], [], [])}
    start = 0
    for e, folder in enumerate(sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle]
        vectors = np.load(os.path.join(folder, "vectors.npz"))
        content = dict(zip(vectors["tectum_content__steps"].tolist(), vectors["tectum_content"]))
        broadcast = dict(zip(vectors["broadcast__steps"].tolist(), vectors["broadcast"]))
        for t in range(1, len(steps)):
            info = steps[t]["internals"]["info_after_step"]
            name = selected_object(history[start + t - 1].get("group", []), info)
            if name is None:
                continue
            label = direction(info, name)
            for key, table in (("content", content), ("broadcast", broadcast)):
                if t not in table or (key == "broadcast" and not steps[t].get("ignited")):
                    continue
                rows[key][0].append(np.asarray(table[t], dtype=float).ravel())
                rows[key][1].append(label)
                rows[key][2].append(e)
        start += len(steps)
    if start != len(history):
        raise ValueError("%s: %d binding records for %d steps" % (run, len(history), start))
    return {key: tuple(np.array(part) for part in value) for key, value in rows.items()}


def checks(link: Dict[str, list], nolink: Dict[str, list]) -> Dict[str, List[bool]]:
    """A1 to A3 per seed, from the scores and the link arm's permutation p."""
    holds = {"A1": [], "A2": [], "A3": []}
    for i in range(len(link["content"])):
        holds["A1"].append(link["content"][i] >= SCORE)
        holds["A2"].append(link["broadcast"][i] >= SCORE)
        holds["A3"].append(link["content"][i] > nolink["content"][i] and link["broadcast"][i] > nolink["broadcast"][i]
                           and nolink["content"][i] <= NULL and nolink["broadcast"][i] <= NULL and link["p"][i] < 0.05)
    return holds


def arm_scores(runs: List[str], with_p: bool) -> tuple:
    scores = {"content": [], "broadcast": [], "p": [], "samples": []}
    for run in runs:
        data = run_samples(run)
        ps = []
        for key in ("content", "broadcast"):
            x, y, groups = data[key]
            score = direction_score(x, y, groups)
            scores[key].append(score)
            if with_p:
                ps.append(permutation_p(x, y, groups, score))
        scores["p"].append(max(ps) if ps else None)
        scores["samples"].append({key: int(len(data[key][1])) for key in data})
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--link", nargs="+", required=True)
    parser.add_argument("--nolink", nargs="+", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if len(args.link) != len(args.nolink):
        parser.error("each arm needs one run for each seed, in the same order")
    names = [seed_label(run) for run in args.link]
    link, nolink = arm_scores(args.link, with_p=True), arm_scores(args.nolink, with_p=False)
    for name, i in zip(names, range(len(names))):
        print("%s | link: content %.3f, broadcast %.3f, p %.4f, samples %s | no link: content %.3f, broadcast %.3f"
              % (name, link["content"][i], link["broadcast"][i], link["p"][i], link["samples"][i],
                 nolink["content"][i], nolink["broadcast"][i]))
    holds = checks(link, nolink)
    result = {"seeds": names, "link": link, "nolink": nolink, "holds": holds, "verdict": verdict(holds)}
    print("holds:", holds)
    print("VERDICT:", result["verdict"])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
