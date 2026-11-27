"""Read-only check: does the broadcast at the choice step hold which tone is the louder one.

Step P0 of design P. The rules, the thresholds and the controls are in
docs/decisions/2026_11_15_workspace_choice_read.md, written before this file. Nothing is trained and
no run of the agent is made. The check reads saved runs.

One row for each episode, the recorded vector at the step where the value gate chooses. A ridge read
of the label (1 when the tone of the light is the louder one) with whole episodes held out, a null of
200 label orders, the same read of the audio vector as the reference, and a control that reads the
episode number.

Two points that the rules leave to the code, both fixed here before the first run.
1. The folds of a read are made from the labels that the read receives. So a null read makes its own
   folds from the labels in their random order, and each held-out fold has both labels.
2. The inner folds of each outer fold come from a new generator with seed 0.

Usage:
    python -m scripts.analysis.check_choice_read --runs-root runs --summary out/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np

STREAMS = ("broadcast", "audio_content", "tectum_content")
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
OUTER_FOLDS = 5
INNER_FOLDS = 4
FOLD_SEED = 0
NULL_SEED = 1
NULL_COUNT = 200
NULL_PERCENTILE = 95
HOLD_RULE = 0.80
REFERENCE_RULE = 0.90
NULL_MEAN_RANGE = (0.40, 0.60)
SALIENCE_TOLERANCE = 1e-9
STEPS_IN_AN_EPISODE = 200
GROUPS = {"vg_learn": (("vg_learn_s601", "vg_learn_s602", "vg_learn_s603"), 40, True),
          "rv_switch": (("rv_switch_s701", "rv_switch_s702", "rv_switch_s703"), 60, True),
          "cg_gate_benefit": (("cg_gate_benefit_s1301", "cg_gate_benefit_s1302", "cg_gate_benefit_s1303"), 60, False)}


class ControlMismatch(RuntimeError):
    """A control with an exact expected value did not give that value. The run is VOID."""


# The calculations of the read.

def find_choice_step(binding_steps: List[dict]) -> int:
    """The first step with a band and at least one group with cells, as the gate wiring does."""
    for index, record in enumerate(binding_steps):
        if record.get("band") is not None and any(record.get("groups", {}).values()):
            return index
    raise ControlMismatch("an episode has no step with a bound group")


def label_of_level(decoy_level: float) -> int:
    """1 when the tone of the light is the louder one (decoy level below 1), 0 above 1."""
    if decoy_level == 1.0:
        raise ControlMismatch("a decoy level of 1 gives no louder tone")
    return 1 if decoy_level < 1.0 else 0


def make_folds(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """The fold of each episode, with each label divided as equally as possible between the folds."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(labels), dtype=int)
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        if len(members) < folds:
            raise ControlMismatch(f"label {label} has {len(members)} episodes for {folds} folds")
        fold_of[rng.permutation(members)] = np.arange(len(members)) % folds
    return fold_of


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The share of (label 1, label 0) pairs in the right order. A tie counts one half."""
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ControlMismatch("the AUC needs both labels")
    difference = positive[:, None] - negative[None, :]
    return float(((difference > 0).sum() + 0.5 * (difference == 0).sum()) / difference.size)


def standardize(train: np.ndarray, test: np.ndarray):
    """Mean 0 and deviation 1 from the training rows. A dimension with no spread is set to 0."""
    mean, deviation = train.mean(axis=0), train.std(axis=0)
    spread = deviation > 0
    scale = np.where(spread, deviation, 1.0)
    return (train - mean) / scale * spread, (test - mean) / scale * spread


def ridge_predictions(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, penalties) -> np.ndarray:
    """Ridge predictions for each penalty, one column each. The intercept is the mean of y, left out."""
    u, s, vt = np.linalg.svd(x_train, full_matrices=False)
    projected_target = u.T @ (y_train - y_train.mean())
    projected_test = x_test @ vt.T
    shrink = np.stack([s / (s ** 2 + penalty) for penalty in penalties], axis=1)
    return projected_test @ (shrink * projected_target[:, None])


def _held_out_aucs(rows, row_labels, train_rows, test_rows) -> np.ndarray:
    """The AUC of the held-out rows for each penalty."""
    train, test = standardize(rows[train_rows], rows[test_rows])
    predictions = ridge_predictions(train, row_labels[train_rows].astype(float), test, PENALTIES)
    return np.array([auc(predictions[:, column], row_labels[test_rows]) for column in range(len(PENALTIES))])


def choose_penalty(rows, episode_of_row, labels, episodes: np.ndarray) -> float:
    """The penalty with the highest mean AUC over the inner folds. The smallest penalty on a tie."""
    inner_fold = make_folds(labels[episodes], INNER_FOLDS, FOLD_SEED)
    row_labels = labels[episode_of_row]
    fold_aucs = []
    for fold in range(INNER_FOLDS):
        train_rows = np.isin(episode_of_row, episodes[inner_fold != fold])
        test_rows = np.isin(episode_of_row, episodes[inner_fold == fold])
        fold_aucs.append(_held_out_aucs(rows, row_labels, train_rows, test_rows))
    return PENALTIES[int(np.argmax(np.mean(fold_aucs, axis=0)))]


def read_score(rows: np.ndarray, episode_of_row: np.ndarray, labels: np.ndarray) -> dict:
    """The read of the rules: the mean of the five held-out AUC values, with whole episodes held out."""
    rows = np.asarray(rows, dtype=np.float64)
    outer_fold, row_labels = make_folds(labels, OUTER_FOLDS, FOLD_SEED), labels[episode_of_row]
    fold_aucs, penalties = [], []
    for fold in range(OUTER_FOLDS):
        train_episodes = np.flatnonzero(outer_fold != fold)
        penalty = choose_penalty(rows, episode_of_row, labels, train_episodes)
        train_rows = np.isin(episode_of_row, train_episodes)
        aucs = _held_out_aucs(rows, row_labels, train_rows, ~train_rows)
        fold_aucs.append(float(aucs[PENALTIES.index(penalty)]))
        penalties.append(penalty)
    return {"score": float(np.mean(fold_aucs)), "fold_aucs": fold_aucs, "penalties": penalties}


def label_orders(labels: np.ndarray, count: int, seed: int) -> List[np.ndarray]:
    """The labels of the episodes in `count` random orders."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(labels) for _ in range(count)]


def null_scores(rows, episode_of_row, labels, count: int = NULL_COUNT, seed: int = NULL_SEED) -> np.ndarray:
    """The score of the same read for each random order of the labels."""
    return np.array([read_score(rows, episode_of_row, order)["score"] for order in label_orders(labels, count, seed)])


def run_label(reading: Dict[str, float]) -> str:
    """VOID, HOLDS, AT NULL or BETWEEN for one run, from its scores and its null values."""
    if reading["audio"] < REFERENCE_RULE or not reading["audio"] > reading["audio_null"]:
        return "VOID"
    above_null = reading["broadcast"] > reading["broadcast_null"]
    if not above_null:
        return "AT NULL"
    if reading["broadcast"] >= HOLD_RULE and reading["broadcast"] > reading["episode_number"]:
        return "HOLDS"
    return "BETWEEN"


def verdict(labels_of_groups: Dict[str, List[str]]) -> str:
    """The verdict over the counted groups, read VOID first, then FAILED, then HOLDS."""
    every = [label for labels in labels_of_groups.values() for label in labels]
    if "VOID" in every:
        return "VOID"
    if all(label == "AT NULL" for label in every):
        return "FAILED"
    if all(label == "HOLDS" for label in every):
        return "HOLDS"
    return "UNDECIDED"


# The reading of a saved run.

def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_steps(folder: str) -> List[dict]:
    """The step records of an episode. Control K1 asks for 200 of them in order."""
    with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if [record.get("step") for record in records] != list(range(STEPS_IN_AN_EPISODE)):
        raise ControlMismatch(f"K1, {folder}: the step records are not 0 to {STEPS_IN_AN_EPISODE - 1}")
    return records


def _check_salience(gate_episode: dict, binding_record: dict, where: str) -> None:
    """Control K3: the gate recorded the salience of the binding record at the choice step."""
    candidates = {band: binding_record["salience"][band] for band, cells in binding_record["groups"].items() if cells}
    recorded = gate_episode["salience"]
    if set(candidates) != set(recorded):
        raise ControlMismatch(f"K3, {where}: the bands of the gate and of the binding differ")
    if any(abs(candidates[band] - recorded[band]) > SALIENCE_TOLERANCE for band in candidates):
        raise ControlMismatch(f"K3, {where}: the salience of the gate and of the binding differ")


def _stream_rows(tables, choice_step: int, where: str) -> Dict[str, np.ndarray]:
    """For each stream, the rows of step 0 to the choice step. Control K4 checks the step index."""
    rows = {}
    for name in STREAMS:
        steps = np.asarray(tables[name + "__steps"])
        if len(steps) <= choice_step or not np.array_equal(steps[:choice_step + 1], np.arange(choice_step + 1)):
            raise ControlMismatch(f"K4, {where}: stream {name} has no row at each step up to {choice_step}")
        rows[name] = np.asarray(tables[name][:choice_step + 1], dtype=np.float64).reshape(choice_step + 1, -1)
    return rows


def _read_episode(run_dir: str, index: int, gate_episode: dict, binding: List[dict]) -> dict:
    """The label, the choice step, the rows of each stream and the workspace state of one episode."""
    where = f"{os.path.basename(run_dir)} episode {index}"
    folder = os.path.join(run_dir, "episodes", f"ep_{index:04d}")
    choice_step = find_choice_step(binding)
    _check_salience(gate_episode, binding[choice_step], where)
    steps = _load_steps(folder)
    with np.load(os.path.join(folder, "vectors.npz")) as tables:
        rows = _stream_rows(tables, choice_step, where)
    return {"label": label_of_level(float(steps[0]["internals"]["info_after_step"]["_truth_decoy_level"])),
            "choice_step": choice_step, "rows": rows,
            "ignited": bool(steps[choice_step]["ignited"]), "winner": str(steps[choice_step]["winner"])}


def read_run(run_dir: str, expected_episodes: int) -> List[dict]:
    """One record for each episode of a run, after controls K1 to K4."""
    gate = _load_json(os.path.join(run_dir, "value_gate.json"))["episodes"]
    history = _load_json(os.path.join(run_dir, "map_binding.json"))["history"]
    vector_files = glob.glob(os.path.join(run_dir, "episodes", "ep_*", "vectors.npz"))
    step_files = glob.glob(os.path.join(run_dir, "episodes", "ep_*", "steps.jsonl"))
    counts = (len(gate), len(vector_files), len(step_files), len(history) / STEPS_IN_AN_EPISODE)
    if any(count != expected_episodes for count in counts):
        raise ControlMismatch(f"K1, {run_dir}: the counts {counts} are not {expected_episodes}")
    episodes = [_read_episode(run_dir, index, gate[index],
                              history[index * STEPS_IN_AN_EPISODE:(index + 1) * STEPS_IN_AN_EPISODE])
                for index in range(expected_episodes)]
    ones = sum(episode["label"] for episode in episodes)
    if ones * 2 != expected_episodes:
        raise ControlMismatch(f"K2, {run_dir}: {ones} of {expected_episodes} episodes have label 1")
    return episodes


def _scored(rows, episode_of_row, labels, where: str, check_null_mean: bool) -> dict:
    """A read with its null. Control K5 checks the mean of the null scores."""
    reading = read_score(rows, episode_of_row, labels)
    null = null_scores(rows, episode_of_row, labels)
    reading.update(null_95=float(np.percentile(null, NULL_PERCENTILE)), null_mean=float(null.mean()),
                   rows=int(len(rows)))
    if check_null_mean and not NULL_MEAN_RANGE[0] <= reading["null_mean"] <= NULL_MEAN_RANGE[1]:
        raise ControlMismatch(f"K5, {where}: the mean of the null scores is {reading['null_mean']:.3f}")
    return reading


def _same_vector_counts(episodes: List[dict]) -> dict:
    """Which vector the broadcast is at the choice step: the largest absolute difference to each."""
    report = {}
    for name in ("audio_content", "tectum_content"):
        differences = [float(np.abs(e["rows"]["broadcast"][-1] - e["rows"][name][-1]).max()) for e in episodes]
        report[name] = {"episodes_with_difference_0": int(sum(d == 0.0 for d in differences)),
                        "largest_difference": float(max(differences)),
                        "median_difference": float(np.median(differences))}
    return report


def _reported_only(episodes: List[dict], labels: np.ndarray, where: str) -> dict:
    """The items of the rules that have no part in the verdict."""
    steps = [episode["choice_step"] for episode in episodes]
    episode_of_row = np.repeat(np.arange(len(episodes)), [step + 1 for step in steps])
    earlier_rows = np.concatenate([episode["rows"]["broadcast"] for episode in episodes])
    return {"same_vector": _same_vector_counts(episodes),
            "ignited_share": float(np.mean([episode["ignited"] for episode in episodes])),
            "winners": dict(Counter(episode["winner"] for episode in episodes)),
            "choice_step": {"min": int(min(steps)), "median": float(np.median(steps)), "max": int(max(steps))},
            "broadcast_rows_to_the_choice_step": _scored(earlier_rows, episode_of_row, labels, where, False)}


def check_run(run_dir: str, expected_episodes: int) -> dict:
    """The label of one run with every score. A failed control gives VOID with its cause."""
    name = os.path.basename(run_dir)
    try:
        episodes = read_run(run_dir, expected_episodes)
        labels, one_each = np.array([e["label"] for e in episodes]), np.arange(expected_episodes)
        streams = {stream: _scored(np.stack([e["rows"][stream][-1] for e in episodes]), one_each, labels,
                                   f"{name} {stream}", stream != "tectum_content") for stream in STREAMS}
        episode_number = read_score(one_each[:, None].astype(float), one_each, labels)["score"]
        reported = _reported_only(episodes, labels, name)
    except ControlMismatch as mismatch:
        return {"run": name, "label": "VOID", "cause": str(mismatch)}
    reading = {"broadcast": streams["broadcast"]["score"], "broadcast_null": streams["broadcast"]["null_95"],
               "audio": streams["audio_content"]["score"], "audio_null": streams["audio_content"]["null_95"],
               "episode_number": episode_number}
    label = run_label(reading)
    cause = "the reference failed" if label == "VOID" else None
    return {"run": name, "label": label, "cause": cause, "reading": reading, "streams": streams, "reported": reported}


def source_table(run_dir: str, expected_episodes: int) -> Dict[str, Dict[str, int]]:
    """Made after the run, with no part in the verdict. For each label, the count of episodes where the
    broadcast at the choice step is the audio vector, the tectum vector, or neither."""
    table = {str(label): {"audio_content": 0, "tectum_content": 0, "neither": 0} for label in (1, 0)}
    for episode in read_run(run_dir, expected_episodes):
        broadcast = episode["rows"]["broadcast"][-1]
        same = [name for name in ("audio_content", "tectum_content")
                if np.array_equal(broadcast, episode["rows"][name][-1])]
        if len(same) > 1:
            raise ControlMismatch(f"{run_dir}: the broadcast equals both vectors in one episode")
        table[str(episode["label"])][same[0] if same else "neither"] += 1
    return table


def _write_source_tables(runs_root: str, path: str) -> None:
    tables = {name: source_table(os.path.join(runs_root, name), episodes)
              for names, episodes, is_counted in GROUPS.values() if is_counted for name in names}
    for name, table in tables.items():
        print(name, table)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(tables, handle, indent=1)


def _print_report(report: dict) -> None:
    if "reading" not in report:
        print(f"{report['run']}: VOID. {report['cause']}")
        return
    r, s = report["reading"], report["streams"]
    print(f"{report['run']}: {report['label']}. broadcast {r['broadcast']:.3f} (null {r['broadcast_null']:.3f}), "
          f"audio {r['audio']:.3f} (null {r['audio_null']:.3f}), "
          f"tectum {s['tectum_content']['score']:.3f} (null {s['tectum_content']['null_95']:.3f}), "
          f"episode number {r['episode_number']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", required=True, help="The folder that holds the nine saved runs.")
    parser.add_argument("--summary", default=None, help="Write the reports as JSON to this path.")
    parser.add_argument("--source-table", default=None,
                        help="Write only the table of the label against the source of the broadcast, as JSON, "
                             "to this path. The read does not run. Added after the first run.")
    args = parser.parse_args()
    if args.source_table:
        _write_source_tables(args.runs_root, args.source_table)
        return
    reports, counted = [], {}
    for group, (names, episodes, is_counted) in GROUPS.items():
        for name in names:
            report = dict(check_run(os.path.join(args.runs_root, name), episodes), group=group, counted=is_counted)
            reports.append(report)
            _print_report(report)
            if is_counted:
                counted.setdefault(group, []).append(report["label"])
    print(f"VERDICT over the counted groups: {verdict(counted)}")
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump({"verdict": verdict(counted), "runs": reports}, handle, indent=1)


if __name__ == "__main__":
    main()
