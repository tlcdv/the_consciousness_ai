"""Would a priority from the workspace winner change the choice of the value gate? An offline check.

docs/decisions/2026_11_11_workspace_priority_check.md holds the rules, pushed before this script ran.
The check reads saved runs only. It trains nothing and changes no default. It is a correlation on
saved runs and not an ablation.

The term. At the choice step of an episode, the loudest band gets 1 when the workspace is ignited
and its winner is `audio`. Every other case gives 0. The score of a band is its recorded priority
plus WEIGHT x term. The predicted choice uses the softmax of the gate with the draw that the gate
used (a generator with the gate's own seed, one draw for each episode).

RULES, for each run.
  1  The share of episodes where the predicted choice differs from the recorded choice is 0.05 or more.
  2  R squared of a least squares fit of the term on the salience is below 0.90.
SUPPORTED: both rules at 3 of 3 seeds in each group. EXCLUDED: the controls pass and SUPPORTED is
not reached. VOID: a control fails. Every control raises ControlMismatch, and no verdict is printed.

Usage:
  python -m scripts.analysis.check_workspace_priority --runs-root <folder with the six runs>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from models.core.value_gate import TEMPERATURE, VALUE_WEIGHT

WEIGHT = 1.0                      # the salience of the loudest tone
REPORT_WEIGHTS = (0.5, 1.0, 2.0)  # 0.5 and 2.0 are printed only
CHANGE_RULE = 0.05
FIT_RULE = 0.90
GATE_SEED_OFFSET = 1000003        # _wire_value_gate, scripts/training/train_rlhf.py
PRIORITY_TOLERANCE = 1e-9
GROUPS = {"vg_learn": (("vg_learn_s601", "vg_learn_s602", "vg_learn_s603"), 40),
          "rv_switch": (("rv_switch_s701", "rv_switch_s702", "rv_switch_s703"), 60)}


class ControlMismatch(RuntimeError):
    """A control with an exact expected value did not give that value. The check is VOID."""


def find_choice_step(binding_steps: List[dict]) -> int:
    """The first step with a band and at least one group with cells, as the gate wiring does."""
    for index, record in enumerate(binding_steps):
        if record.get("band") is not None and any(record.get("groups", {}).values()):
            return index
    raise ControlMismatch("an episode has no step with a bound group")


def workspace_term(step_record: dict, loudest_band: int, bands: List[str]) -> Dict[str, float]:
    """1 for the loudest band when the workspace is ignited with an audio winner, else 0."""
    if "ignited" not in step_record or "winner" not in step_record:
        raise ControlMismatch("a step record has no ignition state or no winner")
    audio_won = bool(step_record["ignited"]) and step_record["winner"] == "audio"
    return {band: 1.0 if audio_won and int(band) == int(loudest_band) else 0.0 for band in bands}


def choice_probabilities(scores) -> np.ndarray:
    """The softmax of ValueGate.probabilities, at its temperature."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 1:
        return np.array([1.0])
    scaled = (scores - scores.max()) / TEMPERATURE
    return np.exp(scaled) / np.exp(scaled).sum()


def pick_index(probabilities: np.ndarray, draw: float) -> int:
    """The index that numpy's Generator.choice gives for one uniform draw."""
    cumulative = np.cumsum(probabilities)
    cumulative /= cumulative[-1]
    return int(cumulative.searchsorted(draw, side="right"))


def predict_choice(bands: List[str], priorities, term: Dict[str, float], weight: float, draw: float) -> str:
    scores = [priority + weight * term[band] for band, priority in zip(bands, priorities)]
    return bands[pick_index(choice_probabilities(scores), draw)]


def r_squared(salience: np.ndarray, term: np.ndarray) -> float:
    """R squared of the least squares line of `term` on `salience`. A constant input has no fit."""
    salience, term = np.asarray(salience, dtype=np.float64), np.asarray(term, dtype=np.float64)
    if np.ptp(term) == 0.0 or np.ptp(salience) == 0.0:
        raise ValueError("R squared has no value when the term or the salience is constant")
    slope, intercept = np.polyfit(salience, term, 1)
    residual = term - (slope * salience + intercept)
    return float(1.0 - (residual ** 2).sum() / ((term - term.mean()) ** 2).sum())


def replay_values(episodes: List[dict], learning_rate: float) -> List[Dict[str, float]]:
    """The value of each band before each episode, by the update rule of the gate, from 0."""
    values, before = {}, []
    for episode in episodes:
        before.append(dict(values))
        band = str(episode["choice"])
        old = values.get(band, 0.0)
        values[band] = old + learning_rate * (episode["outcome"] - old)
    return before


def _load_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_step(run_dir: str, episode: int, step: int) -> dict:
    path = os.path.join(run_dir, "episodes", f"ep_{episode:04d}", "steps.jsonl")
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if rows[step].get("step") != step:
        raise ControlMismatch(f"{path}: row {step} holds step {rows[step].get('step')}")
    return rows[step]


def _check_salience(gate_episode: dict, binding_record: dict, label: str) -> None:
    """Control B: the gate read the salience of the binding record at the choice step."""
    candidates = {b: binding_record["salience"][b] for b, cells in binding_record["groups"].items() if cells}
    if candidates != gate_episode["salience"]:
        raise ControlMismatch(f"control B, {label}: the salience of the gate and of the binding differ")


def _check_priorities(gate_episode: dict, values: Dict[str, float], label: str) -> None:
    """Control C: salience + weight x replayed value gives the recorded priority."""
    for band, salience in gate_episode["salience"].items():
        expected = salience + VALUE_WEIGHT * values.get(band, 0.0)
        if abs(expected - gate_episode["priorities"][band]) > PRIORITY_TOLERANCE:
            raise ControlMismatch(f"control C, {label}, band {band}: {expected} against the record")


def _read_episodes(run_dir: str, expected_episodes: int) -> List[dict]:
    """One row for each episode: bands, priorities, salience, term, recorded choice, winner."""
    gate = _load_json(os.path.join(run_dir, "value_gate.json"))
    history = _load_json(os.path.join(run_dir, "map_binding.json"))["history"]
    max_steps = _load_json(os.path.join(run_dir, "session.json"))["run"]["max_steps"]
    step_files = glob.glob(os.path.join(run_dir, "episodes", "ep_*", "steps.jsonl"))
    if not len(gate["episodes"]) == len(step_files) == expected_episodes == len(history) // max_steps:
        raise ControlMismatch(f"{run_dir}: the episode count is not {expected_episodes}")
    values, rows = replay_values(gate["episodes"], gate["learning_rate"]), []
    for index, episode in enumerate(gate["episodes"]):
        binding = history[index * max_steps:(index + 1) * max_steps]
        step = find_choice_step(binding)
        label = f"{os.path.basename(run_dir)} episode {index}"
        _check_salience(episode, binding[step], label)
        _check_priorities(episode, values[index], label)
        rows.append(_episode_row(episode, binding[step], _load_step(run_dir, index, step)))
    return rows


def _episode_row(episode: dict, binding_record: dict, step_record: dict) -> dict:
    bands = list(episode["salience"])
    return {"bands": bands, "choice": str(episode["choice"]),
            "salience": [episode["salience"][b] for b in bands],
            "priorities": [episode["priorities"][b] for b in bands],
            "term": workspace_term(step_record, binding_record["band"], bands),
            "ignited": bool(step_record["ignited"]), "winner": step_record["winner"]}


def _count_changes(rows: List[dict], seed: int, weight: float) -> int:
    """Episodes where the predicted choice at `weight` differs from the recorded choice."""
    generator = np.random.default_rng(seed + GATE_SEED_OFFSET)
    draws = [generator.random() for _ in rows]
    return sum(predict_choice(r["bands"], r["priorities"], r["term"], weight, u) != r["choice"]
               for r, u in zip(rows, draws))


def _fit(rows: List[dict]) -> Optional[float]:
    """R squared of rule 2, or None when the term is 0 in every row (limit 7 of the rules)."""
    salience = np.array([s for r in rows for s in r["salience"]])
    term = np.array([r["term"][b] for r in rows for b in r["bands"]])
    control = r_squared(salience, 2.0 * salience + 1.0)
    if abs(control - 1.0) > 1e-9:
        raise ControlMismatch(f"control D: the fit of a linear function gave {control}")
    return None if not term.any() else r_squared(salience, term)


def check_run(run_dir: str, expected_episodes: int) -> dict:
    """The controls and the two rules for one saved run."""
    rows = _read_episodes(run_dir, expected_episodes)
    seed = _load_json(os.path.join(run_dir, "session.json"))["run"]["seed"]
    unchanged = len(rows) - _count_changes(rows, seed, weight=0.0)
    if unchanged != len(rows):
        raise ControlMismatch(f"control A, {run_dir}: the replay matched {unchanged} of {len(rows)} choices")
    changed = {weight: _count_changes(rows, seed, weight) for weight in REPORT_WEIGHTS}
    fit = _fit(rows)
    return {"run": os.path.basename(os.path.normpath(run_dir)), "seed": seed, "episodes": len(rows),
            "control_a_matches": unchanged, "changed": changed,
            "changed_share": changed[WEIGHT] / len(rows), "r_squared": fit,
            "ignited_share": sum(r["ignited"] for r in rows) / len(rows),
            "audio_share": sum(any(r["term"].values()) for r in rows) / len(rows),
            "rule_1": changed[WEIGHT] / len(rows) >= CHANGE_RULE,
            "rule_2": fit is not None and fit < FIT_RULE}


def verdict(run_reports: List[dict]) -> str:
    """SUPPORTED when both rules hold at every seed of each group, with 3 seeds in each group."""
    groups: Dict[str, List[bool]] = {}
    for report in run_reports:
        groups.setdefault(report["group"], []).append(report["rule_1"] and report["rule_2"])
    if not groups or any(len(passes) != 3 for passes in groups.values()):
        raise ControlMismatch("each group needs exactly 3 seeds")
    return "SUPPORTED" if all(all(passes) for passes in groups.values()) else "EXCLUDED"


def _print_report(report: dict) -> None:
    fit = "undefined" if report["r_squared"] is None else f"{report['r_squared']:.4f}"
    changed = "  ".join(f"w={w}: {n}" for w, n in report["changed"].items())
    print(f"{report['run']:16s} episodes {report['episodes']}  control A {report['control_a_matches']}"
          f"/{report['episodes']}  ignited {report['ignited_share']:.3f}  audio winner q {report['audio_share']:.3f}"
          f"  changed [{changed}]  share {report['changed_share']:.3f}  rule 1 {report['rule_1']}"
          f"  R2 {fit}  rule 2 {report['rule_2']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-root", required=True, help="The folder that holds the six saved runs.")
    parser.add_argument("--summary", default=None, help="Write the reports as JSON to this path.")
    args = parser.parse_args()
    reports = []
    for group, (names, episodes) in GROUPS.items():
        for name in names:
            reports.append(dict(check_run(os.path.join(args.runs_root, name), episodes), group=group))
            _print_report(reports[-1])
    print(f"Controls A to D passed in {len(reports)} of {len(reports)} runs.")
    print(f"VERDICT at weight {WEIGHT}: {verdict(reports)}")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump({"weight": WEIGHT, "verdict": verdict(reports), "runs": reports}, handle, indent=1)


if __name__ == "__main__":
    main()
