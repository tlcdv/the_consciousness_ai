"""Offline choice head: can a head learn which candidate gives the food, from one outcome for each episode.

Step P1 of design P. The rules are in docs/decisions/2026_11_16_workspace_choice_head.md, written
before this file.

This file holds the two heads, their training, the score, and the test on built data that is the gate
of the run. The test on built data reads no saved run.

It also holds the count table of the recorded feedback. The count table reads value_gate.json of each
run and nothing else: no saved vector, and no head is trained. The part that reads the saved vectors
is not written yet. It follows only after the owner's decision.

The two heads.
  variant  The go and no-go variant: PrefrontalCortex(input, 256) and BasalGanglia(256, 2) of the agent,
           with the loss terms of ActionSelectionCore.update_policy and the departures D0 to D5 of the
           rules. The action of an episode is 1 for the chosen slot and 0 for the other slot.
  plain    The plain value head: the body of StandardActorCritic and one linear layer with 2 outputs,
           with a mean squared error of the output of the chosen slot against the outcome.

A head receives the input rows, the chosen slot and the outcome. It never receives the target.

It also holds the check on the variant: why the go and no-go variant does not learn on built data. The
rules are in docs/decisions/2026_11_16_choice_head_variant_sign.md, written before that code. The check
uses built data only. It reads no saved run, and it changes no file under models/. Conditions C3a, C3b
and C3c are ablations to test a cause. They are not a repair of the loss.

It also holds check 2 of step P2: the choice head of models/core/choice_head.py on built data, before the
build of the choice mode. The rules are in docs/decisions/2026_11_22_workspace_choice_mode.md, with the
sections after the review and on the scaling. Part A gives the head the new row sets of C2 with its own
exploring choices. Part B uses the live rule with the 60 rows of a run. The arm of O2 is part B with a
store of rows, reported only. Each part runs a second pass with each outcome set to 1. It reads no saved
run and runs no agent.

It also holds check 2b of step P2: three learning rules of the choice head (`chosen`, `both`, `hopeful`)
in the setting of part B, with the measure greedy_food, the controls K1 to K6 and the rule that chooses
the learning rule of work 4. The rules are in the same file, sections "Owner decision before work 4, and
the rules of check 2b" and "Changes after the independent review of check 2b". It reads no saved run.
The label mode also reads the committed rows of check 2 (K6) and the committed count table of step P1.

Usage:
    python -m scripts.analysis.check_choice_head --built-test out/built_test.csv
    python -m scripts.analysis.check_choice_head --count-table runs --out out/count_table.json
    python -m scripts.analysis.check_choice_head --variant-sign-first-step
    python -m scripts.analysis.check_choice_head --variant-sign-timing
    python -m scripts.analysis.check_choice_head --variant-sign out_dir --step-p1-csv built_test.csv
    python -m scripts.analysis.check_choice_head --variant-sign-labels out_dir
    python -m scripts.analysis.check_choice_head --choice-mode-timing
    python -m scripts.analysis.check_choice_head --choice-mode-check out_dir
    python -m scripts.analysis.check_choice_head --choice-mode-labels out_dir
    python -m scripts.analysis.check_choice_head --choice-rule-timing
    python -m scripts.analysis.check_choice_head --choice-rule-check out_dir
    python -m scripts.analysis.check_choice_head --choice-rule-labels out_dir
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.core.choice_head import DEFAULT_RULE, EXPLORATION_SHARE, RULES, UPDATES_PER_CHOICE, ChoiceHead
from models.self_model.action_selection_core import BasalGanglia, PrefrontalCortex
from scripts.analysis.check_choice_read import ControlMismatch, auc, make_folds, standardize

HEADS = ("variant", "plain")
CONTEXT_DIM = 256
PLAIN_HIDDEN = 128           # ac_hidden of StandardActorCritic
LEARNING_RATE = 3e-4         # the default of both policy classes
UPDATES = 200
INITIAL_SEEDS = (0, 1, 2)
OUTER_FOLDS = 5
FOLD_SEED = 0

BUILT_DIMENSIONS = 256
BUILT_SIGNAL_DIMENSIONS = 64
BUILT_SHIFT = 1.0
BUILT_ROWS = (30, 40)
BUILT_SIGNAL_SHARES = (1.0, 0.6, 0.0)    # 0.0 is the null of the test: no signal in the input
BUILT_DATA_SEEDS = tuple(range(10))
CHOOSERS = ("random", "slot_0", "food", "food_80")
BUILT_COLUMNS = ("outcomes_at_1", "rows", "signal_share", "chooser", "data_seed", "head", "score",
                 "score_seed_0", "score_seed_1", "score_seed_2")

CANDIDATE_BANDS = (18, 23)
RUN_EPISODES = {"vg_learn": 40, "rv_switch": 60}
MIN_EPISODES_OF_A_TARGET = 10    # control K7
MIN_OF_EACH_OUTCOME = 2          # the count rule of the recorded feedback
# (food share, slot 0 share) that each chooser of the test on built data gives with equal targets.
CHOOSER_POINTS = {"random": (0.5, 0.5), "slot_0": (0.5, 1.0), "food": (1.0, 0.5), "food_80": (0.8, 0.5)}
# The saved food switch runs have --dark-room-food-switch 30 in their command, and the food moves at
# recorded episode 29. They were made before the counter of the room was repaired (the correction of
# 2026-11-16 in the rules).
FOOD_SWITCH_EPISODE = 29
# name, run folder, first episode, episode after the last, band of the food, counted in the verdict
UNITS = tuple(
    [("vg_learn_s%d" % seed, "vg_learn_s%d" % seed, 0, 40, 18, True) for seed in (601, 602, 603)]
    + [("rv_switch_s%d before" % seed, "rv_switch_s%d" % seed, 0, FOOD_SWITCH_EPISODE, 18, True)
       for seed in (701, 702, 703)]
    + [("rv_switch_s%d after" % seed, "rv_switch_s%d" % seed, FOOD_SWITCH_EPISODE, 60, 23, False)
       for seed in (701, 702, 703)])


# The heads.

def build_head(kind: str, input_dim: int, seed: int) -> Dict[str, nn.Module]:
    """A head with the default initial weights of the library, from the initial seed."""
    torch.manual_seed(seed)
    if kind == "variant":
        return {"pfc": PrefrontalCortex(input_dim, CONTEXT_DIM), "bg": BasalGanglia(CONTEXT_DIM, 2)}
    if kind == "plain":
        return {"net": nn.Sequential(nn.Linear(input_dim, PLAIN_HIDDEN), nn.Tanh(),
                                     nn.Linear(PLAIN_HIDDEN, PLAIN_HIDDEN), nn.Tanh(),
                                     nn.Linear(PLAIN_HIDDEN, 2))}
    raise ValueError("head must be one of %s, got %r" % (HEADS, kind))


def head_parameters(head: Dict[str, nn.Module]) -> List[torch.Tensor]:
    return [parameter for part in head.values() for parameter in part.parameters()]


def action_vectors(chosen: torch.Tensor) -> torch.Tensor:
    """Departure D0 of the rules: 1 for the chosen slot and 0 for the other slot."""
    return nn.functional.one_hot(chosen.long(), 2).float()


def variant_loss_terms(value, go, no_go, brake, action, outcome) -> Dict[str, torch.Tensor]:
    """The loss terms of update_policy (lines 417 to 462 of action_selection_core.py)."""
    advantage = outcome - value.detach()
    target_go = action * (advantage > 0).float() - action * (advantage < 0).float()
    target_no_go = (advantage < 0).float().expand_as(no_go)
    terms = {"value": nn.MSELoss()(value, outcome),
             "go": nn.MSELoss()(go, target_go.detach()),
             "no_go": nn.BCELoss()(no_go, target_no_go.detach()),
             "brake": nn.MSELoss()(brake, torch.tanh(torch.abs(advantage).detach()))}
    terms["total"] = terms["go"] + terms["no_go"] + 0.3 * terms["brake"] + 0.5 * terms["value"]
    return terms


def variant_loss(pfc, bg, rows, action, outcome) -> Dict[str, torch.Tensor]:
    """The loss of the variant for a batch, with the working memory at zero (departure D2)."""
    state, _ = pfc(rows, torch.zeros(rows.shape[0], CONTEXT_DIM))
    return variant_loss_terms(bg.critic(state), bg.direct_pathway(state), bg.indirect_pathway(state),
                              bg.stn_pathway(state), action, outcome)


def plain_loss(net, rows, chosen, outcome) -> torch.Tensor:
    """Mean squared error of the output of the chosen slot against the outcome."""
    chosen_output = net(rows).gather(1, chosen.long().unsqueeze(1))
    return nn.MSELoss()(chosen_output, outcome)


def slot_numbers(head: Dict[str, nn.Module], rows: torch.Tensor) -> torch.Tensor:
    """One number for each slot. Variant: go x (1 - no-go), with no brake and no relay."""
    if "net" in head:
        return head["net"](rows)
    state, _ = head["pfc"](rows, torch.zeros(rows.shape[0], CONTEXT_DIM))
    return head["bg"].direct_pathway(state) * (1.0 - head["bg"].indirect_pathway(state))


def train_head(kind: str, seed: int, rows, chosen, outcome, updates: int = UPDATES) -> Dict[str, nn.Module]:
    """A head trained on all rows at once with Adam. The feedback is the chosen slot and its outcome."""
    rows = torch.as_tensor(np.asarray(rows), dtype=torch.float32)
    chosen = torch.as_tensor(np.asarray(chosen), dtype=torch.long)
    outcome = torch.as_tensor(np.asarray(outcome), dtype=torch.float32).unsqueeze(1)
    head = build_head(kind, rows.shape[1], seed)
    optimizer = torch.optim.Adam(head_parameters(head), lr=LEARNING_RATE)
    for _ in range(updates):
        if kind == "variant":
            loss = variant_loss(head["pfc"], head["bg"], rows, action_vectors(chosen), outcome)["total"]
        else:
            loss = plain_loss(head["net"], rows, chosen, outcome)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return head


def preferences(head: Dict[str, nn.Module], rows) -> np.ndarray:
    """The number of slot 0 minus the number of slot 1, for each row."""
    with torch.no_grad():
        numbers = slot_numbers(head, torch.as_tensor(np.asarray(rows), dtype=torch.float32))
    return (numbers[:, 0] - numbers[:, 1]).numpy().astype(np.float64)


# The score.

def score_of_one_seed(kind: str, seed: int, rows, chosen, outcome, target, updates: int = UPDATES) -> float:
    """The mean AUC of the preference against the target over five folds. The target is not given
    to the head. It makes the folds and the AUC only."""
    rows, fold_of = np.asarray(rows, dtype=np.float64), make_folds(target, OUTER_FOLDS, FOLD_SEED)
    fold_aucs = []
    for fold in range(OUTER_FOLDS):
        train, test = fold_of != fold, fold_of == fold
        train_rows, test_rows = standardize(rows[train], rows[test])
        head = train_head(kind, seed, train_rows, chosen[train], outcome[train], updates)
        fold_aucs.append(auc(preferences(head, test_rows), target[test]))
    return float(np.mean(fold_aucs))


# The test on built data.

def built_outcomes(chosen: np.ndarray, target: np.ndarray) -> np.ndarray:
    """1 when the chosen slot is the slot of the food. A target of 1 means that the food is in slot 0."""
    return (chosen == 1 - target).astype(np.float64)


def _built_choices(chooser: str, food_slot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    random_choices, order = rng.integers(0, 2, len(food_slot)), rng.permutation(len(food_slot))
    if chooser == "random":
        return random_choices
    if chooser == "slot_0":
        return np.zeros(len(food_slot), dtype=int)
    if chooser == "food":
        return food_slot.copy()
    if chooser == "food_80":
        chosen = food_slot.copy()
        other = order[int(round(0.8 * len(food_slot))):]
        chosen[other] = 1 - food_slot[other]
        return chosen
    raise ValueError("chooser must be one of %s, got %r" % (CHOOSERS, chooser))


def built_case(rows: int, signal_share: float, chooser: str, data_seed: int, outcomes_at_1: bool = False) -> dict:
    """Built rows of noise with a signal for the target, and the feedback of one chooser."""
    rng = np.random.default_rng(data_seed)
    inputs = rng.normal(size=(rows, BUILT_DIMENSIONS))
    target = rng.permutation(np.arange(rows) % 2)
    with_signal = np.zeros(rows, dtype=bool)
    with_signal[rng.permutation(rows)[:int(round(signal_share * rows))]] = True
    inputs[np.ix_(with_signal & (target == 1), np.arange(BUILT_SIGNAL_DIMENSIONS))] += BUILT_SHIFT
    chosen = _built_choices(chooser, 1 - target, rng)
    outcome = np.ones(rows) if outcomes_at_1 else built_outcomes(chosen, target)
    return {"rows": inputs, "target": target, "with_signal": with_signal, "chosen": chosen, "outcome": outcome}


def built_test_cases() -> List[dict]:
    """Each case of the test: 2 passes x 2 row counts x 3 signal shares x 4 choosers x 10 data sets."""
    names = ("outcomes_at_1", "rows", "signal_share", "chooser", "data_seed")
    grid = itertools.product((False, True), BUILT_ROWS, BUILT_SIGNAL_SHARES, CHOOSERS, BUILT_DATA_SEEDS)
    return [dict(zip(names, combination)) for combination in grid]


def run_built_test(path: str) -> None:
    """One result row for each case and head, written at once so that the row count shows the progress."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BUILT_COLUMNS)
        writer.writeheader()
        for case in built_test_cases():
            built = built_case(**case)
            for kind in HEADS:
                scores = [score_of_one_seed(kind, seed, built["rows"], built["chosen"], built["outcome"],
                                            built["target"]) for seed in INITIAL_SEEDS]
                writer.writerow(dict(case, head=kind, score=float(np.mean(scores)),
                                     **{"score_seed_%d" % seed: scores[seed] for seed in INITIAL_SEEDS}))
                handle.flush()


# The count table of the recorded feedback.

def slot_record(gate_episode: dict, food_band: int, where: str) -> Dict[str, int]:
    """The chosen slot, the outcome and the target of one episode of value_gate.json.

    Slot 0 is the candidate with the higher recorded salience. The target is 1 when the band of the
    food is in slot 0."""
    salience = {int(band): float(number) for band, number in gate_episode["salience"].items()}
    if tuple(sorted(salience)) != CANDIDATE_BANDS:
        raise ControlMismatch("K6, %s: the candidates are %s, not %s" % (where, sorted(salience), CANDIDATE_BANDS))
    if salience[CANDIDATE_BANDS[0]] == salience[CANDIDATE_BANDS[1]]:
        raise ControlMismatch("%s: the two candidates have equal salience" % where)
    by_salience = sorted(salience, key=salience.get, reverse=True)
    slot_of = {band: slot for slot, band in enumerate(by_salience)}
    outcome = float(gate_episode["outcome"])
    if outcome not in (0.0, 1.0):
        raise ControlMismatch("K10, %s: the outcome is %r, not 0 or 1" % (where, outcome))
    return {"chosen": slot_of[int(gate_episode["choice"])], "outcome": int(outcome),
            "target": 1 - slot_of[food_band]}


def count_table(records: List[Dict[str, int]]) -> Dict[str, int]:
    """The count of episodes for each chosen slot, outcome and target. It has 8 cells."""
    cells = {"chosen_%d_outcome_%d_target_%d" % cell: 0 for cell in itertools.product((0, 1), repeat=3)}
    for record in records:
        cells["chosen_%d_outcome_%d_target_%d" % (record["chosen"], record["outcome"], record["target"])] += 1
    return cells


def training_set_counts(records: List[Dict[str, int]]) -> Dict[str, object]:
    """For one training set: the count of each outcome for each chosen slot, and the count of the rule.

    The count of the rule of a slot is the smaller of its two outcome counts. The count of the rule of
    the set is that of its better slot. The set has an informative slot when this count is 2 or more."""
    counts = {}
    for slot in (0, 1):
        outcomes = [record["outcome"] for record in records if record["chosen"] == slot]
        counts["slot_%d" % slot] = {"outcome_1": sum(outcomes), "outcome_0": len(outcomes) - sum(outcomes)}
    rule_count = max(min(counts[name].values()) for name in ("slot_0", "slot_1"))
    return dict(counts, episodes=len(records), rule_count=rule_count,
                informative=rule_count >= MIN_OF_EACH_OUTCOME)


def nearest_chooser(food_share: float, slot_0_share: float) -> str:
    """A description only: the chooser of the test on built data with the nearest pair of shares."""
    def distance(name: str) -> float:
        return (CHOOSER_POINTS[name][0] - food_share) ** 2 + (CHOOSER_POINTS[name][1] - slot_0_share) ** 2
    return min(CHOOSERS, key=distance)


def unit_report(records: List[Dict[str, int]], where: str) -> Dict[str, object]:
    """The count table of a unit, the counts of its five training sets, and VOID or not VOID."""
    target = np.array([record["target"] for record in records])
    if min(int(target.sum()), int(len(target) - target.sum())) < MIN_EPISODES_OF_A_TARGET:
        raise ControlMismatch("K7, %s: %d episodes have target 1 of %d" % (where, target.sum(), len(target)))
    fold_of = make_folds(target, OUTER_FOLDS, FOLD_SEED)
    sets = [training_set_counts([record for record, fold in zip(records, fold_of) if fold != held_out])
            for held_out in range(OUTER_FOLDS)]
    food_share = float(np.mean([record["chosen"] == 1 - record["target"] for record in records]))
    slot_0_share = float(np.mean([record["chosen"] == 0 for record in records]))
    return {"episodes": len(records), "episodes_with_target_1": int(target.sum()),
            "count_table": count_table(records), "training_sets": sets,
            "smallest_rule_count": min(one_set["rule_count"] for one_set in sets),
            "void_for_the_recorded_feedback": any(not one_set["informative"] for one_set in sets),
            "share_food_slot_chosen": food_share, "share_slot_0_chosen": slot_0_share,
            "share_outcome_1": float(np.mean([record["outcome"] for record in records])),
            "nearest_chooser": nearest_chooser(food_share, slot_0_share)}


def read_unit_records(runs_dir: str, run: str, first: int, after_last: int, food_band: int) -> List[Dict[str, int]]:
    """The slot records of the episodes of one unit, from value_gate.json of its run and nothing else."""
    with open(os.path.join(runs_dir, run, "value_gate.json"), encoding="utf-8") as handle:
        episodes = json.load(handle)["episodes"]
    expected = RUN_EPISODES[run.rsplit("_s", 1)[0]]
    if len(episodes) != expected:
        raise ControlMismatch("K1, %s: value_gate.json has %d episodes, not %d" % (run, len(episodes), expected))
    return [slot_record(episodes[index], food_band, "%s episode %d" % (run, index))
            for index in range(first, after_last)]


def run_count_table(runs_dir: str, out_path: str) -> Dict[str, dict]:
    """The report of each unit. A control mismatch makes the unit VOID and is recorded with its text."""
    reports = {}
    for name, run, first, after_last, food_band, counted in UNITS:
        try:
            report = unit_report(read_unit_records(runs_dir, run, first, after_last, food_band), name)
        except ControlMismatch as mismatch:
            report = {"control_mismatch": str(mismatch), "void_for_the_recorded_feedback": True}
        reports[name] = dict(report, counted=counted, food_band=food_band, first_episode=first,
                             last_episode=after_last - 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    return reports


# The check on the variant. Rules: docs/decisions/2026_11_16_choice_head_variant_sign.md.

SIGN_CONDITIONS = ("C1", "C2", "C3a", "C3b", "C3c", "C4")
SIGN_CHECKPOINTS = (0, 1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200)
SIGN_SIGNAL_SHARE = 1.0
# The targets of the loss that use the critic at its initial weights, for each ablation.
UNTRAINED_TARGETS = {"C3a": ("go",), "C3b": ("go", "no_go", "brake"), "C3c": ("no_go",)}
STEP_P1_HEAD = {"C1": "variant", "C4": "plain"}      # controls V1 and V2
STEP_P1_TOLERANCE = 1e-9
UPDATE_MEASURES = ("share_negative", "share_zero", "mean_abs_advantage", "mean_go_chosen", "mean_no_go",
                   "untrained_share_negative", "untrained_share_at_or_below_0")
MEAN_MEASURES = ("score", "score_go_only") + UPDATE_MEASURES
CASE_COLUMNS = ("condition", "rows", "chooser", "data_seed")
MEAN_COLUMNS = CASE_COLUMNS + ("checkpoint",) + MEAN_MEASURES
TRAINING_COLUMNS = (CASE_COLUMNS + ("initial_seed", "fold", "checkpoint") + MEAN_MEASURES
                    + ("first_update_share_above_half",))


def variant_sign_cases() -> List[dict]:
    """Each case of the check: C3a, C3b and C3c have the food chooser only, each other condition has four."""
    cases = []
    for condition in SIGN_CONDITIONS:
        choosers = ("food",) if condition in UNTRAINED_TARGETS else CHOOSERS
        cases += [{"condition": condition, "rows": rows, "chooser": chooser, "data_seed": data_seed}
                  for rows, chooser, data_seed in itertools.product(BUILT_ROWS, choosers, BUILT_DATA_SEEDS)]
    return cases


def describe(case: dict) -> str:
    return "%(condition)s, %(rows)d rows, chooser %(chooser)s, data set %(data_seed)d" % case


# The loss, with the critic that each target uses.

def variant_signals(head: Dict[str, nn.Module], rows: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The four outputs of the variant for a batch, with the working memory at zero (departure D2)."""
    state, _ = head["pfc"](rows, torch.zeros(rows.shape[0], CONTEXT_DIM))
    bg = head["bg"]
    return {"value": bg.critic(state), "go": bg.direct_pathway(state),
            "no_go": bg.indirect_pathway(state), "brake": bg.stn_pathway(state)}


def go_targets(action: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """The go target: the action with the sign of the advantage, and 0 for an advantage of 0."""
    return action * (advantage > 0).float() - action * (advantage < 0).float()


def sourced_loss_terms(signals, action, outcome, untrained_value, untrained_targets) -> Dict[str, torch.Tensor]:
    """The loss terms of variant_loss_terms. Each target named in untrained_targets takes its advantage
    from the untrained critic, and each other target from the learning critic. The value term always
    has the learning critic."""
    learning, untrained = outcome - signals["value"].detach(), outcome - untrained_value
    advantage = {name: untrained if name in untrained_targets else learning for name in ("go", "no_go", "brake")}
    target_no_go = (advantage["no_go"] < 0).float().expand_as(signals["no_go"])
    terms = {"value": nn.MSELoss()(signals["value"], outcome),
             "go": nn.MSELoss()(signals["go"], go_targets(action, advantage["go"]).detach()),
             "no_go": nn.BCELoss()(signals["no_go"], target_no_go.detach()),
             "brake": nn.MSELoss()(signals["brake"], torch.tanh(torch.abs(advantage["brake"]).detach()))}
    terms["total"] = terms["go"] + terms["no_go"] + 0.3 * terms["brake"] + 0.5 * terms["value"]
    return terms


# The measurements.

def advantage_measures(advantage: torch.Tensor) -> Dict[str, float]:
    """Measurements 1 and 2: the shares of rows with an advantage below 0 and of 0, and the mean size."""
    return {"share_negative": float((advantage < 0).float().mean()),
            "share_zero": float((advantage == 0).float().mean()),
            "mean_abs_advantage": float(advantage.abs().mean())}


def update_measures(signals, action, outcome, untrained_value) -> Dict[str, float]:
    """Measurements 1 to 3 of one update, before the weights change. They use the learning critic. With
    an untrained critic, the shares of its advantage are added."""
    with torch.no_grad():
        measures = advantage_measures(outcome - signals["value"])
        measures["mean_go_chosen"] = float((signals["go"] * action).sum(dim=1).mean())
        measures["mean_no_go"] = float(signals["no_go"].mean())
        if untrained_value is not None:
            untrained = outcome - untrained_value
            measures["untrained_share_negative"] = float((untrained < 0).float().mean())
            measures["untrained_share_at_or_below_0"] = float((untrained <= 0).float().mean())
    return measures


def go_only_score(go: np.ndarray, target: np.ndarray) -> float:
    """Measurement 5: the AUC of go of slot 0 minus go of slot 1, with no no-go factor."""
    return auc(go[:, 0] - go[:, 1], target)


def checkpoint_scores(head: Dict[str, nn.Module], split: dict, checkpoint: int) -> dict:
    """Measurements 4 and 5 on the test rows of the fold. The plain value head has no go signal."""
    target = np.asarray(split["test_target"])
    scores = {"checkpoint": checkpoint, "score": auc(preferences(head, split["test_rows"]), target),
              "score_go_only": ""}
    if "bg" in head:
        with torch.no_grad():
            go = variant_signals(head, torch.as_tensor(np.asarray(split["test_rows"]), dtype=torch.float32))["go"]
        scores["score_go_only"] = go_only_score(go.numpy().astype(np.float64), target)
    return scores


def first_turn_update(updates: List[dict]):
    """Measurement 6: the first update with a share of negative advantage above 0.50, or "none"."""
    if not updates or "share_negative" not in updates[0]:
        return ""
    turned = [number for number, update in enumerate(updates, start=1) if update["share_negative"] > 0.5]
    return turned[0] if turned else "none"


def window_means(updates: List[dict], first: int, last: int) -> dict:
    """The mean of each measurement over updates first to last. Blank with no update or no measurement."""
    window = updates[first - 1:last]
    return {name: float(np.mean([update[name] for update in window])) if window and name in window[0] else ""
            for name in UPDATE_MEASURES}


# The rows of built data.

def built_noise(rows: int, seed) -> np.ndarray:
    """The rows of built_case for this seed before the signal is added. It is the first draw there."""
    return np.random.default_rng(seed).normal(size=(rows, BUILT_DIMENSIONS))


def new_row_seed(data_seed: int, fold: int, update: int) -> List[int]:
    """The seed of a new row set of C2. The update is counted from 1 in the job."""
    return [data_seed, fold, update]


def new_row_noise(count: int, data_seed: int, fold: int, update: int) -> np.ndarray:
    return built_noise(count, new_row_seed(data_seed, fold, update))


def new_row_set(count: int, chooser: str, data_seed: int, fold: int, update: int) -> dict:
    """A new set of built rows for one update of C2, from the generator of the test on built data."""
    return built_case(count, SIGN_SIGNAL_SHARE, chooser, new_row_seed(data_seed, fold, update))


def rows_in_common(first: np.ndarray, second: np.ndarray) -> int:
    """The count of rows of the first array that are equal to a row of the second array."""
    known = {np.ascontiguousarray(row).tobytes() for row in np.asarray(second, dtype=np.float64)}
    return sum(np.ascontiguousarray(row).tobytes() in known for row in np.asarray(first, dtype=np.float64))


def fold_split(built: dict, fold: int) -> dict:
    """The training rows and the test rows of one fold, made as score_of_one_seed makes them."""
    rows = np.asarray(built["rows"], dtype=np.float64)
    fold_of = make_folds(built["target"], OUTER_FOLDS, FOLD_SEED)
    train, test = fold_of != fold, fold_of == fold
    train_rows, test_rows = standardize(rows[train], rows[test])
    return {"train_rows": train_rows, "test_rows": test_rows, "raw_train_rows": rows[train],
            "chosen": built["chosen"][train], "outcome": built["outcome"][train],
            "test_target": built["target"][test]}


def c2_split(built: dict, case: dict, fold: int) -> dict:
    """The fold of C2: the test rows of C1, and what the new row set of each update needs."""
    return dict(fold_split(built, fold), chooser=case["chooser"], data_seed=case["data_seed"], fold=fold,
                data_noise=built_noise(case["rows"], case["data_seed"]))


def case_split(built: dict, case: dict, fold: int) -> dict:
    return c2_split(built, case, fold) if case["condition"] == "C2" else fold_split(built, fold)


# The controls.

def check_food_outcomes(outcome: np.ndarray, where: str) -> None:
    """Control V7: with the food chooser the outcome of each row is 1."""
    if not np.all(np.asarray(outcome) == 1):
        raise ControlMismatch("V7, %s: an outcome with the food chooser is not 1" % where)


def check_new_rows(new_noise: np.ndarray, data_noise: np.ndarray, where: str) -> None:
    """Control V8: no row of a new row set is a row of the data set, before the signal is added."""
    common = rows_in_common(new_noise, data_noise)
    if common:
        raise ControlMismatch("V8, %s: %d rows of a new row set are rows of the data set" % (where, common))


def untrained_values(seed: int, rows) -> torch.Tensor:
    """The value of the critic of the variant at its initial weights, for each row."""
    rows = torch.as_tensor(np.asarray(rows), dtype=torch.float32)
    with torch.no_grad():
        return variant_signals(build_head("variant", rows.shape[1], seed), rows)["value"]


def check_untrained_critic(built: dict, where: str) -> Tuple[int, float]:
    """Control V6 before a training: no advantage of the untrained critic is 0 or below on a training
    row. Returns the count of values read and the highest value of the critic."""
    count, highest = 0, -float("inf")
    for seed, fold in itertools.product(INITIAL_SEEDS, range(OUTER_FOLDS)):
        split = fold_split(built, fold)
        critic_values = untrained_values(seed, split["train_rows"])
        outcome = torch.as_tensor(np.asarray(split["outcome"]), dtype=torch.float32).unsqueeze(1)
        if bool(((outcome - critic_values) <= 0).any()):
            raise ControlMismatch("V6, %s, seed %d, fold %d: an advantage of the untrained critic is 0 or "
                                  "below" % (where, seed, fold))
        count, highest = count + critic_values.numel(), max(highest, float(critic_values.max()))
    return count, highest


def check_test_rows_of_c2(built: dict, case: dict) -> int:
    """Control V9: the test rows of each fold of C2 equal those of C1. Returns the count of folds."""
    for fold in range(OUTER_FOLDS):
        if not np.array_equal(c2_split(built, case, fold)["test_rows"], fold_split(built, fold)["test_rows"]):
            raise ControlMismatch("V9, %s, fold %d: the test rows of C2 are not those of C1" % (describe(case), fold))
    return OUTER_FOLDS


def first_step(cases: List[dict]) -> Dict[str, float]:
    """Controls V6, V7 and V9 for each case that they apply to, with no training. The job runs this
    before its first training and stops on a mismatch."""
    report = {"V7_rows": 0, "V9_folds": 0, "V6_highest_value_of_the_untrained_critic": -float("inf")}
    for case in cases:
        built = built_case(case["rows"], SIGN_SIGNAL_SHARE, case["chooser"], case["data_seed"])
        if case["chooser"] == "food":
            check_food_outcomes(built["outcome"], describe(case))
            report["V7_rows"] += len(built["outcome"])
        if case["condition"] in UNTRAINED_TARGETS:
            count, highest = check_untrained_critic(built, describe(case))
            name = "V6_values_%s" % case["condition"]
            report[name] = report.get(name, 0) + count
            report["V6_highest_value_of_the_untrained_critic"] = max(
                report["V6_highest_value_of_the_untrained_critic"], highest)
        if case["condition"] == "C2":
            report["V9_folds"] += check_test_rows_of_c2(built, case)
    return report


def check_measurements_keep_the_training(case: dict) -> None:
    """Control V3: the trained weights of one case of C1 are equal with and without the measurements."""
    built = built_case(case["rows"], SIGN_SIGNAL_SHARE, case["chooser"], case["data_seed"])
    for seed, fold in itertools.product(INITIAL_SEEDS, range(OUTER_FOLDS)):
        split = fold_split(built, fold)
        measured = head_parameters(train_measured("C1", seed, split)["head"])
        expected = head_parameters(train_head("variant", seed, split["train_rows"], split["chosen"], split["outcome"]))
        if any(not torch.equal(a, b) for a, b in zip(measured, expected)):
            raise ControlMismatch("V3, %s, seed %d, fold %d: the measurements change the weights"
                                  % (describe(case), seed, fold))


def load_step_p1_scores(path: str) -> Dict[tuple, float]:
    """The scores of the test on built data of step P1 at the full signal, first pass, for V1 and V2."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {(int(row["rows"]), row["chooser"], int(row["data_seed"]), row["head"]): float(row["score"])
                for row in csv.DictReader(handle)
                if row["outcomes_at_1"] == "False" and float(row["signal_share"]) == SIGN_SIGNAL_SHARE}


def check_against_step_p1(case: dict, mean_rows: List[dict], step_p1_scores: Dict[tuple, float]) -> None:
    """Controls V1 (C1) and V2 (C4): the score at 200 updates equals the score of step P1."""
    if case["condition"] not in STEP_P1_HEAD:
        return
    expected = step_p1_scores[(case["rows"], case["chooser"], case["data_seed"], STEP_P1_HEAD[case["condition"]])]
    last = mean_rows[-1]
    if last["checkpoint"] != UPDATES or abs(last["score"] - expected) > STEP_P1_TOLERANCE:
        raise ControlMismatch("%s, %s: the score at checkpoint %d is %r and step P1 has %r"
                              % ("V1" if case["condition"] == "C1" else "V2", describe(case),
                                 last["checkpoint"], last["score"], expected))


# The training with the measurements.

def as_batch(split: dict) -> Dict[str, torch.Tensor]:
    """The tensors of train_head, from the training rows of a fold."""
    return {"rows": torch.as_tensor(np.asarray(split["train_rows"]), dtype=torch.float32),
            "chosen": torch.as_tensor(np.asarray(split["chosen"]), dtype=torch.long),
            "outcome": torch.as_tensor(np.asarray(split["outcome"]), dtype=torch.float32).unsqueeze(1)}


def c2_batch(split: dict, update: int) -> Dict[str, torch.Tensor]:
    """The new row set of one update of C2, standardized with the training rows of the fold. Controls
    V8 and V7 are checked for each set."""
    count, where = len(split["train_rows"]), "data set %d, fold %d, update %d" % (
        split["data_seed"], split["fold"], update)
    check_new_rows(new_row_noise(count, split["data_seed"], split["fold"], update), split["data_noise"], where)
    new_set = new_row_set(count, split["chooser"], split["data_seed"], split["fold"], update)
    if split["chooser"] == "food":
        check_food_outcomes(new_set["outcome"], where)
    _, new_rows = standardize(split["raw_train_rows"], new_set["rows"])
    return as_batch({"train_rows": new_rows, "chosen": new_set["chosen"], "outcome": new_set["outcome"]})


def measured_loss(condition: str, head, untrained, batch) -> tuple:
    """The loss of one update, its measurements and the value of the untrained critic, from one forward
    pass of the head. C1 and C2 use the loss of step P1 unchanged."""
    if condition == "C4":
        return plain_loss(head["net"], batch["rows"], batch["chosen"], batch["outcome"]), {}, None
    signals, action, outcome = variant_signals(head, batch["rows"]), action_vectors(batch["chosen"]), batch["outcome"]
    if untrained is None:
        terms = variant_loss_terms(signals["value"], signals["go"], signals["no_go"], signals["brake"], action, outcome)
        return terms["total"], update_measures(signals, action, outcome, None), None
    with torch.no_grad():
        untrained_value = variant_signals(untrained, batch["rows"])["value"]
    terms = sourced_loss_terms(signals, action, outcome, untrained_value, UNTRAINED_TARGETS[condition])
    return terms["total"], update_measures(signals, action, outcome, untrained_value), untrained_value


def check_untrained_value(record: dict, measures: dict, untrained_value, last: bool) -> None:
    """Controls V6 (at each update) and V5 (the value is equal at the first and at the last update)."""
    if untrained_value is None:
        return
    if measures["untrained_share_at_or_below_0"] > 0:
        raise ControlMismatch("V6: an advantage of the untrained critic is 0 or below at an update")
    record.setdefault("untrained_value_first", untrained_value)
    if last:
        record["untrained_value_last"] = untrained_value
        if not torch.equal(record["untrained_value_first"], untrained_value):
            raise ControlMismatch("V5: the value of the untrained critic changed between the first and the "
                                  "last update")


def add_checkpoint(record: dict, split: dict, updates_done: int, checkpoints) -> None:
    if updates_done in checkpoints:
        record["checkpoints"].append(checkpoint_scores(record["head"], split, updates_done))


def train_measured(condition: str, seed: int, split: dict, updates: int = UPDATES,
                   checkpoints=SIGN_CHECKPOINTS) -> dict:
    """One training of a condition with the fixed training values of step P1. It returns the head, the
    measurements of each update (before the weights change) and the scores at each checkpoint."""
    batch = as_batch(split)
    head = build_head("plain" if condition == "C4" else "variant", batch["rows"].shape[1], seed)
    untrained = copy.deepcopy(head) if condition in UNTRAINED_TARGETS else None
    optimizer = torch.optim.Adam(head_parameters(head), lr=LEARNING_RATE)
    record = {"head": head, "updates": [], "checkpoints": []}
    for update in range(1, updates + 1):
        add_checkpoint(record, split, update - 1, checkpoints)
        if condition == "C2":
            batch = c2_batch(split, update)
        loss, measures, untrained_value = measured_loss(condition, head, untrained, batch)
        check_untrained_value(record, measures, untrained_value, update == updates)
        record["updates"].append(measures)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    add_checkpoint(record, split, updates, checkpoints)
    return record


# The rows of the check.

def training_rows_of(case: dict, seed: int, fold: int, record: dict) -> List[dict]:
    """One row for each checkpoint of one training. Measurements 1 to 3 are means over the updates
    from the checkpoint before to this one."""
    rows, previous = [], 0
    for scores in record["checkpoints"]:
        rows.append(dict(case, initial_seed=seed, fold=fold, **scores,
                         **window_means(record["updates"], previous + 1, scores["checkpoint"]),
                         first_update_share_above_half=first_turn_update(record["updates"])))
        previous = scores["checkpoint"]
    return rows


def mean_or_blank(numbers: list):
    return "" if any(number == "" for number in numbers) else float(np.mean(numbers))


def mean_rows_of(case: dict, training_rows: List[dict]) -> List[dict]:
    """One row for each checkpoint of a case: the means over the initial seeds and the folds."""
    mean_rows = []
    for checkpoint in sorted({row["checkpoint"] for row in training_rows}):
        members = [row for row in training_rows if row["checkpoint"] == checkpoint]
        mean_rows.append(dict(case, checkpoint=checkpoint,
                              **{name: mean_or_blank([row[name] for row in members]) for name in MEAN_MEASURES}))
    return mean_rows


def run_case(case: dict, updates: int = UPDATES, checkpoints=SIGN_CHECKPOINTS) -> Tuple[List[dict], List[dict]]:
    """Each training of one case. Returns the rows for each training and the mean rows."""
    built = built_case(case["rows"], SIGN_SIGNAL_SHARE, case["chooser"], case["data_seed"])
    training_rows = []
    for seed, fold in itertools.product(INITIAL_SEEDS, range(OUTER_FOLDS)):
        record = train_measured(case["condition"], seed, case_split(built, case, fold), updates, checkpoints)
        training_rows.extend(training_rows_of(case, seed, fold, record))
    return training_rows, mean_rows_of(case, training_rows)


def write_cases(out_dir: str, cases: List[dict], step_p1_scores: Dict[tuple, float]) -> None:
    """The rows of each case, written at once so that the row counts show the progress."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mean_rows.csv"), "w", newline="", encoding="utf-8") as mean_file, \
            open(os.path.join(out_dir, "training_rows.csv"), "w", newline="", encoding="utf-8") as training_file:
        mean_writer = csv.DictWriter(mean_file, fieldnames=MEAN_COLUMNS)
        training_writer = csv.DictWriter(training_file, fieldnames=TRAINING_COLUMNS)
        mean_writer.writeheader()
        training_writer.writeheader()
        for case in cases:
            training_rows, mean_rows = run_case(case)
            check_against_step_p1(case, mean_rows, step_p1_scores)
            training_writer.writerows(training_rows)
            mean_writer.writerows(mean_rows)
            mean_file.flush()
            training_file.flush()


def run_variant_sign(out_dir: str, step_p1_csv: str, cases: List[dict] = None) -> None:
    """The job of the check. V6, V7 and V9 come before the first training, then V3, then each case with
    V1, V2, V5, V6 and V8. A control mismatch stops the job, and the check is then VOID."""
    cases = variant_sign_cases() if cases is None else cases
    print(json.dumps(first_step(cases), indent=2), flush=True)
    step_p1_scores = load_step_p1_scores(step_p1_csv)
    c1_cases = [case for case in cases if case["condition"] == "C1"]
    if c1_cases:
        check_measurements_keep_the_training(c1_cases[0])
    write_cases(out_dir, cases, step_p1_scores)


def time_trainings() -> Dict[str, float]:
    """The seconds of one training of each condition with each measurement active, at 40 rows with the
    food chooser, and the estimate for the full job from these times. It writes no file. One training
    of 5 updates comes first and is not timed, so that no time has the start of the process in it."""
    seconds = {}
    warm_up = built_case(BUILT_ROWS[-1], SIGN_SIGNAL_SHARE, "food", 0)
    train_measured("C1", 0, fold_split(warm_up, 0), updates=5, checkpoints=(0, 5))
    for condition in SIGN_CONDITIONS:
        case = {"condition": condition, "rows": BUILT_ROWS[-1], "chooser": "food", "data_seed": 0}
        built = built_case(case["rows"], SIGN_SIGNAL_SHARE, case["chooser"], case["data_seed"])
        start = time.perf_counter()
        training_rows_of(case, 0, 0, train_measured(condition, 0, case_split(built, case, 0)))
        seconds[condition] = time.perf_counter() - start
    trainings = len(INITIAL_SEEDS) * OUTER_FOLDS
    total = sum(seconds[case["condition"]] * trainings for case in variant_sign_cases())
    seconds["estimate_minutes_of_the_full_job"] = (total + 2 * trainings * seconds["C1"]) / 60.0
    return seconds


# The labels of the check on the variant, from mean_rows.csv and training_rows.csv of the job and from
# nothing else. Each label rule is that of the last section of the rules that names it.

LIMIT_FLOOR = 0.05           # change 19: the limit is the larger of this and 2 standard errors
HIGH_SCORE = 0.80
HALF = 0.50
SMALL_ADVANTAGE = 0.25       # d1
FIRST_READ_CHECKPOINT = 20   # a2 and reading 2 do not use checkpoints 0 and 1
A2_LAST_EARLY_CHECKPOINT = 100
D2_EARLY_CHECKPOINTS = (40, 60, 80, 100)
D2_LATE_CHECKPOINTS = (180, 200)
NO_LABEL_CHOOSERS = ("slot_0", "food_80")      # change 20
TEXT_COLUMNS = ("condition", "chooser")
INTEGER_COLUMNS = ("rows", "data_seed", "initial_seed", "fold", "checkpoint", "first_update_share_above_half")
# The rows of the table for step P2 in the order of the rules, each with its place in the order of
# advice. The row of d is a statement on the cause only and has no place.
P2_ROWS = (("a HOLDS and b FAILS", 2), ("a PARTLY and b FAILS", 2), ("b HOLDS", 3), ("b PARTLY", 2),
           ("a FAILS", 2), ("c HOLDS", 4), ("d HOLDS and a HOLDS", None))
P2_ADVICE = {2: "Do not build with the variant", 3: "A build needs a check at the amount of data of a run",
             4: "A new rule for the owner", 5: "The author reports the table and the owner decides"}


def parse_cell(name: str, text: str):
    """A blank cell is None. The word "none" of measurement 6 stays a word."""
    if name in TEXT_COLUMNS or text == "none":
        return text
    if text == "":
        return None
    return int(text) if name in INTEGER_COLUMNS else float(text)


def read_rows(path: str) -> List[dict]:
    """The rows of mean_rows.csv or of training_rows.csv, with numbers in place of text."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [{name: parse_cell(name, text) for name, text in row.items()} for row in csv.DictReader(handle)]


def check_complete(mean_rows: List[dict], training_rows: List[dict]) -> None:
    """Control V4: the job is complete only with each mean row and each row for a training."""
    cases = variant_sign_cases()
    expected_mean = len(cases) * len(SIGN_CHECKPOINTS)
    expected_training = expected_mean * len(INITIAL_SEEDS) * OUTER_FOLDS
    if (len(mean_rows), len(training_rows)) != (expected_mean, expected_training):
        raise ControlMismatch("V4: %d mean rows and %d rows for each training, not %d and %d. The job is not "
                              "complete" % (len(mean_rows), len(training_rows), expected_mean, expected_training))


class MeanTable:
    """The mean rows of the job, by condition, row count, chooser and checkpoint."""

    def __init__(self, mean_rows: List[dict]):
        self.rows_of = {}
        for row in mean_rows:
            key = (row["condition"], row["rows"], row["chooser"], row["checkpoint"])
            self.rows_of.setdefault(key, {})[row["data_seed"]] = row

    def has(self, condition: str, rows: int, chooser: str) -> bool:
        return (condition, rows, chooser, UPDATES) in self.rows_of

    def numbers(self, condition: str, rows: int, chooser: str, checkpoint: int, measure: str) -> np.ndarray:
        """The 10 numbers of the data sets, in the order of the data seeds. A missing one is an error."""
        members = self.rows_of.get((condition, rows, chooser, checkpoint), {})
        where = "%s, %d rows, chooser %s, checkpoint %d, %s" % (condition, rows, chooser, checkpoint, measure)
        if sorted(members) != list(BUILT_DATA_SEEDS):
            raise ControlMismatch("%s: the rows do not have each of the 10 data sets" % where)
        numbers = [members[data_seed][measure] for data_seed in BUILT_DATA_SEEDS]
        if any(number is None for number in numbers):
            raise ControlMismatch("%s: a number is blank" % where)
        return np.array(numbers, dtype=np.float64)

    def mean(self, condition: str, rows: int, chooser: str, checkpoint: int, measure: str) -> float:
        return float(np.mean(self.numbers(condition, rows, chooser, checkpoint, measure)))


def limit_of_change_19(numbers: np.ndarray) -> Dict[str, float]:
    """The larger of 0.05 and 2 standard errors of the mean over the 10 data sets. The standard error is
    the sample standard deviation with the divisor 9, divided by the square root of 10."""
    if len(numbers) != len(BUILT_DATA_SEEDS):
        raise ControlMismatch("the limit of change 19 needs 10 numbers, got %d" % len(numbers))
    standard_error = float(np.std(numbers, ddof=1) / math.sqrt(len(numbers)))
    return {"standard_error": standard_error, "limit": max(LIMIT_FLOOR, 2.0 * standard_error)}


# What is read at one row count.

def read_a1_a2(table: MeanTable, rows: int) -> dict:
    """Predictions a1 and a2, from C1 with the food chooser."""
    first_shares = table.numbers("C1", rows, "food", 1, "share_negative")
    limit, late_share = limit_of_change_19(first_shares), table.mean("C1", rows, "food", UPDATES, "share_negative")
    early = [table.mean("C1", rows, "food", checkpoint, "score") for checkpoint in SIGN_CHECKPOINTS
             if FIRST_READ_CHECKPOINT <= checkpoint <= A2_LAST_EARLY_CHECKPOINT]
    score_at_200 = table.mean("C1", rows, "food", UPDATES, "score")
    return {"a1_part_1_share_at_update_1": float(np.mean(first_shares)), "a1_part_1_limit": limit,
            "a1_part_1": bool(np.mean(first_shares) <= limit["limit"]),
            "a1_part_2_share": late_share, "a1_part_2": late_share > HALF,
            "a2_highest_score_from_20_to_100": max(early), "a2_score_at_200": score_at_200,
            "a2": max(early) >= HIGH_SCORE and score_at_200 < HALF}


def read_c3(table: MeanTable, rows: int) -> dict:
    """Prediction a3 with measurement 7, and the scores of C3b and C3c at 200."""
    facts = {}
    for name, condition in (("a3", "C3a"), ("C3b", "C3b"), ("C3c", "C3c")):
        facts[name + "_score"] = table.mean(condition, rows, "food", UPDATES, "score")
        facts[name] = facts[name + "_score"] >= HIGH_SCORE
    facts["measurement_7"] = table.mean("C3a", rows, "food", UPDATES, "share_negative")
    facts["c3a_says_nothing_about_c"] = facts["measurement_7"] <= HALF
    return facts


def read_b(table: MeanTable, rows: int) -> dict:
    """Predictions b1, b2 and b3, from C2."""
    facts = {"b1_share": table.mean("C2", rows, "food", UPDATES, "share_negative"),
             "b2_score": table.mean("C2", rows, "food", UPDATES, "score"),
             "b3_score": table.mean("C2", rows, "random", UPDATES, "score")}
    facts.update(b1=facts["b1_share"] <= HALF, b2=facts["b2_score"] >= HIGH_SCORE, b3=facts["b3_score"] >= HIGH_SCORE)
    return facts


def read_reading_2(table: MeanTable, rows: int, a1_part_2: bool) -> dict:
    """Prediction b4 and the label of reading 2 at one row count, from the checkpoints of C1 with the
    food chooser. With no turn, no checkpoint is before the turn."""
    turned = [checkpoint for checkpoint in SIGN_CHECKPOINTS[1:]
              if table.mean("C1", rows, "food", checkpoint, "share_negative") > HALF]
    turn = turned[0] if turned else None
    before = [checkpoint for checkpoint in SIGN_CHECKPOINTS
              if turn is not None and FIRST_READ_CHECKPOINT <= checkpoint < turn]
    scores = [table.mean("C1", rows, "food", checkpoint, "score") for checkpoint in before]
    b4 = all(score > HALF for score in scores) if before else None
    label = "NOT DECIDED"
    if before and not b4:
        label = "FAILS"
    elif before and a1_part_2:
        label = "HOLDS"
    return {"reading_2_turn_checkpoint": turn, "reading_2_checkpoints_before_the_turn": before,
            "reading_2_scores_before_the_turn": scores, "b4": b4, "reading_2_at_this_row_count": label}


def read_c(table: MeanTable, rows: int) -> dict:
    """Prediction c1 and the FAILS rule of c, from C1 with the food chooser at 200."""
    go_only = table.numbers("C1", rows, "food", UPDATES, "score_go_only")
    differences = go_only - table.numbers("C1", rows, "food", UPDATES, "score")
    limit = limit_of_change_19(differences)
    return {"c1_score_go_only": float(np.mean(go_only)), "c1": bool(np.mean(go_only) >= HIGH_SCORE),
            "c_difference": float(np.mean(differences)), "c_limit": limit,
            "c_scores_within_the_limit": bool(abs(np.mean(differences)) <= limit["limit"])}


def read_d(table: MeanTable, rows: int) -> dict:
    """Predictions d1 and d2, from C1 with the random chooser."""
    def mean_score(checkpoints) -> np.ndarray:
        return np.mean([table.numbers("C1", rows, "random", checkpoint, "score") for checkpoint in checkpoints], axis=0)

    differences = mean_score(D2_EARLY_CHECKPOINTS) - mean_score(D2_LATE_CHECKPOINTS)
    limit = limit_of_change_19(differences)
    advantage = table.mean("C1", rows, "random", UPDATES, "mean_abs_advantage")
    return {"d1_mean_abs_advantage": advantage, "d1": advantage < SMALL_ADVANTAGE,
            "d2_difference": float(np.mean(differences)), "d2_limit": limit,
            "d2": bool(np.mean(differences) >= limit["limit"])}


def read_row_count(table: MeanTable, rows: int) -> dict:
    """Each prediction and each number of a label rule, at one row count."""
    facts = dict(read_a1_a2(table, rows), **read_c3(table, rows))
    facts.update(read_b(table, rows))
    facts.update(read_reading_2(table, rows, facts["a1_part_2"]))
    facts.update(read_c(table, rows))
    facts.update(read_d(table, rows))
    return facts


# The labels. A prediction holds only when it holds at both row counts.

def at_both(by_rows: Dict[int, dict], name: str) -> bool:
    return all(by_rows[rows][name] for rows in BUILT_ROWS)


def predictions_of(by_rows: Dict[int, dict]) -> dict:
    names = ("a1_part_1", "a1_part_2", "a2", "a3", "C3b", "C3c", "b1", "b2", "b3", "c1", "d1", "d2")
    predictions = {name: at_both(by_rows, name) for name in names}
    b4 = [by_rows[rows]["b4"] for rows in BUILT_ROWS]      # None at a row count with no checkpoint to read
    predictions["b4"] = None if None in b4 else all(b4)
    return predictions


def label_of_a(by_rows: Dict[int, dict]) -> str:
    """Change 5 with owner decisions 1 and 2 after part 3. FAILS needs the same way at both row counts.
    HOLDS and PARTLY need each condition at both row counts. When part 1 of a1 fails, the label is NOT
    DECIDED also when a way of FAILS is met (owner decision 1 on the label code)."""
    no_turn = all(not by_rows[rows]["a1_part_2"] for rows in BUILT_ROWS)       # the first way
    c3b_low = all(not by_rows[rows]["C3b"] for rows in BUILT_ROWS)             # the second way
    if no_turn or c3b_low:
        return "FAILS" if at_both(by_rows, "a1_part_1") else "NOT DECIDED"
    if not all(at_both(by_rows, name) for name in ("a1_part_1", "a1_part_2", "a2", "C3b")):
        return "NOT DECIDED"
    if at_both(by_rows, "a3"):
        return "HOLDS"
    return "PARTLY" if all(not by_rows[rows]["a3"] for rows in BUILT_ROWS) else "NOT DECIDED"


def label_of_b(by_rows: Dict[int, dict]) -> str:
    """The label of reading 1. b has no NOT DECIDED."""
    if not (at_both(by_rows, "b1") and at_both(by_rows, "b2")):
        return "FAILS"
    return "HOLDS" if at_both(by_rows, "b3") else "PARTLY"


def label_of_reading_2(by_rows: Dict[int, dict]) -> str:
    """Necessary change 1 of part 3. HOLDS and FAILS need both row counts."""
    labels = {by_rows[rows]["reading_2_at_this_row_count"] for rows in BUILT_ROWS}
    return labels.pop() if len(labels) == 1 else "NOT DECIDED"


def label_of_c(by_rows: Dict[int, dict]) -> str:
    """FAILS when the two scores are within the limit at both row counts, also when c1 holds (owner
    decision 2 on the label code). Else HOLDS when c1 holds."""
    if at_both(by_rows, "c_scores_within_the_limit"):
        return "FAILS"
    return "HOLDS" if at_both(by_rows, "c1") else "NOT DECIDED"


def label_of_d(by_rows: Dict[int, dict]) -> str:
    if not at_both(by_rows, "d1"):
        return "FAILS"
    return "HOLDS" if at_both(by_rows, "d2") else "NOT DECIDED"


def step_p2_rows(labels: Dict[str, str]) -> dict:
    """The rows of the table for step P2 that apply, and the advice with the first place in the order.
    The VOID row has place 1. It is not read from the rows: a control mismatch stops the job."""
    a, b = labels["a"], labels["b"]
    applies = {"a HOLDS and b FAILS": a == "HOLDS" and b == "FAILS", "a PARTLY and b FAILS": a == "PARTLY" and b == "FAILS",
               "b HOLDS": b == "HOLDS", "b PARTLY": b == "PARTLY", "a FAILS": a == "FAILS",
               "c HOLDS": labels["c"] == "HOLDS", "d HOLDS and a HOLDS": labels["d"] == "HOLDS" and a == "HOLDS"}
    names = [name for name, _ in P2_ROWS if applies[name]]
    places = [place for name, place in P2_ROWS if applies[name] and place is not None]
    if not places:
        names, places = names + ["Each other case"], [5]
    return {"rows": names, "place": min(places), "advice": P2_ADVICE[min(places)]}


# What is reported with no label.

def count_a2_trainings(training_rows: List[dict], rows: int) -> Dict[str, int]:
    """For C1 with the food chooser: the count of trainings in which the first update of measurement 6
    comes before the first checkpoint with a score below 0.50. Update u comes before checkpoint c when
    u is c or less, because checkpoint c is read after update c."""
    trainings = {}
    for row in training_rows:
        if (row["condition"], row["chooser"], row["rows"]) == ("C1", "food", rows):
            trainings.setdefault((row["data_seed"], row["initial_seed"], row["fold"]), []).append(row)
    counts = {"trainings": len(trainings), "turn_before_the_first_low_score": 0, "no_turn": 0, "no_low_score": 0}
    for members in trainings.values():
        turn = members[0]["first_update_share_above_half"]
        low = [row["checkpoint"] for row in members if row["score"] < HALF]
        if turn is None:
            raise ControlMismatch("a training of C1 has no measurement 6")
        counts["no_turn"] += turn == "none"
        counts["no_low_score"] += not low
        counts["turn_before_the_first_low_score"] += turn != "none" and bool(low) and turn <= min(low)
    return counts


def measurement_8(mean_rows: List[dict]) -> List[dict]:
    """For each case: the score at checkpoint 1 minus the score at checkpoint 0. It is not a control."""
    scores = {(row["condition"], row["rows"], row["chooser"], row["data_seed"], row["checkpoint"]): row["score"]
              for row in mean_rows if row["checkpoint"] in (0, 1)}
    return [{"condition": condition, "rows": rows, "chooser": chooser, "data_seed": data_seed,
             "difference": scores[(condition, rows, chooser, data_seed, 1)] - score_at_0}
            for (condition, rows, chooser, data_seed, checkpoint), score_at_0 in scores.items() if checkpoint == 0]


def report_with_no_label(table: MeanTable) -> List[dict]:
    """Change 20: the slot_0 and food_80 choosers are reported with no label."""
    entries = []
    for condition, chooser, rows in itertools.product(("C1", "C2", "C4"), NO_LABEL_CHOOSERS, BUILT_ROWS):
        if table.has(condition, rows, chooser):
            entries.append({"condition": condition, "chooser": chooser, "rows": rows, "label": "none",
                            "score_at_200": table.mean(condition, rows, chooser, UPDATES, "score")})
    return entries


def label_report_of(mean_rows: List[dict], training_rows: List[dict]) -> dict:
    """Each prediction, each label, each limit, and the rows of the table for step P2 that apply."""
    table = MeanTable(mean_rows)
    by_rows = {rows: read_row_count(table, rows) for rows in BUILT_ROWS}
    labels = {"a": label_of_a(by_rows), "b": label_of_b(by_rows), "reading_2": label_of_reading_2(by_rows),
              "c": label_of_c(by_rows), "d": label_of_d(by_rows)}
    return {"labels": labels, "predictions": predictions_of(by_rows), "row_counts": by_rows,
            "step_p2": step_p2_rows(labels),
            "a2_trainings": {rows: count_a2_trainings(training_rows, rows) for rows in BUILT_ROWS},
            "measurement_8": measurement_8(mean_rows), "no_label": report_with_no_label(table),
            "not_answered": "The question why the variant is below 0.80 is not answered for slot_0 and food_80"}


def run_labels(out_dir: str) -> dict:
    """The label report of a complete job, from its two files. It is written to labels.json there."""
    mean_rows = read_rows(os.path.join(out_dir, "mean_rows.csv"))
    training_rows = read_rows(os.path.join(out_dir, "training_rows.csv"))
    check_complete(mean_rows, training_rows)
    report = label_report_of(mean_rows, training_rows)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


# Check 2 of step P2 of design P (docs/decisions/2026_11_22_workspace_choice_mode.md, with its sections
# "Changes after the independent review" and "Owner decision on the scaling"). Built data only.

MODE_PASSES = ("first", "second")               # the second pass sets each outcome to 1
MODE_PARTS = ("A", "B", "O2")
MODE_A_CHECKPOINTS = SIGN_CHECKPOINTS
MODE_B_CHECKPOINTS = (0, 10, 20, 30, 40, 50, 60)
MODE_B_EPISODES = 60
MODE_B_SCORE_ROWS = 40                          # N3: part B is scored on the data set of 40 rows
MODE_B_LABEL_EPISODES = (40, 60)                # N11
MODE_SEED_STEP = 100                            # O1: the seed of a head is 100 x data seed + initial seed
PART_B_ROWS_TAG, EXPLORATION_TAG = 1, 2         # the last value of the seed sequences of the two streams
MODE_MEASURES = ("score", "count_choices", "count_explored", "share_explored", "share_food", "share_slot_0",
                 "mean_outcome", "informative")
MODE_CASE_COLUMNS = ("part", "pass", "rows", "data_seed")
MODE_MEAN_COLUMNS = MODE_CASE_COLUMNS + ("checkpoint",) + MODE_MEASURES
MODE_TRAINING_COLUMNS = MODE_CASE_COLUMNS + ("initial_seed", "fold", "checkpoint") + MODE_MEASURES
MODE_TEXT_COLUMNS = ("part", "pass")
MODE_INTEGER_COLUMNS = ("rows", "data_seed", "initial_seed", "fold", "checkpoint")
MODE_NEXT_STEP = {
    ("HOLDS", "HOLDS"): "The build (work 4) can start after the owner's yes",
    ("HOLDS", "PARTLY"): "The build can start after the owner's yes. A gate of step P3 then needs runs of 60 "
                         "episodes or more. The owner decides",
    ("HOLDS", "FAILS"): "The author proposes no build. A change of the learning rule is a new rule with its own "
                        "check. The owner decides",
    ("FAILS", "FAILS"): "The author proposes no build. The owner decides"}
MODE_OTHER_CASE = "Each other case: NOT DECIDED. The author reports the tables, and the owner decides"


def mode_head_seed(data_seed: int, initial_seed: int) -> int:
    """O1: the seed of the head of initial seed s in data set d is 100 x d + s, so the 30 heads differ."""
    return MODE_SEED_STEP * data_seed + initial_seed


def exploration_seed(initial_seed: int, data_seed: int, fold: int) -> List[int]:
    """The seed sequence of the exploration. It has no pass and no row count, so these repeat its draws."""
    return [initial_seed, data_seed, fold, EXPLORATION_TAG]


def part_b_row_seed(data_seed: int, repeat: int) -> List[int]:
    """The seed sequence of the rows of the episodes of part B and of the arm of O2."""
    return [data_seed, repeat, 0, PART_B_ROWS_TAG]


def mode_cases() -> List[dict]:
    """Part A at both row counts, then part B and the arm of O2 with their one score, each in both passes."""
    cases = [{"part": "A", "pass": run_pass, "rows": rows, "data_seed": data_seed}
             for run_pass, rows, data_seed in itertools.product(MODE_PASSES, BUILT_ROWS, BUILT_DATA_SEEDS)]
    return cases + [{"part": part, "pass": run_pass, "rows": MODE_B_SCORE_ROWS, "data_seed": data_seed}
                    for part, run_pass, data_seed in itertools.product(("B", "O2"), MODE_PASSES, BUILT_DATA_SEEDS)]


def mode_checkpoints(part: str) -> tuple:
    return MODE_A_CHECKPOINTS if part == "A" else MODE_B_CHECKPOINTS


def mode_expected_counts() -> Tuple[int, int]:
    """Control K1: the mean rows and the training rows of a complete job."""
    mean_count = sum(len(mode_checkpoints(case["part"])) for case in mode_cases())
    return mean_count, mean_count * len(INITIAL_SEEDS) * OUTER_FOLDS


def check_k3(new_noise: np.ndarray, data_noise: np.ndarray, where: str) -> None:
    """Control K3: no new row equals a row of the data set of its case, before the signal is added."""
    common = rows_in_common(new_noise, data_noise)
    if common:
        raise ControlMismatch("K3, %s: %d new rows are rows of the data set" % (where, common))


def mode_outcomes(chosen: np.ndarray, target: np.ndarray, outcomes_at_1: bool) -> np.ndarray:
    """The outcome of each choice. Control K5 checks it before the head receives it."""
    outcome = np.ones(len(chosen)) if outcomes_at_1 else built_outcomes(np.asarray(chosen), np.asarray(target))
    expected = np.ones(len(chosen)) if outcomes_at_1 else (np.asarray(chosen) == 1 - np.asarray(target)) * 1.0
    if not np.array_equal(outcome, expected):
        raise ControlMismatch("K5: an outcome is not the outcome of its choice")
    return outcome


class ChoiceWindow:
    """The choices since the last checkpoint, and the counts of each choice so far for the count rule."""

    def __init__(self):
        self.window = {"chosen": [], "explored": [], "food": [], "outcome": []}
        self.counts = np.zeros((2, 2), dtype=int)       # slot x outcome, over each choice so far

    def add(self, chosen, explored, food_slot, outcome) -> None:
        for name, numbers in (("chosen", chosen), ("explored", explored), ("food", food_slot), ("outcome", outcome)):
            self.window[name].extend(np.asarray(numbers).tolist())
        for slot, result in zip(np.asarray(chosen), np.asarray(outcome)):
            self.counts[int(slot), int(result == 1.0)] += 1

    def close(self) -> dict:
        """The measures of the window, which then starts again. A window with no choice has blank shares."""
        chosen, explored = np.array(self.window["chosen"]), np.array(self.window["explored"], dtype=bool)
        shares = {"share_explored": explored, "share_food": chosen == np.array(self.window["food"]),
                  "share_slot_0": chosen == 0, "mean_outcome": np.array(self.window["outcome"])}
        measures = {"count_choices": len(chosen), "count_explored": int(explored.sum()),
                    "informative": int(self.counts.min(axis=1).max())}
        measures.update({name: float(np.mean(x)) if len(chosen) else "" for name, x in shares.items()})
        self.window = {name: [] for name in self.window}
        return measures


def learn_from_set(head: ChoiceHead, rows, target, outcomes_at_1: bool, window: ChoiceWindow,
                   steps: int) -> np.ndarray:
    """One choice for each row by the exploration rule, the outcomes, and `steps` updates on these rows.
    The choices read the rows only, never the target. Returns the chosen slots."""
    chosen, explored, _ = head.choose(rows)
    outcome = mode_outcomes(chosen, target, outcomes_at_1)
    window.add(chosen, explored, 1 - np.asarray(target), outcome)
    head.update(rows, chosen, outcome, steps)
    return chosen


def part_a_set(split: dict, data_seed: int, fold: int, update: int, data_noise: np.ndarray):
    """The new row set of C2 for this update, standardized with the training rows of the fold as in C2,
    and its targets. Control K3 checks its rows before the signal is added."""
    count = len(split["train_rows"])
    check_k3(new_row_noise(count, data_seed, fold, update), data_noise,
             "part A, data set %d, fold %d, update %d" % (data_seed, fold, update))
    new_set = new_row_set(count, "random", data_seed, fold, update)
    _, rows = standardize(split["raw_train_rows"], new_set["rows"])
    return rows, new_set["target"]


def part_b_rows(data_seed: int, repeat: int) -> dict:
    """The 60 rows of the episodes of part B and their targets, not standardized, as in the live mode.
    Control K3 checks them against the data set of 40 rows before the signal is added."""
    seed = part_b_row_seed(data_seed, repeat)
    check_k3(built_noise(MODE_B_EPISODES, seed), built_noise(MODE_B_SCORE_ROWS, data_seed),
             "part B, data set %d, repeat %d" % (data_seed, repeat))
    built = built_case(MODE_B_EPISODES, SIGN_SIGNAL_SHARE, "random", seed)
    return {"rows": built["rows"], "target": built["target"]}


def mode_score_set(data_seed: int) -> dict:
    """N3: part B and the arm of O2 are scored on all 40 rows of the data set of 40 rows, as they are."""
    built = built_case(MODE_B_SCORE_ROWS, SIGN_SIGNAL_SHARE, "random", data_seed)
    return {"rows": built["rows"], "target": built["target"]}


def mode_checkpoint_row(training: dict, done: int, head: ChoiceHead, scoring: dict, window: ChoiceWindow) -> dict:
    """The row of one checkpoint: the score on the score rows, and the measures of the window."""
    score = auc(head.preferences(scoring["rows"]), scoring["target"])
    return dict(training, checkpoint=done, score=score, **window.close())


def part_a_training(case: dict, initial_seed: int, fold: int, updates: int = UPDATES,
                    checkpoints=MODE_A_CHECKPOINTS) -> List[dict]:
    """One training of part A. At each update the new row set of C2, the choices, the outcomes and one
    update on the set. The score uses the test rows of the fold, standardized as in C2."""
    data_seed = case["data_seed"]
    split = fold_split(built_case(case["rows"], SIGN_SIGNAL_SHARE, "random", data_seed), fold)
    data_noise = built_noise(case["rows"], data_seed)
    head = ChoiceHead(mode_head_seed(data_seed, initial_seed), exploration_seed(initial_seed, data_seed, fold))
    training, window, rows = dict(case, initial_seed=initial_seed, fold=fold), ChoiceWindow(), []
    scoring = {"rows": split["test_rows"], "target": split["test_target"]}
    for update in range(1, updates + 1):
        if update - 1 in checkpoints:
            rows.append(mode_checkpoint_row(training, update - 1, head, scoring, window))
        new_rows, target = part_a_set(split, data_seed, fold, update, data_noise)
        learn_from_set(head, new_rows, target, case["pass"] == "second", window, steps=1)
    if updates in checkpoints:
        rows.append(mode_checkpoint_row(training, updates, head, scoring, window))
    return rows


def choose_one_episode(head: ChoiceHead, row, target, case: dict, window: ChoiceWindow, store: list) -> None:
    """The choice of one episode and its updates: 5 on its row (part B), or one on all rows so far (O2)."""
    outcomes_at_1 = case["pass"] == "second"
    if case["part"] != "O2":
        learn_from_set(head, row, target, outcomes_at_1, window, steps=UPDATES_PER_CHOICE)
        return
    chosen, explored, _ = head.choose(row)
    outcome = mode_outcomes(chosen, target, outcomes_at_1)
    window.add(chosen, explored, 1 - np.asarray(target), outcome)
    store.append((row, chosen, outcome))
    head.update(np.concatenate([r for r, _, _ in store]), np.concatenate([c for _, c, _ in store]),
                np.concatenate([o for _, _, o in store]), steps=1)


def part_b_training(case: dict, initial_seed: int, repeat: int, episodes: int = MODE_B_EPISODES,
                    checkpoints=MODE_B_CHECKPOINTS, checkpoint_row=mode_checkpoint_row) -> List[dict]:
    """One training of part B (the live rule) or of the arm of O2, one row for each episode. The score
    uses the 40 rows of the data set of 40 rows. The learning rule of the head is case["rule"], and
    `chosen` when the case has none (check 2). Check 2b passes rule_checkpoint_row."""
    data_seed = case["data_seed"]
    episode_set, scoring = part_b_rows(data_seed, repeat), mode_score_set(data_seed)
    head = ChoiceHead(mode_head_seed(data_seed, initial_seed), exploration_seed(initial_seed, data_seed, repeat),
                      rule=case.get("rule", DEFAULT_RULE))
    training, window, rows, store = dict(case, initial_seed=initial_seed, fold=repeat), ChoiceWindow(), [], []
    for episode in range(episodes):
        if episode in checkpoints:
            rows.append(checkpoint_row(training, episode, head, scoring, window))
        row, target = episode_set["rows"][episode:episode + 1], episode_set["target"][episode:episode + 1]
        choose_one_episode(head, row, target, case, window, store)
    if episodes in checkpoints:
        rows.append(checkpoint_row(training, episodes, head, scoring, window))
    return rows


def mode_mean_rows(case: dict, training_rows: List[dict], measures=MODE_MEASURES) -> List[dict]:
    """One row for each checkpoint of a case: the means over the initial seeds and the folds."""
    mean_rows = []
    for checkpoint in sorted({row["checkpoint"] for row in training_rows}):
        members = [row for row in training_rows if row["checkpoint"] == checkpoint]
        mean_rows.append(dict(case, checkpoint=checkpoint,
                              **{name: mean_or_blank([row[name] for row in members]) for name in measures}))
    return mean_rows


def mode_case_rows(case: dict, length: int = None, checkpoints=None) -> Tuple[List[dict], List[dict]]:
    """The 15 trainings of one case (3 initial seeds x 5 folds or repeats), and its mean rows. `length` is
    the count of updates of part A or of episodes of part B."""
    checkpoints = mode_checkpoints(case["part"]) if checkpoints is None else checkpoints
    training_rows = []
    for seed, fold in itertools.product(INITIAL_SEEDS, range(OUTER_FOLDS)):
        if case["part"] == "A":
            training_rows += part_a_training(case, seed, fold, length or UPDATES, checkpoints)
        else:
            training_rows += part_b_training(case, seed, fold, length or MODE_B_EPISODES, checkpoints)
    return training_rows, mode_mean_rows(case, training_rows)


def run_mode_check(out_dir: str, cases: List[dict] = None) -> None:
    """The job of check 2. The rows of each case are written when the case is complete, so the row counts
    show the progress. Controls K3 and K5 stop the job on a mismatch."""
    cases = mode_cases() if cases is None else cases
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mean_rows.csv"), "w", newline="", encoding="utf-8") as mean_file, \
            open(os.path.join(out_dir, "training_rows.csv"), "w", newline="", encoding="utf-8") as training_file:
        mean_writer = csv.DictWriter(mean_file, fieldnames=MODE_MEAN_COLUMNS)
        training_writer = csv.DictWriter(training_file, fieldnames=MODE_TRAINING_COLUMNS)
        mean_writer.writeheader()
        training_writer.writeheader()
        for case in cases:
            training_rows, mean_rows = mode_case_rows(case)
            training_writer.writerows(training_rows)
            mean_writer.writerows(mean_rows)
            mean_file.flush()
            training_file.flush()


def mode_timing() -> Dict[str, float]:
    """The mean seconds of 5 trainings of each part, after one short training that is not timed, and the
    estimate of the job from them. It writes no file."""
    part_a_training(mode_cases()[0], 0, 0, updates=5, checkpoints=(0, 5))
    timed = {"A": mode_cases()[10], "B": mode_cases()[40], "O2": mode_cases()[60]}
    trainings = {part: sum(case["part"] == part for case in mode_cases()) * len(INITIAL_SEEDS) * OUTER_FOLDS
                 for part in MODE_PARTS}
    seconds = {}
    for part, case in timed.items():
        start = time.perf_counter()
        for fold in range(OUTER_FOLDS):
            (part_a_training if part == "A" else part_b_training)(case, 0, fold)
        seconds[part] = (time.perf_counter() - start) / OUTER_FOLDS
    estimate = sum(seconds[part] * trainings[part] for part in MODE_PARTS) / 60.0
    return {"seconds_for_one_training": seconds, "trainings": trainings, "estimate_minutes": estimate}


# The label code of check 2. It reads mean_rows.csv and training_rows.csv only, and trains no head.

def parse_mode_cell(name: str, text: str):
    """A blank cell is None."""
    if name in MODE_TEXT_COLUMNS:
        return text
    if text == "":
        return None
    return int(text) if name in MODE_INTEGER_COLUMNS else float(text)


def read_mode_rows(path: str) -> List[dict]:
    """The rows of mean_rows.csv or of training_rows.csv of check 2, with numbers in place of text."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [{name: parse_mode_cell(name, text) for name, text in row.items()} for row in csv.DictReader(handle)]


def check_mode_complete(mean_rows: List[dict], training_rows: List[dict]) -> None:
    """Control K1: the check is complete only with each mean row and each training row."""
    expected = mode_expected_counts()
    if (len(mean_rows), len(training_rows)) != expected:
        raise ControlMismatch("K1: %d mean rows and %d training rows, not %d and %d. The check is not complete"
                              % (len(mean_rows), len(training_rows), expected[0], expected[1]))


def redrawn_explored(seed_sequence, choices_of_windows) -> List[int]:
    """N2: the count of exploration draws of each window, drawn again from a new generator with the calls
    of a choice in their order, rng.random() and then rng.integers(0, 2)."""
    rng, counts = np.random.default_rng(seed_sequence), []
    for choices in choices_of_windows:
        explored = 0
        for _ in range(int(choices)):
            explored += int(rng.random() < EXPLORATION_SHARE)
            rng.integers(0, 2)
        counts.append(explored)
    return counts


def check_exploration_counts(training_rows: List[dict]) -> dict:
    """Control K2: in each training row the recorded count of exploration draws equals the count drawn
    again. Also the share of exploration draws of each part and pass."""
    trainings, totals, mismatches = {}, {}, 0
    for row in training_rows:
        key = (row["part"], row["pass"], row["rows"], row["data_seed"], row["initial_seed"], row["fold"])
        trainings.setdefault(key, []).append(row)
    for (part, run_pass, _, data_seed, seed, fold), members in trainings.items():
        members = sorted(members, key=lambda row: row["checkpoint"])
        counts = redrawn_explored(exploration_seed(seed, data_seed, fold), [row["count_choices"] for row in members])
        mismatches += sum(row["count_explored"] != count for row, count in zip(members, counts))
        total = totals.setdefault("%s %s" % (part, run_pass), [0, 0])
        total[0] += sum(row["count_explored"] for row in members)
        total[1] += sum(row["count_choices"] for row in members)
    shares = {name: (explored / choices if choices else None) for name, (explored, choices) in totals.items()}
    return {"mismatches": int(mismatches), "trainings": len(trainings), "share_of_exploration_draws": shares}


class ModeTable:
    """The mean rows of check 2, by part, pass, row count and checkpoint."""

    def __init__(self, mean_rows: List[dict]):
        self.rows_of = {}
        for row in mean_rows:
            key = (row["part"], row["pass"], row["rows"], row["checkpoint"])
            self.rows_of.setdefault(key, {})[row["data_seed"]] = row

    def numbers(self, part: str, run_pass: str, rows: int, checkpoint: int, measure: str = "score") -> np.ndarray:
        """The 10 numbers of the data sets in the order of the data seeds. A missing or blank one is an error."""
        members = self.rows_of.get((part, run_pass, rows, checkpoint), {})
        where = "%s, %s pass, %d rows, checkpoint %d, %s" % (part, run_pass, rows, checkpoint, measure)
        if sorted(members) != list(BUILT_DATA_SEEDS):
            raise ControlMismatch("%s: the rows do not have each of the 10 data sets" % where)
        numbers = [members[data_seed][measure] for data_seed in BUILT_DATA_SEEDS]
        if any(number is None for number in numbers):
            raise ControlMismatch("%s: a number is blank" % where)
        return np.array(numbers, dtype=np.float64)

    def mean_or_none(self, part: str, run_pass: str, rows: int, checkpoint: int, measure: str):
        """The mean over the data sets, or None when the measure is blank, as at checkpoint 0."""
        members = self.rows_of.get((part, run_pass, rows, checkpoint), {})
        numbers = [members[data_seed][measure] for data_seed in BUILT_DATA_SEEDS if data_seed in members]
        return None if len(numbers) != len(BUILT_DATA_SEEDS) or None in numbers else float(np.mean(numbers))


def paired_reading(table: ModeTable, part: str, rows: int, checkpoint: int) -> dict:
    """N1: the score of the first pass, of the second pass, the paired difference D for each data set, its
    mean and the limit of change 19 over the 10 data sets."""
    first, second = table.numbers(part, "first", rows, checkpoint), table.numbers(part, "second", rows, checkpoint)
    differences = first - second
    limit = limit_of_change_19(differences)
    return {"score": float(first.mean()), "second_pass": float(second.mean()), "d": float(differences.mean()),
            "d_values": differences.tolist(), "d_standard_error": limit["standard_error"], "d_limit": limit["limit"],
            "high": bool(first.mean() >= HIGH_SCORE), "d_at_limit": bool(differences.mean() >= limit["limit"])}


def label_of_part_a(by_rows: Dict[int, dict]) -> str:
    """N1: HOLDS needs a score of 0.80 or more and D at the limit or above at both row counts. FAILS is a
    score below 0.80 at both. Each other case, also a high score with D below the limit, is NOT DECIDED."""
    if all(reading["high"] and reading["d_at_limit"] for reading in by_rows.values()):
        return "HOLDS"
    if not any(reading["high"] for reading in by_rows.values()):
        return "FAILS"
    return "NOT DECIDED"


def label_of_part_b(after: Dict[int, dict]) -> str:
    """N11, with the one score of N3. `after` holds the readings after 40 and after 60 episodes."""
    forty, sixty = after[40], after[60]
    if forty["high"] and sixty["high"] and forty["d_at_limit"] and sixty["d_at_limit"]:
        return "HOLDS"
    if not forty["high"] and sixty["high"] and sixty["d_at_limit"]:
        return "PARTLY"
    if not forty["high"] and not sixty["high"]:
        return "FAILS"
    return "NOT DECIDED"


def mode_series() -> List[Tuple[str, str, int]]:
    """Each part, pass and row count of the mean rows, in the order of the cases."""
    return list(dict.fromkeys((case["part"], case["pass"], case["rows"]) for case in mode_cases()))


def mode_curves(table: ModeTable) -> dict:
    """Reported only: each measure at each checkpoint, as the mean over the data sets, and the first
    checkpoint at which the score reaches 0.80."""
    curves = {}
    for part, run_pass, rows in mode_series():
        points = {str(checkpoint): {name: table.mean_or_none(part, run_pass, rows, checkpoint, name)
                                    for name in MODE_MEASURES} for checkpoint in mode_checkpoints(part)}
        first_high = next((int(c) for c, point in points.items() if point["score"] >= HIGH_SCORE), None)
        curves["%s %s %d" % (part, run_pass, rows)] = {"checkpoints": points, "first_checkpoint_at_0.80": first_high}
    return curves


def mode_labels_of(mean_rows: List[dict]) -> dict:
    """The labels of parts A and B, the label of the arm of O2 (reported only), the row of "What each result
    means", the readings and the curves."""
    table = ModeTable(mean_rows)
    readings = {"A": {rows: paired_reading(table, "A", rows, UPDATES) for rows in BUILT_ROWS},
                "B": {n: paired_reading(table, "B", MODE_B_SCORE_ROWS, n) for n in MODE_B_LABEL_EPISODES},
                "O2": {n: paired_reading(table, "O2", MODE_B_SCORE_ROWS, n) for n in MODE_B_LABEL_EPISODES}}
    labels = {"A": label_of_part_a(readings["A"]), "B": label_of_part_b(readings["B"]),
              "O2": label_of_part_b(readings["O2"])}
    return dict(labels, next_step=MODE_NEXT_STEP.get((labels["A"], labels["B"]), MODE_OTHER_CASE),
                readings=readings, curves=mode_curves(table))


def mode_report(mean_rows: List[dict], training_rows: List[dict]) -> dict:
    """The report of check 2. The verdict is VOID when control K2 fails."""
    k2 = check_exploration_counts(training_rows)
    report = mode_labels_of(mean_rows)
    return dict({"verdict": "VOID" if k2["mismatches"] else "no control failed", "K2": k2}, **report)


def run_mode_labels(out_dir: str) -> dict:
    """The report of a complete check 2, from its two files. It is written to labels.json there."""
    mean_rows = read_mode_rows(os.path.join(out_dir, "mean_rows.csv"))
    training_rows = read_mode_rows(os.path.join(out_dir, "training_rows.csv"))
    check_mode_complete(mean_rows, training_rows)
    report = mode_report(mean_rows, training_rows)
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


# Check 2b of step P2 of design P (docs/decisions/2026_11_22_workspace_choice_mode.md, sections "Owner
# decision before work 4, and the rules of check 2b" and "Changes after the independent review of check
# 2b"). Three learning rules in the setting of part B of check 2. Built data only.

RULE_CASE_COLUMNS = ("rule", "part", "pass", "rows", "data_seed")
RULE_MEASURES = MODE_MEASURES + ("greedy_food",)
RULE_MEAN_COLUMNS = RULE_CASE_COLUMNS + ("checkpoint",) + RULE_MEASURES
RULE_TRAINING_COLUMNS = RULE_CASE_COLUMNS + ("initial_seed", "fold", "checkpoint") + RULE_MEASURES
RULE_TEXT_COLUMNS = ("rule", "part", "pass")
CANDIDATE_RULES = ("both", "hopeful")
RULE_TIE_EPISODES = 60                          # R1: the tie rule reads greedy_food after 60 episodes
K6_MEASURES = MODE_MEASURES                     # K6 with R4: the score and each measure of the window
CHECK_2_TRAINING_ROWS = os.path.join("docs", "results", "data", "workspace_choice_mode_2026_11",
                                     "training_rows.csv")
STEP_P1_COUNT_TABLE = os.path.join("docs", "results", "data", "workspace_choice_head_2026_11", "count_table.json")


def greedy_food(outputs: np.ndarray, target: np.ndarray) -> float:
    """S2: the share of the rows on which the larger output (slot 0 when the two are equal) is the slot of
    the food. Target 1 means that the food is in slot 0."""
    outputs = np.asarray(outputs)
    greedy_slot = np.where(outputs[:, 0] >= outputs[:, 1], 0, 1)
    return float(np.mean(greedy_slot == 1 - np.asarray(target)))


def rule_checkpoint_row(training: dict, done: int, head: ChoiceHead, scoring: dict, window: ChoiceWindow) -> dict:
    """The row of one checkpoint of check 2b: the row of check 2 and greedy_food on the score rows."""
    row = mode_checkpoint_row(training, done, head, scoring, window)
    row["greedy_food"] = greedy_food(head.outputs(scoring["rows"]), scoring["target"])
    return row


def rule_cases() -> List[dict]:
    """Each rule in each pass and data set, in the setting of part B."""
    return [{"rule": rule, "part": "B", "pass": run_pass, "rows": MODE_B_SCORE_ROWS, "data_seed": data_seed}
            for rule, run_pass, data_seed in itertools.product(RULES, MODE_PASSES, BUILT_DATA_SEEDS)]


def rule_expected_counts() -> Tuple[int, int]:
    """Control K1: 420 mean rows and 6300 training rows."""
    mean_count = len(rule_cases()) * len(MODE_B_CHECKPOINTS)
    return mean_count, mean_count * len(INITIAL_SEEDS) * OUTER_FOLDS


def rule_case_rows(case: dict, length: int = None, checkpoints=None) -> Tuple[List[dict], List[dict]]:
    """The 15 trainings of one case of check 2b (3 initial seeds x 5 repeats), and its mean rows."""
    checkpoints = MODE_B_CHECKPOINTS if checkpoints is None else checkpoints
    training_rows = []
    for seed, repeat in itertools.product(INITIAL_SEEDS, range(OUTER_FOLDS)):
        training_rows += part_b_training(case, seed, repeat, length or MODE_B_EPISODES, checkpoints,
                                         checkpoint_row=rule_checkpoint_row)
    return training_rows, mode_mean_rows(case, training_rows, RULE_MEASURES)


def run_rule_check(out_dir: str, cases: List[dict] = None) -> None:
    """The job of check 2b. The rows of each case are written when the case is complete, so the row counts
    show the progress. Controls K3 and K5 stop the job on a mismatch."""
    cases = rule_cases() if cases is None else cases
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mean_rows.csv"), "w", newline="", encoding="utf-8") as mean_file, \
            open(os.path.join(out_dir, "training_rows.csv"), "w", newline="", encoding="utf-8") as training_file:
        mean_writer = csv.DictWriter(mean_file, fieldnames=RULE_MEAN_COLUMNS)
        training_writer = csv.DictWriter(training_file, fieldnames=RULE_TRAINING_COLUMNS)
        mean_writer.writeheader()
        training_writer.writeheader()
        for case in cases:
            training_rows, mean_rows = rule_case_rows(case)
            training_writer.writerows(training_rows)
            mean_writer.writerows(mean_rows)
            mean_file.flush()
            training_file.flush()


def rule_timing() -> Dict[str, float]:
    """The mean seconds of 5 trainings of each rule (first pass, data set 0, initial seed 0, repeats 0 to
    4), after one short training that is not timed, and the estimate of the job. It writes no file."""
    first_cases = {case["rule"]: case for case in rule_cases() if case["pass"] == "first" and case["data_seed"] == 0}
    part_b_training(first_cases[DEFAULT_RULE], 0, 0, episodes=5, checkpoints=(0, 5), checkpoint_row=rule_checkpoint_row)
    trainings = len(rule_cases()) // len(RULES) * len(INITIAL_SEEDS) * OUTER_FOLDS
    seconds = {}
    for rule, case in first_cases.items():
        start = time.perf_counter()
        for repeat in range(OUTER_FOLDS):
            part_b_training(case, 0, repeat, checkpoint_row=rule_checkpoint_row)
        seconds[rule] = (time.perf_counter() - start) / OUTER_FOLDS
    estimate = sum(seconds[rule] * trainings for rule in RULES) / 60.0
    return {"seconds_for_one_training": seconds, "trainings_of_each_rule": trainings, "estimate_minutes": estimate,
            "expected_rows": dict(zip(("mean_rows", "training_rows"), rule_expected_counts()))}


# The label code of check 2b. It reads the two files of check 2b, the committed training rows of check 2
# (K6) and the committed count table of step P1 (reported only). It trains no head.

def parse_rule_cell(name: str, text: str):
    """A blank cell is None."""
    if name in RULE_TEXT_COLUMNS:
        return text
    if text == "":
        return None
    return int(text) if name in MODE_INTEGER_COLUMNS else float(text)


def read_rule_rows(path: str) -> List[dict]:
    """The rows of mean_rows.csv or of training_rows.csv of check 2b, with numbers in place of text."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [{name: parse_rule_cell(name, text) for name, text in row.items()} for row in csv.DictReader(handle)]


def check_rule_complete(mean_rows: List[dict], training_rows: List[dict]) -> None:
    """Control K1: the check is complete only with 420 mean rows and 6300 training rows."""
    expected = rule_expected_counts()
    if (len(mean_rows), len(training_rows)) != expected:
        raise ControlMismatch("K1: %d mean rows and %d training rows, not %d and %d. The check is not complete"
                              % (len(mean_rows), len(training_rows), expected[0], expected[1]))


def rule_k2(training_rows: List[dict]) -> dict:
    """Control K2 (N2) for each rule apart: in each training row the recorded count of exploration draws
    equals the count drawn again. Also the share of exploration draws of each rule and pass."""
    mismatches, trainings, shares = 0, 0, {}
    for rule in RULES:
        reading = check_exploration_counts([row for row in training_rows if row["rule"] == rule])
        mismatches, trainings = mismatches + reading["mismatches"], trainings + reading["trainings"]
        shares.update({"%s %s" % (rule, name): share for name, share in reading["share_of_exploration_draws"].items()})
    return {"mismatches": mismatches, "trainings": trainings, "share_of_exploration_draws": shares}


def same_cell(first, second) -> bool:
    """Two cells are the same number, or both blank. A blank is "" before a write and None after a read."""
    first, second = (None if cell == "" else cell for cell in (first, second))
    return first == second


def rule_k6(training_rows: List[dict], check_2_rows: List[dict]) -> Dict[str, int]:
    """Control K6 with R4: each training row of `chosen` equals the training row of part B of check 2 with
    the same pass, data set, initial seed, repeat and checkpoint, exactly, in the score and in each measure
    of the window. Returns the count of rows compared, of rows that differ, and of rows with no partner."""
    committed = {(row["pass"], row["data_seed"], row["initial_seed"], row["fold"], row["checkpoint"]): row
                 for row in check_2_rows if row["part"] == "B"}
    reading = {"compared": 0, "mismatches": 0, "missing": 0}
    for row in training_rows:
        if row["rule"] != DEFAULT_RULE:
            continue
        partner = committed.get((row["pass"], row["data_seed"], row["initial_seed"], row["fold"], row["checkpoint"]))
        if partner is None:
            reading["missing"] += 1
            continue
        reading["compared"] += 1
        reading["mismatches"] += int(not all(same_cell(row[name], partner[name]) for name in K6_MEASURES))
    return reading


def k6_expected_rows() -> int:
    """The training rows of `chosen` in a complete check 2b, each with a partner in check 2."""
    return rule_expected_counts()[1] // len(RULES)


class RuleTable:
    """The mean rows of check 2b, by rule, pass and checkpoint."""

    def __init__(self, mean_rows: List[dict]):
        self.rows_of = {}
        for row in mean_rows:
            self.rows_of.setdefault((row["rule"], row["pass"], row["checkpoint"]), {})[row["data_seed"]] = row

    def numbers(self, rule: str, run_pass: str, checkpoint: int, measure: str) -> np.ndarray:
        """The 10 numbers of the data sets in the order of the data seeds. A missing or blank one is an error."""
        members = self.rows_of.get((rule, run_pass, checkpoint), {})
        where = "%s, %s pass, checkpoint %d, %s" % (rule, run_pass, checkpoint, measure)
        if sorted(members) != list(BUILT_DATA_SEEDS):
            raise ControlMismatch("%s: the rows do not have each of the 10 data sets" % where)
        numbers = [members[data_seed][measure] for data_seed in BUILT_DATA_SEEDS]
        if any(number is None for number in numbers):
            raise ControlMismatch("%s: a number is blank" % where)
        return np.array(numbers, dtype=np.float64)

    def mean_or_none(self, rule: str, run_pass: str, checkpoint: int, measure: str):
        """The mean over the data sets, or None when the measure is blank, as at checkpoint 0."""
        members = self.rows_of.get((rule, run_pass, checkpoint), {})
        numbers = [members[data_seed][measure] for data_seed in BUILT_DATA_SEEDS if data_seed in members]
        return None if len(numbers) != len(BUILT_DATA_SEEDS) or None in numbers else float(np.mean(numbers))


def paired_difference(first: np.ndarray, second: np.ndarray) -> dict:
    """The difference for each data set, first minus second, its mean, and the limit of change 19 of the 10
    differences. `holds` is a mean at the limit or above."""
    differences = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    limit = limit_of_change_19(differences)
    return {"d": float(differences.mean()), "d_values": differences.tolist(),
            "standard_error": limit["standard_error"], "limit": limit["limit"],
            "holds": bool(differences.mean() >= limit["limit"])}


def score_condition(table: RuleTable, rule: str, episodes: int) -> dict:
    """Condition 1 at one checkpoint (N11): the score is 0.80 or more and D of the score is at the limit."""
    first = table.numbers(rule, "first", episodes, "score")
    reading = paired_difference(first, table.numbers(rule, "second", episodes, "score"))
    reading.update(score=float(first.mean()), d_at_limit=reading["holds"])
    reading["holds"] = bool(first.mean() >= HIGH_SCORE and reading["d_at_limit"])
    return reading


def rule_conditions(table: RuleTable, rule: str) -> dict:
    """The three conditions of a rule `both` or `hopeful`, each with its numbers, and whether it passes.
    Condition 2 uses the paired differences, rule minus `chosen` (O3)."""
    conditions = {}
    for episodes in MODE_B_LABEL_EPISODES:
        conditions["score_%d" % episodes] = score_condition(table, rule, episodes)
        conditions["greedy_food_%d" % episodes] = paired_difference(
            table.numbers(rule, "first", episodes, "greedy_food"),
            table.numbers(DEFAULT_RULE, "first", episodes, "greedy_food"))
    conditions["greedy_food_d_60"] = paired_difference(table.numbers(rule, "first", 60, "greedy_food"),
                                                       table.numbers(rule, "second", 60, "greedy_food"))
    conditions["passes"] = all(reading["holds"] for reading in conditions.values())
    return conditions


def choose_rule(table: RuleTable, mismatches: int) -> dict:
    """The rule that chooses the learning rule of work 4, with R1. A control that fails gives VOID and no
    rule. The tie rule takes `hopeful` when the mean difference of greedy_food after 60 episodes, `both`
    minus `hopeful`, is below the limit in absolute value, else the rule with the higher mean."""
    if mismatches:
        return {"result": "VOID", "rule": None}
    conditions = {rule: rule_conditions(table, rule) for rule in CANDIDATE_RULES}
    passing = [rule for rule in CANDIDATE_RULES if conditions[rule]["passes"]]
    if not passing:
        return {"result": "no rule passes", "rule": DEFAULT_RULE, "conditions": conditions}
    if len(passing) == 1:
        return {"result": "one rule passes", "rule": passing[0], "conditions": conditions}
    tie = paired_difference(table.numbers("both", "first", RULE_TIE_EPISODES, "greedy_food"),
                            table.numbers("hopeful", "first", RULE_TIE_EPISODES, "greedy_food"))
    within = abs(tie["d"]) < tie["limit"]
    rule = "hopeful" if within or tie["d"] < 0 else "both"
    return {"result": "both rules pass", "rule": rule, "tie": dict(tie, within_the_limit=within),
            "conditions": conditions}


def second_pass_report(table: RuleTable) -> dict:
    """R6 and O2, reported only: the mean second pass greedy_food of each rule after 40 and 60 episodes, and
    0.5 minus it with the limit of change 19 of the 10 numbers. `both` is marked when 0.5 minus its mean is
    at the limit or above."""
    report = {}
    for rule, episodes in itertools.product(RULES, MODE_B_LABEL_EPISODES):
        numbers = table.numbers(rule, "second", episodes, "greedy_food")
        below = paired_difference(np.full(len(numbers), HALF), numbers)
        reading = {"mean": float(numbers.mean()), "half_minus_mean": below["d"], "limit": below["limit"]}
        if rule == "both":
            reading["marked"] = below["holds"]
        report["%s %d" % (rule, episodes)] = reading
    return report


def rule_curves(table: RuleTable) -> dict:
    """Reported only: each measure at each checkpoint, as the mean over the data sets."""
    return {"%s %s" % (rule, run_pass): {str(checkpoint): {name: table.mean_or_none(rule, run_pass, checkpoint, name)
                                                           for name in RULE_MEASURES}
                                         for checkpoint in MODE_B_CHECKPOINTS}
            for rule, run_pass in itertools.product(RULES, MODE_PASSES)}


def gate_shares(count_table_path: str) -> dict:
    """Reported only, with no part in the choice: the share of the episodes in which the value gate took
    the slot of the food in each unit of step P1, read from its committed count table. The lowest share of
    the counted units, and the shares of the units after the food switch (O5)."""
    with open(count_table_path, encoding="utf-8") as handle:
        units = json.load(handle)
    counted = {name for name, _, _, _, _, is_counted in UNITS if is_counted}
    shares = {name: units[name]["share_food_slot_chosen"] for name, *_ in UNITS}
    return {"source": count_table_path.replace(os.sep, "/"),
            "lowest_of_the_counted_units": min(shares[name] for name in counted),
            "units_after_the_food_switch": {name: share for name, share in shares.items() if name not in counted}}


def rule_report(mean_rows: List[dict], training_rows: List[dict], check_2_rows: List[dict],
                count_table_path: str = STEP_P1_COUNT_TABLE) -> dict:
    """The report of check 2b. The verdict is VOID when control K2 or K6 fails."""
    k2, k6 = rule_k2(training_rows), rule_k6(training_rows, check_2_rows)
    k6["expected"] = k6_expected_rows()
    k6_fails = k6["mismatches"] or k6["missing"] or k6["compared"] != k6["expected"]
    table = RuleTable(mean_rows)
    choice = choose_rule(table, int(bool(k2["mismatches"] or k6_fails)))
    return {"verdict": "VOID" if choice["result"] == "VOID" else "no control failed", "K2": k2, "K6": k6,
            "choice": choice, "second_pass": second_pass_report(table), "curves": rule_curves(table),
            "value_gate_of_step_p1": gate_shares(count_table_path)}


def run_rule_labels(out_dir: str, check_2_rows_path: str = CHECK_2_TRAINING_ROWS) -> dict:
    """The report of a complete check 2b, from its two files and the committed rows of check 2. It is
    written to labels.json there."""
    mean_rows = read_rule_rows(os.path.join(out_dir, "mean_rows.csv"))
    training_rows = read_rule_rows(os.path.join(out_dir, "training_rows.csv"))
    check_rule_complete(mean_rows, training_rows)
    report = rule_report(mean_rows, training_rows, read_mode_rows(check_2_rows_path))
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--built-test",
                      help="Run the test on built data and write one row for each case and head to this "
                           "CSV file. It reads no saved run.")
    mode.add_argument("--count-table", metavar="RUNS_DIR",
                      help="Make the count table of the recorded feedback from value_gate.json of each run "
                           "in this folder. It reads no saved vector and trains no head. Needs --out.")
    mode.add_argument("--variant-sign", metavar="OUT_DIR",
                      help="Run the check on the variant and write mean_rows.csv and training_rows.csv to "
                           "this folder. Built data only. Needs --step-p1-csv. Expected: 3600 mean rows "
                           "and 54000 rows for each training and checkpoint.")
    mode.add_argument("--variant-sign-first-step", action="store_true",
                      help="Compute controls V6, V7 and V9 of the check on the variant for each case, with "
                           "no training, and print the counts.")
    mode.add_argument("--variant-sign-timing", action="store_true",
                      help="Time one training of each condition of the check on the variant with each "
                           "measurement active, and print an estimate for the full job. It writes no file.")
    mode.add_argument("--variant-sign-labels", metavar="OUT_DIR",
                      help="Compute each prediction and each label of the check on the variant from "
                           "mean_rows.csv and training_rows.csv of this folder, and write labels.json there. "
                           "It reads nothing else and trains no head. The job must be complete.")
    mode.add_argument("--choice-mode-timing", action="store_true",
                      help="Time 5 trainings of each part of check 2 of step P2 and print an estimate for the "
                           "job. It writes no file.")
    mode.add_argument("--choice-mode-check", metavar="OUT_DIR",
                      help="Run check 2 of step P2 on built data and write mean_rows.csv and training_rows.csv "
                           "to this folder. Expected: 760 mean rows and 11400 training rows.")
    mode.add_argument("--choice-mode-labels", metavar="OUT_DIR",
                      help="Compute the controls K1 and K2 and each label of check 2 of step P2 from "
                           "mean_rows.csv and training_rows.csv of this folder, and write labels.json there. "
                           "It reads nothing else and trains no head. The check must be complete.")
    mode.add_argument("--choice-rule-timing", action="store_true",
                      help="Time 5 trainings of each learning rule of check 2b of step P2 and print an estimate "
                           "for the job. It writes no file.")
    mode.add_argument("--choice-rule-check", metavar="OUT_DIR",
                      help="Run check 2b of step P2 on built data (the rules chosen, both and hopeful in the "
                           "setting of part B) and write mean_rows.csv and training_rows.csv to this folder. "
                           "Expected: 420 mean rows and 6300 training rows.")
    mode.add_argument("--choice-rule-labels", metavar="OUT_DIR",
                      help="Compute the controls K1, K2 and K6, the choosing rule and the reports of check 2b "
                           "from mean_rows.csv and training_rows.csv of this folder, the committed rows of check "
                           "2 and the committed count table of step P1, and write labels.json there. It trains "
                           "no head. The check must be complete.")
    parser.add_argument("--check-2-rows", default=CHECK_2_TRAINING_ROWS,
                        help="training_rows.csv of check 2, for control K6 of check 2b.")
    parser.add_argument("--out", help="The JSON file of the count table.")
    parser.add_argument("--step-p1-csv",
                        help="built_test.csv of step P1, for controls V1 and V2 of the check on the variant.")
    args = parser.parse_args()
    if args.count_table:
        if not args.out:
            parser.error("--count-table needs --out")
        print(json.dumps(run_count_table(args.count_table, args.out), indent=2))
        return
    if args.variant_sign_labels:
        print(json.dumps(run_labels(args.variant_sign_labels), indent=2))
        return
    if args.choice_mode_labels:
        print(json.dumps(run_mode_labels(args.choice_mode_labels), indent=2))
        return
    if args.choice_rule_labels:
        print(json.dumps(run_rule_labels(args.choice_rule_labels, args.check_2_rows), indent=2))
        return
    torch.set_num_threads(1)
    if args.choice_rule_timing:
        print(json.dumps(rule_timing(), indent=2))
    elif args.choice_rule_check:
        run_rule_check(args.choice_rule_check)
    elif args.choice_mode_timing:
        print(json.dumps(mode_timing(), indent=2))
    elif args.choice_mode_check:
        run_mode_check(args.choice_mode_check)
    elif args.variant_sign_first_step:
        print(json.dumps(first_step(variant_sign_cases()), indent=2))
    elif args.variant_sign_timing:
        print(json.dumps(time_trainings(), indent=2))
    elif args.variant_sign:
        if not args.step_p1_csv:
            parser.error("--variant-sign needs --step-p1-csv")
        run_variant_sign(args.variant_sign, args.step_p1_csv)
    else:
        run_built_test(args.built_test)


if __name__ == "__main__":
    main()
