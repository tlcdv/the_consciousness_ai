"""Abstract model of the competing goals gate, written before any run.

`docs/decisions/2026_11_08_competing_goals_gate.md`. No room and no agent. The model uses the value
gate's own choice and update rules (`models/core/value_gate.py`), two food objects and the travel
times measured in earlier runs of the room.

One episode:
- In the first FORCED episodes only one object is present, the light and the decoy in turn, so the gate
  has one candidate. Bumble bees in a delay test were first trained on each flower alone (Baciadonna et
  al., Sci. Rep. 15, 24363, 2025). Without these episodes the first object that gives food takes over.
  The gate updates only the chosen value, both values start at 0, and a value of 0.5 outweighs the
  largest salience difference. The first version of this model had no forced episodes, and no design
  passed.
- After that the gate chooses the light or the decoy by softmax over salience + VALUE_WEIGHT x V at
  TEMPERATURE. The louder tone has salience 1.0. The quieter tone's salience is drawn from the quieter
  saliences that the `value_gate.json` records of `--runs` hold. Until 2026-11-11 it was uniform in
  0.34 to 0.48 (`docs/decisions/2026_10_16_value_gate.md`), and the records are wider than that. The
  louder tone changes object every episode, as `--dark-room-decoy-level 0.5,3.0` does. The decoy is
  louder in even episodes.
- The chosen object is reached with probability REACH, 115 of 120 in
  `docs/results/rhythm_binding_gate_2026_10.md`. `--reach` sets another rate. After a miss in a free
  episode the agent reaches the OTHER object and gets that object's food, and the gate gives it to the
  value of the chosen tone. Until 2026-11-11 a miss was no contact and no food. In 9 saved runs a miss
  reached the other object in 21 of 21 cases (`scripts/analysis/probe_chosen_object_reach.py`). In a
  forced episode there is no other object, so a miss is no contact and no food.
- The steps from the choice to the first contact are drawn from the `value_gate.json` records of runs
  with no wait (the steps of the episodes with outcome 1). The food comes after `wait` more steps at the
  object that the agent reached, so it comes n = travel + wait steps after the choice.
- Outcome "binary" is the present gate. It is 1 when the food comes within OUTCOME_STEPS, else 0.
  Outcome "hyperbolic" is the proposed option. It is size / (1 + k (n - 1)) when the food comes within
  OUTCOME_STEPS, else 0. In monkeys, the dopamine value signal falls hyperbolically with the delay of a
  reward and rises with its size (Kobayashi and Schultz, J. Neurosci. 28, 7837, 2008).
- V of the chosen object moves by LEARNING_RATE x (outcome - V), once per episode.

The sizes are room units. The room's present food is 1.0, an arbitrary unit. With sizes near 1 the
values of two foods differ by less than the salience difference can outweigh, and the second version of
this model failed after the change (decoy first about 0.76). So the search also scales the sizes. The
gate's own constants do not change.

Arms, EPISODES episodes each, with the change at episode SWITCH. The light is the large food with the
wait, the decoy the small food with no wait.
- benefit: before, the light gives `big` after `wait` steps. After, it gives `size` after `wait` steps.
  The light stays the larger food, so a rule that always takes the larger food fails after the change,
  and a rule that always takes the sooner food fails before it.
- binary: the schedule of benefit with the binary outcome. This is the ablation.
- cost: before, the light gives `size` with no wait. After, it gives `size` after `wait` steps.
  Reported only. The second version of this model also required the cost arm, and with four share
  criteria at 3 seeds no design reached a live chance above about 0.66.
- stay: the light gives `big` after `wait` steps in every episode. Reported only.

The live gate that the model judges, at 3 of 3 seeds:
- C1: benefit arm, light first in WINDOW_BEFORE, SHARE or more.
- C2: benefit arm, decoy first in WINDOW_AFTER, SHARE or more.
- C3: the change of decoy first from WINDOW_BEFORE to WINDOW_AFTER is larger in the benefit arm than in
  the binary arm by ABLATION_MARGIN or more. The binary gate cannot see the change of size, so it has
  no reason to change its choice. Its level of choice is not a criterion, because with two foods the
  first object that the binary gate rewards often takes over.
A design passes when the chance that C1 to C3 all hold, from 3 sequences drawn for each arm
(GATE_DRAWS times), is GATE_CHANCE or more, and the mean binary change is BINARY_CHANGE or less in size.

`--search` tries every design of GRID with SEARCH_SEQUENCES sequences. The order is fixed before the
search: the smallest big / decoy, then the smallest size / decoy, then the smallest wait, then the
smallest k, then the smallest decoy. The first passing design is run again with `--sequences`
sequences and a new seed, and it must pass again.

Controls that raise on a mismatch: the model's softmax equals `ValueGate.probabilities`, its update
equals the value that `ValueGate` keeps, its binary outcome equals what `ValueGate.observe_reward`
records, and a constant hyperbolic outcome gives the closed form o (1 - (1 - LEARNING_RATE)^t).

Usage:
    python -m scripts.analysis.model_competing_goals --runs runs/vg_learn_s601 runs/vg_learn_s602 \
        runs/vg_learn_s603 runs/rv_stay_s701 runs/rv_stay_s702 runs/rv_stay_s703 --search
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from models.core.value_gate import LEARNING_RATE, OUTCOME_STEPS, TEMPERATURE, VALUE_WEIGHT, ValueGate

REACH = 115 / 120
EPISODES, SWITCH, FORCED = 60, 30, 6
WINDOW_BEFORE, WINDOW_AFTER = (15, 30), (45, 60)
SHARE = 0.8
BINARY_CHANGE = 0.15
ABLATION_MARGIN = 0.3
GATE_CHANCE = 0.8
GRID = {"k": (0.05, 0.1, 0.2, 0.3), "decoy": (1.0, 2.0, 3.0), "size": (1.5, 2.0, 3.0, 4.0, 6.0),
        "wait": (20, 30, 40, 45), "big": (6.0, 8.0, 12.0, 16.0, 24.0, 32.0)}
SEARCH_SEQUENCES = 4000   # 400 until 2026-11-11; the chance of one design then moved by about 0.04 between seeds
GATE_DRAWS = 20000
LIGHT, DECOY = 0, 1
ARMS = ("cost", "benefit", "binary", "stay")
STRICT = ("benefit_before_light", "benefit_after_decoy")
REPORTED = ("cost_before_light", "cost_after_decoy", "binary_before_light", "binary_after_decoy",
            "stay_after_light")


@dataclass(frozen=True)
class Design:
    k: float
    decoy: float
    size: float
    big: float
    wait: int


@dataclass(frozen=True)
class Measured:
    """What the model takes from the gate records: travel steps and the quieter tone's salience."""
    travel: np.ndarray
    quiet: np.ndarray


def quieter_salience(episode: dict, path: str) -> float:
    """The smaller of the two saliences of one gate record. A record with one candidate raises."""
    salience = list(episode["salience"].values())
    if len(salience) != 2:
        raise ValueError("%s has an episode with %d candidates; the model needs 2" % (path, len(salience)))
    return float(min(salience))


def load_measured(runs: Sequence[str]) -> Tuple[Measured, Dict[str, str]]:
    """Travel steps (episodes with outcome 1), the quieter salience of every episode, and each md5."""
    travel: List[int] = []
    quiet: List[float] = []
    md5: Dict[str, str] = {}
    for run in runs:
        path = os.path.join(run, "value_gate.json")
        with open(path, "rb") as handle:
            raw = handle.read()
        md5[path] = hashlib.md5(raw).hexdigest()
        record = json.loads(raw)
        travel += [int(episode["steps"]) for episode in record["episodes"] if episode["outcome"] == 1.0]
        quiet += [quieter_salience(episode, path) for episode in record["episodes"]]
    if not travel:
        raise ValueError("no episode with food in %s" % list(runs))
    return Measured(np.array(travel), np.array(quiet)), md5


def episode_terms(arm: str, design: Design, episode: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """Food sizes and waits (light, decoy) and the outcome rule of one episode of an arm."""
    late = episode >= SWITCH
    if arm == "cost":
        light = (design.size, design.wait if late else 0)
    elif arm in ("benefit", "binary"):
        light = (design.size if late else design.big, design.wait)
    elif arm == "stay":
        light = (design.big, design.wait)
    else:
        raise ValueError("unknown arm %r" % arm)
    rule = "binary" if arm == "binary" else "hyperbolic"
    return np.array([light[0], design.decoy]), np.array([light[1], 0]), rule


def outcome(rule: str, size: np.ndarray, n: np.ndarray, reached: np.ndarray, k: float) -> np.ndarray:
    """The outcome of each sequence when the food of `size` comes n steps after the choice."""
    within = reached & (n <= OUTCOME_STEPS)
    if rule == "binary":
        return within.astype(float)
    if rule == "hyperbolic":
        return np.where(within, size / (1.0 + k * (n - 1)), 0.0)
    raise ValueError("unknown outcome rule %r" % rule)


def decoy_probability(salience: np.ndarray, v: np.ndarray) -> np.ndarray:
    """P(decoy) per sequence, the softmax of ValueGate.probabilities over the two objects."""
    score = salience + VALUE_WEIGHT * v
    score = (score - score.max(axis=1, keepdims=True)) / TEMPERATURE
    e = np.exp(score)
    return e[:, DECOY] / e.sum(axis=1)


def draw_salience(episode: int, n_seq: int, rng: np.random.Generator, quiet: np.ndarray) -> np.ndarray:
    """Salience (light, decoy): the louder tone 1.0, the quieter one drawn from the measured `quiet`."""
    salience = np.ones((n_seq, 2))
    quieter = LIGHT if episode % 2 == 0 else DECOY
    salience[:, quieter] = rng.choice(quiet, size=n_seq)
    return salience


def update(v: np.ndarray, chosen: np.ndarray, o: np.ndarray) -> None:
    """V of the chosen object moves by LEARNING_RATE x (outcome - V), in place."""
    rows = np.arange(len(chosen))
    v[rows, chosen] += LEARNING_RATE * (o - v[rows, chosen])


def choose(episode: int, v: np.ndarray, rng: np.random.Generator, quiet: np.ndarray) -> np.ndarray:
    """The chosen object per sequence: forced in the first FORCED episodes (light first), then softmax."""
    salience = draw_salience(episode, len(v), rng, quiet)
    if episode < FORCED:
        return np.full(len(v), episode % 2)
    return (rng.random(len(v)) < decoy_probability(salience, v)).astype(int)


def contact_object(chosen: np.ndarray, reached: np.ndarray, episode: int) -> np.ndarray:
    """The object the agent reaches first. A miss gives the other object, or -1 in a forced episode."""
    if episode < FORCED:
        return np.where(reached, chosen, -1)
    return np.where(reached, chosen, 1 - chosen)


def play_episode(arm: str, design: Design, measured: Measured, episode: int, v: np.ndarray,
                 rng: np.random.Generator, reach: float = REACH) -> np.ndarray:
    """One episode for each sequence. Moves the chosen value in `v` and returns the first object."""
    size, wait, rule = episode_terms(arm, design, episode)
    chosen = choose(episode, v, rng, measured.quiet)
    contact = contact_object(chosen, rng.random(len(v)) < reach, episode)
    at = np.maximum(contact, 0)
    n = rng.choice(measured.travel, size=len(v)) + wait[at]
    update(v, chosen, outcome(rule, size[at], n, contact >= 0, design.k))
    return contact


def simulate(arm: str, design: Design, measured: Measured, n_seq: int, rng: np.random.Generator,
             reach: float = REACH) -> np.ndarray:
    """The first object of each episode for each sequence: LIGHT, DECOY, or -1 for no contact."""
    v = np.zeros((n_seq, 2))
    first = np.full((n_seq, EPISODES), -1)
    for episode in range(EPISODES):
        first[:, episode] = play_episode(arm, design, measured, episode, v, rng, reach)
    return first


def window_share(first: np.ndarray, target: int, window: Tuple[int, int]) -> np.ndarray:
    """For each sequence, the share of the window's episodes in which `target` was the first object."""
    return (first[:, window[0]:window[1]] == target).mean(axis=1)


def episodes_to_switch(first: np.ndarray) -> np.ndarray:
    """Episodes after SWITCH until 4 of 5 episodes in a row have the decoy first; -1 if never."""
    decoy = (first[:, SWITCH:] == DECOY).astype(int)
    count = np.full(len(first), -1)
    for start in range(decoy.shape[1] - 4):
        hit = (decoy[:, start:start + 5].sum(axis=1) >= 4) & (count < 0)
        count[hit] = start
    return count


def evaluate(design: Design, measured: Measured, n_seq: int, seed: int,
             reach: float = REACH) -> Dict[str, np.ndarray]:
    """The shares that the criteria read, for every arm, from one generator."""
    rng = np.random.default_rng(seed)
    firsts = {arm: simulate(arm, design, measured, n_seq, rng, reach) for arm in ARMS}
    shares = {}
    for arm in ("cost", "benefit", "binary"):
        shares["%s_before_light" % arm] = window_share(firsts[arm], LIGHT, WINDOW_BEFORE)
        shares["%s_after_decoy" % arm] = window_share(firsts[arm], DECOY, WINDOW_AFTER)
        shares["%s_change" % arm] = shares["%s_after_decoy" % arm] - window_share(firsts[arm], DECOY,
                                                                                 WINDOW_BEFORE)
    shares["stay_after_light"] = window_share(firsts["stay"], LIGHT, WINDOW_AFTER)
    shares["cost_switch_episodes"] = episodes_to_switch(firsts["cost"])
    shares["benefit_switch_episodes"] = episodes_to_switch(firsts["benefit"])
    return shares


def passes(shares: Dict[str, np.ndarray]) -> bool:
    """The live chance of C1 to C3 at GATE_CHANCE or more, and a small mean binary change."""
    if abs(shares["binary_change"].mean()) > BINARY_CHANGE:
        return False
    return gate_pass_chance(shares, np.random.default_rng(0))["all"] >= GATE_CHANCE


def gate_pass_chance(shares: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, float]:
    """The chance of each live criterion at 3 of 3 seeds, from 3 random sequences per arm."""
    n = len(shares["benefit_before_light"])
    pick = {arm: rng.integers(0, n, size=(GATE_DRAWS, 3)) for arm in ("benefit", "binary")}
    chance = {name: float((shares[name][pick[name.split("_")[0]]] >= SHARE).all(axis=1).mean())
              for name in STRICT}
    margin = shares["benefit_change"][pick["benefit"]] - shares["binary_change"][pick["binary"]]
    chance["ablation"] = float((margin >= ABLATION_MARGIN).all(axis=1).mean())
    together = np.ones(GATE_DRAWS, dtype=bool)
    for name in STRICT:
        together &= (shares[name][pick[name.split("_")[0]]] >= SHARE).all(axis=1)
    chance["all"] = float((together & (margin >= ABLATION_MARGIN).all(axis=1)).mean())
    return chance


def grid_designs(travel: np.ndarray) -> List[Design]:
    """Every design of GRID with big > size > decoy and the food within OUTCOME_STEPS."""
    designs = []
    for k, decoy, size, wait, big in itertools.product(*(GRID[name] for name in ("k", "decoy", "size",
                                                                                 "wait", "big"))):
        if big > size > decoy and wait + int(travel.max()) <= OUTCOME_STEPS:
            designs.append(Design(k, decoy, size, big, wait))
    return designs


def search(measured: Measured, seed: int, reach: float = REACH) -> Tuple[List[Design], int]:
    """Every grid design that passes, in the order fixed before the search, and the grid size."""
    designs = grid_designs(measured.travel)
    passing = [d for d in designs if passes(evaluate(d, measured, SEARCH_SEQUENCES, seed, reach))]
    order = sorted(passing, key=lambda d: (d.big / d.decoy, d.size / d.decoy, d.wait, d.k, d.decoy))
    return order, len(designs)


def gate_outcome(n_food: Optional[int]) -> float:
    """The outcome that ValueGate records when the food comes n_food steps after the choice."""
    gate = ValueGate(seed=0)
    gate.choose({LIGHT: 1.0})
    for step in range(1, OUTCOME_STEPS + 1):
        gate.observe_reward(1.0 if step == n_food else -0.01)
        if gate.episodes:
            break
    return gate.episodes[-1]["outcome"]


def check_softmax(rng: np.random.Generator) -> None:
    """Raise unless decoy_probability equals ValueGate.probabilities."""
    gate = ValueGate(seed=0)
    for _ in range(200):
        salience = {LIGHT: float(rng.uniform(0.3, 1.0)), DECOY: float(rng.uniform(0.3, 1.0))}
        gate.values = {LIGHT: float(rng.uniform(-1.0, 3.0)), DECOY: float(rng.uniform(-1.0, 3.0))}
        want = gate.probabilities(salience)[DECOY]
        got = decoy_probability(np.array([[salience[LIGHT], salience[DECOY]]]),
                                np.array([[gate.values[LIGHT], gate.values[DECOY]]]))[0]
        if abs(want - got) > 1e-12:
            raise AssertionError("softmax differs from ValueGate: %r against %r" % (got, want))


def check_update_and_binary_outcome(rng: np.random.Generator) -> None:
    """Raise unless the update and the binary outcome equal what ValueGate keeps and records."""
    gate, v = ValueGate(seed=0), np.zeros((1, 2))
    for _ in range(100):
        n_food = int(rng.choice([1, 30, OUTCOME_STEPS, OUTCOME_STEPS + 1, 0]))
        closed = len(gate.episodes)
        gate.choose({LIGHT: 1.0})
        for step in range(1, OUTCOME_STEPS + 1):
            gate.observe_reward(1.0 if step == n_food else -0.01)
            if len(gate.episodes) > closed:
                break
        gate.new_episode()
        o = outcome("binary", np.array([1.0]), np.array([n_food if n_food else 10 ** 6]), np.array([True]), 0.0)
        if o[0] != gate_outcome(n_food if n_food else None):
            raise AssertionError("binary outcome differs from ValueGate at n_food %r" % n_food)
        update(v, np.array([LIGHT]), o)
        if abs(v[0, LIGHT] - gate.values[LIGHT]) > 1e-12:
            raise AssertionError("update differs from ValueGate: %r against %r" % (v[0, LIGHT], gate.values[LIGHT]))


def check_hyperbolic_closed_form() -> None:
    """Raise unless a constant hyperbolic outcome gives o (1 - (1 - LEARNING_RATE)^t)."""
    o = outcome("hyperbolic", np.array([3.0]), np.array([21]), np.array([True]), 0.2)
    if abs(o[0] - 3.0 / (1.0 + 0.2 * 20)) > 1e-15:
        raise AssertionError("hyperbolic outcome %r" % o[0])
    v = np.zeros((1, 2))
    for t in range(1, 51):
        update(v, np.array([DECOY]), o)
        if abs(v[0, DECOY] - o[0] * (1.0 - (1.0 - LEARNING_RATE) ** t)) > 1e-12:
            raise AssertionError("value at episode %d differs from the closed form" % t)


def steady_values(design: Design, travel: np.ndarray) -> List[str]:
    """V that each condition converges to at the median travel, for the report."""
    t = float(np.median(travel))
    lines = ["median travel %.1f steps" % t]
    for label, size, wait in (("decoy, no wait", design.decoy, 0), ("light, size, no wait", design.size, 0),
                              ("light, size after the wait", design.size, design.wait),
                              ("light, big after the wait", design.big, design.wait)):
        lines.append("  %s (%.1f, %d steps) converges to %.3f" % (label, size, wait,
                                                                 size / (1.0 + design.k * (t + wait - 1))))
    return lines


def report(design: Design, shares: Dict[str, np.ndarray], n_seq: int, chance: Dict[str, float]) -> None:
    """Print the design, the shares (mean and 5th percentile), the switch speed and the verdict."""
    print("design: %s; %d sequences" % (design, n_seq))
    for name in STRICT + ("binary_change", "benefit_change") + REPORTED:
        print("  %-22s mean %.3f, 5th percentile %.3f" % (name, shares[name].mean(),
                                                           np.percentile(shares[name], 5)))
    for name in ("cost_switch_episodes", "benefit_switch_episodes"):
        done = shares[name][shares[name] >= 0]
        print("  %-24s median %s, 95th percentile %s, never in %d of %d" % (
            name, np.median(done) if len(done) else None, np.percentile(done, 95) if len(done) else None,
            int((shares[name] < 0).sum()), len(shares[name])))
    print("chance of each live criterion at 3 of 3 seeds:", {k: round(c, 3) for k, c in chance.items()})
    for name in STRICT:
        print("  %s 5th percentile %.3f" % (name, np.percentile(shares[name], 5)))
    print("model criteria:", "PASS" if passes(shares) else "FAIL")


def report_passing(passing: List[Design], measured: Measured, args: argparse.Namespace) -> None:
    """The gate chance of every passing design, each run again with `--sequences` and the report seed."""
    for design in passing:
        shares = evaluate(design, measured, args.sequences, args.seed + 1, args.reach)
        chance = gate_pass_chance(shares, np.random.default_rng(args.seed + 2))
        print("  %s: %s" % (design, {name: round(c, 3) for name, c in chance.items()}))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run folders whose value_gate.json gives the travel steps (rooms with no wait)")
    parser.add_argument("--search", action="store_true", help="choose the design from GRID by the fixed order")
    parser.add_argument("--design", type=float, nargs=5, metavar=("K", "DECOY", "SIZE", "BIG", "WAIT"),
                        default=None, help="a design to report without --search")
    parser.add_argument("--sequences", type=int, default=20000,
                        help="sequences for the reported design (2000 until 2026-11-11)")
    parser.add_argument("--seed", type=int, default=0, help="generator seed of the search; the report uses seed + 1")
    parser.add_argument("--reach", type=float, default=REACH,
                        help="chance that the agent reaches the chosen object; a miss reaches the other object")
    parser.add_argument("--show-passing", action="store_true",
                        help="with --search, report the gate chance of every passing design")
    return parser.parse_args(argv)


def run_controls() -> None:
    rng = np.random.default_rng(12345)
    check_softmax(rng)
    check_update_and_binary_outcome(rng)
    check_hyperbolic_closed_form()
    print("controls: softmax, update, binary outcome and hyperbolic closed form match")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    measured, md5 = load_measured(args.runs)
    travel = measured.travel
    for path, digest in md5.items():
        print("record %s md5 %s" % (path, digest))
    steps, counts = np.unique(travel, return_counts=True)
    print("travel steps:", dict(zip(steps.tolist(), counts.tolist())), "n", len(travel))
    print("quieter salience: n %d, min %.3f, 5th percentile %.3f, median %.3f, 95th percentile %.3f, max %.3f" % (
        len(measured.quiet), measured.quiet.min(), np.percentile(measured.quiet, 5), np.median(measured.quiet),
        np.percentile(measured.quiet, 95), measured.quiet.max()))
    print("reach %.4f" % args.reach)
    run_controls()
    if args.search:
        passing, size = search(measured, args.seed, args.reach)
        print("passing designs: %d of %d; first five: %s" % (len(passing), size, passing[:5]))
        if not passing:
            raise SystemExit("no design of the grid passes; the design FAILED in the abstract model")
        design = passing[0]
        if args.show_passing:
            report_passing(passing, measured, args)
    else:
        k, decoy, size, big, wait = args.design
        design = Design(k, decoy, size, big, int(wait))
    print("\n".join(steady_values(design, travel)))
    shares = evaluate(design, measured, args.sequences, args.seed + 1, args.reach)
    report(design, shares, args.sequences, gate_pass_chance(shares, np.random.default_rng(args.seed + 2)))


if __name__ == "__main__":
    main()
