"""The selection switch: a learned value for each heard tone decides which bound object the tectum follows.

`docs/decisions/2026_10_16_value_gate.md`. In monkeys, the caudate learns the value of objects from
reward, and through the substantia nigra it releases gaze toward high-value objects in the
superior colliculus (Kim and Hikosaka, Neuron 2013; Hikosaka, Takikawa and Kawagoe, Physiol. Rev.
2000). A learned value captures attention beyond salience (Anderson, Laurent and Yantis, PNAS
2011). The value is updated by the reward prediction error that dopamine neurons signal (Schultz,
Dayan and Montague, Science 1997).

Here a value V belongs to a cochlear band (a heard tone). Once in each episode, at the first step
with bound objects, the gate chooses a band by softmax over salience + VALUE_WEIGHT x V with
temperature TEMPERATURE, from a generator of its own, and keeps the choice to the episode end. The
outcome is 1 when food comes (reward above FOOD) within OUTCOME_STEPS steps after the choice, else 0.
The error is outcome - V, and V += learning rate x error, once per episode.

All values were set on 2026-10-16 before any run, and checked in an abstract model only.

`outcome_rule` "hyperbolic" (docs/decisions/2026_11_08_competing_goals_gate.md): the outcome is the
reward of the food step divided by (1 + delay_rate x (n - 1)), where n counts the steps from the choice
to the food. Food is still a reward above FOOD within OUTCOME_STEPS steps, and no food still gives 0.
In monkeys, the dopamine value signal falls hyperbolically with the delay of a reward and rises with
its size (Kobayashi and Schultz, J. Neurosci. 28, 7837, 2008). "binary" is the default and is the gate
as it was.

`salience_floor` (the same document, owner decision of 2026-11-14): at the choice the gate ignores a
candidate whose salience is below the floor, and the record holds the removed candidates. With one
tone in the room the binder also gives a small band (salience 0.068 or less in the offline check),
and without the floor the gate chose it at about 2 choices in 100. When every candidate is below the
floor the gate makes no choice. None is the default and is the gate as it was.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

VALUE_WEIGHT = 2.0
TEMPERATURE = 0.25
LEARNING_RATE = 0.3
OUTCOME_STEPS = 60
FOOD = 0.5          # the reward of a step in the light is about 1.0; outside it is below 0
OUTCOME_RULES = ("binary", "hyperbolic")


class ValueGate:
    """Values per band, one choice per episode, one update per episode."""

    def __init__(self, seed: int, learning_rate: float = LEARNING_RATE, value_weight: float = VALUE_WEIGHT,
                 temperature: float = TEMPERATURE, outcome_steps: int = OUTCOME_STEPS,
                 outcome_rule: str = "binary", delay_rate: float = 0.0,
                 salience_floor: Optional[float] = None):
        if salience_floor is not None and float(salience_floor) < 0.0:
            raise ValueError("salience_floor must be 0 or more, got %r" % (salience_floor,))
        self.salience_floor = None if salience_floor is None else float(salience_floor)
        if outcome_rule not in OUTCOME_RULES:
            raise ValueError("outcome_rule must be one of %s, got %r" % (OUTCOME_RULES, outcome_rule))
        if float(delay_rate) < 0.0:
            raise ValueError("delay_rate must be 0 or more, got %r" % (delay_rate,))
        self.outcome_rule, self.delay_rate = outcome_rule, float(delay_rate)
        self.learning_rate, self.value_weight = learning_rate, value_weight
        self.temperature, self.outcome_steps = temperature, outcome_steps
        self.rng = np.random.default_rng(seed)
        self.values: Dict[int, float] = {}
        self.episodes = []
        self.choice: Optional[int] = None
        self._open = None

    def probabilities(self, salience: Dict[int, float]) -> Dict[int, float]:
        """Softmax over salience + weight x value, at the temperature."""
        bands = list(salience)
        if len(bands) == 1:
            return {bands[0]: 1.0}
        score = np.array([salience[b] + self.value_weight * self.values.get(b, 0.0) for b in bands])
        score = (score - score.max()) / self.temperature
        p = np.exp(score) / np.exp(score).sum()
        return {b: float(q) for b, q in zip(bands, p)}

    def choose(self, salience: Dict[int, float]) -> Optional[int]:
        """The band to follow in this episode. A choice already made is kept.

        With a salience floor, the candidates below it are left out. None, with no draw and no
        record, when no candidate is left."""
        if self.choice is not None:
            return self.choice
        removed = {}
        if self.salience_floor is not None:
            removed = {b: s for b, s in salience.items() if s < self.salience_floor}
            salience = {b: s for b, s in salience.items() if s >= self.salience_floor}
            if not salience:
                return None
        p = self.probabilities(salience)
        bands = list(p)
        self.choice = bands[int(self.rng.choice(len(bands), p=[p[b] for b in bands]))]
        self._open = {"choice": self.choice, "salience": {str(b): float(s) for b, s in salience.items()},
                      "priorities": {str(b): float(salience[b] + self.value_weight * self.values.get(b, 0.0))
                                     for b in salience},
                      "value_before": float(self.values.get(self.choice, 0.0)), "steps": 0}
        if self.salience_floor is not None:
            self._open["removed"] = {str(b): float(s) for b, s in removed.items()}
        return self.choice

    def observe_reward(self, reward: float) -> None:
        """Food within the window closes the choice with its outcome; the end of the window closes it with 0."""
        if self._open is None:
            return
        self._open["steps"] += 1
        if reward > FOOD:
            self._close(self._food_outcome(reward))
        elif self._open["steps"] >= self.outcome_steps:
            self._close(0.0)

    def _food_outcome(self, reward: float) -> float:
        """Binary: 1. Hyperbolic: the food reward divided by (1 + delay_rate x (steps - 1)), and the
        open record keeps the food reward."""
        if self.outcome_rule == "binary":
            return 1.0
        self._open["food_reward"] = float(reward)
        return float(reward) / (1.0 + self.delay_rate * (self._open["steps"] - 1))

    def new_episode(self) -> None:
        """A step without sound starts the next episode: an open choice closes with 0."""
        if self._open is not None:
            self._close(0.0)
        self.choice = None

    def _close(self, outcome: float) -> None:
        band = self._open["choice"]
        before = self.values.get(band, 0.0)
        error = outcome - before
        self.values[band] = before + self.learning_rate * error
        self.episodes.append(dict(self._open, outcome=outcome, error=float(error),
                                  value_after=float(self.values[band])))
        self._open = None
