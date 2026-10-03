"""Learned valence: a linear temporal difference critic over the module bids.

V(bids) = sum over modules of value[m] * bid[m]. After each transition,
    delta = reward + TD_DISCOUNT * V(next bids) - V(bids)
    value[m] += TD_LEARNING_RATE * delta * bid[m]
and at the end of an episode the next value is 0. A module whose value is large in size
predicts reward, or its absence, and its bid gets a salience boost of gain * |value|.

Grounding, paraphrased: valence is assigned to sensory images separately, as a common
currency ranking stimuli by survival value (Feinberg & Mallatt); early vertebrates learn
from the change in predicted reward, the temporal difference error carried by dopamine
(Bennett, breakthrough 2). This replaces, when enabled, the fixed approach/threat module
sets in AffectiveModulator, which are this project's own invention.

The reward is the EXTERNAL task reward only, never the shaped reward that contains
homeostatic terms (ethics rule E2, docs/ethics_framework.md).

TD_DISCOUNT and TD_LEARNING_RATE are engineering values set 2026-09-15 before any run.

The boost rule. A value has no upper limit, so under the default rule `linear` the boost
gain * |value| has none either. In recorded runs the value of vision reached 6 to 8 in one
long episode in the light, and the boost alone then put the vision bid at its upper limit
(docs/results/vision_lockin_2026_10.md). The rule `saturating` gives gain * tanh(|value|),
which is at most the gain and equals the linear rule to first order for small values. It
adds no constant. The rule changes the boost only. What the critic learns is the same.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

TD_DISCOUNT = 0.95
TD_LEARNING_RATE = 0.05
BOOST_RULES = ("linear", "saturating")


class LearnedValence:
    """Per-module values learned from temporal difference error."""

    def __init__(self, boost_rule: str = "linear"):
        if boost_rule not in BOOST_RULES:
            raise ValueError("boost_rule must be one of %s, got %r" % (BOOST_RULES, boost_rule))
        self.boost_rule = boost_rule
        self.values: Dict[str, float] = {}
        self.previous_bids: Optional[Dict[str, float]] = None
        self.previous_reward = 0.0
        self.last_td_error: Optional[float] = None

    def value_of(self, bids: Dict[str, float]) -> float:
        return sum(self.values.get(name, 0.0) * bid for name, bid in bids.items())

    def observe(self, bids: Dict[str, float], reward: float) -> None:
        """Record this step's bids and the reward that followed its action."""
        current = {name: float(bid) for name, bid in bids.items()}
        if self.previous_bids is not None:
            delta = (self.previous_reward + TD_DISCOUNT * self.value_of(current)
                     - self.value_of(self.previous_bids))
            self._learn(delta)
        self.previous_bids = current
        self.previous_reward = float(reward)

    def end_episode(self) -> None:
        """Terminal transition: the value after the last step is 0."""
        if self.previous_bids is None:
            return
        self._learn(self.previous_reward - self.value_of(self.previous_bids))
        self.previous_bids = None
        self.previous_reward = 0.0

    def boost(self, name: str, gain: float) -> float:
        size = abs(self.values.get(name, 0.0))
        if self.boost_rule == "saturating":
            return gain * math.tanh(size)
        return gain * size

    def _learn(self, delta: float) -> None:
        self.last_td_error = float(delta)
        for name, bid in self.previous_bids.items():
            self.values[name] = self.values.get(name, 0.0) + TD_LEARNING_RATE * delta * bid
