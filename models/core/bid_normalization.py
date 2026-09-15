"""A bid that measures change against the module's own running baseline.

Provenance: candidate S5 in scripts/analysis/probe_bid_counterfactual.py, sigmoid of a
causal running z-score of the summed KL. It de-saturated the vision bid at all three
trained checkpoints, whose KL scales differ 8,079x
(docs/results/bid_counterfactual_2026_09.md). The constants are the ones fixed there
before any winner was read.

Biological reading, hypothesis only: sensory neurons adapt to a constant input and
respond to change against a baseline (habituation; Bennett, A Brief History of
Intelligence, chapters 1 and 4). A constant surprise then bids at the midpoint instead
of at the ceiling.

The statistics are read BEFORE they are updated, so a bid never includes its own value.
"""

from __future__ import annotations

import math

EMA_ALPHA = 0.01
SD_FLOOR = 1e-9


def stable_sigmoid(z: float) -> float:
    """1 / (1 + e^-z) without overflow. Live KL jumps by thousands produce |z| far past
    the ~709 where math.exp overflows."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    if z < -700.0:
        return 0.0
    e = math.exp(z)
    return e / (1.0 + e)


class RunningZScoreBid:
    """sigmoid((x - running mean) / running sd), then update the running statistics.

    A series whose running sd is at or below SD_FLOOR gets z = 0 (bid 0.5), so a constant
    input is never amplified into apparent variation. The first reading also bids 0.5.
    """

    def __init__(self, alpha: float = EMA_ALPHA, initial_var: float = 1.0):
        """`initial_var` 1.0 is the validated S5 setting, harmless for the KL (thousands).
        For a small signal it hides real changes for hundreds of steps, so a small-scale
        signal (the auditory prediction error, about 0.01) uses 0.0, which has no scale."""
        self.alpha = float(alpha)
        self.initial_var = float(initial_var)
        self.mean = 0.0
        self.var = self.initial_var
        self.seen = 0

    def bid(self, value: float) -> float:
        z = self._z(float(value))
        self._update(float(value))
        return stable_sigmoid(z)

    def _z(self, value: float) -> float:
        if self.seen == 0:
            return 0.0
        sd = self.var ** 0.5
        if sd <= SD_FLOOR:
            return 0.0
        return (value - self.mean) / (sd + 1e-12)

    def _update(self, value: float) -> None:
        if self.seen == 0:
            self.mean, self.var = value, self.initial_var
        else:
            delta = value - self.mean
            self.mean += self.alpha * delta
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta)
        self.seen += 1
