"""Auditory surprise: a salience bid from the error of predicting the next sound.

A small predictor learns the next cochlear band-energy profile from the current one.
Its squared error is the surprise; the bid is the running z-score of that error, the same
reduction as the vision bid (models/core/bid_normalization.py). A sound that stays the
same becomes predictable and its bid falls (habituation); a new sound is not predicted
and its bid rises.

Grounding, paraphrased: salience in the tectum comes from features such as novelty
(Feinberg & Mallatt); neurons adapt to constant input and respond to change (Bennett,
chapters 1 and 4). The predictor is an engineering stand-in, not a model of a nucleus.

The predictor learns only when the module is in training mode AND gradients are enabled,
so replays and probes under torch.no_grad or eval() never change its weights. Its inputs
are detached, so its loss reaches nothing but the predictor.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.bid_normalization import RunningZScoreBid

SURPRISE_HIDDEN = 32
SURPRISE_LEARNING_RATE = 1e-3


def band_profile(cochleagram: torch.Tensor) -> torch.Tensor:
    """[B, bands, T] filterbank output -> [B, bands] log mean energy per band."""
    return torch.log1p(cochleagram.abs().mean(dim=2))


class AuditorySurprise(nn.Module):
    """Predicts the next band profile; bids by the z-scored prediction error."""

    def __init__(self, num_bands: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(num_bands, SURPRISE_HIDDEN), nn.GELU(),
            nn.Linear(SURPRISE_HIDDEN, num_bands))
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=SURPRISE_LEARNING_RATE)
        # initial_var 0.0: the error is about 0.01, so the validated start of 1.0 would
        # hide every change for hundreds of steps (measured in tests/test_audio_surprise.py).
        self.normalizer = RunningZScoreBid(initial_var=0.0)
        self.previous: Optional[torch.Tensor] = None
        self.last_error: Optional[float] = None
        self.error_history: List[float] = []

    def bid(self, profile: torch.Tensor) -> float:
        current = profile.detach()
        if self.previous is None:
            self.previous = current
            return 0.5
        error = self._prediction_error(self.previous, current)
        self.previous = current
        self.last_error = error
        self.error_history.append(error)
        return self.normalizer.bid(error)

    def _prediction_error(self, previous: torch.Tensor, current: torch.Tensor) -> float:
        if not (self.training and torch.is_grad_enabled()):
            with torch.no_grad():
                return float(F.mse_loss(self.predictor(previous), current))
        loss = F.mse_loss(self.predictor(previous), current)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())
