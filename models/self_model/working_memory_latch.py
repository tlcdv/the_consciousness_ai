"""
Interference-protected working-memory latch.

The 2026-06-11/12 probe (rssm_working_memory_2026_06_12.md) established two facts:
the RSSM deterministic recurrent state `h_state` holds the DMTS sample at ~99%
across the blank delay, but at the choice phase the incoming choice stimuli overwrite
it (the recurrent state is not protected from interference), so routing raw `h_state`
to the policy did not help matching.

This latch protects a held representation from interference. It keeps a frozen copy
of `h_state` captured while the input was "blank" (low stimulus energy) and stops
updating it when a stimulus is present. For DMTS the sequence is fixation (blank) ->
sample (stimulus) -> delay (blank) -> choice (stimulus). During the delay the latch
copies the current `h_state` (which still carries the sample, 99%); when the choices
appear the latch freezes, so at the choice phase it exposes the last delay `h_state`,
i.e. the sample, instead of the choice-contaminated current state.

Phase-agnostic by design: "stimulus present" is detected from frame energy, never
from the task phase (the agent is never told the phase). The blank threshold adapts
per episode from the running min/max of the frame standard deviation, so it does not
need hand-tuning to a specific environment's brightness.
"""
from __future__ import annotations

import torch


class WorkingMemoryLatch:
    def __init__(self, present_frac: float = 0.3, min_range: float = 1e-3):
        # A frame counts as "stimulus present" when its std exceeds
        # min_std + present_frac * (max_std - min_std), with min/max tracked per
        # episode. present_frac in (0, 1): lower = more frames count as stimulus.
        self.present_frac = present_frac
        self.min_range = min_range
        self.reset()

    def reset(self) -> None:
        """Clear the latch and the per-episode frame-energy statistics."""
        self.latched: torch.Tensor | None = None
        self._min_std: float | None = None
        self._max_std: float | None = None

    def _stimulus_present(self, frame: torch.Tensor) -> bool:
        """True if the frame carries a stimulus (high energy), False if blank.

        Tracks the running min/max of the frame std this episode and thresholds at
        a fraction of that range. Before the range is established (first frames) it
        treats the frame as blank, so the latch starts by copying h_state.
        """
        std = float(frame.float().std().item())
        self._min_std = std if self._min_std is None else min(self._min_std, std)
        self._max_std = std if self._max_std is None else max(self._max_std, std)
        rng = self._max_std - self._min_std
        if rng < self.min_range:
            return False  # no contrast seen yet: treat as blank
        threshold = self._min_std + self.present_frac * rng
        return std > threshold

    def update(self, h_state: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Advance one step and return the protected memory.

        While the frame is blank, copy `h_state` into the latch (capturing the
        sample-bearing delay state). While a stimulus is present, hold the frozen
        latch. Returns the latch if one exists, else the (detached) current state.
        """
        if not self._stimulus_present(frame):
            self.latched = h_state.detach().clone()
        if self.latched is not None:
            return self.latched
        return h_state.detach()
