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


class ObsMapSampleMemory:
    """Gated obs_map working memory for DMTS.

    The 2026-06-14 leakage-free correction showed the sample is genuinely absent from
    the RSSM `h_state` but is encoded in `obs_map` (decodes the on-screen stimulus at
    ~1.0). So a usable working memory must capture from `obs_map`, not `h_state`.

    The gate distinguishes the SAMPLE from the CHOICES without phase labels using the
    length of the preceding blank: the sample is the stimulus that follows the short
    fixation blank, the choices follow the long delay blank. On a blank->stimulus
    onset, capture `obs_map` only if the preceding blank run was short (fixation),
    and hold it through the delay and choice. This keeps the captured sample
    available at the decision point.

    Causal and RL-free: depends only on frame energy and the blank-run length, never
    on the task phase. Validate leakage-free (one record per trial).
    """

    def __init__(self, short_blank_max: int = 12, present_frac: float = 0.3,
                 min_range: float = 1e-3):
        self.short_blank_max = short_blank_max
        self.present_frac = present_frac
        self.min_range = min_range
        self.reset()

    def reset(self) -> None:
        self.slot: torch.Tensor | None = None
        self._blank_run = 0
        self._prev_present = True  # so a leading blank starts a run
        self._min_std: float | None = None
        self._max_std: float | None = None

    def _stimulus_present(self, frame: torch.Tensor) -> bool:
        std = float(frame.float().std().item())
        self._min_std = std if self._min_std is None else min(self._min_std, std)
        self._max_std = std if self._max_std is None else max(self._max_std, std)
        rng = self._max_std - self._min_std
        if rng < self.min_range:
            return False
        return std > self._min_std + self.present_frac * rng

    def update(self, obs_map: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Advance one step and return the held sample obs_map (or the current
        obs_map if nothing captured yet)."""
        present = self._stimulus_present(frame)
        if not present:
            self._blank_run += 1
        else:
            # blank -> stimulus onset: capture only if the preceding blank was a
            # short fixation (not the long delay), so we grab the sample not a choice.
            if not self._prev_present and 0 < self._blank_run <= self.short_blank_max:
                self.slot = obs_map.detach().clone()
            self._blank_run = 0
        self._prev_present = present
        return self.slot if self.slot is not None else obs_map.detach()
