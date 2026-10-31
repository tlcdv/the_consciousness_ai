"""Tectal attention: the place the tectum selected gets a gain on the tectum's own map.

docs/decisions/2026_10_17_tectal_attention.md. The normalization model of attention (Reynolds and
Heeger 2009) multiplies the stimulus drive by an attention field, 1 everywhere and above 1 at the
attended place, before normalization. Here the normalization is one global pool: the attended map
is rescaled so that its total absolute activity equals that of the map before. That is a stated
simplification of the model's spatial suppressive field. In the tectum, the loop with the nucleus
isthmi boosts the signals of the selected stimulus (Fernandes et al. 2021).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def attention_field(cells: List[int], grid: int, gain: float) -> np.ndarray:
    """[grid, grid]: 1 everywhere and `gain` on the cells (index row * grid + column)."""
    if gain < 1.0:
        raise ValueError("the attention gain must be 1 or more, got %r" % (gain,))
    field = np.ones((grid, grid))
    for index in cells:
        field[int(index) // grid, int(index) % grid] = gain
    return field


def attend(obs_map: torch.Tensor, field: np.ndarray) -> torch.Tensor:
    """[B, C, H, W] map times the field, rescaled per sample to its total absolute activity before."""
    if np.all(field == 1.0):
        return obs_map
    weights = torch.as_tensor(field, dtype=obs_map.dtype, device=obs_map.device)
    attended = obs_map * weights
    before = obs_map.abs().sum(dim=(1, 2, 3), keepdim=True)
    after = attended.abs().sum(dim=(1, 2, 3), keepdim=True)
    return attended * (before / after.clamp_min(1e-12))
