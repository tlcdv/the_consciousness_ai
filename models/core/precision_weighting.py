"""Weight the two sense bids by how strong each sense's current response is.

Default off (`--bid-precision off`). With `gain`, the vision and audio bids are
multiplied by their share of the total gain, scaled so that equal gains leave
both bids unchanged.

WHY, and what is measured. The vision bid (z-scored KL surprise) and the audio
bid (z-scored prediction error) are both SALIENCE: they fall when an input stays
constant. The multisensory sources separate salience from RELIABILITY, which is
read from the gain of the current response and does not habituate (Ma, Beck,
Latham and Pouget 2006; Fetsch et al. 2009 and 2011; Feldman and Friston 2010;
Stein and Stanford 2008). The shares are a divisive normalization with a zero
constant (Ohshiro, Angelaki and DeAngelis 2011).

STATUS: the offline gain check (scripts/analysis/probe_sense_gain.py) PASSED R1
and R2 at seeds 51 to 53 and FAILED R3 at seed 51, where the light stayed in view
long enough only once, so R3 was not measurable. The owner recorded an override on
2026-09-16 and asked for the live test instead. Nothing here is a result. The
pre-stated Gate B3 judges it.

The gain measures repeat the probe exactly:
    vision  the strongest pooled cell of the frame outside the agent's own disc,
            which always sits at the centre of the agent-centered view
    audio   the waveform RMS over the tone RMS at the source
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

GRID = 16
# The agent's disc has radius 20 in a 96-wide window scaled to the frame, about
# 0.21 of the frame width. The mask is 0.25 of the frame width from the centre.
SELF_MASK_FRACTION = 0.25
# The binaural tone at distance 0 has amplitude 0.5 (simulations/environments/
# audio_mixin.py), so its RMS is 0.5 / sqrt(2).
TONE_RMS_AT_SOURCE = 0.5 / math.sqrt(2.0)


def _self_mask(grid: int, device) -> torch.Tensor:
    """True for cells that hold the agent's own disc."""
    centres = (torch.arange(grid, device=device, dtype=torch.float32) + 0.5) / grid
    y, x = torch.meshgrid(centres, centres, indexing="ij")
    return torch.hypot(x - 0.5, y - 0.5) < SELF_MASK_FRACTION


def vision_gain(frame: torch.Tensor) -> float:
    """Strongest pooled response outside the agent's own disc, in [0, 1].

    frame: [B, 3, H, W] in [0, 1], the tensor the tectum receives.
    """
    if frame is None:
        return 0.0
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    grey = frame.mean(dim=1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(grey, GRID)[:, 0]
    outside = pooled[:, ~_self_mask(GRID, pooled.device)]
    return float(outside.max().clamp(0.0, 1.0).item())


def audio_gain(waveform: Optional[torch.Tensor]) -> float:
    """Waveform RMS over the tone RMS at the source, clipped to [0, 1]."""
    if waveform is None or waveform.numel() == 0:
        return 0.0
    rms = float(torch.sqrt(torch.mean(waveform.float() ** 2)).item())
    return min(1.0, rms / TONE_RMS_AT_SOURCE)


def precision_shares(vision: float, audio: float) -> Tuple[float, float]:
    """Each sense's share of the total gain. Both 0.5 when neither responds."""
    total = vision + audio
    if total <= 0.0:
        return 0.5, 0.5
    return vision / total, audio / total


def weighted_sense_bids(vision_bid: float, audio_bid: float,
                        vision_gain_value: float, audio_gain_value: float
                        ) -> Tuple[float, float]:
    """Both bids multiplied by twice their share, so equal gains change nothing."""
    share_v, share_a = precision_shares(vision_gain_value, audio_gain_value)
    return (_clip(vision_bid * 2.0 * share_v), _clip(audio_bid * 2.0 * share_a))


def _clip(bid: float) -> float:
    return max(0.0, min(1.0, float(bid)))
