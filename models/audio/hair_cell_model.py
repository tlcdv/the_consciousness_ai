"""Inner hair cell transduction model.

Biological counterpart: inner hair cells (IHCs) of the organ of Corti.

IHCs convert mechanical vibration of the basilar membrane into neural
signals via mechanoelectric transduction. The output has two components:

1. **Envelope** (rate code): half-wave rectification + low-pass filtering.
   Represents the slowly varying amplitude of each frequency band.
   Drives auditory nerve fiber firing rate. Critical for loudness
   perception and amplitude modulation detection.

2. **Temporal Fine Structure (TFS)** (temporal code): the fast-varying
   carrier signal after envelope removal. Preserves phase information
   critical for pitch perception, sound localization (ITD), and timbre.

Reference: Joris et al. 2004 (neural coding in auditory nerve).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HairCellModel(nn.Module):
    """Inner hair cell envelope and TFS extraction.

    Input:  [B, num_bands, T_frames] from GammatoneFilterbank
    Output: [B, 2 * num_bands, T_frames] (envelope channels + TFS channels)

    The envelope is computed via half-wave rectification followed by
    temporal smoothing (average pooling acts as a low-pass filter at
    ~50 Hz cutoff). TFS is the residual: input minus smoothed envelope.
    """

    def __init__(self, num_bands: int = 64, envelope_pool_size: int = 9):
        super().__init__()
        self.num_bands = num_bands
        # Pool size controls envelope smoothing cutoff.
        # At 250 frames/sec (stride-4 from 16kHz), pool_size=9 gives
        # effective cutoff around ~28 Hz, appropriate for envelope detection.
        # Must be odd for symmetric padding.
        self.envelope_pool_size = envelope_pool_size if envelope_pool_size % 2 == 1 else envelope_pool_size + 1

    def forward(self, filterbank_out: torch.Tensor) -> torch.Tensor:
        """Extract envelope and temporal fine structure.

        Args:
            filterbank_out: [B, num_bands, T_frames] cochleagram

        Returns:
            [B, 2 * num_bands, T_frames] with first num_bands channels
            being envelope and last num_bands channels being TFS
        """
        # Half-wave rectification (IHC mechanoelectric transduction is
        # asymmetric, responding primarily to one direction of deflection)
        rectified = F.relu(filterbank_out)

        # Low-pass filtering via average pooling along time axis
        # Padding keeps temporal dimension unchanged
        pad = self.envelope_pool_size // 2
        envelope = F.avg_pool1d(
            rectified,
            kernel_size=self.envelope_pool_size,
            stride=1,
            padding=pad,
        )

        # TFS: residual after removing the envelope
        # This preserves the fast oscillatory component
        tfs = filterbank_out - envelope

        # Concatenate along band dimension: [B, 2*num_bands, T_frames]
        return torch.cat([envelope, tfs], dim=1)
