"""Cochlear-inspired gammatone filterbank.

Biological counterpart: basilar membrane frequency decomposition.

The cochlea decomposes incoming sound into ~3500 frequency channels via
mechanical resonance of the basilar membrane. Each location along the
membrane acts as a bandpass filter tuned to a characteristic frequency,
with bandwidth increasing with frequency (ERB scale). The gammatone
filter is the standard linear approximation of this process (Patterson
et al. 1992), widely used in computational auditory models.

This module implements a 64-band gammatone filterbank as a frozen Conv1d
layer. Center frequencies are spaced on the ERB scale (Glasberg & Moore
1990), covering the human audible range from 20 Hz to 16 kHz. The
filterbank is frozen (no trainable parameters), paralleling how DINOv2
is used as a frozen visual feature extractor in the retinotopic encoder.

License: MIT (all dependencies are scipy + torch, both permissive).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import gammatone as scipy_gammatone


def erb_space(low_freq: float, high_freq: float, num_bands: int) -> np.ndarray:
    """Compute center frequencies on the ERB (Equivalent Rectangular Bandwidth) scale.

    Uses the Glasberg & Moore (1990) formula:
        ERB_number = 21.366 * log10(1 + f / 228.7)
        f = 228.7 * (10^(ERB_number / 21.366) - 1)

    Returns:
        1D array of center frequencies in Hz, low to high.
    """
    erb_low = 21.366 * math.log10(1.0 + low_freq / 228.7)
    erb_high = 21.366 * math.log10(1.0 + high_freq / 228.7)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    return 228.7 * (10.0 ** (erb_points / 21.366) - 1.0)


class GammatoneFilterbank(nn.Module):
    """Frozen gammatone filterbank mimicking cochlear frequency decomposition.

    Input:  [B, 1, T] raw waveform at sample_rate Hz
    Output: [B, num_bands, T_frames] cochleagram (downsampled by stride)

    All parameters are frozen. The cochlea's physical structure does not
    change during learning; only downstream cortical processing adapts.
    """

    def __init__(
        self,
        num_bands: int = 64,
        sample_rate: int = 16000,
        freq_low: float = 20.0,
        freq_high: float = 16000.0,
        filter_order: int = 4,
        filter_length: int = 1024,
        stride: int = 4,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.stride = stride

        center_freqs = erb_space(freq_low, min(freq_high, sample_rate / 2 - 1), num_bands)
        self.register_buffer(
            "center_frequencies",
            torch.from_numpy(center_freqs).float(),
        )

        # Build impulse responses for each band using scipy
        kernels = np.zeros((num_bands, 1, filter_length), dtype=np.float32)
        for i, cf in enumerate(center_freqs):
            # scipy.signal.gammatone returns (b, a) coefficients for IIR,
            # or the FIR impulse response depending on ftype
            b, a = scipy_gammatone(cf, ftype="iir", fs=sample_rate, order=filter_order)
            # Generate impulse response from IIR coefficients
            from scipy.signal import lfilter
            impulse = np.zeros(filter_length, dtype=np.float64)
            impulse[0] = 1.0
            ir = lfilter(b, a, impulse).astype(np.float32)
            # Normalize to unit energy
            energy = np.sqrt(np.sum(ir ** 2) + 1e-8)
            kernels[i, 0, :] = ir / energy

        # Store as non-trainable Conv1d weight buffer
        self.register_buffer("filter_weights", torch.from_numpy(kernels))

        # Freeze all parameters
        self.requires_grad_(False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply gammatone filterbank to raw waveform.

        Args:
            waveform: [B, 1, T] mono audio at self.sample_rate Hz

        Returns:
            [B, num_bands, T_frames] cochleagram, where
            T_frames = (T - filter_length) // stride + 1
        """
        # Grouped conv1d: each filter applied independently to the single input channel
        # groups=1 because input has 1 channel and we have num_bands output channels
        cochleagram = torch.nn.functional.conv1d(
            waveform,
            self.filter_weights,
            stride=self.stride,
            padding=self.filter_weights.shape[2] // 2,
        )
        return cochleagram

    def get_center_frequencies(self) -> torch.Tensor:
        """Return the ERB-spaced center frequencies in Hz."""
        return self.center_frequencies
