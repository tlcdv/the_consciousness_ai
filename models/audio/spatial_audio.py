"""Spatial audio computation from binaural cues.

Biological counterpart: medial superior olive (MSO) and lateral superior
olive (LSO) in the brainstem auditory pathway.

Sound localization in mammals relies on two primary binaural cues:
- ITD (Interaural Time Difference): computed by the MSO via coincidence
  detection. Dominant below ~1500 Hz. Maps to azimuth.
- ILD (Interaural Level Difference): computed by the LSO via excitatory-
  inhibitory comparison. Dominant above ~1500 Hz. Also maps to azimuth.

Elevation is estimated from spectral cues (pinna filtering), which we
approximate here or accept from environment metadata.

Reference: Grothe et al. 2010 (mechanisms of sound localization in mammals).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class SpatialAudioComputer(nn.Module):
    """Compute sound source azimuth and elevation.

    Accepts mono waveforms (default center), stereo waveforms (ITD/ILD
    estimation), or environment-provided metadata.

    Output: [B, 2] with (azimuth, elevation) in [-1, 1].
    For tectum compatibility, call expand_for_tectum() to broadcast
    to [B, feature_dim, 2].
    """

    def __init__(self, sample_rate: int = 16000, head_diameter_m: float = 0.18):
        super().__init__()
        self.sample_rate = sample_rate
        # Maximum ITD for a human head (~0.18m diameter)
        # max_itd = head_diameter / speed_of_sound
        self.max_itd_samples = int(head_diameter_m / 343.0 * sample_rate) + 1

    def forward(
        self,
        waveform: torch.Tensor,
        metadata: dict | None = None,
    ) -> torch.Tensor:
        """Compute spatial coordinates.

        Args:
            waveform: [B, C, T] where C=1 (mono), C=2 (left, right) or
                C=4 (left, right, upper, lower)
            metadata: optional dict with "audio_azimuth" and "audio_elevation"
                      keys (floats in [-1, 1]). Overrides computation.

        Returns:
            [B, 2] tensor with (azimuth, elevation) in [-1, 1]
        """
        B = waveform.shape[0]
        device = waveform.device

        # Environment metadata takes priority
        if metadata is not None and "audio_azimuth" in metadata:
            az = float(metadata["audio_azimuth"])
            el = float(metadata.get("audio_elevation", 0.0))
            coords = torch.tensor([[az, el]], device=device).expand(B, -1)
            return coords.clamp(-1.0, 1.0)

        channels = waveform.shape[1]

        if channels == 1:
            # Mono: no spatial information, default to center
            return torch.zeros(B, 2, device=device)

        if channels >= 2:
            # Channels are (left, right) and, with 4 channels, (upper, lower).
            # Azimuth is positive toward the right; elevation positive toward the
            # lower ear (image rows grow downward, so this matches the tectum grid).
            coords = self._estimate_from_stereo(waveform[:, 0, :], waveform[:, 1, :], device)
            if channels >= 4:
                coords[:, 1] = self._estimate_from_stereo(
                    waveform[:, 2, :], waveform[:, 3, :], device)[:, 0]
            return coords

        return torch.zeros(B, 2, device=device)

    def _estimate_from_stereo(
        self, first: torch.Tensor, second: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Direction along one ear pair from ITD and ILD, positive toward `second`.

        Before 2026-09-15 the ILD term had the opposite sign to the ITD term, so a
        louder right ear pulled the estimate toward the left. No training run used
        this path then: every environment sent mono sound.
        """
        B = first.shape[0]
        coords = torch.zeros(B, 2, device=device)
        for b in range(B):
            coords[b, 0] = self._pair_direction(
                first[b].detach().cpu().numpy(), second[b].detach().cpu().numpy())
        return coords

    def _pair_direction(self, first: np.ndarray, second: np.ndarray) -> float:
        if not (np.any(first != 0) and np.any(second != 0)):
            return 0.0
        # ITD: a peak at a positive lag means `first` is delayed, so `second` leads.
        corr = np.correlate(first, second, mode="full")
        mid = len(corr) // 2
        search = min(self.max_itd_samples, mid)
        peak_offset = np.argmax(corr[mid - search: mid + search + 1]) - search
        itd = float(np.clip(peak_offset / max(search, 1), -1.0, 1.0))
        # ILD: positive when `second` is louder.
        first_energy = float(np.sum(first ** 2) + 1e-8)
        second_energy = float(np.sum(second ** 2) + 1e-8)
        ild = (second_energy - first_energy) / (first_energy + second_energy)
        return float(np.clip(0.7 * itd + 0.3 * ild, -1.0, 1.0))

    @staticmethod
    def expand_for_tectum(
        coords: torch.Tensor, feature_dim: int = 64
    ) -> torch.Tensor:
        """Broadcast [B, 2] spatial coords to [B, feature_dim, 2] for tectum.

        The tectum TopographicMap expects [B, feature_dim, 2] where each
        feature channel carries the same (azimuth, elevation) pair.
        """
        # [B, 2] -> [B, 1, 2] -> [B, feature_dim, 2]
        return coords.unsqueeze(1).expand(-1, feature_dim, -1)
