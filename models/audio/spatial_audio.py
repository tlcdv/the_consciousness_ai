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
            waveform: [B, C, T] where C=1 (mono) or C=2 (stereo)
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
            # Stereo: estimate ITD via cross-correlation
            left = waveform[:, 0, :]   # [B, T]
            right = waveform[:, 1, :]  # [B, T]
            return self._estimate_from_stereo(left, right, device)

        return torch.zeros(B, 2, device=device)

    def _estimate_from_stereo(
        self, left: torch.Tensor, right: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Estimate azimuth from ITD and ILD of stereo channels."""
        B = left.shape[0]
        coords = torch.zeros(B, 2, device=device)

        for b in range(B):
            l = left[b].detach().cpu().numpy()
            r = right[b].detach().cpu().numpy()

            # ITD via cross-correlation peak offset
            if np.any(l != 0) and np.any(r != 0):
                corr = np.correlate(l, r, mode="full")
                mid = len(corr) // 2
                search = min(self.max_itd_samples, mid)
                region = corr[mid - search : mid + search + 1]
                peak_offset = np.argmax(region) - search
                # Normalize to [-1, 1]
                azimuth = float(np.clip(peak_offset / max(search, 1), -1.0, 1.0))

                # ILD: energy ratio (rough indicator)
                l_energy = float(np.sum(l ** 2) + 1e-8)
                r_energy = float(np.sum(r ** 2) + 1e-8)
                ild_ratio = (l_energy - r_energy) / (l_energy + r_energy)
                # Blend ITD and ILD estimates
                azimuth = 0.7 * azimuth + 0.3 * float(np.clip(ild_ratio, -1, 1))

                coords[b, 0] = float(np.clip(azimuth, -1.0, 1.0))

        return coords

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
