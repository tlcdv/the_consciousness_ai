"""Acoustic emotion feature extraction and PAD mapping.

Biological counterpart: auditory cortex -> amygdala/insula pathway.

Emotional responses to sound are driven by acoustic features, not semantic
content. Low-level features like roughness, spectral centroid, and loudness
variability correlate with arousal and valence ratings across cultures
(Juslin & Laukka 2003, Eerola & Vuoskoski 2013).

This module extracts 6 acoustic emotion features from the cochlear
representation and maps them to PAD (Pleasure, Arousal, Dominance)
deltas via a small trainable MLP. It also classifies paralinguistic
vocalizations (laughter, crying, screaming, growling) for discrete
emotion labeling.

The 6 features:
1. Spectral centroid: weighted mean of band energies (brightness)
2. Loudness variability: temporal std of total envelope energy
3. Roughness: energy in 15-300 Hz AM range (Vassilakis 2005)
4. Pitch contour slope: rate of change of dominant frequency
5. Spectral flux: frame-to-frame spectral change
6. Harmonic-to-noise ratio: periodic vs aperiodic energy ratio

Reference: Schuller et al. 2013 (ComParE, acoustic features for
emotion/paralinguistics), Eyben et al. 2010 (openSMILE).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PARALINGUISTIC_CLASSES = [
    "speech", "laughter", "crying", "screaming",
    "growling", "sighing", "silence",
]


class AudioAffectExtractor(nn.Module):
    """Extract emotional features from cochlear representation.

    Operates on the envelope and TFS output from HairCellModel, plus
    raw filterbank energy for spectral analysis.

    Returns a dict with PAD deltas, raw acoustic features, and
    paralinguistic classification.
    """

    def __init__(self, num_bands: int = 64, feature_dim: int = 6):
        super().__init__()
        self.num_bands = num_bands
        self.feature_dim = feature_dim

        # PAD mapping: 6 acoustic features -> 3 PAD deltas
        self.pad_mlp = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
            nn.Tanh(),
        )

        # Paralinguistic classifier: 6 features -> 7 classes
        self.para_classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, len(PARALINGUISTIC_CLASSES)),
        )

        # Store band center frequencies for spectral centroid weighting
        # Approximate ERB centers (will be overwritten if filterbank provides them)
        self._band_weights: torch.Tensor | None = None

    def _ensure_band_weights(self, num_bands: int, device: torch.device) -> torch.Tensor:
        """Lazy-init normalized band frequency weights for spectral centroid."""
        if self._band_weights is None or self._band_weights.shape[0] != num_bands:
            # Linear approximation of ERB center frequencies (normalized 0-1)
            w = torch.linspace(0.0, 1.0, num_bands, device=device)
            self._band_weights = w
        return self._band_weights.to(device)

    def extract_features(
        self,
        envelope: torch.Tensor,
        tfs: torch.Tensor,
        filterbank_energy: torch.Tensor,
    ) -> torch.Tensor:
        """Extract 6 acoustic emotion features.

        Args:
            envelope: [B, num_bands, T] from HairCellModel (first half)
            tfs: [B, num_bands, T] from HairCellModel (second half)
            filterbank_energy: [B, num_bands, T] raw filterbank |output|

        Returns:
            [B, 6] feature vector
        """
        B = envelope.shape[0]
        device = envelope.device

        band_weights = self._ensure_band_weights(envelope.shape[1], device)

        # 1. Spectral centroid: brightness
        band_energy = envelope.pow(2).mean(dim=2)  # [B, num_bands]
        total_energy = band_energy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        centroid = (band_energy * band_weights.unsqueeze(0)).sum(dim=1, keepdim=True) / total_energy
        # [B, 1] in [0, 1]

        # 2. Loudness variability: temporal dynamics
        frame_loudness = envelope.pow(2).sum(dim=1)  # [B, T]
        loudness_std = frame_loudness.std(dim=1, keepdim=True).nan_to_num(0.0)
        loudness_var = loudness_std / (frame_loudness.mean(dim=1, keepdim=True) + 1e-8)
        loudness_var = loudness_var.clamp(0, 5.0) / 5.0  # normalize to ~[0, 1]

        # 3. Roughness: energy in AM range (15-300 Hz)
        # Approximate by variance of envelope fluctuations across bands
        env_diff = envelope[:, :, 1:] - envelope[:, :, :-1]
        roughness = env_diff.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        roughness = roughness.view(B, 1).clamp(0, 2.0) / 2.0

        # 4. Pitch contour slope: rate of change of dominant frequency
        # Find dominant band per frame, compute slope
        dominant_band = band_energy.argmax(dim=1).float()  # [B]
        # Use temporal change in dominant band over frames
        frame_dominant = filterbank_energy.abs().argmax(dim=1).float()  # [B, T]
        if frame_dominant.shape[1] > 1:
            pitch_diff = frame_dominant[:, 1:] - frame_dominant[:, :-1]
            pitch_slope = pitch_diff.mean(dim=1, keepdim=True) / max(envelope.shape[1], 1)
        else:
            pitch_slope = torch.zeros(B, 1, device=device)
        pitch_slope = pitch_slope.clamp(-1, 1)

        # 5. Spectral flux: frame-to-frame spectral change
        if filterbank_energy.shape[2] > 1:
            spec_diff = filterbank_energy[:, :, 1:] - filterbank_energy[:, :, :-1]
            flux = spec_diff.pow(2).sum(dim=1).mean(dim=1, keepdim=True).sqrt()
            flux = flux.clamp(0, 3.0) / 3.0
        else:
            flux = torch.zeros(B, 1, device=device)

        # 6. Harmonic-to-noise ratio (HNR)
        # Periodic energy from TFS autocorrelation peak, noise from residual
        periodic_energy = tfs.pow(2).mean(dim=(1, 2))  # [B]
        noise_energy = (filterbank_energy.abs() - envelope).pow(2).mean(dim=(1, 2)).clamp(min=1e-8)
        hnr = (periodic_energy / noise_energy).log10().clamp(-2, 2) / 2.0  # normalize ~[-1, 1]
        hnr = hnr.unsqueeze(1)  # [B, 1]

        # Stack all features
        features = torch.cat([
            centroid, loudness_var, roughness,
            pitch_slope, flux, hnr,
        ], dim=1)  # [B, 6]

        return features

    def forward(
        self,
        envelope: torch.Tensor,
        tfs: torch.Tensor,
        filterbank_energy: torch.Tensor,
    ) -> dict:
        """Extract features, map to PAD, classify paralinguistics.

        Args:
            envelope: [B, num_bands, T] envelope channels
            tfs: [B, num_bands, T] TFS channels
            filterbank_energy: [B, num_bands, T] raw |filterbank output|

        Returns:
            dict with keys:
                "pad_delta": [B, 3] PAD deltas (valence, arousal, dominance)
                "acoustic_features": [B, 6] raw features
                "paralinguistic": str (most likely class name)
                "paralinguistic_logits": [B, 7] class logits
        """
        features = self.extract_features(envelope, tfs, filterbank_energy)

        pad_delta = self.pad_mlp(features)  # [B, 3]
        para_logits = self.para_classifier(features)  # [B, 7]

        # Get most likely paralinguistic class
        para_idx = para_logits.argmax(dim=1)[0].item()
        para_class = PARALINGUISTIC_CLASSES[para_idx]

        return {
            "pad_delta": pad_delta,
            "acoustic_features": features,
            "paralinguistic": para_class,
            "paralinguistic_logits": para_logits,
        }
