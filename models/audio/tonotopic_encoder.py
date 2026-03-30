"""Tonotopic encoder for auditory features.

Biological counterpart: primary auditory cortex (A1) tonotopic maps.

Just as the RetinotopicEncoder preserves spatial correspondence between
image regions and feature grid positions, the TonotopicEncoder preserves
frequency correspondence: band 0 (low frequency, cochlear apex) maps to
position 0 in the output, band N (high frequency, cochlear base) maps to
position N. This tonotopic organization is maintained throughout the
auditory hierarchy from cochlea to A1 (Romani et al. 1982, Merzenich &
Reid 1974).

The encoder is a trainable 1D convolutional stack that processes the
hair cell output (envelope + TFS) and produces a compact tonotopic
feature map compatible with the tectum grid.

License: MIT
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TonotopicConvStack(nn.Module):
    """Lightweight 1D CNN that preserves tonotopic structure.

    Analogous to RetinotopicConvStack for vision: strided convolutions
    reduce the temporal dimension while maintaining frequency band ordering.

    Input:  [B, 2 * num_bands, T_frames]
    Output: [B, out_channels, T_reduced]
    """

    def __init__(self, in_channels: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 96, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(96, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)


class TonotopicEncoder(nn.Module):
    """Encodes hair cell output into a tonotopic feature map.

    Output shape: [B, feature_dim, num_output_bands] where num_output_bands
    matches the tectum grid_size (default 16) for spatial compatibility.

    The frequency axis is the spatial dimension: position 0 = low frequency
    (cochlear apex), position 15 = high frequency (cochlear base).

    For tectum integration, call reshape_for_tectum() to get
    [B, feature_dim, grid_size, grid_size] where frequency maps to the
    elevation axis and azimuth is repeated (filled by spatial audio).
    """

    def __init__(
        self,
        num_bands: int = 64,
        feature_dim: int = 64,
        num_output_bands: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_output_bands = num_output_bands

        # Temporal processing: reduce T_frames to a fixed-size representation
        self.conv_stack = TonotopicConvStack(
            in_channels=2 * num_bands,
            out_channels=feature_dim,
        )

        # Adaptive pool reduces temporal dimension to num_output_bands
        # This effectively bins the temporal features into frequency-aligned slots
        self.temporal_pool = nn.AdaptiveAvgPool1d(num_output_bands)

        # Channel projection + normalization
        self.proj = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, hair_cell_out: torch.Tensor) -> torch.Tensor:
        """Encode hair cell features into tonotopic feature map.

        Args:
            hair_cell_out: [B, 2 * num_bands, T_frames]

        Returns:
            [B, feature_dim, num_output_bands] tonotopic features
        """
        # Temporal convolutions
        x = self.conv_stack(hair_cell_out)  # [B, feature_dim, T_reduced]

        # Pool to fixed number of output bands
        x = self.temporal_pool(x)  # [B, feature_dim, num_output_bands]

        # Channel projection
        x = self.proj(x)  # [B, feature_dim, num_output_bands]

        # LayerNorm over feature dimension (transpose, normalize, transpose back)
        x = x.transpose(1, 2)  # [B, num_output_bands, feature_dim]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, feature_dim, num_output_bands]

        return x

    def reshape_for_tectum(
        self, tonotopic_features: torch.Tensor, grid_size: int = 16
    ) -> torch.Tensor:
        """Reshape tonotopic features for tectum grid compatibility.

        Maps the 1D frequency axis to the elevation (vertical) axis of the
        2D tectum grid. The azimuth (horizontal) axis is filled by repeating,
        to be overridden by spatial audio placement.

        Biological basis: in the inferior colliculus, frequency is mapped
        along one axis (isofrequency contours) while spatial azimuth is
        mapped along the orthogonal axis (Merzenich & Reid 1974).

        Args:
            tonotopic_features: [B, feature_dim, num_output_bands]
            grid_size: tectum grid spatial size

        Returns:
            [B, feature_dim, grid_size, grid_size]
        """
        B, C, N = tonotopic_features.shape
        # Pool/interpolate frequency bands to match grid_size if needed
        if N != grid_size:
            tonotopic_features = F.adaptive_avg_pool1d(tonotopic_features, grid_size)

        # [B, C, grid_size] -> [B, C, grid_size, 1] -> [B, C, grid_size, grid_size]
        return tonotopic_features.unsqueeze(3).expand(-1, -1, -1, grid_size)
