"""Auditory specialist module for Global Workspace competition.

Biological counterpart: auditory cortex (content analysis) combined with
inferior colliculus (spatial analysis) and amygdala (affective evaluation).

This module chains the cochlear front-end (gammatone filterbank, hair cell
model) through the tonotopic encoder and produces:
1. Workspace content tensor for GNW competition
2. Salience bid based on acoustic novelty and loudness change
3. Spatial audio features for tectum IE fusion
4. Affective features (PAD deltas) for emotion integration

The specialist implements receive_broadcast() for reentrant feedback,
following the same pattern as SensoryTectum.

Graceful degradation: when no audio is available (waveform is None or
all zeros), returns zero content with zero bid. This preserves backward
compatibility with environments that have no audio.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.audio.gammatone_filterbank import GammatoneFilterbank
from models.audio.hair_cell_model import HairCellModel
from models.audio.tonotopic_encoder import TonotopicEncoder
from models.audio.spatial_audio import SpatialAudioComputer
from models.audio.audio_affect_extractor import AudioAffectExtractor


class AuditorySpecialist(nn.Module):
    """Workspace-competing auditory processing module.

    Chains: GammatoneFilterbank -> HairCellModel -> TonotopicEncoder
            -> workspace projection + spatial + affect extraction

    Input:  [B, 1, T] raw mono waveform at sample_rate Hz (or None)
    Output: (workspace_content [B, workspace_dim], bid float)
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        self.sample_rate = config.get("audio_sample_rate", 16000)
        self.num_bands = config.get("audio_num_bands", 64)
        self.feature_dim = config.get("tectum_feature_dim", 64)
        self.num_output_bands = config.get("tectum_grid_size", 16)
        workspace_dim = config.get("workspace_dim", 256)

        # Cochlear front-end (frozen)
        self.filterbank = GammatoneFilterbank(
            num_bands=self.num_bands,
            sample_rate=self.sample_rate,
        )

        # Inner hair cell transduction
        self.hair_cell = HairCellModel(num_bands=self.num_bands)

        # Tonotopic encoder (trainable)
        self.encoder = TonotopicEncoder(
            num_bands=self.num_bands,
            feature_dim=self.feature_dim,
            num_output_bands=self.num_output_bands,
        )

        # Spatial audio computation
        self.spatial = SpatialAudioComputer(sample_rate=self.sample_rate)

        # Affect extraction
        self.affect = AudioAffectExtractor(num_bands=self.num_bands)

        # Workspace projection: flatten tonotopic features -> workspace_dim
        self.workspace_proj = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_output_bands, workspace_dim),
            nn.GELU(),
            nn.LayerNorm(workspace_dim),
        )

        # Salience computation: small MLP from acoustic features to scalar bid
        self.salience_net = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Cache for external access (tectum, emotion system)
        self._last_spatial: torch.Tensor | None = None
        self._last_affect: dict | None = None
        self._last_content: torch.Tensor | None = None
        self._last_tonotopic: torch.Tensor | None = None

    def forward(
        self,
        waveform: torch.Tensor | None,
        metadata: dict | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Process raw audio waveform through the full cochlear pipeline.

        Args:
            waveform: [B, 1, T] mono waveform or [B, 2, T] stereo, or None
            metadata: optional dict with spatial info from environment

        Returns:
            (workspace_content [B, workspace_dim], bid scalar in [0, 1])
        """
        # Graceful degradation: no audio available
        if waveform is None:
            return self._zero_output(metadata)

        B = waveform.shape[0]
        device = waveform.device

        # Check for all-zero (silence / stub)
        if waveform.abs().max().item() < 1e-7:
            return self._zero_output(metadata, device=device, batch_size=B)

        # Ensure mono for cochlear processing
        if waveform.shape[1] > 1:
            mono = waveform.mean(dim=1, keepdim=True)
        else:
            mono = waveform

        # 1. Cochlear decomposition (frozen)
        with torch.no_grad():
            cochleagram = self.filterbank(mono)  # [B, num_bands, T_frames]

        # 2. Hair cell transduction
        hair_cell_out = self.hair_cell(cochleagram)  # [B, 2*num_bands, T_frames]

        # Split envelope and TFS for affect extraction
        envelope = hair_cell_out[:, :self.num_bands, :]
        tfs = hair_cell_out[:, self.num_bands:, :]

        # 3. Tonotopic encoding (trainable)
        tonotopic = self.encoder(hair_cell_out)  # [B, feature_dim, num_output_bands]
        self._last_tonotopic = tonotopic

        # 4. Workspace content
        flat = tonotopic.reshape(B, -1)  # [B, feature_dim * num_output_bands]
        content = self.workspace_proj(flat)  # [B, workspace_dim]
        self._last_content = content

        # 5. Spatial audio
        spatial_coords = self.spatial(waveform, metadata)  # [B, 2]
        self._last_spatial = SpatialAudioComputer.expand_for_tectum(
            spatial_coords, self.feature_dim
        )

        # 6. Affect extraction
        filterbank_energy = cochleagram.abs()
        affect_out = self.affect(envelope, tfs, filterbank_energy)
        self._last_affect = affect_out

        # 7. Salience bid from acoustic features
        acoustic_features = affect_out["acoustic_features"]  # [B, 6]
        bid = self.salience_net(acoustic_features).mean().item()

        return content, bid

    def _zero_output(
        self,
        metadata: dict | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, float]:
        """Return zero content and zero bid when no audio is available."""
        workspace_dim = self.workspace_proj[0].out_features
        content = torch.zeros(batch_size, workspace_dim, device=device)
        self._last_content = content
        self._last_spatial = torch.zeros(batch_size, self.feature_dim, 2, device=device)
        self._last_affect = None
        self._last_tonotopic = None
        return content, 0.0

    def receive_broadcast(
        self, broadcast_content: torch.Tensor, current_bid: float
    ) -> float:
        """Reentrant feedback from Global Workspace broadcast.

        Computes prediction error between broadcast and last audio content.
        High PE -> audio was not the broadcast winner, increase bid.
        Low PE -> audio content matches broadcast, settle.

        Same pattern as SensoryTectum.receive_broadcast().
        """
        if self._last_content is None:
            return current_bid

        # Extract a tensor from broadcast_content. Workspace returns one of:
        # - raw torch.Tensor (legacy)
        # - dict with "_fused" key (Phase A attention-weighted fusion)
        # - dict with "tensor" key (legacy winner-take-all vision payload)
        # If none of those, no PE can be computed and we return current_bid.
        if isinstance(broadcast_content, torch.Tensor):
            bc_tensor = broadcast_content
        elif isinstance(broadcast_content, dict):
            bc_tensor = broadcast_content.get("_fused")
            if not isinstance(bc_tensor, torch.Tensor):
                bc_tensor = broadcast_content.get("tensor")
            if not isinstance(bc_tensor, torch.Tensor):
                return current_bid
        else:
            return current_bid

        # Align shapes: bc_tensor may be [1, D] or [D]; _last_content may differ.
        bc_flat = bc_tensor.detach().view(-1)
        last_flat = self._last_content.view(-1)
        if bc_flat.shape != last_flat.shape:
            n = min(bc_flat.shape[0], last_flat.shape[0])
            bc_flat = bc_flat[:n]
            last_flat = last_flat[:n]

        diff = torch.norm(bc_flat - last_flat)
        magnitude = torch.norm(last_flat) + 1e-8
        pe = (diff / magnitude).item()

        if pe > 0.3:
            # Audio not in the spotlight, increase bid for attention
            updated = min(1.0, current_bid + pe * 0.1)
        else:
            # Audio recognized in broadcast, settle
            updated = current_bid * (1.0 - pe * 0.1)

        return max(0.0, min(1.0, updated))

    def get_spatial_for_tectum(self) -> torch.Tensor:
        """Return cached spatial features [B, feature_dim, 2] for TopographicMap."""
        if self._last_spatial is None:
            return torch.zeros(1, self.feature_dim, 2)
        return self._last_spatial

    def get_affect_output(self) -> dict | None:
        """Return cached affect output (PAD deltas, paralinguistic, features)."""
        return self._last_affect

    def get_tonotopic_for_tectum(self, grid_size: int = 16) -> torch.Tensor | None:
        """Return tonotopic features reshaped for tectum grid.

        Returns [B, feature_dim, grid_size, grid_size] where frequency maps
        to elevation and azimuth is repeated.
        """
        if self._last_tonotopic is None:
            return None
        return self.encoder.reshape_for_tectum(self._last_tonotopic, grid_size)
