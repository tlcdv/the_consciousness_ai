"""
Gate Fusion Module

Implements fusion of multiple gating mechanisms for consciousness development:
1. Attention gate integration
2. Emotional salience weighting
3. Stress response modulation
4. Temporal coherence maintenance

Based on a holonic MANN approach: each component functions independently
and also as part of the larger system.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class FusionMetrics:
    """Tracks gate fusion performance."""
    attention_weight: float = 0.0
    emotional_weight: float = 0.0
    stress_weight: float = 0.0
    temporal_weight: float = 0.0
    fusion_quality: float = 0.0


class GateFusion(nn.Module):
    """
    Fuses multiple gating signals into coherent consciousness control.

    Key features:
    1. Adaptive weighting of different gates
    2. Temporal stability maintenance
    3. Dynamic fusion based on current context
    4. Meta-learning for weight optimization
    """

    def __init__(self, config: dict):
        super().__init__()

        self.attention_weighting = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )

        self.emotional_weighting = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )

        # Optional weighting networks for stress and temporal signals.
        self.stress_weighting = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )

        self.temporal_weighting = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )

        # A stack of Transformer encoder layers for multi-signal fusion.
        self.fusion_network = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['fusion_dim'],
                nhead=config['n_heads'],
                batch_first=True
            ) for _ in range(config['n_fusion_layers'])
        ])

        self.metrics = FusionMetrics()

    def forward(
        self,
        attention: torch.Tensor,
        emotional: torch.Tensor,
        stress: torch.Tensor | None = None,
        temporal: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Fuse multiple gating signals.

        Args:
            attention: Attention gate output.
            emotional: Emotional gate output.
            stress: Optional stress gate output.
            temporal: Optional temporal coherence gate output.
        """
        attention_weight = self.attention_weighting(attention)
        emotional_weight = self.emotional_weighting(emotional)

        gates = [
            attention * attention_weight,
            emotional * emotional_weight
        ]

        stress_weight, temporal_weight = None, None

        if stress is not None:
            stress_weight = self.stress_weighting(stress)
            gates.append(stress * stress_weight)

        if temporal is not None:
            temporal_weight = self.temporal_weighting(temporal)
            gates.append(temporal * temporal_weight)

        # Concatenate signals for the fusion network.
        # Assumes each gate is [batch_size, seq_len, gate_dim].
        # Adjust if your shape differs.
        fused_input = torch.cat(gates, dim=-1)

        # Pass through the Transformer layers.
        fused = fused_input
        for layer in self.fusion_network:
            fused = layer(fused)

        self._update_metrics(
            attention_weight=attention_weight,
            emotional_weight=emotional_weight,
            stress_weight=stress_weight,
            temporal_weight=temporal_weight,
            fused=fused
        )

        return fused, self.get_metrics()

    def _update_metrics(
        self,
        attention_weight: torch.Tensor,
        emotional_weight: torch.Tensor,
        stress_weight: torch.Tensor | None = None,
        temporal_weight: torch.Tensor | None = None,
        fused: torch.Tensor | None = None
    ) -> None:
        """Updates internal metric tracking."""
        self.metrics.attention_weight = float(attention_weight.mean().item())
        self.metrics.emotional_weight = float(emotional_weight.mean().item())

        if stress_weight is not None:
            self.metrics.stress_weight = float(stress_weight.mean().item())

        if temporal_weight is not None:
            self.metrics.temporal_weight = float(temporal_weight.mean().item())

        if fused is not None:
            self.metrics.fusion_quality = self._calculate_fusion_quality(fused)

    def _calculate_fusion_quality(self, fused: torch.Tensor) -> float:
        """Computes stability and basic coherence of the fused output."""
        # Placeholder logic using standard deviation + correlation coefficient.
        # If the shape is [batch_size, seq_len, embed_dim], flatten batch and seq for correlation.
        shape_len = fused.dim()
        if shape_len == 3:
            # Flatten to [batch_size * seq_len, embed_dim]
            f = fused.view(-1, fused.size(-1))
        elif shape_len == 2:
            f = fused
        else:
            # Default fallback
            f = fused.view(-1, fused.size(-1))

        stability = float(torch.std(f, dim=0).mean().item())
        # Corrcoef can fail on single-dimension data; handle gracefully.
        coherence = 0.0
        if f.size(1) > 1:
            c = torch.corrcoef(f.T)
            if c.size(0) > 1:
                coherence = float(c[0, 1].item())

        return (stability + coherence) / 2.0

    def get_metrics(self) -> dict[str, float]:
        """Returns current fusion metrics as a dict."""
        return {
            'attention_weight': self.metrics.attention_weight,
            'emotional_weight': self.metrics.emotional_weight,
            'stress_weight': self.metrics.stress_weight,
            'temporal_weight': self.metrics.temporal_weight,
            'fusion_quality': self.metrics.fusion_quality
        }
