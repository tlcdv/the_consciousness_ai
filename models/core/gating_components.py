"""
Gating Component Networks

Implements specialized gating mechanisms for different aspects of consciousness:
1. Attention-based gating
2. Emotional salience gating
3. Stress response gating
4. Temporal coherence gating

Each component functions both independently and as part of the system.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Gates information based on attention levels."""

    def __init__(self, config: dict):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['state_dim']),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, attention_level: float) -> torch.Tensor:
        """Applies attention-based gating to x."""
        gate_values = self.attention_net(x)
        return x * gate_values * attention_level


class EmotionalGate(nn.Module):
    """Gates information based on emotional salience."""

    def __init__(self, config: dict):
        super().__init__()
        self.emotion_encoder = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['state_dim']),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, emotional_context: dict[str, float]) -> torch.Tensor:
        """Encodes emotional context into gate values and applies them to x."""
        # Build tensor from emotional context.
        keys = sorted(emotional_context.keys())
        emotion_tensor = torch.tensor(
            [emotional_context[k] for k in keys],
            dtype=x.dtype,
            device=x.device
        ).unsqueeze(0)  # Shape [1, emotion_dim] for batch dimension if needed.

        gate_values = self.emotion_encoder(emotion_tensor)
        # Expand gate_values to match x if necessary.
        if gate_values.dim() == 2 and x.dim() == 2:
            # If x is [batch_size, state_dim], replicate gate_values across batch.
            gate_values = gate_values.repeat(x.size(0), 1)

        return x * gate_values


class TemporalCoherenceGate(nn.Module):
    """Gates information based on temporal consistency."""

    def __init__(self, config: dict):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config['state_dim'],
            num_heads=config['n_heads'],
            batch_first=True
        )
        self.gate_net = nn.Sequential(
            nn.Linear(config['state_dim'], config['state_dim']),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, temporal_context: torch.Tensor) -> torch.Tensor:
        """
        Applies temporal attention to x using temporal_context, then gates
        with the resulting features.
        """
        # x, temporal_context shapes assumed: [batch_size, seq_len, state_dim].
        # Adjust if different.
        attended_features, _ = self.temporal_attention(
            x,
            temporal_context,
            temporal_context
        )
        gate_values = self.gate_net(attended_features)
        return x * gate_values
