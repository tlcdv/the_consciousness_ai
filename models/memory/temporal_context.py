"""
Temporal Context Management for Memory Formation

This module implements:
1. Temporal sequence tracking and organization
2. Context-based memory retrieval
3. Time-based memory organization
4. Sequence coherence validation

Dependencies:
- models/memory/emotional_memory_core.py for memory storage
- models/memory/temporal_coherence.py for sequence tracking
- models/evaluation/memory_metrics.py for validation
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np

@dataclass
class TimeContext:
    """Tracks temporal context information"""
    timestamp: float
    sequence_id: str
    previous_contexts: list[str]
    next_contexts: list[str]
    attention_level: float

@dataclass
class TemporalMetrics:
    """Tracks temporal processing performance"""
    sequence_coherence: float = 0.0
    attention_stability: float = 0.0
    consolidation_quality: float = 0.0
    temporal_consistency: float = 0.0

class TemporalContextManager:
    def __init__(self, config: dict):
        """Initialize temporal context system"""
        self.config = config
        self.contexts = {}
        self.sequences = {}
        
    def add_context(
        self,
        memory_id: str,
        current_state: dict[str, torch.Tensor],
        attention: float
    ) -> TimeContext:
        """Add temporal context to memory"""
        # Create context
        context = TimeContext(
            timestamp=self._get_timestamp(),
            sequence_id=self._generate_sequence_id(),
            previous_contexts=self._get_previous_contexts(),
            next_contexts=[],
            attention_level=attention
        )
        
        # Update sequence tracking
        self._update_sequence_links(context)
        
        # Store context
        self.contexts[memory_id] = context
        
        return context

class TemporalContextNetwork(nn.Module):
    """
    Processes temporal context for memory formation and retrieval.
    Maintains temporal coherence in consciousness development.
    """

    def __init__(self, config: dict):
        super().__init__()
        
        # Core networks
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config['temporal_dim'],
                nhead=config['n_heads'],
                dim_feedforward=config['ff_dim']
            ),
            num_layers=config['n_layers']
        )
        
        self.time_embedding = nn.Linear(1, config['temporal_dim'])
        
        # Attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config['temporal_dim'],
            num_heads=config['n_heads']
        )
        
        self.metrics = TemporalMetrics()

    def forward(
        self,
        sequence: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Process temporal sequence with attention
        
        Args:
            sequence: Input sequence of states/events
            timestamps: Corresponding timestamps
            attention_mask: Optional attention mask
        """
        # Generate time embeddings
        time_embeddings = self.time_embedding(timestamps.unsqueeze(-1))
        
        # Add temporal embeddings to sequence
        sequence = sequence + time_embeddings
        
        # Process through transformer
        encoded_sequence = self.temporal_encoder(
            sequence,
            src_key_padding_mask=attention_mask
        )
        
        # Apply temporal attention
        attended_sequence, attention_weights = self.temporal_attention(
            encoded_sequence,
            encoded_sequence,
            encoded_sequence,
            key_padding_mask=attention_mask
        )
        
        # Update metrics
        self._update_metrics(
            sequence=sequence,
            attention_weights=attention_weights,
            timestamps=timestamps
        )
        
        return attended_sequence, self.get_metrics()

    def _update_metrics(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """Update temporal processing metrics"""
        # Calculate sequence coherence
        self.metrics.sequence_coherence = self._calculate_sequence_coherence(sequence)
        
        # Calculate attention stability
        self.metrics.attention_stability = self._calculate_attention_stability(
            attention_weights
        )
        
        # Calculate consolidation quality
        self.metrics.consolidation_quality = self._calculate_consolidation_quality(
            sequence,
            timestamps
        )
        
        # Calculate temporal consistency
        self.metrics.temporal_consistency = self._calculate_temporal_consistency(
            sequence,
            timestamps
        )

    def _calculate_sequence_coherence(self, sequence: torch.Tensor) -> float:
        """Calculate coherence between sequential states"""
        coherence_scores = []
        for i in range(sequence.size(0) - 1):
            score = torch.cosine_similarity(
                sequence[i:i+1],
                sequence[i+1:i+2],
                dim=-1
            )
            coherence_scores.append(score.item())
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0