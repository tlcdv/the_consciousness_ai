"""
Attention Mechanism for Predictive Processing

This module implements:
1. Attention modulation for predictive processing
2. Integration with consciousness development
3. Stress-based attention gating
4. Dynamic attention allocation

Dependencies:
- models/core/consciousness_gating.py for gating control
- models/evaluation/consciousness_monitor.py for metrics
- models/memory/emotional_memory_core.py for context
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class AttentionMetrics:
    """Tracks attention mechanism metrics"""
    focus_level: float = 0.0
    stability_score: float = 0.0
    stress_modulation: float = 0.0
    emotional_weight: float = 0.0

@dataclass
class AttentionState:
    """Track attention mechanism state"""
    focus_level: float = 0.0
    emotional_coherence: float = 0.0
    memory_influence: float = 0.0
    narrative_alignment: float = 0.0
    adaptation_rate: float = 0.0

class PredictiveAttention(nn.Module):
    def __init__(self, config: dict):
        """Initialize predictive attention mechanism"""
        super().__init__()
        self.config = config
        self.metrics = AttentionMetrics()
        
        # Initialize attention components
        self.focus_network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        input_state: torch.Tensor,
        stress_level: float | None = None,
        emotional_context: dict | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Process input through attention mechanism"""
        # Calculate base attention
        attention = self.focus_network(input_state)
        
        # Apply stress modulation if provided
        if stress_level is not None:
            attention = self._modulate_attention(
                attention,
                stress_level
            )
            
        # Update metrics
        self.metrics.focus_level = attention.mean().item()
        self.metrics.stress_modulation = stress_level or 0.0
        
        return attention, self.metrics.__dict__

class ConsciousnessAttention(nn.Module):
    """
    Enhanced attention mechanism for consciousness development with:
    1. Stress-modulated attention
    2. Emotional context integration
    3. Temporal memory coherence
    4. Adaptive attention thresholds
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Core attention parameters
        self.hidden_size = config.get('hidden_size', 768)
        self.num_heads = config.get('num_heads', 12)
        self.dropout = config.get('dropout', 0.1)
        
        # Stress-attention coupling
        self.stress_sensitivity = nn.Parameter(
            torch.ones(1) * config.get('stress_sensitivity', 2.0)
        )
        self.attention_baseline = config.get('attention_baseline', 0.5)
        self.min_attention = config.get('min_attention', 0.2)
        
        # Multi-head attention components
        self.query_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.key_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Emotional context integration
        self.emotional_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Memory context integration
        self.memory_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # State tracking
        self.state = AttentionState()
        
    def forward(
        self,
        input_state: torch.Tensor,
        emotional_context: torch.Tensor,
        memory_context: torch.Tensor | None = None,
        stress_level: float | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Process input through enhanced attention mechanism"""
        
        batch_size = input_state.size(0)
        
        # Project inputs
        query = self.query_net(input_state)
        key = self.key_net(input_state)
        value = self.value_net(input_state)
        
        # Process emotional context
        if emotional_context is not None:
            emotional_features = self.emotional_projection(emotional_context)
            key = key + emotional_features
            value = value + emotional_features
            
        # Integrate memory context
        if memory_context is not None:
            memory_features = self.memory_projection(memory_context)
            key = torch.cat([key, memory_features], dim=1)
            value = torch.cat([value, memory_features], dim=1)
            
        # Calculate attention with temporal masking
        attention_output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value
        )
        
        # Calculate stress-modulated attention level
        if stress_level is not None:
            attention_level = self._calculate_attention_level(stress_level)
        else:
            attention_level = torch.sigmoid(attention_weights.mean())
            
        # Update attention state
        self._update_state(attention_level, emotional_context)
        
        # Project output with residual connection
        output = self.output_projection(
            torch.cat([attention_output, input_state], dim=-1)
        )
        
        return output, self._get_metrics()
        
    def _calculate_attention_level(self, stress_level: float) -> float:
        """Calculate attention level based on stress and adaptation"""
        # Base attention from stress
        base_attention = torch.sigmoid(
            self.stress_sensitivity * torch.tensor(stress_level)
        ).item()
        
        # Add adaptation factor
        adapted_attention = base_attention * (1.0 + self.state.stress_adaptation)
        
        # Ensure minimum attention
        return max(self.min_attention, adapted_attention)
        
    def _update_state(
        self,
        attention_level: float,
        emotional_context: torch.Tensor | None
    ):
        """Update attention state with temporal context"""
        # Update history
        self.state.history.append(attention_level)
        if len(self.state.history) > 1000:
            self.state.history = self.state.history[-1000:]
            
        # Update current level with decay
        self.state.current_level = (
            (1 - self.state.decay_rate) * self.state.current_level +
            self.state.decay_rate * attention_level
        )
        
        # Update baseline
        if len(self.state.history) > 100:
            self.state.baseline = np.mean(self.state.history[-100:])
            
        # Update stress adaptation
        self.state.stress_adaptation = self._calculate_stress_adaptation()
        
        # Update temporal coherence
        self.state.temporal_coherence = self._calculate_temporal_coherence()
        
    def _get_metrics(self) -> dict[str, float]:
        """Get current attention metrics"""
        return {
            'attention_level': self.state.current_level,
            'attention_baseline': self.state.baseline,
            'stress_adaptation': self.state.stress_adaptation,
            'temporal_coherence': self.state.temporal_coherence,
            'stability': self._calculate_stability()
        }
        
    def _calculate_stability(self) -> float:
        """Calculate attention stability"""
        if len(self.state.history) < 50:
            return 0.0
            
        recent_attention = self.state.history[-50:]
        return float(1.0 / (1.0 + np.std(recent_attention)))

class ConsciousnessAttention(nn.Module):
    def __init__(self, config=None):
        """Initialize attention mechanism"""
        super().__init__()
        if config is None:
            config = {}

        def _g(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        hidden_size = _g('hidden_size', 128)
        attention_dims = _g('attention_dims', 64)
        llama_hidden_size = _g('llama_hidden_size', 768)

        # Core attention components
        self.query_net = nn.Linear(hidden_size, attention_dims)
        self.key_net = nn.Linear(hidden_size, attention_dims)
        self.value_net = nn.Linear(hidden_size, hidden_size)

        # Meta-memory integration
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Narrative integration
        self.narrative_projection = nn.Linear(llama_hidden_size, hidden_size)
        
        # State tracking
        self.state = AttentionState()
        
    def forward(
        self,
        query: torch.Tensor = None,
        memory_context: dict | None = None,
        narrative_state: dict | None = None,
        emotional_context=None,
        input_state: torch.Tensor = None,
        environment_context=None,
    ) -> tuple[torch.Tensor, dict]:
        """Process attention with consciousness context"""
        # Support input_state as alias for query
        if query is None and input_state is not None:
            query = input_state

        # Ensure query has correct shape for the linear layers
        expected_dim = self.query_net.in_features
        if query.dim() == 1:
            query = query.unsqueeze(0)
        # If input dim doesn't match, project it
        if query.shape[-1] != expected_dim:
            query = torch.nn.functional.adaptive_avg_pool1d(
                query.unsqueeze(1), expected_dim
            ).squeeze(1)

        keys = self.key_net(query)
        values = self.value_net(query)

        # Apply memory-guided attention
        if memory_context:
            attention_weights = self._calculate_memory_attention(
                query, keys, memory_context
            )
        else:
            attention_weights = torch.matmul(
                self.query_net(query),
                keys.transpose(-2, -1)
            )

        # Generate output
        attention_output = torch.matmul(attention_weights, values)

        # Update state
        self._update_state(attention_weights, narrative_state, emotional_context)

        # Compute attention level for metrics
        base_attention = float(torch.sigmoid(attention_output.mean()).item())
        # Emotional arousal boosts attention (higher emotional intensity → more alert)
        emotional_boost = 0.0
        if emotional_context is not None:
            if isinstance(emotional_context, torch.Tensor):
                emotional_boost = float(emotional_context.abs().mean().item()) * 0.15
            elif isinstance(emotional_context, dict):
                emotional_boost = emotional_context.get('arousal', 0.0) * 0.15
        attention_level = min(1.0, base_attention + emotional_boost)
        metrics = {
            'attention_level': attention_level,
            'focus_level': self.state.focus_level,
            'emotional_coherence': self.state.emotional_coherence,
        }
        return attention_output, metrics
        
    def _calculate_memory_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        memory_context: dict
    ) -> torch.Tensor:
        """Calculate attention weights with memory guidance"""
        # Get memory influence
        memory_gate = self.memory_gate(
            torch.cat([query, memory_context['stable_patterns']], dim=-1)
        )
        
        # Calculate base attention
        base_attention = torch.matmul(
            self.query_net(query),
            keys.transpose(-2, -1)
        )
        
        # Apply memory gating
        return base_attention * memory_gate

    def _update_state(self, attention_weights, narrative_state=None, emotional_context=None):
        """Update internal attention state."""
        self.state.focus_level = float(torch.sigmoid(attention_weights.mean()).item())
        if emotional_context is not None and isinstance(emotional_context, torch.Tensor):
            self.state.emotional_coherence = float(torch.sigmoid(emotional_context.mean()).item())
