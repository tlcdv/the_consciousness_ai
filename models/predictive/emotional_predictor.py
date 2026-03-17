from __future__ import annotations

# models/predictive/emotional_predictor.py

"""
Predictive module that handles:
- Emotional outcome prediction
- Simulation evaluation
- Meta-memory integration
- Stability monitoring
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from models.core.consciousness_core import ConsciousnessState
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.emotional_memory_core import EmotionalMemoryCore

@dataclass
class EmotionalState:
    """Tracks emotional state development"""
    valence: float = 0.0  # Pleasure-displeasure
    arousal: float = 0.0  # Energy level
    dominance: float = 0.0  # Control level
    stress_level: float = 0.0
    attention_focus: float = 0.0
    emotional_stability: float = 0.0

@dataclass
class PredictionMetrics:
    """Track prediction system performance"""
    accuracy: float = 0.0
    confidence: float = 0.0
    stability: float = 0.0
    adaptation_rate: float = 0.0
    meta_memory_influence: float = 0.0

class EmotionalPredictor(nn.Module):
    """
    Predicts emotional development and stress responses
    
    Key Features:
    1. Multimodal emotion prediction
    2. Stress-induced attention modulation
    3. Temporal emotional stability tracking
    4. Consciousness-weighted predictions
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Core parameters
        self.hidden_size = config.get('hidden_size', 768)
        self.num_emotions = config.get('num_emotions', 3)  # VAD dimensions
        self.num_heads = config.get('num_heads', 8)
        
        # Neural components
        self.emotional_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Attention for temporal context
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # Emotion prediction heads
        self.valence_head = nn.Linear(self.hidden_size, 1)
        self.arousal_head = nn.Linear(self.hidden_size, 1)
        self.dominance_head = nn.Linear(self.hidden_size, 1)
        
        # Stress prediction
        self.stress_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        )
        
        # State tracking
        self.state = EmotionalState()
        self.history: list[EmotionalState] = []
        
        # Core components
        self.emotional_graph = EmotionalGraphNetwork()
        self.memory_core = EmotionalMemoryCore(config)
        
        # Prediction networks
        self.outcome_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_emotions)
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Metrics tracking
        self.metrics = PredictionMetrics()
        
    def forward(
        self,
        input_state: torch.Tensor,
        attention_context: torch.Tensor | None = None,
        memory_context: torch.Tensor | None = None,
        meta_memory_context: dict | None = None,
        consciousness_state: ConsciousnessState | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Process input state for emotional predictions"""
        
        # Encode emotional features
        emotional_features = self.emotional_encoder(input_state)
        
        # Apply temporal attention if context available
        if attention_context is not None:
            emotional_features, _ = self.temporal_attention(
                query=emotional_features,
                key=attention_context,
                value=attention_context
            )
            
        # Predict emotional dimensions (VAD)
        valence = torch.sigmoid(self.valence_head(emotional_features))
        arousal = torch.sigmoid(self.arousal_head(emotional_features))
        dominance = torch.sigmoid(self.dominance_head(emotional_features))
        
        # Calculate stress level
        stress_input = torch.cat([
            emotional_features,
            memory_context if memory_context is not None else torch.zeros_like(emotional_features)
        ], dim=-1)
        stress_level = torch.sigmoid(self.stress_predictor(stress_input))
        
        # Update emotional state
        self._update_state(
            valence=valence.mean().item(),
            arousal=arousal.mean().item(),
            dominance=dominance.mean().item(),
            stress_level=stress_level.mean().item()
        )
        
        predictions = {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'stress_level': stress_level
        }
        
        # Get emotional embedding
        emotional_embedding = self.emotional_graph(
            input_state,
            meta_memory_context['stable_patterns'] if meta_memory_context else None
        )
        
        # Generate outcome prediction
        predicted_outcome = self.outcome_predictor(emotional_embedding)
        
        # Calculate confidence score
        confidence = self.confidence_predictor(emotional_embedding)
        
        # Update metrics
        self._update_metrics(
            predicted_outcome,
            confidence,
            meta_memory_context,
            consciousness_state
        )
        
        metrics = self.get_metrics()
        
        return predictions, metrics
        
    def _update_state(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        stress_level: float
    ):
        """Update emotional state tracking"""
        # Update current state
        self.state.valence = valence
        self.state.arousal = arousal
        self.state.dominance = dominance
        self.state.stress_level = stress_level
        
        # Calculate stability
        self.state.emotional_stability = self._calculate_stability()
        
        # Calculate attention focus from arousal and stress
        self.state.attention_focus = self._calculate_attention_focus(
            arousal=arousal,
            stress_level=stress_level
        )
        
        # Store state
        self.history.append(EmotionalState(**vars(self.state)))
        
        # Maintain history size
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
            
    def _calculate_stability(self) -> float:
        """Calculate emotional stability from history"""
        if len(self.history) < 2:
            return 1.0
            
        # Calculate variance of emotional dimensions
        recent_states = self.history[-100:]
        valence_var = np.var([s.valence for s in recent_states])
        arousal_var = np.var([s.arousal for s in recent_states])
        dominance_var = np.var([s.dominance for s in recent_states])
        
        # Higher stability = lower variance
        return float(1.0 / (1.0 + (valence_var + arousal_var + dominance_var) / 3))
        
    def _calculate_attention_focus(
        self,
        arousal: float,
        stress_level: float
    ) -> float:
        """Calculate attention focus level"""
        # Attention increases with both arousal and stress
        base_attention = (arousal + stress_level) / 2
        
        # Modulate by stability
        return float(base_attention * (1.0 + self.state.emotional_stability))
        
    def get_metrics(self) -> dict[str, float]:
        """Get current emotional metrics"""
        return {
            'valence': self.state.valence,
            'arousal': self.state.arousal,
            'dominance': self.state.dominance,
            'stress_level': self.state.stress_level,
            'attention_focus': self.state.attention_focus,
            'emotional_stability': self.state.emotional_stability,
            'confidence': self.metrics.confidence,
            'stability': self.metrics.stability,
            'adaptation_rate': self.metrics.adaptation_rate,
            'meta_memory_influence': self.metrics.meta_memory_influence
        }
        
    def _update_metrics(
        self,
        prediction: torch.Tensor,
        confidence: torch.Tensor,
        meta_memory_context: dict | None,
        consciousness_state: ConsciousnessState | None
    ):
        """Update prediction metrics"""
        self.metrics.confidence = confidence.mean().item()
        
        if meta_memory_context:
            self.metrics.meta_memory_influence = self._calculate_memory_influence(
                prediction,
                meta_memory_context
            )
            
        if consciousness_state:
            self.metrics.stability = consciousness_state.memory_stability
            self.metrics.adaptation_rate = self._calculate_adaptation_rate(
                confidence,
                consciousness_state
            )