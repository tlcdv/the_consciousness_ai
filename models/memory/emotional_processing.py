"""
Enhanced Emotional Processing Module

Implements advanced emotional processing features:
1. Multi-dimensional emotion representation
2. Social context integration
3. Meta-emotional learning
4. Temporal emotion tracking

Based on MANN architecture for holonic consciousness development.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

class MetaEmotionalLearner(nn.Module):
    """Meta-learning for emotional processing adaptation"""
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_dim', 64)
        embed = config.get('embedding_dim', 32)
        self.net = nn.Linear(embed, embed)
        self.progress = 0.0

    def update(self, embedding, state):
        self.progress = min(1.0, self.progress + 0.001)
        return self.net(embedding)

    def get_progress(self):
        return self.progress


@dataclass
class EmotionalProcessingMetrics:
    """Tracks emotional processing performance"""
    emotional_stability: float = 0.0
    social_coherence: float = 0.0
    temporal_consistency: float = 0.0
    meta_learning_progress: float = 0.0

class EmotionalProcessingCore(nn.Module):
    """
    Implements advanced emotional processing and integration
    """

    def __init__(self, config: Dict):
        super().__init__()
        
        # Primary emotion processing
        self.emotion_encoder = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['embedding_dim'])
        )
        
        # Social context processing
        self.social_encoder = nn.Sequential(
            nn.Linear(config['social_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['embedding_dim'])
        )
        
        # Temporal processing
        self.temporal_processor = nn.GRU(
            input_size=config['embedding_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['n_layers']
        )
        
        # Meta-learning components
        self.meta_learner = MetaEmotionalLearner(config)
        
        self.metrics = EmotionalProcessingMetrics()

    def process_emotion(
        self,
        emotional_state: Dict[str, float],
        social_context: Optional[Dict] = None,
        temporal_history: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process emotional input with social and temporal context
        """
        # Encode primary emotions
        emotion_embedding = self.emotion_encoder(
            torch.tensor([v for v in emotional_state.values()])
        )
        
        # Process social context if available
        if social_context:
            social_embedding = self.social_encoder(
                torch.tensor([v for v in social_context.values()])
            )
            emotion_embedding = self._integrate_social_context(
                emotion_embedding, 
                social_embedding
            )
            
        # Process temporal context if available
        if temporal_history:
            temporal_embedding = self._process_temporal_context(temporal_history)
            emotion_embedding = self._integrate_temporal_context(
                emotion_embedding,
                temporal_embedding
            )
            
        # Update meta-learning
        meta_features = self.meta_learner.update(
            emotion_embedding,
            emotional_state
        )
        
        # Update metrics
        self._update_metrics(
            emotional_state=emotional_state,
            social_context=social_context,
            temporal_history=temporal_history
        )
        
        return emotion_embedding + meta_features, self.get_metrics()

    def _update_metrics(
        self,
        emotional_state: Dict[str, float],
        social_context: Optional[Dict],
        temporal_history: Optional[List[Dict]]
    ):
        """Update emotional processing metrics"""
        self.metrics.emotional_stability = self._calculate_emotional_stability(
            emotional_state
        )
        
        if social_context:
            self.metrics.social_coherence = self._calculate_social_coherence(
                emotional_state,
                social_context
            )
            
        if temporal_history:
            self.metrics.temporal_consistency = self._calculate_temporal_consistency(
                temporal_history
            )
            
        self.metrics.meta_learning_progress = self.meta_learner.get_progress()