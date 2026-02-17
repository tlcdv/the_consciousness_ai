"""
Enhanced Emotional Context Processing

Implements advanced emotional processing for memory formation:
1. Multi-dimensional emotional representation
2. Temporal emotional coherence
3. Social context integration
4. Meta-emotional learning

Based on MANN architecture principles.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class MetaEmotionalNetwork(nn.Module):
    """Meta-emotional learning network"""
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_dim', 64)
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, embedding, state):
        return self.linear(embedding)


class EnhancedEmotionalContext(nn.Module):
    """
    Processes enhanced emotional context for memory formation
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Emotional embedding networks
        self.primary_emotion_encoder = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU()
        )
        
        self.social_context_encoder = nn.Sequential(
            nn.Linear(config['social_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU()
        )
        
        # Temporal processing
        self.temporal_emotion = nn.GRU(
            input_size=config['hidden_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=config['n_layers']
        )
        
        # Meta-emotional learning
        self.meta_emotional = MetaEmotionalNetwork(config)

    def forward(
        self,
        emotional_state: Dict[str, float],
        social_context: Optional[Dict] = None,
        temporal_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Process emotional context with temporal coherence"""
        
        # Encode primary emotions
        emotion_embedding = self.primary_emotion_encoder(
            torch.tensor([v for v in emotional_state.values()])
        )
        
        # Integrate social context if available
        if social_context is not None:
            social_embedding = self.social_context_encoder(
                torch.tensor([v for v in social_context.values()])
            )
            emotion_embedding = emotion_embedding + social_embedding
            
        # Process temporal context if available
        if temporal_history is not None:
            temporal_features, _ = self.temporal_emotion(
                temporal_history
            )
            emotion_embedding = emotion_embedding + temporal_features[-1]
            
        # Update meta-emotional learning
        meta_features = self.meta_emotional(
            emotion_embedding,
            emotional_state
        )
        
        return emotion_embedding + meta_features