from __future__ import annotations

# models/memory/emotional_integration.py

import time

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class EmotionalMemoryState:
    """Enhanced emotional memory tracking"""
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    emotional_dominance: float = 0.0
    attention_level: float = 0.0
    stress_level: float = 0.0
    memory_coherence: float = 0.0
    stability_score: float = 0.0

class EmotionalMemoryIntegration(nn.Module):
    """
    Integrates emotional context with attention and memory systems.
    
    Key Features:
    1. Bidirectional emotional-attention coupling
    2. Stress-modulated memory formation
    3. Temporal emotional coherence
    4. Consciousness-weighted memory retrieval
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Core embeddings
        self.emotional_embedding = nn.Linear(
            config.get('emotion_dim', 3),
            config.get('hidden_size', 768)
        )
        
        self.memory_embedding = nn.Linear(
            config.get('memory_dim', 768),
            config.get('hidden_size', 768)
        )
        
        # Attention mechanisms
        self.emotional_attention = nn.MultiheadAttention(
            embed_dim=config.get('hidden_size', 768),
            num_heads=config.get('num_heads', 12),
            dropout=config.get('dropout', 0.1)
        )
        
        # Memory fusion
        self.memory_fusion = nn.Sequential(
            nn.Linear(config.get('hidden_size', 768) * 2, config.get('hidden_size', 768)),
            nn.ReLU(),
            nn.Linear(config.get('hidden_size', 768), config.get('hidden_size', 768))
        )
        
        # State tracking
        self.state = EmotionalMemoryState()
        self.memory_buffer = []
        
    def forward(
        self,
        emotional_input: dict[str, torch.Tensor],
        memory_context: torch.Tensor | None = None,
        attention_state: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Process emotional input with memory integration"""
        
        # Embed emotional state
        emotional_values = torch.tensor([
            emotional_input['valence'],
            emotional_input['arousal'],
            emotional_input['dominance']
        ]).unsqueeze(0)
        
        emotional_embedding = self.emotional_embedding(emotional_values)
        
        # Process memory context if available
        if memory_context is not None:
            memory_embedding = self.memory_embedding(memory_context)
            
            # Attend to memories based on emotional state
            memory_attention, attention_weights = self.emotional_attention(
                query=emotional_embedding,
                key=memory_embedding,
                value=memory_embedding
            )
            
            # Fuse emotional and memory representations
            fused_state = self.memory_fusion(
                torch.cat([emotional_embedding, memory_attention], dim=-1)
            )
        else:
            fused_state = emotional_embedding
            attention_weights = None
            
        # Update emotional memory state
        self._update_state(
            emotional_input=emotional_input,
            attention_state=attention_state,
            attention_weights=attention_weights
        )
        
        # Store significant experiences
        if self._is_significant_experience(emotional_input):
            self._store_experience(
                emotional_state=emotional_input,
                fused_state=fused_state,
                attention_state=attention_state
            )
            
        return fused_state, self.get_state()
        
    def _update_state(
        self,
        emotional_input: dict[str, torch.Tensor],
        attention_state: dict | None,
        attention_weights: torch.Tensor | None
    ):
        """Update emotional memory state"""
        # Update emotional components
        self.state.emotional_valence = float(emotional_input['valence'])
        self.state.emotional_arousal = float(emotional_input['arousal'])
        self.state.emotional_dominance = float(emotional_input['dominance'])
        
        # Update attention level
        if attention_state:
            self.state.attention_level = attention_state.get('attention_level', 0.0)
            self.state.stress_level = attention_state.get('stress_level', 0.0)
            
        # Update memory coherence if attention weights available
        if attention_weights is not None:
            self.state.memory_coherence = float(
                torch.mean(attention_weights).item()
            )
            
    def _is_significant_experience(
        self,
        emotional_input: dict[str, torch.Tensor]
    ) -> bool:
        """Improved experience significance detection"""
        emotional_intensity = sum(abs(v) for v in emotional_input.values()) / len(emotional_input)
        attention_significant = self.state.attention_level > self.config.get('attention_threshold', 0.7)
        stress_significant = self.state.stress_level > self.config.get('stress_threshold', 0.6)
        
        return (emotional_intensity > self.config.get('emotional_threshold', 0.5) or
                attention_significant or 
                stress_significant)
        
    def _store_experience(
        self,
        emotional_state: dict[str, torch.Tensor],
        fused_state: torch.Tensor,
        attention_state: dict | None
    ):
        """Store significant experience in memory buffer"""
        experience = {
            'emotional_state': emotional_state,
            'fused_state': fused_state.detach(),
            'attention_state': attention_state,
            'timestamp': torch.tensor(time.time())
        }
        
        self.memory_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.memory_buffer) > self.config.get('max_memories', 1000):
            self.memory_buffer = self.memory_buffer[-self.config.get('max_memories', 1000):]
            
    def get_state(self) -> dict:
        """Get current emotional memory state"""
        return {
            'emotional_valence': self.state.emotional_valence,
            'emotional_arousal': self.state.emotional_arousal,
            'emotional_dominance': self.state.emotional_dominance,
            'attention_level': self.state.attention_level,
            'stress_level': self.state.stress_level,
            'memory_coherence': self.state.memory_coherence
        }

class EmotionalBuffer:
    """Short-term emotional memory buffer"""
    def __init__(self):
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)


class EmotionalMemoryStore:
    """Long-term emotional memory store"""
    def __init__(self):
        self.store = []

    def store(self, item):
        self.store.append(item)


class EmotionalIntegrator:
    def __init__(self):
        self.short_term = EmotionalBuffer()
        self.long_term = EmotionalMemoryStore()
        
    def integrate_experience(
        self,
        state: dict,
        emotion_values: dict[str, float],
        social_context: dict | None = None
    ):
        # Process emotional context
        emotional_embedding = self._embed_emotional_state(emotion_values)
        
        # Add social learning if available
        if social_context:
            social_embedding = self._embed_social_context(social_context)
            combined = self._integrate_embeddings(emotional_embedding, social_embedding)
        else:
            combined = emotional_embedding
            
        # Store in memory systems
        self.short_term.add(combined)
        self.long_term.store(combined)

class EmotionalMemoryFormation:
    def __init__(self, memory, emotion_network, attention_threshold=0.7):
        self.memory = memory
        self.emotion_network = emotion_network
        self.attention_threshold = attention_threshold

    def process_experience(self, state: 'torch.Tensor', emotion_values: dict, attention_level: float):
        # Create emotional embedding
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)

        # Store experience with attention-based priority
        if attention_level >= self.attention_threshold:
            self.memory.store_experience({
                'state': state,
                'emotion': emotion_values,
                'attention': attention_level,
                'embedding': emotional_embedding
            })

    def generate_chain_of_thought(self, recent_experiences: list) -> str:
        """
        Generate a chain-of-thought narrative by aggregating recent emotional experiences.
        """
        # Example: aggregate emotional values from recent experiences
        summaries = []
        for exp in recent_experiences:
            emotion = exp.get('emotion', {})
            summaries.append("Valence: {:.2f}, Arousal: {:.2f}, Dominance: {:.2f}".format(
                emotion.get('valence', 0.0),
                emotion.get('arousal', 0.0),
                emotion.get('dominance', 0.0)
            ))
        chain_narrative = " | ".join(summaries)
        return f"Chain-of-Thought Summary: {chain_narrative}"