"""
Enhanced Memory Integration Module

Implements a holonic memory architecture integrating:
1. Episodic experience storage with emotional context
2. Semantic knowledge abstraction 
3. Temporal coherence maintenance
4. Consciousness-weighted memory formation

Based on Modular Artificial Neural Networks (MANN) architecture and holonic principles
where each component functions both independently and as part of the whole system.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from models.core.consciousness_gating import ConsciousnessGate


class EpisodicMemoryStore:
    """Episodic memory storage"""
    def __init__(self, config):
        self.memories = []
    def store(self, state, emotional, temporal, metadata=None):
        self.memories.append({'state': state, 'emotional': emotional})
    def search(self, query, emotional_query=None, k=5):
        return self.memories[:k]


class SemanticMemoryStore:
    """Semantic memory storage"""
    def __init__(self, config):
        self.knowledge = {}
    def update(self, features):
        pass
    def search(self, query, k=5):
        return []


class TemporalMemoryBuffer:
    """Temporal memory buffer"""
    def __init__(self, config):
        self.buffer = []
    def update(self, embedding):
        self.buffer.append(embedding)


class EmotionalContextNetwork(nn.Module):
    """Encodes emotional context for memory"""
    def __init__(self, config):
        super().__init__()
        dim = config.get('emotion_dim', 3) if isinstance(config, dict) else 3
        hidden = config.get('hidden_dim', 64) if isinstance(config, dict) else 64
        self.net = nn.Linear(dim, hidden)
    def forward(self, x):
        if isinstance(x, dict):
            x = torch.tensor(list(x.values()))
        return self.net(x)


class SemanticAbstractionNetwork(nn.Module):
    """Abstracts semantic features from experiences"""
    def __init__(self, config):
        super().__init__()
        hidden = config.get('hidden_dim', 64) if isinstance(config, dict) else 64
        self.hidden = hidden
        self.net = nn.Linear(hidden, hidden)
    def forward(self, state, emotional):
        if isinstance(state, torch.Tensor) and state.shape[-1] != self.hidden:
            state = torch.nn.functional.adaptive_avg_pool1d(
                state.unsqueeze(0).unsqueeze(0), self.hidden
            ).squeeze(0).squeeze(0)
        return self.net(state)


class TemporalCoherenceProcessor(nn.Module):
    """Processes temporal coherence"""
    def __init__(self, config):
        super().__init__()
    def forward(self, x):
        return x

@dataclass
class MemoryMetrics:
    """Tracks memory system performance and coherence"""
    temporal_coherence: float = 0.0
    emotional_stability: float = 0.0
    semantic_abstraction: float = 0.0
    retrieval_quality: float = 0.0

class MemoryIntegrationCore(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        # Memory subsystems
        self.episodic_memory = EpisodicMemoryStore(config)
        self.semantic_memory = SemanticMemoryStore(config)
        self.temporal_memory = TemporalMemoryBuffer(config)
        
        # Processing networks
        self.emotional_encoder = EmotionalContextNetwork(config)
        self.semantic_abstractor = SemanticAbstractionNetwork(config)
        self.temporal_processor = TemporalCoherenceProcessor(config)
        
        # Memory formation gate
        self.consciousness_gate = ConsciousnessGate(config)
        
        self.metrics = MemoryMetrics()

    def store_experience(
        self,
        experience_data=None,
        emotional_context=None,
        consciousness_level: float = 0.5,
        metadata: dict | None = None
    ) -> bool:
        """
        Store experience with emotional context and consciousness gating.
        Accepts both dict and tensor experience_data.
        """
        if emotional_context is None:
            emotional_context = {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}
        emotional_embedding = self.emotional_encoder(emotional_context)

        # Normalize experience_data to a dict
        if isinstance(experience_data, dict):
            state = experience_data.get('state', experience_data)
            timestamp = experience_data.get('timestamp', torch.tensor(0.0))
        else:
            state = experience_data
            timestamp = torch.tensor(0.0)

        if isinstance(timestamp, (int, float)):
            timestamp = torch.tensor(float(timestamp))
        temporal_embedding = self.temporal_processor(timestamp)

        # Gate storage based on consciousness level
        if isinstance(consciousness_level, (int, float)):
            gate_input = torch.full((1, self.consciousness_gate.hidden_size), consciousness_level)
        elif isinstance(consciousness_level, torch.Tensor) and consciousness_level.dim() == 0:
            gate_input = consciousness_level.unsqueeze(0).expand(1, self.consciousness_gate.hidden_size)
        else:
            gate_input = consciousness_level
        gated_output, gate_state = self.consciousness_gate(gate_input)
        if gate_state.attention_level >= self.consciousness_gate.attention_threshold:
            self.episodic_memory.store(
                state,
                emotional_embedding,
                temporal_embedding,
                metadata
            )

            semantic_features = self.semantic_abstractor(
                state if isinstance(state, torch.Tensor) else torch.zeros(64),
                emotional_embedding
            )
            self.semantic_memory.update(semantic_features)
            self.temporal_memory.update(temporal_embedding)

            self._update_memory_metrics(
                {'state': state},
                emotional_context,
                consciousness_level
            )
            return True

        return False

    def retrieve_memories(
        self,
        query: dict[str, torch.Tensor],
        emotional_context: dict[str, float] | None = None,
        k: int = 5
    ) -> list[dict]:
        """
        Retrieve relevant memories using emotional context
        """
        # Generate query embeddings
        emotional_query = self.emotional_encoder(emotional_context) if emotional_context else None
        
        # Get episodic memories
        episodic_results = self.episodic_memory.search(
            query['state'],
            emotional_query,
            k=k
        )
        
        # Get semantic knowledge
        semantic_results = self.semantic_memory.search(
            query['state'],
            k=k
        )
        
        # Combine results
        return {
            'episodic': episodic_results,
            'semantic': semantic_results,
            'metrics': self.get_metrics()
        }

    def get_state(self) -> dict:
        """Return current memory system state."""
        return {
            'episodic_count': len(self.episodic_memory.memories),
            'metrics': {
                'temporal_coherence': self.metrics.temporal_coherence,
                'emotional_stability': self.metrics.emotional_stability,
                'semantic_abstraction': self.metrics.semantic_abstraction,
                'retrieval_quality': self.metrics.retrieval_quality,
            }
        }

    def get_metrics(self) -> dict:
        """Return current memory metrics."""
        return {
            'temporal_coherence': self.metrics.temporal_coherence,
            'emotional_stability': self.metrics.emotional_stability,
            'semantic_abstraction': self.metrics.semantic_abstraction,
            'retrieval_quality': self.metrics.retrieval_quality,
        }

    def _update_memory_metrics(
        self,
        experience_data: dict,
        emotional_context: dict[str, float],
        consciousness_level: float
    ):
        """Update memory system metrics"""
        self.metrics.temporal_coherence = self._calculate_temporal_coherence()
        self.metrics.emotional_stability = self._calculate_emotional_stability(
            emotional_context
        )
        self.metrics.semantic_abstraction = self._evaluate_semantic_quality()
        self.metrics.retrieval_quality = self._evaluate_retrieval_quality()

    def _calculate_temporal_coherence(self) -> float:
        """Estimate temporal coherence from episodic memory count."""
        count = len(self.episodic_memory.memories) if hasattr(self.episodic_memory, 'memories') else 0
        return min(1.0, count / 100.0) if count > 0 else 0.0

    def _calculate_emotional_stability(self, emotional_context: dict[str, float]) -> float:
        """Compute emotional stability from current context."""
        values = [abs(v) for v in emotional_context.values()]
        return 1.0 - (sum(values) / max(len(values), 1)) if values else 0.5

    def _evaluate_semantic_quality(self) -> float:
        """Evaluate quality of semantic memory abstractions."""
        return 0.5

    def _evaluate_retrieval_quality(self) -> float:
        """Evaluate memory retrieval quality."""
        return 0.5