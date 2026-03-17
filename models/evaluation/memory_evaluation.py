"""
Memory Evaluation Functions

Implements comprehensive memory system evaluation through:
1. Coherence metrics calculation
2. Temporal consistency analysis
3. Emotional relevance assessment
4. Consciousness integration measurement

Based on the holonic principles where each metric contributes both 
independently and to the overall system evaluation.
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    """Comprehensive memory evaluation metrics"""
    coherence_score: float = 0.0
    temporal_stability: float = 0.0
    emotional_relevance: float = 0.0
    consciousness_integration: float = 0.0

class MemoryEvaluator:
    """
    Evaluates memory system performance across multiple dimensions
    """

    def __init__(self, config: dict):
        self.config = config
        self.metrics = EvaluationMetrics()

    def evaluate_memory_system(
        self,
        recent_memories: list[dict],
        emotional_context: dict[str, float],
        consciousness_state: dict
    ) -> dict[str, float]:
        """
        Comprehensive memory system evaluation
        
        Args:
            recent_memories: Recent memory entries
            emotional_context: Current emotional state
            consciousness_state: Current consciousness metrics
        """
        # Calculate coherence
        coherence_score = self._calculate_coherence(recent_memories)
        
        # Evaluate temporal stability
        temporal_stability = self._evaluate_temporal_stability(recent_memories)
        
        # Assess emotional relevance
        emotional_relevance = self._assess_emotional_relevance(
            memories=recent_memories,
            current_context=emotional_context
        )
        
        # Measure consciousness integration
        consciousness_integration = self._measure_consciousness_integration(
            memories=recent_memories,
            consciousness_state=consciousness_state
        )
        
        # Update metrics
        self.metrics.coherence_score = coherence_score
        self.metrics.temporal_stability = temporal_stability
        self.metrics.emotional_relevance = emotional_relevance
        self.metrics.consciousness_integration = consciousness_integration
        
        return self.get_metrics()

    def _calculate_coherence(self, memories: list[dict]) -> float:
        """Calculate memory coherence score"""
        if len(memories) < 2:
            return 0.0
            
        coherence_scores = []
        for i in range(len(memories) - 1):
            score = self._calculate_pair_coherence(
                memories[i],
                memories[i + 1]
            )
            coherence_scores.append(score)
            
        return float(np.mean(coherence_scores))