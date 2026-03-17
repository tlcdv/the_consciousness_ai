"""
Enhanced Consciousness Metrics System

This module implements:
1. Advanced consciousness development tracking
2. Multi-dimensional metric analysis
3. Development stage validation
4. Integration with emotional and memory systems

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion metrics
- models/memory/emotional_memory_core.py for memory validation
- models/evaluation/consciousness_monitor.py for base metrics
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class EnhancedMetrics:
    """Enhanced metrics for consciousness tracking"""
    emotional_coherence: float = 0.0
    memory_stability: float = 0.0
    attention_consistency: float = 0.0
    behavioral_adaptation: float = 0.0
    learning_progress: float = 0.0
    social_awareness: float = 0.0

@dataclass
class ConsciousnessMetrics:
    """Tracks comprehensive consciousness development metrics"""
    emotional_awareness: float = 0.0
    memory_coherence: float = 0.0
    attention_stability: float = 0.0
    temporal_consistency: float = 0.0
    self_model_quality: float = 0.0
    narrative_coherence: float = 0.0
    development_stage: str = 'initial'

class EnhancedConsciousnessEvaluator:
    """
    Evaluates consciousness development across multiple dimensions
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics = ConsciousnessMetrics()
        self.development_history = []
        
        # Initialize thresholds
        self.consciousness_thresholds = {
            'attention_activation': 0.7,
            'emotional_learning': 0.6,
            'self_awareness': 0.8,
            'narrative_coherence': 0.7
        }

    def evaluate_consciousness(
        self,
        current_state: dict,
        memory_state: dict,
        self_model_state: dict,
        emotional_context: dict | None = None
    ) -> dict[str, float]:
        """
        Comprehensive consciousness evaluation across all dimensions
        """
        # Calculate core metrics
        self.metrics.emotional_awareness = self._evaluate_emotional_awareness(
            emotional_context, self_model_state
        )
        
        self.metrics.memory_coherence = self._evaluate_memory_coherence(
            memory_state
        )
        
        self.metrics.attention_stability = self._evaluate_attention_stability(
            current_state
        )
        
        self.metrics.temporal_consistency = self._evaluate_temporal_consistency(
            memory_state
        )
        
        self.metrics.self_model_quality = self._evaluate_self_model(
            self_model_state
        )
        
        self.metrics.narrative_coherence = self._evaluate_narrative_coherence(
            memory_state
        )
        
        # Update development stage
        self.metrics.development_stage = self._determine_development_stage()
        
        # Store metrics
        self.development_history.append(self.get_metrics())
        
        return self.get_metrics()

    def _evaluate_self_model(self, self_model_state: dict) -> float:
        """Evaluate quality of self-model representation"""
        if not self_model_state:
            return 0.0
            
        confidence = self_model_state.get('confidence', 0.5)
        coherence = self_model_state.get('coherence', 0.5)
        stability = self_model_state.get('stability', 0.5)
        
        return (confidence + coherence + stability) / 3.0

    def get_metrics(self) -> dict[str, float]:
        """Get current consciousness metrics"""
        return {
            'emotional_awareness': self.metrics.emotional_awareness,
            'memory_coherence': self.metrics.memory_coherence,
            'attention_stability': self.metrics.attention_stability,
            'temporal_consistency': self.metrics.temporal_consistency,
            'self_model_quality': self.metrics.self_model_quality,
            'narrative_coherence': self.metrics.narrative_coherence,
            'development_stage': self.metrics.development_stage,
            'consciousness_level': self._calculate_consciousness_level()
        }