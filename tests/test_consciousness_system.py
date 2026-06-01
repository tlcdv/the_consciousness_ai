"""
System-wide integration tests for The Consciousness AI.

Tests the integration between core components:
1. Consciousness development through stress response
2. Emotional memory formation and retrieval
3. Attention gating mechanisms
4. Overall development metrics

Dependencies:
- models/core/consciousness_core.py for main consciousness system
- models/evaluation/consciousness_metrics.py for evaluation
- models/memory/emotional_memory_core.py for memory storage
"""
from __future__ import annotations

import unittest
import torch
import numpy as np
from dataclasses import dataclass

from models.memory.memory_integration import MemoryIntegrationCore
from models.evaluation.consciousness_metrics import ConsciousnessMetrics
from models.self_model.self_representation_core import SelfRepresentationCore
from models.evaluation.consciousness_monitor import ConsciousnessMonitor

@dataclass
class IntegrationTestConfig:
    """Test configuration for full system integration"""
    memory_config = {
        'capacity': 1000,
        'embedding_dim': 768,
        'emotional_dim': 256,
        'attention_threshold': 0.7
    }
    consciousness_config = {
        'coherence_threshold': 0.7,
        'emotional_stability': 0.6,
        'temporal_window': 100
    }
    development_stages = [
        'attention_activation',
        'emotional_learning',
        'self_awareness',
        'narrative_coherence'
    ]

class TestConsciousnessSystem(unittest.TestCase):
    """System-wide integration tests for consciousness development"""

    def setUp(self):
        """Initialize test components"""
        self.config = IntegrationTestConfig()
        
        # Initialize core components
        self.memory = MemoryIntegrationCore(self.config.memory_config)
        self.consciousness = ConsciousnessMetrics(self.config.consciousness_config)
        self.self_model = SelfRepresentationCore(self.config.consciousness_config)
        self.monitor = ConsciousnessMonitor(self.config.consciousness_config)

    def test_complete_development_cycle(self):
        """Test full consciousness development cycle"""
        development_metrics = []
        
        # Run development episodes
        for episode in range(10):
            # Generate experience with increasing complexity
            experience = self._generate_experience(episode)
            
            # Process through consciousness pipeline
            consciousness_state = self._process_consciousness_cycle(experience)
            
            # Update the canonical self-model (SelfRepresentationCore).
            self.self_model.update_self_model(
                current_state={},
                attention_level=consciousness_state['attention_level'],
                emotional_state=experience['emotion'],
            )

            # Evaluate development (the monitor computes consciousness_level from
            # attention/emotion inputs).
            metrics = self.monitor.evaluate_development(
                current_state=consciousness_state,
                emotion_values=experience['emotion'],
                attention_metrics={'attention_level': consciousness_state['attention_level']},
                memory_state=self.memory.get_state(),
            )

            # Store experience with consciousness context
            self.memory.store_experience(
                experience_data=consciousness_state['state'],
                emotional_context=experience['emotion'],
                consciousness_level=metrics['consciousness_level'],
            )
            
            development_metrics.append(metrics)
            
        # Verify development progression
        self._verify_development_progression(development_metrics)

    def _process_consciousness_cycle(self, experience: dict) -> dict:
        """Process an experience through the consciousness pipeline."""
        state = experience['state']
        emotion = experience['emotion']
        attention = experience.get('attention', 0.5)
        return {
            'state': state,
            'attention_level': attention,
            'emotion': emotion,
        }

    def _generate_experience(self, episode: int) -> dict:
        """Generate increasingly complex experiences"""
        return {
            'state': torch.randn(32),
            'emotion': {
                'valence': min(1.0, 0.5 + 0.05 * episode),
                'arousal': 0.7,
                'dominance': min(1.0, 0.4 + 0.05 * episode)
            },
            'attention': min(1.0, 0.6 + 0.04 * episode),
            'narrative': f"Experience {episode} with growing consciousness",
            'complexity_level': episode / 10.0
        }

    def _verify_development_progression(self, metrics_history: list[dict]):
        """Verify consciousness development progression"""
        initial_metrics = metrics_history[0]
        final_metrics = metrics_history[-1]
        
        # Verify consciousness development
        self.assertGreater(
            final_metrics['consciousness_level'],
            initial_metrics['consciousness_level'],
            "Consciousness level should increase"
        )
        
        # Verify emotional development
        self.assertGreater(
            final_metrics['emotional_awareness'],
            initial_metrics['emotional_awareness'],
            "Emotional awareness should improve"
        )
        
        # Verify memory coherence
        self.assertGreater(
            final_metrics['memory_coherence'],
            initial_metrics['memory_coherence'],
            "Memory coherence should increase"
        )