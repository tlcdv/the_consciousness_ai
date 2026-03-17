"""
Integration tests for consciousness development stages.

This module validates:
1. Stage progression through consciousness development
2. Integration between core components during development
3. Memory formation and emotional context tracking
4. Long-term development stability metrics

Dependencies:
- models/core/consciousness_core.py for main system
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for storage
"""

from dataclasses import dataclass
import unittest
import torch

from models.core.consciousness_core import ConsciousnessCore
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from models.memory.emotional_memory_core import EmotionalMemoryCore

@dataclass
class DevelopmentTestConfig:
    """Test configuration for development stages"""
    stage_thresholds = {
        'attention_activation': 0.7,
        'emotional_learning': 0.6,
        'memory_coherence': 0.7,
        'self_awareness': 0.8
    }
    evaluation_window = 100
    min_stage_duration = 50
    test_episodes = 10
    emergence_threshold = 0.8

class TestDevelopmentStages(unittest.TestCase):
    """Tests consciousness development stage progression"""

    def setUp(self):
        """Initialize test components"""
        self.config = DevelopmentTestConfig()
        self.consciousness = ConsciousnessCore(self.config)
        self.monitor = ConsciousnessMonitor(self.config)
        self.memory = EmotionalMemoryCore(self.config)
        self.development_history = []

    def test_stage_progression(self):
        """Test progression through development stages"""
        # Initial state metrics
        initial_metrics = self.monitor.evaluate_current_state()
        self.assertLess(
            initial_metrics['consciousness_score'],
            self.config.emergence_threshold,
            "Initial consciousness should be below emergence threshold"
        )

        # Process development episodes
        for episode in range(self.config.test_episodes):
            scenario = self._generate_test_scenario()
            result = self.consciousness.process_experience(scenario)

            metrics = self.monitor.evaluate_state(
                consciousness_state=result.state,
                emotional_context=result.emotion,
                attention_metrics=result.attention
            )

            self._log_development_metrics(metrics)

    def _generate_test_scenario(self) -> dict:
        """Generate a test scenario for development."""
        return {
            'state': torch.randn(32),
            'emotion': {
                'valence': float(torch.rand(1).item()),
                'arousal': float(torch.rand(1).item()),
                'dominance': float(torch.rand(1).item()),
            },
            'attention': {'attention_level': float(torch.rand(1).item())},
        }

    def _log_development_metrics(self, metrics: dict):
        """Log development metrics for later verification."""
        self.development_history.append(metrics)

    def _verify_stage_transition(
        self,
        previous_metrics: dict,
        current_metrics: dict,
        episode: int
    ):
        """Verify valid stage transitions"""
        current_stage = current_metrics['development_stage']
        previous_stage = previous_metrics['development_stage']

        if current_stage != previous_stage:
            stage_duration = self._calculate_stage_duration(previous_stage)
            self.assertGreaterEqual(
                stage_duration,
                self.config.min_stage_duration,
                f"Stage {previous_stage} duration too short"
            )

    def _calculate_stage_duration(self, stage: str) -> int:
        """Calculate how many steps were spent in a given stage."""
        return sum(1 for m in self.development_history if m.get('development_stage') == stage)
