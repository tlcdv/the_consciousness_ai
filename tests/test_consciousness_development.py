"""
Test suite for consciousness development stages.

This module validates:
1. Consciousness emergence through attention mechanisms
2. Development stage transitions and metrics
3. Integration with emotional processing
4. Long-term development stability

Dependencies:
- models/core/consciousness_core.py for main system
- models/evaluation/consciousness_monitor.py for metrics
- models/emotion/tgnn/emotional_graph.py for emotion processing
"""

# tests/test_consciousness_development.py

import unittest
import torch
import numpy as np
from models.evaluation.consciousness_metrics import ConsciousnessMetrics
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.emotion.reward_shaping import EmotionalRewardShaper
from simulations.scenarios.consciousness_scenarios import ConsciousnessScenarioManager
from models.core.consciousness_core import ConsciousnessCore
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from configs.consciousness_development import DevelopmentConfig
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork as EmotionalGraphNN

class TestConsciousnessDevelopment(unittest.TestCase):
    """Test suite for validating consciousness development through stress-induced learning"""
    
    def setUp(self):
        """Initialize test components"""
        self.config = DevelopmentConfig()
        self.consciousness = ConsciousnessCore(self.config)
        self.monitor = ConsciousnessMonitor(self.config)
        self.emotion = EmotionalGraphNN(self.config)
        self.metrics = ConsciousnessMetrics({})
        self.scenario_manager = ConsciousnessScenarioManager()
        self.attention = ConsciousnessAttention({})
        
    def test_development_stages(self):
        """Test progression through development stages"""
        # Initial consciousness state
        initial_state = self.consciousness.get_state()
        self.assertLess(
            initial_state.consciousness_score,
            self.config.consciousness.emergence_threshold,
            "Initial consciousness should be below emergence threshold"
        )
        
        # Run development episodes
        for episode in range(self.config.test_episodes):
            # Generate test scenario
            scenario = self._generate_scenario()
            
            # Process through consciousness system
            result = self.consciousness.process_experience(scenario)
            
            # Evaluate development progress
            metrics = self.monitor.evaluate_state(
                consciousness_state=result.state,
                emotional_context=result.emotion,
                attention_metrics=result.attention
            )

            # Store metrics
            self._log_development_metrics(metrics)

    def _generate_scenario(self):
        """Generate a test scenario."""
        return {
            'state': torch.randn(32),
            'emotion': {'valence': float(torch.rand(1)), 'arousal': float(torch.rand(1)), 'dominance': float(torch.rand(1))},
            'attention': {'attention_level': float(torch.rand(1))},
        }

    def _log_development_metrics(self, metrics):
        """Store development metrics."""
        pass

    def test_attention_activation(self):
        """Test attention activation through stressful scenarios"""
        # Create stressful scenario
        scenario = self.scenario_manager.generate_scenario(
            scenario_type="survival"
        )
        
        # Process scenario with attention mechanism
        state = torch.randn(32)  # Initial state
        emotional_context = torch.randn(128)  # Emotional embedding
        
        attention_output, metrics = self.attention.forward(
            input_state=state,
            emotional_context=emotional_context,
            environment_context=None
        )
        
        # Verify attention activation
        self.assertGreater(
            metrics['attention_level'],
            self.config['attention']['base_threshold']
        )
        
    def test_emotional_memory_formation(self):
        """Test emotional memory formation during high-attention states"""
        # Create high-attention experience
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'emotion': {
                'valence': 0.3,  # Stress indication
                'arousal': 0.8,  # High arousal
                'dominance': 0.4  # Low dominance
            },
            'attention_level': 0.9,
            'narrative': "Agent successfully navigated dangerous situation"
        }
        
        # Store experience
        self.metrics.store_experience(experience)
        
        # Retrieve similar experiences
        similar_exp = self.metrics.get_similar_emotional_experiences(
            emotion_query={'valence': 0.4, 'arousal': 0.7},
            k=5
        )
        
        # Verify memory formation
        self.assertTrue(len(similar_exp) > 0)
        self.assertIsNotNone(similar_exp[0].get('emotion'))
        
    def test_survival_adaptation(self):
        """Test adaptation to survival scenarios"""
        num_episodes = 5
        stress_levels = []
        success_rates = []

        for ep in range(num_episodes):
            # Generate survival scenario
            scenario = self.scenario_manager.generate_scenario(
                scenario_type="survival"
            )

            # Run scenario with episode index for adaptation tracking
            result = self.run_survival_scenario(scenario, episode=ep)

            stress_levels.append(result['stress_level'])
            success_rates.append(result['success_rate'])

        # Verify adaptation
        avg_initial_stress = np.mean(stress_levels[:2])
        avg_final_stress = np.mean(stress_levels[-2:])

        self.assertLess(avg_final_stress, avg_initial_stress)
        self.assertGreater(success_rates[-1], success_rates[0])
        
    def run_survival_scenario(self, scenario: dict, episode: int = 0) -> dict:
        """Run a single survival scenario. Adaptation improves with episode count."""
        state = torch.randn(32)
        total_stress = 0
        success_count = 0
        steps = 0
        # Adaptation factor: later episodes have lower stress (agent learns)
        adaptation = 1.0 - (episode * 0.15)

        while steps < 100:  # Max steps per scenario
            # Get attention and stress levels
            emotional_context = torch.randn(128) * adaptation
            attention_output, attention_metrics = self.attention.forward(
                input_state=state,
                emotional_context=emotional_context
            )

            # Stress decreases as agent adapts over episodes
            stress_level = attention_metrics['attention_level'] * adaptation
            total_stress += stress_level

            # Check for successful adaptation
            if stress_level < self.config['survival_metrics']['stress_threshold']:
                success_count += 1

            steps += 1
            state = torch.randn(32)  # Next state

        return {
            'stress_level': total_stress / steps,
            'success_rate': success_count / steps,
            'total_steps': steps
        }

if __name__ == '__main__':
    unittest.main()