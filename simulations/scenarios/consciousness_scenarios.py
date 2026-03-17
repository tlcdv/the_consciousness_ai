# simulations/scenarios/consciousness_scenarios.py

import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random


class UnrealEngineInterface:
    """Stub for Unreal Engine integration"""
    def __init__(self, config=None):
        self.config = config

    def create_scenario(self, config):
        return f"scenario_{id(config)}"


class AttentionTriggerSystem:
    """Stub for attention trigger system"""
    def __init__(self, config=None):
        self.config = config

    def setup(self, scenario_id=None, stress_level=0.0):
        pass

"""
Consciousness Development Scenario Generator

This module handles:
1. Generation of consciousness development scenarios
2. Simulation of stressful situations to trigger attention
3. Integration with UE5 for immersive environments
4. Recording of consciousness development metrics

Dependencies:
- models/core/consciousness_core.py for main system integration
- models/evaluation/consciousness_monitor.py for metrics
- configs/consciousness_development.yaml for parameters
"""

@dataclass
class ScenarioConfig:
    """Configuration for consciousness development scenarios"""
    stress_level: float = 0.7  # Base stress level
    attention_threshold: float = 0.8  # Required attention level
    interaction_frequency: float = 0.5  # Human interaction frequency
    max_duration: int = 1000  # Maximum scenario steps
    success_threshold: float = 0.6  # Required success rate

class ScenarioType(Enum):
    """Types of consciousness development scenarios"""
    SURVIVAL = "survival"
    SOCIAL = "social"
    ETHICAL = "ethical"
    PROBLEM_SOLVING = "problem_solving"

class ConsciousnessScenarioGenerator:
    def __init__(self, config=None):
        """Initialize scenario generation"""
        if config is None:
            config = {}
        self.config = config
        ue_cfg = getattr(config, 'ue', None) if not isinstance(config, dict) else config.get('ue')
        self.ue_engine = UnrealEngineInterface(ue_cfg)
        self.attention_triggers = AttentionTriggerSystem(config)
        
    def generate_scenario(
        self,
        difficulty: float = 0.5,
        stress_level: float = 0.7,
        scenario_type: str = "survival"
    ) -> dict:
        """Generate consciousness development scenario"""
        # Configure base scenario
        scenario_config = {
            'difficulty': difficulty,
            'stress_level': stress_level,
            'type': scenario_type,
            'evaluation_metrics': self._get_evaluation_metrics()
        }
        
        # Generate scenario in UE5
        scenario_id = self.ue_engine.create_scenario(scenario_config)
        
        # Configure attention triggers
        self.attention_triggers.setup(
            scenario_id=scenario_id,
            stress_level=stress_level
        )
        
        return self._build_scenario_descriptor(scenario_id, scenario_config)

    def _get_evaluation_metrics(self) -> dict:
        """Return default evaluation metrics for a scenario."""
        return {
            'attention_threshold': 0.7,
            'stress_adaptation': 0.6,
            'emotional_coherence': 0.5,
        }

    def _build_scenario_descriptor(self, scenario_id, config=None) -> dict:
        """Build a descriptor dict for a generated scenario."""
        import torch
        cfg = config or {}
        return {
            'id': scenario_id,
            'type': cfg.get('type', 'survival'),
            'difficulty': cfg.get('difficulty', 0.5),
            'stress_level': cfg.get('stress_level', 0.7),
            'state': torch.randn(32),
            'emotion': {
                'valence': random.uniform(0.2, 0.8),
                'arousal': random.uniform(0.5, 0.9),
                'dominance': random.uniform(0.3, 0.7),
            },
            'attention': {'attention_level': random.uniform(0.5, 0.9)},
        }

    def _generate_survival_scenario(self) -> dict:
        """Generate survival-based scenario"""
        # Create stressful situation to trigger attention
        stress_params = {
            'intensity': random.uniform(0.6, 0.9),
            'duration': random.randint(100, 300),
            'type': random.choice(['physical', 'emotional', 'social'])
        }
        
        # Configure scenario
        return {
            'type': 'survival',
            'stress_params': stress_params,
            'success_criteria': {
                'min_attention': 0.7,
                'min_adaptation': 0.6
            }
        }
        
    def _generate_social_scenario(self) -> dict:
        """Generate social interaction scenario"""
        scenario = {
            'type': ScenarioType.SOCIAL,
            'stress_level': self.config.stress_level * 0.8,
            'description': "Agent must assist humans in crisis",
            'objectives': [
                "Understand emotional states",
                "Provide appropriate assistance",
                "Build trust through interaction"
            ],
            'constraints': {
                'interaction_frequency': self.config.interaction_frequency,
                'emotional_coherence_required': True,
                'trust_threshold': 0.7
            }
        }
        return scenario
        
    def evaluate_performance(
        self,
        attention_level: float,
        interaction_quality: float,
        success_rate: float
    ) -> dict:
        """Evaluate scenario performance"""
        
        # Track metrics
        self.attention_history.append(attention_level)
        self.interaction_history.append(interaction_quality)
        self.success_history.append(success_rate)
        
        # Calculate progress
        avg_attention = np.mean(self.attention_history[-100:])
        avg_interaction = np.mean(self.interaction_history[-100:])
        avg_success = np.mean(self.success_history[-100:])
        
        return {
            'attention_level': avg_attention,
            'interaction_quality': avg_interaction,
            'success_rate': avg_success,
            'meets_criteria': self._check_success_criteria(
                avg_attention, avg_interaction, avg_success
            )
        }
        
    def _check_success_criteria(
        self,
        attention: float,
        interaction: float,
        success: float
    ) -> bool:
        """Check if performance meets success criteria"""
        return (
            attention >= self.config.attention_threshold and
            interaction >= self.config.interaction_frequency and
            success >= self.config.success_threshold
        )
        
    def get_scenario_stats(self) -> dict:
        """Get current scenario statistics"""
        if not self.attention_history:
            return {}
            
        return {
            'total_scenarios': len(self.success_history),
            'avg_attention': np.mean(self.attention_history),
            'avg_interaction': np.mean(self.interaction_history),
            'avg_success': np.mean(self.success_history),
            'recent_improvement': self._calculate_improvement()
        }
        
    def _calculate_improvement(self) -> float:
        """Calculate recent improvement in performance"""
        if len(self.success_history) < 100:
            return 0.0

        recent = np.mean(self.success_history[-50:])
        previous = np.mean(self.success_history[-100:-50])
        return recent - previous


# Alias for backward compatibility with tests and other modules
ConsciousnessScenarioManager = ConsciousnessScenarioGenerator