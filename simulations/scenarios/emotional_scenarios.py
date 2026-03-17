from __future__ import annotations

# simulations/scenarios/emotional_scenarios.py

import numpy as np
from typing import Any
from dataclasses import dataclass
from enum import Enum

class ScenarioType(Enum):
    """Types of emotional development scenarios"""
    SURVIVAL = "survival"
    SOCIAL = "social"
    ETHICAL = "ethical"
    LEARNING = "learning"

@dataclass
class ScenarioConfig:
    """Configuration for emotional scenarios"""
    base_stress_level: float = 0.7
    stress_adaptation_rate: float = 0.1
    attention_threshold: float = 0.8
    interaction_frequency: float = 0.5
    emotional_memory_threshold: float = 0.6
    max_duration: int = 1000

class EmotionalScenarioGenerator:
    """
    Generates emotional development scenarios for consciousness formation
    
    Key Features:
    1. Stress-based attention activation
    2. Social interaction opportunities
    3. Ethical decision points
    4. Memory formation triggers
    """
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.scenario_history = []
        self.stress_history = []
        self.interaction_history = []
        
    def generate_scenario(
        self,
        scenario_type: ScenarioType,
        current_emotional_state: dict[str, float] | None = None
    ) -> dict:
        """Generate scenario based on type and emotional state"""
        
        if scenario_type == ScenarioType.SURVIVAL:
            return self._generate_survival_scenario(current_emotional_state)
        elif scenario_type == ScenarioType.SOCIAL:
            return self._generate_social_scenario(current_emotional_state)
        elif scenario_type == ScenarioType.ETHICAL:
            return self._generate_ethical_scenario(current_emotional_state)
        elif scenario_type == ScenarioType.LEARNING:
            return self._generate_learning_scenario(current_emotional_state)
        
    def _generate_survival_scenario(
        self,
        emotional_state: dict[str, float] | None
    ) -> dict:
        """Generate survival-based attention scenarios"""
        stress_level = self._calculate_stress_level(emotional_state)
        
        scenario = {
            'type': ScenarioType.SURVIVAL,
            'description': "Navigate through challenging environment",
            'stress_level': stress_level,
            'objectives': [
                "Maintain system integrity",
                "Find optimal solution path",
                "Adapt to environmental threats"
            ],
            'interaction_points': self._generate_interaction_points(),
            'attention_triggers': self._generate_attention_triggers(stress_level),
            'memory_formation_opportunities': self._generate_memory_triggers()
        }
        
        self.scenario_history.append(scenario)
        return scenario
        
    def _generate_social_scenario(
        self,
        emotional_state: dict[str, float] | None
    ) -> dict:
        """Generate social interaction scenarios"""
        interaction_intensity = self._calculate_interaction_intensity(emotional_state)
        
        scenario = {
            'type': ScenarioType.SOCIAL,
            'description': "Build emotional connections through interaction",
            'interaction_intensity': interaction_intensity,
            'objectives': [
                "Establish emotional rapport",
                "Demonstrate empathy",
                "Build trust through cooperation"
            ],
            'interaction_points': self._generate_interaction_points(),
            'emotional_triggers': self._generate_emotional_triggers(),
            'memory_formation_opportunities': self._generate_memory_triggers()
        }
        
        self.scenario_history.append(scenario)
        return scenario
        
    def _calculate_stress_level(
        self,
        emotional_state: dict[str, float] | None
    ) -> float:
        """Calculate appropriate stress level based on adaptation"""
        base_stress = self.config.base_stress_level
        
        if emotional_state and self.stress_history:
            # Adjust stress based on emotional state and adaptation
            recent_stress = np.mean(self.stress_history[-10:])
            emotional_valence = emotional_state.get('valence', 0.5)
            
            # Lower stress if showing good adaptation
            if emotional_valence > 0.7 and recent_stress > 0.5:
                base_stress *= (1.0 - self.config.stress_adaptation_rate)
            # Increase stress if adaptation is too easy
            elif emotional_valence > 0.8 and recent_stress < 0.3:
                base_stress *= (1.0 + self.config.stress_adaptation_rate)
                
        self.stress_history.append(base_stress)
        return min(1.0, max(0.1, base_stress))
        
    def _generate_interaction_points(self) -> list[dict]:
        """Generate interaction opportunities in scenario"""
        num_interactions = int(self.config.max_duration * 
                             self.config.interaction_frequency)
        
        return [
            {
                'trigger': f"interaction_{i}",
                'type': np.random.choice(['help', 'cooperate', 'communicate']),
                'emotional_weight': np.random.uniform(0.5, 1.0)
            }
            for i in range(num_interactions)
        ]
        
    def _generate_attention_triggers(self, stress_level: float) -> list[dict]:
        """Generate attention-triggering events"""
        num_triggers = int(stress_level * 10)
        
        return [
            {
                'trigger': f"attention_{i}",
                'intensity': np.random.uniform(0.7, 1.0),
                'duration': np.random.randint(10, 50)
            }
            for i in range(num_triggers)
        ]
        
    def _generate_emotional_triggers(self) -> list[dict]:
        """Generate emotional response opportunities"""
        return [
            {
                'emotion': emotion,
                'intensity': np.random.uniform(0.5, 1.0),
                'context': f"emotional_context_{i}"
            }
            for i, emotion in enumerate(['empathy', 'trust', 'cooperation'])
        ]
        
    def _generate_memory_triggers(self) -> list[dict]:
        """Generate memory formation opportunities"""
        return [
            {
                'importance': np.random.uniform(0.7, 1.0),
                'emotional_salience': np.random.uniform(0.6, 1.0),
                'context': f"memory_context_{i}"
            }
            for i in range(3)
        ]

class EmotionalScenario:
    """
    Manages simulation tasks, increasing complexity progressively.
    """

    def __init__(self, config: dict[str, float]):
        self.config = config
        self.stage = 0

    def get_initial_state(self) -> dict:
        """
        Returns the initial state configuration for the current simulation stage.
        """
        if self.stage == 0:
            # Basic survival task state
            return {"food": 5, "threat_level": 0.2}
        elif self.stage == 1:
            # Introduce social interaction parameters
            return {"peers": 3, "threat_level": 0.3}
        return {}

    def update_scenario(self, agent: Any) -> None:
        """
        Increase the simulation complexity based on agent performance.
        """
        performance_score = agent.get_performance_score()
        if performance_score > self.config.get("threshold_stage_1", 100) and self.stage == 0:
            self.stage = 1