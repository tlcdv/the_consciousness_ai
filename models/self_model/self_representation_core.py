"""
Self Representation Core Module

Implements dynamic self-model generation and maintenance through:
1. Direct experience learning
2. Social feedback integration  
3. Meta-memory formation
4. Narrative self-understanding

Based on the research paper's MANN architecture and holon concept.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time
from collections import deque

@dataclass
class SelfState:
    """Comprehensive representation of the system's self-model"""
    # Identity components
    id: str = "ACM-1"
    name: str = "Artificial Consciousness Module"
    
    # Current state tracking
    emotional_state: Dict[str, float] = None
    attention_focus: Dict[str, float] = None
    confidence_levels: Dict[str, float] = None
    
    # Meta-cognitive components
    knowledge_domains: Dict[str, float] = None  # Domain: confidence level
    knowledge_boundaries: List[str] = None      # Known knowledge gaps
    temporal_continuity: float = 0.0
    
    # Self-reflection components
    beliefs: Dict[str, Any] = None
    intentions: Dict[str, Any] = None
    learning_recognition: float = 0.0
    stability: float = 0.0
    
    # Metacognitive metrics
    confidence_calibration: float = 0.0  # How well confidence predicts accuracy
    
    # Biological Self components (Phase 5)
    body_schema: torch.Tensor = None            # Spatial representation of the physical self
    interoceptive_state: Dict[str, float] = None # Internal needs (energy, damage, fatigue)
    capability_model: Dict[str, float] = None    # Action-to-outcome confidence mappings
    
    def __post_init__(self):
        """Initialize empty containers"""
        if self.emotional_state is None:
            self.emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        if self.attention_focus is None:
            self.attention_focus = {}
        if self.confidence_levels is None:
            self.confidence_levels = {}
        if self.knowledge_domains is None:
            self.knowledge_domains = {}
        if self.knowledge_boundaries is None:
            self.knowledge_boundaries = []
        if self.beliefs is None:
            self.beliefs = {}
        if self.intentions is None:
            self.intentions = {}
        if self.body_schema is None:
            self.body_schema = torch.zeros(1, 10, 8) # Default 10 body parts, 8 features
        if self.interoceptive_state is None:
            self.interoceptive_state = {"energy": 1.0, "damage": 0.0, "fatigue": 0.0}
        if self.capability_model is None:
            self.capability_model = {}

class SelfRepresentationCore:
    """
    Core implementation of the system's representation of itself.
    
    This is the foundation for self-awareness, integrating:
    1. Emotional recognition
    2. Attention tracking
    3. Confidence calibration
    4. Epistemological structures (what the system knows about what it knows)
    5. Temporal self-continuity
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.state = SelfState()
        self.state_history = []
        self.max_history = config.get("max_history", 100)
        self.direct_learner = DirectExperienceLearner(config.get("learning", {}))
        self.social_network = SocialLearningNetwork(config.get("social", {}))
        self.meta_learner = MetaLearningModule(config.get("meta_learning", {}))
        
    def update_self_model(
        self,
        current_state: Dict[str, Any],
        attention_level: float,
        action: Optional[np.ndarray] = None,
        emotional_state: Optional[Dict] = None,
        rpe: float = 0.0,
        social_feedback: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update the self-model based on new experience and feedback
        """
        if timestamp is None:
            timestamp = time.time()
            
        if emotional_state is None:
            emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            
        # Direct experience learning (Capabilities)
        direct_update = self.direct_learner(
            action=action,
            emotional_outcome=emotional_state,
            current_state=self.state
        )
        
        # Meta learning (Learning Velocity)
        meta_update = self.meta_learner(
            rpe=rpe,
            current_state=self.state
        )
        
        # Social learning (if feedback provided)
        social_update = {}
        if social_feedback:
            social_embedding = self.social_network(social_feedback)
            social_update = self._integrate_social_feedback(social_embedding)
            
        # Epistemological update - update what the system knows about what it knows
        epistemic_update = self._update_epistemic_model(current_state)
        
        # Temporal continuity - track changes over time
        temp_update = self._update_temporal_continuity(timestamp)
        
        # Update confidence calibration
        if 'prediction_outcomes' in current_state:
            self._update_confidence_calibration(current_state['prediction_outcomes'])
        
        # Store history
        self._store_state_history()
        
        # Return update results
        return {
            'direct_update': direct_update,
            'meta_update': meta_update,
            'social_update': social_update,
            'epistemic_update': epistemic_update,
            'temporal_update': temp_update,
            'timestamp': timestamp
        }
        
    def _integrate_social_feedback(self, social_embedding: torch.Tensor) -> Dict:
        """Integrate feedback from social interactions"""
        pass
    
    def _update_epistemic_model(self, current_state: Dict[str, Any]) -> Dict:
        """
        Update the system's model of what it knows.
        
        This is critical for "knowing that one knows" - metacognitive awareness
        """
        # Check for successful predictions to update knowledge confidence
        if 'prediction_outcomes' in current_state:
            outcomes = current_state['prediction_outcomes']
            for domain, result in outcomes.items():
                # Update confidence in this knowledge domain based on prediction success
                prev_confidence = self.state.knowledge_domains.get(domain, 0.5)
                correct = result.get('correct', False)
                
                # Increase confidence for correct predictions, decrease for incorrect
                update_rate = self.config.get("knowledge_update_rate", 0.05)
                new_confidence = prev_confidence + update_rate if correct else prev_confidence - update_rate
                self.state.knowledge_domains[domain] = max(0.0, min(1.0, new_confidence))
        
        # Identify knowledge boundaries when uncertain predictions occur
        if 'uncertain_areas' in current_state:
            for area in current_state['uncertain_areas']:
                if area not in self.state.knowledge_boundaries:
                    self.state.knowledge_boundaries.append(area)
        
        return {
            'domains_updated': list(self.state.knowledge_domains.keys()),
            'boundaries_identified': self.state.knowledge_boundaries
        }
    
    def _update_temporal_continuity(self, timestamp: float) -> Dict:
        """Update the system's sense of continuity across time"""
        # Calculate temporal continuity based on consistency of self-representation
        if self.state_history:
            last_state = self.state_history[-1]
            time_diff = timestamp - last_state.get('timestamp', timestamp)
            
            # Calculate state similarity
            similarity = self._calculate_state_similarity(self.state, last_state.get('state'))
            
            # Update continuity score (higher for similar states close in time)
            prev_continuity = self.state.temporal_continuity
            decay_rate = self.config.get("continuity_decay_rate", 0.1)
            time_factor = max(0.0, 1.0 - (time_diff / 3600))  # Normalize to hours
            
            new_continuity = prev_continuity * (1.0 - decay_rate) + similarity * time_factor * decay_rate
            self.state.temporal_continuity = new_continuity
            
            return {
                'previous_continuity': prev_continuity,
                'new_continuity': new_continuity,
                'time_difference': time_diff
            }
        
        return {'initialized': True}
    
    def _update_confidence_calibration(self, prediction_outcomes: Dict) -> None:
        """
        Update how well calibrated the system's confidence is with actual accuracy.
        
        This is essential for accurate metacognition.
        """
        confidences = []
        accuracies = []
        
        # Collect confidence-accuracy pairs
        for domain, outcome in prediction_outcomes.items():
            if 'confidence' in outcome and 'correct' in outcome:
                confidences.append(outcome['confidence'])
                accuracies.append(1.0 if outcome['correct'] else 0.0)
        
        if confidences:
            # Calculate calibration (how well confidence predicts accuracy)
            # Perfect calibration: confidence matches accuracy
            confidences = np.array(confidences)
            accuracies = np.array(accuracies)
            
            # Calculate calibration error (lower is better)
            calibration_error = np.mean(np.abs(confidences - accuracies))
            
            # Update calibration score (higher is better)
            self.state.confidence_calibration = 1.0 - calibration_error
    
    def _store_state_history(self) -> None:
        """Store current state in history"""
        self.state_history.append({
            'state': self.state,
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
    
    def _calculate_state_similarity(self, current_state: SelfState, previous_state: Optional[SelfState]) -> float:
        """Calculate similarity between current and previous states"""
        if not previous_state:
            return 0.0
            
        # Compare key aspects of state (emotional, attention, beliefs)
        # Implementation depends on specific comparison metrics
        # This is a placeholder
        return 0.8
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current self-model state"""
        return {
            'id': self.state.id,
            'name': self.state.name,
            'emotional_state': self.state.emotional_state,
            'attention_focus': self.state.attention_focus,
            'confidence_levels': self.state.confidence_levels,
            'knowledge_domains': self.state.knowledge_domains,
            'knowledge_boundaries': self.state.knowledge_boundaries,
            'temporal_continuity': self.state.temporal_continuity,
            'beliefs': self.state.beliefs,
            'intentions': self.state.intentions,
            'learning_recognition': self.state.learning_recognition,
            'stability': self.state.stability,
            'confidence_calibration': self.state.confidence_calibration
        }
        
# Real Implementations for Phase 5 Phase 5 Self-Model Learning
class DirectExperienceLearner:
    """
    Learns 'what I can do'. Maps recent actions to emotional outcomes,
    building a capability model of the agent's agency in the world.
    """
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.get("capability_lr", 0.1)
        
    def __call__(self, action: Optional[np.ndarray], emotional_outcome: Dict[str, float], current_state: SelfState) -> Dict:
        if action is None:
            return {}
            
        # Simplified: We hash the action sector to create a discrete 'capability' bucket
        # In a full neural architecture, this would be an MLP predicting Delta-Valence from Action
        action_mag = np.linalg.norm(action)
        if action_mag < 0.1:
            action_type = "idle"
        else:
            main_dim = np.argmax(np.abs(action))
            sign = "pos" if action[main_dim] > 0 else "neg"
            action_type = f"move_dim_{main_dim}_{sign}"
            
        # Track expected emotional outcome of this action
        current_valence_exp = current_state.capability_model.get(f"{action_type}_valence", 0.0)
        actual_valence = emotional_outcome.get("valence", 0.0)
        
        # EMA update
        new_valence_exp = current_valence_exp + self.learning_rate * (actual_valence - current_valence_exp)
        current_state.capability_model[f"{action_type}_valence"] = new_valence_exp
        
        return {
            "action_type": action_type,
            "expected_valence_shift": new_valence_exp
        }

class SocialLearningNetwork:
    """Stub for future multi-agent interaction."""
    def __init__(self, config):
        self.config = config
        
    def __call__(self, social_feedback):
        return torch.zeros(128)
        
class MetaLearningModule:
    """
    Tracks learning velocity. If RPE variance is dropping, the agent is 
    successfully learning. If RPE variance spikes, the agent is in a novel situation.
    """
    def __init__(self, config):
        self.config = config
        self.rpe_window_size = config.get("rpe_window_size", 50)
        self.rpe_history = deque(maxlen=self.rpe_window_size)
        self.learning_velocity = 0.0
        
    def __call__(self, rpe: float, current_state: SelfState) -> Dict:
        self.rpe_history.append(rpe)
        
        if len(self.rpe_history) < 10:
            return {"learning_velocity": 0.0, "novelty_spike": False}
            
        # Calculate recent variance vs older variance
        recent_var = np.var(list(self.rpe_history)[-10:])
        overall_var = np.var(list(self.rpe_history))
        
        # If recent variance is much lower than overall, we are converging (learning)
        # If it's much higher, we hit something novel/confusing
        variance_ratio = recent_var / (overall_var + 1e-8)
        
        novelty_spike = variance_ratio > 2.0
        
        # Velocity is positive when variance is dropping
        self.learning_velocity = 1.0 - variance_ratio
        current_state.learning_recognition = self.learning_velocity
        
        return {
            "learning_velocity": self.learning_velocity,
            "rpe_variance_ratio": variance_ratio,
            "novelty_spike": novelty_spike
        }