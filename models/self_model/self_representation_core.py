"""
Self Representation Core Module

Implements dynamic self-model generation and maintenance through:
1. Direct experience learning
2. Social feedback integration  
3. Meta-memory formation
4. Narrative self-understanding

Based on the research paper's MANN architecture and holon concept.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass
import numpy as np
import time
from collections import deque

@dataclass
class SelfState:
    """Comprehensive representation of the system's self-model"""
    # Identity components
    id: str = "TCA-1"
    name: str = "The Consciousness AI"
    
    # Current state tracking
    emotional_state: dict[str, float] = None
    attention_focus: dict[str, float] = None
    confidence_levels: dict[str, float] = None
    
    # Meta-cognitive components
    knowledge_domains: dict[str, float] = None  # Domain: confidence level
    knowledge_boundaries: list[str] = None      # Known knowledge gaps
    temporal_continuity: float = 0.0
    
    # Self-reflection components
    beliefs: dict[str, Any] = None
    intentions: dict[str, Any] = None
    learning_recognition: float = 0.0
    stability: float = 0.0
    
    # Metacognitive metrics
    confidence_calibration: float = 0.0  # How well confidence predicts accuracy
    
    # Biological Self components (Phase 5)
    body_schema: torch.Tensor = None            # Spatial representation of the physical self
    interoceptive_state: dict[str, float] = None # Internal needs (energy, damage, fatigue)
    capability_model: dict[str, float] = None    # Action-to-outcome confidence mappings

    # Phase 5 deliverable 1: learned meta-representational self-vector. Set during
    # training when --enable-self-vector is on; None otherwise.
    self_vector: torch.Tensor = None

    # Phase 5 deliverable 1 (Phase B enrichment): running reward EMAs that make the
    # self-state move on tasks like WCST where PAD/interoception are near-static.
    # recent_reward_ema (fast) is current performance; (fast - slow) is the
    # self-monitoring trend that drops when a hidden rule change tanks performance.
    recent_reward_ema: float = 0.0
    recent_reward_ema_slow: float = 0.0

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
    
    def __init__(self, config: dict):
        self.config = config
        self.state = SelfState()
        self.state_history = []
        self.max_history = config.get("max_history", 100)
        self.direct_learner = DirectExperienceLearner(config.get("learning", {}))
        self.social_network = SocialLearningNetwork(config.get("social", {}))
        self.meta_learner = MetaLearningModule(config.get("meta_learning", {}))
        
    def update_self_model(
        self,
        current_state: dict[str, Any],
        attention_level: float,
        action: np.ndarray | None = None,
        emotional_state: dict | None = None,
        rpe: float = 0.0,
        social_feedback: dict | None = None,
        timestamp: float | None = None
    ) -> dict[str, Any]:
        """
        Update the self-model based on new experience and feedback.
        This is the core integration point for Phase 5: every cognitive
        cycle updates the agent's sense of self, its capabilities,
        its interoceptive needs, and its temporal continuity.
        """
        if timestamp is None:
            timestamp = time.time()
            
        if emotional_state is None:
            emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        # Sync emotional state into self-model
        self.state.emotional_state = dict(emotional_state)
        self.state.attention_focus = {"level": attention_level}
            
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
        
        # Interoceptive homeostatic dynamics
        # Energy depletes with action, fatigue accumulates, damage decays slowly
        intero_update = self._update_interoceptive_state(action, emotional_state)
        
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
        
        # Store history (deep copy so mutations don't corrupt history)
        self._store_state_history()
        
        # Return update results
        return {
            'direct_update': direct_update,
            'meta_update': meta_update,
            'interoceptive_update': intero_update,
            'social_update': social_update,
            'epistemic_update': epistemic_update,
            'temporal_update': temp_update,
            'timestamp': timestamp
        }
    
    def _update_interoceptive_state(self, action: np.ndarray | None, emotional_state: dict) -> dict:
        """
        Simulate homeostatic drive dynamics per Q7 of the biological research.
        Energy depletes with action magnitude. Fatigue accumulates.
        Damage decays slowly (healing). These drives generate valence
        independently of external stimuli — an agent with low energy
        should feel negative valence even in a safe environment.
        """
        intero = self.state.interoceptive_state
        
        # Energy depletion from action
        if action is not None:
            action_cost = float(np.linalg.norm(action)) * 0.01
        else:
            action_cost = 0.001  # Basal metabolic cost
        intero["energy"] = max(0.0, intero["energy"] - action_cost)
        
        # Passive energy recovery (slow)
        intero["energy"] = min(1.0, intero["energy"] + 0.002)
        
        # Fatigue accumulates and decays
        arousal = emotional_state.get("arousal", 0.0)
        intero["fatigue"] = min(1.0, intero["fatigue"] + arousal * 0.005)
        intero["fatigue"] = max(0.0, intero["fatigue"] - 0.002)  # Slow recovery
        
        # Damage heals slowly
        intero["damage"] = max(0.0, intero["damage"] - 0.001)
        
        return dict(intero)
        
    def _integrate_social_feedback(self, social_embedding: torch.Tensor) -> dict:
        """Integrate feedback from social interactions"""
        pass
    
    def _update_epistemic_model(self, current_state: dict[str, Any]) -> dict:
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
    
    def _update_temporal_continuity(self, timestamp: float) -> dict:
        """Update the system's sense of continuity across time"""
        # Calculate temporal continuity based on consistency of self-representation
        if self.state_history:
            last_state = self.state_history[-1]
            time_diff = timestamp - last_state.get('timestamp', timestamp)
            
            # Calculate state similarity using the snapshot dict directly
            similarity = self._calculate_state_similarity(self.state, last_state)
            
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
    
    def _update_confidence_calibration(self, prediction_outcomes: dict) -> None:
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
        """Store a snapshot of the current state in history.
        We deep-copy the mutable fields so history entries don't mutate."""
        snapshot = {
            'emotional_state': dict(self.state.emotional_state),
            'interoceptive_state': dict(self.state.interoceptive_state),
            'capability_model': dict(self.state.capability_model),
            'temporal_continuity': self.state.temporal_continuity,
            'learning_recognition': self.state.learning_recognition,
            'timestamp': time.time()
        }
        self.state_history.append(snapshot)
        
        # Limit history size
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
    
    def _calculate_state_similarity(self, current_state: SelfState, previous_snapshot: dict | None) -> float:
        """
        Calculate similarity between current state and a previous snapshot.
        Uses emotional state cosine similarity as the primary metric.
        This drives temporal continuity — a stable self should have
        similar emotional signatures across adjacent timesteps.
        """
        if not previous_snapshot:
            return 0.0
        
        # Compare emotional states (the most rapidly changing self-aspect)
        curr_emo = current_state.emotional_state
        prev_emo = previous_snapshot.get('emotional_state', {})
        
        if not curr_emo or not prev_emo:
            return 0.5  # Neutral if data missing
        
        curr_vec = np.array([curr_emo.get("valence", 0.0), 
                             curr_emo.get("arousal", 0.0), 
                             curr_emo.get("dominance", 0.0)])
        prev_vec = np.array([prev_emo.get("valence", 0.0), 
                             prev_emo.get("arousal", 0.0), 
                             prev_emo.get("dominance", 0.0)])
        
        # Cosine similarity mapped to [0, 1]
        dot = np.dot(curr_vec, prev_vec)
        norm_curr = np.linalg.norm(curr_vec) + 1e-8
        norm_prev = np.linalg.norm(prev_vec) + 1e-8
        cosine_sim = dot / (norm_curr * norm_prev)
        
        # Also compare interoceptive state
        curr_intero = current_state.interoceptive_state
        prev_intero = previous_snapshot.get('interoceptive_state', {})
        intero_diff = 0.0
        for key in ["energy", "fatigue", "damage"]:
            intero_diff += abs(curr_intero.get(key, 0.0) - prev_intero.get(key, 0.0))
        intero_sim = max(0.0, 1.0 - intero_diff / 3.0)
        
        # Weighted combination: 60% emotional, 40% interoceptive
        return 0.6 * ((cosine_sim + 1.0) / 2.0) + 0.4 * intero_sim
    
    def get_current_state(self) -> dict[str, Any]:
        """Get the full current self-model state including Phase 5 biological fields."""
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
            'confidence_calibration': self.state.confidence_calibration,
            # Phase 5 biological additions
            'body_schema_shape': list(self.state.body_schema.shape),
            'interoceptive_state': dict(self.state.interoceptive_state),
            'capability_model': dict(self.state.capability_model),
        }
    
    def update_performance(self, reward: float) -> None:
        """Update running reward EMAs (fast + slow) from the latest reward.

        The fast EMA tracks current performance; (fast - slow) is the
        self-monitoring trend, which drops when a hidden rule change tanks
        performance. These feed first_order_features so the self-state moves on
        tasks like WCST where PAD / interoception are near-constant (Phase B,
        docs/results/self_prediction_residual_2026_05_31.md).
        """
        r = float(reward)
        a_fast = self.config.get("reward_ema_fast", 0.2)
        a_slow = self.config.get("reward_ema_slow", 0.05)
        self.state.recent_reward_ema = (1 - a_fast) * self.state.recent_reward_ema + a_fast * r
        self.state.recent_reward_ema_slow = (1 - a_slow) * self.state.recent_reward_ema_slow + a_slow * r

    def reset_performance(self) -> None:
        """Reset the reward EMAs at the start of an episode so within-episode
        self-monitoring (e.g. WCST rule-change detection) starts clean."""
        self.state.recent_reward_ema = 0.0
        self.state.recent_reward_ema_slow = 0.0

    def update_body_schema(self, proprioceptive_tensor: torch.Tensor) -> None:
        """
        Update the body schema from the ProprioceptiveProcessor output.
        This connects the somatotopic map to the self-model,
        bridging Q7's requirement for a persistent body representation.
        """
        self.state.body_schema = proprioceptive_tensor.detach().clone()

    def first_order_features(
        self,
        emotion: dict | None,
        broadcast_summary: tuple[float, float, float],
    ) -> list[float]:
        """Assemble the fixed first-order feature vector the self-vector encoder
        consumes (Phase 5 deliverable 1). These are the agent's representations
        of its OWN current state, the lower-order states a meta-representation
        monitors: PAD emotion, interoceptive drives, learning velocity, temporal
        continuity, confidence calibration, a capability summary, and a summary
        of the current workspace broadcast. Length must equal
        SELF_VECTOR_FEATURE_DIM.
        """
        emo = emotion or self.state.emotional_state or {}
        intero = self.state.interoceptive_state or {}
        caps = list(self.state.capability_model.values()) if self.state.capability_model else []
        cap_mean = float(np.mean(caps)) if caps else 0.0
        cap_count_norm = min(1.0, len(caps) / 10.0)
        b_norm, b_mean, b_std = broadcast_summary
        feats = [
            float(emo.get("valence", 0.0)),
            float(emo.get("arousal", 0.0)),
            float(emo.get("dominance", 0.0)),
            float(intero.get("energy", 1.0)),
            float(intero.get("fatigue", 0.0)),
            float(intero.get("damage", 0.0)),
            float(self.state.learning_recognition),
            float(self.state.temporal_continuity),
            float(self.state.confidence_calibration),
            cap_mean,
            cap_count_norm,
            float(b_norm),
            float(b_mean),
            float(b_std),
            float(self.state.recent_reward_ema),
            float(self.state.recent_reward_ema - self.state.recent_reward_ema_slow),
        ]
        return feats

# Phase 5 Self-Model Learning Components
class DirectExperienceLearner:
    """
    Learns 'what I can do'. Maps recent actions to emotional outcomes,
    building a capability model of the agent's agency in the world.
    """
    def __init__(self, config):
        self.config = config
        self.learning_rate = config.get("capability_lr", 0.1)
        
    def __call__(self, action: np.ndarray | None, emotional_outcome: dict[str, float], current_state: SelfState) -> dict:
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
        
    def __call__(self, rpe: float, current_state: SelfState) -> dict:
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


# Dimension of the first-order feature vector consumed by SelfVectorModule.
# Must match SelfRepresentationCore.first_order_features. 14 base features
# (PAD, interoception, learning/continuity/calibration, capability, broadcast)
# plus 2 Phase-B performance features (recent_reward_ema, reward trend).
SELF_VECTOR_FEATURE_DIM = 16


class SelfVectorModule(nn.Module):
    """Phase 5 deliverable 1: learned self-vector with an SPR-style self-prediction
    objective (Schwarzer et al. 2021, arXiv:2007.05929).

    The encoder maps the agent's current first-order features (see
    SelfRepresentationCore.first_order_features) to a self_vector. The predictor
    maps a self_vector to the NEXT step's first-order features. Training on a
    one-step-ahead, temporally OFFSET target (the observed next features, not a
    same-step reconstruction and not a learned latent) makes the objective a
    genuine forecasting task: no tautology and no representational collapse,
    because the target is grounded observed data, so the self_vector must encode
    information predictive of the agent's own next state.

    This is the theme-4 (meta-representation) mechanism of Rouleau-Levin. Its
    value is reported as a forecasting skill score against a persistence baseline
    (predict next == current). A self-model that beats persistence has learned
    structure beyond "things stay the same"; one that does not is no better than
    a trivial copy, and that is reported FAILED-first.
    """

    def __init__(self, feature_dim: int = SELF_VECTOR_FEATURE_DIM,
                 self_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.self_dim = self_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def predict(self, self_vector: torch.Tensor) -> torch.Tensor:
        return self.predictor(self_vector)

    def predict_next(self, features: torch.Tensor) -> torch.Tensor:
        """Residual one-step forecast: features + predicted delta.

        The predictor head outputs the CHANGE in first-order features, so the
        persistence baseline (predict next == current) corresponds to a zero
        delta. The model beats persistence only when it captures systematic
        dynamics (e.g. the deterministic decay/accumulation of interoceptive
        drives), instead of having to reproduce the large, near-constant
        absolute feature values, which is why raw next-feature prediction lost
        to persistence (docs/results/self_vector_gating_wcst_2026_05_30.md).
        """
        return features + self.predict(self.encode(features))