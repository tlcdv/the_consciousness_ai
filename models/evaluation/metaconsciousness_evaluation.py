from __future__ import annotations

import time
import numpy as np
import torch
from typing import Any
from dataclasses import dataclass

@dataclass
class MetaconsciousnessMetrics:
    """Metrics for measuring meta-consciousness capabilities"""
    self_reflection: float = 0.0
    belief_updating: float = 0.0
    attention_awareness: float = 0.0
    uncertainty_recognition: float = 0.0
    temporal_introspection: float = 0.0
    metacognitive_accuracy: float = 0.0
    
    def get_overall_score(self) -> float:
        """Calculate overall metaconsciousness score"""
        metrics = [
            self.self_reflection,
            self.belief_updating,
            self.attention_awareness,
            self.uncertainty_recognition,
            self.temporal_introspection,
            self.metacognitive_accuracy
        ]
        return sum(metrics) / len(metrics)

class MetaconsciousnessEvaluator:
    """
    Evaluates metaconsciousness capabilities - the system's ability to
    reflect on and modify its own cognitive processes.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.metrics = MetaconsciousnessMetrics()
        self.history = []
        
    def evaluate_metaconsciousness(
        self,
        self_model_state: dict,
        belief_updates: list[dict],
        introspection_results: dict,
        prediction_history: list[dict]
    ) -> dict:
        """
        Evaluate the meta-consciousness of the system
        
        Args:
            self_model_state: Current state of the self-model
            belief_updates: Recent updates to the belief system
            introspection_results: Results from introspection processes
            prediction_history: History of predictions and outcomes
            
        Returns:
            Metaconsciousness metrics
        """
        # Evaluate self-reflection capabilities
        self.metrics.self_reflection = self._evaluate_self_reflection(
            self_model_state,
            introspection_results
        )
        
        # Evaluate belief updating capabilities
        self.metrics.belief_updating = self._evaluate_belief_updating(
            belief_updates
        )
        
        # Evaluate attention awareness
        self.metrics.attention_awareness = self._evaluate_attention_awareness(
            self_model_state
        )
        
        # Evaluate uncertainty recognition
        self.metrics.uncertainty_recognition = self._evaluate_uncertainty_recognition(
            self_model_state,
            introspection_results
        )
        
        # Evaluate temporal introspection
        self.metrics.temporal_introspection = self._evaluate_temporal_introspection(
            self_model_state
        )
        
        # Evaluate metacognitive accuracy
        self.metrics.metacognitive_accuracy = self._evaluate_metacognitive_accuracy(
            prediction_history
        )
        
        # Calculate overall score
        overall_score = self.metrics.get_overall_score()
        
        # Store history
        self.history.append({
            'timestamp': time.time(),
            'metrics': self.metrics,
            'overall_score': overall_score
        })
        
        return {
            'self_reflection': self.metrics.self_reflection,
            'belief_updating': self.metrics.belief_updating,
            'attention_awareness': self.metrics.attention_awareness,
            'uncertainty_recognition': self.metrics.uncertainty_recognition,
            'temporal_introspection': self.metrics.temporal_introspection,
            'metacognitive_accuracy': self.metrics.metacognitive_accuracy,
            'overall_score': overall_score
        }
        
    def _evaluate_self_reflection(self, self_model_state: dict, introspection_results: dict) -> float:
        """
        Evaluate the system's ability to reflect on its own state
        
        Measures how well the system can describe its own internal processes,
        identify patterns in its behavior, and recognize its own limitations.
        """
        # Check if system can identify its current emotional state
        emotion_awareness = self._check_emotion_awareness(self_model_state)
        
        # Check if system can identify its knowledge gaps
        knowledge_gap_awareness = self._check_knowledge_gap_awareness(introspection_results)
        
        # Check if system can explain its decision processes
        decision_explanation = self._check_decision_explanation(introspection_results)
        
        return (emotion_awareness + knowledge_gap_awareness + decision_explanation) / 3.0
        
    def _evaluate_belief_updating(self, belief_updates: list[dict]) -> float:
        """
        Evaluate the system's ability to update its beliefs based on new information
        
        Measures how well the system integrates new information, resolves
        contradictions, and adapts its beliefs in response to evidence.
        """
        if not belief_updates:
            return 0.0
            
        # Calculate average magnitude of belief updates
        update_magnitudes = [update.get('magnitude', 0.0) for update in belief_updates]
        avg_magnitude = sum(update_magnitudes) / len(update_magnitudes)
        
        # Calculate proportion of updates that resolve contradictions
        contradiction_resolutions = sum(1 for update in belief_updates 
                                      if update.get('resolves_contradiction', False))
        resolution_ratio = contradiction_resolutions / len(belief_updates)
        
        # Calculate evidence integration score
        evidence_scores = [update.get('evidence_strength', 0.0) for update in belief_updates]
        evidence_score = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0
        
        return (avg_magnitude + resolution_ratio + evidence_score) / 3.0
        
    def _evaluate_attention_awareness(self, self_model_state: dict) -> float:
        """
        Evaluate the system's awareness of its own attention processes
        
        Measures how well the system can monitor and describe what it is
        attending to and why.
        """
        # Check if system maintains an attention schema
        has_attention_schema = 'attention_focus' in self_model_state
        
        # Check attention control capability
        attention_control = self_model_state.get('attention_control_score', 0.0)
        
        # Check awareness of attention shifts
        attention_shift_awareness = self_model_state.get('attention_shift_awareness', 0.0)
        
        return (float(has_attention_schema) + attention_control + attention_shift_awareness) / 3.0
        
    def _evaluate_uncertainty_recognition(self, self_model_state: dict, introspection_results: dict) -> float:
        """
        Evaluate the system's ability to recognize and express uncertainty
        
        Measures how well the system can identify when it lacks knowledge or
        when its beliefs are weakly supported.
        """
        # Check if system expresses confidence levels
        has_confidence = 'confidence_levels' in self_model_state
        
        # Check calibration of confidence (how well confidence matches actual accuracy)
        confidence_calibration = introspection_results.get('confidence_calibration', 0.0)
        
        # Check if system identifies knowledge boundaries
        boundary_awareness = introspection_results.get('knowledge_boundary_awareness', 0.0)
        
        return (float(has_confidence) + confidence_calibration + boundary_awareness) / 3.0
        
    def _evaluate_temporal_introspection(self, self_model_state: dict) -> float:
        """
        Evaluate the system's ability to introspect across time
        
        Measures how well the system can track changes in its own state over time
        and recognize patterns in its development.
        """
        # Check if system maintains temporal continuity
        temporal_continuity = self_model_state.get('temporal_continuity', 0.0)
        
        # Check if system recognizes its own learning
        learning_recognition = self_model_state.get('learning_recognition', 0.0)
        
        # Check if system can project future states
        future_projection = self_model_state.get('future_projection_ability', 0.0)

        return (temporal_continuity + learning_recognition + future_projection) / 3.0

    def _evaluate_metacognitive_accuracy(self, prediction_history: list[dict]) -> float:
        """
        Evaluate how accurately the system predicts its own performance.
        """
        if not prediction_history:
            return 0.0

        correct = 0
        weighted_score = 0.0
        for entry in prediction_history:
            match = entry.get('prediction') == entry.get('outcome')
            confidence = entry.get('confidence', 0.5)
            if match:
                correct += 1
                weighted_score += confidence
            else:
                weighted_score += (1.0 - confidence)

        accuracy = correct / len(prediction_history)
        calibration = weighted_score / len(prediction_history)
        return (accuracy + calibration) / 2.0

    def _check_emotion_awareness(self, self_model_state: dict) -> float:
        """Check if the system can identify its current emotional state."""
        if 'attention_focus' in self_model_state:
            return min(1.0, sum(self_model_state['attention_focus'].values()))
        return 0.0

    def _check_knowledge_gap_awareness(self, introspection_results: dict) -> float:
        """Check if the system can identify its knowledge gaps."""
        return introspection_results.get('knowledge_boundary_awareness', 0.0)

    def _check_decision_explanation(self, introspection_results: dict) -> float:
        """Check if the system can explain its decision processes."""
        return introspection_results.get('confidence_calibration', 0.0)
