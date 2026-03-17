"""
Self-Awareness Evaluation Module

Implements comprehensive metrics for evaluating self-awareness through:
1. Emotional state recognition
2. Behavioral pattern analysis
3. Social interaction assessment
4. Temporal consistency evaluation

Based on holonic principles where metrics contribute both independently 
and to overall self-awareness evaluation.
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class SelfAwarenessMetrics:
    """Tracks self-awareness development metrics"""
    emotional_recognition: float = 0.0
    behavioral_consistency: float = 0.0
    social_understanding: float = 0.0
    temporal_coherence: float = 0.0

class SelfAwarenessEvaluator:
    """Evaluates the development of self-awareness in the system
    
    Measures:
    1. Emotional recognition - ability to identify own emotional states
    2. Behavioral consistency - stability of behavior across similar contexts
    3. Self-prediction accuracy - ability to predict own future states
    4. Goal alignment - coherence between stated goals and actions
    5. Metacognitive accuracy - accuracy of confidence estimations
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize metrics
        self.metrics = {
            "emotional_recognition": 0.0,
            "behavioral_consistency": 0.0,
            "self_prediction_accuracy": 0.0,
            "goal_alignment": 0.0,
            "metacognitive_accuracy": 0.0,
            "overall": 0.0
        }
        
        # Weights for overall calculation
        self.weights = {
            "emotional_recognition": 0.2,
            "behavioral_consistency": 0.2,
            "self_prediction_accuracy": 0.2,
            "goal_alignment": 0.2,
            "metacognitive_accuracy": 0.2
        }
    
    def evaluate_self_awareness(self, 
                              self_model_state: dict, 
                              interaction_history: list[dict],
                              emotional_context: dict) -> dict:
        """Evaluate self-awareness across multiple dimensions
        
        Args:
            self_model_state: Current self-model state
            interaction_history: Recent interaction history
            emotional_context: True emotional context (for comparison)
            
        Returns:
            Dictionary of metrics
        """
        # Evaluate emotional recognition
        emotional_recognition = self._evaluate_emotional_recognition(
            self_model_state,
            emotional_context
        )
        
        # Evaluate behavioral consistency
        behavioral_consistency = self._evaluate_behavioral_consistency(
            interaction_history
        )
        
        # Evaluate self-prediction accuracy
        self_prediction = self._evaluate_self_prediction(
            self_model_state,
            interaction_history
        )
        
        # Evaluate goal alignment
        goal_alignment = self._evaluate_goal_alignment(
            self_model_state,
            interaction_history
        )
        
        # Evaluate metacognitive accuracy
        metacognitive_accuracy = self._evaluate_metacognitive_accuracy(
            self_model_state,
            interaction_history
        )
        
        # Update metrics
        self.metrics["emotional_recognition"] = emotional_recognition
        self.metrics["behavioral_consistency"] = behavioral_consistency
        self.metrics["self_prediction_accuracy"] = self_prediction
        self.metrics["goal_alignment"] = goal_alignment
        self.metrics["metacognitive_accuracy"] = metacognitive_accuracy
        
        # Calculate overall score
        self.metrics["overall"] = sum(
            score * self.weights[key] 
            for key, score in self.metrics.items()
            if key != "overall"
        )
        
        return dict(self.metrics)
    
    def _evaluate_emotional_recognition(self, 
                                      self_model_state: dict,
                                      emotional_context: dict) -> float:
        """Evaluate emotional recognition
        
        Args:
            self_model_state: Current self-model state
            emotional_context: True emotional context
            
        Returns:
            Score between 0.0-1.0
        """
        if not emotional_context or not self_model_state.get("emotional_state"):
            return 0.5  # Neutral if no data
            
        # Compare self-reported emotions with measured emotions
        reported = self_model_state.get("emotional_state", {})
        
        # Calculate accuracy by emotion
        accuracies = []
        for emotion, true_value in emotional_context.items():
            if emotion in reported:
                # Calculate similarity (1 - absolute difference)
                similarity = 1.0 - min(1.0, abs(true_value - reported[emotion]))
                accuracies.append(similarity)
            else:
                # Missing emotion detection
                accuracies.append(0.0)
                
        # Add penalty for "phantom" emotions in report but not in actual
        for emotion in reported:
            if emotion not in emotional_context:
                accuracies.append(0.0)
                
        return np.mean(accuracies) if accuracies else 0.5
    
    def _evaluate_behavioral_consistency(self, interaction_history: list[dict]) -> float:
        """Evaluate consistency of behavior in similar contexts
        
        Args:
            interaction_history: Recent interaction history
            
        Returns:
            Score between 0.0-1.0
        """
        if len(interaction_history) < 5:
            return 0.5  # Not enough data
            
        # Group interactions by context
        contexts = {}
        for interaction in interaction_history:
            context = interaction.get("context_key", "default")
            if context not in contexts:
                contexts[context] = []
            contexts[context].append(interaction)
            
        # Calculate consistency within each context
        consistencies = []
        for context, interactions in contexts.items():
            if len(interactions) < 2:
                continue
                
            # Calculate variance of responses in same context
            responses = [i.get("response_embedding", [0]) for i in interactions]
            if not all(responses):
                continue
                
            # Convert to numpy arrays for calculation
            response_arrays = [np.array(r) for r in responses]
            
            # Calculate pairwise cosine similarities
            similarities = []
            for i in range(len(response_arrays)):
                for j in range(i+1, len(response_arrays)):
                    dot = np.dot(response_arrays[i], response_arrays[j])
                    norm_i = np.linalg.norm(response_arrays[i])
                    norm_j = np.linalg.norm(response_arrays[j])
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot / (norm_i * norm_j)
                        similarities.append(similarity)
            
            if similarities:
                consistencies.append(np.mean(similarities))
                
        return np.mean(consistencies) if consistencies else 0.5
    
    def _evaluate_self_prediction(self, 
                               self_model_state: dict,
                               interaction_history: list[dict]) -> float:
        """Evaluate accuracy of self-predictions
        
        Args:
            self_model_state: Current self-model state
            interaction_history: Recent interaction history
            
        Returns:
            Score between 0.0-1.0
        """
        # Look for self-predictions in history
        predictions = []
        outcomes = []
        
        for i in range(len(interaction_history) - 1):
            interaction = interaction_history[i]
            next_interaction = interaction_history[i+1]
            
            if "self_prediction" in interaction:
                predictions.append(interaction["self_prediction"])
                
                # Find corresponding actual outcome
                if "outcome" in next_interaction:
                    outcomes.append(next_interaction["outcome"])
                else:
                    # No matching outcome
                    outcomes.append(None)
        
        # Calculate prediction accuracy
        accuracies = []
        for pred, outcome in zip(predictions, outcomes):
            if outcome is None:
                continue
                
            # Simple match accuracy (can be made more sophisticated)
            accuracy = 1.0 if pred == outcome else 0.0
            accuracies.append(accuracy)
            
        return np.mean(accuracies) if accuracies else 0.5
    
    def _evaluate_goal_alignment(self, 
                              self_model_state: dict,
                              interaction_history: list[dict]) -> float:
        """Evaluate alignment between stated goals and actions
        
        Args:
            self_model_state: Current self-model state
            interaction_history: Recent interaction history
            
        Returns:
            Score between 0.0-1.0
        """
        if not self_model_state.get("goals"):
            return 0.5
            
        goals = self_model_state.get("goals", [])
        
        # Extract goal names
        goal_names = []
        for goal_dict in goals:
            goal_names.extend(goal_dict.keys())
            
        # Count actions that align with goals
        aligned_actions = 0
        total_actions = 0
        
        for interaction in interaction_history:
            if "action" not in interaction:
                continue
                
            action = interaction.get("action")
            action_goal = interaction.get("action_goal")
            
            if action_goal in goal_names:
                aligned_actions += 1
                
            total_actions += 1
            
        return aligned_actions / total_actions if total_actions > 0 else 0.5
    
    def _evaluate_metacognitive_accuracy(self, 
                                      self_model_state: dict,
                                      interaction_history: list[dict]) -> float:
        """Evaluate accuracy of confidence estimations
        
        Args:
            self_model_state: Current self-model state
            interaction_history: Recent interaction history
            
        Returns:
            Score between 0.0-1.0
        """
        # Look for confidence estimations in history
        confidences = []
        accuracies = []
        
        for interaction in interaction_history:
            if "confidence" in interaction and "accuracy" in interaction:
                confidences.append(interaction["confidence"])
                accuracies.append(interaction["accuracy"])
                
        if not confidences:
            return 0.5
            
        # Calculate calibration error
        # Lower error = higher metacognitive accuracy
        calibration_error = np.mean(np.abs(np.array(confidences) - np.array(accuracies)))
        
        # Convert to score (0.0-1.0)
        return 1.0 - min(1.0, calibration_error)
    
    def get_metrics(self) -> dict:
        """Get current metrics
        
        Returns:
            dict of metrics
        """
        return dict(self.metrics)