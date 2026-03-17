"""
Reinforcement Learning Metrics for Emotional Development

Tracks:
1. Emotional learning curves
2. Reward shaping evaluation
3. Policy adaptation metrics
4. Consciousness integration
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class EmotionalMetrics:
    """Stores emotional learning metrics."""
    emotional_awareness: float = 0.0
    reward_stability: float = 0.0
    learning_progress: float = 0.0
    memory_coherence: float = 0.0
    narrative_consistency: float = 0.0


@dataclass
class EmotionalRLMetrics:
    emotional_reward: float = 0.0
    policy_adaptation: float = 0.0  
    learning_stability: float = 0.0
    exploration_ratio: float = 0.0
    consciousness_alignment: float = 0.0


class EmotionalRLTracker:
    """
    Tracks and analyzes emotional RL metrics.
    """

    def __init__(self, config: dict):
        self.config = config
        self.reward_history = deque(maxlen=1000)
        self.emotion_history = deque(maxlen=1000)
        self.narrative_history = deque(maxlen=100)

        # Thresholds from config.
        self.reward_stability_threshold = config.get('reward_stability_threshold', 0.1)
        self.emotional_awareness_threshold = config.get('emotional_awareness_threshold', 0.7)

    def update(self, metrics: dict) -> EmotionalMetrics:
        """
        Update tracker with new data.
        
        Args:
            metrics: dict containing fields like 'reward', 'emotion_values', 'narrative'.
        
        Returns:
            EmotionalMetrics with updated calculations.
        """
        if 'reward' in metrics:
            self.reward_history.append(metrics['reward'])
        if 'emotion_values' in metrics:
            self.emotion_history.append(metrics['emotion_values'])
        if 'narrative' in metrics:
            self.narrative_history.append(metrics['narrative'])

        current_metrics = EmotionalMetrics(
            emotional_awareness=self._calculate_emotional_awareness(),
            reward_stability=self._calculate_reward_stability(),
            learning_progress=self._calculate_learning_progress(),
            memory_coherence=self._calculate_memory_coherence(),
            narrative_consistency=self._calculate_narrative_consistency()
        )
        return current_metrics

    def _calculate_emotional_awareness(self) -> float:
        """Evaluate continuity across consecutive emotional states."""
        if len(self.emotion_history) < 2:
            return 0.0

        scores = []
        for i in range(len(self.emotion_history) - 1):
            curr = self.emotion_history[i]
            nxt = self.emotion_history[i + 1]
            continuity = 1.0 - np.mean([abs(curr[k] - nxt[k]) for k in curr.keys()])
            scores.append(continuity)

        return float(np.mean(scores))

    def _calculate_reward_stability(self) -> float:
        """Compute stability of recent rewards."""
        if len(self.reward_history) < 10:
            return 0.0
        recent = list(self.reward_history)[-10:]
        return float(1.0 / (1.0 + np.std(recent)))

    def _calculate_learning_progress(self) -> float:
        """Estimate slope of reward trends."""
        if len(self.reward_history) < 100:
            return 0.0
        x = np.arange(len(self.reward_history))
        y = np.array(self.reward_history)
        slope = np.polyfit(x, y, 1)[0]
        return float(1.0 / (1.0 + np.exp(-10 * slope)))

    def _calculate_memory_coherence(self) -> float:
        """Use emotional continuity as a stand-in for memory coherence."""
        if len(self.emotion_history) < 10:
            return 0.0

        scores = []
        for i in range(len(self.emotion_history) - 1):
            curr = self.emotion_history[i]
            nxt = self.emotion_history[i + 1]
            continuity = 1.0 - np.mean([abs(curr[k] - nxt[k]) for k in curr.keys()])
            scores.append(continuity)

        return float(np.mean(scores))

    def _calculate_narrative_consistency(self) -> float:
        """Compute textual overlap as a stand-in for narrative consistency."""
        if len(self.narrative_history) < 2:
            return 0.0

        consistency_scores = []
        for i in range(len(self.narrative_history) - 1):
            curr = self.narrative_history[i].split()
            nxt = self.narrative_history[i + 1].split()
            overlap = len(set(curr) & set(nxt))
            union = len(set(curr) | set(nxt))
            if union > 0:
                consistency_scores.append(overlap / union)

        return float(np.mean(consistency_scores))

    def get_summary(self) -> dict:
        """Return current metrics and thresholds checks."""
        current = self.update({})
        return {
            'emotional_awareness': current.emotional_awareness,
            'reward_stability': current.reward_stability,
            'learning_progress': current.learning_progress,
            'memory_coherence': current.memory_coherence,
            'narrative_consistency': current.narrative_consistency,
            'meets_thresholds': self._check_thresholds(current)
        }

    def _check_thresholds(self, metrics: EmotionalMetrics) -> bool:
        """
        Check if current metrics meet minimum thresholds.
        """
        return (
            metrics.emotional_awareness >= self.emotional_awareness_threshold
            and metrics.reward_stability >= self.reward_stability_threshold
            and metrics.learning_progress > 0
        )


class EmotionalRLEvaluator:
    """
    Evaluates higher-level emotional RL metrics.
    """

    def __init__(self, config: dict):
        self.config = config
        self.metrics = EmotionalRLMetrics()
        self.history = []

    def evaluate_learning(
        self,
        episode_data: dict,
        emotion_values: dict[str, float],
        policy_info: dict
    ) -> dict:
        """
        Evaluate RL performance with emotional focus.
        
        Args:
            episode_data: Contains 'rewards', 'losses', etc.
            emotion_values: Emotional signals.
            policy_info: Info about current policy or network.
        
        Returns:
            Updated metrics dictionary.
        """
        emotional_reward = self._calculate_emotional_reward(
            episode_data.get('rewards', []),
            emotion_values
        )
        policy_adaptation = self._evaluate_policy_adaptation(
            policy_info,
            emotion_values
        )
        learning_stability = self._calculate_learning_stability(
            episode_data.get('losses', [])
        )
        exploration_ratio = self._calculate_exploration_ratio(
            policy_info
        )
        consciousness_alignment = self._calculate_consciousness_alignment(
            emotion_values
        )

        self.metrics.emotional_reward = emotional_reward
        self.metrics.policy_adaptation = policy_adaptation
        self.metrics.learning_stability = learning_stability
        self.metrics.exploration_ratio = exploration_ratio
        self.metrics.consciousness_alignment = consciousness_alignment

        updated = self.get_metrics()
        self.history.append(updated)
        return updated

    def _calculate_emotional_reward(
        self,
        rewards: list[float],
        emotion_values: dict[str, float]
    ) -> float:
        """Compute emotional reward, adjusting raw rewards by an emotional factor."""
        raw_mean = np.mean(rewards) if rewards else 0.0
        valence = emotion_values.get('valence', 0.5)
        # Example: multiply by valence as a placeholder.
        return float(raw_mean * valence)

    def _evaluate_policy_adaptation(
        self,
        policy_info: dict,
        emotion_values: dict[str, float]
    ) -> float:
        """Assess how the policy adapts under emotional influence."""
        # Placeholder uses 'policy_entropy' and 'arousal' as example.
        policy_entropy = policy_info.get('policy_entropy', 0.0)
        arousal = emotion_values.get('arousal', 0.5)
        return float(policy_entropy * arousal)

    def _calculate_learning_stability(self, losses: list[float]) -> float:
        """Compute stability from variance of recent losses."""
        if len(losses) < 5:
            return 0.0
        return float(1.0 / (1.0 + np.std(losses[-5:])))

    def _calculate_exploration_ratio(self, policy_info: dict) -> float:
        """
        Placeholder for exploration ratio, e.g., fraction of random actions.
        """
        return float(policy_info.get('exploration_ratio', 0.0))

    def _calculate_consciousness_alignment(self, emotion_values: dict[str, float]) -> float:
        """
        Dummy alignment measure with a single emotional dimension.
        """
        dominance = emotion_values.get('dominance', 0.5)
        return dominance

    def get_metrics(self) -> dict[str, float]:
        """Return the current RL metrics."""
        return {
            'emotional_reward': self.metrics.emotional_reward,
            'policy_adaptation': self.metrics.policy_adaptation,
            'learning_stability': self.metrics.learning_stability,
            'exploration_ratio': self.metrics.exploration_ratio,
            'consciousness_alignment': self.metrics.consciousness_alignment
        }

    def calculate_consciousness_alignment(self, emotion_values: dict[str, float]) -> float:
        """
        Calculate how well the emotional responses align with consciousness development
        This quantifies if emotional responses are becoming more nuanced over time
        """
        # No history to compare against
        if len(self.history) < 10:
            return 0.0
            
        # Get recent emotion distributions
        recent_emotions = [h.get('emotion_distribution', {}) for h in self.history[-10:]]
        
        # Calculate entropy increase (more diverse emotional responses)
        initial_entropy = self._calculate_emotion_entropy(recent_emotions[0])
        current_entropy = self._calculate_emotion_entropy(recent_emotions[-1])
        
        # Higher entropy suggests more nuanced emotional understanding
        entropy_change = current_entropy - initial_entropy
        
        # Calculate valence stability (more consistent emotional evaluations)
        valences = [e.get('valence', 0) for e in recent_emotions if 'valence' in e]
        valence_stability = 1.0 / (1.0 + np.std(valences)) if valences else 0.0
        
        # Combine metrics (higher is better)
        return 0.5 * (np.clip(entropy_change, 0, 1) + valence_stability)
