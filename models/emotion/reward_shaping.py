"""
Emotional reward shaping for consciousness development.
Integrates with LLaMA 3.3 narrative states and meta-memory system.

Key features:
- Emotion-based reward modulation
- Meta-memory reinforcement
- Controlled adaptation rates
- Narrative coherence rewards
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork


@dataclass
class RewardMetrics:
    """Track reward shaping metrics."""
    emotional_coherence: float = 0.0
    memory_influence: float = 0.0
    narrative_alignment: float = 0.0
    adaptation_rate: float = 0.0


class EmotionalRewardShaper(nn.Module):
    """
    Shapes rewards based on emotional responses and learning progress.
    """

    def __init__(self, config: dict):
        """
        Initializes the reward shaping system.

        Args:
            config: Dictionary containing reward shaping parameters:
                - 'emotional_dims': Size of the input emotion vector
                - 'hidden_size': Embedding dimension
                - 'reward': Sub-dict with 'base_scale', 'memory_influence', 'coherence_weight'
        """
        super().__init__()
        self.config = config

        # Core components.

        emotional_dims = config.get('emotional_dims', 3)
        hidden_size = config.get('hidden_size', 64)

        self.emotion_encoder = nn.Linear(emotional_dims, hidden_size)

        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Configuration.
        reward_cfg = config.get('reward', {})
        self.base_reward_scale = reward_cfg.get('base_scale', 1.0)
        self.memory_influence = reward_cfg.get('memory_influence', 0.5)
        self.coherence_weight = reward_cfg.get('coherence_weight', 0.5)

        # Metrics tracking.
        self.metrics = RewardMetrics()

        self.valence_weight = config.get("valence_weight", 0.1)
        self.dominance_weight = config.get("dominance_weight", 0.05)
        # Homeostatic arousal term: penalise deviation from optimal arousal level.
        # Formula: -arousal_lambda * (arousal - arousal_target)^2
        # Target ~0.3 keeps the agent in calm-curious state without catatonia.
        # Setting target to 0.0 and lambda to 0.1 reproduces the old threshold penalty
        # only at extreme arousal, but the homeostatic form is theoretically correct.
        self.arousal_lambda = config.get("arousal_lambda", 0.1)
        self.arousal_target = config.get("arousal_target", 0.3)

        # Existence-bias ablation (Metzinger ethics, default off). When True,
        # compute_emotional_reward drops the homeostatic arousal penalty and the
        # dominance/agency term (the "stay alive / in control" survival terms),
        # keeping external reward plus the valence term. Baseline is
        # bit-identical when False. See docs/ethics_framework.md.
        self.ablate_existence_bias = bool(config.get("ablate_existence_bias", False))

    def compute_reward(
        self,
        emotion_values: dict[str, float],
        attention_level: float,
        meta_memory: dict | None = None
    ) -> float:
        """
        Compute the shaped reward based on emotional context.

        Args:
            emotion_values: dict of emotional signals (valence, arousal, etc.).
            attention_level: Current attention level or weighting factor.
            meta_memory: Additional memory-based data or patterns.

        Returns:
            Shaped reward value (float).
        """
        # Encode emotions into embeddings
        emotional_embedding = self._encode_emotions(emotion_values)
        base_reward = self._calculate_base_reward(emotional_embedding)

        # Apply memory influence
        if meta_memory:
            memory_gate_val = self._calculate_memory_influence(
                emotional_embedding, 
                meta_memory
            )
            base_reward *= (1.0 + memory_gate_val)

        # Modulate by attention
        return base_reward * (1.0 + attention_level)

    def _encode_emotions(self, emotion_values: dict[str, float]) -> torch.Tensor:
        """
        Encode emotional values into a tensor for further processing.
        Placeholder logic; adjust as needed.
        """
        # Example: sorted keys for deterministic ordering.
        keys = sorted(emotion_values.keys())
        vec = torch.tensor([emotion_values[k] for k in keys], dtype=torch.float).unsqueeze(0)
        return self.emotion_encoder(vec).squeeze(0)

    def _calculate_base_reward(self, emotional_embedding: torch.Tensor) -> float:
        """
        Derive a base reward from the emotional embedding.
        Placeholder logic: sum the embedding and scale.
        """
        base_val = torch.sum(emotional_embedding).item()
        return float(base_val * self.base_reward_scale)

    def _calculate_memory_influence(
        self,
        emotional_embedding: torch.Tensor,
        meta_memory: dict
    ) -> float:
        """
        Compute how meta-memory influences the reward.
        Placeholder logic: feed combined embedding to a gating net.
        """
        # Dummy memory embedding from meta_memory; or your real approach.
        memory_vec = torch.tensor(
            [meta_memory.get('stability_score', 0.5)],
            dtype=torch.float
        )
        combined = torch.cat([emotional_embedding, memory_vec], dim=0)
        gate_val = self.memory_gate(combined.unsqueeze(0)).squeeze(0).item()
        return float(gate_val * self.memory_influence)

    def _update_metrics(
        self,
        emotional_embedding: torch.Tensor,
        base_reward: float,
        attention_level: float
    ) -> None:
        """
        Update reward shaping metrics with placeholder logic.
        """
        self.metrics.emotional_coherence = float(torch.norm(emotional_embedding).item())
        self.metrics.memory_influence = float(base_reward)
        self.metrics.narrative_alignment = 0.0  # Adjust if you integrate narratives.
        self.metrics.adaptation_rate = attention_level

    def compute_emotional_reward(self, emotion_values, base_reward=1.0, context=None):
        """
        Compute shaped reward from full PAD state using homeostatic formulation.

        Formula (published thesis, corrected):
            Rtotal = Rext
                   + λ1 * ΔValence                         (reward positive affect)
                   - λ2 * (Arousal - Arousal_target)^2     (homeostatic arousal penalty)
                   + λ3 * Dominance                        (reward sense of agency)

        The old threshold-based arousal penalty is replaced with a quadratic
        homeostatic term. This lets the agent seek moderate arousal (curiosity/
        exploration) rather than minimising arousal entirely (catatonia).
        Dominance encodes agency: it distinguishes anger from fear (both are
        high-arousal/negative-valence, but only fear is submissive).

        Args:
            emotion_values: dict with keys 'valence', 'arousal', 'dominance'.
            base_reward: External task reward (Rext).
            context: Optional dict with 'emotional_history' and 'adaptation_detected'.
        """
        reward = base_reward

        # λ1 * ΔValence: reward increases in positive affect.
        if 'valence' in emotion_values:
            reward += emotion_values['valence'] * self.valence_weight

        # λ3 * Dominance: reward sense of control and agency.
        # Gated by the existence-bias ablation: dominance/agency is part of the
        # "stay in control" survival drive.
        if 'dominance' in emotion_values and not self.ablate_existence_bias:
            reward += emotion_values['dominance'] * self.dominance_weight

        # -λ2 * (Arousal - Arousal_target)^2: homeostatic arousal.
        # Gated by the existence-bias ablation: this is the homeostatic survival
        # term (keep arousal near a viable set point).
        if 'arousal' in emotion_values and not self.ablate_existence_bias:
            deviation = emotion_values['arousal'] - self.arousal_target
            reward -= self.arousal_lambda * (deviation ** 2)

        # Valence trend bonus: reward improvement over recent history.
        if context and 'emotional_history' in context and len(context['emotional_history']) > 5:
            recent = context['emotional_history'][-5:]
            valence_trend = sum(e.get('valence', 0) for e in recent) / len(recent)
            valence_delta = emotion_values.get('valence', 0) - valence_trend
            reward += valence_delta * self.config.get('trend_weight', 0.1)

        # Adaptation bonus when learning progress is detected.
        if context and context.get('adaptation_detected', False):
            reward += self.config.get('adaptation_bonus', 0.2)

        return reward
