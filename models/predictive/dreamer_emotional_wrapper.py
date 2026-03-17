"""
DreamerV3 Integration Wrapper for Emotional Processing

Implements:
1. Emotional context integration with DreamerV3
2. Dream-based emotion prediction and simulation
3. Emotional reward shaping for world model learning
4. Integration with consciousness development
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass

from models.predictive.dreamerv3_wrapper import DreamerV3Wrapper as DreamerV3
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore
from models.evaluation.consciousness_metrics import ConsciousnessMetrics


@dataclass
class EmotionalMetrics:
    """Tracks emotional learning metrics."""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    reward_history: list[float] = None
    consciousness_score: float = 0.0


@dataclass
class EmotionalDreamState:
    """Tracks emotional state during dream generation."""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    attention: float = 0.0


class DreamerEmotionalWrapper:
    """
    Integrates DreamerV3 with emotional learning.
    """

    def __init__(self, config: dict):
        """Initialize emotional dreamer wrapper."""
        self.config = config

        # Load DreamerV3 with matching config key (use full config as fallback).
        dreamer_cfg = config.get('dreamerV3', config)
        self.dreamer = DreamerV3(dreamer_cfg)

        self.emotion_network = None  # Lazy init when EmotionalGraphNetwork is available
        self.reward_shaper = EmotionalRewardShaper(config)

        # Build MemoryConfig from dict if needed
        from models.memory.memory_core import MemoryConfig
        mem_cfg_dict = config.get('memory_config', config)
        if isinstance(mem_cfg_dict, dict):
            mem_config = MemoryConfig(
                max_memories=mem_cfg_dict.get('max_memories', 10000),
                vector_dim=mem_cfg_dict.get('vector_dim', 128),
                attention_threshold=mem_cfg_dict.get('attention_threshold', 0.5),
            )
        else:
            mem_config = mem_cfg_dict
        self.memory = MemoryCore(mem_config)
        self.consciousness_metrics = ConsciousnessMetrics(config)

        self.dream_state = EmotionalDreamState()
        self.metrics = EmotionalMetrics(reward_history=[])

        # Training parameters.
        self.world_model_lr = config.get('world_model_lr', 1e-4)
        self.actor_lr = config.get('actor_lr', 8e-5)
        self.critic_lr = config.get('critic_lr', 8e-5)
        self.gamma = config.get('gamma', 0.99)

        # Default base_reward if missing in config.
        self.base_reward = float(config.get('base_reward', 1.0))

    def process_interaction(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        emotion_values: dict[str, float],
        done: bool
    ) -> dict:
        """Process interaction with emotional context."""
        self.update_emotional_state(emotion_values)

        # Compute shaped reward using reward_shaper
        arousal = emotion_values.get('arousal', 0.5)
        shaped_reward = self.reward_shaper.compute_reward(
            emotion_values=emotion_values,
            attention_level=arousal,
        )

        self.store_experience(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            emotion_values=emotion_values,
            done=done
        )

        # Compute attention level from emotion values
        arousal = emotion_values.get('arousal', 0.5)
        valence = emotion_values.get('valence', 0.5)
        attention_level = min(1.0, arousal * 0.7 + (1.0 - valence) * 0.3 + 0.1)

        emotional_state = self.get_emotional_state()
        emotional_state['attention_level'] = attention_level

        return {
            'shaped_reward': shaped_reward,
            'emotional_state': emotional_state,
        }

    def compute_reward(self, state: torch.Tensor, emotion_values: dict[str, float],
                       action_info: dict | None = None) -> float:
        """Compute a shaped reward from state and emotion values."""
        valence = emotion_values.get('valence', 0.5)
        arousal = emotion_values.get('arousal', 0.5)
        dominance = emotion_values.get('dominance', 0.5)
        intensity = action_info.get('intensity', 0.5) if action_info else 0.5
        # Base reward from emotional state
        base = valence * 0.5 + dominance * 0.3 + arousal * 0.2
        # Scale by action intensity and config scale
        scale = float(self.config.get('emotional_scale', 2.0))
        shaped = base * scale * (0.5 + 0.5 * intensity)
        return max(0.01, min(scale * 2.0, shaped))

    def update_emotional_state(self, emotion_values: dict[str, float]):
        """Update internal emotional state tracking."""
        self.metrics.valence = emotion_values.get('valence', self.metrics.valence)
        self.metrics.arousal = emotion_values.get('arousal', self.metrics.arousal)
        self.metrics.dominance = emotion_values.get('dominance', self.metrics.dominance)

    def calculate_learning_progress(self) -> float:
        """Calculate recent learning progress from reward history."""
        if not self.metrics.reward_history:
            return 0.0
        recent_rewards = self.metrics.reward_history[-100:]
        return float(np.mean(np.diff(recent_rewards)))

    def store_experience(self, **kwargs):
        """Store experience with emotional context."""
        self.memory.store_experience(
            state=kwargs.get('state', torch.zeros(32)),
            action=kwargs.get('action', torch.zeros(8)),
            reward=kwargs.get('reward', 0.0),
            emotion_values=kwargs.get('emotion_values', {}),
            attention_level=kwargs.get('attention_level', 0.5),
        )
        if 'reward' in kwargs:
            self.metrics.reward_history.append(kwargs['reward'])

    def get_emotional_state(self) -> dict:
        """Return current emotional state."""
        return {
            'valence': self.metrics.valence,
            'arousal': self.metrics.arousal,
            'dominance': self.metrics.dominance,
            'consciousness_score': self.metrics.consciousness_score
        }

    def get_action(
        self,
        state: torch.Tensor,
        emotion_context: dict | None = None
    ) -> torch.Tensor:
        """Get action with optional emotional context."""
        if emotion_context is not None:
            emotional_embedding = self.emotion_network.get_embedding(emotion_context)
            action = self.dreamer.get_action(state, additional_context=emotional_embedding)
        else:
            action = self.dreamer.get_action(state)
        return action

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'dreamer_state': self.dreamer.state_dict(),
            'emotion_network_state': self.emotion_network.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.dreamer.load_state_dict(checkpoint['dreamer_state'])
        self.emotion_network.load_state_dict(checkpoint['emotion_network_state'])
        self.metrics = checkpoint['metrics']
        self.config = checkpoint['config']

    def imagine_trajectory(
        self,
        current_state: torch.Tensor,
        emotional_context: dict[str, float],
        horizon: int = 10
    ) -> tuple[torch.Tensor, dict]:
        """Generate imagined trajectory with emotional context."""
        imagined_trajectory = []
        for _ in range(horizon):
            action = self.get_action(current_state, emotional_context)
            next_state = self.dreamer.predict_next_state(current_state, action)
            imagined_trajectory.append((current_state, action, next_state))
            current_state = next_state
        return imagined_trajectory, self.get_emotional_state()

    def process_dream(
        self,
        dream_state: torch.Tensor,
        emotional_context: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Process dream state with optional emotional context."""
        emotional_features = self.emotion_network.extract_features(dream_state)
        self.dream_state = self._update_dream_state(emotional_features, emotional_context)

        shaped_reward = self._shape_emotional_reward(dream_state, self.dream_state)
        reward_scaling = shaped_reward / self.base_reward

        return shaped_reward, {
            'emotional_state': self.dream_state,
            'reward_scaling': reward_scaling
        }

    def _update_dream_state(
        self,
        emotional_features: torch.Tensor,
        emotional_context: dict | None
    ) -> EmotionalDreamState:
        """Update the dream state using extracted emotional features."""
        updated_state = EmotionalDreamState(
            valence=float(emotional_features[0].item()),
            arousal=float(emotional_features[1].item()) if emotional_features.size(0) > 1 else 0.0,
            dominance=float(emotional_features[2].item()) if emotional_features.size(0) > 2 else 0.0,
            attention=emotional_context.get('attention', 0.0) if emotional_context else 0.0
        )
        return updated_state

    def _shape_emotional_reward(
        self,
        dream_state: torch.Tensor,
        dream_emotional_state: EmotionalDreamState
    ) -> float:
        """Derive an emotional reward from dream state and dream emotional state."""
        # Very basic example: sum of valence and arousal.
        return dream_emotional_state.valence + dream_emotional_state.arousal
