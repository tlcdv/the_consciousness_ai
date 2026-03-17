import unittest
import torch
import numpy as np
from typing import Any

from models.self_model.action_selection_core import ActionSelectionCore
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore, MemoryConfig


class TestEmotionalReinforcementIntegration(unittest.TestCase):
    """Integration tests for the emotional reinforcement learning system."""

    def setUp(self):
        self.config = {
            "workspace_dim": 32,
            "context_dim": 32,
            "action_dim": 4,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "device": "cpu",
            "emotional_dims": 3,
            "hidden_size": 16,
            "reward": {
                "base_scale": 1.0
            },
            "max_memories": 100,
            "cleanup_threshold": 0.4,
            "vector_dim": 32,
            "index_batch_size": 10,
            "attention_threshold": 0.5,
        }

        self.emotion_shaper = EmotionalRewardShaper(self.config)

        mem_config = MemoryConfig(
            max_memories=100,
            vector_dim=32,
            attention_threshold=0.5,
        )
        self.memory = MemoryCore(mem_config)

        self.action_core = ActionSelectionCore(self.config, self.emotion_shaper, self.memory)

    def test_end_to_end_learning(self):
        """Test a complete learning cycle with emotional integration."""
        state = torch.randn(self.config["workspace_dim"])

        for step_i in range(12):
            action, value = self.action_core.select_action(state, emotion_arousal=0.5)
            next_state = torch.randn(self.config["workspace_dim"])
            raw_reward = float(torch.rand(1).item())

            emotion_state = {
                "valence": float(np.random.uniform(-1, 1)),
                "arousal": float(np.random.uniform(0, 1)),
                "dominance": float(np.random.uniform(-1, 1)),
            }

            step_info = self.action_core.step(
                workspace_broadcast=state,
                action=action,
                raw_reward=raw_reward,
                next_broadcast=next_state,
                done=(step_i == 11),
                emotion_state=emotion_state,
                attention_level=0.7,
            )

            self.assertIn("raw_reward", step_info)
            self.assertIn("shaped_reward", step_info)
            state = next_state

        update_info = self.action_core.update_policy()
        self.assertIn("policy_loss", update_info)
        self.assertIn("value_loss", update_info)

    def test_emotional_memory_integration(self):
        """Test that emotional experiences are stored via the RL step."""
        state = torch.randn(self.config["workspace_dim"])
        for i in range(5):
            action, _ = self.action_core.select_action(state, emotion_arousal=0.5)
            next_state = torch.randn(self.config["workspace_dim"])
            emotion_state = {
                "valence": 0.7 + 0.05 * i,
                "arousal": 0.5 + 0.05 * i,
                "dominance": 0.3,
            }

            self.action_core.step(
                workspace_broadcast=state,
                action=action,
                raw_reward=0.5 + 0.1 * i,
                next_broadcast=next_state,
                done=False,
                emotion_state=emotion_state,
                attention_level=0.8,
            )
            state = next_state

        self.assertEqual(len(self.action_core.rollout_buffer), 5)

    def test_reward_shaping(self):
        """Test that emotional reward shaping modulates the base reward."""
        positive_emotion = {"valence": 0.9, "arousal": 0.3, "dominance": 0.5}
        negative_emotion = {"valence": -0.9, "arousal": 0.8, "dominance": -0.5}

        reward_pos = self.emotion_shaper.compute_emotional_reward(
            emotion_values=positive_emotion, base_reward=1.0
        )
        reward_neg = self.emotion_shaper.compute_emotional_reward(
            emotion_values=negative_emotion, base_reward=1.0
        )

        self.assertGreater(reward_pos, reward_neg)


if __name__ == "__main__":
    unittest.main()
