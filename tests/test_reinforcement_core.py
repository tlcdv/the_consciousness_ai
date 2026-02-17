import unittest
import torch
import numpy as np
from models.self_model.reinforcement_core import ReinforcementCore
from models.memory.memory_core import MemoryCore, MemoryConfig
from models.emotion.reward_shaping import EmotionalRewardShaper

class TestReinforcementCore(unittest.TestCase):
    def setUp(self):
        self.config = {
            'state_dim': 32,
            'action_dim': 8,
            'gamma': 0.99,
            'learning_rate': 0.001,
            'device': 'cpu',
            'emotional_dims': 3,
            'hidden_size': 16,
            'reward': {'base_scale': 1.0},
            'emotional_scale': 2.0,
            'positive_emotion_bonus': 0.5,
        }
        self.emotion_shaper = EmotionalRewardShaper(self.config)
        mem_config = MemoryConfig(max_memories=1000, vector_dim=32, attention_threshold=0.5)
        self.memory = MemoryCore(mem_config)
        self.rl_core = ReinforcementCore(self.config, self.emotion_shaper, self.memory)

    def test_compute_reward(self):
        """Test emotional reward computation via the shaper."""
        emotion_values = {
            'valence': 0.8,
            'arousal': 0.6,
            'dominance': 0.7
        }
        reward = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_values,
            base_reward=1.0
        )
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0.0)

    def test_adaptation(self):
        """Test select_action and step cycle."""
        state = torch.randn(self.config['state_dim'])
        action, value = self.rl_core.select_action(state)
        self.assertEqual(len(action), self.config['action_dim'])
        self.assertIsInstance(value, float)

    def test_memory_integration(self):
        """Test that steps populate the rollout buffer."""
        state = torch.randn(self.config['state_dim'])
        for i in range(5):
            action, _ = self.rl_core.select_action(state)
            next_state = torch.randn(self.config['state_dim'])
            self.rl_core.step(
                state=state,
                action=action,
                raw_reward=0.5,
                next_state=next_state,
                done=False,
                emotion_state={'valence': 0.8, 'arousal': 0.5, 'dominance': 0.6},
                attention_level=0.7,
            )
            state = next_state
        self.assertEqual(len(self.rl_core.rollout_buffer), 5)

if __name__ == '__main__':
    unittest.main()