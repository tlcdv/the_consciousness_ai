"""
Test suite for emotional reinforcement learning integration.

Validates:
1. Emotional reward shaping based on attention and arousal
2. Integration between emotional memory and RL
3. Learning progress based on emotional states
"""

import unittest
import torch
import numpy as np
from models.self_model.action_selection_core import ActionSelectionCore
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore, MemoryConfig


class TestEmotionalReinforcement(unittest.TestCase):
    def setUp(self):
        self.config = {
            'workspace_dim': 32,
            'context_dim': 32,
            'action_dim': 8,
            'gamma': 0.99,
            'learning_rate': 0.001,
            'device': 'cpu',
            'emotional_dims': 3,
            'hidden_size': 16,
            'reward': {'base_scale': 1.0},
            'emotional_scale': 2.0,
            'valence_weight': 0.1,
            'dominance_weight': 0.05,
            'arousal_penalty': 0.1,
            'arousal_threshold': 0.8,
        }
        self.emotion_shaper = EmotionalRewardShaper(self.config)
        mem_config = MemoryConfig(max_memories=1000, vector_dim=32, attention_threshold=0.5)
        self.memory = MemoryCore(mem_config)
        self.action_core = ActionSelectionCore(self.config, self.emotion_shaper, self.memory)

    def test_reward_shaping(self):
        """Test emotional reward shaping"""
        positive_state = {'valence': 0.8, 'arousal': 0.5, 'dominance': 0.6}
        negative_state = {'valence': -0.8, 'arousal': 0.9, 'dominance': -0.5}

        reward_pos = self.emotion_shaper.compute_emotional_reward(
            emotion_values=positive_state, base_reward=1.0
        )
        reward_neg = self.emotion_shaper.compute_emotional_reward(
            emotion_values=negative_state, base_reward=1.0
        )
        self.assertGreater(reward_pos, reward_neg)

    def test_emotional_reward_computation(self):
        """Test if emotional rewards are computed correctly"""
        emotion_values = {'valence': 0.8, 'arousal': 0.6, 'dominance': 0.7}
        reward = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_values, base_reward=1.0
        )
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)

    def test_meta_learning_adaptation(self):
        """Test action selection and policy update cycle"""
        state = torch.randn(self.config['workspace_dim'])
        for i in range(12):
            action, _ = self.action_core.select_action(state, emotion_arousal=0.5)
            next_state = torch.randn(self.config['workspace_dim'])
            self.action_core.step(
                workspace_broadcast=state, action=action, raw_reward=0.5,
                next_broadcast=next_state, done=(i == 11),
                emotion_state={'valence': 0.7, 'arousal': 0.5, 'dominance': 0.6},
                attention_level=0.8,
                narrative="Test"
            )
            state = next_state

        result = self.action_core.update_policy()
        self.assertIn('policy_loss', result)
        self.assertIn('value_loss', result)

    def test_memory_integration(self):
        """Test emotional experience storage via step"""
        state = torch.randn(self.config['workspace_dim'])
        action, _ = self.action_core.select_action(state, emotion_arousal=0.5)
        next_state = torch.randn(self.config['workspace_dim'])

        step_result = self.action_core.step(
            workspace_broadcast=state, action=action, raw_reward=0.5,
            next_broadcast=next_state, done=False,
            emotion_state={'valence': 0.8, 'arousal': 0.5, 'dominance': 0.6},
            attention_level=0.7,
            narrative="Test"
        )
        self.assertIn('shaped_reward', step_result)
        self.assertEqual(len(self.action_core.rollout_buffer), 1)

    def test_full_interaction_loop(self):
        """Test complete interaction loop with emotional reinforcement"""
        state = torch.randn(self.config['workspace_dim'])
        total_shaped = 0.0
        for step in range(10):
            action, _ = self.action_core.select_action(state, emotion_arousal=0.5)
            next_state = torch.randn(self.config['workspace_dim'])
            emotion = {
                'valence': float(np.random.uniform(-1, 1)),
                'arousal': float(np.random.uniform(0, 1)),
                'dominance': float(np.random.uniform(-1, 1)),
            }
            result = self.action_core.step(
                workspace_broadcast=state, action=action,
                raw_reward=float(np.random.uniform(0, 1)),
                next_broadcast=next_state, done=(step == 9),
                emotion_state=emotion, attention_level=0.7,
                narrative="Test loop"
            )
            total_shaped += result['shaped_reward']
            state = next_state

        self.assertEqual(len(self.action_core.rollout_buffer), 10)
        self.assertNotEqual(total_shaped, 0.0)


if __name__ == '__main__':
    unittest.main()
