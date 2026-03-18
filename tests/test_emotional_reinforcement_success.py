"""
Test suite for evaluating emotional reinforcement learning success metrics.

Validates that the RL system produces meaningful emotional reward shaping,
stores experiences correctly, and that the narrative engine integrates properly.
"""

import unittest
import torch
import numpy as np
from models.self_model.action_selection_core import ActionSelectionCore
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore, MemoryConfig
from models.narrative.narrative_engine import NarrativeEngine


class MockModel:
    def generate(self, prompt):
        return f"The agent adapted emotionally based on: {prompt}"


class MockMemory:
    def retrieve_relevant(self, input_text):
        return "Relevant emotional memory."


class MockEmotion:
    def analyze(self, input_text):
        return "Positive valence, moderate arousal."


class TestEmotionalReinforcementSuccess(unittest.TestCase):
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

        mock_model = MockModel()
        self.narrative = NarrativeEngine(
            foundational_model=mock_model,
            memory=MockMemory(),
            emotion=MockEmotion(),
            llm=mock_model,
        )

    def test_emotional_memory_formation(self):
        """Test if emotional experiences are stored via RL step"""
        state = torch.randn(self.config['workspace_dim'])
        for i in range(5):
            action, _ = self.action_core.select_action(state, emotion_arousal=0.5)
            next_state = torch.randn(self.config['workspace_dim'])
            self.action_core.step(
                workspace_broadcast=state, action=action, raw_reward=0.5 + 0.1 * i,
                next_broadcast=next_state, done=False,
                emotion_state={'valence': 0.8, 'arousal': 0.6, 'dominance': 0.7},
                attention_level=0.9,
                narrative="Test"
            )
            state = next_state
        self.assertEqual(len(self.action_core.rollout_buffer), 5)

    def test_reward_shaping(self):
        """Test emotional reward shaping mechanism"""
        emotion_values = {'valence': 0.9, 'arousal': 0.7, 'dominance': 0.8}
        reward = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_values, base_reward=1.0
        )
        self.assertGreater(reward, 0)
        self.assertLessEqual(reward, self.config['emotional_scale'] * 2)

    def test_learning_progression(self):
        """Test policy update after enough steps"""
        state = torch.randn(self.config['workspace_dim'])
        for i in range(15):
            action, _ = self.action_core.select_action(state, emotion_arousal=0.5)
            next_state = torch.randn(self.config['workspace_dim'])
            self.action_core.step(
                workspace_broadcast=state, action=action,
                raw_reward=float(np.random.uniform(0, 1)),
                next_broadcast=next_state, done=(i == 14),
                emotion_state={
                    'valence': float(np.random.uniform(0, 1)),
                    'arousal': float(np.random.uniform(0, 1)),
                    'dominance': float(np.random.uniform(0, 1)),
                },
                attention_level=0.7,
                narrative="Test loop"
            )
            state = next_state

        result = self.action_core.update_policy()
        self.assertIn('total_loss', result)

    def test_meta_adaptation(self):
        """Test that select_action returns valid actions"""
        state = torch.randn(self.config['workspace_dim'])
        action, value = self.action_core.select_action(state, emotion_arousal=0.5)
        self.assertEqual(len(action), self.config['action_dim'])
        self.assertIsInstance(value, float)

    def test_narrative_integration(self):
        """Test if emotional experiences generate coherent narratives"""
        input_text = "Agent showed empathy in interaction"
        result = self.narrative.generate_narrative(input_text)
        self.assertIsNotNone(result.text)
        self.assertGreater(len(result.text), 0)


if __name__ == '__main__':
    unittest.main()
