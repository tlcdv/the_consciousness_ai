import unittest
import torch
import numpy as np

from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3
from models.narrative.narrative_engine import NarrativeEngine


class TestEmotionalReinforcementIntegration(unittest.TestCase):
    """Integration tests for the emotional reinforcement learning system."""

    def setUp(self):
        # Restructure config so ReinforcementCore can read keys like 'dreamerV3' directly.
        self.config = {
            'emotional_scale': 2.0,
            'positive_emotion_bonus': 0.5,
            'dreamerV3': {
                'hidden_size': 256,
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'lambda_gae': 0.95
            },
            'meta_config': {
                'enabled': True,
                'adaptation_steps': 5,
                'inner_learning_rate': 0.01
            },
            # Memory capacity for the ReinforcementCore's internal MemoryCore.
            'memory_capacity': 1000,
        }

        # Initialize RL core (which uses DreamerV3, memory, etc.).
        self.rl_core = ReinforcementCore(self.config)

        # Initialize other components individually for integration testing.
        self.emotion_network = EmotionalGraphNetwork()
        self.memory = MemoryCore(capacity=1000)
        self.dreamer = DreamerV3(self.config['dreamerV3'])
        self.narrative = NarrativeEngine()

        # Optionally, you could add a simple placeholder in ReinforcementCore for get_action:
        # def get_action(self, state):
        #     return torch.zeros(8)  # or call self.dreamer.get_action(...) if defined

    def test_end_to_end_learning(self):
        """Test a complete learning cycle with emotional integration."""
        # Mock environment-style state.
        state = torch.randn(32)

        for step in range(10):
            # For the test, define a get_action method or just mock an action here:
            action = torch.randn(8)  # placeholder
            next_state = torch.randn(32)
            reward = float(torch.rand(1).item())

            # Process emotional response.
            emotion_output = self.emotion_network.process_interaction(
                state=state,
                action=action,
                next_state=next_state
            )

            # Compute shaped emotional reward.
            emotional_reward = self.rl_core.compute_reward(
                state=state,
                emotion_values=emotion_output,
                action_info={'step': step}
            )

            # Update RL core.
            update_info = self.rl_core.update(
                state=state,
                action=action,
                reward=emotional_reward,
                next_state=next_state,
                done=(step == 9),
                emotion_context=emotion_output
            )

            # Check that update returns expected keys.
            self.assertIn('world_model_loss', update_info)
            self.assertIn('actor_loss', update_info)
            self.assertIn('critic_loss', update_info)

            state = next_state

    def test_emotional_memory_integration(self):
        """Test that emotional experiences are stored and retrieved."""
        # Create test experiences.
        experiences = []
        for i in range(5):
            exp = {
                'state': torch.randn(32),
                'action': torch.randn(8),
                'emotion': {
                    'valence': 0.7 + 0.1 * i,
                    'arousal': 0.5 + 0.1 * i
                },
                'reward': 0.5 + 0.1 * i,
                'narrative': f"Experience {i} with emotional response"
            }
            experiences.append(exp)
            self.memory.store_experience(exp)

        # Suppose we have a memory method that retrieves experiences by emotion similarity.
        # Adjust if your actual method name or arguments differ.
        query_emotion = {'valence': 0.8, 'arousal': 0.6}
        if hasattr(self.memory, 'get_similar_emotional_experiences'):
            similar_experiences = self.memory.get_similar_emotional_experiences(
                emotion_query=query_emotion, k=3
            )
            self.assertEqual(len(similar_experiences), 3)
            self.assertTrue(all('emotion' in exp for exp in similar_experiences))
        else:
            # Skip or assertNotImplemented if your memory doesn't have such a method.
            self.skipTest("get_similar_emotional_experiences not implemented.")

    def test_meta_learning_adaptation(self):
        """Test meta-learning adaptation to new emotional scenarios."""
        base_scenario = {
            'states': torch.randn(10, 32),
            'actions': torch.randn(10, 8),
            'emotions': torch.randn(10, 3),
            'rewards': torch.randn(10)
        }

        pre_adaptation_perf = self.evaluate_scenario(base_scenario)

        # Adapt to scenario if meta-learning is enabled.
        adaptation_result = self.rl_core.adapt_to_scenario(base_scenario)
        # Possibly check for a known key if your meta-learner returns one.
        # self.assertIn('adapted_params', adaptation_result)

        post_adaptation_perf = self.evaluate_scenario(base_scenario)
        # Check for improvement in some mock metric
        self.assertGreater(
            post_adaptation_perf['emotional_accuracy'],
            pre_adaptation_perf['emotional_accuracy'],
            "Meta-learning adaptation should improve emotional accuracy."
        )

    def evaluate_scenario(self, scenario):
        """Mock scenario evaluation. Returns some performance metrics."""
        total_reward = 0.0
        emotional_correct = 0.0
        count = len(scenario['states'])

        for i in range(count):
            state = scenario['states'][i]
            action = torch.randn(8)  # placeholder
            predicted_emotion = self.emotion_network.predict_emotion(
                state=state,
                action=action
            )

            actual_emotion = scenario['emotions'][i]
            emotional_correct += self.calculate_emotion_accuracy(
                predicted_emotion,
                actual_emotion
            )
            total_reward += scenario['rewards'][i].item()

        return {
            'total_reward': total_reward,
            'emotional_accuracy': emotional_correct / count if count else 0.0
        }

    def calculate_emotion_accuracy(self, predicted, target):
        """Mock method to measure how close predicted emotions are to targets."""
        # If your code returns a dict of floats vs. a tensor, adapt accordingly.
        # Here, we do a simple difference-based measure for valence/arousal.
        if isinstance(predicted, dict):
            accuracy = 0.0
            c = 0
            for key in ['valence', 'arousal']:
                if key in predicted:
                    accuracy += max(0.0, 1.0 - abs(predicted[key] - target[c].item()))
                    c += 1
            return accuracy / (c or 1)
        else:
            # If predicted is a tensor or list, assume the first 2 elements are valence/arousal.
            if len(predicted) >= 2:
                val_err = abs(predicted[0] - target[0].item())
                aro_err = abs(predicted[1] - target[1].item())
                avg_err = (val_err + aro_err) / 2.0
                return max(0.0, 1.0 - avg_err)
            return 0.0


if __name__ == '__main__':
    unittest.main()
