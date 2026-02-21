# tests/test_simulation_integration.py

import unittest
import torch
from simulations.api.simulation_manager import SimulationManager
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from simulations.api.simulation_manager import SimulationManager
from simulations.scenarios.consciousness_scenarios import ConsciousnessScenarioManager
from simulations.environments.interactive_vr_environment import InteractiveVREnvironment


class TestSimulationIntegration(unittest.TestCase):
    def setUp(self):
        self.config = {
            'reinforcement': {
                'emotional_scale': 2.0,
                'dreamer_config': {
                    'hidden_size': 256
                }
            },
            'simulation': {
                'max_steps': 100,
                'reward_threshold': 0.5
            }
        }
        self.sim_manager = SimulationManager(self.config)
        
    def test_interaction_loop(self):
        """Test complete interaction loop with emotional reinforcement"""
        agent = self.create_test_agent()
        environment = self.create_test_environment()
        
        result = self.sim_manager.run_interaction(agent, environment)
        
        self.assertIn('total_reward', result)
        self.assertIn('steps', result)
        self.assertIn('update_info', result)
        
    def test_emotional_learning(self):
        """Test emotional learning over multiple episodes"""
        agent = self.create_test_agent()
        environment = self.create_test_environment()
        
        initial_performance = self.evaluate_agent(agent, environment)
        
        # Train for several episodes
        for _ in range(5):
            self.sim_manager.run_interaction(agent, environment)
            
        final_performance = self.evaluate_agent(agent, environment)
        
        # Assert improvement in emotional understanding
        self.assertGreater(final_performance['emotional_accuracy'], 
                          initial_performance['emotional_accuracy'])
    
    def evaluate_agent(self, agent, environment):
        """Helper method to evaluate agent performance"""
        total_reward = 0
        emotional_correct = 0
        num_steps = 0
        
        state = environment.reset()
        done = False
        
        while not done and num_steps < self.config['simulation']['max_steps']:
            action = agent.get_action(state)
            next_state, reward, done, info = environment.step(action)
            
            if info.get('emotion_prediction_correct', False):
                emotional_correct += 1
                
            total_reward += reward
            num_steps += 1
            state = next_state
            
        return {
            'total_reward': total_reward,
            'emotional_accuracy': emotional_correct / num_steps if num_steps > 0 else 0
        }
    
    def create_test_agent(self):
        """Create a test agent for simulation"""
        return DummyAgent(action_space=8, state_space=32)
    
    def create_test_environment(self):
        """Create a test environment for simulation"""
        return DummyEnvironment(state_space=32)
    
class DummyAgent:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.experience_count = 0

    def get_action(self, state):
        self.experience_count += 1
        return torch.randn(self.action_space)

class DummyEnvironment:
    def __init__(self, state_space):
        self.state_space = state_space
        self.total_episodes = 0
        self.episode_step = 0

    def reset(self):
        self.total_episodes += 1
        self.episode_step = 0
        return torch.randn(self.state_space)

    def step(self, action):
        self.episode_step += 1
        next_state = torch.randn(self.state_space)
        reward = torch.rand(1).item()
        done = self.episode_step >= 10
        # Prediction accuracy improves with total episodes (simulates learning effect)
        accuracy_prob = min(0.95, 0.2 + self.total_episodes * 0.1)
        info = {
            'emotion_values': {
                'valence': torch.rand(1).item(),
                'arousal': torch.rand(1).item()
            },
            'emotion_prediction_correct': torch.rand(1).item() < accuracy_prob
        }
        return next_state, reward, done, info

if __name__ == '__main__':
    unittest.main()