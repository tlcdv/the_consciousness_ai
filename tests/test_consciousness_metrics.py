# tests/test_consciousness_metrics.py

import unittest
import torch
import numpy as np
from models.evaluation.consciousness_metrics import ConsciousnessMetrics
from models.evaluation.emotional_rl_metrics import EmotionalRLTracker
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.evaluation.consciousness_monitor import ConsciousnessMonitor

"""
Unit tests for consciousness development metrics.

Tests the following metrics:
1. Emotional awareness scoring
2. Attention stability metrics
3. Memory coherence evaluation
4. Development stage progression
5. Integration with consciousness core

Dependencies:
- models/core/consciousness_core.py for main system
- models/evaluation/consciousness_monitor.py for metrics
- models/memory/emotional_memory_core.py for memory validation
"""

class TestConsciousnessMetrics(unittest.TestCase):
    """Test suite for consciousness development metrics"""
    
    def setUp(self):
        self.config = {
            'emotional_scale': 2.0,
            'emotion_embedding_size': 128,
            'consciousness_thresholds': {
                'emotional_awareness': 0.7,
                'memory_coherence': 0.6,
                'attention_level': 0.8
            },
            'dreamer_config': {
                'hidden_size': 256,
                'learning_rate': 0.0001
            }
        }
        
        self.metrics = ConsciousnessMetrics(self.config)
        self.rl_tracker = EmotionalRLTracker(self.config)
        self.dreamer = DreamerEmotionalWrapper(self.config)
        self.monitor = ConsciousnessMonitor(self.config)
        
    def test_survival_learning(self):
        """Test learning through survival-based experiences"""
        # Simulate stressful scenario
        state = torch.randn(32)
        action = torch.randn(8)
        
        # Create stressed emotional state
        emotion_values = {
            'valence': 0.3,  # Low valence indicating stress
            'arousal': 0.8,  # High arousal
            'dominance': 0.4  # Low dominance
        }
        
        # Process interaction
        result = self.dreamer.process_interaction(
            state=state,
            action=action,
            reward=0.5,
            next_state=torch.randn(32),
            emotion_values=emotion_values,
            done=False
        )
        
        # Verify emotional processing
        self.assertIn('emotional_state', result)
        self.assertIn('shaped_reward', result)
        
        # Verify attention activation
        self.assertTrue(result['emotional_state']['attention_level'] > 0.7)
        
    def test_emotional_memory_formation(self):
        """Test emotional memory formation and retrieval"""
        # Create emotional experience
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'emotion': {
                'valence': 0.8,
                'arousal': 0.6,
                'dominance': 0.7
            },
            'attention_level': 0.9,
            'narrative': "Agent successfully helped human in challenging situation"
        }
        
        # Store experience
        self.metrics.store_experience(experience)
        
        # Retrieve similar experiences
        similar_exp = self.metrics.get_similar_emotional_experiences(
            emotion_query={'valence': 0.7, 'arousal': 0.5},
            k=5
        )
        
        # Verify memory formation
        self.assertTrue(len(similar_exp) > 0)
        self.assertIsNotNone(similar_exp[0].get('emotion'))
        
    def test_consciousness_development(self):
        """Test overall consciousness development metrics"""
        # Create interaction history
        interactions = []
        for _ in range(10):
            interaction = {
                'state': torch.randn(32),
                'action': torch.randn(8),
                'emotion_values': {
                    'valence': np.random.random(),
                    'arousal': np.random.random(),
                    'dominance': np.random.random()
                },
                'attention_level': np.random.random(),
                'reward': np.random.random()
            }
            interactions.append(interaction)
            
        # Evaluate consciousness metrics
        metrics = self.metrics.evaluate_consciousness_development(interactions)
        
        # Verify metrics
        self.assertIn('emotional_awareness', metrics)
        self.assertIn('memory_coherence', metrics)
        self.assertIn('attention_level', metrics)
        self.assertIn('learning_progress', metrics)
        
        # Verify values are within expected ranges
        self.assertTrue(0 <= metrics['emotional_awareness'] <= 1)
        self.assertTrue(0 <= metrics['memory_coherence'] <= 1)
        
    def test_reward_shaping(self):
        """Test emotional reward shaping mechanism"""
        state = torch.randn(32)
        emotion_values = {
            'valence': 0.9,  # Very positive emotion
            'arousal': 0.7,
            'dominance': 0.8
        }
        action_info = {'action_type': 'help_human', 'intensity': 0.8}
        
        shaped_reward = self.dreamer.compute_reward(
            state=state,
            emotion_values=emotion_values,
            action_info=action_info
        )
        
        # Verify reward properties
        self.assertGreater(shaped_reward, 0)
        self.assertLessEqual(
            shaped_reward, 
            self.config['emotional_scale'] * 2
        )

    def test_emotional_awareness_scoring(self):
        """Test emotional awareness metric calculation"""
        test_state = {
            'emotion': {
                'valence': 0.7,
                'arousal': 0.6,
                'dominance': 0.5
            },
            'attention': 0.8
        }
        
        metrics = self.monitor.evaluate_emotional_awareness(test_state)
        
        assert 0.0 <= metrics['emotional_awareness'] <= 1.0, "Emotional awareness out of range"
        assert 'confidence' in metrics, "Confidence score missing"

if __name__ == '__main__':
    unittest.main()