import unittest
import torch
import sys
import os
import numpy as np

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.evaluation.levin_consciousness_metrics import LevinConsciousnessEvaluator, LevinConsciousnessMetrics

class TestLevinConsciousnessMetrics(unittest.TestCase):
    """Tests for the Levin-inspired consciousness evaluation metrics"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            "bioelectric": {
                "field_dimension": 32,
                "bioelectric_channels": 4
            },
            "holonic": {
                "num_holons": 4,
                "integration_heads": 2
            }
        }
        self.evaluator = LevinConsciousnessEvaluator(self.config)
        
    def test_bioelectric_complexity_evaluation(self):
        """Test evaluation of bioelectric field complexity"""
        # Create mock bioelectric state
        bioelectric_state = {
            'memory': torch.randn(4, 32),
            'attention': torch.randn(4, 32),
            'narrative': torch.randn(4, 32)
        }
        
        # Test the function
        result = self.evaluator.evaluate_bioelectric_complexity(bioelectric_state)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        
    def test_empty_bioelectric_state(self):
        """Test handling of empty bioelectric state"""
        result = self.evaluator.evaluate_bioelectric_complexity({})
        self.assertEqual(result, 0.0)
        
    def test_collective_intelligence_evaluation(self):
        """Test evaluation of collective intelligence through holonic integration"""
        # Create mock holonic output
        holonic_output = {
            'attention_weights': torch.softmax(torch.randn(4, 4), dim=1),
            'holon_states': torch.randn(4, 128)
        }
        
        # Test the function
        result = self.evaluator.evaluate_collective_intelligence(holonic_output)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
        
    def test_full_levin_consciousness_evaluation(self):
        """Test the full Levin consciousness evaluation process with mock data"""
        # Create mock data for evaluation
        bioelectric_state = {
            'memory': torch.randn(4, 32),
            'attention': torch.randn(4, 32)
        }
        
        holonic_output = {
            'attention_weights': torch.softmax(torch.randn(4, 4), dim=1),
            'holon_states': torch.randn(4, 128),
            'integrated_state': torch.randn(1, 128)
        }
        
        past_states = [
            {'integrated_state': torch.randn(1, 128)} for _ in range(5)
        ]
        
        current_state = {
            'integrated_state': torch.randn(1, 128)
        }
        
        actions = [
            {'embedding': torch.randn(64)} for _ in range(3)
        ]
        
        goals = [
            {'embedding': torch.randn(64)} for _ in range(3)
        ]
        
        outcomes = [
            {'embedding': torch.randn(64)} for _ in range(3)
        ]
        
        component_states = {
            'memory': torch.randn(32),
            'attention': torch.randn(32)
        }
        
        # Run evaluation
        results = self.evaluator.evaluate_levin_consciousness(
            bioelectric_state,
            holonic_output,
            past_states,
            current_state,
            actions,
            goals,
            outcomes,
            component_states
        )
        
        # Check that all expected metrics are present
        expected_keys = [
            'bioelectric_complexity', 
            'morphological_adaptation',
            'collective_intelligence', 
            'goal_directed_behavior',
            'basal_cognition',
            'overall_levin_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], float)
            self.assertGreaterEqual(results[key], 0.0)
            self.assertLessEqual(results[key], 1.0)

if __name__ == '__main__':
    unittest.main()