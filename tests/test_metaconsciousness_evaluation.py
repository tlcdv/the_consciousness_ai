import unittest
import sys
import os
import numpy as np

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.evaluation.metaconsciousness_evaluation import MetaconsciousnessEvaluator, MetaconsciousnessMetrics

class TestMetaconsciousnessEvaluation(unittest.TestCase):
    """Tests for the metaconsciousness evaluation system"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            "broadcast_threshold": 0.7,
            "knowledge_update_rate": 0.05,
            "max_history": 10
        }
        self.evaluator = MetaconsciousnessEvaluator(self.config)
        
    def test_belief_updating_evaluation(self):
        """Test evaluation of belief updating capabilities"""
        # Create test belief updates
        belief_updates = [
            {
                'magnitude': 0.8,
                'resolves_contradiction': True,
                'evidence_strength': 0.9
            },
            {
                'magnitude': 0.3,
                'resolves_contradiction': False,
                'evidence_strength': 0.5
            },
            {
                'magnitude': 0.6,
                'resolves_contradiction': True,
                'evidence_strength': 0.7
            }
        ]
        
        # Calculate expected outcome
        avg_magnitude = (0.8 + 0.3 + 0.6) / 3
        resolution_ratio = 2 / 3
        evidence_score = (0.9 + 0.5 + 0.7) / 3
        expected = (avg_magnitude + resolution_ratio + evidence_score) / 3
        
        # Test the function
        result = self.evaluator._evaluate_belief_updating(belief_updates)
        self.assertAlmostEqual(result, expected, places=5)
        
    def test_empty_belief_updates(self):
        """Test handling of empty belief updates"""
        result = self.evaluator._evaluate_belief_updating([])
        self.assertEqual(result, 0.0)
        
    def test_overall_score_calculation(self):
        """Test calculation of overall metaconsciousness score"""
        metrics = MetaconsciousnessMetrics(
            self_reflection=0.7,
            belief_updating=0.8,
            attention_awareness=0.6,
            uncertainty_recognition=0.5,
            temporal_introspection=0.9,
            metacognitive_accuracy=0.7
        )
        
        expected = (0.7 + 0.8 + 0.6 + 0.5 + 0.9 + 0.7) / 6
        self.assertAlmostEqual(metrics.get_overall_score(), expected, places=5)
        
    def test_full_evaluation(self):
        """Test the full evaluation process with mock data"""
        # Create mock data for evaluation
        self_model_state = {
            'attention_focus': {'task': 0.8, 'environment': 0.2},
            'confidence_levels': {'physics': 0.9, 'history': 0.3},
            'attention_control_score': 0.7,
            'attention_shift_awareness': 0.6,
            'temporal_continuity': 0.8,
            'learning_recognition': 0.7,
            'future_projection_ability': 0.5
        }
        
        belief_updates = [
            {'magnitude': 0.7, 'resolves_contradiction': True, 'evidence_strength': 0.8},
            {'magnitude': 0.5, 'resolves_contradiction': False, 'evidence_strength': 0.6}
        ]
        
        introspection_results = {
            'confidence_calibration': 0.7,
            'knowledge_boundary_awareness': 0.6
        }
        
        prediction_history = [
            {'prediction': 'A', 'outcome': 'A', 'confidence': 0.8},
            {'prediction': 'B', 'outcome': 'C', 'confidence': 0.4}
        ]
        
        # Run evaluation
        results = self.evaluator.evaluate_metaconsciousness(
            self_model_state,
            belief_updates,
            introspection_results,
            prediction_history
        )
        
        # Check that all expected metrics are present
        expected_keys = [
            'self_reflection', 'belief_updating', 'attention_awareness',
            'uncertainty_recognition', 'temporal_introspection',
            'metacognitive_accuracy', 'overall_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], float)
            self.assertGreaterEqual(results[key], 0.0)
            self.assertLessEqual(results[key], 1.0)

if __name__ == '__main__':
    unittest.main()