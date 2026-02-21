import unittest
import numpy as np
import torch
from models.self_model.self_representation_core import (
    SelfRepresentationCore,
    SelfState,
    DirectExperienceLearner,
    MetaLearningModule
)

class TestSelfModelExpansion(unittest.TestCase):
    """Tests for the Phase 5 Self-Model biological expansions."""
    
    def setUp(self):
        self.config = {
            "max_history": 10,
            "learning": {"capability_lr": 0.5},
            "meta_learning": {"rpe_window_size": 20}
        }
        self.core = SelfRepresentationCore(self.config)

    def test_self_state_initialization(self):
        """Test that the Phase 5 biological structures initialize correctly."""
        state = self.core.state
        
        # Check explicit interoceptive state
        self.assertIn("energy", state.interoceptive_state)
        self.assertEqual(state.interoceptive_state["energy"], 1.0)
        self.assertEqual(state.interoceptive_state["fatigue"], 0.0)
        
        # Check body schema (10 parts, 8 features each)
        self.assertIsInstance(state.body_schema, torch.Tensor)
        self.assertEqual(state.body_schema.shape, (1, 10, 8))
        
        # Check empty capability model
        self.assertIsInstance(state.capability_model, dict)
        self.assertEqual(len(state.capability_model), 0)

    def test_direct_experience_learner(self):
        """Test that action->outcome mapping updates the capability model (EMA)."""
        learner = DirectExperienceLearner({"capability_lr": 0.5})
        state = SelfState()
        
        # 1. Action is High Magnitude Positive in dimension 0
        action = np.array([1.0, 0.0, 0.0])
        # Outcome is highly positive valence
        emotion = {"valence": 1.0, "arousal": 0.5, "dominance": 0.5}
        
        res = learner(action, emotion, state)
        self.assertEqual(res["action_type"], "move_dim_0_pos")
        
        # Expected valence should move halfway to 1.0 (LR is 0.5)
        self.assertAlmostEqual(state.capability_model["move_dim_0_pos_valence"], 0.5)
        
        # 2. Do it again
        res = learner(action, emotion, state)
        # Expected valence should move halfway from 0.5 to 1.0 -> 0.75
        self.assertAlmostEqual(state.capability_model["move_dim_0_pos_valence"], 0.75)
        
        # 3. Idle action
        idle_action = np.array([0.05, 0.0, 0.0])
        res_idle = learner(idle_action, {"valence": 0.0}, state)
        self.assertEqual(res_idle["action_type"], "idle")
        self.assertAlmostEqual(state.capability_model["idle_valence"], 0.0)

    def test_meta_learning_velocity(self):
        """Test that MetaLearningModule detects dropping variance as learning velocity."""
        meta = MetaLearningModule({"rpe_window_size": 20})
        state = SelfState()
        
        # Initial phase: not enough data
        for i in range(5):
            res = meta(0.5, state)
        self.assertEqual(res["learning_velocity"], 0.0)
        
        # Provide highly variant RPEs indicating confusion/novelty
        variant_rpes = [1.0, -1.0, 0.8, -0.9, 1.2, -1.1, 0.7, -0.8]
        for r in variant_rpes:
            meta(r, state)
            
        # Provide stable/converging RPEs indicating learning is succeeding
        stable_rpes = [0.1, 0.05, 0.08, 0.02, -0.01, 0.03, -0.02, 0.01]
        for r in stable_rpes:
            res = meta(r, state)
            
        # Recent variance should be much lower than overall variance
        self.assertTrue(res["learning_velocity"] > 0)
        self.assertFalse(res["novelty_spike"])
        self.assertEqual(state.learning_recognition, res["learning_velocity"])

    def test_update_self_model_integration(self):
        """Test the full update_self_model loop with action and RPE."""
        action = np.array([0.5, -0.5, 0.0])
        emotion = {"valence": 0.8, "arousal": 0.2, "dominance": 0.5}
        rpe = 0.5
        
        res = self.core.update_self_model(
            current_state={"prediction_outcomes": {}},
            attention_level=0.5,
            action=action,
            emotional_state=emotion,
            rpe=rpe
        )
        
        # Verify sub-components ran
        self.assertIn("direct_update", res)
        self.assertIn("meta_update", res)
        self.assertIn("action_type", res["direct_update"])
        
        # Verify state history tracked it
        self.assertEqual(len(self.core.state_history), 1)

if __name__ == '__main__':
    unittest.main()
