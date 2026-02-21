import unittest
import numpy as np
import torch
from models.self_model.self_representation_core import (
    SelfRepresentationCore,
    SelfState,
    DirectExperienceLearner,
    MetaLearningModule
)

class TestSelfStateInitialization(unittest.TestCase):
    """Test that the Phase 5 biological structures initialize correctly."""
    
    def test_body_schema_shape(self):
        state = SelfState()
        self.assertIsInstance(state.body_schema, torch.Tensor)
        self.assertEqual(state.body_schema.shape, (1, 10, 8))

    def test_interoceptive_defaults(self):
        state = SelfState()
        self.assertEqual(state.interoceptive_state["energy"], 1.0)
        self.assertEqual(state.interoceptive_state["damage"], 0.0)
        self.assertEqual(state.interoceptive_state["fatigue"], 0.0)

    def test_capability_model_empty(self):
        state = SelfState()
        self.assertIsInstance(state.capability_model, dict)
        self.assertEqual(len(state.capability_model), 0)
    
    def test_emotional_state_has_pad(self):
        state = SelfState()
        self.assertIn("valence", state.emotional_state)
        self.assertIn("arousal", state.emotional_state)
        self.assertIn("dominance", state.emotional_state)


class TestDirectExperienceLearner(unittest.TestCase):
    """Test action → emotion capability EMA tracking."""
    
    def setUp(self):
        self.learner = DirectExperienceLearner({"capability_lr": 0.5})
        self.state = SelfState()
    
    def test_ema_update_positive(self):
        """Positive valence should increase expected valence for the action type."""
        action = np.array([1.0, 0.0, 0.0])
        emotion = {"valence": 1.0}
        
        res = self.learner(action, emotion, self.state)
        self.assertEqual(res["action_type"], "move_dim_0_pos")
        self.assertAlmostEqual(self.state.capability_model["move_dim_0_pos_valence"], 0.5)
        
        # Second call: 0.5 + 0.5*(1.0 - 0.5) = 0.75
        self.learner(action, emotion, self.state)
        self.assertAlmostEqual(self.state.capability_model["move_dim_0_pos_valence"], 0.75)

    def test_idle_action(self):
        """Very small action magnitude should be classified as idle."""
        action = np.array([0.05, 0.0, 0.0])
        res = self.learner(action, {"valence": 0.0}, self.state)
        self.assertEqual(res["action_type"], "idle")
    
    def test_none_action_returns_empty(self):
        res = self.learner(None, {"valence": 0.5}, self.state)
        self.assertEqual(res, {})


class TestMetaLearningModule(unittest.TestCase):
    """Test RPE variance-based learning velocity detection."""
    
    def setUp(self):
        self.meta = MetaLearningModule({"rpe_window_size": 30})
        self.state = SelfState()
    
    def test_insufficient_data(self):
        """Should return 0 velocity until enough data collected."""
        for i in range(5):
            res = self.meta(0.5, self.state)
        self.assertEqual(res["learning_velocity"], 0.0)
        self.assertFalse(res["novelty_spike"])

    def test_learning_convergence(self):
        """When RPE variance drops, velocity should be positive."""
        # Phase 1: chaotic RPEs (high variance)
        for r in [1.0, -1.0, 0.8, -0.9, 1.2, -1.1, 0.7, -0.8, 1.0, -0.5]:
            self.meta(r, self.state)
            
        # Phase 2: stable RPEs (learning)
        for r in [0.1, 0.05, 0.08, 0.02, -0.01, 0.03, -0.02, 0.01, 0.0, 0.01]:
            res = self.meta(r, self.state)
            
        self.assertGreater(res["learning_velocity"], 0)
        self.assertFalse(res["novelty_spike"])
        self.assertEqual(self.state.learning_recognition, res["learning_velocity"])


class TestSelfRepresentationCore(unittest.TestCase):
    """Integration tests for the full self-model update loop."""
    
    def setUp(self):
        self.config = {
            "max_history": 10,
            "learning": {"capability_lr": 0.5},
            "meta_learning": {"rpe_window_size": 20}
        }
        self.core = SelfRepresentationCore(self.config)

    def test_interoceptive_dynamics(self):
        """Energy should deplete with action over multiple steps."""
        initial_energy = self.core.state.interoceptive_state["energy"]
        
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        emotion = {"valence": 0.5, "arousal": 0.7, "dominance": 0.3}
        
        for _ in range(50):
            self.core.update_self_model(
                current_state={},
                attention_level=0.5,
                action=action,
                emotional_state=emotion,
                rpe=0.1
            )
        
        final_energy = self.core.state.interoceptive_state["energy"]
        # Energy should have depleted
        self.assertLess(final_energy, initial_energy)
        
        # Fatigue should have accumulated (high arousal)
        self.assertGreater(self.core.state.interoceptive_state["fatigue"], 0.0)
    
    def test_emotional_state_synced(self):
        """update_self_model should sync emotion into SelfState."""
        emotion = {"valence": 0.9, "arousal": 0.1, "dominance": 0.7}
        self.core.update_self_model(
            current_state={},
            attention_level=0.5,
            emotional_state=emotion
        )
        self.assertEqual(self.core.state.emotional_state["valence"], 0.9)
        self.assertEqual(self.core.state.emotional_state["dominance"], 0.7)

    def test_history_is_snapshot_not_reference(self):
        """History should store snapshots, not references to the live state."""
        emotion1 = {"valence": 0.5, "arousal": 0.3, "dominance": 0.5}
        self.core.update_self_model(
            current_state={}, attention_level=0.5, emotional_state=emotion1
        )
        
        # Change the state
        emotion2 = {"valence": -0.9, "arousal": 0.9, "dominance": -0.5}
        self.core.update_self_model(
            current_state={}, attention_level=0.5, emotional_state=emotion2
        )
        
        # The first history entry should still have the original values
        first_snapshot = self.core.state_history[0]
        self.assertAlmostEqual(first_snapshot["emotional_state"]["valence"], 0.5)
        # The second should have the new values
        second_snapshot = self.core.state_history[1]
        self.assertAlmostEqual(second_snapshot["emotional_state"]["valence"], -0.9)

    def test_temporal_continuity_updates(self):
        """Temporal continuity should be non-zero after multiple updates."""
        emotion = {"valence": 0.5, "arousal": 0.3, "dominance": 0.5}
        for _ in range(5):
            self.core.update_self_model(
                current_state={}, attention_level=0.5, emotional_state=emotion
            )
        # Should have non-zero continuity after multiple consistent updates
        self.assertGreater(self.core.state.temporal_continuity, 0.0)

    def test_get_current_state_includes_phase5_fields(self):
        """get_current_state() must expose body_schema, interoceptive, capability."""
        state_dict = self.core.get_current_state()
        self.assertIn("body_schema_shape", state_dict)
        self.assertIn("interoceptive_state", state_dict)
        self.assertIn("capability_model", state_dict)
        self.assertEqual(state_dict["body_schema_shape"], [1, 10, 8])

    def test_update_body_schema(self):
        """update_body_schema should replace the body tensor."""
        new_schema = torch.ones(1, 10, 8)
        self.core.update_body_schema(new_schema)
        self.assertTrue(torch.all(self.core.state.body_schema == 1.0))
    
    def test_full_update_returns_all_sections(self):
        """Full update should return direct, meta, interoceptive, epistemic, temporal."""
        action = np.array([0.5, -0.5])
        emotion = {"valence": 0.8, "arousal": 0.2, "dominance": 0.5}
        res = self.core.update_self_model(
            current_state={}, attention_level=0.5,
            action=action, emotional_state=emotion, rpe=0.3
        )
        self.assertIn("direct_update", res)
        self.assertIn("meta_update", res)
        self.assertIn("interoceptive_update", res)
        self.assertIn("epistemic_update", res)
        self.assertIn("temporal_update", res)


if __name__ == '__main__':
    unittest.main()
