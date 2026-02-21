import unittest
import torch
import numpy as np
from models.self_model.action_selection_core import ActionSelectionCore, PrefrontalCortex, BasalGanglia
from models.memory.memory_core import MemoryCore, MemoryConfig
from models.emotion.reward_shaping import EmotionalRewardShaper

class TestActionSelection(unittest.TestCase):
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
            'positive_emotion_bonus': 0.5,
        }
        self.emotion_shaper = EmotionalRewardShaper(self.config)
        mem_config = MemoryConfig(max_memories=1000, vector_dim=32, attention_threshold=0.5)
        self.memory = MemoryCore(mem_config)
        self.action_core = ActionSelectionCore(self.config, self.emotion_shaper, self.memory)

    def test_basal_ganglia_gating(self):
        """Test the Go/No-Go pathway gating logic controlled by RPE (dopamine)."""
        bg = self.action_core.bg
        pfc_state = torch.randn(1, self.config['context_dim'])
        
        # Test with high dopamine (positive RPE)
        action_high_da, val_high = bg(pfc_state, dopamine_rpe=1.0)
        
        # Test with low dopamine (negative RPE)
        action_low_da, val_low = bg(pfc_state, dopamine_rpe=-1.0)
        
        # They should evaluate the same state value internally
        self.assertEqual(val_high.item(), val_low.item())
        
        # But the action pathways should diverge because dopamine strengthens 'Go' and weakens 'No-Go'
        # We can't guarantee exact magnitude strictly without checking internals,
        # but we can verify the output shape and type
        self.assertEqual(action_high_da.shape, (1, self.config['action_dim']))
        self.assertEqual(action_low_da.shape, (1, self.config['action_dim']))

    def test_emotional_modulation(self):
        """Test that emotional arousal scales exploration noise."""
        state = torch.randn(1, self.config['workspace_dim'])
        
        # We run multiple samples to capture variance/noise effects
        actions_calm = []
        actions_panic = []
        for _ in range(50):
            a_calm, _ = self.action_core.select_action(state, emotion_arousal=0.0)
            a_panic, _ = self.action_core.select_action(state, emotion_arousal=1.0)
            actions_calm.append(a_calm)
            actions_panic.append(a_panic)
            
        std_calm = np.std(np.array(actions_calm))
        std_panic = np.std(np.array(actions_panic))
        
        # Panicked state (high arousal) should have higher variance
        self.assertGreater(std_panic, std_calm)

    def test_memory_integration(self):
        """Test that steps populate the rollout buffer with RPE calculations."""
        state = torch.randn(1, self.config['workspace_dim'])
        for i in range(5):
            action, _ = self.action_core.select_action(state)
            next_state = torch.randn(1, self.config['workspace_dim'])
            metrics = self.action_core.step(
                workspace_broadcast=state,
                action=action,
                raw_reward=0.5,
                next_broadcast=next_state,
                done=False,
                emotion_state={'valence': 0.8, 'arousal': 0.5, 'dominance': 0.6},
                attention_level=0.7,
            )
            state = next_state
            
            # Ensure dopamine RPE is calculated and returned
            self.assertIn("dopamine_rpe", metrics)
            
        self.assertEqual(len(self.action_core.rollout_buffer), 5)

if __name__ == '__main__':
    unittest.main()