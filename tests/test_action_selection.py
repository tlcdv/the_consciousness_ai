import unittest
import torch
import numpy as np
from models.self_model.action_selection_core import ActionSelectionCore, PrefrontalCortex, BasalGanglia
from models.memory.memory_core import MemoryCore, MemoryConfig
from models.emotion.reward_shaping import EmotionalRewardShaper

class TestPrefrontalCortex(unittest.TestCase):
    """Tests for PFC Working Memory stabilization."""
    
    def setUp(self):
        self.workspace_dim = 32
        self.context_dim = 32
        self.pfc = PrefrontalCortex(self.workspace_dim, self.context_dim)
    
    def test_working_memory_persistence(self):
        """PFC should stabilize rapidly changing inputs into a slowly-evolving context."""
        hidden = torch.zeros(1, self.context_dim)
        
        # Feed the same broadcast repeatedly, hidden should converge
        broadcast = torch.randn(1, self.workspace_dim)
        states = []
        for _ in range(10):
            state, hidden = self.pfc(broadcast, hidden)
            states.append(state.clone())
        
        # Later states should be more similar to each other than early states
        # (working memory settling)
        early_diff = torch.norm(states[1] - states[0]).item()
        late_diff = torch.norm(states[-1] - states[-2]).item()
        self.assertLess(late_diff, early_diff + 0.01)  # Allow tiny tolerance
    
    def test_output_shape(self):
        """PFC output should match context_dim."""
        hidden = torch.zeros(1, self.context_dim)
        broadcast = torch.randn(1, self.workspace_dim)
        state, new_hidden = self.pfc(broadcast, hidden)
        self.assertEqual(state.shape, (1, self.context_dim))
        self.assertEqual(new_hidden.shape, (1, self.context_dim))


class TestBasalGanglia(unittest.TestCase):
    """Tests for BG Go/No-Go/STN pathway logic."""

    def setUp(self):
        torch.manual_seed(0)
        self.context_dim = 32
        self.action_dim = 8
        self.bg = BasalGanglia(self.context_dim, self.action_dim)
    
    def test_go_nogo_dopamine_modulation(self):
        """High dopamine should produce larger action magnitudes than low dopamine."""
        pfc_state = torch.randn(1, self.context_dim)
        
        # Positive RPE (high dopamine) strengthens Go, weakens No-Go
        action_high_da, val_high = self.bg(pfc_state, dopamine_rpe=1.0)
        
        # Negative RPE (low dopamine) weakens Go, strengthens No-Go
        action_low_da, val_low = self.bg(pfc_state, dopamine_rpe=-1.0)
        
        # Value should be the same (critic doesn't depend on dopamine)
        self.assertAlmostEqual(val_high.item(), val_low.item(), places=5)
        
        # High dopamine should generally produce larger absolute action magnitudes
        # (Go pathway is boosted, No-Go is suppressed)
        mag_high = torch.abs(action_high_da).mean().item()
        mag_low = torch.abs(action_low_da).mean().item()
        # Run multiple trials with fixed seed for reproducibility
        torch.manual_seed(99)
        wins = 0
        n_trials = 100
        for _ in range(n_trials):
            pfc_state = torch.randn(1, self.context_dim)
            a_high, _ = self.bg(pfc_state, dopamine_rpe=1.0)
            a_low, _ = self.bg(pfc_state, dopamine_rpe=-1.0)
            if torch.abs(a_high).mean() > torch.abs(a_low).mean():
                wins += 1
        # High dopamine should win the majority of the time
        self.assertGreater(wins, n_trials // 4,
            f"High dopamine should produce larger actions more often. Won {wins}/{n_trials}.")
    
    def test_stn_global_inhibition(self):
        """STN output should be a scalar that gates all action dimensions."""
        pfc_state = torch.randn(1, self.context_dim)
        stn_output = self.bg.stn_pathway(pfc_state)
        
        # STN should produce a single scalar per batch element
        self.assertEqual(stn_output.shape, (1, 1))
        # Should be in [0, 1] range (sigmoid)
        self.assertGreaterEqual(stn_output.item(), 0.0)
        self.assertLessEqual(stn_output.item(), 1.0)
    
    def test_thalamic_relay_shape(self):
        """Thalamic relay should preserve action dimensionality."""
        raw_action = torch.randn(1, self.action_dim)
        output = self.bg.thalamic_relay(raw_action)
        self.assertEqual(output.shape, (1, self.action_dim))
    
    def test_output_shapes(self):
        """Final output should be [B, action_dim] for action and [B, 1] for value."""
        pfc_state = torch.randn(1, self.context_dim)
        action, value = self.bg(pfc_state)
        self.assertEqual(action.shape, (1, self.action_dim))
        self.assertEqual(value.shape, (1, 1))
        # Actions should be bounded in [-1, 1]
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))


class TestActionSelectionCore(unittest.TestCase):
    """Integration tests for the full Action Selection pipeline."""
    
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

    def test_select_action_shape(self):
        """select_action should return a 1D numpy array with shape (action_dim,)."""
        state = torch.randn(self.config['workspace_dim'])  # No batch dim
        action, value = self.action_core.select_action(state)
        self.assertEqual(action.shape, (self.config['action_dim'],))
        self.assertIsInstance(value, float)

    def test_policy_input_dim_defaults_to_workspace_dim(self):
        """Without policy_input_dim the PFC input width equals workspace_dim
        (baseline bit-identical, the GRUCell input size is unchanged)."""
        self.assertEqual(self.action_core.policy_input_dim, self.config['workspace_dim'])
        self.assertEqual(self.action_core.pfc.working_memory.input_size,
                         self.config['workspace_dim'])

    def test_spatial_conv_pfc_processes_flattened_map(self):
        """A conv-front-end PFC accepts the flattened topographic map [B, C*H*W],
        reshapes it, convolves it, and returns a [B, context_dim] state. This is
        the spatial-processing path for --policy-input spatial-conv."""
        shape = (64, 16, 16)
        flat = shape[0] * shape[1] * shape[2]
        pfc = PrefrontalCortex(flat, 32, spatial_conv=True, spatial_shape=shape)
        self.assertIsNotNone(pfc.conv)
        hidden = torch.zeros(2, 32)
        state, new_hidden = pfc(torch.randn(2, flat), hidden)
        self.assertEqual(state.shape, (2, 32))
        self.assertEqual(new_hidden.shape, (2, 32))

    def test_spatial_conv_action_core_runs(self):
        """ActionSelectionCore with policy_spatial_conv builds the conv PFC and
        select_action runs end-to-end on a flattened obs_map input."""
        shape = (64, 16, 16)
        flat = shape[0] * shape[1] * shape[2]
        cfg = dict(self.config)
        cfg['policy_input_dim'] = flat
        cfg['policy_spatial_conv'] = True
        cfg['policy_spatial_shape'] = shape
        core = ActionSelectionCore(cfg, self.emotion_shaper, self.memory)
        self.assertIsNotNone(core.pfc.conv)
        action, value = core.select_action(torch.randn(flat))
        self.assertEqual(action.shape, (cfg['action_dim'],))

    def test_policy_input_dim_override_sizes_pfc(self):
        """With policy_input_dim set (e.g. the --policy-input spatial obs_map tap),
        the PFC accepts that-sized input and select_action runs. Previously the PFC
        was hardcoded to workspace_dim, so a wider tap crashed the Go/No-Go policy."""
        cfg = dict(self.config)
        cfg['policy_input_dim'] = 256
        core = ActionSelectionCore(cfg, self.emotion_shaper, self.memory)
        self.assertEqual(core.pfc.working_memory.input_size, 256)
        state = torch.randn(256)  # wider policy input
        action, value = core.select_action(state)
        self.assertEqual(action.shape, (cfg['action_dim'],))

    def test_emotional_modulation(self):
        """Emotional arousal should scale exploration noise variance."""
        state = torch.randn(1, self.config['workspace_dim'])
        
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

    def test_rpe_calculation(self):
        """Step should return a dopamine RPE value."""
        state = torch.randn(1, self.config['workspace_dim'])
        action, _ = self.action_core.select_action(state)
        next_state = torch.randn(1, self.config['workspace_dim'])
        
        metrics = self.action_core.step(
            workspace_broadcast=state,
            action=action,
            raw_reward=1.0,
            next_broadcast=next_state,
            done=False,
            emotion_state={'valence': 0.5, 'arousal': 0.3, 'dominance': 0.5},
            attention_level=0.7,
        )
        
        self.assertIn("dopamine_rpe", metrics)
        self.assertIn("shaped_reward", metrics)
        self.assertIsInstance(metrics["dopamine_rpe"], float)

    def test_rollout_buffer_population(self):
        """Steps should populate the rollout buffer for training."""
        state = torch.randn(1, self.config['workspace_dim'])
        for i in range(5):
            action, _ = self.action_core.select_action(state)
            next_state = torch.randn(1, self.config['workspace_dim'])
            self.action_core.step(
                workspace_broadcast=state,
                action=action,
                raw_reward=0.5,
                next_broadcast=next_state,
                done=False,
                emotion_state={'valence': 0.8, 'arousal': 0.5, 'dominance': 0.6},
                attention_level=0.7,
            )
            state = next_state
            
        self.assertEqual(len(self.action_core.rollout_buffer), 5)

    def test_update_policy_trains(self):
        """update_policy should compute losses and update weights when buffer is full."""
        state = torch.randn(1, self.config['workspace_dim'])
        # Fill buffer with 12 steps (> 10 minimum)
        for _ in range(12):
            action, _ = self.action_core.select_action(state)
            next_state = torch.randn(1, self.config['workspace_dim'])
            self.action_core.step(
                workspace_broadcast=state,
                action=action,
                raw_reward=np.random.uniform(-1, 1),
                next_broadcast=next_state,
                done=False,
                emotion_state={'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5},
                attention_level=0.5,
            )
            state = next_state
        
        # Capture weights before update
        go_weight_before = self.action_core.bg.direct_pathway[0].weight.data.clone()
        
        metrics = self.action_core.update_policy()
        
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("total_loss", metrics)
        
        # Weights should have changed
        go_weight_after = self.action_core.bg.direct_pathway[0].weight.data
        self.assertFalse(torch.allclose(go_weight_before, go_weight_after),
            "Go pathway weights should change after update_policy")
        
        # Buffer should be cleared
        self.assertEqual(len(self.action_core.rollout_buffer), 0)

    def test_reset_state(self):
        """reset_state should zero out PFC hidden memory."""
        # Run a few actions to populate hidden state
        state = torch.randn(1, self.config['workspace_dim'])
        self.action_core.select_action(state)
        self.action_core.select_action(state)
        
        # Hidden should be non-zero
        self.assertFalse(torch.all(self.action_core.pfc_hidden == 0))
        
        # Reset
        self.action_core.reset_state()
        self.assertTrue(torch.all(self.action_core.pfc_hidden == 0))
        self.assertEqual(self.action_core.last_value, 0.0)


if __name__ == '__main__':
    unittest.main()