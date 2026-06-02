"""
Tests for the DQNPolicy P5 confirmation tool (off-policy DQN on the broadcast).

Mechanism-level only: interface parity with ActionSelectionCore, action shape +
finiteness for continuous and discrete configs, that step/update_policy/replay_update
run and produce finite losses, and that the target net tracks the online net. The
LEARNING evidence (does DQN-on-broadcast stay far below DQN-on-pixels) comes from
the dark_room run, not a unit test.
"""
import unittest

import numpy as np
import torch

from models.self_model.dqn_policy import DQNPolicy
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore


class TestDQNPolicy(unittest.TestCase):
    def _core(self, action_dim=2, wd=32, continuous=True):
        torch.manual_seed(0)
        cfg = {"workspace_dim": wd, "action_dim": action_dim, "device": "cpu",
               "learning_rate": 1e-3, "dqn_hidden": 32, "dqn_batch_size": 4,
               "env_continuous": continuous}
        return DQNPolicy(cfg, EmotionalRewardShaper({}), MemoryCore({}))

    def test_interface_parity_with_action_core(self):
        c = self._core()
        for m in ("select_action", "step", "update_policy", "replay_update", "reset_state"):
            self.assertTrue(callable(getattr(c, m)), f"missing {m}")
        self.assertTrue(hasattr(c, "pfc_hidden"))

    def test_continuous_action_shape_and_finite(self):
        c = self._core(action_dim=2, wd=32, continuous=True)
        self.assertEqual(c.n_actions, 9)
        a, v = c.select_action(torch.randn(1, 32), emotion_arousal=0.5)
        self.assertEqual(a.shape[0], 2)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(v))
        self.assertTrue((a >= -1.0).all() and (a <= 1.0).all())

    def test_discrete_action_is_onehot(self):
        c = self._core(action_dim=5, wd=32, continuous=False)
        self.assertEqual(c.n_actions, 5)
        a, _ = c.select_action(torch.randn(1, 32))
        self.assertEqual(a.shape[0], 5)
        self.assertEqual(int(a.sum()), 1)  # one-hot

    def test_step_and_update_policy(self):
        c = self._core(action_dim=2, wd=32, continuous=True)
        emo = {"valence": 0.1, "arousal": 0.5, "dominance": 0.0}
        for _ in range(20):
            b = torch.randn(1, 32)
            a, _ = c.select_action(b)
            c.step(b, a, 1.0, torch.randn(1, 32), False, emo, 0.5)
        out = c.update_policy()
        self.assertIn("total_loss", out)
        self.assertTrue(np.isfinite(out["total_loss"]))
        self.assertGreater(len(c.buffer), 0)

    def test_target_net_updates(self):
        c = self._core(action_dim=2, wd=32, continuous=True)
        c.target_update = 1  # update target every train step
        emo = {"valence": 0.0, "arousal": 0.5, "dominance": 0.0}
        for _ in range(10):
            b = torch.randn(1, 32)
            a, _ = c.select_action(b)
            c.step(b, a, 0.5, torch.randn(1, 32), False, emo, 0.5)
        # After several training steps the target net mirrors the online net.
        for p, tp in zip(c.qnet.parameters(), c.target_net.parameters()):
            self.assertTrue(torch.allclose(p, tp))

    def test_replay_update_runs(self):
        c = self._core(action_dim=2, wd=32, continuous=True)
        experiences = [
            {"state": torch.randn(32), "action": np.zeros(2, dtype=np.float32), "reward": 0.5}
            for _ in range(6)
        ]
        out = c.replay_update(experiences)
        self.assertIn("replay_total_loss", out)
        self.assertTrue(np.isfinite(out["replay_total_loss"]))


if __name__ == "__main__":
    unittest.main()
