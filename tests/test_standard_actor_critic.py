"""
Tests for the StandardActorCritic P5 diagnostic policy (A2C on the broadcast).

Mechanism-level only: interface parity with ActionSelectionCore, action/value
shapes, and that update_policy / replay_update run and produce finite losses. The
LEARNING evidence (does a plain A2C on the broadcast beat the Go/No-Go core) comes
from the dark_room comparison run, not a unit test.
"""
import unittest

import numpy as np
import torch

from models.self_model.standard_actor_critic import StandardActorCritic
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore


class TestStandardActorCritic(unittest.TestCase):
    def _core(self, action_dim=2, wd=32):
        torch.manual_seed(0)
        cfg = {"workspace_dim": wd, "action_dim": action_dim, "device": "cpu",
               "learning_rate": 1e-3, "ac_hidden": 32}
        return StandardActorCritic(cfg, EmotionalRewardShaper({}), MemoryCore({}))

    def test_interface_parity_with_action_core(self):
        c = self._core()
        for m in ("select_action", "step", "update_policy", "replay_update", "reset_state"):
            self.assertTrue(callable(getattr(c, m)), f"missing {m}")
        self.assertTrue(hasattr(c, "pfc_hidden"))

    def test_select_action_shape_and_finite(self):
        c = self._core(action_dim=2, wd=32)
        a, v = c.select_action(torch.randn(1, 32), emotion_arousal=0.5)
        self.assertEqual(a.shape[0], 2)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(v))

    def test_step_and_update_policy(self):
        c = self._core(action_dim=2, wd=32)
        emo = {"valence": 0.1, "arousal": 0.5, "dominance": 0.0}
        for _ in range(12):
            b = torch.randn(1, 32)
            a, _ = c.select_action(b)
            c.step(b, a, 1.0, torch.randn(1, 32), False, emo, 0.5)
        out = c.update_policy()
        self.assertIn("total_loss", out)
        self.assertTrue(np.isfinite(out["total_loss"]))
        # buffer cleared after update
        self.assertEqual(len(c.rollout_buffer), 0)

    def test_replay_update_runs(self):
        c = self._core(action_dim=2, wd=32)
        experiences = [
            {"state": torch.randn(32), "action": np.zeros(2, dtype=np.float32), "reward": 0.5}
            for _ in range(6)
        ]
        out = c.replay_update(experiences)
        self.assertIn("replay_total_loss", out)
        self.assertTrue(np.isfinite(out["replay_total_loss"]))


if __name__ == "__main__":
    unittest.main()
