"""Tests for the vanilla DQN baseline."""
from __future__ import annotations

import unittest
import torch
import numpy as np
from scripts.training.train_baseline_dqn import DQN, ReplayBuffer, frame_to_tensor


class TestDQN(unittest.TestCase):

    def test_forward_produces_q_values(self):
        net = DQN(action_dim=5)
        x = torch.randn(1, 3, 224, 224)
        q = net(x)
        self.assertEqual(q.shape, (1, 5))

    def test_batch_forward(self):
        net = DQN(action_dim=4)
        x = torch.randn(8, 3, 224, 224)
        q = net(x)
        self.assertEqual(q.shape, (8, 4))

    def test_no_nan_output(self):
        net = DQN(action_dim=5)
        x = torch.randn(2, 3, 224, 224)
        q = net(x)
        self.assertFalse(torch.isnan(q).any())


class TestReplayBuffer(unittest.TestCase):

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            s = torch.randn(3, 224, 224)
            buf.push(s, i % 5, float(i), s, False)
        self.assertEqual(len(buf), 10)
        states, actions, rewards, next_states, dones = buf.sample(4)
        self.assertEqual(states.shape[0], 4)
        self.assertEqual(actions.shape[0], 4)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(torch.zeros(3, 8, 8), 0, 0.0, torch.zeros(3, 8, 8), False)
        self.assertEqual(len(buf), 5)


class TestFrameConversion(unittest.TestCase):

    def test_frame_to_tensor_shape(self):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        t = frame_to_tensor(frame)
        self.assertEqual(t.shape, (1, 3, 224, 224))
        self.assertTrue(t.max() <= 1.0)
        self.assertTrue(t.min() >= 0.0)


if __name__ == "__main__":
    unittest.main()
