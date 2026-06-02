"""
Tests for the control-relevant representation objective (P5 fix).

Mechanism-level: head output shape, obs_features downsampling shape/range, the loss
is finite, and crucially that the loss gradient reaches the CONTENT input (which is
how it shapes the tectum, not just the head). The LEARNING evidence (does the
objective raise dark_room competence) comes from the ON-vs-OFF run, not a unit test.

The baseline-bit-identical-when-off guarantee is structural in train_rlhf.py (the
control loss is only added inside `if control_repr_head is not None`), so it is not
re-tested here.
"""
import unittest

import torch

from models.core.control_representation import ControlRepresentationHead, obs_features


class TestObsFeatures(unittest.TestCase):
    def test_shape_and_range(self):
        frame = torch.rand(1, 3, 224, 224)
        feats = obs_features(frame, grid=8)
        self.assertEqual(feats.shape, (1, 3 * 8 * 8))
        self.assertTrue((feats >= 0.0).all() and (feats <= 1.0).all())

    def test_accepts_unbatched(self):
        feats = obs_features(torch.rand(3, 32, 32), grid=4)
        self.assertEqual(feats.shape, (1, 3 * 4 * 4))


class TestControlRepresentationHead(unittest.TestCase):
    def _head(self, content_dim=32, action_dim=2, target_dim=192):
        torch.manual_seed(0)
        return ControlRepresentationHead(content_dim, action_dim, target_dim, hidden_dim=32)

    def test_predict_shape(self):
        h = self._head()
        out = h.predict(torch.randn(1, 32), torch.randn(1, 2))
        self.assertEqual(out.shape, (1, 192))

    def test_loss_finite(self):
        h = self._head()
        loss = h.loss(torch.randn(1, 32), torch.randn(1, 2), torch.rand(1, 192))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_gradient_reaches_content(self):
        """The objective must shape the tectum: gradient flows into the content
        input (a detached target would train only the head)."""
        h = self._head()
        content = torch.randn(1, 32, requires_grad=True)
        action = torch.randn(1, 2)
        target = torch.rand(1, 192)
        loss = h.loss(content, action, target)
        loss.backward()
        self.assertIsNotNone(content.grad)
        self.assertGreater(float(content.grad.abs().sum()), 0.0)

    def test_target_is_stop_grad(self):
        """The target must not carry gradient (stop-grad), so the representation
        is not trained to make the target trivially predictable."""
        h = self._head()
        target = torch.rand(1, 192, requires_grad=True)
        loss = h.loss(torch.randn(1, 32), torch.randn(1, 2), target)
        loss.backward()
        self.assertIsNone(target.grad)


if __name__ == "__main__":
    unittest.main()
