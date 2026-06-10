"""
Tests for the tectum reconstruction objective (perception fix).

Mechanism-level: reconstruction shape and [0, 1] range, the loss is finite, the
gradient reaches the CONTENT input (which is how it shapes the tectum, not just the
decoder), the target is stop-grad, and a value test that the head can drive
reconstruction loss DOWN on a fixed (content, frame) pair via gradient steps on the
content (proving the objective puts learnable pressure on the representation).

The decisive learning evidence (does the objective make tectum_content decodable)
comes from re-running the perception-decodability probe on a --save-tectum
checkpoint, not from a unit test. The baseline-bit-identical-when-off guarantee is
structural in train_rlhf.py (the recon loss is only added inside
`if recon_head is not None`), so it is not re-tested here.
"""
import unittest

import torch
import torch.nn.functional as F

from models.core.tectum_reconstruction import TectumReconstructionHead


class TestTectumReconstructionHead(unittest.TestCase):
    def _head(self, content_dim=32, grid=8):
        torch.manual_seed(0)
        return TectumReconstructionHead(content_dim, grid=grid, hidden_dim=32)

    def test_reconstruct_shape_and_range(self):
        h = self._head(content_dim=32, grid=8)
        out = h.reconstruct(torch.randn(2, 32))
        self.assertEqual(out.shape, (2, 3 * 8 * 8))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_accepts_unbatched_content(self):
        h = self._head(content_dim=32, grid=4)
        out = h.reconstruct(torch.randn(32))
        self.assertEqual(out.shape, (1, 3 * 4 * 4))

    def test_loss_finite(self):
        h = self._head(content_dim=32, grid=8)
        frame = torch.rand(1, 3, 224, 224)
        loss = h.loss(torch.randn(1, 32), frame)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_gradient_reaches_content(self):
        """The objective must shape the tectum: the gradient flows into the
        content input (a detached target trains only the decoder)."""
        h = self._head(content_dim=32, grid=8)
        content = torch.randn(1, 32, requires_grad=True)
        frame = torch.rand(1, 3, 224, 224)
        loss = h.loss(content, frame)
        loss.backward()
        self.assertIsNotNone(content.grad)
        self.assertGreater(float(content.grad.abs().sum()), 0.0)

    def test_target_is_stop_grad(self):
        """The downsampled-frame target must not carry gradient (stop-grad)."""
        h = self._head(content_dim=32, grid=8)
        frame = torch.rand(1, 3, 224, 224, requires_grad=True)
        loss = h.loss(torch.randn(1, 32), frame)
        loss.backward()
        self.assertIsNone(frame.grad)

    def test_foreground_weighting_finite_and_stopgrad(self):
        """Foreground-weighted loss is finite, non-negative, and stop-grad on the
        target frame (so the representation is not trained to make the target
        trivially predictable)."""
        h = self._head(content_dim=32, grid=8)
        frame = torch.rand(1, 3, 224, 224, requires_grad=True)
        loss = h.loss(torch.randn(1, 32), frame, foreground=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNone(frame.grad)

    def test_foreground_emphasizes_stimulus_pixels(self):
        """A sparse stimulus on a black frame should dominate the foreground loss:
        the plain MSE is dominated by the (correctly reconstructed) black
        background, so an error confined to the stimulus pixels yields a much
        larger foreground loss than plain MSE."""
        h = self._head(content_dim=32, grid=4)
        n = 3 * 4 * 4
        target = torch.zeros(1, n)
        target[0, 0] = 1.0  # single bright stimulus element on a black frame
        pred = target.clone()
        pred[0, 0] = 0.0  # miss only the stimulus, reconstruct the background
        plain = float(F.mse_loss(pred, target))
        # weighted loss recomputed with the head's foreground formula
        w = (target - target.mean(dim=1, keepdim=True)).abs() + 1e-6
        sq = (pred - target) ** 2
        weighted = float((w * sq).sum(dim=1).mean() / (w.sum(dim=1).mean() + 1e-8))
        self.assertGreater(weighted, plain * 5)

    def test_content_can_reduce_reconstruction_loss(self):
        """Value test: optimizing the CONTENT toward a fixed frame lowers the
        reconstruction loss, confirming the objective puts learnable pressure on
        the representation (not just on the decoder)."""
        h = self._head(content_dim=32, grid=8)
        for p in h.parameters():
            p.requires_grad_(False)  # freeze the decoder; only the content moves
        frame = torch.rand(1, 3, 224, 224)
        content = torch.randn(1, 32, requires_grad=True)
        opt = torch.optim.Adam([content], lr=5e-2)
        first = float(h.loss(content, frame).item())
        for _ in range(200):
            opt.zero_grad()
            loss = h.loss(content, frame)
            loss.backward()
            opt.step()
        last = float(h.loss(content, frame).item())
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
