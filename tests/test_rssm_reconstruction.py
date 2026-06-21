"""
Tests for the R1 world-model reconstruction head (active-inference stage 1).

The head reconstructs the current downsampled frame from the PRE-CAPSULE RSSM latent
(state_tensor = cat([h_t, z_flat])), which is where the collapse-locus probe (confirmed
on a trained tectum 2026-06-21) shows stimulus identity dies. The unit tests cover the
mechanism: reconstruction shape and [0, 1] range, finite loss, the gradient reaching the
LATENT input (which is how it shapes the RSSM, not just the decoder), the stop-grad
target, foreground weighting, and a value test that the head can drive reconstruction
loss DOWN by moving the latent (proving the objective puts learnable pressure on the
RSSM latent).

The decisive learning evidence (does the objective make z_state decodable) comes from
re-running the collapse-locus probe on a --save-tectum checkpoint, not from a unit test.
The baseline-bit-identical-when-off guarantee is structural in train_rlhf.py (the
wm-recon loss is only added inside `if wm_recon_head is not None`), so it is not
re-tested here.
"""
import unittest

import torch
import torch.nn.functional as F

from models.core.rssm_reconstruction import RSSMReconstructionHead


class TestRSSMReconstructionHead(unittest.TestCase):
    def _head(self, rssm_channels=16, grid=8):
        torch.manual_seed(0)
        return RSSMReconstructionHead(rssm_channels, grid=grid,
                                      reduce_channels=8, hidden_dim=32)

    def test_reconstruct_shape_and_range(self):
        h = self._head(rssm_channels=16, grid=8)
        out = h.reconstruct(torch.randn(2, 16, 8, 8))
        self.assertEqual(out.shape, (2, 3 * 8 * 8))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_accepts_unbatched_latent(self):
        h = self._head(rssm_channels=16, grid=4)
        out = h.reconstruct(torch.randn(16, 4, 4))
        self.assertEqual(out.shape, (1, 3 * 4 * 4))

    def test_loss_finite(self):
        h = self._head(rssm_channels=16, grid=8)
        frame = torch.rand(1, 3, 224, 224)
        loss = h.loss(torch.randn(1, 16, 8, 8), frame)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_gradient_reaches_latent(self):
        """The objective must shape the RSSM: the gradient flows into the latent
        input (a detached target trains only the decoder). This is the property
        that distinguishes R1 (source at the RSSM latent) from the FAILED 2026-06-10
        recon (source at the post-collapse tectum_content)."""
        h = self._head(rssm_channels=16, grid=8)
        latent = torch.randn(1, 16, 8, 8, requires_grad=True)
        frame = torch.rand(1, 3, 224, 224)
        loss = h.loss(latent, frame)
        loss.backward()
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

    def test_target_is_stop_grad(self):
        """The downsampled-frame target must not carry gradient (stop-grad)."""
        h = self._head(rssm_channels=16, grid=8)
        frame = torch.rand(1, 3, 224, 224, requires_grad=True)
        loss = h.loss(torch.randn(1, 16, 8, 8), frame)
        loss.backward()
        self.assertIsNone(frame.grad)

    def test_foreground_weighting_finite_and_stopgrad(self):
        """Foreground-weighted loss (the default) is finite, non-negative, and
        stop-grad on the target frame."""
        h = self._head(rssm_channels=16, grid=8)
        frame = torch.rand(1, 3, 224, 224, requires_grad=True)
        loss = h.loss(torch.randn(1, 16, 8, 8), frame, foreground=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNone(frame.grad)

    def test_latent_can_reduce_reconstruction_loss(self):
        """Value test: optimizing the LATENT toward a fixed frame lowers the
        reconstruction loss, confirming the objective puts learnable pressure on
        the RSSM latent (not just on the decoder)."""
        h = self._head(rssm_channels=16, grid=8)
        for p in h.parameters():
            p.requires_grad_(False)  # freeze the decoder; only the latent moves
        frame = torch.rand(1, 3, 224, 224)
        latent = torch.randn(1, 16, 8, 8, requires_grad=True)
        opt = torch.optim.Adam([latent], lr=5e-2)
        first = float(h.loss(latent, frame).item())
        for _ in range(200):
            opt.zero_grad()
            loss = h.loss(latent, frame)
            loss.backward()
            opt.step()
        last = float(h.loss(latent, frame).item())
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
