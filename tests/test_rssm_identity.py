"""
Tests for the latent identity gate (Path B stage 1 diagnostic ceiling test).

The head classifies the DMTS sample's shape and color from the PRE-CAPSULE RSSM latent
(state_tensor = cat([h_t, z_flat])), the confirmed collapse locus. The unit tests cover
the mechanism: logit shapes, finite loss, the gradient reaching the LATENT input (which
is how the objective pressures the RSSM, not just the classifier), accuracy semantics,
the label-map consistency with the stimulus renderer, and a value test that the loss can
be driven DOWN by moving the latent (proving the objective puts learnable identity
pressure on the RSSM latent).

The decisive evidence (does supervised pressure make z_state decodable) comes from
re-running the collapse-locus probe on a --save-tectum checkpoint, not from a unit test.
The baseline-bit-identical-when-off guarantee is structural in train_rlhf.py (the
latent-id loss is only added inside `if latent_id_head is not None`), so it is not
re-tested here, matching test_rssm_reconstruction.py.
"""
import unittest

import torch

from models.core.rssm_identity import RSSMIdentityHead
from simulations.environments._stimulus_renderer import SHAPE_NAMES, COLOR_NAMES


class TestRSSMIdentityHead(unittest.TestCase):
    def _head(self, rssm_channels=16, grid=8, num_shapes=6, num_colors=6):
        torch.manual_seed(0)
        return RSSMIdentityHead(rssm_channels, grid=grid, reduce_channels=8,
                                hidden_dim=32, num_shapes=num_shapes,
                                num_colors=num_colors)

    def test_forward_shapes(self):
        h = self._head(rssm_channels=16, grid=8)
        shape_logits, color_logits = h(torch.randn(2, 16, 8, 8))
        self.assertEqual(shape_logits.shape, (2, 6))
        self.assertEqual(color_logits.shape, (2, 6))

    def test_accepts_unbatched_latent(self):
        h = self._head(rssm_channels=16, grid=4)
        shape_logits, color_logits = h(torch.randn(16, 4, 4))
        self.assertEqual(shape_logits.shape, (1, 6))
        self.assertEqual(color_logits.shape, (1, 6))

    def test_loss_finite_and_accuracy_bounded(self):
        h = self._head(rssm_channels=16, grid=8)
        loss, acc = h.loss(torch.randn(1, 16, 8, 8), 2, 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        # acc is the mean of two argmax hits: one of {0.0, 0.5, 1.0}.
        self.assertIn(acc, (0.0, 0.5, 1.0))

    def test_gradient_reaches_latent(self):
        """The objective must shape the RSSM: the gradient flows into the latent
        input. A classifier that only trained its own weights would put no
        identity pressure on the upstream RSSM."""
        h = self._head(rssm_channels=16, grid=8)
        latent = torch.randn(1, 16, 8, 8, requires_grad=True)
        loss, _ = h.loss(latent, 1, 3)
        loss.backward()
        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

    def test_label_maps_match_renderer(self):
        """The default head sizes must match the renderer's label vocabularies,
        and the names must be unique (the train-loop index maps depend on it)."""
        self.assertEqual(len(SHAPE_NAMES), 6)
        self.assertEqual(len(COLOR_NAMES), 6)
        self.assertEqual(len(set(SHAPE_NAMES)), len(SHAPE_NAMES))
        self.assertEqual(len(set(COLOR_NAMES)), len(COLOR_NAMES))

    def test_latent_can_reduce_identity_loss(self):
        """Value test: optimizing the LATENT toward fixed labels with a frozen
        head lowers the CE, confirming the objective puts learnable identity
        pressure on the RSSM latent (not just on the classifier)."""
        h = self._head(rssm_channels=16, grid=8)
        for p in h.parameters():
            p.requires_grad_(False)  # freeze the classifier; only the latent moves
        latent = torch.randn(1, 16, 8, 8, requires_grad=True)
        opt = torch.optim.Adam([latent], lr=5e-2)
        first = float(h.loss(latent, 0, 5)[0].item())
        for _ in range(200):
            opt.zero_grad()
            loss, _ = h.loss(latent, 0, 5)
            loss.backward()
            opt.step()
        last = float(h.loss(latent, 0, 5)[0].item())
        self.assertLess(last, first)

    def test_distinct_labels_get_distinct_pressure(self):
        """Gradients for two different shape labels on the same latent must
        differ; otherwise the objective could not separate identities."""
        h = self._head(rssm_channels=16, grid=8)
        latent_a = torch.randn(1, 16, 8, 8)
        la = latent_a.clone().requires_grad_(True)
        lb = latent_a.clone().requires_grad_(True)
        loss_a, _ = h.loss(la, 0, 0)
        loss_a.backward()
        loss_b, _ = h.loss(lb, 3, 0)
        loss_b.backward()
        self.assertGreater(float((la.grad - lb.grad).abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
