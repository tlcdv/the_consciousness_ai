"""Tests for the value-equivalent world-model objective (Stage 1, model-based path).

Covers the new module (reward/continue heads + balanced KL loss) and the RSSM
action-conditioning, at the mechanism level: shapes, finite losses, gradient reaches the
latent (so it shapes the RSSM, not just the heads), a value test that the latent can
reduce the reward loss, and the baseline-bit-identical guarantee when action-conditioning
is off (action_dim=0). The decisive learning evidence (does it repair working memory)
comes from the leakage-free probe on a trained checkpoint, not a unit test.
"""
import unittest

import torch

from models.core.world_model_objective import WorldModelObjective
from models.core.sensory_tectum import RSSMCore


def _latent(b=2, c=16, g=4):
    return torch.randn(b, c, g, g)


class TestWorldModelObjective(unittest.TestCase):
    def _obj(self, c=16, g=4):
        torch.manual_seed(0)
        return WorldModelObjective(latent_channels=c, grid=g, hidden_dim=32)

    def test_reward_continue_shapes(self):
        o = self._obj()
        self.assertEqual(o.predict_reward(_latent()).shape, (2, 1))
        self.assertEqual(o.predict_continue_logit(_latent()).shape, (2, 1))

    def test_accepts_unbatched_latent(self):
        o = self._obj()
        self.assertEqual(o.predict_reward(torch.randn(16, 4, 4)).shape, (1, 1))

    def test_reward_loss_finite_and_grad_reaches_latent(self):
        o = self._obj()
        lat = _latent()
        lat.requires_grad_(True)
        loss = o.reward_loss(lat, None, torch.randn(2, 1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(lat.grad)
        self.assertGreater(float(lat.grad.abs().sum()), 0.0)

    def test_reward_target_is_stop_grad(self):
        o = self._obj()
        tgt = torch.randn(2, 1, requires_grad=True)
        loss = o.reward_loss(_latent(), None, tgt)
        loss.backward()
        self.assertIsNone(tgt.grad)

    def test_continue_loss_finite(self):
        o = self._obj()
        loss = o.continue_loss(_latent(), torch.ones(2, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_kl_loss_finite_and_grad_reaches_logits(self):
        # logits shape [B, categories, classes, H, W]
        prior = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
        post = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
        loss = WorldModelObjective.kl_loss(prior, post, beta=1.0, free_bits=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(prior.grad.abs().sum() > 0 or post.grad.abs().sum() > 0)

    def test_kl_free_bits_floor(self):
        # identical prior/posterior -> raw KL is 0, but free_bits clamps it to >= floor
        logits = torch.randn(2, 4, 4, 4, 4)
        loss = WorldModelObjective.kl_loss(logits, logits.clone(), beta=1.0, free_bits=1.0)
        self.assertGreater(float(loss), 0.5)  # floored near free_bits, not 0

    def test_latent_can_reduce_reward_loss(self):
        """Value test: optimizing the latent toward a fixed reward lowers the loss,
        confirming the objective puts learnable pressure on the RSSM latent."""
        o = self._obj()
        for p in o.parameters():
            p.requires_grad_(False)  # freeze the heads; only the latent moves
        lat = _latent()
        lat.requires_grad_(True)
        tgt = torch.randn(2, 1)
        opt = torch.optim.Adam([lat], lr=5e-2)
        first = float(o.reward_loss(lat, None, tgt).item())
        for _ in range(200):
            opt.zero_grad()
            loss = o.reward_loss(lat, None, tgt)
            loss.backward()
            opt.step()
        self.assertLess(float(o.reward_loss(lat, None, tgt).item()), first)

    def test_action_conditioned_reward_head(self):
        """With action_dim>0 the reward head is conditioned on the action: different
        actions give different predicted rewards (a DMTS choice reward depends on the
        action), and the gradient reaches the action input."""
        torch.manual_seed(0)
        o = WorldModelObjective(latent_channels=16, grid=4, hidden_dim=32, action_dim=5)
        lat = _latent()
        a1 = torch.zeros(2, 5); a1[:, 0] = 1.0
        a2 = torch.zeros(2, 5); a2[:, 3] = 1.0
        self.assertFalse(torch.allclose(o.predict_reward(lat, a1),
                                        o.predict_reward(lat, a2)))
        a = torch.randn(2, 5, requires_grad=True)
        o.reward_loss(lat, a, torch.randn(2, 1)).backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)


class TestRSSMActionConditioning(unittest.TestCase):
    def _rssm(self, action_dim=0):
        torch.manual_seed(0)
        return RSSMCore(feature_dim=8, grid_size=4, num_categories=4, num_classes=4,
                        action_dim=action_dim)

    def _states(self, b=2, f=8, g=4, cat=4, cls=4):
        obs = torch.randn(b, f, g, g)
        h = torch.zeros(b, f, g, g)
        z = torch.full((b, cat, cls, g, g), 1.0 / cls)
        return obs, h, z

    def test_action_dim_zero_has_no_embed(self):
        r = self._rssm(action_dim=0)
        self.assertIsNone(r.action_embed)

    def test_action_dim_zero_ignores_action_bit_identical(self):
        """With action_dim=0 the step must be identical whether or not an action is
        passed (the baseline path is untouched)."""
        torch.manual_seed(1)
        r = self._rssm(action_dim=0)
        obs, h, z = self._states()
        a = torch.randn(2, 5)
        torch.manual_seed(7)
        h1, z1, pr1, po1 = r.step(obs, h, z, action=None)
        torch.manual_seed(7)
        h2, z2, pr2, po2 = r.step(obs, h, z, action=a)
        self.assertTrue(torch.allclose(h1, h2))
        self.assertTrue(torch.allclose(pr1, pr2))

    def test_action_changes_step_when_enabled(self):
        """With action_dim>0, a non-zero action changes the deterministic state."""
        r = self._rssm(action_dim=5)
        obs, h, z = self._states()
        torch.manual_seed(3)
        h_none, *_ = r.step(obs, h, z, action=None)
        torch.manual_seed(3)
        h_act, *_ = r.step(obs, h, z, action=torch.randn(2, 5))
        self.assertFalse(torch.allclose(h_none, h_act))

    def test_action_gradient_reaches_embed(self):
        r = self._rssm(action_dim=5)
        obs, h, z = self._states()
        a = torch.randn(2, 5, requires_grad=True)
        h_t, *_ = r.step(obs, h, z, action=a)
        h_t.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertGreater(float(a.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
