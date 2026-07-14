"""Tests for the continuous RSSM latent (Path B1).

The continuous mode replaces the gumbel-softmax categorical z_t with a Gaussian latent of
the SAME [B, categories, classes, grid, grid] shape (prior/posterior nets reused as the
mean, one shared per-channel log-variance added). These tests cover the mechanism:
  - shape parity with the discrete latent (so downstream stages are untouched),
  - the only new parameter is cont_logvar (discrete state_dict is bit-identical),
  - eval is deterministic (returns the mean) and training samples (reparameterization),
  - the gradient reaches the posterior net AND cont_logvar,
  - the tectum forward runs end to end and z_state carries continuous (non-one-hot) values.

Whether the continuous latent can HOLD stimulus identity is decided by the ceiling test +
collapse-locus probe on a trained checkpoint, not by a unit test.
"""
import unittest

import torch

from models.core.sensory_tectum import RSSMCore, SensoryTectum


class TestContinuousRSSMCore(unittest.TestCase):
    def _core(self, mode):
        torch.manual_seed(0)
        return RSSMCore(feature_dim=8, grid_size=4, num_categories=4, num_classes=4,
                        latent_mode=mode)

    def _inputs(self, core, B=2):
        h = torch.zeros(B, core.feature_dim, core.grid_size, core.grid_size)
        z = torch.zeros(B, core.categories, core.classes, core.grid_size, core.grid_size)
        obs = torch.randn(B, core.feature_dim, core.grid_size, core.grid_size)
        return h, z, obs

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            RSSMCore(latent_mode="fuzzy")

    def test_discrete_has_no_logvar_param(self):
        core = self._core("discrete")
        self.assertFalse(any(n == "cont_logvar" for n, _ in core.named_parameters()))

    def test_continuous_adds_only_logvar(self):
        disc = {n for n, _ in self._core("discrete").named_parameters()}
        cont = {n for n, _ in self._core("continuous").named_parameters()}
        self.assertEqual(cont - disc, {"cont_logvar"})
        self.assertEqual(disc - cont, set())

    def test_continuous_z_shape_matches_discrete(self):
        for mode in ("discrete", "continuous"):
            core = self._core(mode)
            core.eval()
            h, z, obs = self._inputs(core)
            h_t, z_t, prior, post = core.step(obs, h, z)
            self.assertEqual(z_t.shape,
                             (2, core.categories, core.classes, core.grid_size, core.grid_size))

    def test_eval_is_deterministic_training_samples(self):
        core = self._core("continuous")
        h, z, obs = self._inputs(core)
        core.eval()
        _, z1, _, _ = core.step(obs, h, z)
        _, z2, _, _ = core.step(obs, h, z)
        self.assertTrue(torch.allclose(z1, z2), "eval latent must be deterministic (mode)")
        core.train()
        _, z3, _, _ = core.step(obs, h, z)
        _, z4, _, _ = core.step(obs, h, z)
        self.assertFalse(torch.allclose(z3, z4), "training latent must be sampled")

    def test_continuous_latent_not_one_hot(self):
        """The discrete latent is one-hot over classes (sums to 1 per category cell);
        the continuous latent is real-valued and generally is not."""
        core = self._core("continuous")
        core.eval()
        h, z, obs = self._inputs(core)
        _, z_t, _, _ = core.step(obs, h, z)
        sums = z_t.sum(dim=2)  # sum over classes
        self.assertFalse(torch.allclose(sums, torch.ones_like(sums)))

    def test_gradient_reaches_posterior_and_logvar(self):
        core = self._core("continuous")
        core.train()
        h, z, obs = self._inputs(core)
        _, z_t, _, _ = core.step(obs, h, z)
        z_t.sum().backward()
        last_post = core.posterior_net[-1]
        self.assertIsNotNone(last_post.weight.grad)
        self.assertGreater(float(last_post.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(core.cont_logvar.grad)
        # logvar grad can be zero only by coincidence; sampling makes it nonzero
        self.assertGreater(float(core.cont_logvar.grad.abs().sum()), 0.0)


class TestContinuousTectumForward(unittest.TestCase):
    def _tectum(self, mode):
        torch.manual_seed(0)
        return SensoryTectum({
            "tectum_feature_dim": 8, "tectum_grid_size": 4, "workspace_dim": 32,
            "rssm_latent_mode": mode,
        })

    def test_forward_runs_and_bid_finite(self):
        tectum = self._tectum("continuous")
        tectum.eval()
        frame = torch.rand(1, 3, 224, 224)
        audio = torch.zeros(1, 8, 2)
        content, bid = tectum(frame, audio)
        self.assertTrue(torch.isfinite(content).all())
        self.assertTrue(float(bid) == float(bid))  # not NaN
        # z_state cached and continuous-shaped
        self.assertEqual(tectum.z_state.shape[1:3], (tectum.rssm.categories, tectum.rssm.classes))

    def test_discrete_tectum_state_dict_has_no_logvar(self):
        keys = set(self._tectum("discrete").state_dict().keys())
        self.assertFalse(any("cont_logvar" in k for k in keys))
        cont_keys = set(self._tectum("continuous").state_dict().keys())
        self.assertEqual(cont_keys - keys, {"rssm.cont_logvar"})


if __name__ == "__main__":
    unittest.main()
