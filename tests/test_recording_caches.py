"""Values the session recorder reads that used to exist only as local variables.

    tectum._last_kl_div        the KL before tanh; the vision bid is tanh of it
    tectum._last_kl_map        the per-cell KL map it sums (continuous latent only)
    workspace.last_modulated_bids   bids after affective modulation, before binding

Without the modulated bids, the logged raw bids cannot explain the logged winner,
because modulation and binding happen in between.
"""

import argparse
import unittest

import torch

from scripts.training.train_rlhf import build_config, frame_to_tensor, init_components
from simulations.environments.simple_visual_env import SimpleVisualEnv


def _components(latent_mode):
    args = argparse.Namespace(env="dark_room", episodes=1, max_steps=5, action_dim=2,
                              lr=1e-3, enable_audio=False, seed=0, existence_drive="on",
                              rssm_latent_mode=latent_mode)
    config = build_config(args)
    return config, init_components(config)


def _one_tectum_step(config, tectum):
    env = SimpleVisualEnv(width=224, height=224)
    obs, _ = env.reset()
    spatial = torch.zeros(1, config["tectum_feature_dim"], 2, device=config["device"])
    with torch.no_grad():
        return tectum(frame_to_tensor(obs, config["device"]), spatial)


class TestTectumKLCache(unittest.TestCase):

    def test_continuous_kl_map_sums_to_kl_div(self):
        config, components = _components("continuous")
        tectum = components[0]
        _, bid = _one_tectum_step(config, tectum)
        kl_map = tectum._last_kl_map
        self.assertEqual(tuple(kl_map.shape), (1, 32, 32, 16, 16))
        self.assertAlmostEqual(float(kl_map.sum()), tectum._last_kl_div,
                               delta=1e-4 * max(1.0, abs(tectum._last_kl_div)))
        self.assertEqual(bid, float(torch.tanh(torch.tensor(tectum._last_kl_div))))

    def test_discrete_latent_caches_kl_div_and_no_map(self):
        config, components = _components("discrete")
        tectum = components[0]
        _one_tectum_step(config, tectum)
        self.assertIsInstance(tectum._last_kl_div, float)
        self.assertIsNone(tectum._last_kl_map)


class TestWorkspaceModulatedBids(unittest.TestCase):
    BIDS = {"vision": 0.6, "audio": 0.2, "memory": 0.1, "body": 0.15, "semantic": 0.0}
    PAD = {"valence": -0.4, "arousal": 0.7, "dominance": 0.3}

    def _run(self, pad_state):
        config, components = _components("continuous")
        workspace, modulator = components[1], components[3]
        zero = torch.zeros(1, config["workspace_dim"], device=config["device"])
        payloads = {name: {"tensor": zero, "source": name} for name in self.BIDS}
        workspace.run_competition(
            inputs={}, goal_vector=torch.tensor([1.0, -1.0, 1.0], device=config["device"]),
            bids=dict(self.BIDS), payloads=payloads, pad_state=pad_state)
        return workspace, modulator

    def test_cache_equals_the_modulator_output(self):
        workspace, modulator = self._run(self.PAD)
        expected, _ = modulator.modulate(dict(self.BIDS), self.PAD)
        self.assertEqual(workspace.last_modulated_bids, expected)

    def test_without_pad_the_cache_equals_the_raw_bids(self):
        workspace, _ = self._run(None)
        self.assertEqual(workspace.last_modulated_bids, self.BIDS)


if __name__ == "__main__":
    unittest.main()
