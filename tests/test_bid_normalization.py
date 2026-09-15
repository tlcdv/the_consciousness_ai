"""The vision bid measures change against its own running baseline.

The default bid tanh(KL) was exactly 1.0 on every measured step (KL 1462 to 282061), and
the KL scale differs 8,079x between checkpoints of the same configuration
(docs/results/bid_counterfactual_2026_09.md). The S5 reduction, sigmoid of a causal
running z-score, de-saturated at all three checkpoints there. These tests pin it.
"""

import argparse
import math
import unittest

import numpy as np
import torch

from models.core.bid_normalization import (
    EMA_ALPHA,
    SD_FLOOR,
    RunningZScoreBid,
)
from scripts.training.train_rlhf import build_config, frame_to_tensor, init_components
from simulations.environments.simple_visual_env import SimpleVisualEnv


def _bids(values, alpha=EMA_ALPHA):
    normalizer = RunningZScoreBid(alpha)
    return [normalizer.bid(v) for v in values]


class TestRunningZScoreBid(unittest.TestCase):

    def test_constants_match_the_validated_reduction(self):
        self.assertEqual(EMA_ALPHA, 0.01)
        self.assertEqual(SD_FLOOR, 1e-9)

    def test_first_reading_has_no_baseline(self):
        self.assertEqual(_bids([1234.5]), [0.5])

    def test_second_reading_closed_form(self):
        # After one reading: mean = x0, variance = 1.0, so z = x1 - x0.
        first, second = _bids([10.0, 11.5])
        self.assertEqual(first, 0.5)
        self.assertAlmostEqual(second, 1.0 / (1.0 + math.exp(-1.5)), places=12)

    def test_a_constant_input_stays_at_the_midpoint(self):
        self.assertEqual(set(_bids([7.0] * 500)), {0.5})

    def test_a_rise_above_baseline_bids_high_and_a_fall_bids_low(self):
        bids = _bids([100.0] * 200 + [100.0 + 5.0] + [100.0 - 5.0])
        self.assertGreater(bids[-2], 0.5)
        self.assertLess(bids[-1], 0.5)

    def test_bid_does_not_depend_on_the_kl_scale(self):
        rng = np.random.default_rng(3)
        series = 2000.0 + 50.0 * rng.standard_normal(3000)
        small, large = _bids(series), _bids(series * 1000.0)
        self.assertLess(max(abs(a - b) for a, b in zip(small[-500:], large[-500:])), 1e-3)

    def test_a_huge_jump_does_not_overflow(self):
        # Measured live KL ranged 1462 to 282061; the running variance starts at 1.0.
        low, high, back = _bids([1462.0, 282061.0, 1462.0])
        self.assertEqual(low, 0.5)
        self.assertEqual(high, 1.0)
        # Closed form for the third reading: the jump moved the mean and the variance.
        delta = 282061.0 - 1462.0
        mean = 1462.0 + EMA_ALPHA * delta
        var = (1.0 - EMA_ALPHA) * (1.0 + EMA_ALPHA * delta * delta)
        z = (1462.0 - mean) / (var ** 0.5 + 1e-12)
        self.assertAlmostEqual(back, 1.0 / (1.0 + math.exp(-z)), places=12)
        self.assertEqual(_bids([0.0, -1e9])[1], 0.0)

    def test_zero_initial_variance_is_scale_free_from_the_start(self):
        rng = np.random.default_rng(4)
        series = 0.01 + 0.002 * rng.standard_normal(200)
        small = [RunningZScoreBid(initial_var=0.0) for _ in range(2)]
        a = [small[0].bid(v) for v in series]
        b = [small[1].bid(v * 1e6) for v in series]
        self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-9)

    def test_uses_no_random_state(self):
        np.random.seed(5)
        before = np.random.get_state()[1].copy()
        _bids([1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before))


def _tectum(reduction=None):
    fields = dict(env="dark_room", episodes=1, max_steps=5, action_dim=2, lr=1e-3,
                  enable_audio=False, seed=0, existence_drive="on",
                  rssm_latent_mode="continuous")
    if reduction is not None:
        fields["vision_bid_reduction"] = reduction
    config = build_config(argparse.Namespace(**fields))
    return config, init_components(config)[0]


def _tectum_bids(config, tectum, steps):
    env = SimpleVisualEnv(width=224, height=224)
    np.random.seed(0)
    obs, _ = env.reset()
    spatial = torch.zeros(1, config["tectum_feature_dim"], 2, device=config["device"])
    bids = []
    with torch.no_grad():
        for _ in range(steps):
            _, bid = tectum(frame_to_tensor(obs, config["device"]), spatial)
            bids.append(bid)
            obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    return bids


class TestTectumVisionBidReduction(unittest.TestCase):

    def test_default_keeps_tanh_and_builds_no_normalizer(self):
        config, tectum = _tectum()
        self.assertIsNone(tectum.bid_normalizer)
        bid = _tectum_bids(config, tectum, 1)[0]
        self.assertEqual(bid, float(torch.tanh(torch.tensor(tectum._last_kl_div))))

    def test_zscore_first_bid_is_the_midpoint_and_bids_vary(self):
        config, tectum = _tectum("zscore")
        bids = _tectum_bids(config, tectum, 6)
        self.assertEqual(bids[0], 0.5)
        self.assertGreater(len({round(b, 9) for b in bids}), 1)

    def test_unknown_reduction_is_rejected(self):
        with self.assertRaises(ValueError):
            _tectum("average")


if __name__ == "__main__":
    unittest.main()
