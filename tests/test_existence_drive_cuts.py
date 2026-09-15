"""With the existence drive off, homeostatic variables reach no part of the agent.

docs/ethics_framework.md lists the paths by which energy, fatigue and damage act
on the agent. The original ablation cut two (interoceptive affect, reward terms;
see tests/test_existence_bias_ablation.py). These tests pin the other three:

    1. the environment battery is not copied into the self-model's energy
    2. the body bid does not depend on energy
    3. the self-vector features carry neutral interoceptive values

and pin that the drive declaration is enforced where the agent is built, so no
caller of init_components can skip it.
"""

import argparse
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.ethics.framework import EthicsViolation
from models.self_model.self_representation_core import SelfRepresentationCore
from scripts.training.metrics_logger import ConsciousnessMetricsLogger
from scripts.training.train_rlhf import build_config, init_components, run_episode
from simulations.environments.simple_visual_env import SimpleVisualEnv

LOW_ENERGY = 0.1
STEPS = 3


def _args(**overrides):
    base = argparse.Namespace(
        episodes=1, max_steps=STEPS, action_dim=2, lr=1e-3, render=False,
        env="dark_room", log_dir="runs/_test", enable_audio=False, seed=0,
    )
    for key, flag in overrides.items():
        setattr(base, key, flag)
    return base


def _run_dark_room(existence_drive):
    """Run a few dark_room steps from a low-energy self-model; return (energy, body bids)."""
    torch.manual_seed(0)
    np.random.seed(0)
    config = build_config(_args(existence_drive=existence_drive))
    components = init_components(config)
    tectum, workspace, reentrant, modulator = components[0], components[1], components[2], components[3]
    memory, action_core, gate, self_model = components[5], components[6], components[8], components[14]
    self_model.state.interoceptive_state["energy"] = LOW_ENERGY
    env = SimpleVisualEnv(width=224, height=224)
    log_dir = tempfile.mkdtemp()
    logger = ConsciousnessMetricsLogger(log_dir=log_dir, use_tensorboard=False)
    try:
        run_episode(0, config, tectum, workspace, reentrant, modulator, action_core, env,
                    gate=gate, memory=memory, metrics_logger=logger, self_model=self_model)
    finally:
        logger.close()
    with open(Path(log_dir) / "metrics.csv", newline="") as handle:
        body_bids = [float(row["bid_body"]) for row in csv.DictReader(handle)]
    shutil.rmtree(log_dir, ignore_errors=True)
    return self_model.state.interoceptive_state["energy"], body_bids


class TestDeclarationEnforcedWhereTheAgentIsBuilt(unittest.TestCase):

    def test_init_components_refuses_an_undeclared_run(self):
        config = build_config(_args())
        with self.assertRaises(EthicsViolation) as caught:
            init_components(config)
        self.assertIn("E1", str(caught.exception))

    def test_ablate_flag_is_an_alias_for_off(self):
        config = build_config(_args(ablate_existence_bias=True))
        self.assertEqual(config["existence_drive"], "off")
        self.assertTrue(config["ablate_existence_bias"])

    def test_drive_on_keeps_the_ablation_key_false(self):
        config = build_config(_args(existence_drive="on"))
        self.assertEqual(config["existence_drive"], "on")
        self.assertFalse(config["ablate_existence_bias"])


class TestBatteryAndBodyBidCuts(unittest.TestCase):
    """Paired arms from the same low-energy start. Drive on is the control."""

    @classmethod
    def setUpClass(cls):
        cls.energy_on, cls.body_on = _run_dark_room("on")
        cls.energy_off, cls.body_off = _run_dark_room("off")

    def test_rows_were_logged(self):
        self.assertEqual(len(self.body_on), STEPS)
        self.assertEqual(len(self.body_off), STEPS)

    def test_drive_on_copies_the_battery_into_energy(self):
        self.assertGreater(self.energy_on, 0.9)

    def test_drive_off_does_not_copy_the_battery(self):
        self.assertLess(self.energy_off, 0.4)

    def test_drive_on_body_bid_reacts_to_low_energy(self):
        self.assertEqual(self.body_on[0], 0.15)

    def test_drive_off_body_bid_ignores_energy(self):
        self.assertEqual(set(self.body_off), {0.05})


class TestSelfVectorFeaturesCut(unittest.TestCase):
    BROADCAST = (1.0, 0.0, 0.1)
    DEPLETED = {"energy": 0.1, "fatigue": 0.8, "damage": 0.6}

    def _features(self, config):
        core = SelfRepresentationCore(config)
        core.state.interoceptive_state = dict(self.DEPLETED)
        return core.first_order_features({"valence": 0.0}, self.BROADCAST)

    def test_default_features_carry_the_interoceptive_state(self):
        self.assertEqual(self._features({})[3:6], [0.1, 0.8, 0.6])

    def test_drive_off_features_are_neutral(self):
        features = self._features({"ablate_existence_bias": True})
        self.assertEqual(features[3:6], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
