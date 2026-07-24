"""Tests for ConsciousnessMetricsLogger."""
from __future__ import annotations

import os
import csv
import tempfile
import unittest

import numpy as np

from scripts.training.metrics_logger import (
    ConsciousnessMetricsLogger,
    StepMetrics,
)


class TestStepLogging(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ConsciousnessMetricsLogger(
            log_dir=self.tmpdir, use_tensorboard=False
        )

    def tearDown(self):
        self.logger.close()

    def test_csv_created(self):
        """CSV file should be created on init."""
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "metrics.csv")))

    def test_step_written_to_csv(self):
        """log_step should write a row to CSV."""
        m = StepMetrics(global_step=0, phi=0.5, sync_r=0.8,
                        is_conscious=True, reward=1.0, broadcast_mag=0.9)
        self.logger.log_step(m)
        self.logger.close()

        with open(os.path.join(self.tmpdir, "metrics.csv")) as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 2)  # header + 1 row
        self.assertEqual(reader[1][0], "0")  # global_step

    def test_multiple_steps(self):
        """Multiple log_step calls produce multiple rows."""
        for i in range(5):
            m = StepMetrics(global_step=i, phi=0.1 * i, sync_r=0.5,
                            is_conscious=i > 2, reward=float(i), broadcast_mag=0.5)
            self.logger.log_step(m)
        self.logger.close()

        with open(os.path.join(self.tmpdir, "metrics.csv")) as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 6)  # header + 5


class TestEpisodeLogging(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ConsciousnessMetricsLogger(
            log_dir=self.tmpdir, use_tensorboard=False
        )

    def tearDown(self):
        self.logger.close()

    def test_episode_csv_created(self):
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "episodes.csv")))

    def test_episode_written(self):
        self.logger.log_episode(
            episode=0, total_reward=10.0, steps=100,
            avg_phi=0.3, consciousness_ratio=0.5
        )
        self.logger.close()

        with open(os.path.join(self.tmpdir, "episodes.csv")) as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 2)


class TestInsightDetection(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ConsciousnessMetricsLogger(
            log_dir=self.tmpdir, use_tensorboard=False
        )

    def tearDown(self):
        self.logger.close()

    def test_novel_high_reward_high_broadcast_is_insight(self):
        """All 4 criteria met should return True."""
        # Seed cross-episode rewards (need >= 200) and broadcast mags (need >= 10)
        for i in range(210):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=0.5, broadcast_mag=0.5)
            self.logger.log_step(m)

        result = self.logger.detect_insight_moment(
            state_hash="state_42", action=3, reward=5.0, broadcast_mag=0.9
        )
        self.assertTrue(result)

    def test_repeated_state_action_not_insight(self):
        """Same state-action pair seen twice should not be insight."""
        for i in range(210):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=0.5, broadcast_mag=0.5)
            self.logger.log_step(m)

        self.logger.detect_insight_moment("s1", 1, 5.0, 0.9)
        # Advance global step past cooldown
        for i in range(210, 270):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=0.5, broadcast_mag=0.5)
            self.logger.log_step(m)
        result = self.logger.detect_insight_moment("s1", 1, 5.0, 0.9)
        self.assertFalse(result)

    def test_low_reward_not_insight(self):
        """Reward below 1.5x positive average should not be insight."""
        for i in range(210):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=10.0, broadcast_mag=0.5)
            self.logger.log_step(m)

        result = self.logger.detect_insight_moment("unique_state", 1, 1.0, 0.9)
        self.assertFalse(result)

    def test_low_broadcast_not_insight(self):
        """Broadcast below 75th percentile should not be insight."""
        for i in range(210):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=0.5,
                            broadcast_mag=0.1 + (i % 20) * 0.04)
            self.logger.log_step(m)

        # Very low broadcast
        result = self.logger.detect_insight_moment("novel_s", 1, 10.0, 0.01)
        self.assertFalse(result)

    def test_insufficient_baseline_not_insight(self):
        """With fewer than 10 reward samples, no insight can be detected."""
        for i in range(5):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=1.0, broadcast_mag=0.5)
            self.logger.log_step(m)
        result = self.logger.detect_insight_moment("s_new", 1, 5.0, 0.9)
        self.assertFalse(result)

    def test_cooldown_prevents_rapid_insights(self):
        """Two insights within 50 steps should be blocked by cooldown."""
        for i in range(210):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=0.5, broadcast_mag=0.5)
            self.logger.log_step(m)
        first = self.logger.detect_insight_moment("s_a", 1, 5.0, 0.9)
        self.assertTrue(first)
        # Immediately try another (within cooldown)
        result = self.logger.detect_insight_moment("s_b", 2, 5.0, 0.9)
        self.assertFalse(result)


class TestEIComputation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ConsciousnessMetricsLogger(
            log_dir=self.tmpdir, use_tensorboard=False
        )

    def tearDown(self):
        self.logger.close()

    def test_insufficient_data_returns_zeros(self):
        """With < 10 trajectory points, EI should be zero."""
        result = self.logger.compute_and_log_ei(episode=0)
        self.assertEqual(result["ei_gates"], 0.0)
        self.assertFalse(result["emergent"])

    def test_with_trajectory_data(self):
        """With enough trajectory data, EI values should be computed."""
        for i in range(50):
            gate = (float(i % 3), float(i % 5), float(i % 2))
            ws = (float(i % 4), float(i % 3))
            m = StepMetrics(
                global_step=i, phi=0.1, sync_r=0.5,
                is_conscious=True, reward=1.0, broadcast_mag=0.5,
                gate_state=gate, workspace_state=ws,
            )
            self.logger.log_step(m)

        result = self.logger.compute_and_log_ei(episode=0, num_gate_states=8, num_workspace_states=8)
        self.assertIsInstance(result["ei_gates"], float)
        self.assertIsInstance(result["ei_workspace"], float)
        self.assertIn("emergent", result)


class TestResetAndClose(unittest.TestCase):

    def test_reset_clears_episode_state(self):
        tmpdir = tempfile.mkdtemp()
        logger = ConsciousnessMetricsLogger(log_dir=tmpdir, use_tensorboard=False)
        for i in range(5):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5,
                            is_conscious=True, reward=1.0, broadcast_mag=0.5)
            logger.log_step(m)
        logger.reset_episode_state()
        self.assertEqual(len(logger._episode_broadcast_mags), 0)
        self.assertEqual(len(logger._seen_state_actions), 0)
        # Cross-episode rewards are preserved across resets
        self.assertEqual(len(logger._cross_episode_rewards), 5)
        logger.close()

    def test_close_idempotent(self):
        tmpdir = tempfile.mkdtemp()
        logger = ConsciousnessMetricsLogger(log_dir=tmpdir, use_tensorboard=False)
        logger.close()
        # Second close should not raise
        logger.close()


class TestCausalEmergence2(unittest.TestCase):
    """CE 2.0 (SVD heuristic) wiring in the logger. The metric itself is tested in
    tests/test_causal_emergence_svd.py; here we check the logger integration."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = ConsciousnessMetricsLogger(
            log_dir=self.tmpdir, use_tensorboard=False
        )

    def tearDown(self):
        self.logger.close()

    def test_episode_header_has_ce2_columns(self):
        self.logger.close()
        with open(os.path.join(self.tmpdir, "episodes.csv")) as f:
            header = next(csv.reader(f))
        for col in ("ce2_gates", "ce2_workspace", "ce2_ratio",
                    "ce2_complexity_gates", "ce2_complexity_workspace",
                    "ce2_rssm", "ce2_complexity_rssm",
                    "ce2_gates_states", "ce2_workspace_states"):
            self.assertIn(col, header)

    def test_disabled_by_default_leaves_buffers_empty(self):
        self.assertFalse(self.logger._ce2_enabled)
        m = StepMetrics(global_step=0, phi=0.1, sync_r=0.5, is_conscious=True,
                        reward=1.0, broadcast_mag=0.5,
                        gate_state=(0.1, 0.2, 0.3, 0.4, 0.5),
                        workspace_state=(0.5, 0.1, 0.5))
        self.logger.log_step(m)
        self.assertEqual(len(self.logger._ce2_gate_trajectory), 0)

    def test_ce2_zero_with_insufficient_data(self):
        self.logger.enable_ce2(num_classes=4)
        res = self.logger.compute_and_log_ce2(episode=0)
        self.assertEqual(res["ce2_gates"], 0.0)
        self.assertEqual(res["ce2_complexity_gates"], 0)
        self.assertEqual(res["ce2_rssm"], 0.0)

    def test_ce2_gate_workspace_finite_and_buffers_cleared(self):
        self.logger.enable_ce2(num_classes=4)
        rng = np.random.default_rng(0)
        for i in range(30):
            m = StepMetrics(global_step=i, phi=0.1, sync_r=0.5, is_conscious=True,
                            reward=1.0, broadcast_mag=0.5,
                            gate_state=tuple(rng.random(5).tolist()),
                            workspace_state=tuple(rng.random(3).tolist()))
            self.logger.log_step(m)
        self.assertGreater(len(self.logger._ce2_gate_trajectory), 0)
        res = self.logger.compute_and_log_ce2(episode=0)
        self.assertGreaterEqual(res["ce2_gates"], 0.0)
        self.assertGreaterEqual(res["ce2_workspace"], 0.0)
        self.assertIn("ce2_emergent", res)
        self.assertEqual(len(self.logger._ce2_gate_trajectory), 0)  # cleared

    def test_ce2_rssm_from_recorded_latents_and_counts_reset(self):
        self.logger.enable_ce2(num_classes=4)
        for field in ([0, 2], [1, 3], [0, 2], [1, 3], [0, 2]):
            self.logger.record_latent_step(np.array(field))
        res = self.logger.compute_and_log_ce2(episode=0)
        self.assertGreaterEqual(res["ce2_rssm"], 0.0)
        self.assertIsInstance(res["ce2_complexity_rssm"], int)
        self.assertEqual(self.logger._latent_counts.sum(), 0.0)  # reset after use

    def test_reset_latent_window_breaks_cross_episode_pairing(self):
        self.logger.enable_ce2(num_classes=4)
        self.logger.record_latent_step(np.array([0, 1]))
        self.logger.reset_latent_window()
        self.logger.record_latent_step(np.array([2, 3]))
        # No transition counted across the reset boundary.
        self.assertEqual(self.logger._latent_counts.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
