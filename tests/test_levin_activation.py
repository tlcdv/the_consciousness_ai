"""
Tests for Phase 5 deliverable 4: activating the dormant Levin modules.

Covers:
  - BioelectricSignalingNetwork.forward runs across multiple components without
    the 4-D-stack / MultiheadAttention shape crash (the latent bug fixed when
    the modules were activated), and passes a single component through unchanged.
  - HolonicSystem.forward produces the keys LevinConsciousnessEvaluator reads.
  - The 5 LevinConsciousnessMetrics computed from REAL HolonicSystem output are
    finite and in [0, 1] (the contract the training-loop logging relies on).
  - LevinConsciousnessEvaluator.evaluate() adapter (used by ConsciousnessMonitor)
    does not crash on empty or partial state dicts.

These are baseline-apparatus tests: the modules run in inference mode as fixed
measurement functions, not as trained policy components.
"""
import math
import unittest

import torch

from models.self_model.bioelectric_signaling import BioelectricSignalingNetwork
from models.self_model.holonic_intelligence import HolonicSystem
from models.evaluation.levin_consciousness_metrics import LevinConsciousnessEvaluator


def _small_config():
    return {
        "hidden_size": 32,
        "num_holons": 4,
        "field_dimension": 16,
        "bioelectric_channels": 4,
        "signaling_layers": 2,
        "gap_junction_heads": 4,
        "gap_junction_dropout": 0.0,
        "integration_heads": 4,
    }


class TestBioelectricSignalingShapes(unittest.TestCase):
    """The gap-junction cross-attention must accept 3-D fields and not crash."""

    def setUp(self):
        torch.manual_seed(0)
        self.net = BioelectricSignalingNetwork(_small_config())
        self.net.eval()

    def test_multi_component_forward_runs(self):
        states = {
            "vision": torch.randn(1, 32),
            "memory": torch.randn(1, 32),
            "gate": torch.randn(1, 32),
        }
        with torch.no_grad():
            out = self.net(states)
        self.assertEqual(set(out.keys()), set(states.keys()))
        for field in out.values():
            # [B, num_channels, field_dim]
            self.assertEqual(tuple(field.shape), (1, 4, 16))
            self.assertTrue(torch.isfinite(field).all())

    def test_single_component_passthrough(self):
        states = {"solo": torch.randn(1, 32)}
        with torch.no_grad():
            out = self.net(states)
        self.assertIn("solo", out)
        self.assertEqual(tuple(out["solo"].shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(out["solo"]).all())

    def test_batch_dimension_preserved(self):
        states = {"a": torch.randn(3, 32), "b": torch.randn(3, 32)}
        with torch.no_grad():
            out = self.net(states)
        for field in out.values():
            self.assertEqual(field.shape[0], 3)


class TestHolonicSystemOutput(unittest.TestCase):
    """HolonicSystem must produce the keys the Levin evaluator reads."""

    def setUp(self):
        torch.manual_seed(0)
        self.system = HolonicSystem(_small_config())
        self.system.eval()

    def test_forward_keys_and_finite(self):
        with torch.no_grad():
            out = self.system(torch.randn(1, 32))
        for key in ("integrated_state", "holon_states", "attention_weights",
                    "bioelectric_fields"):
            self.assertIn(key, out)
        self.assertTrue(torch.isfinite(out["integrated_state"]).all())
        self.assertTrue(torch.isfinite(out["attention_weights"]).all())
        self.assertIsInstance(out["bioelectric_fields"], dict)
        self.assertEqual(len(out["bioelectric_fields"]), 4)  # num_holons


class TestLevinMetricsFromHolonicOutput(unittest.TestCase):
    """The 5 metrics from REAL holonic output are finite and in [0, 1]."""

    def setUp(self):
        torch.manual_seed(0)
        cfg = _small_config()
        self.system = HolonicSystem(cfg)
        self.system.eval()
        self.evaluator = LevinConsciousnessEvaluator(cfg)

    def _metrics(self, past_states):
        with torch.no_grad():
            out = self.system(torch.randn(1, 32))
        current = {"integrated_state": out["integrated_state"].detach()}
        component_states = {
            "broadcast": torch.randn(1, 32),
            "tectum": torch.randn(1, 32),
            "gate": torch.randn(5),
        }
        result = self.evaluator.evaluate_levin_consciousness(
            bioelectric_state=out["bioelectric_fields"],
            holonic_output=out,
            past_states=past_states,
            current_state=current,
            actions=[], goals=[], outcomes=[],
            component_states=component_states,
        )
        return result, current

    def test_all_metrics_finite_and_bounded(self):
        # First step: no history (morphological_adaptation falls back to 0.0)
        result, current = self._metrics(past_states=[])
        # Second step: with one prior integrated state so morphological is live
        result2, _ = self._metrics(past_states=[current])

        keys = [
            "bioelectric_complexity", "morphological_adaptation",
            "collective_intelligence", "goal_directed_behavior",
            "basal_cognition", "overall_levin_score",
        ]
        for res in (result, result2):
            for k in keys:
                self.assertIn(k, res)
                v = res[k]
                self.assertIsInstance(v, float)
                self.assertTrue(math.isfinite(v), f"{k} not finite: {v}")
                self.assertGreaterEqual(v, 0.0, f"{k} below 0: {v}")
                self.assertLessEqual(v, 1.0, f"{k} above 1: {v}")

    def test_goal_directed_is_zero_in_baseline(self):
        # Baseline wiring passes empty actions/goals/outcomes, so goal_directed
        # is 0.0 until the substrate-independence test defines the embeddings.
        result, _ = self._metrics(past_states=[])
        self.assertEqual(result["goal_directed_behavior"], 0.0)

    def test_collective_intelligence_nonzero_with_real_attention(self):
        # Real holonic attention weights are not perfectly uniform, so the
        # integration (1 - normalized entropy) should be strictly positive.
        result, _ = self._metrics(past_states=[])
        self.assertGreater(result["collective_intelligence"], 0.0)


class TestEvaluateAdapter(unittest.TestCase):
    """The evaluate() adapter (used by ConsciousnessMonitor) never crashes."""

    def setUp(self):
        self.evaluator = LevinConsciousnessEvaluator(_small_config())

    def test_empty_state(self):
        result = self.evaluator.evaluate()
        self.assertIn("overall_levin_score", result)
        for v in result.values():
            self.assertTrue(math.isfinite(v))

    def test_none_state(self):
        result = self.evaluator.evaluate(None)
        self.assertIn("overall_levin_score", result)

    def test_partial_state(self):
        result = self.evaluator.evaluate({"component_states": {"a": torch.randn(8)}})
        self.assertIn("basal_cognition", result)
        self.assertTrue(0.0 <= result["basal_cognition"] <= 1.0)


class TestCollectiveIntelligenceFix(unittest.TestCase):
    """Phase 1a (2026-05-29): collective_intelligence must respond to input.

    The previous entropy-of-attention implementation was inert (constant ~2e-6
    across 64 diverse inputs; see docs/results/levin_derisk_2026_05_29.md). The
    holon_states-based measure must vary with input and have sensible extremes.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.evaluator = LevinConsciousnessEvaluator(_small_config())

    def test_identical_holons_give_max_integration(self):
        # All holon states identical -> pairwise cosine 1 -> integration ~1.0
        row = torch.randn(1, 32)
        states = row.repeat(6, 1)
        ci = self.evaluator.evaluate_collective_intelligence({"holon_states": states})
        self.assertGreater(ci, 0.99)

    def test_different_inputs_give_different_ci(self):
        ci_a = self.evaluator.evaluate_collective_intelligence(
            {"holon_states": torch.randn(8, 32)})
        ci_b = self.evaluator.evaluate_collective_intelligence(
            {"holon_states": torch.randn(8, 32) * 3.0 + 1.0})
        self.assertNotEqual(ci_a, ci_b)
        for ci in (ci_a, ci_b):
            self.assertTrue(0.0 <= ci <= 1.0)

    def test_missing_holon_states_returns_zero(self):
        self.assertEqual(self.evaluator.evaluate_collective_intelligence({}), 0.0)

    def test_single_holon_returns_zero(self):
        self.assertEqual(
            self.evaluator.evaluate_collective_intelligence(
                {"holon_states": torch.randn(1, 32)}),
            0.0,
        )

    def test_handles_3d_holon_states(self):
        # Real HolonicSystem returns [num_holons, batch, hidden]
        ci = self.evaluator.evaluate_collective_intelligence(
            {"holon_states": torch.randn(8, 1, 32)})
        self.assertTrue(0.0 <= ci <= 1.0)


class TestLevinDynamicRange(unittest.TestCase):
    """Phase 1b (2026-05-29): activated metrics must have real dynamic range,
    not merely be bounded in [0, 1]. Addresses the standing audit gap that tests
    only check shapes/bounds and never that a metric actually responds to input.
    """

    def test_metrics_respond_to_diverse_inputs(self):
        from scripts.analysis.diagnose_levin_variance import run_probe
        summary = run_probe(trials=32, seed=0, hidden_size=128, num_holons=6)

        # collective_intelligence was inert before the fix; it must now move.
        self.assertTrue(
            summary["collective_intelligence"]["usable"],
            f"collective_intelligence still inert: {summary['collective_intelligence']}",
        )
        # The other input-driven metrics must also have dynamic range.
        for k in ("bioelectric_complexity", "morphological_adaptation",
                  "basal_cognition"):
            self.assertTrue(summary[k]["usable"], f"{k} inert: {summary[k]}")
        # goal_directed is a documented constant-0 placeholder (deliverable 5).
        self.assertEqual(summary["goal_directed_behavior"]["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
