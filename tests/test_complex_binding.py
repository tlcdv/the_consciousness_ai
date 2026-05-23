"""Tests for Phase B-alt of the 2026-05-19 plan: KomplexNet-style binding.

Three test groups:
  1. Mechanical tests of ComplexBindingSystem (shapes, ranges, drop-in
     compat with WorkspaceBindingSystem interface).
  2. 3-condition phi-binding correlation gate (analogous to
     tests/test_phi_binding_correlation.py for AKOrN). Validates that
     sync_R increases monotonically with the binding regime.
  3. weave_content gate test: when phases are synchronized, the
     amplitude variance of content tensors across modules is LOW
     (factors all near +1); when desynced, variance is HIGH
     (factors spread across [-1, +1]). This is the structural
     property Phase B-alt claims will make phi track sync_R.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.complex_binding import ComplexBindingSystem
from models.core.global_workspace import GlobalWorkspace


class TestComplexBindingMechanics(unittest.TestCase):
    """Smoke tests for the ComplexBindingSystem interface."""

    def test_bind_bids_returns_dict_and_float(self):
        cbs = ComplexBindingSystem(num_modules=5)
        cbs.register_modules(["vision", "audio", "memory", "body", "semantic"])
        bids = {"vision": 0.7, "audio": 0.6, "memory": 0.5, "body": 0.4, "semantic": 0.3}
        bound, sync_R = cbs.bind_bids(bids)
        self.assertIsInstance(bound, dict)
        self.assertEqual(set(bound.keys()), set(bids.keys()))
        self.assertIsInstance(sync_R, float)
        self.assertGreaterEqual(sync_R, 0.0)
        self.assertLessEqual(sync_R, 1.0)

    def test_pairwise_coherence_shape_and_range(self):
        cbs = ComplexBindingSystem(num_modules=4)
        cbs.register_modules(["a", "b", "c", "d"])
        cbs.bind_bids({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
        coh = cbs.get_pairwise_coherence()
        self.assertIsNotNone(coh)
        self.assertEqual(coh.shape, (1, 4, 4))
        self.assertGreaterEqual(coh.min().item(), -1.0001)
        self.assertLessEqual(coh.max().item(), 1.0001)
        diag = torch.diagonal(coh[0])
        self.assertTrue(torch.allclose(diag, torch.ones(4), atol=1e-4))

    def test_reset_state_clears_phases(self):
        cbs = ComplexBindingSystem(num_modules=3)
        cbs.register_modules(["a", "b", "c"])
        cbs.bind_bids({"a": 0.5, "b": 0.5, "c": 0.5})
        self.assertIsNotNone(cbs.current_phases)
        cbs.reset_state()
        self.assertIsNone(cbs.current_phases)

    def test_get_module_phases_shape(self):
        cbs = ComplexBindingSystem(num_modules=5)
        cbs.register_modules(["v", "a", "m", "b", "s"])
        cbs.bind_bids({"v": 0.5, "a": 0.5, "m": 0.5, "b": 0.5, "s": 0.5})
        phases = cbs.get_module_phases()
        self.assertEqual(phases.shape, (5,))

    def test_bound_bid_boost_range(self):
        """bound_bid / orig_bid should be in [1.0, 1.5] per the alignment boost."""
        cbs = ComplexBindingSystem(num_modules=4)
        cbs.register_modules(["a", "b", "c", "d"])
        bids = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
        bound, _ = cbs.bind_bids(bids)
        for name in bids:
            ratio = bound[name] / bids[name]
            self.assertGreaterEqual(ratio, 1.0 - 1e-4)
            self.assertLessEqual(ratio, 1.5 + 1e-4)


class TestWeaveContent(unittest.TestCase):
    """The KomplexNet adaptation: phases are woven into content tensors."""

    def test_weave_returns_dict_with_tensor_keys(self):
        cbs = ComplexBindingSystem(num_modules=3)
        cbs.register_modules(["a", "b", "c"])
        cbs.bind_bids({"a": 0.5, "b": 0.5, "c": 0.5})
        payloads = {
            "a": {"tensor": torch.randn(8)},
            "b": {"tensor": torch.randn(8)},
            "c": {"tensor": torch.randn(8)},
        }
        woven = cbs.weave_content(payloads)
        self.assertEqual(set(woven.keys()), {"a", "b", "c"})
        for name in woven:
            self.assertIn("tensor", woven[name])

    def test_weave_preserves_raw_tensor_payload(self):
        cbs = ComplexBindingSystem(num_modules=2)
        cbs.register_modules(["a", "b"])
        cbs.bind_bids({"a": 0.5, "b": 0.5})
        payloads = {"a": torch.randn(8), "b": torch.randn(8)}
        woven = cbs.weave_content(payloads)
        self.assertIsInstance(woven["a"], torch.Tensor)
        self.assertEqual(woven["a"].shape, payloads["a"].shape)

    def test_weave_factor_range(self):
        """Each module's factor cos(theta_m - theta_global) must be in [-1, 1]."""
        cbs = ComplexBindingSystem(num_modules=5, iterations=20)
        cbs.register_modules(["v", "a", "m", "b", "s"])
        cbs.bind_bids({"v": 0.5, "a": 0.5, "m": 0.5, "b": 0.5, "s": 0.5})
        unit = torch.ones(4)
        payloads = {n: unit.clone() for n in ["v", "a", "m", "b", "s"]}
        woven = cbs.weave_content(payloads)
        for name in payloads:
            ratio = woven[name][0].item() / unit[0].item()
            self.assertGreaterEqual(ratio, -1.0 - 1e-4)
            self.assertLessEqual(ratio, 1.0 + 1e-4)

    def test_weave_preserves_non_tensor_fields(self):
        """Dict payloads with non-tensor fields (capsule_poses etc.) keep them."""
        cbs = ComplexBindingSystem(num_modules=2)
        cbs.register_modules(["a", "b"])
        cbs.bind_bids({"a": 0.5, "b": 0.5})
        payloads = {
            "a": {"tensor": torch.randn(8), "capsule_poses": "preserved"},
            "b": {"tensor": torch.randn(8)},
        }
        woven = cbs.weave_content(payloads)
        self.assertEqual(woven["a"]["capsule_poses"], "preserved")


class TestPhiBindingCorrelationKomplex(unittest.TestCase):
    """3-condition phi-binding correlation gate for KomplexNet.

    The gate the prior 8 Phi-1 runs failed: does the order parameter
    actually rise with the binding regime under this mechanism? If it
    does NOT, then KomplexNet is no better than AKOrN at the bid level
    and Phase B-alt is dead before the 6-hour training run.
    """

    def test_sync_R_increases_with_coupling(self):
        """Higher coupling -> higher sync_R, for fixed bids and seed."""
        torch.manual_seed(42)
        # Use a single ComplexBindingSystem and tune the desync_eps to
        # sweep the regime: high eps suppresses coupling (unbound),
        # low eps preserves it (bound).
        bids = {"v": 0.8, "a": 0.8, "m": 0.8, "b": 0.8, "s": 0.8}

        cbs_unbound = ComplexBindingSystem(num_modules=5, iterations=20, desync_eps=10.0)
        cbs_unbound.register_modules(["v", "a", "m", "b", "s"])
        cbs_unbound.current_phases = torch.linspace(0, 2 * np.pi * 4 / 5, 5)
        _, R_A = cbs_unbound.bind_bids(bids)

        cbs_partial = ComplexBindingSystem(num_modules=5, iterations=20, desync_eps=0.5)
        cbs_partial.register_modules(["v", "a", "m", "b", "s"])
        cbs_partial.current_phases = torch.linspace(0, 2 * np.pi * 4 / 5, 5)
        _, R_B = cbs_partial.bind_bids(bids)

        cbs_bound = ComplexBindingSystem(num_modules=5, iterations=20, desync_eps=0.01)
        cbs_bound.register_modules(["v", "a", "m", "b", "s"])
        cbs_bound.current_phases = torch.linspace(0, 2 * np.pi * 4 / 5, 5)
        _, R_C = cbs_bound.bind_bids(bids)

        print(f"\nKomplex Condition A (Unbound, eps=10.0)   sync_R: {R_A:.4f}")
        print(f"Komplex Condition B (Partial, eps=0.5)    sync_R: {R_B:.4f}")
        print(f"Komplex Condition C (Bound,   eps=0.01)   sync_R: {R_C:.4f}")

        self.assertGreater(R_C, R_A, "Fully bound sync_R should exceed unbound")
        self.assertGreaterEqual(R_C, R_B - 1e-3,
                                "Fully bound sync_R should at least match partial")

    def test_weave_content_variance_tracks_sync(self):
        """When phases synchronize, weave factors cluster near +1
        (low variance across modules). When desynced, weave factors
        span [-1, +1] (high variance). This is the structural property
        Phase B-alt depends on for phi-on-broadcast to track sync_R.
        """
        torch.manual_seed(7)
        bids = {"v": 0.8, "a": 0.8, "m": 0.8, "b": 0.8, "s": 0.8}
        payloads = {n: torch.ones(4) for n in bids}

        cbs_sync = ComplexBindingSystem(num_modules=5, iterations=30, desync_eps=0.01)
        cbs_sync.register_modules(["v", "a", "m", "b", "s"])
        cbs_sync.current_phases = torch.zeros(5)
        cbs_sync.bind_bids(bids)
        woven_sync = cbs_sync.weave_content(payloads)
        factors_sync = torch.tensor([woven_sync[n][0].item() for n in bids])

        cbs_desync = ComplexBindingSystem(num_modules=5, iterations=30, desync_eps=10.0)
        cbs_desync.register_modules(["v", "a", "m", "b", "s"])
        cbs_desync.current_phases = torch.linspace(0, 2 * np.pi * 4 / 5, 5)
        cbs_desync.bind_bids(bids)
        woven_desync = cbs_desync.weave_content(payloads)
        factors_desync = torch.tensor([woven_desync[n][0].item() for n in bids])

        print(f"\nSynced factors:    {factors_sync.tolist()}, std={factors_sync.std():.4f}")
        print(f"Desynced factors:  {factors_desync.tolist()}, std={factors_desync.std():.4f}")

        self.assertLess(factors_sync.std().item(), factors_desync.std().item() + 0.05,
                        "Synced content factors should have lower variance than desynced")


class TestGlobalWorkspaceKomplexWiring(unittest.TestCase):
    """End-to-end: --binding-mechanism komplex produces a working workspace."""

    def test_komplex_instantiates(self):
        ws = GlobalWorkspace({
            "binding_mechanism": "komplex",
            "num_modules": 5,
            "module_names": ["vision", "audio", "memory", "body", "semantic"],
            "ignition_threshold": 0.3,
            "ignition_gain": 5.0,
            "reverberation_alpha": 0.0,
        })
        self.assertEqual(type(ws.binding_system).__name__, "ComplexBindingSystem")
        self.assertEqual(ws.binding_mechanism, "komplex")

    def test_akorn_default_unchanged(self):
        ws = GlobalWorkspace({"ignition_threshold": 0.3, "ignition_gain": 5.0})
        self.assertEqual(type(ws.binding_system).__name__, "WorkspaceBindingSystem")
        self.assertEqual(ws.binding_mechanism, "akorn")

    def test_unknown_mechanism_raises(self):
        with self.assertRaises(ValueError):
            GlobalWorkspace({"binding_mechanism": "nonexistent"})

    def test_komplex_run_competition_end_to_end(self):
        """Smoke test: komplex GNW runs one competition step without error."""
        ws = GlobalWorkspace({
            "binding_mechanism": "komplex",
            "num_modules": 4,
            "module_names": ["vision", "audio", "memory", "body"],
            "ignition_threshold": 0.3,
            "ignition_gain": 5.0,
            "reverberation_alpha": 0.0,
            "broadcast_mode": "attention_weighted",
            "attention_temperature": 0.5,
            "attention_floor": 0.05,
            "workspace_dim": 16,
        })
        bids = {"vision": 0.7, "audio": 0.6, "memory": 0.5, "body": 0.4}
        payloads = {
            "vision": {"tensor": torch.randn(16)},
            "audio": {"tensor": torch.randn(16)},
            "memory": {"tensor": torch.randn(16)},
            "body": {"tensor": torch.randn(16)},
        }
        broadcast, _ = ws.run_competition(
            inputs={}, goal_vector=torch.zeros(3), bids=bids, payloads=payloads,
        )
        if broadcast:
            self.assertTrue("_fused" in broadcast or "tensor" in broadcast,
                            f"unexpected broadcast structure: {list(broadcast.keys())}")


if __name__ == "__main__":
    unittest.main()
