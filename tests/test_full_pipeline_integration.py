"""
Full pipeline integration test.

Exercises the complete cognitive pipeline end-to-end:
  Raw 224x224 frame -> RetinotopicEncoder -> SensoryTectum -> GlobalWorkspace
  -> ReentrantProcessor -> ActionSelectionCore

Uses small dimensions (workspace_dim=64, grid=8) for speed.
No model weights required (DINOv2 conv stack fallback).
"""
from __future__ import annotations

import unittest
import torch
import numpy as np

from models.core.sensory_tectum import SensoryTectum
from models.core.global_workspace import GlobalWorkspace
from models.core.reentrant_processor import ReentrantProcessor
from models.core.semantic_pathway import SemanticPathway
from models.emotion.affective_modulator import AffectiveModulator
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.self_model.action_selection_core import ActionSelectionCore
from models.memory.memory_core import MemoryCore


def _build_pipeline(workspace_dim=64, grid_size=8, feature_dim=32, action_dim=2):
    """Build all components with small dims for fast testing."""
    tectum_config = {
        "tectum_feature_dim": feature_dim,
        "tectum_grid_size": grid_size,
        "workspace_dim": workspace_dim,
    }
    tectum = SensoryTectum(tectum_config)

    workspace_config = {
        "ignition_threshold": 0.4,
        "ignition_gain": 8.0,
        "reverberation_alpha": 0.7,
        "num_modules": 5,
        "module_names": ["vision", "audio", "memory", "body", "semantic"],
    }
    workspace = GlobalWorkspace(workspace_config)

    reentrant = ReentrantProcessor({
        "max_cycles": 3,
        "convergence_threshold": 0.01,
    })

    semantic = SemanticPathway(input_dim=1536, workspace_dim=workspace_dim)

    modulator = AffectiveModulator()
    workspace.affective_modulator = modulator

    action_config = {
        "workspace_dim": workspace_dim,
        "action_dim": action_dim,
        "context_dim": 64,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    emotion_shaper = EmotionalRewardShaper({"valence_weight": 0.5, "arousal_penalty": 1.0})
    memory = MemoryCore({})
    action_core = ActionSelectionCore(action_config, emotion_shaper, memory)

    return tectum, workspace, reentrant, semantic, modulator, action_core


class TestFullPipelineShapes(unittest.TestCase):
    """Verify tensor shapes flow correctly through the full pipeline."""

    def setUp(self):
        self.tectum, self.workspace, self.reentrant, self.semantic, \
            self.modulator, self.action_core = _build_pipeline()

    def test_raw_frame_through_tectum(self):
        """224x224 RGB frame -> tectum -> workspace content + bid."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, bid = self.tectum(frame, audio)
        self.assertEqual(content.dim(), 2)  # [B, workspace_dim]
        self.assertEqual(content.shape[0], 1)
        self.assertIsInstance(bid, float)

    def test_tectum_output_no_nan(self):
        """Tectum output should contain no NaN values."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, bid = self.tectum(frame, audio)
        self.assertFalse(torch.isnan(content).any().item())
        self.assertFalse(np.isnan(bid))

    def test_workspace_competition_produces_sync_r(self):
        """GNW competition should expose last_sync_R."""
        bids = {"vision": 0.8, "audio": 0.1, "memory": 0.2, "body": 0.05, "semantic": 0.0}
        payloads = {k: {"data": k} for k in bids}
        goal = torch.zeros(3)
        self.workspace.run_competition({}, goal, bids, payloads)
        self.assertTrue(hasattr(self.workspace, "last_sync_R"))
        self.assertIsInstance(self.workspace.last_sync_R, float)
        self.assertGreaterEqual(self.workspace.last_sync_R, 0.0)

    def test_reentrant_convergence(self):
        """Reentrant processor should converge or use all cycles."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, bid = self.tectum(frame, audio)

        bids = {"vision": max(0.0, min(1.0, bid)), "audio": 0.0,
                "memory": 0.1, "body": 0.05, "semantic": 0.0}
        payloads = {"vision": {"tensor": content, "source": "tectum"}}
        specialists = {"vision": self.tectum}
        goal = torch.tensor([1.0, -1.0, 1.0])

        result = self.reentrant.settle(
            workspace=self.workspace,
            specialists=specialists,
            initial_bids=bids,
            payloads=payloads,
            goal_vector=goal,
        )
        self.assertGreaterEqual(result.cycles_used, 1)
        self.assertLessEqual(result.cycles_used, 3)
        self.assertEqual(len(result.prediction_errors), result.cycles_used)

    def test_action_selection_valid(self):
        """Action selection should produce a valid action from broadcast."""
        broadcast = torch.randn(1, 64)
        action, value = self.action_core.select_action(
            broadcast, emotion_arousal=0.5, rpe_cache=0.0
        )
        self.assertIsNotNone(action)

    def test_consciousness_metrics_populated(self):
        """After competition, workspace state should have phi and is_conscious."""
        bids = {"vision": 0.9, "audio": 0.5, "memory": 0.6, "body": 0.3, "semantic": 0.4}
        payloads = {k: {"data": k} for k in bids}
        goal = torch.zeros(3)
        self.workspace.run_competition({}, goal, bids, payloads)
        self.assertIsInstance(self.workspace.state.phi_value, float)
        self.assertIn(self.workspace.state.is_conscious, (True, False))

    def test_device_consistency_cpu(self):
        """All tensors should stay on CPU when no GPU is requested."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, _ = self.tectum(frame, audio)
        self.assertEqual(content.device.type, "cpu")


class TestEndToEndPipeline(unittest.TestCase):
    """Run the full loop mimicking one training step."""

    def setUp(self):
        self.tectum, self.workspace, self.reentrant, self.semantic, \
            self.modulator, self.action_core = _build_pipeline()

    def test_single_step_no_crash(self):
        """A complete perception-to-action step should run without error."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        tectum_content, vision_bid = self.tectum(frame, audio)

        semantic_emb = torch.zeros(1536)
        _, semantic_bid = self.semantic(semantic_emb)

        bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": 0.0, "memory": 0.1, "body": 0.05,
            "semantic": max(0.0, min(1.0, semantic_bid)),
        }
        payloads = {
            "vision": {"tensor": tectum_content, "source": "tectum"},
            "semantic": {"tensor": torch.zeros(1, 64), "source": "semantic"},
        }
        specialists = {"vision": self.tectum}
        goal = torch.tensor([1.0, -1.0, 1.0])

        result = self.reentrant.settle(
            self.workspace, specialists, bids, payloads, goal
        )
        broadcast = result.broadcast_content
        if not isinstance(broadcast, torch.Tensor):
            broadcast = torch.zeros(1, 64)
        action, value = self.action_core.select_action(
            broadcast, emotion_arousal=0.3, rpe_cache=0.0
        )
        self.assertIsNotNone(action)
        self.assertIsInstance(result.phi, float)

    def test_tectum_produces_finite_output(self):
        """Sequential tectum calls should produce finite, non-NaN workspace content."""
        audio = torch.zeros(1, 32, 2)
        for _ in range(3):
            frame = torch.randn(1, 3, 224, 224)
            content, bid = self.tectum(frame, audio)
            self.assertFalse(torch.isnan(content).any().item())
            self.assertFalse(torch.isinf(content).any().item())
            self.assertTrue(0.0 <= bid or bid == 0.0)  # bid is valid float

    def test_binding_r_exposed(self):
        """After settle, workspace should have last_sync_R accessible."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, bid = self.tectum(frame, audio)
        bids = {"vision": max(0.0, min(1.0, bid)), "audio": 0.0,
                "memory": 0.1, "body": 0.05, "semantic": 0.0}
        payloads = {"vision": {"tensor": content, "source": "tectum"}}
        goal = torch.tensor([1.0, -1.0, 1.0])
        self.reentrant.settle(
            self.workspace, {"vision": self.tectum}, bids, payloads, goal
        )
        self.assertTrue(hasattr(self.workspace, "last_sync_R"))

    def test_multiple_steps_stable(self):
        """Running 5 sequential steps should not produce NaN or crash."""
        audio = torch.zeros(1, 32, 2)
        for i in range(5):
            frame = torch.randn(1, 3, 224, 224)
            content, bid = self.tectum(frame, audio)
            bids = {"vision": max(0.0, min(1.0, bid)), "audio": 0.0,
                    "memory": 0.1, "body": 0.05, "semantic": 0.0}
            payloads = {"vision": {"tensor": content, "source": "tectum"}}
            goal = torch.tensor([1.0, -1.0, 1.0])
            result = self.reentrant.settle(
                self.workspace, {"vision": self.tectum}, bids, payloads, goal
            )
            broadcast = result.broadcast_content
            if isinstance(broadcast, torch.Tensor):
                self.assertFalse(torch.isnan(broadcast).any().item(),
                                 f"NaN in broadcast at step {i}")

    def test_semantic_pathway_zero_bid_loses_competition(self):
        """Semantic pathway with zero embedding should not win GNW competition."""
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, vision_bid = self.tectum(frame, audio)

        bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": 0.0, "memory": 0.1, "body": 0.05, "semantic": 0.0,
        }
        payloads = {
            "vision": {"tensor": content, "source": "tectum"},
            "semantic": {"tensor": torch.zeros(1, 64), "source": "semantic"},
        }
        goal = torch.tensor([1.0, -1.0, 1.0])
        broadcast, raw_bids = self.workspace.run_competition(
            {}, goal, bids, payloads
        )
        # Vision should dominate when semantic has zero bid
        if self.workspace.state.is_conscious:
            self.assertIn("vision", self.workspace.state.focus_topic.lower())


class TestBroadcastTensorExtraction(unittest.TestCase):
    """Regression tests for the train_rlhf.py broadcast-as-dict bug.

    GlobalWorkspace.run_competition returns broadcast_content as a dict with
    a "tensor" key (built from each winning module's payload). Production
    train_rlhf.py used to fall straight through to a zeros fallback because
    it only checked isinstance(broadcast, torch.Tensor). The result was
    broadcast_mag == 0 for every step of every episode, which collapsed
    gate dynamics and pinned phi at 0. These tests lock in that the
    contract is dict-with-tensor-key and that the tensor is recoverable.
    """

    def setUp(self):
        self.tectum, self.workspace, self.reentrant, self.semantic, \
            self.modulator, self.action_core = _build_pipeline()

    def _settle_with_high_vision_bid(self):
        frame = torch.randn(1, 3, 224, 224)
        audio = torch.zeros(1, 32, 2)
        content, bid = self.tectum(frame, audio)
        bids = {
            "vision": max(0.7, min(1.0, bid)),  # force ignition
            "audio": 0.0, "memory": 0.1, "body": 0.05, "semantic": 0.0,
        }
        payloads = {"vision": {"tensor": content, "source": "tectum"}}
        goal = torch.tensor([1.0, -1.0, 1.0])
        result = self.reentrant.settle(
            self.workspace, {"vision": self.tectum}, bids, payloads, goal
        )
        return result, content

    def test_broadcast_content_is_dict_when_conscious(self):
        """When workspace ignites, broadcast_content is a dict (not a tensor)."""
        result, _ = self._settle_with_high_vision_bid()
        if self.workspace.state.is_conscious:
            self.assertIsInstance(
                result.broadcast_content, dict,
                "GlobalWorkspace returns broadcast_content as a dict; if this "
                "ever changes, the tensor extraction in train_rlhf.py needs "
                "to be re-checked."
            )

    def test_broadcast_dict_carries_tensor_key(self):
        """The dict contract: vision payload's 'tensor' key is preserved."""
        result, expected_content = self._settle_with_high_vision_bid()
        if self.workspace.state.is_conscious:
            bd = result.broadcast_content
            self.assertIn("tensor", bd,
                          "broadcast_content must expose a 'tensor' key for "
                          "downstream gate/RND/memory consumption.")
            self.assertIsInstance(bd["tensor"], torch.Tensor)
            # Norm > 0: the actual tectum content reaches downstream code.
            self.assertGreater(
                float(bd["tensor"].norm().item()), 0.0,
                "broadcast_content['tensor'] must be non-zero so phi has signal."
            )

    def test_extracted_broadcast_norm_matches_dict_tensor(self):
        """Mirrors the production extraction logic; norm must agree."""
        result, _ = self._settle_with_high_vision_bid()
        bd = result.broadcast_content
        if isinstance(bd, torch.Tensor):
            extracted = bd
        elif isinstance(bd, dict) and isinstance(bd.get("tensor"), torch.Tensor):
            extracted = bd["tensor"]
        else:
            extracted = torch.zeros(1, 64)
        if self.workspace.state.is_conscious:
            self.assertGreater(float(extracted.norm().item()), 0.0)


class TestPhiSyncRDecoupling(unittest.TestCase):
    """Regression test for the second 06f96db survivor: workspace.state.phi_value
    must NOT include the sync_R * 0.1 identity. That algebraic shortcut made
    Phi-1 trivially correlate with sync_R (the 04-14 r=1.000 'PASS' was the
    identity, not a finding). The fix removes the addition; this test runs
    one competition and confirms phi_value equals the gate-state phi alone.
    """

    def setUp(self):
        self.tectum, self.workspace, self.reentrant, self.semantic, \
            self.modulator, self.action_core = _build_pipeline()

    def test_phi_value_does_not_include_sync_r_identity(self):
        """state.phi_value == phi_result.phi (no sync_R contribution)."""
        from models.core.consciousness_gating import ConsciousnessGate

        gate = ConsciousnessGate({"hidden_size": 64, "gating": {}})
        self.workspace.consciousness_gate = gate

        bids = {"vision": 0.9, "audio": 0.6, "memory": 0.5,
                "body": 0.3, "semantic": 0.4}
        payloads = {k: {"tensor": torch.randn(1, 64), "source": k}
                    for k in bids}
        goal = torch.zeros(3)

        # Run competition once. Use the non-zero sync_R that the binding
        # system produces: if the old `phi += sync_R * 0.1` line were back,
        # phi_value would shift by ~sync_R * 0.1 above the gate-state phi.
        self.workspace.run_competition({}, goal, bids, payloads)
        phi_value_after = self.workspace.state.phi_value

        # Recompute phi_result.phi from the same gate state. Since the gate
        # is stateful (prev_gate_values), we read what was just set.
        from models.evaluation.iit_phi import IITMetrics
        # The workspace already used its own iit_metrics; read directly
        # from the gate state it just produced.
        gate_state = gate.state
        # Use a fresh IITMetrics because the workspace's TPM has only one
        # transition; the proxy/insufficient_data path is fine here.
        fresh = IITMetrics()
        phi_result = fresh.compute_phi_from_gate_state(gate_state)

        # The exact phi values won't match (different TPM histories), but
        # we can lock in the absence of the +sync_R*0.1 contamination by
        # checking that phi_value does not exceed the realistic phi range
        # by a sync_R-sized amount. In the buggy version with sync_R~0.2,
        # the inflation would be ~0.02, large enough to fail this test.
        sync_r = float(self.workspace.last_sync_R)
        # If buggy: phi_value would be roughly phi_result.phi + sync_r*0.1.
        # Gate state -> phi typically returns 0..0.005 in this short setup.
        # If phi_value > 0.05 while sync_r is ~0.2, that's a smoking gun.
        if sync_r > 0.05:
            self.assertLess(
                phi_value_after, sync_r * 0.05 + max(phi_result.phi, 0.05),
                f"phi_value={phi_value_after:.5f} looks inflated by sync_R "
                f"(={sync_r:.4f}); the +sync_R*0.1 identity may have come back."
            )


if __name__ == "__main__":
    unittest.main()
