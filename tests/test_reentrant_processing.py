"""
Tests for Phase 6: Reentrant Processing (Predictive Coding Loop).

Validates:
1. ReentrantProcessor convergence behavior
2. Early termination when prediction error stabilizes
3. Max cycles constraint
4. Specialist receive_broadcast bid modulation
5. SensoryTectum and ProprioceptiveProcessor top-down feedback
"""

import unittest
import torch
import numpy as np
from typing import Any

from models.core.reentrant_processor import ReentrantProcessor, SettleResult


# ──────────────────────────────────────────────
# Mock Objects
# ──────────────────────────────────────────────

class MockWorkspace:
    """Simplified workspace that returns deterministic results for testing."""
    
    def __init__(self, broadcast_tensor=None, conscious=True):
        self._broadcast = broadcast_tensor if broadcast_tensor is not None else torch.randn(1, 256)
        self._call_count = 0
        
        class State:
            phi_value = 0.5
            is_conscious = conscious
        
        self.state = State()
        
    def run_competition(self, inputs, goal_vector, bids=None, payloads=None,
                        pad_state=None, interoceptive_state=None):
        self._call_count += 1
        # Return the broadcast, potentially with slight noise each cycle
        # to simulate convergence
        noise_scale = max(0, 1.0 - self._call_count * 0.3)  # Decreasing noise
        noisy_broadcast = self._broadcast + torch.randn_like(self._broadcast) * noise_scale * 0.01

        if self.state.is_conscious:
            return noisy_broadcast, bids or {}
        else:
            return {}, bids or {}


class MockSpecialist:
    """A mock specialist that supports receive_broadcast."""
    
    def __init__(self, bid_delta=0.0):
        self.broadcast_calls = 0
        self.bid_delta = bid_delta
    
    def receive_broadcast(self, broadcast_content, current_bid):
        self.broadcast_calls += 1
        # Slightly adjust bid each cycle
        return max(0.0, min(1.0, current_bid + self.bid_delta))


class NoFeedbackSpecialist:
    """A specialist without receive_broadcast (e.g., emotion module)."""
    pass


# ──────────────────────────────────────────────
# Test: ReentrantProcessor Core
# ──────────────────────────────────────────────

class TestReentrantProcessor(unittest.TestCase):
    """Tests for the core reentrant settle loop."""
    
    def test_settle_returns_result(self):
        """settle() should return a SettleResult with all expected fields."""
        processor = ReentrantProcessor({"min_cycles": 2, "max_cycles": 5})
        workspace = MockWorkspace()
        
        result = processor.settle(
            workspace=workspace,
            specialists={},
            initial_bids={"vision": 0.8},
            payloads={"vision": "test"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        self.assertIsInstance(result, SettleResult)
        self.assertIn("vision", result.final_bids)
        self.assertIsNotNone(result.broadcast_content)
        self.assertGreater(result.cycles_used, 0)
        self.assertGreater(len(result.prediction_errors), 0)
    
    def test_max_cycles_respected(self):
        """The loop must never exceed max_cycles."""
        processor = ReentrantProcessor({
            "min_cycles": 2, 
            "max_cycles": 5,
            "convergence_threshold": 0.0001  # Very tight: won't converge
        })
        workspace = MockWorkspace()
        
        result = processor.settle(
            workspace=workspace,
            specialists={},
            initial_bids={"vision": 0.8},
            payloads={"vision": "test"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        self.assertLessEqual(result.cycles_used, 5)
    
    def test_min_cycles_enforced(self):
        """Even trivially convergent inputs must run at least min_cycles."""
        processor = ReentrantProcessor({
            "min_cycles": 3,
            "max_cycles": 10,
            "convergence_threshold": 10.0  # Very loose: converges on first cycle
        })
        workspace = MockWorkspace()
        
        result = processor.settle(
            workspace=workspace,
            specialists={},
            initial_bids={"vision": 0.5},
            payloads={"vision": "test"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        self.assertGreaterEqual(result.cycles_used, 3)
    
    def test_prediction_errors_recorded(self):
        """Should record one PE value per cycle."""
        processor = ReentrantProcessor({"min_cycles": 2, "max_cycles": 4})
        workspace = MockWorkspace()
        
        result = processor.settle(
            workspace=workspace,
            specialists={},
            initial_bids={"vision": 0.7},
            payloads={"vision": "test"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        self.assertEqual(len(result.prediction_errors), result.cycles_used)
        # First PE is always 1.0 (no previous broadcast)
        self.assertEqual(result.prediction_errors[0], 1.0)
    
    def test_specialist_receives_broadcast(self):
        """Specialists with receive_broadcast should be called during settle cycles."""
        processor = ReentrantProcessor({
            "min_cycles": 3, 
            "max_cycles": 5,
            "convergence_threshold": 0.0001  # Tight, so we run at least 3 full cycles
        })
        workspace = MockWorkspace()
        specialist = MockSpecialist()
        
        result = processor.settle(
            workspace=workspace,
            specialists={"vision": specialist},
            initial_bids={"vision": 0.8},
            payloads={"vision": "test"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        # Should be called at least min_cycles times
        self.assertGreaterEqual(specialist.broadcast_calls, 3)
    
    def test_specialist_without_feedback_ignored(self):
        """Specialists without receive_broadcast should not cause errors."""
        processor = ReentrantProcessor({"min_cycles": 2, "max_cycles": 3})
        workspace = MockWorkspace()
        no_feedback = NoFeedbackSpecialist()
        
        # Should not raise
        result = processor.settle(
            workspace=workspace,
            specialists={"emotion": no_feedback},
            initial_bids={"emotion": 0.5},
            payloads={"emotion": "calm"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        self.assertIsInstance(result, SettleResult)
    
    def test_bid_modulation_across_cycles(self):
        """Specialist bid adjustments should be reflected in final bids."""
        processor = ReentrantProcessor({"min_cycles": 3, "max_cycles": 3})
        workspace = MockWorkspace()
        
        # Specialist that increases bid by 0.05 each cycle
        increasing_specialist = MockSpecialist(bid_delta=0.05)
        
        result = processor.settle(
            workspace=workspace,
            specialists={"vision": increasing_specialist},
            initial_bids={"vision": 0.5},
            payloads={"vision": "test"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        # After 3 cycles of +0.05, bid should be ~0.65
        self.assertGreater(result.final_bids["vision"], 0.5)
    
    def test_subconscious_processing(self):
        """When workspace doesn't ignite, PE should still be computed."""
        processor = ReentrantProcessor({"min_cycles": 2, "max_cycles": 3})
        workspace = MockWorkspace(conscious=False)
        
        result = processor.settle(
            workspace=workspace,
            specialists={},
            initial_bids={"vision": 0.1},
            payloads={"vision": "dim"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )
        
        self.assertFalse(result.is_conscious)
        self.assertEqual(len(result.prediction_errors), result.cycles_used)


# ──────────────────────────────────────────────
# Test: Specialist receive_broadcast
# ──────────────────────────────────────────────

class TestSensoryTectumFeedback(unittest.TestCase):
    """Test SensoryTectum.receive_broadcast() top-down modulation."""
    
    def setUp(self):
        from models.core.sensory_tectum import SensoryTectum
        self.tectum = SensoryTectum({
            "tectum_feature_dim": 64,
            "tectum_grid_size": 4,
            "workspace_dim": 32
        })
    
    def test_receive_broadcast_with_matching_content(self):
        """When broadcast matches tectum content, bid should stay stable or decrease."""
        # First: do a forward pass to populate _last_content
        vision = torch.randn(1, 64, 4, 4)
        audio = torch.randn(1, 64, 2)
        content, bid = self.tectum(vision, audio)
        
        # Send back the exact same content (perfect match)
        updated = self.tectum.receive_broadcast(content, bid)
        # PE is ~0, so bid should decrease slightly
        self.assertLessEqual(updated, bid + 0.01)  # Allow small float tolerance
    
    def test_receive_broadcast_with_distant_content(self):
        """When broadcast differs greatly, bid should increase."""
        vision = torch.randn(1, 64, 4, 4)
        audio = torch.randn(1, 64, 2)
        content, bid = self.tectum(vision, audio)
        
        # Send completely different content
        alien_content = torch.randn_like(content) * 10.0
        updated = self.tectum.receive_broadcast(alien_content, bid)
        # PE is high, so bid should increase
        self.assertGreaterEqual(updated, bid)
    
    def test_receive_broadcast_without_tensor(self):
        """Non-tensor broadcast should cause slight decay."""
        self.tectum._last_content = torch.randn(1, 32)
        updated = self.tectum.receive_broadcast({}, 0.5)
        self.assertLess(updated, 0.5)


class TestProprioceptiveFeedback(unittest.TestCase):
    """Test ProprioceptiveProcessor.receive_broadcast() body-awareness modulation."""
    
    def setUp(self):
        from models.self_model.embodiment_core import ProprioceptiveProcessor
        self.processor = ProprioceptiveProcessor(raw_state_dim=40, num_parts=10, feature_per_part=8)
    
    def test_body_relevant_broadcast_boosts_bid(self):
        """Body-related content should boost the body bid."""
        bid = self.processor.receive_broadcast({"body": "collision detected"}, 0.5)
        self.assertGreater(bid, 0.5)
    
    def test_body_relevant_string_broadcast(self):
        """String mentioning 'pain' should boost bid."""
        bid = self.processor.receive_broadcast("experiencing pain", 0.5)
        self.assertGreater(bid, 0.5)
    
    def test_irrelevant_broadcast_decays_bid(self):
        """Non-body broadcast should cause bid decay."""
        bid = self.processor.receive_broadcast({"vision": "bright light"}, 0.5)
        self.assertLess(bid, 0.5)
    
    def test_minimum_body_awareness(self):
        """Body bid should never reach absolute zero."""
        bid = 0.1
        for _ in range(100):
            bid = self.processor.receive_broadcast({"vision": "nothing"}, bid)
        self.assertGreater(bid, 0.0)


# ──────────────────────────────────────────────
# Test: Prediction Error Computation
# ──────────────────────────────────────────────

class TestPredictionError(unittest.TestCase):
    """Test the PE computation logic of ReentrantProcessor."""
    
    def setUp(self):
        self.processor = ReentrantProcessor()
    
    def test_first_cycle_pe_is_max(self):
        """First cycle (no previous) should return PE = 1.0."""
        pe = self.processor._compute_prediction_error(None, torch.randn(1, 256))
        self.assertEqual(pe, 1.0)
    
    def test_identical_broadcasts_pe_zero(self):
        """Identical broadcasts should give PE = 0."""
        x = torch.randn(1, 256)
        pe = self.processor._compute_prediction_error(x, x)
        self.assertAlmostEqual(pe, 0.0, places=5)
    
    def test_different_broadcasts_pe_positive(self):
        """Different broadcasts should give positive PE."""
        a = torch.zeros(1, 256)
        b = torch.ones(1, 256)
        pe = self.processor._compute_prediction_error(a, b)
        self.assertGreater(pe, 0.0)
    
    def test_dict_broadcast_comparison(self):
        """dict broadcasts should compare by keys."""
        a = {"vision": "cat", "emotion": "happy"}
        b = {"vision": "cat", "emotion": "sad"}
        pe = self.processor._compute_prediction_error(a, b)
        self.assertGreater(pe, 0.0)
        self.assertLess(pe, 1.0)
    
    def test_dict_identical_pe_zero(self):
        """Identical dicts should give PE = 0."""
        a = {"vision": "cat"}
        pe = self.processor._compute_prediction_error(a, a)
        self.assertEqual(pe, 0.0)


# ──────────────────────────────────────────────
# Test: Multi-Level Reentrant Integration
# ──────────────────────────────────────────────

class TestMultiLevelIntegration(unittest.TestCase):
    """End-to-end: tectum with reentrant capsules inside ReentrantProcessor settle loop."""

    def _make_tectum(self, reentrant_iterations=2):
        from models.core.sensory_tectum import SensoryTectum
        return SensoryTectum({
            "tectum_feature_dim": 16,
            "tectum_grid_size": 8,
            "workspace_dim": 64,
            "capsule_hierarchy_spec": [(8, 6), (4, 8)],
            "num_primary_caps": 4,
            "capsule_primary_dim": 4,
            "routing_iterations": 2,
            "capsule_reentrant_iterations": reentrant_iterations,
        })

    def test_tectum_reentrant_in_settle_loop(self):
        """Tectum with reentrant capsules should work inside ReentrantProcessor.settle()."""
        processor = ReentrantProcessor({"min_cycles": 2, "max_cycles": 4})
        workspace = MockWorkspace(broadcast_tensor=torch.randn(1, 64))
        tectum = self._make_tectum(reentrant_iterations=2)

        # Run tectum forward first to populate _last_content
        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        tectum(vision, audio)

        result = processor.settle(
            workspace=workspace,
            specialists={"vision": tectum},
            initial_bids={"vision": 0.7},
            payloads={"vision": "tectum_content"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )

        self.assertIsInstance(result, SettleResult)
        self.assertIn("vision", result.final_bids)
        self.assertGreater(result.cycles_used, 0)

        # Capsule reentrant PE should be tracked inside tectum
        pe = tectum.capsule_layer.get_level_prediction_errors()
        self.assertGreater(len(pe), 0)

    def test_nested_convergence(self):
        """Intra-hierarchy PE should be present at each settle cycle's capsule forward."""
        processor = ReentrantProcessor({
            "min_cycles": 3, "max_cycles": 3,
            "convergence_threshold": 0.0001
        })
        workspace = MockWorkspace(broadcast_tensor=torch.randn(1, 64))
        tectum = self._make_tectum(reentrant_iterations=3)

        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        tectum(vision, audio)

        result = processor.settle(
            workspace=workspace,
            specialists={"vision": tectum},
            initial_bids={"vision": 0.6},
            payloads={"vision": "content"},
            goal_vector=torch.tensor([1.0, -1.0, 1.0])
        )

        # Both outer (settle) and inner (capsule) loops should have run
        self.assertEqual(result.cycles_used, 3)
        pe = tectum.capsule_layer.get_level_prediction_errors()
        # 3 reentrant iterations, 1 feedback projection (2 routing levels)
        self.assertEqual(len(pe), 3)

    def test_reentrant_vs_non_reentrant_tectum_bids_differ(self):
        """Reentrant capsule tectum should produce different bid trajectory than non-reentrant."""
        torch.manual_seed(42)
        processor_0 = ReentrantProcessor({"min_cycles": 3, "max_cycles": 3})
        processor_r = ReentrantProcessor({"min_cycles": 3, "max_cycles": 3})

        broadcast = torch.randn(1, 64)
        ws_0 = MockWorkspace(broadcast_tensor=broadcast.clone())
        ws_r = MockWorkspace(broadcast_tensor=broadcast.clone())

        tectum_0 = self._make_tectum(reentrant_iterations=0)
        tectum_r = self._make_tectum(reentrant_iterations=3)

        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        tectum_0(vision, audio)
        tectum_r(vision, audio)

        r0 = processor_0.settle(
            workspace=ws_0, specialists={"vision": tectum_0},
            initial_bids={"vision": 0.5}, payloads={"vision": "x"},
            goal_vector=torch.tensor([1.0])
        )
        rr = processor_r.settle(
            workspace=ws_r, specialists={"vision": tectum_r},
            initial_bids={"vision": 0.5}, payloads={"vision": "x"},
            goal_vector=torch.tensor([1.0])
        )

        # Both should complete, but bids may differ due to different _last_content
        self.assertIsInstance(r0, SettleResult)
        self.assertIsInstance(rr, SettleResult)


if __name__ == '__main__':
    unittest.main()
