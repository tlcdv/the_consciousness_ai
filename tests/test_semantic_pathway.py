"""Tests for the SemanticPathway module."""
from __future__ import annotations

import unittest
import torch

from models.core.semantic_pathway import SemanticPathway


class TestSemanticPathway(unittest.TestCase):

    def setUp(self):
        self.pathway = SemanticPathway(input_dim=1536, workspace_dim=256)

    def test_forward_shape(self):
        """Output content should be [1, workspace_dim]."""
        embedding = torch.randn(1536)
        content, bid = self.pathway(embedding)
        self.assertEqual(content.shape, (1, 256))

    def test_forward_batched(self):
        """Batched input should produce matching batch output."""
        embedding = torch.randn(4, 1536)
        content, bid = self.pathway(embedding)
        self.assertEqual(content.shape, (4, 256))

    def test_zero_embedding_zero_bid(self):
        """Zero embedding should produce zero bid."""
        embedding = torch.zeros(1536)
        content, bid = self.pathway(embedding)
        self.assertAlmostEqual(bid, 0.0, places=5)

    def test_nonzero_embedding_positive_bid(self):
        """Non-zero embedding should produce positive bid."""
        embedding = torch.randn(1536)
        content, bid = self.pathway(embedding)
        self.assertGreater(bid, 0.0)

    def test_bid_clamped_to_unit(self):
        """Bid should be clamped to [0, 1]."""
        embedding = torch.randn(1536) * 1000
        _, bid = self.pathway(embedding)
        self.assertLessEqual(bid, 1.0)
        self.assertGreaterEqual(bid, 0.0)

    def test_gradient_flow(self):
        """Gradients should flow through the projection."""
        # Check that projection weights receive gradients
        embedding = torch.randn(1, 1536)
        content, _ = self.pathway(embedding)
        loss = content.pow(2).sum()
        loss.backward()
        proj_weight = self.pathway.projection[0].weight
        self.assertIsNotNone(proj_weight.grad)
        self.assertGreater(proj_weight.grad.abs().sum().item(), 0.0)

    def test_receive_broadcast_tensor(self):
        """receive_broadcast with tensor should return updated bid."""
        embedding = torch.randn(1536)
        content, bid = self.pathway(embedding)
        broadcast = content.detach().clone()
        updated = self.pathway.receive_broadcast(broadcast, bid)
        # Broadcast matches content exactly -> PE near 0 -> bid increases
        self.assertGreaterEqual(updated, bid)

    def test_receive_broadcast_no_content(self):
        """receive_broadcast before any forward should return current bid."""
        pathway = SemanticPathway()
        result = pathway.receive_broadcast(torch.zeros(256), 0.5)
        self.assertEqual(result, 0.5)


class TestSemanticPathwayGNWIntegration(unittest.TestCase):

    def test_gnw_with_semantic_module(self):
        """GlobalWorkspace should accept 5-module bids including semantic."""
        from models.core.global_workspace import GlobalWorkspace

        config = {"ignition_threshold": 0.5, "ignition_gain": 10.0}
        gnw = GlobalWorkspace(config)

        bids = {
            "vision": 0.7, "audio": 0.3, "memory": 0.4,
            "body": 0.2, "semantic": 0.6,
        }
        payloads = {k: {"data": k} for k in bids}
        goal = torch.zeros(3)

        broadcast, result_bids = gnw.run_competition(
            inputs={}, goal_vector=goal, bids=bids, payloads=payloads
        )
        self.assertIsInstance(result_bids, dict)
        # last_sync_R should be exposed
        self.assertTrue(hasattr(gnw, 'last_sync_R'))
        self.assertIsInstance(gnw.last_sync_R, float)


if __name__ == "__main__":
    unittest.main()
