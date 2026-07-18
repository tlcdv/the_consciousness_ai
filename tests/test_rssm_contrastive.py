"""Tests for RSSMContrastiveHead (Path B stage 2, label-free identity objective)."""
from __future__ import annotations

import torch
import pytest

from models.core.rssm_contrastive import RSSMContrastiveHead


class TestRSSMContrastiveHead:
    """RSSMContrastiveHead construction, forward shape, loss shape, and gradient flow."""

    def _head(self, rssm_channels=16, grid=8, proj_dim=8, temperature=0.1):
        return RSSMContrastiveHead(
            rssm_channels=rssm_channels, grid=grid,
            reduce_channels=8, hidden_dim=32,
            proj_dim=proj_dim, temperature=temperature,
        )

    def test_forward_shape(self):
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        x = torch.randn(2, 16, 8, 8)
        out = h(x)
        assert out.shape == (2, 8), f"expected (2, 8), got {out.shape}"
        # Check L2-normalized: each row should have norm ~1
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward_single_sample(self):
        """3-D input (no batch dim) should be handled."""
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        x = torch.randn(16, 8, 8)
        out = h(x)
        assert out.shape == (1, 8)

    def test_forward_reduced_grid(self):
        """Works with a smaller grid (e.g. 4x4)."""
        h = self._head(rssm_channels=16, grid=4, proj_dim=8)
        x = torch.randn(2, 16, 4, 4)
        out = h(x)
        assert out.shape == (2, 8)

    def test_loss_shape(self):
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        anchor = torch.randn(4, 8)
        positive = torch.randn(4, 8)
        negatives = torch.randn(4, 6, 8)
        loss = h.loss(anchor, positive, negatives)
        assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"
        assert loss.item() >= 0.0, "loss should be non-negative"

    def test_loss_with_negative_broadcast(self):
        """2-D negatives (no per-anchor dimension) should broadcast to all anchors."""
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        anchor = torch.randn(4, 8)
        positive = torch.randn(4, 8)
        negatives = torch.randn(6, 8)  # no batch dim
        loss = h.loss(anchor, positive, negatives)
        assert loss.dim() == 0

    def test_loss_lower_for_matching_pairs(self):
        """Loss should be lower when anchor matches positive vs a different positive."""
        h = self._head(rssm_channels=16, grid=8, proj_dim=8, temperature=0.5)

        # Matching pairs: anchor[i] aligns with positive[i] (same direction)
        anchor = torch.randn(4, 8)
        positive = anchor.clone() + torch.randn(4, 8) * 0.1  # small noise
        negatives = torch.randn(4, 6, 8) * 3.0  # far away

        loss_match = h.loss(anchor, positive, negatives).item()

        # Non-matching: positive is random
        positive_random = torch.randn(4, 8)
        loss_random = h.loss(anchor, positive_random, negatives).item()

        assert loss_match < loss_random, (
            f"matching loss {loss_match:.4f} should be < random {loss_random:.4f}"
        )

    def test_gradient_flow_to_input(self):
        """Gradients should flow from the contrastive loss back to the input tensor."""
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        out = h(x)
        anchor, positive = out[:1], out[1:2]
        negatives = torch.randn(1, 3, 8)  # shared negatives for the single anchor
        loss = h.loss(anchor, positive, negatives)
        loss.backward()
        assert x.grad is not None, "gradient is None"
        assert x.grad.abs().sum().item() > 0, "gradient is zero"

    def test_gradient_flow_to_parameters(self):
        """Gradients should flow through the head's parameters."""
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        x = torch.randn(2, 16, 8, 8)
        out = h(x)
        anchor, positive = out[:1], out[1:2]
        negatives = torch.randn(1, 3, 8)
        loss = h.loss(anchor, positive, negatives)
        loss.backward()
        has_grad = False
        for p in h.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "no parameter received a gradient"

    def test_full_rssm_channels(self):
        """Head initializes and forward-passes at the full rssm_channels (1088) used in training."""
        h = RSSMContrastiveHead(
            rssm_channels=1088, grid=16,
            reduce_channels=32, hidden_dim=256,
            proj_dim=128, temperature=0.1,
        )
        x = torch.randn(2, 1088, 16, 16)
        out = h(x)
        assert out.shape == (2, 128)

    def test_loss_with_one_negative(self):
        """Edge case: single negative per anchor."""
        h = self._head(rssm_channels=16, grid=8, proj_dim=8)
        anchor = torch.randn(4, 8)
        positive = torch.randn(4, 8)
        negatives = torch.randn(4, 1, 8)
        loss = h.loss(anchor, positive, negatives)
        assert loss.dim() == 0
        assert loss.item() >= 0.0
