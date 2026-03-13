import unittest
import torch

from models.core.topographic_loss import topographic_spatial_loss


class TestTopographicSpatialLoss(unittest.TestCase):

    def test_returns_scalar(self):
        """Loss should be a scalar tensor."""
        x = torch.randn(2, 16, 8, 8)
        loss = topographic_spatial_loss(x)
        self.assertEqual(loss.dim(), 0)

    def test_finite_output(self):
        """Loss should be finite for normal input."""
        x = torch.randn(2, 16, 8, 8)
        loss = topographic_spatial_loss(x)
        self.assertTrue(torch.isfinite(loss))

    def test_smooth_gradient_lower_loss(self):
        """A spatially smooth feature map (nearby cells similar) should
        produce lower loss than a random one, since topographic loss
        penalizes spatial disorganization."""
        torch.manual_seed(42)

        # Smooth: each cell's features depend on its spatial position
        H, W, C = 8, 8, 16
        coords_y = torch.linspace(0, 1, H).unsqueeze(1).unsqueeze(0).expand(1, -1, W)
        coords_x = torch.linspace(0, 1, W).unsqueeze(0).unsqueeze(0).expand(1, H, -1)
        # Stack and repeat across channels: nearby cells will have similar values
        smooth_map = torch.cat([
            coords_y.unsqueeze(1).expand(-1, C // 2, -1, -1),
            coords_x.unsqueeze(1).expand(-1, C // 2, -1, -1),
        ], dim=1)  # [1, C, H, W]

        # Random: no spatial structure
        random_map = torch.randn(1, C, H, W)

        loss_smooth = topographic_spatial_loss(smooth_map, alpha=1.0)
        loss_random = topographic_spatial_loss(random_map, alpha=1.0)

        # Smooth should have lower (more negative) loss than random
        self.assertLess(loss_smooth.item(), loss_random.item())

    def test_alpha_scaling(self):
        """Loss should scale proportionally with alpha."""
        x = torch.randn(1, 16, 8, 8)
        loss_025 = topographic_spatial_loss(x, alpha=0.25)
        loss_050 = topographic_spatial_loss(x, alpha=0.50)

        # loss_050 should be roughly 2x loss_025
        ratio = loss_050.item() / (loss_025.item() + 1e-10)
        self.assertAlmostEqual(ratio, 2.0, places=1)

    def test_gradient_flows(self):
        """Loss should be differentiable."""
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        loss = topographic_spatial_loss(x)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))

    def test_batch_independence(self):
        """Loss should handle different batch sizes."""
        x1 = torch.randn(1, 16, 8, 8)
        x4 = torch.randn(4, 16, 8, 8)
        loss1 = topographic_spatial_loss(x1)
        loss4 = topographic_spatial_loss(x4)
        self.assertTrue(torch.isfinite(loss1))
        self.assertTrue(torch.isfinite(loss4))


if __name__ == '__main__':
    unittest.main()
