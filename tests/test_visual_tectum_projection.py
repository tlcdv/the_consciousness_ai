import unittest
import torch

from models.core.visual_tectum_projection import VisualTectumProjection


class TestVisualTectumProjection(unittest.TestCase):

    def test_output_shape_from_14x14_stub(self):
        """Qwen2-VL stub returns [1536, 14, 14]. Output should be [1, 64, 16, 16]."""
        proj = VisualTectumProjection(in_channels=1536, out_channels=64, target_grid=16)
        x = torch.zeros(1536, 14, 14)
        out = proj(x)
        self.assertEqual(out.shape, (1, 64, 16, 16))

    def test_output_shape_from_28x28(self):
        """Higher resolution input should still produce [1, 64, 16, 16]."""
        proj = VisualTectumProjection(in_channels=1536, out_channels=64, target_grid=16)
        x = torch.randn(1536, 28, 28)
        out = proj(x)
        self.assertEqual(out.shape, (1, 64, 16, 16))

    def test_batched_input(self):
        """4D input [B, C, H, W] should preserve batch dimension."""
        proj = VisualTectumProjection(in_channels=1536, out_channels=64, target_grid=16)
        x = torch.randn(2, 1536, 14, 14)
        out = proj(x)
        self.assertEqual(out.shape, (2, 64, 16, 16))

    def test_zero_input_near_zero_output(self):
        """Zero input (stub mode) should produce near-zero output."""
        proj = VisualTectumProjection(in_channels=1536, out_channels=64, target_grid=16)
        x = torch.zeros(1536, 14, 14)
        out = proj(x)
        self.assertLess(out.abs().max().item(), 1e-5)

    def test_configurable_out_channels(self):
        """out_channels=32 should produce [1, 32, 16, 16]."""
        proj = VisualTectumProjection(in_channels=1536, out_channels=32, target_grid=16)
        x = torch.randn(1536, 14, 14)
        out = proj(x)
        self.assertEqual(out.shape, (1, 32, 16, 16))

    def test_configurable_target_grid(self):
        """target_grid=8 should produce [1, 64, 8, 8]."""
        proj = VisualTectumProjection(in_channels=1536, out_channels=64, target_grid=8)
        x = torch.randn(1536, 14, 14)
        out = proj(x)
        self.assertEqual(out.shape, (1, 64, 8, 8))


if __name__ == '__main__':
    unittest.main()
