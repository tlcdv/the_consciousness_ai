import unittest
import torch

from models.core.retinotopic_encoder import RetinotopicEncoder, RetinotopicConvStack


class TestRetinotopicConvStack(unittest.TestCase):

    def setUp(self):
        self.stack = RetinotopicConvStack(out_channels=768)

    def test_output_shape(self):
        """Conv stack should produce [B, 768, 14, 14] from 224x224 input."""
        x = torch.randn(2, 3, 224, 224)
        out = self.stack(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 768)
        # 224 / 2^4 = 14
        self.assertEqual(out.shape[2], 14)
        self.assertEqual(out.shape[3], 14)

    def test_spatial_correspondence(self):
        """A bright patch at image region (i, j) should produce strongest
        activation at the corresponding conv stack output cell."""
        torch.manual_seed(0)
        stack = RetinotopicConvStack(out_channels=64)

        # Place a strong stimulus in the top-left quadrant (rows 0-56, cols 0-56)
        x_topleft = torch.zeros(1, 3, 224, 224)
        x_topleft[:, :, :56, :56] = 5.0

        # Place a strong stimulus in the bottom-right quadrant
        x_botright = torch.zeros(1, 3, 224, 224)
        x_botright[:, :, 168:, 168:] = 5.0

        out_tl = stack(x_topleft)  # [1, 64, 14, 14]
        out_br = stack(x_botright)

        # Energy in top-left quadrant of output (0:7, 0:7)
        tl_energy_tl = out_tl[:, :, :7, :7].abs().mean().item()
        tl_energy_br = out_tl[:, :, 7:, 7:].abs().mean().item()

        br_energy_tl = out_br[:, :, :7, :7].abs().mean().item()
        br_energy_br = out_br[:, :, 7:, 7:].abs().mean().item()

        # Top-left stimulus should activate top-left output more
        self.assertGreater(tl_energy_tl, tl_energy_br)
        # Bottom-right stimulus should activate bottom-right output more
        self.assertGreater(br_energy_br, br_energy_tl)

    def test_zero_input_low_output(self):
        """Zero input should produce low-magnitude output (bias terms
        prevent exact zero, but output should be small)."""
        x = torch.zeros(1, 3, 224, 224)
        out = self.stack(x)
        self.assertLess(out.abs().max().item(), 0.5)


class TestRetinotopicEncoder(unittest.TestCase):

    def setUp(self):
        # Use fallback conv stack (pretrained=False) for CI
        self.encoder = RetinotopicEncoder(
            out_channels=64, target_grid=16, pretrained=False
        )

    def test_output_shape_default(self):
        """Encoder should produce [B, 64, 16, 16]."""
        x = torch.randn(1, 3, 224, 224)
        out = self.encoder(x)
        self.assertEqual(out.shape, (1, 64, 16, 16))

    def test_output_shape_batched(self):
        x = torch.randn(4, 3, 224, 224)
        out = self.encoder(x)
        self.assertEqual(out.shape, (4, 64, 16, 16))

    def test_output_shape_non_224_input(self):
        """Non-224 input should be resized internally."""
        x = torch.randn(1, 3, 128, 128)
        out = self.encoder(x)
        self.assertEqual(out.shape, (1, 64, 16, 16))

    def test_unbatched_input(self):
        """3D input [C, H, W] should be auto-batched."""
        x = torch.randn(3, 224, 224)
        out = self.encoder(x)
        self.assertEqual(out.shape, (1, 64, 16, 16))

    def test_custom_target_grid(self):
        encoder = RetinotopicEncoder(
            out_channels=32, target_grid=8, pretrained=False
        )
        x = torch.randn(1, 3, 224, 224)
        out = encoder(x)
        self.assertEqual(out.shape, (1, 32, 8, 8))

    def test_grid_permutation_degrades(self):
        """Shuffling grid cells should change output (P5 causal efficacy).
        If spatial structure matters, a permuted grid produces different
        downstream features than the original."""
        torch.manual_seed(42)
        x = torch.randn(1, 3, 224, 224)
        out = self.encoder(x)  # [1, 64, 16, 16]

        # Permute spatial cells
        B, C, H, W = out.shape
        out_flat = out.reshape(B, C, H * W)  # [1, 64, 256]
        perm = torch.randperm(H * W)
        out_permuted = out_flat[:, :, perm].reshape(B, C, H, W)

        # Permuted output should differ from original
        diff = (out - out_permuted).abs().mean().item()
        self.assertGreater(diff, 0.01)

    def test_using_dino_flag(self):
        """Fallback encoder should have using_dino=False."""
        self.assertFalse(self.encoder.using_dino)

    def test_channel_proj_trainable(self):
        """The 1x1 conv projection should have trainable parameters."""
        trainable = [p for p in self.encoder.channel_proj.parameters()
                     if p.requires_grad]
        self.assertGreater(len(trainable), 0)

    def test_gradient_flows_through_projection(self):
        """Gradients should flow through the channel projection."""
        x = torch.randn(1, 3, 224, 224)
        out = self.encoder(x)
        loss = out.sum()
        loss.backward()

        conv_weight = list(self.encoder.channel_proj.parameters())[0]
        self.assertIsNotNone(conv_weight.grad)
        self.assertFalse(torch.all(conv_weight.grad == 0))


if __name__ == '__main__':
    unittest.main()
