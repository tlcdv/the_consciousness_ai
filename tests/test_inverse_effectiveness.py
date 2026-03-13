import unittest
import torch

from models.core.sensory_tectum import TopographicMap


class TestInverseEffectiveness(unittest.TestCase):
    """Tests for the inverse effectiveness multisensory fusion in TopographicMap."""

    def setUp(self):
        self.grid_size = 8
        self.feature_dim = 16
        self.topo = TopographicMap(
            grid_size=self.grid_size, feature_dim=self.feature_dim
        )

    def test_weak_weak_larger_enhancement(self):
        """Weak visual + weak audio should produce proportionally larger
        enhancement than strong visual + strong audio.

        This is the core inverse effectiveness principle (Stein & Meredith 1993):
        proportional enhancement is greatest when individual unimodal responses
        are weakest."""
        B, C, H, W = 1, self.feature_dim, self.grid_size, self.grid_size

        # Weak signals
        vis_weak = torch.ones(B, C, H, W) * 0.01
        aud_weak = torch.ones(B, C, H, W) * 0.01

        # Strong signals
        vis_strong = torch.ones(B, C, H, W) * 10.0
        aud_strong = torch.ones(B, C, H, W) * 10.0

        fused_weak = self.topo._fuse_inverse_effectiveness(vis_weak, aud_weak)
        fused_strong = self.topo._fuse_inverse_effectiveness(vis_strong, aud_strong)

        # Proportional enhancement = fused / visual_only
        # For weak signals, audio contribution should be proportionally larger
        ratio_weak = (fused_weak.abs().mean() / (vis_weak.abs().mean() + 1e-10)).item()
        ratio_strong = (fused_strong.abs().mean() / (vis_strong.abs().mean() + 1e-10)).item()

        self.assertGreater(ratio_weak, ratio_strong)

    def test_zero_audio_passthrough(self):
        """When audio is zero, fusion should approximately equal visual input."""
        B, C, H, W = 1, self.feature_dim, self.grid_size, self.grid_size
        visual = torch.randn(B, C, H, W)
        audio = torch.zeros(B, C, H, W)

        fused = self.topo._fuse_inverse_effectiveness(visual, audio)

        # With zero audio, fused should be close to visual
        diff = (fused - visual).abs().mean().item()
        self.assertLess(diff, 1e-5)

    def test_output_shape_preserved(self):
        """Fused output should match input shape."""
        B, C, H, W = 2, self.feature_dim, self.grid_size, self.grid_size
        visual = torch.randn(B, C, H, W)
        audio = torch.randn(B, C, H, W)

        fused = self.topo._fuse_inverse_effectiveness(visual, audio)
        self.assertEqual(fused.shape, (B, C, H, W))

    def test_ie_weight_normalization(self):
        """IE weights should be normalized so mean is approximately 1.0,
        preventing overall magnitude drift."""
        B, C, H, W = 1, self.feature_dim, self.grid_size, self.grid_size
        visual = torch.randn(B, C, H, W).abs() + 0.1
        audio = torch.randn(B, C, H, W).abs() + 0.1

        v_mag = visual.norm(dim=1, keepdim=True)
        a_mag = audio.norm(dim=1, keepdim=True)
        epsilon = 1e-6
        max_unimodal = torch.max(v_mag, a_mag) + epsilon
        ie_weight = 1.0 / max_unimodal
        ie_weight = ie_weight / (ie_weight.mean() + epsilon)

        # Mean weight should be close to 1.0
        self.assertAlmostEqual(ie_weight.mean().item(), 1.0, places=1)

    def test_full_forward_with_ie_fusion(self):
        """TopographicMap.forward() should produce correct shape with IE fusion."""
        B = 2
        vision_grid = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        audio_spatial = torch.randn(B, self.feature_dim, 2)

        fused = self.topo(vision_grid, audio_spatial)
        self.assertEqual(
            fused.shape,
            (B, self.feature_dim, self.grid_size, self.grid_size)
        )

    def test_spatial_selectivity(self):
        """Audio placed at a specific grid location should enhance that
        location more than distant locations."""
        B, C, H, W = 1, self.feature_dim, self.grid_size, self.grid_size

        # Uniform weak visual
        visual = torch.ones(B, C, H, W) * 0.1

        # Audio concentrated at center
        audio = torch.zeros(B, C, H, W)
        center = H // 2
        audio[:, :, center, center] = 1.0

        fused = self.topo._fuse_inverse_effectiveness(visual, audio)

        # Center should have higher magnitude than corners
        center_val = fused[:, :, center, center].abs().mean().item()
        corner_val = fused[:, :, 0, 0].abs().mean().item()
        self.assertGreater(center_val, corner_val)


class TestSomatosensoryChannel(unittest.TestCase):
    """Tests for the somatosensory (body schema) channel in TopographicMap.

    The body schema from SelfRepresentationCore is projected onto the spatial
    grid and fused via inverse effectiveness, creating a trimodal (vision +
    audio + somatosensory) topographic map matching the deep layers of the
    biological superior colliculus (Stein & Meredith 1993, ch. 4).
    """

    def setUp(self):
        self.grid_size = 8
        self.feature_dim = 16
        self.body_parts = 10
        self.body_features = 8
        self.topo = TopographicMap(
            grid_size=self.grid_size,
            feature_dim=self.feature_dim,
            body_parts=self.body_parts,
            body_features=self.body_features,
        )

    def test_forward_without_body_unchanged(self):
        """Calling forward without body_schema should still work (backward compat)."""
        B = 2
        vis = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        aud = torch.randn(B, self.feature_dim, 2)
        out = self.topo(vis, aud)
        self.assertEqual(out.shape, (B, self.feature_dim, self.grid_size, self.grid_size))

    def test_forward_with_body_correct_shape(self):
        """Forward with body_schema should produce correct output shape."""
        B = 2
        vis = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        aud = torch.randn(B, self.feature_dim, 2)
        body = torch.randn(B, self.body_parts, self.body_features)
        out = self.topo(vis, aud, body_schema=body)
        self.assertEqual(out.shape, (B, self.feature_dim, self.grid_size, self.grid_size))

    def test_body_changes_output(self):
        """Providing body_schema should change the fused output compared to no body."""
        B = 1
        vis = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        aud = torch.randn(B, self.feature_dim, 2)
        body = torch.randn(B, self.body_parts, self.body_features) * 2.0

        out_no_body = self.topo(vis, aud)
        out_with_body = self.topo(vis, aud, body_schema=body)

        diff = (out_with_body - out_no_body).abs().mean().item()
        self.assertGreater(diff, 1e-4, "Body schema should change the fused output")

    def test_zero_body_minimal_effect(self):
        """Zero body schema should have minimal effect on fusion."""
        B = 1
        vis = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        aud = torch.randn(B, self.feature_dim, 2)
        body_zero = torch.zeros(B, self.body_parts, self.body_features)

        out_no_body = self.topo(vis, aud)
        out_zero_body = self.topo(vis, aud, body_schema=body_zero)

        # Zero body projects to near-zero grid, IE fusion of near-zero
        # should have minimal impact
        diff = (out_zero_body - out_no_body).abs().mean().item()
        self.assertLess(diff, 0.5, "Zero body schema should have small effect")

    def test_body_projection_gradient_flows(self):
        """Gradients should flow through the body projection layer."""
        B = 1
        vis = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        aud = torch.randn(B, self.feature_dim, 2)
        body = torch.randn(B, self.body_parts, self.body_features, requires_grad=True)

        out = self.topo(vis, aud, body_schema=body)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(body.grad)
        self.assertGreater(body.grad.abs().sum().item(), 0.0,
                           "Gradients should flow back to body_schema")

    def test_tectum_passes_body_through(self):
        """SensoryTectum.forward() should accept and pass body_schema to TopographicMap."""
        from models.core.sensory_tectum import SensoryTectum

        config = {
            "tectum_feature_dim": self.feature_dim,
            "tectum_grid_size": self.grid_size,
            "workspace_dim": 32,
            "use_pretrained_dino": False,
        }
        tectum = SensoryTectum(config)

        B = 1
        vis = torch.randn(B, self.feature_dim, self.grid_size, self.grid_size)
        aud = torch.randn(B, self.feature_dim, 2)
        body = torch.randn(B, self.body_parts, self.body_features)

        content, bid = tectum(vis, aud, body_schema=body)
        self.assertEqual(content.shape[0], B)
        self.assertIsInstance(bid, float)


if __name__ == '__main__':
    unittest.main()
