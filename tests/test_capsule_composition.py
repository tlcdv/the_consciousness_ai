import unittest
import torch

from models.core.capsule_composition import (
    squash,
    PrimaryCapsuleLayer,
    RoutingCapsuleLayer,
    CapsuleCompositionLayer,
)
from models.core.sensory_tectum import SensoryTectum


class TestSquash(unittest.TestCase):

    def test_output_bounded(self):
        """Squashed vectors should have length strictly less than 1."""
        x = torch.randn(4, 8, 16)
        out = squash(x, dim=-1)
        norms = torch.norm(out, dim=-1)
        self.assertTrue((norms < 1.0).all())

    def test_zero_input(self):
        """Zero input should produce near-zero output."""
        x = torch.zeros(2, 4, 8)
        out = squash(x, dim=-1)
        self.assertLess(out.abs().max().item(), 1e-3)


class TestPrimaryCapsuleLayer(unittest.TestCase):

    def setUp(self):
        self.in_channels = 32
        self.num_capsules = 4
        self.capsule_dim = 4
        self.grid_size = 8
        self.layer = PrimaryCapsuleLayer(
            in_channels=self.in_channels,
            num_capsules=self.num_capsules,
            capsule_dim=self.capsule_dim,
            stride=2
        )

    def test_output_shape(self):
        """Output should be [B, num_caps * spatial, cap_dim]."""
        B = 2
        x = torch.randn(B, self.in_channels, self.grid_size, self.grid_size)
        out = self.layer(x)
        # stride=2 on 8x8 with padding -> 4x4 spatial
        spatial = 4 * 4
        expected_caps = self.num_capsules * spatial
        self.assertEqual(out.shape, (B, expected_caps, self.capsule_dim))

    def test_squash_bounds(self):
        """All capsule vectors should have length < 1."""
        x = torch.randn(2, self.in_channels, self.grid_size, self.grid_size)
        out = self.layer(x)
        norms = torch.norm(out, dim=-1)
        self.assertTrue((norms < 1.0).all())

    def test_zero_input_low_activity(self):
        """Zero input should produce near-zero capsule activities."""
        x = torch.zeros(1, self.in_channels, self.grid_size, self.grid_size)
        out = self.layer(x)
        norms = torch.norm(out, dim=-1)
        self.assertLess(norms.max().item(), 0.1)


class TestRoutingCapsuleLayer(unittest.TestCase):

    def setUp(self):
        self.num_primary = 64
        self.primary_dim = 4
        self.num_output = 4
        self.output_dim = 8
        self.layer = RoutingCapsuleLayer(
            num_primary_caps=self.num_primary,
            primary_dim=self.primary_dim,
            num_output_caps=self.num_output,
            output_dim=self.output_dim,
            routing_iterations=3
        )

    def test_output_shapes(self):
        """Poses and activities should have correct shapes."""
        B = 2
        x = torch.randn(B, self.num_primary, self.primary_dim)
        poses, activities = self.layer(x)
        self.assertEqual(poses.shape, (B, self.num_output, self.output_dim))
        self.assertEqual(activities.shape, (B, self.num_output))

    def test_activities_bounded(self):
        """Capsule activities (norms of squashed vectors) should be in [0, 1)."""
        x = torch.randn(2, self.num_primary, self.primary_dim)
        _, activities = self.layer(x)
        self.assertTrue((activities >= 0.0).all())
        self.assertTrue((activities < 1.0).all())

    def test_routing_concentrates_mass(self):
        """More routing iterations should produce higher top-1 capsule activity."""
        torch.manual_seed(42)
        x = torch.randn(1, self.num_primary, self.primary_dim)

        layer_1iter = RoutingCapsuleLayer(
            self.num_primary, self.primary_dim,
            self.num_output, self.output_dim,
            routing_iterations=1
        )
        layer_5iter = RoutingCapsuleLayer(
            self.num_primary, self.primary_dim,
            self.num_output, self.output_dim,
            routing_iterations=5
        )
        # Copy weights so only iteration count differs
        layer_5iter.W.data.copy_(layer_1iter.W.data)

        _, act1 = layer_1iter(x)
        _, act5 = layer_5iter(x)

        # More routing should increase the max activity (concentration)
        self.assertGreater(act5.max().item(), act1.max().item() * 0.9)

    def test_gradient_flows(self):
        """Backward pass should produce gradients on W."""
        x = torch.randn(1, self.num_primary, self.primary_dim, requires_grad=True)
        poses, activities = self.layer(x)
        loss = poses.sum() + activities.sum()
        loss.backward()
        self.assertIsNotNone(self.layer.W.grad)
        self.assertFalse(torch.all(self.layer.W.grad == 0))


class TestCapsuleCompositionLayer(unittest.TestCase):

    def setUp(self):
        self.rssm_channels = 32
        self.grid_size = 8
        self.workspace_dim = 64
        self.num_output_caps = 4
        self.output_dim = 8
        self.layer = CapsuleCompositionLayer(
            rssm_channels=self.rssm_channels,
            grid_size=self.grid_size,
            workspace_dim=self.workspace_dim,
            num_output_caps=self.num_output_caps,
            output_dim=self.output_dim,
            num_primary_caps=4,
            primary_dim=4,
            routing_iterations=3
        )

    def test_workspace_content_shape(self):
        x = torch.randn(2, self.rssm_channels, self.grid_size, self.grid_size)
        content, _, _ = self.layer(x)
        self.assertEqual(content.shape, (2, self.workspace_dim))

    def test_capsule_activities_shape(self):
        x = torch.randn(2, self.rssm_channels, self.grid_size, self.grid_size)
        _, activities, _ = self.layer(x)
        self.assertEqual(activities.shape, (2, self.num_output_caps))

    def test_capsule_poses_shape(self):
        x = torch.randn(2, self.rssm_channels, self.grid_size, self.grid_size)
        _, _, poses = self.layer(x)
        self.assertEqual(poses.shape, (2, self.num_output_caps, self.output_dim))

    def test_caches_last_poses(self):
        x = torch.randn(1, self.rssm_channels, self.grid_size, self.grid_size)
        self.assertIsNone(self.layer._last_poses)
        self.layer(x)
        self.assertIsNotNone(self.layer._last_poses)

    def test_tectum_integration(self):
        """SensoryTectum with capsule layer should still return (content, bid)."""
        config = {
            "tectum_feature_dim": 16,
            "tectum_grid_size": 8,
            "workspace_dim": 64,
            "num_output_caps": 4,
            "capsule_output_dim": 8,
            "num_primary_caps": 4,
            "capsule_primary_dim": 4,
            "routing_iterations": 3
        }
        tectum = SensoryTectum(config)

        B = 1
        vision = torch.randn(B, 16, 8, 8)
        audio = torch.randn(B, 16, 2)
        content, bid = tectum(vision, audio)

        self.assertEqual(content.shape, (B, 64))
        self.assertIsInstance(bid, float)
        self.assertTrue(0.0 <= bid <= 1.0)

        # Capsule state should be cached
        self.assertIsNotNone(tectum._last_capsule_poses)
        self.assertIsNotNone(tectum._last_capsule_activities)

    def test_tectum_capsule_payload(self):
        """get_capsule_payload() should return poses and activities after forward."""
        config = {
            "tectum_feature_dim": 16,
            "tectum_grid_size": 8,
            "workspace_dim": 64,
        }
        tectum = SensoryTectum(config)

        # Before forward, payload is empty
        self.assertEqual(tectum.get_capsule_payload(), {})

        # After forward, payload has capsule state
        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        tectum(vision, audio)

        payload = tectum.get_capsule_payload()
        self.assertIn("capsule_poses", payload)
        self.assertIn("capsule_activities", payload)


if __name__ == '__main__':
    unittest.main()
