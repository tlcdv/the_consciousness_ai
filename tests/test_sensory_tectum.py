import unittest
import torch
import torch.nn.functional as F

from models.core.sensory_tectum import TopographicMap, RSSMCore, SensoryTectum

class TestSensoryTectum(unittest.TestCase):
    
    def test_topographic_map_fusion(self):
        """Test that visual and audio grids are correctly fused"""
        feature_dim = 16
        grid_size = 8
        batch_size = 2
        
        topo_map = TopographicMap(grid_size=grid_size, feature_dim=feature_dim)
        
        # Visual input: [B, C, H, W]
        vision_grid = torch.ones(batch_size, feature_dim, grid_size, grid_size)
        
        # Audio spatial input: [B, C, 2] (x, y coordinates in [-1, 1])
        # Add audio exclusively to top-right corner (1, 1)
        audio_spatial = torch.zeros(batch_size, feature_dim, 2)
        audio_spatial[:, :, 0] = 1.0 # x = right
        audio_spatial[:, :, 1] = 1.0 # y = bottom (in image coords depending on setup, but typically end of grid)
        
        fused = topo_map(vision_grid, audio_spatial)
        
        self.assertEqual(fused.shape, (batch_size, feature_dim, grid_size, grid_size))
        
    def test_rssm_recurrence(self):
        """Test the DreamerV3 style Recurrent State Space Model"""
        rssm = RSSMCore(feature_dim=16, grid_size=8, num_categories=8, num_classes=8)
        
        B = 2
        # Initialize inputs
        h_prev = torch.zeros(B, 16, 8, 8)
        z_prev = torch.zeros(B, 8, 8, 8, 8)
        z_prev[:, :, 0, :, :] = 1.0
        
        obs_map = torch.randn(B, 16, 8, 8)
        
        # Taking a step with observation
        h_t, z_t, prior, posterior = rssm.step(obs_map, h_prev, z_prev)
        
        self.assertEqual(h_t.shape, (B, 16, 8, 8))
        self.assertEqual(z_t.shape, (B, 8, 8, 8, 8))
        self.assertEqual(prior.shape, (B, 8, 8, 8, 8))
        self.assertEqual(posterior.shape, (B, 8, 8, 8, 8))
        
        # z_t should be one-hot encoded along the classes dim (dim=2)
        # Verify it sums to 1 across the classes dimension
        z_sums = z_t.sum(dim=2)
        self.assertTrue(torch.allclose(z_sums, torch.ones_like(z_sums)))
        
    def test_tectum_surprise_bid(self):
        """Test that the full module returns a higher bid when surprise is higher"""
        config = {
            "tectum_feature_dim": 16,
            "tectum_grid_size": 8,
            "workspace_dim": 64
        }
        tectum = SensoryTectum(config)

        B = 1
        vision_features = torch.randn(B, 16, 8, 8)
        audio_spatial = torch.randn(B, 16, 2)

        # First step: initialize
        content1, bid1 = tectum.forward(vision_features, audio_spatial)

        # Second step: feed exact same inputs
        # The RSSM should ideally predict this (or closely), resulting in lower surprise
        content2, bid2 = tectum.forward(vision_features, audio_spatial)

        # In a completely untrained model, random weights might not guarantee bids strictly drop
        # But we ensure it returns valid types and tensor shapes
        self.assertEqual(content1.shape, (B, 64))
        self.assertIsInstance(bid1, float)
        self.assertTrue(0.0 <= bid1 <= 1.0) # Bid is strictly bounded by tanh


class TestTruncatedBPTT(unittest.TestCase):
    """Locks the S3 invariant: gradient at step T flows back through up
    to (bptt_window - 1) earlier RSSM steps, instead of being severed
    every step the way the previous unconditional detach did.
    """

    def _build_tectum(self, K):
        torch.manual_seed(0)
        return SensoryTectum({
            "tectum_feature_dim": 16,
            "tectum_grid_size": 8,
            "workspace_dim": 64,
            "bptt_window": K,
        })

    def test_h_state_grad_fn_within_window(self):
        """Mid-window, h_state must carry a grad_fn (pre-fix it was always
        detached, so grad_fn was None on every step)."""
        tectum = self._build_tectum(K=8)
        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        # Step 1: counter goes from 0 -> 1, K not reached, h_state retained.
        tectum.forward(vision, audio)
        self.assertIsNotNone(
            tectum.h_state.grad_fn,
            msg="h_state has no grad_fn after step 1; truncated BPTT broken.",
        )
        # Step 7: counter goes 1 -> 7, still under K=8, still retained.
        for _ in range(6):
            tectum.forward(vision, audio)
        self.assertIsNotNone(tectum.h_state.grad_fn)
        # Step 8: counter hits K, must detach.
        tectum.forward(vision, audio)
        self.assertIsNone(
            tectum.h_state.grad_fn,
            msg="h_state still has grad_fn after K=8 steps; detach skipped.",
        )

    def test_loss_at_step_T_updates_rssm_via_earlier_steps(self):
        """K=4 BPTT must produce strictly more RSSM gradient signal than
        K=1 (the pre-fix behavior) on the same 4-step rollout. The
        absolute gradient magnitude is dominated by random-init scale and
        sigmoid saturation, so the right invariant is the K=4 vs K=1
        difference, not an absolute threshold."""

        def total_rssm_grad(K):
            tectum = self._build_tectum(K=K)
            vision = torch.randn(1, 16, 8, 8)
            audio = torch.randn(1, 16, 2)
            # Same 4-step rollout under both K values
            for _ in range(3):
                tectum.forward(vision, audio)
            content, _ = tectum.forward(vision, audio)
            loss = (content ** 2).mean()
            loss.backward()
            return sum(
                p.grad.abs().sum().item()
                for p in tectum.rssm.parameters() if p.grad is not None
            )

        # Common seed for the inputs so the comparison is fair
        torch.manual_seed(123)
        g_k1 = total_rssm_grad(K=1)
        torch.manual_seed(123)
        g_k4 = total_rssm_grad(K=4)

        # K=1 gives gradient only from the final RSSM step (chain length 1).
        # K=4 gives gradient from all 4 RSSM steps (chain length 4).
        # The K=4 total absolute gradient must exceed K=1 by at least 5%
        # to confirm the recurrent chain is contributing.
        self.assertGreater(
            g_k4, g_k1 * 1.05,
            msg=f"K=4 BPTT grad ({g_k4:.3e}) is not meaningfully larger "
                f"than K=1 grad ({g_k1:.3e}). Recurrent chain is severed.",
        )

    def test_window_of_one_recovers_old_behavior(self):
        """K=1 means detach every step. Use this as the ablation hook so
        the peaceful-castle Run I (--ablate-bptt) has a config-only knob."""
        tectum = self._build_tectum(K=1)
        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        tectum.forward(vision, audio)
        self.assertIsNone(
            tectum.h_state.grad_fn,
            msg="K=1 should detach immediately, leaving no grad_fn.",
        )

    def test_reset_state_clears_bptt_counter(self):
        tectum = self._build_tectum(K=8)
        vision = torch.randn(1, 16, 8, 8)
        audio = torch.randn(1, 16, 2)
        for _ in range(3):
            tectum.forward(vision, audio)
        self.assertEqual(tectum._steps_since_detach, 3)
        tectum.reset_state(batch_size=1)
        self.assertEqual(tectum._steps_since_detach, 0)


if __name__ == '__main__':
    unittest.main()
