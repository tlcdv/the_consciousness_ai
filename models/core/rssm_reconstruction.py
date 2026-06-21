"""World-model reconstruction objective on the RSSM latent (active-inference stage 1, R1).

The collapse-locus probe (docs/results/collapse_locus_2026_06_16.md, confirmed on a
trained tectum 2026-06-21 in collapse_locus_trained_2026_06_21.md) localized the loss of
task-relevant stimulus identity to the RSSM step: obs_map decodes shape/color at ~1.0,
but the RSSM latent (z_state) and everything downstream sit at chance, trained and
untrained. The cause: the RSSM is trained to predict reward (reward-predictor MSE +
TDANN), not to reconstruct observations, so it has no pressure to keep identity.

This head adds the missing generative pressure AT THE LOCUS: reconstruct the current
(downsampled) frame from the RSSM latent state_tensor = cat([h_t, z_flat]) (the exact
pre-capsule tensor the capsule layer consumes). Minimizing the reconstruction error
forces the RSSM latent to retain enough of the frame to rebuild it, which requires
keeping shape/color in h_t/z_t. This is the likelihood term of a variational
free-energy / world-model objective (active_inference_unification.md, roadmap Phase 6;
the DINO-WM-style "predict the content map from the latent", here with a raw-frame
target because training uses the conv-fallback encoder, not a frozen DINOv2 feature
space; see docs/generative_world_models_perception.md).

CRITICAL difference from TectumReconstructionHead (tectum_reconstruction.py): that head
reconstructs from tectum_content (256-D, AFTER the collapse) and FAILED
(tectum_reconstruction_2026_06_10.md, "architectural"), because the identity is already
gone before tectum_content. This head reconstructs from the RSSM latent (BEFORE the
capsule collapse), which is where the collapse-locus probe shows the loss happens. That
is the untested R1 the diagnosis points to.

Default-off (--enable-wm-recon); the baseline is bit-identical when off. The decisive
test is re-running the collapse-locus probe on a --save-tectum checkpoint: if z_state
(and h_state) decode identity above chance toward obs_map's level, the collapse is
repaired at its source.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.control_representation import obs_features


class RSSMReconstructionHead(nn.Module):
    """Reconstruct a downsampled RGB frame from the spatial RSSM latent.

    Input is the pre-capsule state tensor [B, rssm_channels, grid, grid] (h_t plus the
    flattened categorical z_t). A 1x1 conv reduces channels, then a small MLP maps the
    spatial latent to the flat downsampled frame. The decoder is intentionally modest:
    its job is to put reconstruction pressure on the upstream RSSM latent, not to be a
    strong decoder. A weak decoder makes that pressure stronger (the latent has to carry
    the information itself).
    """

    def __init__(self, rssm_channels: int, grid: int = 16,
                 reduce_channels: int = 32, hidden_dim: int = 256,
                 target_mode: str = "frame", feature_dim: int | None = None):
        super().__init__()
        self.grid = grid
        self.target_mode = target_mode
        if target_mode == "obs_map":
            # Reconstruct the dense, identity-rich obs_map feature map (decodes
            # identity at ~1.0), the DINO-WM-style "predict the content map" target
            # without needing a frozen DINOv2. A dense target resists the
            # average-collapse that a sparse pixel target invites. obs_map is a raw
            # post-GELU feature map (not in [0, 1]), so the output is linear.
            if feature_dim is None:
                raise ValueError("target_mode='obs_map' requires feature_dim")
            self.target_dim = feature_dim * grid * grid
            out_activation: nn.Module = nn.Identity()
        else:  # frame (default): downsampled RGB, in [0, 1]
            self.target_dim = 3 * grid * grid
            out_activation = nn.Sigmoid()
        self.reduce = nn.Conv2d(rssm_channels, reduce_channels, kernel_size=1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduce_channels * grid * grid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.target_dim),
            out_activation,
        )

    def reconstruct(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct the downsampled frame from the spatial RSSM latent."""
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        reduced = self.reduce(state_tensor)
        return self.net(reduced)

    def loss(self, state_tensor: torch.Tensor, target_source: torch.Tensor,
             foreground: bool = True) -> torch.Tensor:
        """Reconstruction error between the prediction from the RSSM latent and a
        stop-grad target derived from target_source.

        Gradient flows into state_tensor (and thus into the RSSM: posterior/prior nets,
        the ConvGRU, and back through obs_map into the encoder), forcing the latent to
        retain the current stimulus. The target is detached so the loss trains the RSSM
        representation, not the target.

        target_mode='frame' (default): target_source is the raw RGB frame; the target is
        the downsampled frame in [0, 1]. foreground=True (per-element weighted MSE,
        weights proportional to deviation from the mean) is the default, so the sparse
        stimulus dominates the trivial black background (naive MSE was verified to FAIL on
        sparse DMTS frames).

        target_mode='obs_map': target_source is the obs_map feature map
        [B, feature_dim, grid, grid]; the target is its flattened, detached values. This
        is a DENSE, identity-rich target (obs_map decodes identity at ~1.0), so plain MSE
        is used (no sparse-background problem) and foreground is ignored.
        """
        pred = self.reconstruct(state_tensor)
        if self.target_mode == "obs_map":
            tgt = target_source
            if tgt.dim() == 3:
                tgt = tgt.unsqueeze(0)
            target = tgt.reshape(tgt.shape[0], -1).detach()
            return F.mse_loss(pred, target)
        target = obs_features(target_source, grid=self.grid).detach()
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if not foreground:
            return F.mse_loss(pred, target)
        w = (target - target.mean(dim=1, keepdim=True)).abs()
        w = w + 1e-6  # floor so a flat frame degrades to ~uniform weighting
        sq = (pred - target) ** 2
        return (w * sq).sum(dim=1).mean() / (w.sum(dim=1).mean() + 1e-8)
