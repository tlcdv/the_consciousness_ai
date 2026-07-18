"""Label-free contrastive identity objective on the RSSM latent (Path B stage 2).

The supervised ceiling test (--enable-latent-id, B0) showed the discrete gumbel-softmax
latent cannot carry stimulus identity even under direct supervision. The continuous latent
(B1) passed: with --rssm-latent-mode continuous, z_state decodes identity well above
chance (replicated 3 seeds, docs/results/b1_continuous_latent_2026_07.md).

This head replaces the privileged-label CE with a self-supervised InfoNCE objective that
uses the DMTS trial structure: the same stimulus identity appears in both the sample phase
and the choice phase (as the correct match). Pulling same-trial sample and choice latents
together, and pushing cross-trial pairs apart, puts identity pressure on the continuous
latent without env labels.

Input is the pre-capsule RSSM latent cat([h_t, z_flat]) from SensoryTectum._last_state_tensor,
the same locus as RSSMIdentityHead and RSSMReconstructionHead.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSMContrastiveHead(nn.Module):
    """InfoNCE contrastive head on the pre-capsule RSSM latent.

    Architecture:
      1. Channel reduction: 1x1 conv (rssm_channels -> reduce_channels)
      2. Global average pool -> flatten
      3. Projection MLP: hidden_dim -> proj_dim (LayerNorm + GELU)
      4. The projected vectors are used in the InfoNCE loss.

    The loss operates on aggregated trial representations: sample-phase latents
    (multiple steps) are averaged, choice-phase latents are averaged, and same-trial
    pairs are pulled together while cross-trial pairs are pushed apart.
    """

    def __init__(
        self,
        rssm_channels: int,
        grid: int = 16,
        reduce_channels: int = 32,
        hidden_dim: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.proj_dim = proj_dim

        self.reduce = nn.Conv2d(rssm_channels, reduce_channels, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduce_channels * grid * grid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Project a single pre-capsule RSSM latent to contrastive space.

        Args:
            state_tensor: [B, rssm_channels, grid, grid] from the RSSM latent.

        Returns:
            projected: [B, proj_dim] L2-normalized contrastive embedding.
        """
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        h = self.reduce(state_tensor)
        proj = self.projection(h)
        return F.normalize(proj, p=2, dim=1)

    def loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE loss.

        For each anchor-positive pair, the loss is:
          -log( exp(sim(a, p) / τ) / (exp(sim(a, p) / τ) + sum_i exp(sim(a, n_i) / τ)) )

        Args:
            anchor: [B, proj_dim] trial-aggregated sample-phase latents.
            positive: [B, proj_dim] same-trial choice-phase latents.
            negatives: [B, K, proj_dim] or [K, proj_dim] cross-trial negatives.
                If 2-D, broadcast to B identical sets of K negatives per anchor.

        Returns:
            Scalar loss (mean over batch).
        """
        B = anchor.shape[0]

        # Negatives shape: handle [B, K, D] or [K, D]
        if negatives.dim() == 2:
            negatives = negatives.unsqueeze(0).expand(B, -1, -1)

        # Similarity of anchor to positive: [B]
        pos_sim = (anchor * positive).sum(dim=1) / self.temperature

        # Similarity of anchor to each negative: [B, K]
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Numerator: exp(positive sim)
        numerator = torch.exp(pos_sim)  # [B]

        # Denominator: exp(positive) + sum(exp(negatives))
        denominator = numerator + torch.exp(neg_sim).sum(dim=1)  # [B]

        loss = -torch.log(numerator / denominator + 1e-8).mean()
        return loss
