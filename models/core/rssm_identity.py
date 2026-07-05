"""Supervised identity gate on the RSSM latent (Path B stage 1, diagnostic).

Every objective tried at the collapse locus belongs to the variance-weighted error
family (reward MSE, frame reconstruction, obs_map reconstruction, the value-equivalent
reward/continue/KL objective), and each FAILED to make the RSSM latent carry stimulus
identity (docs/results/collapse_locus_*.md, wm_predict_stage1_2026_06_24.md,
perception_collapse_synthesis_2026_06_21.md). The confirmed mechanism: identity is a
low-variance direction of obs_map, and variance-weighted objectives drop it.

This head asks the remaining decisive question about the discrete latent itself. With
the maximum possible identity pressure, a supervised cross-entropy on the DMTS sample's
shape and color applied to the pre-capsule latent cat([h_t, z_flat]) during the sample
phase, does identity enter the latent at all?

- KILL: the CE trains down at the head but the offline collapse-locus probe still
  decodes z_state at chance. The gumbel-softmax categorical latent (or its optimization
  path) cannot carry identity even under direct supervision, so no label-free objective
  (InfoNCE and relatives) will do better. Escalate to the continuous-latent stage.
- PASS: z_state decodes identity well above chance on the probe. The latent is
  trainable and the wall was the objective family; a label-free contrastive objective
  becomes the mechanism candidate to build next.

Like the DMTS match head, this is a PROBE with privileged labels (the env's
sample_shape / sample_color), a ceiling test, not a shipped mechanism. Default off
(--enable-latent-id); the baseline is bit-identical when off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSMIdentityHead(nn.Module):
    """Classify the sample's shape and color from the spatial RSSM latent.

    Input is the pre-capsule state tensor [B, rssm_channels, grid, grid] (h_t plus the
    flattened categorical z_t), the same locus RSSMReconstructionHead used. A 1x1 conv
    reduces channels, a small MLP trunk feeds two linear class heads. The head is
    intentionally modest for the same reason the recon head was: its job is to put
    identity pressure on the upstream RSSM latent, not to be a strong classifier.
    """

    def __init__(self, rssm_channels: int, grid: int = 16,
                 reduce_channels: int = 32, hidden_dim: int = 256,
                 num_shapes: int = 6, num_colors: int = 6):
        super().__init__()
        self.reduce = nn.Conv2d(rssm_channels, reduce_channels, kernel_size=1)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduce_channels * grid * grid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.shape_head = nn.Linear(hidden_dim, num_shapes)
        self.color_head = nn.Linear(hidden_dim, num_colors)

    def forward(self, state_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (shape_logits [B, num_shapes], color_logits [B, num_colors])."""
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        h = self.trunk(self.reduce(state_tensor))
        return self.shape_head(h), self.color_head(h)

    def loss(self, state_tensor: torch.Tensor, shape_idx: int,
             color_idx: int) -> tuple[torch.Tensor, float]:
        """Cross-entropy on shape plus color for a single labeled step.

        Gradient flows into state_tensor (and thus into the RSSM: posterior/prior
        nets, the ConvGRU, and back through obs_map into the encoder), directly
        pressuring the latent to separate stimulus identities. Returns
        (loss, accuracy) where accuracy is the mean of the shape and color argmax
        hits for this step (an in-training signal to distinguish "head never
        trained" from "latent cannot carry identity").
        """
        shape_logits, color_logits = self.forward(state_tensor)
        device = shape_logits.device
        s = torch.tensor([shape_idx], dtype=torch.long, device=device)
        c = torch.tensor([color_idx], dtype=torch.long, device=device)
        loss = F.cross_entropy(shape_logits, s) + F.cross_entropy(color_logits, c)
        acc = 0.5 * (float(int(shape_logits.argmax(dim=1).item()) == shape_idx)
                     + float(int(color_logits.argmax(dim=1).item()) == color_idx))
        return loss, acc
