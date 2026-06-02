"""
Control-relevant representation objective for the tectum (P5 fix).

The 2026-06-01 competence diagnosis showed the broadcast representation, not the
policy, is the bottleneck: the tectum is trained only by reward-prediction MSE +
TDANN topographic loss + gate-diversity loss, never by the control objective, so
the 256-D broadcast is never shaped to be a controllable state. A known-working RL
learner on the broadcast plateaus ~10x below the same learner family on raw pixels.

This adds an action-conditioned forward model: given the current tectum content and
the action taken, predict the NEXT observation (downsampled). Trained on the same
tectum_optimizer path as the reward predictor, the gradient flows into the tectum
through the current content, forcing the representation to encode the consequences
of actions (controllable dynamics), which a state representation needs for control.

Why predict the next OBSERVATION rather than the next tectum content:
  - It is computable in the same step (the next frame is available right after
    env.step, before the auxiliary optimizer update), so no cross-step BPTT graph
    is retained and the in-place tectum-optimizer obstacle (train_rlhf.py:725-734)
    is untouched.
  - The target is grounded raw pixels, not a moving learned latent, so there is no
    trivial-constant collapse (the failure mode a same-step or non-stop-grad latent
    target would invite).
  - The gradient reaches the tectum through the current content as the model INPUT,
    which is what actually shapes perception (a detached-content target would train
    only the head).

Forward-model idea: Schwarzer et al. 2021 (SPR, arXiv:2007.05929) and the DreamerV3
world model already used elsewhere in this project. Default-off
(--enable-control-repr); the baseline is bit-identical when off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def obs_features(frame: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """Downsample an RGB frame to a fixed-length feature vector.

    Accepts [B, 3, H, W] or [3, H, W]; returns [B, 3*grid*grid] in [0, 1]
    (adaptive average pool to grid x grid, then flatten). Used as the grounded,
    stop-grad prediction target for the forward model.
    """
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(frame, (grid, grid))
    return pooled.flatten(1)


class ControlRepresentationHead(nn.Module):
    def __init__(self, content_dim: int, action_dim: int, target_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.content_dim = content_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.net = nn.Sequential(
            nn.Linear(content_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def predict(self, content: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Action-conditioned forecast of the next-observation features."""
        if content.dim() == 1:
            content = content.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.net(torch.cat([content, action], dim=-1))

    def loss(self, content_t: torch.Tensor, action_t: torch.Tensor,
             next_obs_features: torch.Tensor) -> torch.Tensor:
        """MSE between the action-conditioned forecast from (content_t, action_t)
        and the observed next-observation features (stop-grad target). The gradient
        flows into `content_t` (and thus the tectum); the target is detached."""
        pred = self.predict(content_t, action_t)
        target = next_obs_features.detach()
        if target.dim() == 1:
            target = target.unsqueeze(0)
        return F.mse_loss(pred, target)
