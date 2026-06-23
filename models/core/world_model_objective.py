"""Value-equivalent world-model objective on the RSSM latent (Stage 1 of the model-based path).

Makes the RSSM a value-equivalent (MuZero / Dreamer-without-decoder) world model:
predict reward + continue from the RSSM latent, and train the prior/posterior KL as an
actual loss. There is deliberately NO observation decoder, which structurally avoids the
reconstruction-discards-low-variance-identity failure of the prior reconstruction heads
(collapse_locus_wmobs_2026_06_21.md): there is no reconstruction at all.

The working-memory pressure comes from predicting the DELAYED reward/continue at the
choice phase with BPTT through the delay (wired in the training loop), not from
reconstructing the trivially-blank delay frames. The latent it scores is the pre-capsule
RSSM state cat([h_t, z_flat]) (cached as SensoryTectum._last_state_tensor).

Default-off (--enable-wm-predict); the baseline is bit-identical when off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _head(latent_channels: int, grid: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """A small conv-reducer + MLP head on the spatial RSSM latent."""
    return nn.Sequential(
        nn.Conv2d(latent_channels, 32, kernel_size=1),
        nn.GELU(),
        nn.Flatten(),
        nn.Linear(32 * grid * grid, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class WorldModelObjective(nn.Module):
    """Reward + continue prediction heads on the RSSM latent, plus the balanced KL loss.

    No decoder. The heads are intentionally modest: their job is to put predictive
    pressure on the upstream RSSM latent so it carries what is needed to predict the
    delayed outcome, which (via BPTT through the delay) forces the recurrent state to
    retain the task-relevant working memory.
    """

    def __init__(self, latent_channels: int, grid: int = 16, hidden_dim: int = 256,
                 action_dim: int = 0):
        super().__init__()
        self.grid = grid
        self.action_dim = action_dim
        # Reward is predicted from the latent AND the action (value-equivalent / MuZero:
        # the reward depends on the action taken, e.g. which DMTS choice). A latent-only
        # reward head is ill-posed for an action-dependent reward and gives weak, ambiguous
        # gradient pressure on the latent. The continue head stays latent-only (episode end
        # is not action-dependent here).
        self.reward_reduce = nn.Conv2d(latent_channels, 32, kernel_size=1)
        self.reward_mlp = nn.Sequential(
            nn.Linear(32 * grid * grid + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continue_head = _head(latent_channels, grid, hidden_dim, 1)

    def predict_reward(self, latent: torch.Tensor,
                       action: torch.Tensor | None = None) -> torch.Tensor:
        """latent [B, C, H, W] (+ action [B, action_dim]) -> predicted reward [B, 1]."""
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        reduced = self.reward_reduce(latent).flatten(1)
        if self.action_dim > 0:
            if action is None:
                action = torch.zeros(reduced.shape[0], self.action_dim,
                                     device=reduced.device)
            elif action.dim() == 1:
                action = action.unsqueeze(0)
            reduced = torch.cat([reduced, action], dim=1)
        return self.reward_mlp(reduced)

    def predict_continue_logit(self, latent: torch.Tensor) -> torch.Tensor:
        """latent -> logit for P(not done) [B, 1]."""
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        return self.continue_head(latent)

    def reward_loss(self, latent: torch.Tensor, action: torch.Tensor | None,
                    target_reward: torch.Tensor) -> torch.Tensor:
        """MSE between predicted reward (from latent + action) and the observed reward
        (stop-grad target). The gradient flows into the latent, so the RSSM must carry
        whatever predicts the action-dependent reward; at the choice phase, with BPTT
        through the delay, that requires holding the sample."""
        pred = self.predict_reward(latent, action)
        tgt = target_reward.detach().reshape(pred.shape)
        return F.mse_loss(pred, tgt)

    def continue_loss(self, latent: torch.Tensor, not_done: torch.Tensor) -> torch.Tensor:
        """BCE between predicted continue logit and the observed not-done flag (stop-grad)."""
        logit = self.predict_continue_logit(latent)
        tgt = not_done.detach().reshape(logit.shape).float()
        return F.binary_cross_entropy_with_logits(logit, tgt)

    @staticmethod
    def kl_loss(prior_logits: torch.Tensor, post_logits: torch.Tensor,
                beta: float = 1.0, free_bits: float = 1.0,
                balance: float = 0.8) -> torch.Tensor:
        """DreamerV3 balanced categorical KL with free bits.

        logits shape [B, categories, classes, H, W]; the categorical is over the class
        axis (dim=2). Balanced: a `balance`-weighted dynamics term (train the prior toward
        a stop-grad posterior) plus a (1-balance) representation term (train the posterior
        toward a stop-grad prior). Free bits clamp the per-category KL so the model is not
        penalised below `free_bits` nats, which stabilises training and prevents posterior
        collapse.
        """
        def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
            # KL(softmax(p) || softmax(q)) over the class axis -> [B, categories, H, W]
            p = F.softmax(p_logits, dim=2)
            return (p * (F.log_softmax(p_logits, dim=2)
                         - F.log_softmax(q_logits, dim=2))).sum(dim=2)

        kl_dyn = _kl(post_logits.detach(), prior_logits)   # move prior to posterior
        kl_rep = _kl(post_logits, prior_logits.detach())   # move posterior to prior
        kl_dyn = torch.clamp(kl_dyn, min=free_bits).mean()
        kl_rep = torch.clamp(kl_rep, min=free_bits).mean()
        return beta * (balance * kl_dyn + (1.0 - balance) * kl_rep)
