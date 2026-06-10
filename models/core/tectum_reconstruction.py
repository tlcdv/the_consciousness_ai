"""
Reconstruction objective for the tectum (active-inference stage-1 likelihood term).

The 2026-06-09 perception-decodability probe localized the agent's competence
bottleneck to the obs_map -> tectum_content collapse: a linear decoder reads
stimulus identity (shape/color/size/count) off the spatial obs_map perfectly
(1.000) but only at chance off the 256-D tectum_content. The RSSM + capsule
compression discards stimulus identity because nothing in the tectum's training
objective (reward-prediction MSE + TDANN topographic loss) asks it to preserve
that identity. Training 100 episodes did not recover it.

This adds the missing pressure: reconstruct the current (downsampled) frame from
the current tectum_content. Minimizing reconstruction error forces the 256-D
bottleneck to retain enough of the frame to rebuild it, which requires keeping
shape/color/size in the latent. This is the likelihood term of a variational
free-energy / autoencoding objective (the first stage of the active-inference
unification in docs/active_inference_unification.md), the principled fix the probe
pointed to, not a control-specific patch.

Difference from ControlRepresentationHead (control_representation.py): that head
predicts the NEXT observation conditioned on the action (controllable dynamics);
this head reconstructs the CURRENT observation unconditioned (stimulus identity).
The probe measures current-stimulus decodability, so current-frame reconstruction
is the direct pressure for the measured gap.

Default-off (--enable-recon); the baseline is bit-identical when off. The decisive
test is re-running the perception-decodability probe on a --save-tectum checkpoint:
if tectum_content decode rises above chance toward obs_map's levels, the collapse
is repaired.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.control_representation import obs_features


class TectumReconstructionHead(nn.Module):
    """Reconstruct a downsampled RGB frame from the tectum content vector.

    The decoder is intentionally a plain MLP. Its job is not to be a strong
    decoder; it is to put reconstruction pressure on the upstream tectum_content
    so the 256-D bottleneck must preserve the stimulus. A weak decoder makes that
    pressure stronger (the content has to carry the information itself).
    """

    def __init__(self, content_dim: int, grid: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.content_dim = content_dim
        self.grid = grid
        self.target_dim = 3 * grid * grid
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.target_dim),
            nn.Sigmoid(),  # frame features are in [0, 1]
        )

    def reconstruct(self, content: torch.Tensor) -> torch.Tensor:
        """Reconstruct the downsampled frame features from tectum content."""
        if content.dim() == 1:
            content = content.unsqueeze(0)
        return self.net(content)

    def loss(self, content_t: torch.Tensor, frame: torch.Tensor,
             foreground: bool = False) -> torch.Tensor:
        """Reconstruction error between the prediction from content_t and the
        downsampled current frame (stop-grad target).

        The gradient flows into content_t (and thus into the tectum: encoder,
        RSSM, capsules), forcing it to retain the current stimulus. The target
        frame is detached so the loss trains the tectum representation, not the
        pixels.

        foreground=False: plain MSE over all target elements. On sparse stimuli
        (mostly-black frames) this is dominated by the trivial background, so the
        pressure to encode the small stimulus is weak (verified FAILED on DMTS).

        foreground=True: per-element weighted MSE, weights proportional to each
        element's deviation from the frame's mean. Background pixels (near the
        mean) get ~0 weight; the stimulus (bright or dark outlier) dominates. This
        concentrates the pressure on the stimulus identity the probe measures.
        """
        pred = self.reconstruct(content_t)
        target = obs_features(frame, grid=self.grid).detach()
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if not foreground:
            return F.mse_loss(pred, target)
        w = (target - target.mean(dim=1, keepdim=True)).abs()
        w = w + 1e-6  # floor so a flat frame degrades to ~uniform weighting
        sq = (pred - target) ** 2
        return (w * sq).sum(dim=1).mean() / (w.sum(dim=1).mean() + 1e-8)
