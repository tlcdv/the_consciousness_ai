"""
Supervised match head for DMTS (delayed match-to-sample).

Why this exists. The 2026-06-14 localization (rssm_working_memory_2026_06_12.md,
"Final localization") established that the correct DMTS match action is
supervised-decodable at 0.845 from the policy input [current obs_map ; held
sample obs_map] via a PCA+MLP probe, yet the Go/No-Go RL policy given the SAME
input does not learn it from reward (behavioral trials_correct ~1.19). Every
upstream layer (perception, working-memory availability, representation) is
ruled out; the wall is RL credit assignment.

This module puts that decoder INSIDE the agent and trains it on the env's own
`target_position` label, in two modes:

  acting  A standalone conv head whose argmax drives the action at the choice
          phase. Capability / pipeline test: does the LIVE in-loop pipeline (live
          obs_map + live working-memory latch + this head) support correct match
          behavior end to end? It uses the privileged target label, so it is NOT
          a claim of autonomous learning. It validates the pipeline and sets the
          behavioral ceiling (expected near the 0.845 offline decodability).

  aux     A linear head on the policy's own PFC features (the shared conv trunk
          the Go/No-Go policy uses). Its cross-entropy gradient flows into the
          shared PFC conv, so it tests whether a dense supervised signal shapes
          the representation enough for the RL policy to act correctly. The head
          does NOT drive the action in this mode; the RL policy still selects.

Input layout (obsmem-conv tap). The policy_state is
torch.cat([current_obs_map.flatten(), held_sample.flatten()]) =>
[B, 2*64*16*16] = [B, 32768]. Reshaped to [B, 128, 16, 16] this is the current
obs_map on channels 0..63 and the held sample on channels 64..127, so a conv can
compare the two spatial maps to find which choice position matches the sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchHead(nn.Module):
    """Standalone conv head over the stacked [current ; held] obs_map.

    Used by the `acting` mode: its argmax is the action at the choice phase.
    Owns its own parameters and is trained by its own optimizer, so it does not
    touch the policy networks.
    """

    def __init__(self, spatial_shape: tuple[int, int, int], num_actions: int,
                 hidden: int = 32):
        super().__init__()
        c, h, w = spatial_shape
        self.spatial_shape = spatial_shape
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Conv2d(c, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(16 * 4 * 4, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, C*H*W] (flattened policy_state) or [B, C, H, W]. -> logits."""
        if state.dim() == 2:
            state = state.view(state.shape[0], *self.spatial_shape)
        return self.net(state)

    @staticmethod
    def loss(logits: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the integer target position (class index)."""
        return F.cross_entropy(logits, target_position)

    def predict(self, state: torch.Tensor) -> int:
        """Argmax action (the matching choice position) for a single observation."""
        with torch.no_grad():
            return int(self.forward(state).argmax(dim=1)[0].item())


class AuxMatchHead(nn.Module):
    """Linear classifier over the policy's PFC features, for the `aux` mode.

    Applied on the PFC's `pfc_state` (the same conv trunk the Go/No-Go policy
    consumes), so backprop of its cross-entropy loss shapes the shared
    representation. Tiny on purpose: the question is whether the trunk carries a
    linearly-readable match signal once the dense gradient pushes it there, not
    whether a deep head can decode it.
    """

    def __init__(self, context_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Linear(context_dim, num_actions)

    def forward(self, pfc_state: torch.Tensor) -> torch.Tensor:
        return self.fc(pfc_state)

    @staticmethod
    def loss(logits: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target_position)
