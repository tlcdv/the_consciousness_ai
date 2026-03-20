"""
Semantic Pathway: Qwen2-VL cortical route into the Global Workspace.

Projects Qwen2-VL ViT embeddings (1536-D) to workspace_dim and generates
a salience bid based on embedding magnitude. Supports top-down reentrant
feedback via receive_broadcast(), computing prediction error between the
current broadcast and the pathway's last submitted content.

When Qwen2-VL is unavailable (no weights / stub mode), the bid degrades
to zero so the pathway never wins competition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPathway(nn.Module):
    """
    Cortical semantic pathway projecting Qwen2-VL features into workspace space.

    forward(embedding) -> (workspace_content, bid)
    receive_broadcast(broadcast, current_bid) -> updated_bid
    """

    def __init__(self, input_dim: int = 1536, workspace_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.workspace_dim = workspace_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, workspace_dim),
            nn.GELU(),
            nn.LayerNorm(workspace_dim),
        )

        # Cache last projected content for prediction error in reentrant loop
        self._last_content: torch.Tensor | None = None

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Project a Qwen2-VL embedding into workspace content with a salience bid.

        Args:
            embedding: [input_dim] or [B, input_dim] tensor from Qwen2-VL ViT.

        Returns:
            (workspace_content [B, workspace_dim], bid scalar clamped to [0, 1])
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        content = self.projection(embedding)
        self._last_content = content.detach()

        # Bid = clamped L2 norm of the raw embedding (zero embedding -> zero bid)
        bid = float(torch.clamp(embedding.norm(dim=-1).mean() / (self.input_dim ** 0.5), 0.0, 1.0))

        return content, bid

    def receive_broadcast(self, broadcast: object, current_bid: float) -> float:
        """
        Top-down reentrant feedback from the workspace broadcast.

        Computes prediction error between broadcast and last content.
        If the broadcast is close to what this pathway submitted, the bid
        increases (confirmation). If far, the bid decreases (surprise).

        Args:
            broadcast: Workspace broadcast content (tensor or dict).
            current_bid: The pathway's current bid value.

        Returns:
            Updated bid value.
        """
        if self._last_content is None:
            return current_bid

        if isinstance(broadcast, torch.Tensor):
            # Reshape broadcast to match if needed
            flat_broadcast = broadcast.flatten()
            flat_content = self._last_content.flatten()

            min_len = min(len(flat_broadcast), len(flat_content))
            if min_len == 0:
                return current_bid

            pe = F.mse_loss(flat_broadcast[:min_len], flat_content[:min_len]).item()
            # Low PE -> boost bid (confirmed), high PE -> reduce bid
            adjustment = 0.1 * (1.0 - min(pe, 1.0))
            return max(0.0, min(1.0, current_bid + adjustment))

        # Non-tensor broadcast: no meaningful PE computation
        return current_bid
