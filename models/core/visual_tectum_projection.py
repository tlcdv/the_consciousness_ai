import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualTectumProjection(nn.Module):
    """
    Bridges Qwen2-VL ViT output to the SensoryTectum input format.

    Qwen2-VL produces spatial grids of shape [1536, H, W] (variable H/W
    depending on image resolution). The tectum expects [B, feature_dim, grid_size, grid_size].
    This module handles spatial interpolation and channel reduction.
    """

    def __init__(self, in_channels=1536, out_channels=64, target_grid=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_grid = target_grid

        self.channel_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.LayerNorm([out_channels, target_grid, target_grid]),
            nn.GELU()
        )

    def forward(self, qwen_grid):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Args:
            qwen_grid: [C, H, W] or [B, C, H, W] from Qwen2-VL get_visual_embeddings(return_spatial_grid=True)

        Returns:
            [B, out_channels, target_grid, target_grid]
        """
        if qwen_grid.dim() == 3:
            qwen_grid = qwen_grid.unsqueeze(0)

        # Spatial interpolation to fixed grid size
        if qwen_grid.shape[2] != self.target_grid or qwen_grid.shape[3] != self.target_grid:
            qwen_grid = F.interpolate(
                qwen_grid,
                size=(self.target_grid, self.target_grid),
                mode='bilinear',
                align_corners=False
            )

        return self.channel_proj(qwen_grid)
