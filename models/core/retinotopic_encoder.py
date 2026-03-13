import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class RetinotopicConvStack(nn.Module):
    """
    Lightweight convolutional stack that preserves retinotopic structure
    by construction. Each strided conv halves spatial dimensions while
    maintaining neighborhood relationships.

    Used as fallback when DINOv2 weights are unavailable (testing, CI),
    and as the reference implementation for pure geometric isomorphism.

    Input:  [B, 3, 224, 224]
    Output: [B, 64, 16, 16]

    Receptive field at output: each grid cell covers a 29x29 pixel region
    centered on a 14x14 stride grid. Adjacent cells overlap, preserving
    smooth spatial gradients.
    """

    def __init__(self, out_channels=64):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 (close to 16, pad to 16)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, pixel_values):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Args:
            pixel_values: [B, 3, 224, 224] raw image tensor

        Returns:
            [B, out_channels, 14, 14] spatial feature map
        """
        return self.conv_stack(pixel_values)


class RetinotopicEncoder(nn.Module):
    """
    Wraps DINOv2-B/14 (frozen) to produce spatially faithful patch tokens.

    DINOv2 was trained with self-supervised masked image modeling at patch
    level. Each patch token at grid position (i, j) corresponds to the
    14x14 pixel region at (i*14, j*14) in the input image. This preserves
    isomorphic spatial mapping: adjacent patches map to adjacent image
    regions, and distances in the grid are proportional to distances in
    the image.

    For 224x224 input, DINOv2-B/14 produces 16x16 = 256 patch tokens at
    768 dimensions each. A 1x1 Conv2d reduces channels from 768 to 64
    (tectum feature_dim), matching the SensoryTectum input format.

    Falls back to RetinotopicConvStack when DINOv2 weights are unavailable.
    The conv stack preserves retinotopy by construction (strided convolutions
    maintain spatial correspondence), though with weaker features.
    """

    def __init__(self, out_channels=64, target_grid=16, pretrained=True):
        super().__init__()
        self.out_channels = out_channels
        self.target_grid = target_grid
        self.using_dino = False

        if pretrained:
            try:
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(
                    "facebook/dinov2-base",
                    add_pooling_layer=False,
                )
                self.backbone.requires_grad_(False)
                self.backbone.eval()
                self.using_dino = True
                self._dino_hidden_dim = self.backbone.config.hidden_size  # 768
                logger.info("DINOv2-B/14 loaded (frozen, %d-dim patches)",
                            self._dino_hidden_dim)
            except Exception as e:
                logger.warning("DINOv2 unavailable (%s), using conv fallback", e)
                pretrained = False

        if not pretrained or not self.using_dino:
            self.backbone = RetinotopicConvStack(out_channels=768)
            self._dino_hidden_dim = 768
            self.using_dino = False

        self.channel_proj = nn.Sequential(
            nn.Conv2d(self._dino_hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.LayerNorm([out_channels, target_grid, target_grid]),
            nn.GELU(),
        )

    def _extract_patch_grid(self, pixel_values):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Extract patch tokens and reshape to spatial grid.

        Args:
            pixel_values: [B, 3, H, W]

        Returns:
            [B, hidden_dim, H_patches, W_patches]
        """
        if self.using_dino:
            with torch.no_grad():
                outputs = self.backbone(pixel_values)
                # last_hidden_state: [B, 1 + num_patches, hidden_dim]
                # First token is CLS, rest are patch tokens
                patch_tokens = outputs.last_hidden_state[:, 1:, :]

            B, N, C = patch_tokens.shape
            H_patches = int(N ** 0.5)
            W_patches = H_patches
            # Reshape to spatial grid: [B, C, H, W]
            grid = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patches, W_patches)
        else:
            # Conv stack already outputs spatial grid
            grid = self.backbone(pixel_values)

        return grid

    def forward(self, pixel_values):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Args:
            pixel_values: [B, 3, 224, 224] raw image tensor
                          or [B, 3, H, W] (will be resized)

        Returns:
            [B, out_channels, target_grid, target_grid]
        """
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        # Resize to 224x224 if needed (DINOv2 expects this)
        if pixel_values.shape[2] != 224 or pixel_values.shape[3] != 224:
            pixel_values = F.interpolate(
                pixel_values, size=(224, 224),
                mode='bilinear', align_corners=False
            )

        # Extract spatial patch grid
        grid = self._extract_patch_grid(pixel_values)

        # Interpolate to target grid size if needed
        if grid.shape[2] != self.target_grid or grid.shape[3] != self.target_grid:
            grid = F.interpolate(
                grid, size=(self.target_grid, self.target_grid),
                mode='bilinear', align_corners=False
            )

        # Channel reduction: 768 -> out_channels
        return self.channel_proj(grid)
