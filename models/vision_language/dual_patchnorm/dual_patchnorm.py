import torch
import torch.nn as nn
import einops
from typing import Tuple, Optional 
from dataclasses import dataclass

@dataclass
class DualPatchNormConfig:
    """Configuration for Dual PatchNorm layer"""
    patch_size: Tuple[int, int] = (16, 16)
    hidden_size: int = 768
    eps: float = 1e-6
    elementwise_affine: bool = True
    dropout: float = 0.1
    num_heads: int = 12

class DualPatchNorm(nn.Module):
    """
    Dual PatchNorm implementation for vision transformers.
    Combines spatial and channel normalization for improved feature learning.
    """
    
    def __init__(self, config: DualPatchNormConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Spatial normalization
        self.spatial_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine
        )
        
        # Channel normalization
        self.channel_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Multi-head attention for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Dual PatchNorm
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Normalized tensor of shape (batch_size, num_patches, hidden_size)
        """
        # Patch embedding
        x = einops.rearrange(x, 'b h w c -> b c h w')
        patches = self.patch_embed(x)
        patches = einops.rearrange(patches, 'b c h w -> b (h w) c')
        
        # Spatial normalization
        spatial_normed = self.spatial_norm(patches)
        
        # Channel normalization
        channel_normed = einops.rearrange(patches, 'b n c -> b c n')
        channel_normed = self.channel_norm(channel_normed)
        channel_normed = einops.rearrange(channel_normed, 'b c n -> b n c')
        
        # Concatenate normalized features
        dual_normed = torch.cat([spatial_normed, channel_normed], dim=-1)
        
        # Project to hidden size
        output = self.output_projection(dual_normed)
        
        # Self-attention for feature refinement
        output = einops.rearrange(output, 'b n c -> n b c')
        output, _ = self.attention(output, output, output)
        output = einops.rearrange(output, 'n b c -> b n c')
        
        return output