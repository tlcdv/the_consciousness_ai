import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def squash(x, dim=-1):
    # type: (torch.Tensor, int) -> torch.Tensor
    """
    Squash activation: maps vectors to length in [0, 1) while preserving direction.
    Short vectors get shrunk to near zero, long vectors approach unit length.
    """
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm_sq + 1e-8)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * (x / norm)


class PrimaryCapsuleLayer(nn.Module):
    """
    Converts spatial feature maps into primary capsule pose vectors.

    Takes the RSSM spatial state [B, C, H, W] and produces a set of
    capsule vectors per spatial location via a strided convolution.
    Each capsule vector encodes a local feature part with its pose
    (position, orientation, scale encoded implicitly in the vector).
    """

    def __init__(self, in_channels, num_capsules=8, capsule_dim=8,
                 kernel_size=3, stride=2):
        # type: (int, int, int, int, int) -> None
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Args:
            x: [B, in_channels, H, W]

        Returns:
            [B, num_capsules * H_out * W_out, capsule_dim]
        """
        B = x.shape[0]
        out = self.conv(x)  # [B, num_caps * cap_dim, H_out, W_out]
        H_out, W_out = out.shape[2], out.shape[3]

        # Reshape to [B, num_caps, cap_dim, H_out, W_out] then flatten spatial
        out = out.view(B, self.num_capsules, self.capsule_dim, H_out, W_out)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_caps, H, W, cap_dim]
        out = out.view(B, self.num_capsules * H_out * W_out, self.capsule_dim)

        return squash(out, dim=-1)


class RoutingCapsuleLayer(nn.Module):
    """
    Dynamic routing by agreement (Sabour et al. 2017).

    Primary capsules "vote" for higher level capsules by predicting their
    pose vectors. Routing iteratively adjusts coupling coefficients so that
    primary capsules route to the higher level capsule whose actual pose
    best matches their prediction. This implements compositional binding:
    parts that agree on a whole get bound together.
    """

    def __init__(self, num_primary_caps, primary_dim,
                 num_output_caps=4, output_dim=16, routing_iterations=3):
        # type: (int, int, int, int, int) -> None
        super().__init__()
        self.num_primary = num_primary_caps
        self.num_output = num_output_caps
        self.output_dim = output_dim
        self.routing_iterations = routing_iterations

        # Prediction weight matrix: each primary capsule predicts each output capsule
        # W[i, j] transforms primary_i's pose into a prediction for output_j
        self.W = nn.Parameter(
            torch.randn(num_primary_caps, num_output_caps, output_dim, primary_dim) * 0.01
        )

    def forward(self, primary_caps):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Args:
            primary_caps: [B, num_primary_caps, primary_dim]

        Returns:
            capsule_poses: [B, num_output_caps, output_dim]
            capsule_activities: [B, num_output_caps]
        """
        B = primary_caps.shape[0]

        # Compute vote predictions: u_hat[b, i, j, d] = W[i, j] @ primary[b, i]
        # primary_caps: [B, num_primary, primary_dim]
        # W: [num_primary, num_output, output_dim, primary_dim]
        u_hat = torch.einsum('bip,iojp->bioj', primary_caps, self.W)
        # u_hat: [B, num_primary, num_output, output_dim]

        # Initialize routing logits to zero (uniform coupling)
        b_logits = torch.zeros(B, self.num_primary, self.num_output,
                               device=primary_caps.device)

        # Iterative routing
        v = None
        for r in range(self.routing_iterations):
            # Coupling coefficients: how much each primary routes to each output
            c = F.softmax(b_logits, dim=2)  # [B, num_primary, num_output]

            # Weighted sum of predictions per output capsule
            s = torch.einsum('bio,biod->bod', c, u_hat)  # [B, num_output, output_dim]

            # Squash to get output capsule poses
            v = squash(s, dim=-1)  # [B, num_output, output_dim]

            # Update routing logits (except on last iteration)
            if r < self.routing_iterations - 1:
                # Agreement: dot product between prediction and actual output
                agreement = torch.einsum('biod,bod->bio', u_hat, v)
                b_logits = b_logits + agreement

        # Capsule activity = length of pose vector (already in [0, 1) from squash)
        capsule_activities = torch.norm(v, dim=-1)  # [B, num_output]

        return v, capsule_activities


class CapsuleCompositionLayer(nn.Module):
    """
    Full capsule composition pipeline for the SensoryTectum.

    Takes the RSSM spatial state and produces:
    1. workspace_content: projected capsule poses for workspace competition
    2. capsule_activities: per capsule activation scalars
    3. capsule_poses: raw pose vectors for structured payloads

    This replaces the previous global_pool + linear projection, preserving
    compositional structure through the capsule hierarchy.
    """

    def __init__(self, rssm_channels, grid_size, workspace_dim=256,
                 num_output_caps=4, output_dim=16,
                 num_primary_caps=8, primary_dim=8,
                 routing_iterations=3):
        # type: (int, int, int, int, int, int, int, int) -> None
        super().__init__()

        self.primary = PrimaryCapsuleLayer(
            in_channels=rssm_channels,
            num_capsules=num_primary_caps,
            capsule_dim=primary_dim
        )

        # After stride=2 conv, spatial dims halve
        reduced_h = (grid_size + 1) // 2  # ceiling division for odd grid sizes
        reduced_w = (grid_size + 1) // 2
        total_primary = num_primary_caps * reduced_h * reduced_w

        self.routing = RoutingCapsuleLayer(
            num_primary_caps=total_primary,
            primary_dim=primary_dim,
            num_output_caps=num_output_caps,
            output_dim=output_dim,
            routing_iterations=routing_iterations
        )

        # Project concatenated output capsule poses to workspace dimension
        self.workspace_proj = nn.Linear(num_output_caps * output_dim, workspace_dim)

        # Cache for reentrant feedback and inspection
        self._last_poses = None
        self._last_activities = None

    def forward(self, state_tensor):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Args:
            state_tensor: [B, rssm_channels, grid_size, grid_size]

        Returns:
            workspace_content: [B, workspace_dim]
            capsule_activities: [B, num_output_caps]
            capsule_poses: [B, num_output_caps, output_dim]
        """
        B = state_tensor.shape[0]

        # Primary capsules from spatial features
        primary_caps = self.primary(state_tensor)

        # Route to higher level capsules
        capsule_poses, capsule_activities = self.routing(primary_caps)

        # Project to workspace dimension
        flat_poses = capsule_poses.reshape(B, -1)
        workspace_content = self.workspace_proj(flat_poses)

        # Cache for external access
        self._last_poses = capsule_poses.detach()
        self._last_activities = capsule_activities.detach()

        return workspace_content, capsule_activities, capsule_poses
