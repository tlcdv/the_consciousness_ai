from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



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
        # type: (torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
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
        # type: (torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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


class HierarchicalCapsuleComposition(nn.Module):
    """
    Multi-level capsule hierarchy with 3-4 routing levels and reentrant
    top-down feedback between levels.

    Biological basis: Feinberg & Mallatt require 3-4+ hierarchical levels
    with genuine compositional transformation at each level. Each routing
    layer implements dynamic routing by agreement (Sabour 2017), where
    capsules at level N vote for capsules at level N+1.

    Reentrant processing (Lamme & Roelfsema 2000): after the initial
    bottom-up sweep, higher levels send predictions back to lower levels.
    Lower levels compute prediction errors and re-route. This models
    V1-LGN, V2-V1, V4-V2 type reciprocal connections. The feedback gain
    (alpha) is weaker than bottom-up, matching biological asymmetry.

    Default hierarchy (4 levels total):
        Level 1: PrimaryCapsuleLayer (stride-2 conv) -> local features
        Level 2: RoutingCapsuleLayer -> object primitives (16 caps, 12-D)
        Level 3: RoutingCapsuleLayer -> object categories (8 caps, 16-D)
        Level 4: RoutingCapsuleLayer -> scene/workspace (4 caps, 16-D)
    """

    DEFAULT_HIERARCHY = [
        (16, 12),  # Level 2: 16 intermediate capsules, 12-D poses
        (8, 16),   # Level 3: 8 higher capsules, 16-D poses
        (4, 16),   # Level 4: 4 output capsules, 16-D poses
    ]

    def __init__(self, rssm_channels, grid_size, workspace_dim=256,
                 num_primary_caps=8, primary_dim=8,
                 hierarchy_spec=None, routing_iterations=3,
                 reentrant_iterations=2, feedback_alpha=0.5):
        # type: (int, int, int, int, int, list[tuple[int, int]] | None, int, int, float) -> None
        super().__init__()

        if hierarchy_spec is None:
            hierarchy_spec = list(self.DEFAULT_HIERARCHY)

        self.num_levels = 1 + len(hierarchy_spec)  # primary + routing levels
        self.reentrant_iterations = reentrant_iterations
        self.feedback_alpha = feedback_alpha

        # Level 1: primary capsules from spatial features
        self.primary = PrimaryCapsuleLayer(
            in_channels=rssm_channels,
            num_capsules=num_primary_caps,
            capsule_dim=primary_dim
        )

        # Compute total primary capsule count after stride-2 spatial reduction
        reduced_h = (grid_size + 1) // 2
        reduced_w = (grid_size + 1) // 2
        total_primary = num_primary_caps * reduced_h * reduced_w

        # Build routing layers: each transforms level N capsules into level N+1
        self.routing_layers = nn.ModuleList()
        prev_num_caps = total_primary
        prev_dim = primary_dim

        # Track dimensions at each level for feedback projections
        level_dims = [primary_dim]  # level 0 = primary capsule dim

        for num_caps, cap_dim in hierarchy_spec:
            self.routing_layers.append(RoutingCapsuleLayer(
                num_primary_caps=prev_num_caps,
                primary_dim=prev_dim,
                num_output_caps=num_caps,
                output_dim=cap_dim,
                routing_iterations=routing_iterations
            ))
            level_dims.append(cap_dim)
            prev_num_caps = num_caps
            prev_dim = cap_dim

        # Feedback projections: level N+1 -> level N dimension.
        # feedback_projections[i] projects from level (i+1) dim to level i dim.
        # Only needed between routing levels (not from primary to spatial).
        self.feedback_projections = nn.ModuleList()
        for i in range(len(hierarchy_spec) - 1):
            higher_dim = level_dims[i + 2]   # level i+2 in full hierarchy = routing level i+1
            lower_dim = level_dims[i + 1]    # level i+1 in full hierarchy = routing level i
            self.feedback_projections.append(nn.Linear(higher_dim, lower_dim))

        # Final projection to workspace dimension
        final_num_caps, final_dim = hierarchy_spec[-1]
        self.workspace_proj = nn.Linear(final_num_caps * final_dim, workspace_dim)

        # Cache for reentrant feedback and inspection
        self._last_poses = None
        self._last_activities = None
        self._level_poses = []     # poses at each routing level
        self._level_activities = [] # activities at each routing level
        self._level_prediction_errors = []  # PE per level per reentrant iteration

    def _bottom_up_pass(self, primary_caps):
        # type: (torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]
        """Run bottom-up routing through all levels. Returns (poses, activities) per level."""
        level_results = []
        caps = primary_caps
        for routing in self.routing_layers:
            caps, activities = routing(caps)
            level_results.append((caps, activities))
        return level_results

    def _top_down_feedback(self, level_results):
        # type: (list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float]]
        """
        Apply top-down predictions from higher levels to lower levels.

        For each pair of adjacent routing levels (from top to bottom):
        1. Higher level poses are projected to lower level dimension
        2. Prediction is broadcast (mean over higher capsules, tiled to lower count)
        3. Error = lower_poses - prediction
        4. Lower level input is refined: lower_poses + alpha * error
        5. Re-route from the refined lower level upward

        Returns updated level results and per-level prediction error norms.
        """
        poses_list = [lr[0] for lr in level_results]
        per_level_pe = []

        # Top-down: from highest routing level down to the second routing level.
        # feedback_projections[i] projects from routing level (i+1) to routing level i.
        for i in range(len(self.feedback_projections) - 1, -1, -1):
            higher_poses = poses_list[i + 1]  # [B, num_higher, higher_dim]
            lower_poses = poses_list[i]        # [B, num_lower, lower_dim]

            # Project higher level prediction to lower dimension
            prediction = self.feedback_projections[i](higher_poses)  # [B, num_higher, lower_dim]

            # Broadcast: average over higher capsules, expand to lower count
            pred_mean = prediction.mean(dim=1, keepdim=True)  # [B, 1, lower_dim]
            pred_broadcast = pred_mean.expand_as(lower_poses)  # [B, num_lower, lower_dim]

            # Prediction error
            error = lower_poses - pred_broadcast
            pe_norm = error.norm(dim=-1).mean().item()
            per_level_pe.append((i, pe_norm))

            # Refine lower level with residual error and re-route upward
            refined_input = lower_poses + self.feedback_alpha * error
            refined_poses, refined_acts = self.routing_layers[i + 1](refined_input)
            poses_list[i + 1] = refined_poses
            level_results[i + 1] = (refined_poses, refined_acts)

        # Sort PE by level index (was computed top-down)
        per_level_pe.sort(key=lambda x: x[0])
        pe_values = [pe for _, pe in per_level_pe]

        return level_results, pe_values

    def forward(self, state_tensor):
        # type: (torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Args:
            state_tensor: [B, rssm_channels, grid_size, grid_size]

        Returns:
            workspace_content: [B, workspace_dim]
            capsule_activities: [B, num_output_caps] (final level)
            capsule_poses: [B, num_output_caps, output_dim] (final level)
        """
        B = state_tensor.shape[0]

        # Level 1: primary capsules
        primary_caps = self.primary(state_tensor)

        # Initial bottom-up pass
        level_results = self._bottom_up_pass(primary_caps)

        # Reentrant top-down/bottom-up iterations
        all_pe = []
        for _ in range(self.reentrant_iterations):
            if len(self.feedback_projections) > 0:
                level_results, pe_values = self._top_down_feedback(level_results)
                all_pe.append(pe_values)

        # Extract final outputs
        capsule_poses, capsule_activities = level_results[-1]

        # Project to workspace
        flat_poses = capsule_poses.reshape(B, -1)
        workspace_content = self.workspace_proj(flat_poses)

        # Cache for external access
        self._last_poses = capsule_poses.detach()
        self._last_activities = capsule_activities.detach()
        self._level_poses = [lr[0].detach() for lr in level_results]
        self._level_activities = [lr[1].detach() for lr in level_results]
        self._level_prediction_errors = all_pe

        return workspace_content, capsule_activities, capsule_poses

    def get_all_level_poses(self):
        # type: () -> list[tuple[torch.Tensor, torch.Tensor]]
        """
        Returns cached (poses, activities) for each routing level.

        Level indices correspond to routing layers:
            index 0 = first routing level (Level 2 in the full hierarchy)
            index 1 = second routing level (Level 3)
            ...
        """
        return list(zip(self._level_poses, self._level_activities))

    def get_level_prediction_errors(self):
        # type: () -> list[list[float]]
        """
        Returns per-level prediction errors for each reentrant iteration.

        Outer list: one entry per reentrant iteration.
        Inner list: PE values indexed by feedback projection index
        (0 = between routing levels 0 and 1, etc.).

        Empty if reentrant_iterations=0 or hierarchy has fewer than 2 routing levels.
        """
        return self._level_prediction_errors
