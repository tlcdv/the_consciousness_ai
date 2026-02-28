import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from models.core.visual_tectum_projection import VisualTectumProjection
from models.core.capsule_composition import CapsuleCompositionLayer

class TopographicMap(nn.Module):
    """
    Biological Counterpart: Optic Tectum / Superior Colliculus
    
    Maintains a 2D spatial grid representing the agent's egocentric space.
    Sensory inputs (vision, audio) are mapped into this shared coordinate frame,
    preserving spatial relationships (isomorphism).
    """
    def __init__(self, grid_size: int = 16, feature_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        
        # 2D Convolutional layers to fuse modalities into the spatial grid
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([feature_dim, grid_size, grid_size]),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
    def forward(self, visual_grid: torch.Tensor, audio_spatial: torch.Tensor) -> torch.Tensor:
        """
        Fuses visual and spatial audio into a single topographic map.
        
        Args:
            visual_grid: [B, feature_dim, grid_size, grid_size] - from CNN backbone
            audio_spatial: [B, feature_dim, 2] - bearing and elevation features
            
        Returns:
            fused_map: [B, feature_dim, grid_size, grid_size]
        """
        B, C, H, W = visual_grid.shape
        device = visual_grid.device
        
        # Project audio spatial vector into a 2D grid representation
        # Biologically, auditory space is computed, not natively mapped on the retina
        audio_grid = torch.zeros(B, C, H, W, device=device)
        
        # Simple heuristic: audio_spatial holds [x, y] coordinates in [-1, 1]
        # We place a Gaussian bump at that coordinate in the grid
        for b in range(B):
            ax = audio_spatial[b, 0, 0].item() # x (azimuth)
            ay = audio_spatial[b, 0, 1].item() # y (elevation)
            
            # Map [-1, 1] to [0, grid_size-1]
            gx = torch.clamp(torch.tensor((ax + 1) / 2 * (W - 1)), 0, W - 1).int()
            gy = torch.clamp(torch.tensor((ay + 1) / 2 * (H - 1)), 0, H - 1).int()
            
            # Add audio features to that location (with a small blur for uncertainty)
            audio_grid[b, :, gy, gx] = audio_spatial[b, :, 0]
            
        # Apply slight blur to audio grid to represent spatial uncertainty
        audio_grid = F.avg_pool2d(audio_grid, kernel_size=3, stride=1, padding=1)
        
        # Concatenate and fuse
        combined = torch.cat([visual_grid, audio_grid], dim=1) # [B, 2*feature_dim, H, W]
        fused_map = self.fusion_conv(combined)
        
        return fused_map

class RSSMCore(nn.Module):
    """
    Recurrent State Space Model (DreamerV3 style) applied to topographic maps.
    
    Biological Counterpart: Tectal Temporal Integration / Cortical Predictive Coding
    
    Instead of flat vectors, this maintains a recurrent *spatial* state.
    It splits state into:
      h_t: Deterministic hidden state (GRU)
      z_t: Stochastic latent state (Discrete Categorical per spatial location)
    """
    def __init__(self, feature_dim: int = 64, grid_size: int = 16, num_categories: int = 32, num_classes: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.categories = num_categories
        self.classes = num_classes
        
        # Deterministic Recurrence: ConvGRU
        # (Using standard Conv2d logic manually for a simplified GRU step)
        self.gru_update = nn.Conv2d(feature_dim + (num_categories * num_classes), feature_dim * 2, kernel_size=3, padding=1)
        self.gru_reset = nn.Conv2d(feature_dim + (num_categories * num_classes), feature_dim, kernel_size=3, padding=1)
        self.gru_candidate = nn.Conv2d(feature_dim + (num_categories * num_classes), feature_dim, kernel_size=3, padding=1)
        
        # Prior/Posterior Predictors (Encoder/Dynamics)
        # Returns logits for categorical distribution [B, categories*classes, H, W]
        self.posterior_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, num_categories * num_classes, kernel_size=1)
        )
        
        self.prior_net = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, num_categories * num_classes, kernel_size=1)
        )
        
    def step(self, 
             obs_map: Optional[torch.Tensor], 
             h_prev: torch.Tensor, 
             z_prev: torch.Tensor, 
             action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RSSM forward step.
        If obs_map is provided: calculates posterior (z_t|obs, h_t)
        If obs_map is None: calculates prior (z_t|h_t) (Imagination/Prediction)
        """
        # 1. Deterministic update (h_t) based on previous state and action
        # Simplifying action integration: just project and add if present
        z_reshaped = z_prev.view(-1, self.categories * self.classes, self.grid_size, self.grid_size)
        gru_in = torch.cat([h_prev, z_reshaped], dim=1)
        
        # ConvGRU math
        update_gate = torch.sigmoid(self.gru_update(gru_in))
        reset_gate = torch.sigmoid(self.gru_reset(gru_in))
        
        cand_in = torch.cat([h_prev * reset_gate, z_reshaped], dim=1)
        candidate = torch.tanh(self.gru_candidate(cand_in))
        
        # Update h_t
        h_t = (1 - update_gate[:, :self.feature_dim]) * h_prev + update_gate[:, :self.feature_dim] * candidate
        
        # 2. Prior prediction (Dreaming / Prediction)
        prior_logits = self.prior_net(h_t)
        prior_logits = prior_logits.view(-1, self.categories, self.classes, self.grid_size, self.grid_size)
        
        # Straight-Through Estimator (STE) for discrete sampling
        # In a real training loop we use reparameterization, here we use argmax/gumbel
        prior_sample = F.gumbel_softmax(prior_logits, tau=1.0, hard=True, dim=2)
        
        if obs_map is not None:
            # 3. Posterior update (Reality)
            post_in = torch.cat([h_t, obs_map], dim=1)
            posterior_logits = self.posterior_net(post_in)
            posterior_logits = posterior_logits.view(-1, self.categories, self.classes, self.grid_size, self.grid_size)
            
            # STE Sample
            z_t = F.gumbel_softmax(posterior_logits, tau=1.0, hard=True, dim=2)
            return h_t, z_t, prior_logits, posterior_logits
        else:
            # Blind prediction
            z_t = prior_sample
            return h_t, z_t, prior_logits, prior_logits

class SensoryTectum(nn.Module):
    """
    The full midbrain sensory integration layer.
    Replaces raw visual/audio processing with a coherent, spatial world model.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.feature_dim = config.get("tectum_feature_dim", 64)
        self.grid_size = config.get("tectum_grid_size", 16)
        workspace_dim = config.get("workspace_dim", 256)

        self.topo_map = TopographicMap(self.grid_size, self.feature_dim)
        self.rssm = RSSMCore(self.feature_dim, self.grid_size)

        # Qwen2-VL grid adapter: [1536, H, W] -> [B, feature_dim, grid_size, grid_size]
        self.visual_proj = VisualTectumProjection(
            in_channels=config.get("vit_dim", 1536),
            out_channels=self.feature_dim,
            target_grid=self.grid_size
        )

        # Capsule composition layer replaces global_pool + linear projection.
        # Preserves compositional structure through dynamic routing by agreement.
        rssm_channels = self.feature_dim + (self.rssm.categories * self.rssm.classes)
        self.capsule_layer = CapsuleCompositionLayer(
            rssm_channels=rssm_channels,
            grid_size=self.grid_size,
            workspace_dim=workspace_dim,
            num_output_caps=config.get("num_output_caps", 4),
            output_dim=config.get("capsule_output_dim", 16),
            num_primary_caps=config.get("num_primary_caps", 8),
            primary_dim=config.get("capsule_primary_dim", 8),
            routing_iterations=config.get("routing_iterations", 3)
        )

        self.register_buffer('h_state', None)
        self.register_buffer('z_state', None)

        # Cache for reentrant feedback
        self._last_content = None
        self._last_raw_bid = 0.0
        self._last_capsule_poses = None
        self._last_capsule_activities = None
        
    def reset_state(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self.h_state = torch.zeros(batch_size, self.feature_dim, self.grid_size, self.grid_size, device=device)
        self.z_state = torch.zeros(batch_size, self.rssm.categories, self.rssm.classes, self.grid_size, self.grid_size, device=device)
        # Initialize Z randomly
        self.z_state[:, :, 0, :, :] = 1.0 
        
    def forward(self, vision_features: torch.Tensor, audio_spatial: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Process incoming streams, update the world model, and generate a bid for the workspace.
        """
        B = vision_features.shape[0]
        if self.h_state is None or self.h_state.shape[0] != B:
            self.reset_state(B)
            
        # 1. Create Egocentric Topographic Map
        obs_map = self.topo_map(vision_features, audio_spatial)
        
        # 2. Update RSSM World Model
        h_t, z_t, prior_logits, post_logits = self.rssm.step(obs_map, self.h_state, self.z_state)
        
        # Save state
        self.h_state = h_t.detach()
        self.z_state = z_t.detach()
        
        # 3. Calculate Prediction Error (Surprise)
        # KL Divergence: KL(posterior || prior) = sum q * log(q/p)
        # This measures how much the observed reality (posterior) diverges from 
        # the model's expectation (prior). High KL = high surprise = novel input.
        # F.kl_div expects (log_input, target) and computes sum(target * (log(target) - log_input))
        # So: F.kl_div(log_prior, posterior) = KL(posterior || prior)
        q = F.softmax(post_logits, dim=2)   # posterior (reality)
        log_p = F.log_softmax(prior_logits, dim=2)  # prior (prediction)
        kl_div = F.kl_div(log_p, q, reduction='batchmean')
        
        # Scale bid to [0, 1] using tanh
        bid = torch.tanh(kl_div).item()
        
        # 4. Extract content via capsule composition
        z_flat = z_t.view(B, -1, self.grid_size, self.grid_size)
        state_tensor = torch.cat([h_t, z_flat], dim=1)  # [B, C, H, W]

        workspace_content, capsule_activities, capsule_poses = self.capsule_layer(state_tensor)

        # Cache for reentrant feedback
        self._last_content = workspace_content.detach()
        self._last_raw_bid = bid
        self._last_capsule_poses = capsule_poses.detach()
        self._last_capsule_activities = capsule_activities.detach()

        return workspace_content, bid

    def get_capsule_payload(self):
        # type: () -> Dict[str, Any]
        """Returns cached capsule state for structured workspace payloads."""
        if self._last_capsule_poses is None:
            return {}
        return {
            "capsule_poses": self._last_capsule_poses,
            "capsule_activities": self._last_capsule_activities
        }
    
    def receive_broadcast(self, broadcast_content: Any, current_bid: float) -> float:
        """
        Receive top-down feedback from the workspace (Phase 6 Reentrant Processing).
        
        If the broadcast matches our own content closely (low PE), we are already 
        aligned with consciousness and can lower our bid slightly (settled).
        If the broadcast is far from our content (high PE), we should increase 
        our bid to compete harder in the next cycle.
        
        Args:
            broadcast_content: The current workspace broadcast (tensor or dict)
            current_bid: Our current bid value
            
        Returns:
            Updated bid value incorporating top-down context
        """
        if not isinstance(broadcast_content, torch.Tensor) or self._last_content is None:
            # If broadcast is not a tensor (subconscious/empty), maintain baseline
            return current_bid * 0.95  # Slight decay
        
        # Compute prediction error: how different is the broadcast from what we sent?
        with torch.no_grad():
            diff = torch.norm(broadcast_content - self._last_content)
            magnitude = torch.norm(self._last_content) + 1e-8
            pe = (diff / magnitude).item()
        
        # High PE = broadcast diverges from our content = need to push harder
        # Low PE = we're already in the broadcast = can relax slightly
        if pe > 0.3:
            # We're not in the spotlight. Increase bid.
            updated_bid = min(1.0, current_bid + pe * 0.1)
        else:
            # We're recognized. Settle slightly.
            updated_bid = current_bid * (1.0 - pe * 0.1)
        
        return max(0.0, min(1.0, updated_bid))
