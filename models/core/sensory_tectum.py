from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from models.core.retinotopic_encoder import RetinotopicEncoder
from models.core.capsule_composition import HierarchicalCapsuleComposition

class TopographicMap(nn.Module):
    """
    Biological Counterpart: Optic Tectum / Superior Colliculus

    Maintains a 2D spatial grid representing the agent's egocentric space.
    Sensory inputs (vision, audio, somatosensory) are mapped into this shared
    coordinate frame, preserving spatial relationships (isomorphism).

    Uses inverse effectiveness fusion (Stein & Meredith 1993): proportional
    enhancement is greatest when individual unimodal responses are weakest.
    This is how the biological SC detects faint multimodal stimuli that would
    be missed by either modality alone.

    The somatosensory channel projects the body schema (10 body parts x 8 features)
    onto the spatial grid via a learned linear map. Biologically, the deep layers
    of the SC contain somatotopic maps aligned with the visual and auditory maps
    (Stein & Meredith 1993, ch. 4).
    """
    def __init__(self, grid_size: int = 16, feature_dim: int = 64,
                 body_parts: int = 10, body_features: int = 8):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        # Somatosensory projection: body_schema [B, body_parts, body_features]
        # -> [B, feature_dim, grid_size, grid_size]
        self.body_proj = nn.Linear(body_parts * body_features, feature_dim * grid_size * grid_size)

        # Refinement conv applied after inverse effectiveness additive fusion.
        # Input is feature_dim (not 2x) because IE does weighted addition.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([feature_dim, grid_size, grid_size]),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU()
        )

    def _place_audio_on_grid(self, audio_spatial, B, C, H, W, device):
        # type: (torch.Tensor, int, int, int, int, torch.device) -> torch.Tensor
        """
        Project audio bearing/elevation into a 2D spatial grid.

        Biologically, auditory space is computed (from ITD, ILD, spectral cues)
        and calibrated to match the visual map during development (Knudsen &
        Brainard 1991). We place audio features at the corresponding grid
        location with spatial blur for receptive field uncertainty.
        """
        audio_grid = torch.zeros(B, C, H, W, device=device)

        for b in range(B):
            ax = audio_spatial[b, 0, 0].item()  # azimuth
            ay = audio_spatial[b, 0, 1].item()  # elevation

            gx = torch.clamp(torch.tensor((ax + 1) / 2 * (W - 1)), 0, W - 1).int()
            gy = torch.clamp(torch.tensor((ay + 1) / 2 * (H - 1)), 0, H - 1).int()

            audio_grid[b, :, gy, gx] = audio_spatial[b, :, 0]

        # Spatial blur: auditory RFs are 40-80 degrees in the SC,
        # much coarser than visual RFs (10-30 degrees)
        audio_grid = F.avg_pool2d(audio_grid, kernel_size=3, stride=1, padding=1)
        return audio_grid

    def _fuse_inverse_effectiveness(self, visual, audio, epsilon=1e-6):
        # type: (torch.Tensor, torch.Tensor, float) -> torch.Tensor
        """
        Inverse effectiveness fusion (Stein & Meredith 1993, Ohshiro et al. 2011).

        When both unimodal responses at a grid cell are weak, the proportional
        enhancement from combining them is large. When both are strong, the
        enhancement is modest. This follows from the sigmoid response function:
        weak inputs operate on the steep part of the curve (large gain from
        combination), strong inputs are near saturation (small gain).

        Args:
            visual: [B, C, H, W] visual feature grid
            audio:  [B, C, H, W] audio feature grid (sparse, most cells zero)
            epsilon: numerical stability

        Returns:
            [B, C, H, W] fused feature grid
        """
        v_mag = visual.norm(dim=1, keepdim=True)  # [B, 1, H, W]
        a_mag = audio.norm(dim=1, keepdim=True)

        # Weight inversely proportional to the stronger unimodal signal
        max_unimodal = torch.max(v_mag, a_mag) + epsilon
        ie_weight = 1.0 / max_unimodal
        # Normalize so mean weight is 1.0 (preserves overall magnitude)
        ie_weight = ie_weight / (ie_weight.mean() + epsilon)

        # Additive fusion: visual is the anchor, audio is modulated
        fused = visual + audio * ie_weight
        return fused

    def _project_body_to_grid(self, body_schema: torch.Tensor, B: int, device: torch.device) -> torch.Tensor:
        """
        Project the body schema onto the spatial grid.

        The body schema [B, body_parts, body_features] is flattened and linearly
        mapped to [B, feature_dim, grid_size, grid_size]. This creates a
        somatotopic spatial representation analogous to the deep layer maps
        in the biological superior colliculus.
        """
        flat = body_schema.reshape(B, -1)  # [B, body_parts * body_features]
        projected = self.body_proj(flat)    # [B, feature_dim * grid_size * grid_size]
        return projected.view(B, self.feature_dim, self.grid_size, self.grid_size)

    def forward(self, visual_grid: torch.Tensor, audio_spatial: torch.Tensor,
                body_schema: torch.Tensor | None = None) -> torch.Tensor:
        """
        Fuses visual, spatial audio, and somatosensory input into a single
        topographic map using inverse effectiveness.

        Args:
            visual_grid: [B, feature_dim, grid_size, grid_size] from RetinotopicEncoder
            audio_spatial: [B, feature_dim, 2] bearing and elevation features
            body_schema: Optional [B, body_parts, body_features] from SelfRepresentationCore.
                When provided, projected onto the grid and fused via inverse effectiveness.

        Returns:
            fused_map: [B, feature_dim, grid_size, grid_size]
        """
        B, C, H, W = visual_grid.shape
        device = visual_grid.device

        audio_grid = self._place_audio_on_grid(audio_spatial, B, C, H, W, device)

        # Inverse effectiveness fusion: vision + audio
        fused = self._fuse_inverse_effectiveness(visual_grid, audio_grid)

        # Trimodal fusion: add somatosensory channel if available
        if body_schema is not None:
            body_grid = self._project_body_to_grid(body_schema, B, device)
            fused = self._fuse_inverse_effectiveness(fused, body_grid)

        # Refinement convolutions
        fused_map = self.fusion_conv(fused)
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
    def __init__(self, feature_dim: int = 64, grid_size: int = 16, num_categories: int = 32, num_classes: int = 32, action_dim: int = 0, latent_mode: str = "discrete"):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.categories = num_categories
        self.classes = num_classes
        # Latent representation for z_t. "discrete" (default): per-cell categorical with
        # gumbel-softmax STE, the baseline. "continuous" (Path B1): a Gaussian latent of
        # the SAME [B, categories, classes, grid, grid] shape, reusing the prior/posterior
        # nets as the mean and adding one log-variance parameter. The B1 bet tests whether
        # the discrete latent was the wall that blocked stimulus identity (the 2026-07
        # ceiling test showed even direct supervision could not push identity into the
        # discrete latent). Every downstream shape is identical, so the capsule/workspace
        # path is unchanged; only the sampling and KL differ.
        if latent_mode not in ("discrete", "continuous"):
            raise ValueError(f"latent_mode must be discrete or continuous, got {latent_mode}")
        self.latent_mode = latent_mode
        # Optional action-conditioning for the value-equivalent world model. Built only
        # when action_dim > 0, so the baseline ConvGRU step is bit-identical when off.
        self.action_embed = nn.Linear(action_dim, feature_dim) if action_dim > 0 else None
        # Continuous mode: one shared log-variance per (category, class) channel. The
        # prior/posterior nets supply the MEAN (reused unchanged); this is the only new
        # parameter. Not created in discrete mode -> baseline state_dict is bit-identical.
        if latent_mode == "continuous":
            self.cont_logvar = nn.Parameter(
                torch.zeros(1, num_categories, num_classes, 1, 1))

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
             obs_map: torch.Tensor | None, 
             h_prev: torch.Tensor, 
             z_prev: torch.Tensor, 
             action: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RSSM forward step.
        If obs_map is provided: calculates posterior (z_t|obs, h_t)
        If obs_map is None: calculates prior (z_t|h_t) (Imagination/Prediction)
        """
        # 1. Deterministic update (h_t) based on previous state and action
        # Simplifying action integration: just project and add if present
        z_reshaped = z_prev.view(-1, self.categories * self.classes, self.grid_size, self.grid_size)

        # Action-conditioning: add the projected action to the deterministic state used to
        # compute the GRU gates and candidate. The residual update still carries h_prev.
        # No-op when action_embed is None or action is None -> baseline bit-identical.
        h_cond = h_prev
        if action is not None and self.action_embed is not None:
            a = self.action_embed(action)  # [B, feature_dim]
            h_cond = h_prev + a.view(a.shape[0], self.feature_dim, 1, 1)

        gru_in = torch.cat([h_cond, z_reshaped], dim=1)

        # ConvGRU math
        update_gate = torch.sigmoid(self.gru_update(gru_in))
        reset_gate = torch.sigmoid(self.gru_reset(gru_in))

        cand_in = torch.cat([h_cond * reset_gate, z_reshaped], dim=1)
        candidate = torch.tanh(self.gru_candidate(cand_in))
        
        # Update h_t
        h_t = (1 - update_gate[:, :self.feature_dim]) * h_prev + update_gate[:, :self.feature_dim] * candidate
        
        # 2. Prior prediction (Dreaming / Prediction)
        prior_logits = self.prior_net(h_t)
        prior_logits = prior_logits.view(-1, self.categories, self.classes, self.grid_size, self.grid_size)

        if self.latent_mode == "continuous":
            # Gaussian latent: prior_logits/posterior_logits are reinterpreted as MEANS
            # (same nets, same shape). Sample by reparameterization when training, take
            # the mode (mean) in eval so decoding/probes are deterministic. The returned
            # "logits" slots carry the means, consumed by the Gaussian KL in the tectum.
            prior_sample = self._reparam(prior_logits)
            if obs_map is not None:
                post_in = torch.cat([h_t, obs_map], dim=1)
                posterior_mean = self.posterior_net(post_in).view(
                    -1, self.categories, self.classes, self.grid_size, self.grid_size)
                z_t = self._reparam(posterior_mean)
                return h_t, z_t, prior_logits, posterior_mean
            return h_t, prior_sample, prior_logits, prior_logits

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

    def _reparam(self, mean: torch.Tensor) -> torch.Tensor:
        """Gaussian reparameterization with the shared per-channel log-variance.

        Training: mean + eps * std (a real sample, gradient flows to mean and logvar).
        Eval: the mean (mode), so probes and decoding see a deterministic latent.
        """
        if not self.training:
            return mean
        std = torch.exp(0.5 * self.cont_logvar)
        return mean + torch.randn_like(mean) * std

class SensoryTectum(nn.Module):
    """
    The full midbrain sensory integration layer.
    Replaces raw visual/audio processing with a coherent, spatial world model.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.feature_dim = config.get("tectum_feature_dim", 64)
        self.grid_size = config.get("tectum_grid_size", 16)
        workspace_dim = config.get("workspace_dim", 256)

        self.topo_map = TopographicMap(self.grid_size, self.feature_dim)
        self.latent_mode = config.get("rssm_latent_mode", "discrete")
        self.rssm = RSSMCore(self.feature_dim, self.grid_size,
                             action_dim=config.get("wm_action_dim", 0),
                             latent_mode=self.latent_mode)

        # Retinotopic encoder: DINOv2-B/14 (frozen) -> [B, feature_dim, grid_size, grid_size]
        # Falls back to conv stack when DINOv2 weights unavailable (CI/testing)
        use_pretrained = config.get("use_pretrained_dino", False)
        self.retinotopic_encoder = RetinotopicEncoder(
            out_channels=self.feature_dim,
            target_grid=self.grid_size,
            pretrained=use_pretrained,
        )

        # Hierarchical capsule composition: 4 levels of compositional
        # transformation via dynamic routing by agreement (Sabour 2017).
        # Required by Feinberg & Mallatt Feature #3 (3-4+ hierarchical levels).
        rssm_channels = self.feature_dim + (self.rssm.categories * self.rssm.classes)
        self.capsule_layer = HierarchicalCapsuleComposition(
            rssm_channels=rssm_channels,
            grid_size=self.grid_size,
            workspace_dim=workspace_dim,
            num_primary_caps=config.get("num_primary_caps", 8),
            primary_dim=config.get("capsule_primary_dim", 8),
            hierarchy_spec=config.get("capsule_hierarchy_spec", None),
            routing_iterations=config.get("routing_iterations", 3),
            reentrant_iterations=config.get("capsule_reentrant_iterations", 2),
            feedback_alpha=config.get("capsule_feedback_alpha", 0.5)
        )

        self.register_buffer('h_state', None)
        self.register_buffer('z_state', None)

        # Truncated BPTT window. The previous behavior of detaching every
        # step severed the gradient at every recurrent step, making the
        # RSSM a one-step encoder despite being labeled a world model.
        # With bptt_window=K, gradient flows through K consecutive RSSM
        # steps before being detached, so a loss at step t can update
        # the RSSM weights based on the previous K-1 hidden states.
        self.bptt_window = config.get("bptt_window", 8)
        self._steps_since_detach = 0

        # Cache for reentrant feedback
        self._last_content = None
        self._last_raw_bid = 0.0
        self._last_capsule_poses = None
        self._last_capsule_activities = None

        # When True, the capsule/workspace path uses a detached RSSM latent so its graph
        # is not retained across a value-equivalent world-model BPTT trial (set by the
        # training loop under --enable-wm-predict). Default False -> baseline unchanged.
        self.detach_downstream = False

    def reset_state(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self.h_state = torch.zeros(batch_size, self.feature_dim, self.grid_size, self.grid_size, device=device)
        # Uniform categorical prior: every class equally likely. Avoids the
        # pre-fix bias where every episode started with class 0 peaked, which
        # gave the world model the same arbitrary initial belief on every
        # reset and biased ablation comparisons.
        if self.latent_mode == "continuous":
            # Continuous latent: start at the zero mean (standard-normal prior mode).
            self.z_state = torch.zeros(
                batch_size, self.rssm.categories, self.rssm.classes,
                self.grid_size, self.grid_size, device=device)
        else:
            uniform_p = 1.0 / self.rssm.classes
            self.z_state = torch.full(
                (batch_size, self.rssm.categories, self.rssm.classes,
                 self.grid_size, self.grid_size),
                uniform_p,
                device=device,
            )
        # Episode boundary: reset the BPTT cycle counter so the new
        # episode starts with a fresh K-step window.
        self._steps_since_detach = 0
        
    def forward(self, vision_features: torch.Tensor, audio_spatial: torch.Tensor,
                body_schema: torch.Tensor | None = None,
                action: torch.Tensor | None = None) -> tuple[torch.Tensor, float]:
        """
        Process incoming streams, update the world model, and generate a bid for the workspace.

        Args:
            vision_features: either raw image frames [B, 3, 224, 224] (will be
                encoded by RetinotopicEncoder) or pre-encoded features
                [B, feature_dim, grid_size, grid_size] (passed through directly)
            audio_spatial: [B, feature_dim, 2] bearing and elevation features
            body_schema: Optional [B, body_parts, body_features] from self-model
        """
        if vision_features.dim() == 3:
            vision_features = vision_features.unsqueeze(0)

        # Auto-detect: raw frames (3 channels) vs pre-encoded (feature_dim channels)
        if vision_features.shape[1] <= 3:
            vision_features = self.retinotopic_encoder(vision_features)

        B = vision_features.shape[0]
        if self.h_state is None or self.h_state.shape[0] != B:
            self.reset_state(B)

        # 1. Create Egocentric Topographic Map (inverse effectiveness fusion)
        obs_map = self.topo_map(vision_features, audio_spatial, body_schema=body_schema)
        # Cache spatial features for topographic loss computation in training loop
        self._last_obs_map = obs_map

        # 2. Update RSSM World Model (optionally action-conditioned). When the value-
        # equivalent world model owns the per-trial BPTT graph, detach obs_map into the
        # RSSM so the heavy full-resolution encoder activations are not retained across
        # the whole trial. The WM then trains the RSSM (gru / posterior / prior, where
        # working memory lives), not the encoder, which already produces an identity-rich
        # obs_map. Off by default -> baseline bit-identical.
        rssm_obs = obs_map.detach() if self.detach_downstream else obs_map
        h_t, z_t, prior_logits, post_logits = self.rssm.step(
            rssm_obs, self.h_state, self.z_state, action=action)
        # Cache the grad-bearing prior/posterior logits for the optional value-equivalent
        # world-model objective (--enable-wm-predict). Unused when off -> bit-identical.
        self._last_prior_logits = prior_logits
        self._last_post_logits = post_logits

        # 2b. Truncated BPTT save. Detach every bptt_window steps so the
        # graph stays bounded but a loss at the end of the window can flow
        # gradient back through up to (window - 1) RSSM steps.
        self._steps_since_detach += 1
        if self._steps_since_detach >= self.bptt_window:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
            self._steps_since_detach = 0
        else:
            self.h_state = h_t
            self.z_state = z_t
        
        # 3. Calculate Prediction Error (Surprise)
        if self.latent_mode == "continuous":
            # Gaussian KL between posterior N(post_mean, var) and prior N(prior_mean, var)
            # with the shared per-channel variance. Equal variances cancel the log and
            # trace terms, leaving KL = 0.5 * sum (post_mean - prior_mean)^2 / var,
            # averaged over the batch (batchmean, matching the discrete branch).
            var = torch.exp(self.rssm.cont_logvar)
            kl_map = 0.5 * (post_logits - prior_logits) ** 2 / (var + 1e-8)
            kl_div = kl_map.sum() / post_logits.shape[0]
        else:
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
        # Cache the grad-bearing pre-capsule RSSM latent (h_t + z_t) for the optional
        # world-model reconstruction objective (R1, --enable-wm-recon). This is the
        # exact tensor the collapse-locus probe shows loses stimulus identity; the
        # recon head reconstructs the frame from it to force the latent to keep
        # identity. Unconditional cache mirrors _last_obs_map (used by TDANN); when
        # wm-recon is off it is simply unused, so the baseline stays bit-identical.
        self._last_state_tensor = state_tensor

        # When the value-equivalent world model owns the recurrent graph (it trains the
        # RSSM directly from state_tensor with per-trial BPTT), the capsule/workspace path
        # does not need to be grad-bearing, so detaching its input keeps the heavy
        # downstream graph (capsule + reentrant settle) from being retained across the
        # whole trial. Off by default -> baseline bit-identical.
        capsule_input = state_tensor.detach() if self.detach_downstream else state_tensor
        workspace_content, capsule_activities, capsule_poses = self.capsule_layer(capsule_input)

        # Cache for reentrant feedback
        self._last_content = workspace_content.detach()
        self._last_raw_bid = bid
        self._last_capsule_poses = capsule_poses.detach()
        self._last_capsule_activities = capsule_activities.detach()

        return workspace_content, bid

    def get_capsule_payload(self):
        # type: () -> dict[str, Any]
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
        # Workspace returns one of: raw tensor; dict with "_fused" key
        # (Phase A attention-weighted fusion); dict with "tensor" key (legacy
        # winner-take-all vision payload). Extract a tensor or decay.
        if isinstance(broadcast_content, torch.Tensor):
            bc_tensor = broadcast_content
        elif isinstance(broadcast_content, dict):
            bc_tensor = broadcast_content.get("_fused")
            if not isinstance(bc_tensor, torch.Tensor):
                bc_tensor = broadcast_content.get("tensor")
            if not isinstance(bc_tensor, torch.Tensor):
                return current_bid * 0.95  # subconscious/empty broadcast
        else:
            return current_bid * 0.95

        if self._last_content is None:
            return current_bid * 0.95

        # Compute prediction error: how different is the broadcast from what we sent?
        with torch.no_grad():
            bc_flat = bc_tensor.view(-1)
            last_flat = self._last_content.view(-1)
            if bc_flat.shape != last_flat.shape:
                n = min(bc_flat.shape[0], last_flat.shape[0])
                bc_flat = bc_flat[:n]
                last_flat = last_flat[:n]
            diff = torch.norm(bc_flat - last_flat)
            magnitude = torch.norm(last_flat) + 1e-8
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
