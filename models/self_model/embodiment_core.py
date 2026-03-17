from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class SomatotopicMap(nn.Module):
    """
    Biological Counterpart: Primary Somatosensory Cortex (S1) / Homunculus.
    
    Creates a topologically preserved representation of the agent's body.
    Instead of a flat vector of joint angles, it maps joints, forces, and skin 
    sensors into a spatial arrangement that mirrors the physical body structure.
    """
    def __init__(self, num_body_parts: int = 10, feature_per_part: int = 8, map_size: int = 8):
        super().__init__()
        self.num_parts = num_body_parts
        self.part_dim = feature_per_part
        self.map_size = map_size
        
        # Hardcoded layout of body parts into a 2D grid (Homunculus)
        # e.g. 0: Head, 1: Torso, 2: L_Arm, 3: R_Arm, etc.
        # This mapping allows spatial convolutions to detect full-body gestures
        self.register_buffer('body_grid_indices', self._create_homunculus(num_body_parts, map_size))
        
        # Convolutional processing over the body map
        self.body_conv = nn.Sequential(
            nn.Conv2d(feature_per_part, feature_per_part * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([feature_per_part * 2, map_size, map_size]),
            nn.Conv2d(feature_per_part * 2, feature_per_part * 2, kernel_size=3, padding=1),
            nn.GELU()
        )
        
    def _create_homunculus(self, num_parts: int, map_size: int) -> torch.Tensor:
        """
        Creates a 2D grid assigning body parts to specific (y, x) coordinates.
        This provides the structural priors for the Somatotopic Map.
        Returns tensor of shape [num_parts, 2] containing (y,x) coords.
        """
        grid = torch.zeros(num_parts, 2, dtype=torch.long)
        
        # Simplified bipedal mappings (can be overridden by config later)
        if num_parts >= 6 and map_size >= 5:
            grid[0] = torch.tensor([0, map_size//2])           # Head (Top center)
            grid[1] = torch.tensor([map_size//3, map_size//2]) # Torso 
            grid[2] = torch.tensor([map_size//3, 1])           # L_Arm
            grid[3] = torch.tensor([map_size//3, map_size-2])  # R_Arm
            grid[4] = torch.tensor([map_size-1, 2])            # L_Leg
            grid[5] = torch.tensor([map_size-1, map_size-3])   # R_Leg
            # Fill remaining parts near torso
            for i in range(6, num_parts):
                grid[i] = torch.tensor([map_size//2, map_size//2])
        else:
            # Linear fallback
            for i in range(num_parts):
                grid[i] = torch.tensor([i % map_size, i % map_size])
                
        return grid
        
    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        """
        Maps flat body part features into the 2D Homunculus grid.
        
        Args:
            part_features: [B, num_parts, feature_per_part]
        Returns:
            body_map: [B, feature_per_part * 2, map_size, map_size]
        """
        B = part_features.shape[0]
        device = part_features.device
        
        # Initialize empty map [B, C, H, W]
        body_map = torch.zeros(B, self.part_dim, self.map_size, self.map_size, device=device)
        
        # Scatter features into the spatial grid based on the Homunculus layout
        for i in range(self.num_parts):
            y, x = self.body_grid_indices[i]
            body_map[:, :, y, x] = part_features[:, i, :]
            
        # Optional: Apply slight Gaussian blur here for spatial receptive fields (overlapping representations)
        body_map = F.avg_pool2d(body_map, kernel_size=3, stride=1, padding=1)
        
        # Process spatially
        processed_map = self.body_conv(body_map)
        return processed_map

class ProprioceptiveProcessor(nn.Module):
    """
    Biological Counterpart: Spinocerebellar tract / Thalamus (VPL/VPM)
    
    Translates raw joint states and collision data into the unified Neural format
    expected by the Somatotopic Map. Also maintains a short temporal history 
    to detect motion/velocity implicitly.
    """
    def __init__(self, raw_state_dim: int, num_parts: int = 10, feature_per_part: int = 8):
        super().__init__()
        self.num_parts = num_parts
        self.feature_per_part = feature_per_part
        self.raw_dim = raw_state_dim
        
        # Learnable projection from flat raw state to structured parts
        # If the environment gives us a flat vector (e.g., 60 dims for 10 joints * 6 DOF)
        # We need to map it. Ideally, the Unity env groups them, but we handle flat arrays here.
        self.feature_extractor = nn.Sequential(
            nn.Linear(raw_state_dim, num_parts * feature_per_part),
            nn.LayerNorm(num_parts * feature_per_part),
            nn.GELU()
        )
        
        self.somato_map = SomatotopicMap(num_parts, feature_per_part, map_size=8)
        
        # To generate a bid for the Workspace Competition
        self.bid_estimator = nn.Sequential(
            nn.Linear(feature_per_part * 2 * 64, 128), # 64 = 8x8 map_size
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Bid is always [0, 1]
        )
        
    def forward(self, raw_proprioception: torch.Tensor, collision_flags: torch.Tensor = None) -> tuple[torch.Tensor, float]:
        """
        Processes physical state and estimates salience (e.g. pain/contact = high bid).
        
        Args:
            raw_proprioception: [B, raw_state_dim]
            collision_flags: [B, num_parts] Optional collision/pain sensors
            
        Returns:
            spatial_state: [B, C, H, W] The Somatotopic body schema
            bid: float (0.0 to 1.0) Salience of the body state
        """
        B = raw_proprioception.shape[0]
        device = raw_proprioception.device
        
        # 1. Extract structural part features
        flat_features = self.feature_extractor(raw_proprioception)
        part_features = flat_features.view(B, self.num_parts, self.feature_per_part)
        
        # Modulate part features by pain/collision (Pain strongly amplifies salience)
        if collision_flags is not None:
            # collision_flags: 1.0 = collision, 0.0 = clear
            pain_multiplier = 1.0 + (collision_flags.unsqueeze(-1) * 5.0) 
            part_features = part_features * pain_multiplier
            
        # 2. Build the Somatotopic Map
        body_schema = self.somato_map(part_features)
        
        # 3. Calculate Bid for Global Workspace
        flat_schema = body_schema.view(B, -1)
        bid_tensor = self.bid_estimator(flat_schema)
        
        # If there are hard collisions, we guarantee a high baseline bid (Pain Reflex)
        base_bid = bid_tensor.mean().item()
        if collision_flags is not None and collision_flags.max() > 0.1:
            base_bid = max(base_bid, 0.85) # Pain forces its way into consciousness
        
        # Cache for reentrant feedback
        self._last_bid = base_bid
            
        return body_schema, base_bid
    
    def receive_broadcast(self, broadcast_content: Any, current_bid: float) -> float:
        """
        Receive top-down feedback from the workspace (Phase 6 Reentrant Processing).
        
        Body awareness increases when the broadcast contains body-relevant 
        content (e.g., pain or collision keywords). Otherwise body salience 
        decays slightly during reentrant cycles (the brain doesn't constantly 
        attend to the body unless something is wrong).
        
        Args:
            broadcast_content: The current workspace broadcast
            current_bid: Our current bid value
            
        Returns:
            Updated bid value
        """
        # Check if broadcast content relates to the body
        body_relevant = False
        if isinstance(broadcast_content, dict):
            body_relevant = any(
                k in broadcast_content for k in ['body', 'collision', 'pain', 'Physical']
            )
            # Also check string values
            for v in broadcast_content.values():
                if isinstance(v, str) and any(kw in v.lower() for kw in ['body', 'pain', 'physical', 'collision']):
                    body_relevant = True
                    break
        elif isinstance(broadcast_content, str):
            body_relevant = any(kw in broadcast_content.lower() for kw in ['body', 'pain', 'physical'])
        
        if body_relevant:
            # Body is in the spotlight. Maintain or boost bid.
            return min(1.0, current_bid * 1.05)
        else:
            # Body not in focus. Slight decay (background awareness).
            return max(0.05, current_bid * 0.9)  # Never fully zero (always some body awareness)
