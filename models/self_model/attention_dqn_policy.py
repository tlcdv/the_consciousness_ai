"""
Attention-based DQN policy for DMTS non-local comparison.

STATUS (2026-07-19): UNVALIDATED. Committed so the `--policy attention-dqn` wiring in
train_rlhf.py resolves its import, NOT because it has been shown to work. It runs end to
end (verified: 60-step DMTS episode, no errors) and nothing more. There is no unit test
coverage, no verdict doc, and no measured result.

Three runs exist on disk (`runs/c1_attention`, `_seed43`, `_seed44`, 4 episodes each).
Their rewards are NOT a valid comparison against any other arm, because the arms differ in
flags. Do not cite them as evidence that attention helps. A real test needs a matched
baseline (`--policy dqn`, identical flags and seeds) run alongside.

The obsmem-conv input is [current_obs_map (64, 16, 16); held_sample (64, 16, 16)]
stacked on channel axis -> [128, 16, 16].

The attention head compares the held sample against the 4 choice positions
in the current frame, enabling the non-local match decision that the conv
head cannot express (read-only ceiling: PCA+MLP 0.786 vs conv 0.655).
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore


class AttentionDQNPolicy:
    """
    DQN with spatial attention front-end for DMTS match-to-sample.
    
    Input: [128, 16, 16] = [current(64), sample(64), H, W]
    4 choice positions on 16x16 grid (image 224x224, stride 14):
      action 0 (left):  grid (8, 2)  <- pixel (38, 112) / 14
      action 1 (right): grid (8, 13) <- pixel (186, 112) / 14
      action 2 (up):    grid (2, 8)  <- pixel (112, 38) / 14
      action 3 (down):  grid (13, 8) <- pixel (112, 186) / 14
    
    Architecture:
    1. Project sample (64ch) -> query vector [d_model]
    2. Extract 4 patches at choice positions from current (64ch) -> 4 key vectors [d_model]
    3. Cross-attention: similarity(query, key_i) for i=1..4
    4. MLP on 4 similarities -> 4 Q-values
    
    This directly implements the non-local comparison the read-only diagnosis
    showed is necessary (conv cannot express it).
    """

    # (row, col) on 16x16 grid for the 4 choice positions
    CHOICE_GRID_POS = [
        (8, 2),   # action 0: left
        (8, 13),  # action 1: right
        (2, 8),   # action 2: up
        (13, 8),  # action 3: down
    ]

    def __init__(self, config: dict, emotion_shaper: EmotionalRewardShaper, memory: MemoryCore):
        self.config = config
        self.emotion_shaper = emotion_shaper
        self.memory = memory
        
        # Input shape from config (set by train_rlhf.py for obsmem-conv)
        spatial_shape = config.get("policy_spatial_shape", (128, 16, 16))
        self.in_channels = spatial_shape[0]       # 128 (current + sample)
        self.grid_h = spatial_shape[1]             # 16
        self.grid_w = spatial_shape[2]             # 16
        self.sample_ch = self.in_channels // 2     # 64
        self.current_ch = self.sample_ch           # 64
        
        self.action_dim = config.get("action_dim", 4)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-3)
        
        # Attention architecture hyperparameters
        d_model = config.get("attn_d_model", 128)
        d_ff = config.get("attn_d_ff", 256)
        patch_size = config.get("attn_patch_size", 3)  # 3x3 patch around choice position
        
        # Project sample (global pooled) to query
        self.sample_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global pool [64, 1, 1]
            nn.Flatten(),              # [64]
            nn.Linear(self.sample_ch, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        ).to(self.device)
        
        # Extract patches at 4 choice positions from current frame
        patch_dim = self.current_ch * patch_size * patch_size
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        ).to(self.device)
        
        # Cross-attention scaling
        self.attn_scale = d_model ** -0.5
        
        # Final Q-head from 4 attention scores
        self.q_head = nn.Sequential(
            nn.Linear(4, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, self.action_dim),
        ).to(self.device)
        
        # Target network (same architecture, copied weights)
        self.target_sample_proj = self._make_sample_proj(d_model)
        self.target_patch_proj = self._make_patch_proj(d_model, patch_dim)
        self.target_q_head = self._make_q_head(d_ff)
        self.target_sample_proj.load_state_dict(self.sample_proj.state_dict())
        self.target_patch_proj.load_state_dict(self.patch_proj.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())
        self.target_sample_proj.eval()
        self.target_patch_proj.eval()
        self.target_q_head.eval()
        
        # Optimizer
        all_params = (list(self.sample_proj.parameters()) +
                      list(self.patch_proj.parameters()) +
                      list(self.q_head.parameters()))
        self.optimizer = torch.optim.Adam(all_params, lr=self.lr)
        
        # DQN params
        self.batch_size = config.get("dqn_batch_size", 32)
        self.target_update = config.get("dqn_target_update", 200)
        self.eps_start = config.get("dqn_epsilon_start", 1.0)
        self.eps_end = config.get("dqn_epsilon_end", 0.05)
        self.eps_decay_steps = config.get("dqn_epsilon_decay_steps", 50000)
        
        self.buffer: deque = deque(maxlen=config.get("dqn_buffer", 10000))
        
        self.pfc_hidden = None
        self.act_steps = 0
        self.train_steps = 0
        self.last_action_idx = 0
        self.last_loss = 0.0
        
        # Patch extraction
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

    def _get_state_dict(self) -> dict:
        return {
            "sample_proj": self.sample_proj.state_dict(),
            "patch_proj": self.patch_proj.state_dict(),
            "q_head": self.q_head.state_dict(),
        }

    def _make_sample_proj(self, d_model: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.sample_ch, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        ).to(self.device)

    def _make_patch_proj(self, d_model: int, patch_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        ).to(self.device)

    def _make_q_head(self, d_ff: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, self.action_dim),
        ).to(self.device)

    def _update_target(self) -> None:
        self.target_sample_proj.load_state_dict(self.sample_proj.state_dict())
        self.target_patch_proj.load_state_dict(self.patch_proj.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())

    def _update_target(self) -> None:
        self.target_sample_proj.load_state_dict(self.sample_proj.state_dict())
        self.target_patch_proj.load_state_dict(self.patch_proj.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())

    def reset_state(self, batch_size: int = 1) -> None:
        self.pfc_hidden = None

    def _epsilon(self) -> float:
        frac = max(0.0, 1.0 - self.act_steps / max(1, self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * frac

    def _idx_to_action(self, idx: int) -> np.ndarray:
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[idx] = 1.0
        return onehot

    def _extract_patches(self, current: torch.Tensor) -> torch.Tensor:
        """
        Extract 3x3 patches at the 4 choice positions from current frame.
        
        Args:
            current: [B, 64, 16, 16] - current obs_map
        Returns:
            patches: [B, 4, 64*9] - flattened patches
        """
        B, C, H, W = current.shape
        patches = []
        for gy, gx in self.CHOICE_GRID_POS:
            # Clamp to valid range
            gy = max(self.half_patch, min(H - 1 - self.half_patch, gy))
            gx = max(self.half_patch, min(W - 1 - self.half_patch, gx))
            
            y1 = gy - self.half_patch
            y2 = gy + self.half_patch + 1
            x1 = gx - self.half_patch
            x2 = gx + self.half_patch + 1
            
            patch = current[:, :, y1:y2, x1:x2]  # [B, C, 3, 3]
            patches.append(patch.flatten(1))      # [B, C*9]
        
        return torch.stack(patches, dim=1)  # [B, 4, C*9]

    def forward_q(self, spatial_input: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values from spatial input [B, 128, 16, 16].
        
        Returns: [B, 4] Q-values for 4 actions
        """
        B = spatial_input.shape[0]
        
        # Split: current [64, 16, 16], sample [64, 16, 16]
        current = spatial_input[:, :self.current_ch]   # [B, 64, 16, 16]
        sample = spatial_input[:, self.current_ch:]     # [B, 64, 16, 16]
        
        # Sample -> query vector [B, d_model]
        query = self.sample_proj(sample)  # [B, d_model]
        
        # Current -> 4 patch keys [B, 4, d_model]
        patches = self._extract_patches(current)        # [B, 4, 64*9]
        keys = self.patch_proj(patches)                 # [B, 4, d_model]
        
        # Cross-attention: similarity between query and each key
        # query: [B, d_model] -> [B, 1, d_model]
        # keys:  [B, 4, d_model]
        attn_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attn_scale  # [B, 1, 4]
        attn_scores = attn_scores.squeeze(1)  # [B, 4]
        
        # Q-head from 4 attention scores
        q = self.q_head(attn_scores)  # [B, 4]
        
        return q

    def _target_forward_q(self, spatial_input: torch.Tensor) -> torch.Tensor:
        """Forward pass using target network."""
        B = spatial_input.shape[0]
        current = spatial_input[:, :self.current_ch]
        sample = spatial_input[:, self.current_ch:]
        
        query = self.target_sample_proj(sample)
        patches = self._extract_patches(current)
        keys = self.target_patch_proj(patches)
        
        attn_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.attn_scale
        attn_scores = attn_scores.squeeze(1)
        
        q = self.target_q_head(attn_scores)
        return q

    def select_action(self, workspace_broadcast: torch.Tensor, emotion_arousal: float = 0.5,
                      rpe_cache: float = 0.0, self_vector: Optional[torch.Tensor] = None):
        """
        workspace_broadcast is the flattened input [1, 128*16*16] for obsmem-conv.
        We reshape to [1, 128, 16, 16] internally.
        """
        self.act_steps += 1
        
        # Reshape flat input to spatial [1, 128, 16, 16]
        if workspace_broadcast.dim() == 2 and workspace_broadcast.shape[1] == self.in_channels * self.grid_h * self.grid_w:
            spatial_input = workspace_broadcast.view(1, self.in_channels, self.grid_h, self.grid_w)
        elif workspace_broadcast.dim() == 3:
            spatial_input = workspace_broadcast.unsqueeze(0)
        elif workspace_broadcast.dim() == 4:
            spatial_input = workspace_broadcast
        else:
            raise ValueError(f"Unexpected input shape: {workspace_broadcast.shape}")
        spatial_input = spatial_input.detach().to(self.device)
        
        with torch.no_grad():
            q = self.forward_q(spatial_input)
            value = float(q.max(dim=1)[0].item())
        
        if random.random() < self._epsilon():
            idx = random.randrange(self.action_dim)
        else:
            idx = int(q.argmax(dim=1).item())
        
        self.last_action_idx = idx
        return self._idx_to_action(idx), value

    def _train_batch(self) -> float:
        if len(self.buffer) < self.batch_size:
            return self.last_loss
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        s = torch.stack(states).to(self.device)
        ns = torch.stack(next_states).to(self.device)
        a = torch.tensor(actions, dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        q = self.forward_q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self._target_forward_q(ns).max(1)[0]
            target = r + self.gamma * next_q * (1.0 - d)
        
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self._update_target()
        
        self.last_loss = float(loss.item())
        return self.last_loss

    def step(self, workspace_broadcast: torch.Tensor, action: np.ndarray, raw_reward: float,
             next_broadcast: torch.Tensor, done: bool, emotion_state: dict[str, float],
             attention_level: float, narrative: str = "",
             self_vector: Optional[torch.Tensor] = None,
             next_self_vector: Optional[torch.Tensor] = None) -> dict[str, float]:
        shaped = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_state, base_reward=raw_reward,
            context={"adaptation_detected": False})
        
        # Convert flat input to spatial [C, H, W] for buffer storage
        def _to_spatial(x):
            if x.dim() == 2 and x.shape[1] == self.in_channels * self.grid_h * self.grid_w:
                return x.view(self.in_channels, self.grid_h, self.grid_w)
            elif x.dim() == 3:
                return x
            elif x.dim() == 4:
                return x.squeeze(0)
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        s = _to_spatial(workspace_broadcast).detach().cpu()
        ns = _to_spatial(next_broadcast).detach().cpu()
        action_t = torch.tensor(action, dtype=torch.float, device=self.device)
        
        self.memory.store_experience(
            state=s.flatten(), action=action_t, reward=shaped,
            emotion_values=emotion_state, attention_level=attention_level, narrative=narrative)
        
        self.buffer.append((s, self.last_action_idx, shaped, ns, float(done)))
        self._train_batch()
        
        return {"raw_reward": raw_reward, "shaped_reward": shaped, "dopamine_rpe": 0.0}

    def update_policy(self) -> dict[str, float]:
        return {"total_loss": float(self.last_loss)}

    def replay_update(self, experiences: list[dict]) -> dict[str, float]:
        valid = [e for e in experiences if "state" in e and "reward" in e]
        if len(valid) < 4:
            return {}
        states = []
        for e in valid:
            t = torch.as_tensor(e["state"], dtype=torch.float, device=self.device)
            states.append(t.view(-1))
        s = torch.stack(states)
        expected_dim = self.in_channels * self.grid_h * self.grid_w
        if s.shape[1] != expected_dim:
            return {}
        rewards = torch.tensor([float(e["reward"]) for e in valid],
                               dtype=torch.float32, device=self.device)
        q = self.forward_q(s.view(-1, self.in_channels, self.grid_h, self.grid_w))
        target = q.detach().clone()
        max_idx = q.argmax(dim=1)
        target[torch.arange(q.shape[0]), max_idx] = rewards
        loss = 0.5 * F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"replay_total_loss": float(loss.item())}


# For train_rlhf.py to import
__all__ = ["AttentionDQNPolicy"]