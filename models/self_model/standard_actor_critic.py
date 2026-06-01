"""
Standard advantage actor-critic (A2C) on the workspace broadcast.

A P5 diagnostic alternative to the Go/No-Go `ActionSelectionCore`, with the same
call interface (`pfc_hidden` attribute, `select_action`, `step`, `update_policy`,
`replay_update`) so `train_rlhf.py` can swap it in via `--policy standard`.

Purpose: isolate whether the competence bottleneck is the Go/No-Go policy + its
stylized go/no-go/STN loss, or the broadcast representation. By training a plain,
known-working RL head on the SAME broadcast, a large gap in either direction is
informative:
  - standard >> gonogo  => the Go/No-Go policy is the bottleneck (and this is a
    prototype fix).
  - standard ~= gonogo and both far below the DQN-on-pixels baseline => the
    broadcast representation is the bottleneck.

This is a diagnostic, not a default. Gaussian continuous policy + value baseline,
standard A2C loss (policy = -logprob * advantage, value = MSE to discounted
returns, small entropy bonus).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore


class StandardActorCritic:
    def __init__(self, config: dict, emotion_shaper: EmotionalRewardShaper, memory: MemoryCore):
        self.config = config
        self.emotion_shaper = emotion_shaper
        self.memory = memory
        self.workspace_dim = config.get("workspace_dim", 256)
        self.action_dim = config.get("action_dim", 4)
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.device = config.get("device", "cpu")
        hidden = config.get("ac_hidden", 128)

        self.body = nn.Sequential(
            nn.Linear(self.workspace_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        ).to(self.device)
        self.mean_head = nn.Linear(hidden, self.action_dim).to(self.device)
        self.value_head = nn.Linear(hidden, 1).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))

        params = (list(self.body.parameters()) + list(self.mean_head.parameters())
                  + list(self.value_head.parameters()) + [self.log_std])
        self.optimizer = optim.Adam(params, lr=self.lr)

        self.pfc_hidden = None  # interface compatibility (unused)
        self.rollout_buffer: list[dict] = []
        self.last_value = 0.0

    # --- interface parity with ActionSelectionCore ---

    def reset_state(self, batch_size: int = 1) -> None:
        self.pfc_hidden = None
        self.last_value = 0.0

    def _features(self, broadcast: torch.Tensor) -> torch.Tensor:
        if broadcast.dim() == 1:
            broadcast = broadcast.unsqueeze(0)
        return self.body(broadcast.to(self.device))

    def select_action(self, workspace_broadcast: torch.Tensor, emotion_arousal: float = 0.5,
                      rpe_cache: float = 0.0, self_vector: torch.Tensor | None = None):
        with torch.no_grad():
            h = self._features(workspace_broadcast)
            mean = torch.tanh(self.mean_head(h))
            std = torch.exp(self.log_std).clamp(0.05, 1.0)
            # Arousal scales exploration, mirroring the Go/No-Go core's behaviour.
            std = std * max(0.5, float(emotion_arousal) * 2.0)
            action = torch.normal(mean, std).clamp(-1.0, 1.0)
            self.last_value = float(self.value_head(h).item())
        return action.squeeze(0).cpu().numpy(), self.last_value

    def step(self, workspace_broadcast: torch.Tensor, action: np.ndarray, raw_reward: float,
             next_broadcast: torch.Tensor, done: bool, emotion_state: dict[str, float],
             attention_level: float, narrative: str = "",
             self_vector: torch.Tensor | None = None,
             next_self_vector: torch.Tensor | None = None) -> dict[str, float]:
        shaped = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_state, base_reward=raw_reward,
            context={"adaptation_detected": False})
        if workspace_broadcast.dim() == 1:
            workspace_broadcast = workspace_broadcast.unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float, device=self.device)
        self.memory.store_experience(
            state=workspace_broadcast.squeeze(0), action=action_t, reward=shaped,
            emotion_values=emotion_state, attention_level=attention_level, narrative=narrative)
        self.rollout_buffer.append({
            "state": workspace_broadcast.detach(),
            "action": action_t,
            "reward": shaped,
            "done": done,
        })
        return {"raw_reward": raw_reward, "shaped_reward": shaped, "dopamine_rpe": 0.0}

    def _a2c_loss(self, states: torch.Tensor, actions: torch.Tensor,
                  returns: torch.Tensor, entropy_coef: float):
        h = self.body(states)
        mean = torch.tanh(self.mean_head(h))
        std = torch.exp(self.log_std).clamp(0.05, 1.0)
        values = self.value_head(h)
        dist = torch.distributions.Normal(mean, std)
        logprob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
        advantage = (returns - values).detach()
        policy_loss = -(logprob * advantage).mean()
        value_loss = nn.MSELoss()(values, returns)
        return policy_loss, value_loss, entropy

    def update_policy(self) -> dict[str, float]:
        if len(self.rollout_buffer) < 10:
            return {}
        states = torch.cat([x["state"] for x in self.rollout_buffer], dim=0).to(self.device)
        actions = torch.stack([x["action"] for x in self.rollout_buffer]).to(self.device)
        rewards = [x["reward"] for x in self.rollout_buffer]
        dones = [x["done"] for x in self.rollout_buffer]
        returns_list: list[float] = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns_list.insert(0, R)
        returns = torch.tensor(returns_list, dtype=torch.float, device=self.device).unsqueeze(1)

        policy_loss, value_loss, entropy = self._a2c_loss(states, actions, returns, 0.01)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.rollout_buffer = []
        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "total_loss": float(loss.item())}

    def replay_update(self, experiences: list[dict]) -> dict[str, float]:
        valid = [e for e in experiences if "state" in e and "action" in e and "reward" in e]
        if len(valid) < 4:
            return {}
        states = []
        for e in valid:
            s = torch.as_tensor(e["state"], dtype=torch.float, device=self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            states.append(s)
        states = torch.cat(states, dim=0)
        actions = torch.stack([
            torch.as_tensor(e["action"], dtype=torch.float, device=self.device) for e in valid
        ])
        rewards = [float(e["reward"]) for e in valid]
        returns_list: list[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns_list.insert(0, R)
        returns = torch.tensor(returns_list, dtype=torch.float, device=self.device).unsqueeze(1)
        policy_loss, value_loss, _ = self._a2c_loss(states, actions, returns, 0.0)
        loss = 0.5 * (policy_loss + 0.5 * value_loss)  # scaled down like the Go/No-Go replay
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"replay_total_loss": float(loss.item())}
