"""
DQN policy on the workspace broadcast (P5 confirmation tool).

Same call interface as the Go/No-Go `ActionSelectionCore` and the `StandardActorCritic`
(`pfc_hidden` attr, `select_action`, `step`, `update_policy`, `replay_update`,
`reset_state`) so `train_rlhf.py` can swap it in via `--policy dqn`.

Purpose: the competence diagnosis (docs/results/agent_competence_diagnosis_2026_06_01.md)
showed two on-policy learners (Go/No-Go, A2C) tie on the broadcast at ~15, while an
off-policy DQN on raw pixels reaches 92. The one remaining confound is learner
FAMILY (on-policy vs off-policy), not the input. This runs the SAME family (DQN)
on the broadcast to isolate it:
  - DQN-on-broadcast ~= 15 (and << DQN-on-pixels 92) => the broadcast representation
    is the bottleneck, learner-independent.
  - DQN-on-broadcast approaches 92 => the prior on-policy learners were the limit,
    not the representation.

Mirrors the Q-learning core of scripts/training/train_baseline_dqn.py (MLP Q-net +
target net + replay buffer + epsilon-greedy), but consumes the 256-D broadcast
instead of pixels. Continuous environments (dark_room, navigation) use the same
9-bin discretization the pixel baseline uses; discrete environments (DMTS, WCST)
map one Q-output per action. This is a diagnostic, not a default.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore


class DQNPolicy:
    def __init__(self, config: dict, emotion_shaper: EmotionalRewardShaper, memory: MemoryCore):
        self.config = config
        self.emotion_shaper = emotion_shaper
        self.memory = memory
        self.workspace_dim = config.get("workspace_dim", 256)
        # The policy's input dim. Defaults to workspace_dim (broadcast/tectum taps),
        # but the spatial localization tap is much larger, so it can be overridden.
        self.input_dim = config.get("policy_input_dim", self.workspace_dim)
        self.action_dim = config.get("action_dim", 4)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-3)
        hidden = config.get("dqn_hidden", 128)

        # Continuous envs (dark_room, navigation) discretize into 9 bins (3x3),
        # matching the pixel DQN baseline. Discrete envs use one Q per action.
        self.continuous = config.get("env_continuous", True)
        self.n_actions = 9 if self.continuous else self.action_dim

        self.batch_size = config.get("dqn_batch_size", 32)
        self.target_update = config.get("dqn_target_update", 200)
        self.eps_start = config.get("dqn_epsilon_start", 1.0)
        self.eps_end = config.get("dqn_epsilon_end", 0.05)
        self.eps_decay_steps = config.get("dqn_epsilon_decay_steps", 50000)

        def _qnet() -> nn.Module:
            return nn.Sequential(
                nn.Linear(self.input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, self.n_actions),
            ).to(self.device)

        self.qnet = _qnet()
        self.target_net = _qnet()
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.buffer: deque = deque(maxlen=config.get("dqn_buffer", 10000))

        self.pfc_hidden = None  # interface compatibility (unused)
        self.act_steps = 0
        self.train_steps = 0
        self.last_action_idx = 0
        self.last_loss = 0.0

    # --- interface parity ---

    def reset_state(self, batch_size: int = 1) -> None:
        self.pfc_hidden = None

    def _epsilon(self) -> float:
        frac = max(0.0, 1.0 - self.act_steps / max(1, self.eps_decay_steps))
        return self.eps_end + (self.eps_start - self.eps_end) * frac

    def _idx_to_action(self, idx: int) -> np.ndarray:
        """Discrete bin -> the action vector train_rlhf expects."""
        if self.continuous:
            vals = [-1.0, 0.0, 1.0]
            return np.array([vals[idx // 3], vals[idx % 3]], dtype=np.float32)
        onehot = np.zeros(self.action_dim, dtype=np.float32)
        onehot[idx] = 1.0
        return onehot

    def _flat(self, broadcast: torch.Tensor) -> torch.Tensor:
        if broadcast.dim() == 1:
            broadcast = broadcast.unsqueeze(0)
        return broadcast.detach().to(self.device)

    def select_action(self, workspace_broadcast: torch.Tensor, emotion_arousal: float = 0.5,
                      rpe_cache: float = 0.0, self_vector: torch.Tensor | None = None):
        self.act_steps += 1
        b = self._flat(workspace_broadcast)
        with torch.no_grad():
            q = self.qnet(b)
            value = float(q.max(dim=1)[0].item())
        if random.random() < self._epsilon():
            idx = random.randrange(self.n_actions)
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

        q = self.qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target = r + self.gamma * next_q * (1.0 - d)
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def step(self, workspace_broadcast: torch.Tensor, action: np.ndarray, raw_reward: float,
             next_broadcast: torch.Tensor, done: bool, emotion_state: dict[str, float],
             attention_level: float, narrative: str = "",
             self_vector: torch.Tensor | None = None,
             next_self_vector: torch.Tensor | None = None) -> dict[str, float]:
        shaped = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_state, base_reward=raw_reward,
            context={"adaptation_detected": False})
        s = self._flat(workspace_broadcast).squeeze(0).cpu()
        ns = self._flat(next_broadcast).squeeze(0).cpu()
        action_t = torch.tensor(action, dtype=torch.float, device=self.device)
        self.memory.store_experience(
            state=self._flat(workspace_broadcast).squeeze(0), action=action_t, reward=shaped,
            emotion_values=emotion_state, attention_level=attention_level, narrative=narrative)
        self.buffer.append((s, self.last_action_idx, shaped, ns, float(done)))
        # Per-step Q-learning, matching the pixel baseline's training cadence.
        self._train_batch()
        return {"raw_reward": raw_reward, "shaped_reward": shaped, "dopamine_rpe": 0.0}

    def update_policy(self) -> dict[str, float]:
        # DQN trains every step inside step(); expose the latest loss for parity.
        return {"total_loss": float(self.last_loss)}

    def replay_update(self, experiences: list[dict]) -> dict[str, float]:
        """Crude memory replay: regress Q(s, a_taken) toward the stored reward
        (treated as a terminal target). Keeps the off-policy learner using the
        phi-prioritized replay batch like the other policies, without needing
        next-state in the stored entries."""
        valid = [e for e in experiences if "state" in e and "reward" in e]
        if len(valid) < 4:
            return {}
        states = []
        for e in valid:
            t = torch.as_tensor(e["state"], dtype=torch.float, device=self.device)
            states.append(t.view(-1))
        s = torch.stack(states)
        if s.shape[1] != self.input_dim:  # raw memory states may differ; skip
            return {}
        rewards = torch.tensor([float(e["reward"]) for e in valid],
                               dtype=torch.float32, device=self.device)
        q = self.qnet(s)
        target = q.detach().clone()
        # Regress the max-Q action toward the observed reward (terminal bootstrap).
        max_idx = q.argmax(dim=1)
        target[torch.arange(q.shape[0]), max_idx] = rewards
        loss = 0.5 * F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"replay_total_loss": float(loss.item())}
