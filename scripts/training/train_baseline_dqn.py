"""
Vanilla DQN baseline for comparison with the consciousness agent.

Minimal implementation: CNN encoder + MLP Q-network, epsilon-greedy,
replay buffer. No consciousness machinery.

Usage:
    python -m scripts.training.train_baseline_dqn --env dark_room --episodes 100
    python -m scripts.training.train_baseline_dqn --env dmts --episodes 500
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """3-layer CNN + 2-layer MLP Q-network."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        # Compute conv output size for 224x224 input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            conv_out = self.conv(dummy).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x).flatten(1)
        return self.fc(features)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert [H, W, 3] uint8 to [1, 3, H, W] float."""
    t = torch.from_numpy(frame).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def make_env(env_name: str, render_mode: str, difficulty: int = 0):
    if env_name == "dark_room":
        from simulations.environments.simple_visual_env import SimpleVisualEnv
        return SimpleVisualEnv(render_mode=render_mode, width=224, height=224), 2, True
    elif env_name == "navigation":
        from simulations.environments.navigation_env import NavigationEnv
        return NavigationEnv(render_mode=render_mode, width=224, height=224), 2, True
    elif env_name == "dmts":
        from simulations.environments.dmts_env import DMTSEnv
        return DMTSEnv(render_mode=render_mode, width=224, height=224,
                       distractor_overlap=difficulty), 5, False
    elif env_name == "wcst":
        from simulations.environments.wcst_env import WCSTEnv
        return WCSTEnv(render_mode=render_mode, width=224, height=224), 4, False
    else:
        raise ValueError(f"Unknown env: {env_name}")


def main():
    parser = argparse.ArgumentParser(description="DQN baseline agent")
    parser.add_argument("--env", type=str, default="dark_room",
                        choices=["dark_room", "navigation", "dmts", "wcst"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=500)
    parser.add_argument("--log-dir", type=str, default="runs_baseline")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    render_mode = "human" if args.render else "rgb_array"
    env, action_dim, is_continuous = make_env(args.env, render_mode, args.difficulty)

    # For continuous action spaces, discretize into 9 bins (3x3 grid)
    if is_continuous:
        action_dim = 9  # 3 levels for each of 2 dims

    device = "cuda" if torch.cuda.is_available() else "cpu"
    qnet = DQN(action_dim).to(device)
    target_net = DQN(action_dim).to(device)
    target_net.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    buffer = ReplayBuffer()

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, f"baseline_{args.env}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "steps", "epsilon"])

    def discrete_to_continuous(a: int) -> np.ndarray:
        """Convert discrete action to [move_x, move_y] in [-1, 1]."""
        vals = [-1.0, 0.0, 1.0]
        return np.array([vals[a // 3], vals[a % 3]], dtype=np.float32)

    global_step = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        state = frame_to_tensor(obs).to(device)
        total_reward = 0.0
        steps = 0

        epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
            max(0.0, 1.0 - ep / args.epsilon_decay)

        for step in range(args.max_steps):
            # Epsilon-greedy
            if random.random() < epsilon:
                action_idx = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    action_idx = qnet(state).argmax(dim=1).item()

            # Execute
            if is_continuous:
                env_action = discrete_to_continuous(action_idx)
            else:
                env_action = action_idx

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            next_state = frame_to_tensor(next_obs).to(device)

            buffer.push(state.cpu().squeeze(0), action_idx, reward,
                        next_state.cpu().squeeze(0), float(done))

            state = next_state
            total_reward += reward
            steps += 1
            global_step += 1

            # Train
            if len(buffer) >= args.batch_size:
                s, a, r, ns, d = buffer.sample(args.batch_size)
                s, ns = s.to(device), ns.to(device)
                r, d = r.to(device), d.to(device)

                q_vals = qnet(s).gather(1, a.unsqueeze(1).to(device)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns).max(1)[0]
                    target = r + 0.99 * next_q * (1.0 - d)

                loss = F.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if global_step % 200 == 0:
                target_net.load_state_dict(qnet.state_dict())

            if done:
                break

        csv_writer.writerow([ep, f"{total_reward:.3f}", steps, f"{epsilon:.4f}"])
        csv_file.flush()

        if (ep + 1) % 5 == 0:
            logger.info(f"Episode {ep+1}/{args.episodes} | reward={total_reward:.2f} | "
                        f"steps={steps} | eps={epsilon:.3f}")

    csv_file.close()
    env.close()
    logger.info(f"Training complete. Logs: {csv_path}")


if __name__ == "__main__":
    main()
