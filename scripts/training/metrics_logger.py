"""
Consciousness Metrics Logger

Logs step-level and episode-level metrics for evaluating the pre-registered
predictions (docs/preregistered_predictions.md). Supports TensorBoard when
available, with CSV fallback.

Tracked metrics per step:
  - Phi (proxy), AKOrN sync R, is_conscious, reward, PAD state,
    broadcast magnitude, gate state, workspace state

Tracked metrics per episode:
  - Total reward, steps, avg Phi, consciousness ratio, EI comparison

Insight moment detection uses the 4-criterion operational definition from
preregistered_predictions.md.
"""
from __future__ import annotations

import csv
import os
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


@dataclass
class StepMetrics:
    """Metrics collected at each training step."""
    global_step: int
    phi: float
    sync_r: float
    is_conscious: bool
    reward: float
    broadcast_mag: float
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    gate_state: tuple[float, ...] | None = None
    workspace_state: tuple[float, ...] | None = None


class ConsciousnessMetricsLogger:
    """
    Logs consciousness metrics to TensorBoard and/or CSV.

    Usage:
        logger = ConsciousnessMetricsLogger(log_dir="runs/exp1")
        logger.log_step(step_metrics)
        logger.log_episode(episode, total_reward, steps, avg_phi, consciousness_ratio)
        logger.compute_and_log_ei(episode, gate_trajectories, workspace_trajectories)
    """

    def __init__(self, log_dir: str = "runs", use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and _TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        else:
            logger.info("TensorBoard unavailable. Using CSV only.")

        # CSV fallback
        self._csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._init_csv()

        # Episode CSV
        self._ep_csv_path = os.path.join(log_dir, "episodes.csv")
        self._ep_csv_file = None
        self._ep_csv_writer = None
        self._init_episode_csv()

        # Insight detection state
        self._seen_state_actions: set[str] = set()
        self._cross_episode_rewards: deque[float] = deque(maxlen=500)
        self._episode_broadcast_mags: list[float] = []
        self._last_insight_step: int = -100
        self._global_insight_step: int = 0

        # Trajectory buffers for EI computation
        self._gate_trajectory: list[tuple[float, ...]] = []
        self._workspace_trajectory: list[tuple[float, ...]] = []

    def _init_csv(self):
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "global_step", "phi", "sync_r", "is_conscious", "reward",
            "broadcast_mag", "valence", "arousal", "dominance",
        ])

    def _init_episode_csv(self):
        self._ep_csv_file = open(self._ep_csv_path, "w", newline="")
        self._ep_csv_writer = csv.writer(self._ep_csv_file)
        self._ep_csv_writer.writerow([
            "episode", "total_reward", "steps", "avg_phi",
            "consciousness_ratio", "ei_gates", "ei_workspace", "ei_ratio",
        ])

    def log_step(self, metrics: StepMetrics):
        """Log a single training step."""
        step = metrics.global_step

        # CSV
        self._csv_writer.writerow([
            step, f"{metrics.phi:.6f}", f"{metrics.sync_r:.6f}",
            int(metrics.is_conscious), f"{metrics.reward:.6f}",
            f"{metrics.broadcast_mag:.6f}",
            f"{metrics.valence:.4f}", f"{metrics.arousal:.4f}",
            f"{metrics.dominance:.4f}",
        ])
        self._csv_file.flush()

        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("consciousness/phi", metrics.phi, step)
            self.writer.add_scalar("consciousness/sync_R", metrics.sync_r, step)
            self.writer.add_scalar("consciousness/is_conscious", int(metrics.is_conscious), step)
            self.writer.add_scalar("reward/step", metrics.reward, step)
            self.writer.add_scalar("consciousness/broadcast_mag", metrics.broadcast_mag, step)
            self.writer.add_scalar("emotion/valence", metrics.valence, step)
            self.writer.add_scalar("emotion/arousal", metrics.arousal, step)
            self.writer.add_scalar("emotion/dominance", metrics.dominance, step)

        # Buffer for insight detection
        self._cross_episode_rewards.append(metrics.reward)
        self._episode_broadcast_mags.append(metrics.broadcast_mag)
        self._global_insight_step += 1

        # Buffer for EI
        if metrics.gate_state is not None:
            self._gate_trajectory.append(metrics.gate_state)
        if metrics.workspace_state is not None:
            self._workspace_trajectory.append(metrics.workspace_state)

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        avg_phi: float,
        consciousness_ratio: float,
        ei_gates: float = 0.0,
        ei_workspace: float = 0.0,
        ei_ratio: float = 0.0,
    ):
        """Log episode-level summary."""
        # CSV
        self._ep_csv_writer.writerow([
            episode, f"{total_reward:.4f}", steps, f"{avg_phi:.6f}",
            f"{consciousness_ratio:.4f}",
            f"{ei_gates:.6f}", f"{ei_workspace:.6f}", f"{ei_ratio:.4f}",
        ])
        self._ep_csv_file.flush()

        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("episode/total_reward", total_reward, episode)
            self.writer.add_scalar("episode/steps", steps, episode)
            self.writer.add_scalar("episode/avg_phi", avg_phi, episode)
            self.writer.add_scalar("episode/consciousness_ratio", consciousness_ratio, episode)
            if ei_ratio > 0:
                self.writer.add_scalar("emergence/ei_gates", ei_gates, episode)
                self.writer.add_scalar("emergence/ei_workspace", ei_workspace, episode)
                self.writer.add_scalar("emergence/ei_ratio", ei_ratio, episode)

        # Reset per-episode buffers
        self._episode_broadcast_mags.clear()

    def compute_and_log_ei(self, episode: int, num_gate_states: int = 32,
                           num_workspace_states: int = 8) -> dict:
        """
        Compute EI at gate and workspace levels from buffered trajectories.

        Gate states use joint binning: each of the 5 gate dimensions is binned
        to 2 levels, giving 2^5 = 32 joint states. This preserves multivariate
        structure instead of collapsing to a scalar sum.

        Returns dict with ei_gates, ei_workspace, ratio, emergent.
        """
        from models.evaluation.effective_information import (
            compute_effective_information,
            discretize_continuous,
        )

        result = {"ei_gates": 0.0, "ei_workspace": 0.0, "ratio": 0.0, "emergent": False}

        if len(self._gate_trajectory) < 10:
            return result

        # Joint binning for gate trajectories: each dimension -> 2 bins, combined as sum(b_i * 2^i)
        gate_discrete = []
        for g in self._gate_trajectory:
            joint_idx = 0
            for i, val in enumerate(g):
                bit = 1 if val >= 0.5 else 0
                joint_idx += bit * (2 ** i)
            gate_discrete.append(joint_idx)

        ws_flat = [sum(w) for w in self._workspace_trajectory] if self._workspace_trajectory else [0.0] * len(gate_discrete)
        ws_discrete = discretize_continuous(ws_flat, num_workspace_states)

        ei_gates = compute_effective_information(
            [np.array(gate_discrete)], num_gate_states
        )
        ei_workspace = compute_effective_information(
            [np.array(ws_discrete)], num_workspace_states
        )

        ratio = ei_workspace / max(ei_gates, 1e-8)
        emergent = ei_workspace > ei_gates

        result = {
            "ei_gates": ei_gates,
            "ei_workspace": ei_workspace,
            "ratio": ratio,
            "emergent": emergent,
        }

        if self.writer is not None:
            self.writer.add_scalar("emergence/ei_gates", ei_gates, episode)
            self.writer.add_scalar("emergence/ei_workspace", ei_workspace, episode)
            self.writer.add_scalar("emergence/ei_ratio", ratio, episode)
            self.writer.add_scalar("emergence/emergent", int(emergent), episode)

        # Clear trajectory buffers for next window
        self._gate_trajectory.clear()
        self._workspace_trajectory.clear()

        return result

    def detect_insight_moment(
        self,
        state_hash: str,
        action: int | str,
        reward: float,
        broadcast_mag: float,
    ) -> bool:
        """
        Detect an insight moment using the 4-criterion operational definition.

        1. Novel state-action pair
        2. Reward >= 1.5x running average (with minimum absolute threshold)
        3. First attempt in this state (same as criterion 1 for hash-based)
        4. Broadcast magnitude above 75th percentile
        """
        # Cooldown: skip if an insight was detected too recently (50 steps minimum gap)
        if self._last_insight_step >= 0 and (self._global_insight_step - self._last_insight_step) < 50:
            return False

        sa_key = f"{state_hash}_{action}"

        # Criterion 1 & 3: novel state-action pair
        is_novel = sa_key not in self._seen_state_actions
        self._seen_state_actions.add(sa_key)

        if not is_novel:
            return False

        # Criterion 2: reward jump with minimum absolute threshold
        # Require reward > 0.5 AND >= 2x running average (positive portion)
        if reward < 0.5:
            return False

        if len(self._cross_episode_rewards) >= 200:
            positive_rewards = [r for r in self._cross_episode_rewards if r > 0]
            if positive_rewards:
                avg_positive = np.mean(positive_rewards)
                reward_jump = reward >= 2.0 * avg_positive
            else:
                reward_jump = True  # first positive reward ever
        else:
            # Not enough data to establish baseline
            return False

        if not reward_jump:
            return False

        # Criterion 4: high broadcast magnitude (above 75th percentile)
        if len(self._episode_broadcast_mags) >= 10:
            threshold = np.percentile(self._episode_broadcast_mags, 75)
            high_broadcast = broadcast_mag >= threshold
        else:
            return False

        if high_broadcast:
            self._last_insight_step = self._global_insight_step

        return high_broadcast

    def reset_episode_state(self):
        """Reset per-episode tracking (call at start of each episode).

        Cross-episode rewards are preserved to maintain a stable baseline
        for insight detection. Only per-episode state-action novelty and
        broadcast magnitude buffers are cleared.
        """
        self._episode_broadcast_mags.clear()
        self._seen_state_actions.clear()

    def close(self):
        """Flush and close all writers."""
        if self.writer is not None:
            self.writer.close()
        if self._csv_file is not None:
            self._csv_file.close()
        if self._ep_csv_file is not None:
            self._ep_csv_file.close()
