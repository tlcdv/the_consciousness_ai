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
    # Which phi computation produced the value: "pyphi" (exact),
    # "proxy" (unvalidated geometric heuristic), "insufficient_data"
    # (early TPM, returns 0.0), or "" when not produced via the gate
    # pathway. Logged so post-hoc analysis can tell whether a phi
    # value is scientifically grounded.
    phi_method: str = ""
    # Parallel phi from the RIIU pathway (sliding-window SVD residual).
    # Computed alongside the pyphi value when --enable-riiu is on, zero
    # otherwise. Lets the analysis script compare both phi pathways on
    # the same trajectory. See docs/decisions/2026_05_16_riiu_license.md.
    #
    # `phi_riiu` is whichever substrate the --riiu-source flag picked as
    # the reward source (backward-compat alias). When --riiu-probe-all is
    # on, all three explicit per-substrate fields below carry the value
    # from their substrate, regardless of which one drives reward.
    phi_riiu: float = 0.0
    phi_riiu_broadcast: float = 0.0
    phi_riiu_tectum: float = 0.0
    phi_riiu_audio: float = 0.0
    # Levin consciousness metrics (Rouleau-Levin theme set). Computed when
    # --enable-levin-metrics is on, zero otherwise. These are diagnostic
    # measurements (the holonic/bioelectric modules run in inference mode and
    # are NOT part of the policy gradient); they are the baseline apparatus for
    # Phase 5's substrate-independence falsification test. goal_directed is 0.0
    # in this baseline until goal/outcome embeddings are defined at that test's
    # pre-registration. See models/evaluation/levin_consciousness_metrics.py.
    levin_bioelectric_complexity: float = 0.0
    levin_morphological_adaptation: float = 0.0
    levin_collective_intelligence: float = 0.0
    levin_goal_directed: float = 0.0
    levin_basal_cognition: float = 0.0
    # Phase 5 deliverable 1 self-vector loop. self_pred_mse is the one-step
    # self-prediction error of the learned self-model; self_pred_skill is the
    # forecasting skill score vs a persistence baseline (1 - mse/persistence,
    # clamped to [-1, 1]). > 0 means the self-model predicts its own next state
    # better than "no change". Zero when --enable-self-vector is off.
    self_pred_mse: float = 0.0
    self_pred_skill: float = 0.0
    # Perception fix: current-frame reconstruction MSE off tectum_content. Zero
    # when --enable-recon is off. A falling trajectory means the reconstruction
    # objective is training the tectum to retain stimulus identity.
    recon_loss: float = 0.0


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

        # Env-specific per-episode CSV (e.g. WCST recovery: rule_changes,
        # trials_correct). Lazily created on the first log_env_episode call so
        # non-WCST runs do not write an empty file.
        self._env_ep_csv_path = os.path.join(log_dir, "env_episodes.csv")
        self._env_ep_csv_file = None
        self._env_ep_csv_writer = None
        self._env_ep_keys: list[str] = []

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
            "phi_method", "phi_riiu",
            "phi_riiu_broadcast", "phi_riiu_tectum", "phi_riiu_audio",
            "levin_bioelectric_complexity", "levin_morphological_adaptation",
            "levin_collective_intelligence", "levin_goal_directed",
            "levin_basal_cognition",
            "self_pred_mse", "self_pred_skill",
            "recon_loss",
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

        # CSV. phi uses scientific notation because realized values can be
        # below 5e-7 (which 6-decimal float would truncate to 0.000000).
        # The pre-fix ablation runs all logged phi=0.0 not because pyphi
        # returned zero but because the values were sub-microsecond and
        # got rounded away.
        self._csv_writer.writerow([
            step, f"{metrics.phi:.6e}", f"{metrics.sync_r:.6f}",
            int(metrics.is_conscious), f"{metrics.reward:.6f}",
            f"{metrics.broadcast_mag:.6f}",
            f"{metrics.valence:.4f}", f"{metrics.arousal:.4f}",
            f"{metrics.dominance:.4f}",
            metrics.phi_method, f"{metrics.phi_riiu:.6e}",
            f"{metrics.phi_riiu_broadcast:.6e}",
            f"{metrics.phi_riiu_tectum:.6e}", f"{metrics.phi_riiu_audio:.6e}",
            f"{metrics.levin_bioelectric_complexity:.6f}",
            f"{metrics.levin_morphological_adaptation:.6f}",
            f"{metrics.levin_collective_intelligence:.6f}",
            f"{metrics.levin_goal_directed:.6f}",
            f"{metrics.levin_basal_cognition:.6f}",
            f"{metrics.self_pred_mse:.6e}", f"{metrics.self_pred_skill:.6f}",
            f"{metrics.recon_loss:.6e}",
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
            if metrics.phi_riiu != 0.0:
                self.writer.add_scalar("consciousness/phi_riiu", metrics.phi_riiu, step)
            if metrics.phi_riiu_broadcast != 0.0:
                self.writer.add_scalar("consciousness/phi_riiu_broadcast", metrics.phi_riiu_broadcast, step)
            if metrics.phi_riiu_tectum != 0.0:
                self.writer.add_scalar("consciousness/phi_riiu_tectum", metrics.phi_riiu_tectum, step)
            if metrics.phi_riiu_audio != 0.0:
                self.writer.add_scalar("consciousness/phi_riiu_audio", metrics.phi_riiu_audio, step)
            # Levin metrics: log only when active (any non-zero), so a disabled
            # run does not flood TensorBoard with constant-zero series.
            levin_vals = (
                metrics.levin_bioelectric_complexity,
                metrics.levin_morphological_adaptation,
                metrics.levin_collective_intelligence,
                metrics.levin_goal_directed,
                metrics.levin_basal_cognition,
            )
            if any(v != 0.0 for v in levin_vals):
                self.writer.add_scalar("levin/bioelectric_complexity", metrics.levin_bioelectric_complexity, step)
                self.writer.add_scalar("levin/morphological_adaptation", metrics.levin_morphological_adaptation, step)
                self.writer.add_scalar("levin/collective_intelligence", metrics.levin_collective_intelligence, step)
                self.writer.add_scalar("levin/goal_directed", metrics.levin_goal_directed, step)
                self.writer.add_scalar("levin/basal_cognition", metrics.levin_basal_cognition, step)
            # Self-vector loop: log only when active (skill non-zero or mse set).
            if metrics.self_pred_mse != 0.0 or metrics.self_pred_skill != 0.0:
                self.writer.add_scalar("self_model/self_pred_mse", metrics.self_pred_mse, step)
                self.writer.add_scalar("self_model/self_pred_skill", metrics.self_pred_skill, step)

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
            episode, f"{total_reward:.4f}", steps, f"{avg_phi:.6e}",
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

    def log_env_episode(self, episode: int, env_metrics: dict):
        """Log environment-specific per-episode metrics (e.g. WCST rule_changes,
        trials_correct) to env_episodes.csv. The header is created lazily from
        the first call's keys, so only runs that report env metrics produce the
        file."""
        if not env_metrics:
            return
        if self._env_ep_csv_writer is None:
            self._env_ep_csv_file = open(self._env_ep_csv_path, "w", newline="")
            self._env_ep_csv_writer = csv.writer(self._env_ep_csv_file)
            self._env_ep_keys = sorted(env_metrics.keys())
            self._env_ep_csv_writer.writerow(["episode"] + self._env_ep_keys)
        self._env_ep_csv_writer.writerow(
            [episode] + [env_metrics.get(k, "") for k in self._env_ep_keys]
        )
        self._env_ep_csv_file.flush()

    def compute_and_log_ei(self, episode: int, num_gate_states: int = 243,
                           num_workspace_states: int = 8) -> dict:
        """
        Compute EI at gate and workspace levels from buffered trajectories.

        Gate states use fixed tertile boundaries [1/3, 2/3]: each of the 5 gate
        dimensions (sigmoid-bounded [0, 1]) is binned to 3 levels (low/mid/high),
        giving 3^5 = 243 joint states. Fixed thresholds avoid the bias of computing
        percentiles from the same trajectory being discretized, which guarantees
        roughly uniform distributions and inflates EI artificially.

        Returns dict with ei_gates, ei_workspace, ratio, emergent.
        """
        from models.evaluation.effective_information import (
            compute_effective_information,
            discretize_continuous,
        )

        result = {"ei_gates": 0.0, "ei_workspace": 0.0, "ratio": 0.0, "emergent": False}

        if len(self._gate_trajectory) < 10:
            return result

        # Fixed tertile boundaries: gate outputs are sigmoid-bounded [0, 1]
        gate_discrete = []
        for g in self._gate_trajectory:
            joint_idx = 0
            for i, val in enumerate(g):
                if val < 1 / 3:
                    trit = 0
                elif val < 2 / 3:
                    trit = 1
                else:
                    trit = 2
                joint_idx += trit * (3 ** i)
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
        # Require reward > 0.5 AND >= 1.5x running average (positive portion)
        if reward < 0.5:
            return False

        if len(self._cross_episode_rewards) >= 200:
            positive_rewards = [r for r in self._cross_episode_rewards if r > 0]
            if positive_rewards:
                avg_positive = np.mean(positive_rewards)
                reward_jump = reward >= 1.5 * avg_positive
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
        if self._env_ep_csv_file is not None:
            self._env_ep_csv_file.close()
