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
    # DMTS supervised match head: choice-phase cross-entropy loss and accuracy.
    # Zero when --enable-match-head is off. acc rising toward the 0.845 offline
    # decodability means the live in-loop pipeline supports the match.
    match_head_loss: float = 0.0
    match_head_acc: float = 0.0


def _gate_cells(gate_state, n: int = 5) -> list[str]:
    """
    Format the raw gate node values for the per-step CSV, padded to n columns.

    Returns empty strings when no gate is active (or when the fallback
    competition-results path supplies fewer than n values), so the column count
    stays fixed regardless of configuration.
    """
    vals = list(gate_state) if gate_state is not None else []
    return [f"{float(vals[i]):.6f}" if i < len(vals) else "" for i in range(n)]


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

        # Gate discretization mode, shared by EI and CE 2.0 so both score the same
        # TPM. "tertile" (default) = fixed [1/3, 2/3] boundaries, baseline
        # bit-identical. "quantile" = per-dimension terciles from the window's own
        # distribution, set via set_gate_binning(). The 2026-07 gate-binning diagnosis
        # found the gate nodes vary in a ~0.01-wide band around 0.49, which fixed
        # tertiles collapse to one joint state of 243 (see
        # docs/results/ce2_pilot_calibration_2026_07.md).
        self._gate_binning: str = "tertile"

        # Causal Emergence 2.0 (SVD heuristic) state. Kept separate from the EI
        # buffers above so enabling CE 2.0 never alters EI's window. Everything
        # here stays inert until enable_ce2() is called (only when
        # --log-ce2-every > 0), so the default path is bit-identical.
        self._ce2_enabled: bool = False
        self._ce2_num_classes: int = 32
        self._ce2_gate_trajectory: list[tuple[float, ...]] = []
        self._ce2_workspace_trajectory: list[tuple[float, ...]] = []
        self._latent_counts = None   # [num_classes, num_classes] transition counts
        self._latent_prev = None     # previous step's latent index field

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
            "match_head_loss", "match_head_acc",
            # Raw ConsciousnessGate node values. Logged so the gate discretization can
            # be diagnosed directly: the 2026-07 assessment inferred the gates were
            # saturated into one tertile joint state from the EI floor, but the raw
            # per-dimension variation was never measured. Empty when no gate is active.
            "gate_attention", "gate_stability", "gate_adaptation",
            "gate_coherence", "gate_confidence",
        ])

    def _init_episode_csv(self):
        self._ep_csv_file = open(self._ep_csv_path, "w", newline="")
        self._ep_csv_writer = csv.writer(self._ep_csv_file)
        self._ep_csv_writer.writerow([
            "episode", "total_reward", "steps", "avg_phi",
            "consciousness_ratio", "ei_gates", "ei_workspace", "ei_ratio",
            # Floor-corrected EI (constant-trajectory Laplace baseline subtracted;
            # see models/evaluation/effective_information.py). The raw columns are
            # kept unchanged for continuity with pre-2026-07 runs.
            "ei_gates_corr", "ei_workspace_corr", "ei_ratio_corr",
            # Causal Emergence 2.0 (SVD heuristic; see models/evaluation/
            # causal_emergence_svd.py). Zeros unless --log-ce2-every > 0. ce2_gates/
            # ce2_workspace mirror the EI gate/workspace TPMs; ce2_rssm scores the
            # RSSM discrete-latent class-transition TPM. The complexity columns count
            # causally contributing scales (singular values above the non-trivial mean).
            "ce2_gates", "ce2_workspace", "ce2_ratio",
            "ce2_complexity_gates", "ce2_complexity_workspace",
            "ce2_rssm", "ce2_complexity_rssm",
            # Distinct discretized states actually visited in the window. CE 2.0
            # rises as a trajectory freezes, so a value with a state count of 1 is a
            # discretization artifact, not emergence. Always read the two together.
            "ce2_gates_states", "ce2_workspace_states",
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
            f"{metrics.match_head_loss:.6e}", f"{metrics.match_head_acc:.6f}",
            *_gate_cells(metrics.gate_state),
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
            # DMTS match head: log only when active.
            if metrics.match_head_loss != 0.0 or metrics.match_head_acc != 0.0:
                self.writer.add_scalar("match_head/loss", metrics.match_head_loss, step)
                self.writer.add_scalar("match_head/acc", metrics.match_head_acc, step)

        # Buffer for insight detection
        self._cross_episode_rewards.append(metrics.reward)
        self._episode_broadcast_mags.append(metrics.broadcast_mag)
        self._global_insight_step += 1

        # Buffer for EI
        if metrics.gate_state is not None:
            self._gate_trajectory.append(metrics.gate_state)
        if metrics.workspace_state is not None:
            self._workspace_trajectory.append(metrics.workspace_state)

        # Buffer for CE 2.0 (separate lists so the CE 2.0 window is independent of
        # the EI window). No-op unless enable_ce2() has been called.
        if self._ce2_enabled:
            if metrics.gate_state is not None:
                self._ce2_gate_trajectory.append(metrics.gate_state)
            if metrics.workspace_state is not None:
                self._ce2_workspace_trajectory.append(metrics.workspace_state)

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
        ei_gates_corr: float = 0.0,
        ei_workspace_corr: float = 0.0,
        ei_ratio_corr: float = 0.0,
        ce2_gates: float = 0.0,
        ce2_workspace: float = 0.0,
        ce2_ratio: float = 0.0,
        ce2_complexity_gates: int = 0,
        ce2_complexity_workspace: int = 0,
        ce2_rssm: float = 0.0,
        ce2_complexity_rssm: int = 0,
        ce2_gates_states: int = 0,
        ce2_workspace_states: int = 0,
    ):
        """Log episode-level summary."""
        # CSV
        self._ep_csv_writer.writerow([
            episode, f"{total_reward:.4f}", steps, f"{avg_phi:.6e}",
            f"{consciousness_ratio:.4f}",
            f"{ei_gates:.6f}", f"{ei_workspace:.6f}", f"{ei_ratio:.4f}",
            f"{ei_gates_corr:.6f}", f"{ei_workspace_corr:.6f}",
            f"{ei_ratio_corr:.4f}",
            f"{ce2_gates:.6f}", f"{ce2_workspace:.6f}", f"{ce2_ratio:.4f}",
            ce2_complexity_gates, ce2_complexity_workspace,
            f"{ce2_rssm:.6f}", ce2_complexity_rssm,
            ce2_gates_states, ce2_workspace_states,
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
                self.writer.add_scalar("emergence/ei_gates_corr", ei_gates_corr, episode)
                self.writer.add_scalar("emergence/ei_workspace_corr",
                                       ei_workspace_corr, episode)
                self.writer.add_scalar("emergence/ei_ratio_corr", ei_ratio_corr, episode)

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

    def set_gate_binning(self, mode: str):
        """Select the gate discretization used by EI and CE 2.0. 'tertile' (default,
        baseline bit-identical) or 'quantile'."""
        if mode not in ("tertile", "quantile"):
            raise ValueError(f"gate binning must be 'tertile' or 'quantile', got {mode}")
        self._gate_binning = mode

    def _discretize_gate_window(self, trajectory, var_floor: float = 1e-4) -> list:
        """
        Map a window of gate-state tuples to joint tertile indices in [0, 3^d).

        'tertile' (default): fixed [1/3, 2/3] boundaries. Bit-identical to the
        pre-2026-07 inline code, so the baseline is unchanged when the flag is off.

        'quantile': per-dimension terciles from the window's own distribution, so a
        gate signal that lives in a narrow band (the 2026-07 diagnosis found ~0.01
        wide around 0.49) is actually resolved. A dimension whose std is at or below
        var_floor is treated as dead and pinned to bin 0, so a near-constant channel
        (e.g. gate_adaptation, std ~6e-6) is not split into noise.
        """
        if not trajectory:
            return []
        if self._gate_binning != "quantile":
            out = []
            for g in trajectory:
                joint_idx = 0
                for i, val in enumerate(g):
                    if val < 1 / 3:
                        trit = 0
                    elif val < 2 / 3:
                        trit = 1
                    else:
                        trit = 2
                    joint_idx += trit * (3 ** i)
                out.append(joint_idx)
            return out
        mat = np.asarray(trajectory, dtype=float)
        edges = []
        for i in range(mat.shape[1]):
            col = mat[:, i]
            edges.append(None if col.std() <= var_floor
                         else np.quantile(col, [1 / 3, 2 / 3]))
        out = []
        for g in mat:
            joint_idx = 0
            for i in range(mat.shape[1]):
                e = edges[i]
                if e is None:
                    trit = 0
                else:
                    trit = 0 if g[i] < e[0] else (1 if g[i] < e[1] else 2)
                joint_idx += trit * (3 ** i)
            out.append(joint_idx)
        return out

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
            corrected_effective_information,
            discretize_continuous,
        )

        result = {"ei_gates": 0.0, "ei_workspace": 0.0, "ratio": 0.0, "emergent": False,
                  "ei_gates_corr": 0.0, "ei_workspace_corr": 0.0, "ratio_corr": 0.0,
                  "emergent_corr": False}

        if len(self._gate_trajectory) < 10:
            return result

        # Gate discretization (tertile by default, or quantile via --gate-binning).
        gate_discrete = self._discretize_gate_window(self._gate_trajectory)

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

        # Floor-corrected EI: subtract the constant-trajectory Laplace baseline so a
        # frozen trajectory reports 0 instead of a state-count-dependent constant
        # (the 2026-07 assessment showed the raw ~12x "emergence ratio" can be the
        # ratio of two such floors). Raw values stay logged for continuity.
        ei_gates_corr = corrected_effective_information(
            [np.array(gate_discrete)], num_gate_states
        )
        ei_workspace_corr = corrected_effective_information(
            [np.array(ws_discrete)], num_workspace_states
        )
        if ei_gates_corr > 0:
            ratio_corr = ei_workspace_corr / ei_gates_corr
        elif ei_workspace_corr > 0:
            ratio_corr = float("inf")
        else:
            ratio_corr = 0.0
        emergent_corr = ei_workspace_corr > ei_gates_corr

        result = {
            "ei_gates": ei_gates,
            "ei_workspace": ei_workspace,
            "ratio": ratio,
            "emergent": emergent,
            "ei_gates_corr": ei_gates_corr,
            "ei_workspace_corr": ei_workspace_corr,
            "ratio_corr": ratio_corr,
            "emergent_corr": emergent_corr,
        }

        if self.writer is not None:
            self.writer.add_scalar("emergence/ei_gates", ei_gates, episode)
            self.writer.add_scalar("emergence/ei_workspace", ei_workspace, episode)
            self.writer.add_scalar("emergence/ei_ratio", ratio, episode)
            self.writer.add_scalar("emergence/emergent", int(emergent), episode)
            self.writer.add_scalar("emergence/ei_gates_corr", ei_gates_corr, episode)
            self.writer.add_scalar("emergence/ei_workspace_corr",
                                   ei_workspace_corr, episode)
            self.writer.add_scalar("emergence/emergent_corr",
                                   int(emergent_corr), episode)

        # Clear trajectory buffers for next window
        self._gate_trajectory.clear()
        self._workspace_trajectory.clear()

        return result

    # ----------------------------------------------------------------- #
    # Causal Emergence 2.0 (SVD heuristic, Hoel 2025, arXiv:2503.13395v3)
    # ----------------------------------------------------------------- #
    def enable_ce2(self, num_classes: int = 32):
        """Turn on CE 2.0 capture (call when --log-ce2-every > 0).

        Allocates the RSSM latent transition-count matrix and starts the separate
        gate/workspace buffers used by compute_and_log_ce2. Idempotent."""
        from models.evaluation.causal_emergence_svd import new_transition_counts
        self._ce2_enabled = True
        self._ce2_num_classes = int(num_classes)
        self._latent_counts = new_transition_counts(self._ce2_num_classes)
        self._latent_prev = None

    def reset_latent_window(self):
        """Drop the previous latent field so no transition is counted across an
        episode reset (the RSSM state resets between episodes). No-op when CE 2.0
        is disabled."""
        self._latent_prev = None

    def record_latent_step(self, index_field):
        """Accumulate one RSSM latent transition (prev -> current) into the pooled
        count matrix. index_field is a flat int array of per-variable class indices
        for the current step (from causal_emergence_svd.latent_class_indices)."""
        if not self._ce2_enabled or self._latent_counts is None:
            return
        from models.evaluation.causal_emergence_svd import update_transition_counts
        if self._latent_prev is not None:
            update_transition_counts(self._latent_counts, self._latent_prev, index_field)
        self._latent_prev = index_field

    def compute_and_log_ce2(self, episode: int, num_gate_states: int = 243,
                            num_workspace_states: int = 8) -> dict:
        """
        CE 2.0 (SVD heuristic) at the gate and workspace levels, plus the RSSM
        latent, from CE 2.0's own buffers.

        The gate/workspace discretization mirrors compute_and_log_ei exactly (fixed
        tertile bins for the 5 gate dims -> 3^5 = 243 states; workspace sum binned to
        num_workspace_states), so CE 2.0 and EI score identical TPMs and can be
        compared honestly. The RSSM TPM is the pooled class->class transition matrix
        accumulated by record_latent_step. Returns a dict of ce2_* scores and
        emergent-complexity counts, then clears its buffers and the latent counts.
        """
        from models.evaluation.effective_information import discretize_continuous
        from models.evaluation.causal_emergence_svd import (
            compute_ce2_from_trajectories, compute_ce2_from_tpm, counts_to_tpm,
            trajectory_degeneracy,
        )

        result = {"ce2_gates": 0.0, "ce2_workspace": 0.0, "ce2_ratio": 0.0,
                  "ce2_emergent": False, "ce2_complexity_gates": 0,
                  "ce2_complexity_workspace": 0, "ce2_rssm": 0.0,
                  "ce2_complexity_rssm": 0, "ce2_gates_states": 0,
                  "ce2_workspace_states": 0}

        # Gate + workspace CE 2.0 (needs a minimum window for a meaningful TPM).
        if len(self._ce2_gate_trajectory) >= 10:
            gate_discrete = self._discretize_gate_window(self._ce2_gate_trajectory)
            ws_flat = ([sum(w) for w in self._ce2_workspace_trajectory]
                       if self._ce2_workspace_trajectory else [0.0] * len(gate_discrete))
            ws_discrete = discretize_continuous(ws_flat, num_workspace_states)

            g_traj = [np.array(gate_discrete)]
            w_traj = [np.array(ws_discrete)]
            # Degeneracy first: CE 2.0 RISES as a trajectory freezes, so a value
            # computed on a frozen input is a discretization artifact, not emergence.
            g_deg = trajectory_degeneracy(g_traj, num_gate_states)
            w_deg = trajectory_degeneracy(w_traj, num_workspace_states)
            g_res = compute_ce2_from_trajectories(g_traj, num_gate_states)
            w_res = compute_ce2_from_trajectories(w_traj, num_workspace_states)
            result["ce2_gates"] = g_res.causal_emergence
            result["ce2_workspace"] = w_res.causal_emergence
            result["ce2_ratio"] = w_res.causal_emergence / max(g_res.causal_emergence, 1e-8)
            result["ce2_emergent"] = w_res.causal_emergence > g_res.causal_emergence
            result["ce2_complexity_gates"] = g_res.emergent_complexity
            result["ce2_complexity_workspace"] = w_res.emergent_complexity
            result["ce2_gates_states"] = g_deg["distinct_states"]
            result["ce2_workspace_states"] = w_deg["distinct_states"]
            if g_deg["degenerate"] or w_deg["degenerate"]:
                logger.warning(
                    "CE 2.0 computed on a degenerate trajectory (gate distinct "
                    "states=%d, workspace=%d). CE 2.0 rises as the input freezes, so "
                    "these values are discretization artifacts, not emergence. See "
                    "docs/results/ce2_pilot_calibration_2026_07.md",
                    g_deg["distinct_states"], w_deg["distinct_states"],
                )

        # RSSM latent CE 2.0 from the pooled transition counts.
        if self._latent_counts is not None and self._latent_counts.sum() > 0:
            r_res = compute_ce2_from_tpm(counts_to_tpm(self._latent_counts, laplace=1.0))
            result["ce2_rssm"] = r_res.causal_emergence
            result["ce2_complexity_rssm"] = r_res.emergent_complexity
        elif self._ce2_enabled:
            # No transitions pooled this window: the per-step record_latent_step
            # wiring did not fire. Surface it rather than silently logging zeros.
            logger.warning(
                "CE 2.0 enabled but no RSSM latent transitions were recorded this "
                "window; check the run_episode latent capture (z_state argmax)."
            )

        if self.writer is not None:
            self.writer.add_scalar("emergence/ce2_gates", result["ce2_gates"], episode)
            self.writer.add_scalar("emergence/ce2_workspace", result["ce2_workspace"], episode)
            self.writer.add_scalar("emergence/ce2_ratio", result["ce2_ratio"], episode)
            self.writer.add_scalar("emergence/ce2_rssm", result["ce2_rssm"], episode)
            self.writer.add_scalar("emergence/ce2_complexity_rssm",
                                   result["ce2_complexity_rssm"], episode)

        # Reset CE 2.0 buffers and latent counts for the next window.
        self._ce2_gate_trajectory.clear()
        self._ce2_workspace_trajectory.clear()
        if self._latent_counts is not None:
            self._latent_counts[:] = 0.0
        self._latent_prev = None

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
