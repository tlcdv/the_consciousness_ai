"""
IIT Phi integration for consciousness measurement.

Computes Integrated Information (Phi) per IIT 3.0 (Oizumi, Albantakis, Tononi 2014)
using the consciousness gate subsystem as the causal network.

The five gate nodes (attention, stability, adaptation, coherence, confidence) form a
causally interconnected subsystem:
  - attention -> stability (attention_net output feeds stability_net via gated state)
  - stability -> adaptation (stability_score multiplies base adaptation_rate)
  - coherence -> adaptation (meta_memory context modulates adaptation)
  - confidence -> attention (narrator confidence feeds back into next cycle)
  - attention -> coherence (gated output determines what reaches meta-memory)

The empirical TPM is built from observed binary state transitions. When pyphi is
available, exact Big Phi is computed. Otherwise, a geometric proxy based on TPM
structure approximates integration.
"""
from __future__ import annotations

import torch
import numpy as np

try:
    import pyphi
except ImportError:
    pyphi = None

from typing import Any
import logging
import warnings
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PhiResult:
    """Result of a Phi computation."""
    phi: float
    num_nodes: int
    num_transitions: int
    method: str  # "pyphi", "proxy", or "insufficient_data"
    node_labels: tuple[str, ...]
    current_state: tuple[int, ...]


# Causal connectivity matrix for the 5-node gate subsystem.
# Row i, col j = 1 means node i causally influences node j.
# Nodes: [attention, stability, adaptation, coherence, confidence]
#
# Every node has in-degree >= 1 and out-degree >= 1, and every node
# lies inside a directed cycle, so the system is irreducible under IIT.
# pyphi.compute.sia returns non-zero phi when the empirical TPM has
# structure. Cycles include:
#     att -> stb -> adp -> conf -> att
#     att -> coh -> adp -> conf -> att
#     att -> coh -> conf -> att
#
# Verified against ConsciousnessGate.forward() in
# models/core/consciousness_gating.py: every edge below has a matching
# input concatenation in the gate networks.
GATE_CM = np.array([
    # att  stb  adp  coh  conf
    [0,    1,   0,   1,   0],   # attention -> stability, coherence
    [0,    0,   1,   0,   0],   # stability -> adaptation
    [0,    0,   0,   0,   1],   # adaptation -> confidence (closes cycle)
    [0,    0,   1,   0,   1],   # coherence -> adaptation, confidence
    [1,    0,   0,   0,   0],   # confidence -> attention (cross-step feedback)
], dtype=int)

GATE_NODE_LABELS = (
    "attention",
    "stability",
    "adaptation",
    "coherence",
    "confidence",
)

# Adaptive binarization floors per gate dimension. The adaptation_rate floor
# was lowered from 0.001 to 1e-5 on 2026-05-17 (Phase C of the Phi-1 retest
# plan, ~/.claude/plans/let-s-plan-the-next-misty-parasol.md): adaptation
# is scaled to [0, 0.02] in models/core/consciousness_gating.py:187, so the
# running median was below 0.001 and binarization was always 0, removing
# adaptation from the 5-bit state entirely. Lowering the floor lets
# adaptation actually contribute to TPM state diversity.
_DEFAULT_BINARIZATION_FLOORS = (0.1, 0.1, 1e-5, 0.05, 0.1)


class IITMetrics:
    """
    Computes Integrated Information (Phi) from consciousness gate states.

    The gate subsystem is a 5-node causal network. Binary states are derived
    by thresholding each gate value against adaptive medians computed from
    the running history. The empirical TPM captures how the system's causal
    structure actually behaves during operation.

    When pyphi is installed, exact Big Phi is computed via the MIP algorithm.
    Without pyphi, a proxy metric based on TPM determinism and integration
    structure is returned instead.
    """

    def __init__(self, history_len: int = 200, tpm_window: int = 200,
                 tpm_decay: float = 0.995, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.history_len = history_len
        self.tpm_window = tpm_window
        self.tpm_decay = tpm_decay

        # Raw continuous gate values for adaptive thresholding
        self._raw_history = deque(maxlen=history_len)
        # Binarized state history for TPM construction.
        # Uses tpm_window (not history_len) so the TPM reflects recent dynamics
        # instead of converging to a fixed point from all-time averages.
        self.state_history = deque(maxlen=tpm_window)
        # Detached continuous gate values for the temporal variance loss in
        # train_rlhf.py. Mirrors the rolling-window pattern of state_history
        # (no per-episode reset) so the diversity pressure and the TPM track
        # the same window of gate dynamics.
        self._gate_buffer: deque = deque(maxlen=32)

        # Adaptive thresholds (updated from running medians)
        self._thresholds = np.array([0.5, 0.5, 0.01, 0.5, 0.5])
        # Per-dimension flag: have we already warned that the running median
        # is below 5x the binarization floor? Mirror-mode warning fires once
        # per dimension per instance to surface "gate is uninformative"
        # without spamming.
        self._floor_pinned_warned = np.zeros(5, dtype=bool)

        # Node labels
        self.node_labels = GATE_NODE_LABELS

        # PyPhi config
        if pyphi is not None:
            pyphi.config.PROGRESS_BARS = False
            pyphi.config.PARALLEL_CUT_EVALUATION = False

    def reset_tpm(self) -> None:
        """No-op kept for backward compatibility.

        Previously cleared state_history at episode boundaries. That made
        the empirical TPM rebuild from <=5 transitions every episode start,
        with Laplace alpha=0.1 dominating the 27 unvisited rows of the
        32-state TPM. pyphi then returned phi values <5e-7 (truncated to
        0 by 6-decimal CSV logging) for ~99% of calls.

        Exponential decay (tpm_decay=0.995) on the rolling window already
        handles cross-episode staleness: a transition from 200 steps ago
        contributes 0.995^200 = 0.367 of its original weight, and one from
        500 steps ago drops to 0.082. Keeping state_history rolling yields
        a TPM with ~6-11 reliably-visited rows instead of 1-3 sparse rows.

        Diagnostic chain in scripts/analysis/diagnose_phi_zero_v[2,3,4].py
        confirmed the cause and verified pyphi runs cleanly on the rolling
        TPM.
        """
        return

    def _gate_state_to_raw(self, gate_state) -> np.ndarray:
        """Extract raw continuous values from a GatingState dataclass."""
        return np.array([
            gate_state.attention_level,
            gate_state.stability_score,
            gate_state.adaptation_rate,
            gate_state.meta_memory_coherence,
            gate_state.narrator_confidence,
        ], dtype=np.float64)

    def _update_thresholds(self) -> None:
        """Recompute adaptive binarization thresholds from running medians.

        Floor values prevent degenerate splits when a gate dimension never
        crosses 0.5 in either direction. The adaptation_rate floor was
        lowered from 0.001 to 1e-5 on 2026-05-17 (Phase C of the Phi-1
        retest plan): adaptation is scaled to [0, 0.02] in
        models/core/consciousness_gating.py:187, so the running median is
        often below 0.001 and binarization was always 0, removing
        adaptation from the 5-bit state entirely. Lowering the floor lets
        adaptation actually contribute to state diversity.

        Median-vs-floor warning: if a dimension's running median stays
        below 5x the floor for a long stretch, the binarization is being
        pinned by the floor and the gate dimension is not informative.
        The warning fires once per dimension per `IITMetrics` instance.
        """
        if len(self._raw_history) < 10:
            return
        raw = np.array(self._raw_history)
        medians = np.median(raw, axis=0)
        floors = np.array(_DEFAULT_BINARIZATION_FLOORS)
        self._thresholds = np.maximum(medians, floors)
        # Mirror-mode warning: when the running median is strictly below the
        # floor, the threshold is being clamped to the floor and binarization
        # is always 1 (because `v > floor` rarely fails for the actual
        # distribution). This is the "always 1" mirror of the pre-2026-05-17
        # "always 0" bug. The warning fires once per dimension per instance.
        if len(self._raw_history) >= 50:
            pinned = (medians < floors) & ~self._floor_pinned_warned
            for i in np.where(pinned)[0]:
                warnings.warn(
                    f"IITMetrics: gate dimension '{GATE_NODE_LABELS[i]}' median "
                    f"({medians[i]:.3e}) is below floor ({floors[i]:.3e}); "
                    f"binarization threshold is clamped to the floor and may be "
                    f"saturated to 1. The dimension may not be informative in "
                    f"the TPM. Consider lowering the floor further or revising "
                    f"the gate scaling.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._floor_pinned_warned[i] = True

    def _binarize(self, raw: np.ndarray) -> tuple[int, ...]:
        """Binarize raw gate values using adaptive thresholds."""
        return tuple(int(v > t) for v, t in zip(raw, self._thresholds))

    def push_gate_values(self, gate_values_tensor: torch.Tensor) -> None:
        """Append detached continuous gate values to the rolling buffer.

        Used by the temporal variance loss in scripts/training/train_rlhf.py
        as the historical-mean target. The current step's live tensor stays
        outside the buffer (only its detached value is stored) so gradient
        only flows through the current step, not through past steps.
        """
        self._gate_buffer.append(gate_values_tensor.detach().squeeze(0).cpu())

    def update_from_gate_state(self, gate_state) -> tuple[int, ...]:
        """
        Record the current causal state from the consciousness gate.

        Extracts continuous values, updates adaptive thresholds, and appends
        the binarized state to history.

        Args:
            gate_state: GatingState dataclass with attention_level,
                stability_score, adaptation_rate, meta_memory_coherence,
                narrator_confidence.

        Returns:
            The binarized state tuple.
        """
        raw = self._gate_state_to_raw(gate_state)
        self._raw_history.append(raw)
        self._update_thresholds()
        state = self._binarize(raw)
        self.state_history.append(state)
        return state

    def build_empirical_tpm(self, num_nodes: int = 5) -> np.ndarray:
        """
        Build a state-by-node TPM from observed transitions.

        Uses reduced Laplace smoothing (alpha=0.1) and exponential decay on
        transition counts so recent transitions dominate. This prevents TPM
        saturation that makes phi converge to a fixed point.

        Args:
            num_nodes: Number of nodes (default 5 for the gate subsystem).

        Returns:
            TPM of shape (2^N, N) in state-by-node format.
        """
        num_states = 2 ** num_nodes
        # Reduced Laplace smoothing (alpha=0.1 instead of 1.0)
        tpm_counts = np.full((num_states, num_nodes), 0.1)
        state_visit_counts = np.full(num_states, 0.2)

        if len(self.state_history) < 2:
            # Uniform TPM when no transitions observed
            return np.full((num_states, num_nodes), 0.5)

        history = list(self.state_history)
        num_transitions = len(history) - 1
        for i in range(num_transitions):
            state_t = history[i]
            state_next = history[i + 1]

            if len(state_t) != num_nodes or len(state_next) != num_nodes:
                continue

            # Convert binary tuple to row index (little-endian per pyphi convention)
            state_idx = sum(bit << j for j, bit in enumerate(state_t))
            if state_idx >= num_states:
                continue

            # Exponential decay: recent transitions weigh more.
            # Transition i has weight decay^(num_transitions - 1 - i).
            weight = self.tpm_decay ** (num_transitions - 1 - i)

            state_visit_counts[state_idx] += weight
            for node_idx in range(num_nodes):
                if state_next[node_idx] == 1:
                    tpm_counts[state_idx, node_idx] += weight

        tpm = tpm_counts / state_visit_counts[:, None]
        return tpm

    def calculate_phi(self, tpm: np.ndarray, current_state: tuple[int, ...],
                      cm: np.ndarray | None = None) -> float:
        """
        Compute Big Phi using PyPhi.

        Args:
            tpm: State-by-node TPM, shape (2^N, N).
            current_state: Binary state tuple of length N.
            cm: Connectivity matrix, shape (N, N). Uses GATE_CM by default.

        Returns:
            Big Phi value, or 0.0 on error / pyphi unavailable.
        """
        if pyphi is None:
            return 0.0

        try:
            num_nodes = len(current_state)
            if num_nodes > 8:
                # Phi computation is exponential; cap at 8 nodes
                return 0.0

            if cm is None:
                cm = GATE_CM[:num_nodes, :num_nodes]

            network = pyphi.Network(tpm, cm=cm)
            subsystem = pyphi.Subsystem(network, current_state)
            sia = pyphi.compute.sia(subsystem)
            return float(sia.phi) if sia is not None else 0.0

        except Exception as e:
            self.logger.debug("Phi computation error: %s", e)
            return 0.0

    def compute_phi_proxy_from_tpm(self, tpm: np.ndarray,
                                    current_state: tuple[int, ...]) -> float:
        """
        Geometric proxy for Phi when pyphi is not available.

        **WARNING: UNVALIDATED APPROXIMATION**
        This proxy has NO published basis and has NOT been validated against
        the actual Phi computed by PyPhi. It is a heuristic that combines
        TPM determinism and integration metrics, but the formula
        determinism * (1 + integration) is arbitrary.

        Use ONLY as a fallback when PyPhi is unavailable. To properly validate
        this proxy, correlate it against pyphi.compute_phi() on the same TPM
        over 100+ timesteps and report Pearson r.

        Measures two properties:
        1. Determinism: how far TPM rows are from uniform (0.5).
           High determinism means the system has strong causal structure.
        2. Integration: variance of determinism across rows.
           A system where all rows are equally deterministic is less
           integrated than one with heterogeneous causal structure.

        Returns a value in [0, ~2].
        """
        num_nodes = tpm.shape[1]

        # Determinism: mean absolute deviation from 0.5 across all entries
        deviation = np.abs(tpm - 0.5)
        determinism = float(np.mean(deviation)) * 2.0  # Scale to [0, 1]

        # Per-row determinism
        row_det = np.mean(deviation, axis=1)
        # Integration: std of row determinism. Uniform = 0, heterogeneous > 0
        integration = float(np.std(row_det)) * 2.0

        # Proxy phi = determinism * (1 + integration)
        # Fully random TPM: determinism ~ 0, proxy ~ 0
        # Fully deterministic but uniform: integration ~ 0, proxy = determinism
        # Structured causal system: both high, proxy > determinism alone
        return determinism * (1.0 + integration)

    def compute_phi_from_gate_state(self, gate_state) -> PhiResult:
        """
        Primary entry point: compute Phi from consciousness gate state.

        Updates history, builds empirical TPM, and computes Phi (exact if
        pyphi available, proxy otherwise).

        Args:
            gate_state: GatingState dataclass.

        Returns:
            PhiResult with phi value, metadata, and method used.
        """
        current_state = self.update_from_gate_state(gate_state)
        num_transitions = max(0, len(self.state_history) - 1)

        if num_transitions < 5:
            return PhiResult(
                phi=0.0,
                num_nodes=5,
                num_transitions=num_transitions,
                method="insufficient_data",
                node_labels=self.node_labels,
                current_state=current_state,
            )

        tpm = self.build_empirical_tpm(5)

        if pyphi is not None:
            phi = self.calculate_phi(tpm, current_state, cm=GATE_CM)
            method = "pyphi"
        else:
            phi = self.compute_phi_proxy_from_tpm(tpm, current_state)
            method = "proxy"

        return PhiResult(
            phi=phi,
            num_nodes=5,
            num_transitions=num_transitions,
            method=method,
            node_labels=self.node_labels,
            current_state=current_state,
        )

    def compute_phi_from_gate_state_scalar(self, gate_state) -> float:
        """Convenience wrapper returning just the phi float value."""
        return self.compute_phi_from_gate_state(gate_state).phi

    def get_tpm_stats(self) -> dict[str, Any]:
        """
        Return diagnostic statistics about the current TPM.

        Useful for monitoring whether the gate subsystem has sufficient
        causal structure for meaningful Phi computation.
        """
        if len(self.state_history) < 2:
            return {"status": "insufficient_data", "transitions": 0}

        tpm = self.build_empirical_tpm(5)
        deviation = np.abs(tpm - 0.5)

        # Count unique states visited
        unique_states = len(set(self.state_history))

        # Per-node determinism (how predictable each node is)
        node_determinism = {}
        for i, label in enumerate(self.node_labels):
            node_determinism[label] = float(np.mean(deviation[:, i]) * 2.0)

        return {
            "status": "ready",
            "transitions": len(self.state_history) - 1,
            "unique_states": unique_states,
            "possible_states": 32,
            "state_coverage": unique_states / 32.0,
            "mean_determinism": float(np.mean(deviation) * 2.0),
            "node_determinism": node_determinism,
        }

    # --- Legacy methods (preserved for backward compatibility) ---

    def update_history(self, current_state: tuple[int, ...]) -> None:
        """Add a pre-binarized state to history. Prefer update_from_gate_state()."""
        self.state_history.append(current_state)

    def compute_phi_proxy(self, global_workspace_state: torch.Tensor) -> float:
        """
        DEPRECATED. Uses workspace bid values, not causal gate states.
        Phi values from this method are methodologically invalid.
        Use compute_phi_from_gate_state() instead.
        """
        import warnings
        warnings.warn(
            "compute_phi_proxy() uses bid values, not causal states. "
            "Use compute_phi_from_gate_state() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        subsystem_data = self._extract_subsystem_state(global_workspace_state)
        if not subsystem_data:
            return 0.0
        current_state = subsystem_data["state"]
        self.update_history(current_state)
        tpm = self.build_empirical_tpm(len(current_state))
        return self.calculate_phi(tpm, current_state)

    def _extract_subsystem_state(self, attention_weights: torch.Tensor,
                                  threshold: float = 0.1) -> dict[str, Any] | None:
        """Extract binary state from workspace attention weights (legacy)."""
        if attention_weights.ndim < 1:
            return None
        k = min(4, attention_weights.numel())
        top_values, top_indices = torch.topk(attention_weights, k)
        current_state = tuple((top_values > threshold).int().tolist())
        return {"state": current_state, "node_indices": top_indices.tolist()}

    # Keep old name as alias for backward compatibility
    extract_subsystem_state = _extract_subsystem_state
