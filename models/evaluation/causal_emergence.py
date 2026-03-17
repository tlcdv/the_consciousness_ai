"""
Causal Emergence Metrics.

Implements Effective Information (EI) following Hoel et al. (2013), adapted
to the state-by-node TPM format used by IITMetrics.build_empirical_tpm().

Purpose: test the strong emergence claim in the Functionalist Emergentism thesis.

Falsification criterion
-----------------------
If EI(workspace) <= EI(gates) consistently across training, the integrated
workspace state does not have greater causal power than its components.
Strong emergence is not occurring in the current architecture.

Evidence for strong emergence
-----------------------------
If EI(workspace) > EI(gates) as training progresses and complexity increases,
the macro level is exerting genuine causal constraint beyond what the micro
components alone predict (Hoel 2013: "macro can beat micro").

EI formula (adapted for state-by-node TPM)
-------------------------------------------
For each node j across all input states i:

    EI_j = H(mean_j) - mean_i[ H(P(node_j=1 | state_i)) ]

Where H is binary entropy. The first term measures integration (how variable
the effect of node j is across inputs), the second measures degeneracy (how
uncertain the output is for each specific input). EI = sum_j EI_j.

Higher EI = more deterministic (lower row entropy) and more integrated
(higher column mean entropy). This captures the Hoel "effectiveness" criterion.

Reference
---------
Hoel EP, Albantakis L, Tononi G (2013). Quantifying causal emergence shows
that macro can beat micro. PNAS 110(49):19790-19795.
https://www.pnas.org/doi/10.1073/pnas.1314922110
"""
from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _binary_entropy(p: float) -> float:
    """H(p) = -p*log2(p) - (1-p)*log2(1-p). Returns 0 for p in {0, 1}."""
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def effective_information(tpm: np.ndarray) -> float:
    """
    Compute Effective Information from a state-by-node TPM.

    Args:
        tpm: Shape (num_states, num_nodes). TPM[i][j] = P(node_j=1 | state_i).
             Built by IITMetrics.build_empirical_tpm().

    Returns:
        EI value (>= 0). Higher means more causally effective (more deterministic
        and more integrated across input states).
    """
    if tpm is None or tpm.ndim != 2 or tpm.size == 0:
        return 0.0

    num_nodes = tpm.shape[1]
    total_ei = 0.0

    for j in range(num_nodes):
        col = tpm[:, j]

        # Integration: entropy of the mean effect of node j across all input states.
        p_mean = float(np.mean(col))
        h_mean = _binary_entropy(p_mean)

        # Degeneracy: average entropy of node j's output for each specific input.
        mean_h_row = float(np.mean([_binary_entropy(p) for p in col]))

        # EI contribution: integration minus degeneracy. Clamped at 0.
        total_ei += max(0.0, h_mean - mean_h_row)

    return total_ei


@dataclass
class EISnapshot:
    """One recorded measurement of EI at both system levels."""
    step: int
    gate_ei: float
    workspace_ei: float
    emergence_ratio: float  # workspace_ei / gate_ei. >1 means macro beats micro.


class CausalEmergenceTracker:
    """
    Tracks EI at gate level and workspace level over training.

    Usage
    -----
    In the main simulation loop, after each IITMetrics.build_empirical_tpm() call:

        gate_tpm = iit.build_empirical_tpm(num_nodes=5)       # gate subsystem
        workspace_tpm = iit.build_empirical_tpm(num_nodes=4)  # workspace bids
        snapshot = ce_tracker.record(gate_tpm, workspace_tpm)

        if ce_tracker.is_strong_emergence():
            log("Macro-level causal power confirmed over window.")

    Pre-registered falsification prediction
    ----------------------------------------
    After 10k training steps in a novel-problem scenario, EI(workspace) should
    exceed EI(gates) for at least 60% of steps. If it does not, the strong
    emergence claim is falsified for the current architecture.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.history: list[EISnapshot] = []
        self._step = 0

    def record(self, gate_tpm: np.ndarray, workspace_tpm: np.ndarray) -> EISnapshot:
        """
        Compute and record EI at both levels.

        Args:
            gate_tpm: TPM built from gate state history (5 nodes: attention,
                      stability, adaptation, coherence, confidence).
            workspace_tpm: TPM built from workspace bid history (4 nodes).

        Returns:
            EISnapshot with gate_ei, workspace_ei, and emergence_ratio.
        """
        gate_ei = effective_information(gate_tpm)
        workspace_ei = effective_information(workspace_tpm)
        ratio = workspace_ei / (gate_ei + _EPS)

        snapshot = EISnapshot(
            step=self._step,
            gate_ei=gate_ei,
            workspace_ei=workspace_ei,
            emergence_ratio=ratio,
        )
        self.history.append(snapshot)
        self._step += 1

        if self._step % self.log_interval == 0:
            label = "MACRO>MICRO (emergence)" if ratio > 1.0 else "no emergence"
            logger.info(
                f"[CausalEmergence step {self._step}] "
                f"gate_EI={gate_ei:.4f} workspace_EI={workspace_ei:.4f} "
                f"ratio={ratio:.3f} ({label})"
            )

        return snapshot

    def is_strong_emergence(self, window: int = 100, threshold: float = 0.6) -> bool:
        """
        Test the falsification criterion over the recent window.

        Returns True if EI(workspace) > EI(gates) for at least `threshold`
        fraction of the last `window` steps.

        Pre-registered criterion: threshold=0.6, window=10000 after novel-problem training.
        """
        if len(self.history) < window:
            return False
        recent = self.history[-window:]
        fraction = sum(1 for s in recent if s.emergence_ratio > 1.0) / len(recent)
        return fraction >= threshold

    def get_summary(self) -> dict:
        """Return summary statistics for logging and the consciousness dashboard."""
        if not self.history:
            return {
                "steps": 0,
                "mean_gate_ei": 0.0,
                "mean_workspace_ei": 0.0,
                "mean_emergence_ratio": 0.0,
                "strong_emergence": False,
            }

        gate_eis = [s.gate_ei for s in self.history]
        workspace_eis = [s.workspace_ei for s in self.history]
        ratios = [s.emergence_ratio for s in self.history]

        return {
            "steps": len(self.history),
            "mean_gate_ei": float(np.mean(gate_eis)),
            "mean_workspace_ei": float(np.mean(workspace_eis)),
            "mean_emergence_ratio": float(np.mean(ratios)),
            "strong_emergence": self.is_strong_emergence(),
        }