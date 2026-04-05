"""
Effective Information (EI) for Causal Emergence Detection

Implements Erik Hoel's causal emergence framework (PNAS 2013) to
measure whether macro-level states (workspace) carry more causal
information than micro-level states (individual gates).

EI(X) = H(effect | do(cause = uniform)) - noise
       = determinism - degeneracy

If EI(workspace) > EI(gates), the workspace level is causally
emergent. The strong emergence claim is falsified if this never
occurs across training.

Reference: Hoel et al. (2013), "Quantifying causal emergence shows
that macro can beat micro", PNAS 110(49).
"""
from __future__ import annotations

import numpy as np


def _build_tpm(trajectories: list[np.ndarray], num_states: int) -> np.ndarray:
    """
    Build a Transition Probability Matrix from observed state trajectories.

    Args:
        trajectories: list of 1D arrays, each being a sequence of discrete
                      state indices (integers in [0, num_states)).
        num_states: Number of possible states.

    Returns:
        TPM of shape [num_states, num_states] where TPM[i, j] = P(next=j | current=i).
        Uses Laplace smoothing to avoid zero probabilities.
    """
    counts = np.ones((num_states, num_states))  # Laplace smoothing

    for traj in trajectories:
        for t in range(len(traj) - 1):
            s_from = int(traj[t])
            s_to = int(traj[t + 1])
            if 0 <= s_from < num_states and 0 <= s_to < num_states:
                counts[s_from, s_to] += 1.0

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    tpm = counts / row_sums
    return tpm


def _entropy_row(row: np.ndarray) -> float:
    """Shannon entropy of a single probability distribution (one TPM row)."""
    # Filter out zeros to avoid log(0)
    p = row[row > 0]
    return -np.sum(p * np.log2(p))


def compute_effective_information(
    trajectories: list[np.ndarray],
    num_states: int,
) -> float:
    """
    Compute Effective Information (EI) from state trajectories.

    EI measures how deterministic the system's transitions are when
    intervened upon uniformly. Higher EI means cleaner causal structure.

    EI = log2(num_states) - <H(effect | do(cause = i))>_i

    This equals: max possible entropy minus average noise entropy.

    Args:
        trajectories: list of state sequences (1D integer arrays).
        num_states: Number of discrete states the system can be in.

    Returns:
        EI value (float). Range: [0, log2(num_states)].
        0 = maximally noisy (identity/random TPM).
        log2(num_states) = fully deterministic.
    """
    if num_states < 2:
        return 0.0

    tpm = _build_tpm(trajectories, num_states)

    # Maximum entropy = uniform distribution over num_states
    max_entropy = np.log2(num_states)

    # Average conditional entropy: H(effect | cause) averaged over uniform cause
    avg_noise = np.mean([_entropy_row(tpm[i]) for i in range(num_states)])

    # EI = determinism = max_entropy - avg_noise
    ei = max_entropy - avg_noise
    return float(max(0.0, ei))


def compare_ei_levels(
    gate_trajectories: list[np.ndarray],
    gate_num_states: int,
    workspace_trajectories: list[np.ndarray],
    workspace_num_states: int,
) -> dict[str, float]:
    """
    Compare Effective Information at gate level vs workspace level.

    This is the core falsification test: if EI(workspace) > EI(gates),
    the workspace exhibits causal emergence.

    Args:
        gate_trajectories: State sequences from individual gates
                          (attention, emotional, temporal gate activations).
        gate_num_states: Number of discrete states at gate level.
        workspace_trajectories: State sequences from workspace output.
        workspace_num_states: Number of discrete states at workspace level.

    Returns:
        dict with:
            - ei_gates: EI at the gate (micro) level
            - ei_workspace: EI at the workspace (macro) level
            - ratio: EI_workspace / EI_gates (> 1.0 = causal emergence)
            - emergent: bool, whether workspace EI exceeds gate EI
    """
    ei_gates = compute_effective_information(gate_trajectories, gate_num_states)
    ei_workspace = compute_effective_information(workspace_trajectories, workspace_num_states)

    ratio = ei_workspace / ei_gates if ei_gates > 0 else float("inf")

    return {
        "ei_gates": ei_gates,
        "ei_workspace": ei_workspace,
        "ratio": ratio,
        "emergent": ei_workspace > ei_gates,
    }


def discretize_continuous(
    values: np.ndarray,
    num_bins: int = 8,
) -> np.ndarray:
    """
    Discretize continuous state values into integer bin indices.

    Useful for converting gate activations (floats in [0,1]) or
    workspace bid vectors into discrete states for EI computation.

    Args:
        values: 1D array of continuous values.
        num_bins: Number of discrete bins.

    Returns:
        1D integer array of bin indices in [0, num_bins).
    """
    # Clip to [0, 1] range, then bin
    clipped = np.clip(values, 0.0, 1.0)
    bins = np.floor(clipped * (num_bins - 1)).astype(int)
    return np.clip(bins, 0, num_bins - 1)
