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


def constant_trajectory_floor(num_states: int, traj_len: int) -> float:
    """EI reported for a trajectory frozen in ONE state, from Laplace smoothing alone.

    The 2026-07 signature assessment found this floor bit-identical across five runs
    (ei_gates = 0.031178 for 243 states at window 10000): with Laplace smoothing, a
    constant trajectory gives the visited row probability mass (L)/(N+L-1) on the
    self-transition and the other N-1 rows stay uniform, so the resulting EI is a pure
    function of (num_states, trajectory length) and NOT a measurement of causal
    structure. Any EI at or near this floor means "the trajectory never moved".

    Closed form (L = traj_len, N = num_states, one visited row k):
        EI = (log2 N - H_k) / N
        H_k = -( p_kk log2 p_kk + (N-1) p_o log2 p_o ),
        p_kk = L / (N + L - 1),  p_o = 1 / (N + L - 1)
    Matches compute_effective_information([np.zeros(traj_len)], num_states) exactly
    (unit-tested).
    """
    if num_states < 2 or traj_len < 2:
        return 0.0
    n = float(num_states)
    p_kk = traj_len / (n + traj_len - 1.0)
    p_o = 1.0 / (n + traj_len - 1.0)
    h_k = -(p_kk * np.log2(p_kk) + (n - 1.0) * p_o * np.log2(p_o))
    return float(max(0.0, (np.log2(n) - h_k) / n))


def corrected_effective_information(
    trajectories: list[np.ndarray],
    num_states: int,
) -> float:
    """EI with the constant-trajectory Laplace floor subtracted.

    Removes the bias the 2026-07 assessment identified: the raw EI of a frozen
    trajectory is a positive constant that depends only on (num_states, window
    length), and because that floor is HIGHER for smaller state spaces, a frozen
    macro level spuriously "beats" a frozen micro level. After subtraction a
    constant trajectory reports 0 and only actual transition structure counts.

    The raw compute_effective_information is left untouched (pre-registered
    thresholds and historical csv values refer to it); this is an additional,
    corrected reading.
    """
    raw = compute_effective_information(trajectories, num_states)
    total_len = sum(len(t) for t in trajectories)
    n_traj = len([t for t in trajectories if len(t) >= 2])
    # The floor is defined for one contiguous trajectory; for multiple, use the
    # total transition count as the effective length (the smoothing bias scales
    # with observed transitions, which is what total_len - n_traj counts).
    eff_len = (total_len - max(n_traj, 1)) + 1
    floor = constant_trajectory_floor(num_states, eff_len)
    return float(max(0.0, raw - floor))


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
            - ei_gates_corr / ei_workspace_corr: floor-corrected EI per level
            - ratio_corr: corrected ratio (0.0 when both levels are frozen)
            - emergent_corr: corrected emergence flag; False when the comparison
              is only the ratio of two Laplace floors
    """
    ei_gates = compute_effective_information(gate_trajectories, gate_num_states)
    ei_workspace = compute_effective_information(workspace_trajectories, workspace_num_states)

    ratio = ei_workspace / ei_gates if ei_gates > 0 else float("inf")

    ei_gates_corr = corrected_effective_information(gate_trajectories, gate_num_states)
    ei_workspace_corr = corrected_effective_information(
        workspace_trajectories, workspace_num_states)
    if ei_gates_corr > 0:
        ratio_corr = ei_workspace_corr / ei_gates_corr
    elif ei_workspace_corr > 0:
        ratio_corr = float("inf")
    else:
        ratio_corr = 0.0

    return {
        "ei_gates": ei_gates,
        "ei_workspace": ei_workspace,
        "ratio": ratio,
        "emergent": ei_workspace > ei_gates,
        "ei_gates_corr": ei_gates_corr,
        "ei_workspace_corr": ei_workspace_corr,
        "ratio_corr": ratio_corr,
        "emergent_corr": ei_workspace_corr > ei_gates_corr,
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
