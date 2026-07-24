"""
Causal Emergence 2.0 via the SVD heuristic (Hoel 2025, arXiv:2503.13395v3, S3).

CE 2.0 grounds macroscale causation in the causal primitives (sufficiency and
necessity, generalized to determinism and specificity) and estimates them from
the singular values of a row-stochastic Transition Probability Matrix (TPM),
without searching over coarse-grainings (which would be a combinatorial
explosion). Unlike Hoel's 2013 Effective Information (see
models/evaluation/effective_information.py), it carries no log2(n) size term.

The heuristic, verified against the source paper (Supplementary S3):
  1. SVD of the TPM gives singular values sigma_1 >= sigma_2 >= ... >= sigma_n.
  2. Discard the trivial sigma_1 (always >= 1 for a row-stochastic TPM, so it
     reflects nothing about the system's causation).
  3. gamma_star = mean(sigma_2 .. sigma_n) approximates determinism + specificity.
  4. Total causal emergence CE = sigma_2 - gamma_star (the highest non-trivial
     gain available).
  5. Emergent complexity = the count of sigma_i (2 <= i <= n) with
     sigma_i > (1/(n-1)) sum_{j=2..n} sigma_j. That averaging term is exactly
     gamma_star, so the test reduces to sigma_i > gamma_star. The differences
     (sigma_i - gamma_star) for the qualifiers are the per-scale causal
     contributions (the causal apportioning).

DEGENERACY CONFOUND (read before interpreting any value). CE 2.0 measures the
coarse-graining gain still AVAILABLE, so it RISES as the input trajectory becomes
more degenerate. Measured over 2000 steps with 243 states:

    distinct states:   1        2        10       50       243
    CE 2.0:            0.8779   0.7978   0.4329   0.1108   0.0148

A higher value therefore does NOT mean "more emergent"; it can mean "more frozen".
Always read a value alongside trajectory_degeneracy(), and compare it against
frozen_trajectory_ce2_value() to check the input was not frozen. Comparing two
levels with different state-space sizes (e.g. 243-state gates vs 8-state workspace)
is confounded by this. The 2026-07 dark_room pilot found the gate and workspace
trajectories were both frozen, so their CE 2.0 values were exactly the frozen-input
reference (docs/results/ce2_pilot_calibration_2026_07.md).

HONESTY CAVEAT. Applying a heuristic designed for coarse-grains of model Markov
chains to a trained neural world-model latent is exploratory and is NOT validated
by the source paper. S3 itself states the SVD adaptation is "just one proposal ...
future research may either fully explicate this method or propose alternatives."
Treat the outputs here as a diagnostic signal, not a validated measure of
consciousness or causal emergence. No result derived from this module may be
reported without at least 3-seed replication (see the verify-results protocol).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# Reuse the existing, tested TPM builder and discretizer rather than duplicating.
from models.evaluation.effective_information import _build_tpm, discretize_continuous

__all__ = [
    "CE2Result",
    "compute_ce2_from_tpm",
    "compute_ce2_from_trajectories",
    "frozen_trajectory_ce2_value",
    "trajectory_degeneracy",
    "latent_class_indices",
    "build_latent_tpm",
    "new_transition_counts",
    "update_transition_counts",
    "counts_to_tpm",
]


@dataclass
class CE2Result:
    """Outcome of the CE 2.0 SVD heuristic on one TPM."""

    causal_emergence: float          # sigma_2 - gamma_star  (>= 0; 0 when n < 3)
    gamma_star: float                # mean(sigma_2..sigma_n) ~ determinism + specificity
    emergent_complexity: int         # count of sigma_i (i >= 2) with sigma_i > gamma_star
    contributions: list = field(default_factory=list)  # [sigma_i - gamma_star] for qualifiers
    n_states: int = 0
    sigma: list = field(default_factory=list)           # full spectrum, for logging/debug


def _as_float64_matrix(tpm) -> torch.Tensor:
    """Accept a numpy array or torch tensor, return a CPU float64 2-D tensor."""
    if torch.is_tensor(tpm):
        t = tpm.detach().to("cpu", torch.float64)
    else:
        t = torch.as_tensor(np.asarray(tpm), dtype=torch.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError(f"TPM must be a square matrix, got shape {tuple(t.shape)}")
    return t


def compute_ce2_from_tpm(tpm, eps: float = 1e-9) -> CE2Result:
    """
    CE 2.0 SVD heuristic on a row-stochastic n x n TPM (rows sum to 1).

    Args:
        tpm: [n, n] numpy array or torch tensor, TPM[i, j] = P(next=j | current=i).
        eps: numerical tolerance. Singular values below eps are treated as zero,
            and a scale counts toward emergent complexity only if its singular
            value exceeds gamma_star by more than eps. This floors SVD round-off
            and Laplace-smoothing residues (sigma ~ 1e-16), which would otherwise
            register as spurious causal contributions.

    Returns:
        CE2Result. For n < 3 the metric is undefined (n = 1) or trivially zero
        (n = 2), so zeros are returned.
    """
    t = _as_float64_matrix(tpm)
    n = int(t.shape[0])
    if n < 3:
        return CE2Result(0.0, 0.0, 0, [], n, [])

    with torch.no_grad():
        sv = torch.linalg.svdvals(t)          # descending, length n; sigma_1 >= 1

    nontrivial = sv[1:]                        # discard the trivial sigma_1
    gamma_star = float(nontrivial.mean().item())
    ce = float(sv[1].item() - gamma_star)      # sigma_2 - gamma_star, always >= 0
    if ce < eps:
        ce = 0.0                               # floor numerical noise to exactly 0

    # Causal apportioning: count only genuinely positive singular values that
    # exceed the mean by more than numerical noise.
    significant = (nontrivial > eps) & (nontrivial > gamma_star + eps)
    contributions = (nontrivial[significant] - gamma_star).tolist()

    return CE2Result(
        causal_emergence=ce,
        gamma_star=gamma_star,
        emergent_complexity=int(significant.sum().item()),
        contributions=contributions,
        n_states=n,
        sigma=sv.tolist(),
    )


def compute_ce2_from_trajectories(trajectories: list[np.ndarray],
                                  num_states: int, eps: float = 1e-9) -> CE2Result:
    """Build a TPM from integer-index trajectories (via _build_tpm) then score it."""
    tpm = _build_tpm(trajectories, num_states)
    return compute_ce2_from_tpm(tpm, eps=eps)


def frozen_trajectory_ce2_value(num_states: int, traj_len: int,
                                eps: float = 1e-9) -> float:
    """
    The CE 2.0 value a fully frozen (single-state) trajectory produces.

    This is a CEILING, not a floor, and it must NOT be subtracted. CE 2.0 decreases
    monotonically as a trajectory gets richer, because sigma_2 minus gamma_star
    measures the coarse-graining gain still AVAILABLE, and a maximally degenerate
    microscale has the most available. Measured over 2000 steps with 243 states:

        distinct states:      1        2        10       50       243
        CE 2.0:               0.8779   0.7978   0.4329   0.1108   0.0148

    So an observed value at or near this reference means the input trajectory is
    degenerate, NOT that the level is strongly emergent. Use it as a diagnostic:
    compare an observed CE 2.0 against this reference to detect a frozen input.

    On the 2026-07 dark_room pilot the gate level (243 states, 2000-step window)
    matched this reference exactly (0.877884) in all 50 windows, and the workspace
    level (8 states) matched at 0.662874, proving both trajectories were frozen.
    See docs/results/ce2_pilot_calibration_2026_07.md.
    """
    frozen = [np.zeros(int(traj_len), dtype=np.int64)]
    return compute_ce2_from_trajectories(frozen, num_states, eps=eps).causal_emergence


def trajectory_degeneracy(trajectories: list[np.ndarray], num_states: int) -> dict:
    """
    Diagnose whether a trajectory set is rich enough for CE 2.0 to be interpretable.

    Because CE 2.0 rises as the input degenerates, a value computed on a frozen or
    near-frozen trajectory is an artifact of the discretization, not a causal
    property of the system. This reports the evidence needed to judge that:

        distinct_states  - how many of num_states the trajectory actually visits
        n_transitions    - number of t -> t+1 pairs contributing to the TPM
        coverage         - distinct_states / num_states
        degenerate       - True when fewer than 2 distinct states are visited, so
                           the TPM carries no observed transition structure at all

    A True `degenerate` flag means the accompanying CE 2.0 value should not be read
    as signal.
    """
    visited: set[int] = set()
    n_transitions = 0
    for traj in trajectories:
        arr = np.asarray(traj).reshape(-1)
        visited.update(int(x) for x in arr if 0 <= int(x) < num_states)
        n_transitions += max(0, len(arr) - 1)
    distinct = len(visited)
    return {
        "distinct_states": distinct,
        "n_transitions": n_transitions,
        "coverage": distinct / float(num_states) if num_states else 0.0,
        "degenerate": distinct < 2,
    }


# --------------------------------------------------------------------------- #
# RSSM latent -> TPM extraction
# --------------------------------------------------------------------------- #
def latent_class_indices(z_state: torch.Tensor, latent_mode: str = "discrete",
                         num_bins: int = 32) -> np.ndarray:
    """
    Reduce one RSSM latent time step to a flat 1-D array of per-variable state indices.

    discrete mode: z_state is [B, C, K, H, W] one-hot on the class axis (dim=2).
        argmax over dim=2 gives a [B, C, H, W] field of class indices in [0, K),
        i.e. one categorical variable per (batch, category, grid cell). This is the
        primary path and the one the instruction targets ("discrete categorical
        RSSM latent").

    continuous mode (secondary, documented as lossy): z_state holds Gaussian means
        of the same [B, C, K, H, W] shape. The class axis is collapsed by mean and
        each scalar is binned into num_bins via discretize_continuous. Note that
        discretize_continuous clips to [0, 1], so continuous means outside that
        range saturate at the edge bins; this path is an approximation only.

    Returns a flattened int64 numpy array (one entry per categorical variable).
    """
    if z_state.ndim != 5:
        raise ValueError(f"expected z_state [B, C, K, H, W], got {tuple(z_state.shape)}")
    if latent_mode == "discrete":
        idx = z_state.argmax(dim=2)                       # [B, C, H, W]
        return idx.reshape(-1).detach().cpu().numpy().astype(np.int64)
    # continuous secondary path
    vals = z_state.mean(dim=2).reshape(-1).detach().cpu().numpy()
    return np.asarray(discretize_continuous(vals, num_bins), dtype=np.int64)


def build_latent_tpm(step_index_fields: list[np.ndarray], num_states: int) -> np.ndarray:
    """
    Pooled class->class TPM from a per-step list of equal-length index fields.

    Each element is latent_class_indices(...) for one step; column v is the
    trajectory of categorical variable v over time. All variables are pooled into
    one num_states x num_states count matrix via _build_tpm. Intended for offline
    or small-batch use; the training loop uses the incremental counters below to
    stay O(num_states^2) in memory.
    """
    if not step_index_fields:
        return _build_tpm([], num_states)
    stacked = np.stack(step_index_fields, axis=0)         # [T, V]
    trajectories = [stacked[:, v] for v in range(stacked.shape[1])]
    return _build_tpm(trajectories, num_states)


# --------------------------------------------------------------------------- #
# Incremental transition counting (memory-bounded, used by the training loop)
# --------------------------------------------------------------------------- #
def new_transition_counts(num_states: int) -> np.ndarray:
    """Fresh [num_states, num_states] float64 count matrix (unsmoothed)."""
    return np.zeros((num_states, num_states), dtype=np.float64)


def update_transition_counts(counts: np.ndarray, prev_field: np.ndarray,
                             curr_field: np.ndarray) -> np.ndarray:
    """
    Accumulate prev->curr transitions into counts, pooled over all variables.

    prev_field and curr_field are equal-length 1-D int arrays of class indices
    (the same variable v at consecutive time steps). Only paired within a
    trajectory: the caller must NOT pair across episode resets (drop prev at the
    episode boundary), because a reset is not a real dynamical transition.
    """
    prev = np.asarray(prev_field).reshape(-1)
    curr = np.asarray(curr_field).reshape(-1)
    if prev.shape != curr.shape:
        raise ValueError(f"prev/curr shape mismatch: {prev.shape} vs {curr.shape}")
    n = counts.shape[0]
    valid = (prev >= 0) & (prev < n) & (curr >= 0) & (curr < n)
    np.add.at(counts, (prev[valid], curr[valid]), 1.0)
    return counts


def counts_to_tpm(counts: np.ndarray, laplace: float = 1.0) -> np.ndarray:
    """Row-normalize a count matrix into a TPM, with Laplace smoothing (matches
    the smoothing convention in effective_information._build_tpm)."""
    c = np.asarray(counts, dtype=np.float64) + laplace
    return c / c.sum(axis=1, keepdims=True)
