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
