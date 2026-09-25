"""
Traveling-wave measures for step-indexed spatial fields.

UNITS. The agent has no millisecond clock. Time is the environment step and
space is the tectum grid cell, so a wave speed is in cells per step and a
frequency is in cycles per step. There is no Hz and no metres per second here,
for the reason given in coupling_measures.py and docs/thalamic_gating_evidence.md
section 4.

WHAT IT MEASURES. Two complementary indices, because no single index detects
every wave.

  phase gradient directionality (PGD)   |mean phase gradient| / mean |gradient|
      over the grid at one step. It is 1 for a plane wave and near 0 for
      independent noise. It is ALSO near 0 for spiral and target waves,
      because their local directions cancel. A low PGD therefore does not
      mean "no wave". The index follows Rubino, Robbins and Hatsopoulos
      (2006, Nat. Neurosci.).
  phase singularities   the number of 2 x 2 plaquettes around which the wrapped
      phase winds by 2 pi. A spiral has one. A plane wave has none. Noise has
      many, so this count is read against the shuffle null too.

NULL. A spatial shuffle moves every cell's whole time series to a random grid
position. Each cell keeps its own signal, amplitude and spectrum, and the
spatial layout is destroyed. Muller et al. (2026, Neuron, Box 2) name two
confounds for wave detection. Narrowband filtering can create artificial
waves, and coordinated but non-propagating fluctuations can look like waves.
The default here applies no band filter, and the shuffle null keeps
coordinated amplitude changes while breaking propagation.

HONESTY CAVEAT. These are standard estimators applied to an artificial
substrate. A result is a diagnostic until it replicates on 3 seeds against the
null, per the verify-results protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.evaluation.coupling_measures import analytic_phase

__all__ = [
    "WaveResult",
    "count_phase_singularities",
    "measure_waves",
    "phase_field",
    "phase_gradient_directionality",
    "project_channels",
]

EDGE_FRACTION = 0.125   # Hilbert edge samples dropped from each end
MIN_STEPS = 16


@dataclass
class WaveResult:
    """PGD and singularity summary of one field against its shuffle null."""
    pgd_mean: float
    null_mean: float
    null_p95: float
    singularities_median: float
    null_singularities_median: float
    steps_used: int


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def phase_field(signal: np.ndarray, band: tuple | None = None) -> np.ndarray:
    """Instantaneous phase per cell, [T, H, W] radians, from a [T, H, W] signal."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"signal must be [T, H, W], got shape {arr.shape}")
    T, H, W = arr.shape
    flat = arr.reshape(T, H * W)
    phases = np.stack([analytic_phase(flat[:, i], band) for i in range(H * W)], axis=1)
    return phases.reshape(T, H, W)


def _gradients(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrapped central differences on the grid interior, each [T, H-2, W-2]."""
    gx = _wrap(phases[:, 1:-1, 2:] - phases[:, 1:-1, :-2]) / 2
    gy = _wrap(phases[:, 2:, 1:-1] - phases[:, :-2, 1:-1]) / 2
    return gx, gy


def phase_gradient_directionality(phases: np.ndarray) -> np.ndarray:
    """PGD per step, [T]. NaN at a step whose gradients are all zero."""
    gx, gy = _gradients(np.asarray(phases, dtype=np.float64))
    mean_norm = np.hypot(gx.mean(axis=(1, 2)), gy.mean(axis=(1, 2)))
    norm_mean = np.hypot(gx, gy).mean(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm_mean > 0, mean_norm / norm_mean, np.nan)


def count_phase_singularities(phase: np.ndarray) -> int:
    """Number of 2 x 2 plaquettes with a 2 pi phase winding, for one [H, W] step."""
    p = np.asarray(phase, dtype=np.float64)
    loop = (_wrap(p[:-1, 1:] - p[:-1, :-1]) + _wrap(p[1:, 1:] - p[:-1, 1:])
            + _wrap(p[1:, :-1] - p[1:, 1:]) + _wrap(p[:-1, :-1] - p[1:, :-1]))
    return int(np.sum(np.abs(np.round(loop / (2 * np.pi))) >= 1))


def project_channels(h: np.ndarray) -> np.ndarray:
    """Project a [T, C, H, W] state onto its first principal channel direction.

    The direction is fitted over all steps and cells, so one scalar field
    [T, H, W] carries the channel combination with the largest variance.
    """
    arr = np.asarray(h, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"h must be [T, C, H, W], got shape {arr.shape}")
    T, C, H, W = arr.shape
    samples = arr.transpose(0, 2, 3, 1).reshape(-1, C)
    centered = samples - samples.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[0]).reshape(T, H, W)


def _summarize(phases: np.ndarray) -> tuple[float, float]:
    pgd = phase_gradient_directionality(phases)
    sing = [count_phase_singularities(p) for p in phases]
    return float(np.nanmean(pgd)), float(np.median(sing))


def measure_waves(signal: np.ndarray, band: tuple | None = None,
                  n_surrogates: int = 50, seed: int = 0) -> WaveResult:
    """PGD and singularities of a [T, H, W] field against a spatial-shuffle null.

    Raises ValueError for fewer than MIN_STEPS steps or for a field with a
    constant cell, where no phase is defined.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < MIN_STEPS:
        raise ValueError(f"signal must be [T >= {MIN_STEPS}, H, W], got {arr.shape}")
    if np.any(arr.std(axis=0) == 0):
        raise ValueError("a cell is constant over time, so its phase is undefined")
    T, H, W = arr.shape
    edge = int(T * EDGE_FRACTION)
    phases = phase_field(arr, band)[edge:T - edge]
    pgd_mean, sing_median = _summarize(phases)

    rng = np.random.default_rng(seed)
    flat = phases.reshape(phases.shape[0], H * W)
    null_pgd, null_sing = [], []
    for _ in range(n_surrogates):
        shuffled = flat[:, rng.permutation(H * W)].reshape(phases.shape)
        p, s = _summarize(shuffled)
        null_pgd.append(p)
        null_sing.append(s)
    return WaveResult(
        pgd_mean=pgd_mean,
        null_mean=float(np.mean(null_pgd)),
        null_p95=float(np.percentile(null_pgd, 95)),
        singularities_median=sing_median,
        null_singularities_median=float(np.median(null_sing)),
        steps_used=int(phases.shape[0]),
    )
