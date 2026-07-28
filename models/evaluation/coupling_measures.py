"""
Directional and cross-scale coupling measures for step-indexed internal signals.

UNITS WARNING, READ FIRST. This agent has no millisecond clock. Its time axes are
the environment step and the two to ten settle cycles inside
ReentrantProcessor.settle. Neither is a sampling rate. Every "frequency" here is
therefore in CYCLES PER STEP, a dimensionless quantity, and carries NO Hz
interpretation and NO correspondence to any published frequency band.

Specifically: do not write "the agent's theta band", do not map a slow_band here
onto the 2-8 Hz band in Fang et al. (2024), and do not map a fast band onto the
20-45 Hz oscillation in Chowdhury et al. (2025). The correspondence is unknown and
is not established by using the same measure. This constraint is stated in
docs/thalamic_gating_evidence.md section 4 and is binding on every doc, column name
and verdict derived from this module.

WHAT THESE ARE FOR. Fang et al. (2024, bioRxiv 2024.04.02.587714, PREPRINT, not
peer reviewed) report three properties of a thalamic gate that are measurable
rather than merely assertable:

  ordering      phase locking rises within the hub first, then hub to cortex,
                then within cortex (measured with phase_locking_value)
  direction     phase information flows hub to cortex more than the reverse
                (measured with phase_transfer_entropy)
  cross-scale   the hub's slow phase modulates the amplitude of faster activity
                elsewhere, dissociably from the phase locking
                (measured with phase_amplitude_coupling)

The point of implementing them BEFORE any hub exists is that they can be pointed at
the current workspace first. If the existing architecture already shows the ordering
and the directional asymmetry, a hub adds nothing and should not be built.

HONESTY CAVEAT. These are standard neurophysiology estimators applied to a substrate
they were not designed for, on short records, with no established null model for
this system. Treat outputs as diagnostics. Single seed is a hypothesis; no value
from this module may be reported as a result without at least 3-seed replication
(see the verify-results protocol). Phase transfer entropy in particular is biased at
small sample sizes, which is why phase_transfer_entropy returns the raw value and a
surrogate-corrected value, and only the corrected one should be compared across
conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PTEResult",
    "analytic_phase",
    "bandpass",
    "amplitude_envelope",
    "phase_locking_value",
    "phase_transfer_entropy",
    "phase_amplitude_coupling",
]


def _as_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 4:
        raise ValueError(f"{name} needs at least 4 samples, got {arr.size}")
    return arr


def _hilbert(x: np.ndarray) -> np.ndarray:
    """
    Analytic signal via FFT. Implemented here rather than imported from
    scipy.signal so this module has no new dependency and its behaviour on short,
    even and odd length records is pinned by the tests.
    """
    n = x.size
    spectrum = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def bandpass(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Zero-phase brick-wall band filter in cycles per step.

    Args:
        x: 1-D signal sampled once per step.
        low, high: band edges in CYCLES PER STEP, each in [0, 0.5]. 0.5 is the
            Nyquist limit of a once-per-step signal. See the units warning.

    Returns:
        The band-limited signal, real valued, same length as x.
    """
    arr = _as_1d(x, "x")
    if not (0.0 <= low < high <= 0.5):
        raise ValueError(
            f"band edges must satisfy 0 <= low < high <= 0.5 cycles/step, "
            f"got low={low}, high={high}"
        )
    freqs = np.fft.fftfreq(arr.size, d=1.0)
    spectrum = np.fft.fft(arr)
    spectrum[(np.abs(freqs) < low) | (np.abs(freqs) > high)] = 0.0
    return np.real(np.fft.ifft(spectrum))


def analytic_phase(x: np.ndarray, band: tuple | None = None) -> np.ndarray:
    """Instantaneous phase in radians. Optionally band-limit first (cycles/step)."""
    arr = _as_1d(x, "x")
    if band is not None:
        arr = bandpass(arr, band[0], band[1])
    return np.angle(_hilbert(arr))


def amplitude_envelope(x: np.ndarray, band: tuple | None = None) -> np.ndarray:
    """Instantaneous amplitude envelope. Optionally band-limit first."""
    arr = _as_1d(x, "x")
    if band is not None:
        arr = bandpass(arr, band[0], band[1])
    return np.abs(_hilbert(arr))


def phase_locking_value(
    x: np.ndarray, y: np.ndarray, band: tuple | None = None
) -> float:
    """
    Phase locking value between two signals (Lachaux et al. 1999).

    PLV is the magnitude of the mean unit vector of the phase difference: 1.0 when
    the phase lag is perfectly constant (whatever its value), 0.0 when the lag is
    uniformly distributed. It is symmetric and says nothing about direction; use
    phase_transfer_entropy for that.

    Returns:
        A float in [0, 1].
    """
    px = analytic_phase(x, band)
    py = analytic_phase(y, band)
    if px.size != py.size:
        raise ValueError(f"length mismatch: {px.size} vs {py.size}")
    return float(np.abs(np.mean(np.exp(1j * (px - py)))))


@dataclass
class PTEResult:
    """
    Directed phase coupling from source to target.

    `pte` is the raw estimate, which is positively biased on short records.
    `corrected` subtracts the mean of a surrogate null built by circularly shifting
    the source phase, which destroys the timing relationship while preserving each
    signal's own distribution. **Compare conditions on `corrected`, never on `pte`.**
    `surrogate_mean` and `surrogate_std` are reported so the size of the bias is
    visible rather than hidden.
    """

    pte: float
    corrected: float
    surrogate_mean: float
    surrogate_std: float
    n_bins: int
    n_surrogates: int


def phase_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    band: tuple | None = None,
    delay: int = 1,
    n_bins: int = 8,
    n_surrogates: int = 20,
    seed: int = 0,
) -> PTEResult:
    """
    Phase transfer entropy from source to target (Lobier et al. 2014).

    Estimates how much knowing the source's past phase reduces uncertainty about
    the target's future phase, beyond what the target's own past already explains:

        PTE = H(target_future, target_past) + H(target_past, source_past)
              - H(target_past) - H(target_future, target_past, source_past)

    computed on phases discretized into `n_bins` equal angular bins.

    Asymmetry is the quantity of interest: compare PTE(a -> b).corrected against
    PTE(b -> a).corrected. A single value in isolation is not interpretable.

    Args:
        source, target: 1-D signals of equal length.
        band: optional band limits in CYCLES PER STEP (see the units warning).
        delay: prediction horizon in steps.
        n_bins: angular bins for the phase histogram.
        n_surrogates: circular-shift surrogates for the bias correction.
        seed: fixed so the surrogate null is reproducible.
    """
    ps = analytic_phase(source, band)
    pt = analytic_phase(target, band)
    if ps.size != pt.size:
        raise ValueError(f"length mismatch: {ps.size} vs {pt.size}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")
    if ps.size <= delay + 1:
        raise ValueError("signal too short for the requested delay")

    raw = _pte_from_phases(ps, pt, delay, n_bins)

    rng = np.random.default_rng(seed)
    surrogates = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        shift = int(rng.integers(1, ps.size))
        surrogates[i] = _pte_from_phases(np.roll(ps, shift), pt, delay, n_bins)

    surrogate_mean = float(surrogates.mean()) if n_surrogates else 0.0
    surrogate_std = float(surrogates.std(ddof=1)) if n_surrogates > 1 else 0.0

    return PTEResult(
        pte=raw,
        corrected=float(raw - surrogate_mean),
        surrogate_mean=surrogate_mean,
        surrogate_std=surrogate_std,
        n_bins=n_bins,
        n_surrogates=n_surrogates,
    )


def _discretize_phase(phase: np.ndarray, n_bins: int) -> np.ndarray:
    """Map phases in [-pi, pi) onto equal angular bins 0..n_bins-1."""
    shifted = (phase + np.pi) / (2.0 * np.pi)
    return np.clip((shifted * n_bins).astype(np.int64), 0, n_bins - 1)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _pte_from_phases(
    source_phase: np.ndarray, target_phase: np.ndarray, delay: int, n_bins: int
) -> float:
    s_past = _discretize_phase(source_phase[:-delay], n_bins)
    t_past = _discretize_phase(target_phase[:-delay], n_bins)
    t_future = _discretize_phase(target_phase[delay:], n_bins)

    # Joint histograms via a mixed-radix index, avoiding an N-D histogram call.
    h_tp = _entropy_from_counts(np.bincount(t_past, minlength=n_bins))
    h_tf_tp = _entropy_from_counts(
        np.bincount(t_future * n_bins + t_past, minlength=n_bins**2)
    )
    h_tp_sp = _entropy_from_counts(
        np.bincount(t_past * n_bins + s_past, minlength=n_bins**2)
    )
    h_tf_tp_sp = _entropy_from_counts(
        np.bincount(
            (t_future * n_bins + t_past) * n_bins + s_past, minlength=n_bins**3
        )
    )
    return float(h_tf_tp + h_tp_sp - h_tp - h_tf_tp_sp)


def phase_amplitude_coupling(
    phase_signal: np.ndarray,
    amplitude_signal: np.ndarray,
    phase_band: tuple,
    amplitude_band: tuple,
    n_bins: int = 18,
) -> float:
    """
    Modulation index (Tort et al. 2010): how much the slow phase of one signal
    modulates the fast amplitude envelope of another.

    The amplitude envelope is averaged within equal phase bins and the resulting
    distribution is compared to uniform by normalized Kullback-Leibler divergence.
    0.0 means the amplitude is flat across the phase cycle (no coupling); larger
    values mean amplitude concentrates at a preferred phase.

    Direction matters and is carried by argument order: this measures
    phase_signal's phase driving amplitude_signal's amplitude. Compute both
    orderings to test an asymmetry.

    Args:
        phase_signal: signal whose (slow) phase does the modulating.
        amplitude_signal: signal whose (fast) envelope is modulated.
        phase_band, amplitude_band: band edges in CYCLES PER STEP. The phase band
            must be strictly slower than the amplitude band, which the function
            checks, because the measure is meaningless otherwise.
        n_bins: phase bins.

    Returns:
        A non-negative float. It is 0.0 exactly when the mean envelope is identical
        in every phase bin.
    """
    if phase_band[1] > amplitude_band[0]:
        raise ValueError(
            f"phase_band {phase_band} must be strictly slower than amplitude_band "
            f"{amplitude_band}; overlapping bands make the modulation index "
            f"uninterpretable"
        )

    phase = analytic_phase(phase_signal, phase_band)
    envelope = amplitude_envelope(amplitude_signal, amplitude_band)
    if phase.size != envelope.size:
        raise ValueError(f"length mismatch: {phase.size} vs {envelope.size}")

    bins = _discretize_phase(phase, n_bins)
    mean_amplitude = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            mean_amplitude[b] = envelope[mask].mean()

    total = mean_amplitude.sum()
    if total <= 0:
        return 0.0

    p = mean_amplitude / total
    nonzero = p[p > 0]
    # Normalized KL divergence from uniform: (log(n) - H(p)) / log(n).
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float((np.log(n_bins) - entropy) / np.log(n_bins))
