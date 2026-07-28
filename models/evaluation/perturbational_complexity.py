"""
Perturbational Complexity Index (PCI_LZ), after Casali et al. (2013).

Casali, A. G. et al. "A theoretically based index of consciousness independent of
sensory processing and behavior." Science Translational Medicine 5:198ra105 (2013).
Reviewed as the one surviving marker in Koch, Massimini, Boly & Tononi (2016),
Nature Reviews Neuroscience 17:307-321 (doi:10.1038/nrn.2016.22), which reports
gamma synchrony and the P3b as failed markers over the same period.
See docs/thalamic_gating_evidence.md for the review and the alignment audit.

WHY THIS MEASURE AND NOT ANOTHER. PCI perturbs the system and scores the CAUSAL
response for integration (how far the perturbation spreads) and differentiation
(how incompressible the spread is) at once. The 2016 review's stated reason it
succeeds where measures of spontaneous activity fail: because it evaluates
deterministic responses to a perturbation, it is insensitive to random processes
and to locally generated patterns that are not genuinely integrated. That is
exactly the failure mode this repository has documented three times over on
spontaneous trajectories:
  - ei_gates was bit-identical at the constant-trajectory Laplace floor
    (docs/results/instrument_repair_2026_07.md),
  - CE 2.0 gate and workspace values were reproduced exactly by a frozen input
    (docs/results/ce2_pilot_calibration_2026_07.md),
  - phi sits near zero throughout.
PCI does not inherit that failure mode, because it supplies its own variation.

THE THREE REGIMES THIS MEASURE SEPARATES (and how the tests pin them down):
  1. Response is LOCAL or dies out    -> few significant entries -> low PCI.
  2. Response is GLOBAL but STEREOTYPED (every channel does the same periodic
     thing) -> highly compressible -> low PCI.
  3. Response is GLOBAL and NON-REPEATING -> incompressible -> high PCI.

WHERE THE INSENSITIVITY TO RANDOMNESS ACTUALLY COMES FROM. It comes from the
SIGNIFICANCE BINARIZATION in binarize_response(), not from any normalization.
Fluctuations no larger than the channel's own pre-perturbation variability do not
cross threshold, so a response that is pure noise binarizes to almost all zeros
and scores near zero.

DELIBERATE DEVIATION FROM CASALI'S NORMALIZATION (read before comparing to
published PCI values). Casali et al. normalize the LZ count by the asymptotic
complexity of a random source at the OBSERVED activity level, L * H / log2(L).
That term goes to zero as the response gets sparse, so the ratio diverges: on a
32-channel, 64-step response with 4 significant entries out of 2048, it returns
1.32, ranking a dying local response ABOVE a fully differentiated one. The
asymptotic identity it rests on needs a large L * H, which a sparse causal response
does not supply.

This module therefore normalizes by the maximum-entropy asymptote for the matrix
SIZE, L / log2(L), and reports `pci` on that basis. The consequence is explicit:
`pci` retains a dependence on HOW MUCH of the substrate responded. That is
intentional here. Casali divides activity level out because in TMS/EEG the number
of significant sources is confounded by stimulator intensity and electrode
montage; in this repository the perturbation magnitude and the channel set are
fixed by the caller across every comparison, so the confound is absent and spatial
spread is exactly the integration half of what the measure is meant to capture.
`pci_casali` is still computed and reported alongside, so the original quantity is
always recoverable, and `source_entropy` and `active_fraction` are reported so the
sparse regime is visible rather than hidden.

HONESTY CAVEAT. PCI_LZ is validated for TMS-evoked EEG in humans. Applying it to
the internal substrate of a reinforcement-learning agent is exploratory and is NOT
validated by the source. The absolute scale does not transfer: the published
conscious/unconscious cutoff near 0.31 has no meaning here and must never be quoted
against these values. What transfers is the comparative use: the same measure, same
perturbation magnitude, same substrate, across conditions. Single seed is a
hypothesis; no value from this module may be reported as a result without at least
3-seed replication (see the verify-results protocol).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PCIResult",
    "lempel_ziv_complexity",
    "binarize_response",
    "compute_pci",
    "binary_entropy",
]

# Matches the var_floor convention in scripts/training/metrics_logger.py
# (_discretize_gate_window). A channel whose baseline variability is below this is
# treated as inactive rather than allowed to cross threshold on float noise. This
# matters concretely: the gate's `adaptation` node measures std 6.08e-06, pinned at
# 0.0101 (docs/results/gate_binning_2026_07.md). Without a floor it would register
# as significantly responding to any perturbation at all.
DEFAULT_VAR_FLOOR = 1e-4


@dataclass
class PCIResult:
    """
    One perturbation trial's result.

    `pci` is the LZ count over the maximum-entropy asymptote for the matrix size,
    L / log2(L). It runs from ~0 (no response, or a fully compressible one) to ~1
    (a response as incompressible as a random binary matrix of the same size).

    `pci_casali` is the same LZ count under the published normalization,
    c_LZ * log2(L) / (L * H). It is reported for recoverability only and DIVERGES
    for sparse responses; see the module docstring. Do not rank conditions by it.

    A PCI of 0.0 with active_fraction 0.0 means the perturbation had no measurable
    causal effect at all, which is a finding rather than a failed measurement.
    """

    pci: float
    pci_casali: float
    lz_complexity: int
    normalizer: float
    source_entropy: float
    active_fraction: float
    n_channels: int
    n_timesteps: int


def lempel_ziv_complexity(sequence: np.ndarray | list[int]) -> int:
    """
    LZ76 exhaustive-history complexity (Lempel & Ziv 1976), the variant used by the
    PCI literature via Kaspar & Schuster's formulation.

    Counts the number of distinct patterns produced when the sequence is parsed
    left to right, each new component being the shortest prefix not yet seen in the
    history.

    Args:
        sequence: 1-D sequence of ints (any alphabet; PCI uses {0, 1}).

    Returns:
        The number of parsed components. A constant sequence of length >= 2 gives 2.

    Worked examples (hand-parseable, and the values asserted in the tests):
        "0000" -> "0" | "000"        -> 2
        "01"   -> "0" | "1"          -> 2
        "0010" -> "0" | "01" | "0"   -> 3
        "0101" -> "0" | "1" | "01"   -> 3
    """
    s = np.asarray(sequence).ravel()
    n = int(s.size)
    if n == 0:
        return 0
    if n == 1:
        return 1

    i = 0
    k = 1
    l = 1
    k_max = 1
    complexity = 1

    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                complexity += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                complexity += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    return int(complexity)


def binary_entropy(p: float) -> float:
    """Shannon entropy in bits of a Bernoulli(p) source. 0 at p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def binarize_response(
    response: np.ndarray,
    baseline: np.ndarray,
    threshold_sigma: float = 3.0,
    var_floor: float = DEFAULT_VAR_FLOOR,
) -> np.ndarray:
    """
    Mark each (channel, timestep) as significantly responding or not.

    A channel is significant at time t when the magnitude of its causal response
    exceeds `threshold_sigma` times its own pre-perturbation variability. This is
    the step that makes PCI insensitive to noise: a response no larger than the
    channel's ordinary fluctuation does not count.

    Args:
        response: [n_channels, n_timesteps]. The CAUSAL response, i.e. the
            perturbed trajectory minus an unperturbed replay from the identical
            state with the identical action stream. Passing a raw trajectory here
            instead of a difference measures ongoing activity, not causation, and
            is a misuse of the measure.
        baseline: [n_channels, n_baseline_steps]. Pre-perturbation fluctuation of
            the same channels, used per channel to set the threshold.
        threshold_sigma: multiples of the per-channel baseline standard deviation.
        var_floor: channels whose baseline std is at or below this are treated as
            inactive and never mark significant. See DEFAULT_VAR_FLOOR.

    Returns:
        A uint8 array of shape [n_channels, n_timesteps] with entries in {0, 1}.
    """
    response = np.atleast_2d(np.asarray(response, dtype=np.float64))
    baseline = np.atleast_2d(np.asarray(baseline, dtype=np.float64))

    if response.shape[0] != baseline.shape[0]:
        raise ValueError(
            f"channel mismatch: response has {response.shape[0]} channels, "
            f"baseline has {baseline.shape[0]}"
        )
    if baseline.shape[1] < 2:
        raise ValueError("baseline needs at least 2 timesteps to estimate variability")

    sigma = baseline.std(axis=1, ddof=1)
    dead = sigma <= var_floor
    thresholds = threshold_sigma * sigma
    # A dead channel gets an unreachable threshold rather than a near-zero one.
    thresholds[dead] = np.inf

    return (np.abs(response) > thresholds[:, None]).astype(np.uint8)


def compute_pci(
    response: np.ndarray,
    baseline: np.ndarray,
    threshold_sigma: float = 3.0,
    var_floor: float = DEFAULT_VAR_FLOOR,
) -> PCIResult:
    """
    Compute PCI_LZ for one perturbation trial.

    The binarized spatiotemporal response is flattened time-major (all channels at
    t=0, then all channels at t=1, ...), so spatial structure at each instant stays
    contiguous. LZ parsing is order dependent, so this ordering is a convention and
    is held fixed across every comparison made with this module.

    `pci` divides the LZ count by L / log2(L), the maximum-entropy asymptote for a
    binary string of that length. `pci_casali` divides instead by L * H / log2(L),
    the published normalization, which diverges for sparse responses. The module
    docstring states why the headline uses the former.

    Args:
        response: [n_channels, n_timesteps] causal response (see binarize_response).
        baseline: [n_channels, n_baseline_steps] pre-perturbation fluctuation.
        threshold_sigma: significance threshold in baseline standard deviations.
        var_floor: dead-channel floor.

    Returns:
        PCIResult. Both `pci` and `pci_casali` are 0.0 when no entry crosses
        threshold or when every entry does, since either case carries zero source
        entropy and therefore no differentiation.
    """
    binary = binarize_response(
        response, baseline, threshold_sigma=threshold_sigma, var_floor=var_floor
    )
    n_channels, n_timesteps = binary.shape
    length = int(binary.size)

    active_fraction = float(binary.mean()) if length else 0.0
    entropy = binary_entropy(active_fraction)

    # L < 2 leaves log2(L) undefined or zero; H == 0 means an all-quiet or
    # all-active response, which carries no differentiation either way.
    if length < 2 or entropy <= 0.0:
        return PCIResult(
            pci=0.0,
            pci_casali=0.0,
            lz_complexity=lempel_ziv_complexity(binary.T.ravel()) if length else 0,
            normalizer=0.0,
            source_entropy=entropy,
            active_fraction=active_fraction,
            n_channels=n_channels,
            n_timesteps=n_timesteps,
        )

    # Time-major flatten: binary is [channels, time], so transpose first.
    complexity = lempel_ziv_complexity(binary.T.ravel())
    log_length = np.log2(length)
    normalizer = length / log_length
    casali_normalizer = (length * entropy) / log_length

    return PCIResult(
        pci=float(complexity / normalizer),
        pci_casali=float(complexity / casali_normalizer),
        lz_complexity=complexity,
        normalizer=float(normalizer),
        source_entropy=entropy,
        active_fraction=active_fraction,
        n_channels=n_channels,
        n_timesteps=n_timesteps,
    )
