"""
Protected wrapper around thermospike, this project's open-source sampler library
(NumPy, Python 3.10+, https://github.com/tlcdv/thermospike).

thermospike is optional and never in requirements.txt. Every function imports it on
call; without it, the call raises ImportError naming the missing package. No function
returns a fallback sample.

Convention. This project uses spins m in {-1, +1} with
H(m) = -sum_{i<j} J_ij m_i m_j - sum_i b_i m_i (ising_energy). thermospike uses neurons
z in {0, 1} with E(z) = -1/2 z.W.z - c.z. Substituting m = 2z - 1 gives
W = 4J and c = 2b - 2 * (row sums of J), and H(2z - 1) = E(z) + constant, so both
describe the same distribution exp(-beta H) at the same beta.

thermospike offers two samplers. "pbit" is sequential Gibbs sampling with p-bits.
"spiking" is neural sampling with an absolute refractory period of tau steps
(Buesing et al. 2011); with tau = 1 it equals "pbit". Each call also returns the
event counts of the run (random bits, spikes, synaptic events, multiply adds). They
are counts, not joules. This samples in software and touches no hardware.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from typing import Dict, Tuple

import numpy as np
import torch

from models.thermodynamic.p_bit_emulator import _check_couplings

SAMPLERS = ("pbit", "spiking")
NOISE_SOURCES = ("numpy", "lfsr")


def thermospike_available() -> bool:
    """True when thermospike can be imported."""
    return importlib.util.find_spec("thermospike") is not None


def _require_thermospike():
    if not thermospike_available():
        raise ImportError("thermospike is not installed; it needs Python 3.10+: pip install thermospike")
    import thermospike

    return thermospike


def ising_to_boltzmann(J: torch.Tensor, b: torch.Tensor):
    """The thermospike BoltzmannMachine with the same distribution as the +-1 Ising model (J, b)."""
    _check_couplings(J)
    thermospike = _require_thermospike()
    j = J.detach().to(torch.float64).cpu().numpy()
    bias = 2.0 * b.detach().to(torch.float64).cpu().numpy() - 2.0 * j.sum(axis=1)
    return thermospike.BoltzmannMachine(4.0 * j, bias)


def _build_sampler(thermospike, model, beta: float, seed: int, sampler: str, tau: int, noise: str):
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    if noise not in NOISE_SOURCES:
        raise ValueError(f"noise must be one of {NOISE_SOURCES}, got {noise!r}")
    # An LFSR seed must be nonzero, so seed 0 maps to 1.
    source = thermospike.LFSRNoise(seed=seed + 1) if noise == "lfsr" else thermospike.NumpyNoise(seed=seed)
    if sampler == "pbit":
        return thermospike.PBitSampler(model, source, beta=beta)
    return thermospike.SpikingSampler(model, source, tau=tau, beta=beta)


def sample_with_thermospike(
    J: torch.Tensor,
    b: torch.Tensor,
    beta: float,
    n_samples: int,
    burn_in: int,
    seed: int,
    sampler: str = "pbit",
    tau: int = 1,
    noise: str = "numpy",
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """One chain of thermospike samples as +-1, shape [n_samples, n], and the event counts of the run.

    The counts include the burn in sweeps.
    """
    thermospike = _require_thermospike()
    model = ising_to_boltzmann(J, b)
    chain = _build_sampler(thermospike, model, beta, seed, sampler, tau, noise)
    neurons = chain.sample(n_samples, burn_in=burn_in)
    spins = torch.from_numpy(2.0 * neurons.astype(np.float64) - 1.0)
    return spins, dataclasses.asdict(chain.counters)
