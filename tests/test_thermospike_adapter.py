"""
Tests for the thermospike wrapper (models/thermodynamic/interfaces/thermospike_adapter.py).

thermospike is optional and not in requirements.txt, so these tests skip where it is
missing. Where it is present, the coupling transfer from +-1 spins to 0/1 neurons must
keep every energy difference, and both thermospike samplers must match the exact
Boltzmann distribution computed here by enumeration.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.interfaces.thermospike_adapter import (  # noqa: E402
    ising_to_boltzmann,
    sample_with_thermospike,
    thermospike_available,
)
from models.thermodynamic.p_bit_emulator import exact_boltzmann, ising_energy  # noqa: E402


def _frustrated_graph(n: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    upper = torch.randn(n, n, generator=generator, dtype=torch.float64).triu(1)
    upper[upper.abs() < 0.3] = 0.0
    return upper + upper.T, 0.4 * torch.randn(n, generator=generator, dtype=torch.float64)


def _total_variation(samples: torch.Tensor, states: torch.Tensor, probs: torch.Tensor) -> float:
    n = states.shape[1]
    weights = 2 ** torch.arange(n - 1, -1, -1)
    sample_index = ((samples > 0).long() * weights).sum(dim=1)
    state_index = ((states > 0).long() * weights).sum(dim=1)
    counts = torch.bincount(sample_index, minlength=2 ** n).double()
    return 0.5 * (counts[state_index] / counts.sum() - probs).abs().sum().item()


@pytest.mark.skipif(thermospike_available(), reason="only meaningful where thermospike is missing")
def test_missing_thermospike_raises_instead_of_sampling():
    J, b = _frustrated_graph()
    with pytest.raises(ImportError, match="thermospike"):
        sample_with_thermospike(J, b, 0.7, n_samples=2, burn_in=1, seed=0)


class TestWithThermospike:
    @pytest.fixture(autouse=True)
    def _needs_thermospike(self):
        pytest.importorskip("thermospike")

    def test_transfer_keeps_every_energy_difference(self):
        # H(2z - 1) - E(z) must be the same constant for all 2^n states.
        J, b = _frustrated_graph()
        model = ising_to_boltzmann(J, b)
        states, _ = exact_boltzmann(J, b, 1.0)
        neurons = (states + 1) / 2

        gap = ising_energy(states, J, b) - torch.tensor([model.energy(z.numpy()) for z in neurons])

        assert torch.allclose(gap, gap[0].expand_as(gap), atol=1e-9)

    @pytest.mark.parametrize("sampler,tau", [("pbit", 1), ("spiking", 3)])
    def test_samples_match_exact_boltzmann(self, sampler, tau):
        J, b = _frustrated_graph()
        states, probs = exact_boltzmann(J, b, 0.7)

        spins, _ = sample_with_thermospike(J, b, 0.7, n_samples=40000, burn_in=200, seed=1,
                                           sampler=sampler, tau=tau)

        assert spins.shape == (40000, 6)
        assert _total_variation(spins, states, probs) < 0.03

    def test_spins_are_plus_or_minus_one_and_costs_are_counted(self):
        J, b = _frustrated_graph()

        spins, counters = sample_with_thermospike(J, b, 0.7, n_samples=50, burn_in=5, seed=2, noise="lfsr")

        assert set(spins.unique().tolist()) <= {-1.0, 1.0}
        assert counters["updates"] == 55 * 6
        assert counters["random_bits"] == 16 * counters["random_draws"]

    def test_unknown_sampler_is_rejected(self):
        J, b = _frustrated_graph()
        with pytest.raises(ValueError, match="sampler"):
            sample_with_thermospike(J, b, 0.7, n_samples=2, burn_in=1, seed=0, sampler="annealer")
