"""
Tests for the p-bit (Ising) emulator (models/thermodynamic/p_bit_emulator.py).

Every expected value is derived analytically, never copied from a run:

  * Energy: H(m) = -sum_{i<j} J_ij m_i m_j - sum_i b_i m_i. Two coupled spins with
    J = 1 give -1 when aligned and +1 when opposed.
  * One spin with bias b at gain beta: P(m = +1) = (1 + tanh(beta * b)) / 2.
  * beta = 0 gives the uniform distribution over all 2^n states.
  * The mean energy <E> falls as beta rises, because d<E>/d(beta) = -Var(E) <= 0.
  * A ferromagnet with a small positive field has the all +1 state as its only
    ground state, with energy -(number of edges) * J - n * field.

A Gibbs sampler at finite beta does NOT lower the energy at every step, so no test
here asserts a per-step decrease.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.p_bit_emulator import (  # noqa: E402
    BlockGibbsSampler,
    exact_boltzmann,
    grid_couplings,
    greedy_coloring,
    ising_energy,
)

NEAREST_NEIGHBOR_OFFSETS = ((0, 1), (1, 0))


def _random_symmetric_couplings(n: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    upper = torch.randn(n, n, generator=generator, dtype=torch.float64).triu(1) * scale
    return upper + upper.T


def _random_biases(n: int, seed: int, scale: float = 0.5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=generator, dtype=torch.float64) * scale


def _total_variation(samples: torch.Tensor, states: torch.Tensor, probs: torch.Tensor) -> float:
    n = states.shape[1]
    weights = 2 ** torch.arange(n - 1, -1, -1)
    state_index = ((states > 0).long() * weights).sum(dim=1)
    sample_index = ((samples > 0).long() * weights).sum(dim=1)
    counts = torch.bincount(sample_index, minlength=2 ** n).double()
    empirical = torch.zeros_like(probs)
    empirical[state_index] = counts[state_index] / counts.sum()
    return 0.5 * (empirical - probs).abs().sum().item()


class TestIsingEnergy:
    def test_two_spin_ferromagnet_energy(self):
        J = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        b = torch.zeros(2, dtype=torch.float64)
        spins = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], dtype=torch.float64)

        energies = ising_energy(spins, J, b)

        assert energies.tolist() == [-1.0, -1.0, 1.0]

    def test_energy_matches_explicit_pair_sum(self):
        n = 6
        J = _random_symmetric_couplings(n, seed=0)
        b = _random_biases(n, seed=1)
        m = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=torch.float64)

        explicit = -sum(
            J[i, j] * m[i] * m[j] for i in range(n) for j in range(i + 1, n)
        ) - (b * m).sum()

        assert ising_energy(m, J, b).item() == pytest.approx(explicit.item(), abs=1e-12)

    def test_asymmetric_couplings_are_rejected(self):
        J = torch.tensor([[0.0, 1.0], [0.5, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            ising_energy(torch.ones(2), J, torch.zeros(2))

    def test_non_square_couplings_are_rejected(self):
        with pytest.raises(ValueError, match="square"):
            ising_energy(torch.ones(2), torch.zeros(2, 3), torch.zeros(2))

    def test_self_coupling_is_rejected(self):
        J = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="diagonal"):
            ising_energy(torch.ones(2), J, torch.zeros(2))


class TestGreedyColoring:
    def test_no_coupled_pair_shares_a_color(self):
        J = _random_symmetric_couplings(12, seed=2)
        J[J.abs() < 0.8] = 0.0

        classes = greedy_coloring(J)

        for members in classes:
            block = J[members][:, members]
            assert torch.count_nonzero(block) == 0

    def test_every_spin_gets_exactly_one_color(self):
        J = _random_symmetric_couplings(12, seed=3)

        classes = greedy_coloring(J)

        assert sorted(torch.cat(classes).tolist()) == list(range(12))

    def test_even_periodic_lattice_needs_two_colors(self):
        J = grid_couplings(4, 4, NEAREST_NEIGHBOR_OFFSETS, coupling=1.0, periodic=True)

        assert len(greedy_coloring(J)) == 2


class TestGridCouplings:
    def test_periodic_nearest_neighbor_lattice_has_degree_four(self):
        J = grid_couplings(4, 4, NEAREST_NEIGHBOR_OFFSETS, coupling=1.0, periodic=True)

        degrees = torch.count_nonzero(J, dim=1)

        assert degrees.tolist() == [4] * 16
        assert torch.equal(J, J.T)

    def test_open_lattice_corner_has_degree_two(self):
        J = grid_couplings(3, 3, NEAREST_NEIGHBOR_OFFSETS, coupling=1.0, periodic=False)

        assert torch.count_nonzero(J, dim=1).tolist() == [2, 3, 2, 3, 4, 3, 2, 3, 2]

    def test_sixteen_offsets_give_degree_sixteen(self):
        offsets = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (0, 3), (3, 0))

        J = grid_couplings(8, 8, offsets, coupling=1.0, periodic=True)

        assert torch.count_nonzero(J, dim=1).tolist() == [16] * 64


class TestExactBoltzmann:
    def test_probabilities_sum_to_one(self):
        J = _random_symmetric_couplings(5, seed=4)
        _, probs = exact_boltzmann(J, _random_biases(5, seed=5), beta=0.7)

        assert probs.sum().item() == pytest.approx(1.0, abs=1e-12)

    def test_zero_gain_is_uniform(self):
        J = _random_symmetric_couplings(4, seed=6)
        _, probs = exact_boltzmann(J, _random_biases(4, seed=7), beta=0.0)

        assert torch.allclose(probs, torch.full((16,), 1.0 / 16, dtype=torch.float64))

    def test_single_biased_spin_matches_tanh_rule(self):
        beta, bias = 0.8, 0.6
        states, probs = exact_boltzmann(
            torch.zeros(1, 1, dtype=torch.float64), torch.tensor([bias], dtype=torch.float64), beta
        )

        p_up = probs[states[:, 0] > 0].item()

        assert p_up == pytest.approx((1 + torch.tanh(torch.tensor(beta * bias)).item()) / 2)

    def test_more_than_twenty_spins_is_refused(self):
        with pytest.raises(ValueError, match="20"):
            exact_boltzmann(torch.zeros(21, 21), torch.zeros(21), beta=1.0)


class TestBlockGibbsSampler:
    def test_samples_match_exact_boltzmann_on_frustrated_graph(self):
        n, beta = 8, 0.6
        J = _random_symmetric_couplings(n, seed=8)
        b = _random_biases(n, seed=9)
        states, probs = exact_boltzmann(J, b, beta)
        sampler = BlockGibbsSampler(J, b, beta, seed=10)

        samples = sampler.sample(n_chains=4000, n_sweeps=60, burn_in=20, thin=2)

        assert _total_variation(samples.reshape(-1, n), states, probs) < 0.03

    def test_coloring_with_coupled_spins_in_one_class_is_rejected(self):
        J = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="coupled"):
            BlockGibbsSampler(J, torch.zeros(2), beta=1.0, coloring=[torch.tensor([0, 1])])

    def test_coloring_that_misses_a_spin_is_rejected(self):
        with pytest.raises(ValueError, match="exactly once"):
            BlockGibbsSampler(torch.zeros(3, 3), torch.zeros(3), beta=1.0, coloring=[torch.tensor([0, 1])])

    def test_mean_energy_falls_as_gain_rises(self):
        J = _random_symmetric_couplings(10, seed=11)
        b = _random_biases(10, seed=12)

        def mean_energy(beta: float) -> float:
            samples = BlockGibbsSampler(J, b, beta, seed=13).sample(2000, 30, 10, 1)
            return ising_energy(samples.reshape(-1, 10), J, b).mean().item()

        assert mean_energy(0.2) > mean_energy(0.8) > mean_energy(2.0)

    def test_annealing_finds_ferromagnet_ground_state_on_64_spins(self):
        field = 0.1
        J = grid_couplings(8, 8, NEAREST_NEIGHBOR_OFFSETS, coupling=1.0, periodic=True)
        b = torch.full((64,), field, dtype=torch.float64)
        ground_energy = -128.0 - 64 * field
        sampler = BlockGibbsSampler(J, b, beta=0.1, seed=14)

        final = sampler.anneal(n_chains=8, betas=torch.linspace(0.1, 3.0, 400))

        assert ising_energy(final, J, b).min().item() == pytest.approx(ground_energy)

    def test_same_seed_gives_same_samples(self):
        J = _random_symmetric_couplings(6, seed=15)
        b = _random_biases(6, seed=16)

        first = BlockGibbsSampler(J, b, 1.0, seed=17).sample(4, 5, 0, 1)
        second = BlockGibbsSampler(J, b, 1.0, seed=17).sample(4, 5, 0, 1)

        assert torch.equal(first, second)

    def test_spins_stay_plus_or_minus_one(self):
        J = _random_symmetric_couplings(6, seed=18)
        samples = BlockGibbsSampler(J, _random_biases(6, seed=19), 1.0, seed=20).sample(8, 5, 0, 1)

        assert set(samples.unique().tolist()) <= {-1.0, 1.0}
