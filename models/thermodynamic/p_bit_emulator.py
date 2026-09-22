"""
P-bit (Ising) emulation in PyTorch: energy, graph coloring, exact enumeration, and a
chromatic block Gibbs sampler.

A p-bit m_i in {-1, +1} updates by the rule of Camsari et al. (PRX 7, 031014, 2017):

    m_i = sgn(tanh(beta * h_i) - r_i),  h_i = b_i + sum_j J_ij m_j,  r_i ~ U(-1, 1)

so P(m_i = +1) = (1 + tanh(beta * h_i)) / 2. beta is a dimensionless gain (inverse
sampling temperature), not the physical temperature of any chip.

Two p-bits that share a coupling must not update at the same time: a simultaneous
update does not sample the Boltzmann distribution. The sampler therefore updates one
color class of a proper graph coloring at a time (chromatic block Gibbs), and it
rejects a coloring that puts coupled spins in one class.

This is software emulation only. It says nothing about any physical device.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

MAX_EXACT_SPINS = 20


def _check_couplings(J: torch.Tensor) -> None:
    if J.dim() != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"couplings must be a square matrix, got shape {tuple(J.shape)}")
    if not torch.equal(J, J.T):
        raise ValueError("couplings must be symmetric (J_ij == J_ji)")
    if torch.count_nonzero(torch.diagonal(J)) > 0:
        raise ValueError("couplings must have a zero diagonal (no self-coupling)")


def ising_energy(spins: torch.Tensor, J: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """H(m) = -sum_{i<j} J_ij m_i m_j - sum_i b_i m_i, over the last dimension of spins."""
    _check_couplings(J)
    pair_term = 0.5 * torch.einsum("...i,ij,...j->...", spins, J, spins)
    return -pair_term - spins @ b


def grid_couplings(
    rows: int, cols: int, offsets: Sequence[Tuple[int, int]], coupling: float, periodic: bool = True
) -> torch.Tensor:
    """Couplings for a rows x cols grid; each offset (dr, dc) links a site to two neighbors.

    Extropic's public material gives the Z1 a sparse graph of degree 16 but not its
    layout, so any 16-neighbor offset set used with this function is a stand-in.
    """
    n = rows * cols
    J = torch.zeros(n, n, dtype=torch.float64)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if periodic:
                    rr, cc = rr % rows, cc % cols
                elif not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                J[r * cols + c, rr * cols + cc] = J[rr * cols + cc, r * cols + c] = coupling
    return J


def greedy_coloring(J: torch.Tensor) -> List[torch.Tensor]:
    """Proper coloring of the coupling graph: no two coupled spins share a class."""
    _check_couplings(J)
    n = J.shape[0]
    colors = [-1] * n
    for i in range(n):
        taken = {colors[j] for j in torch.nonzero(J[i]).flatten().tolist()}
        colors[i] = next(c for c in range(n) if c not in taken)
    color_tensor = torch.tensor(colors)
    return [torch.nonzero(color_tensor == c).flatten() for c in range(max(colors) + 1)]


def exact_boltzmann(J: torch.Tensor, b: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """All 2^n states (as +-1) and their exact probabilities exp(-beta H) / Z. n <= 20."""
    n = J.shape[0]
    if n > MAX_EXACT_SPINS:
        raise ValueError(f"exact enumeration is limited to {MAX_EXACT_SPINS} spins, got {n}")
    codes = torch.arange(2 ** n).unsqueeze(1)
    bits = (codes >> torch.arange(n - 1, -1, -1)) & 1
    states = (2 * bits - 1).to(torch.float64)
    energies = ising_energy(states, J.to(torch.float64), b.to(torch.float64))
    return states, torch.softmax(-beta * energies, dim=0)


def _check_coloring(J: torch.Tensor, coloring: Sequence[torch.Tensor]) -> None:
    covered = torch.cat(list(coloring)).sort().values
    if not torch.equal(covered, torch.arange(J.shape[0])):
        raise ValueError("coloring must list every spin exactly once")
    for members in coloring:
        if torch.count_nonzero(J[members][:, members]) > 0:
            raise ValueError("coloring puts coupled spins in one class; they would update together")


class BlockGibbsSampler:
    """Chromatic block Gibbs sampler for p-bits on a sparse, symmetric coupling graph."""

    def __init__(
        self,
        J: torch.Tensor,
        b: torch.Tensor,
        beta: float,
        seed: Optional[int] = None,
        coloring: Optional[Sequence[torch.Tensor]] = None,
    ):
        _check_couplings(J)
        self.coloring = list(coloring) if coloring is not None else greedy_coloring(J)
        _check_coloring(J, self.coloring)
        self.J, self.b, self.beta = J, b.to(J.dtype), beta
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def random_spins(self, n_chains: int) -> torch.Tensor:
        """Uniform random +-1 start states, one row per chain."""
        coin = torch.randint(0, 2, (n_chains, self.J.shape[0]), generator=self.generator)
        return (2 * coin - 1).to(self.J.dtype)

    def step(self, spins: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
        """One sweep: update every color class once, in order. Updates spins in place."""
        gain = self.beta if beta is None else beta
        for members in self.coloring:
            field = self.b[members] + spins @ self.J[:, members]
            noise = 2 * torch.rand(field.shape, generator=self.generator, dtype=field.dtype) - 1
            spins[:, members] = torch.where(torch.tanh(gain * field) > noise, 1.0, -1.0).to(spins.dtype)
        return spins

    def sample(self, n_chains: int, n_sweeps: int, burn_in: int, thin: int) -> torch.Tensor:
        """Samples of shape [kept sweeps, n_chains, n] after burn_in, keeping every thin-th sweep."""
        spins = self.random_spins(n_chains)
        for _ in range(burn_in):
            self.step(spins)
        kept = []
        for sweep in range(n_sweeps):
            self.step(spins)
            if sweep % thin == 0:
                kept.append(spins.clone())
        return torch.stack(kept)

    def anneal(self, n_chains: int, betas: torch.Tensor) -> torch.Tensor:
        """One sweep per gain in betas (in order); returns the final spins [n_chains, n]."""
        spins = self.random_spins(n_chains)
        for beta in betas.tolist():
            self.step(spins, beta=beta)
        return spins
