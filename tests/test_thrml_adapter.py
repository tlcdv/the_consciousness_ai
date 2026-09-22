"""
Tests for the THRML wrapper (models/thermodynamic/interfaces/thrml_adapter.py).

THRML needs Python 3.10+ and JAX and is not in requirements.txt, so the sampling
tests skip where it is missing. Where it is present, THRML's samples must match
the exact Boltzmann distribution computed here by enumeration, which also pins
the sign convention of the coupling transfer.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.interfaces.thrml_adapter import (  # noqa: E402
    sample_with_thrml,
    thrml_available,
)
from models.thermodynamic.p_bit_emulator import exact_boltzmann  # noqa: E402


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


@pytest.mark.skipif(thrml_available(), reason="only meaningful where thrml is missing")
def test_missing_thrml_raises_instead_of_sampling():
    J, b = _frustrated_graph()
    with pytest.raises(ImportError, match="thrml"):
        sample_with_thrml(J, b, 0.7, n_chains=2, n_samples=2, n_warmup=1, steps_per_sample=1, seed=0)


class TestWithThrml:
    @pytest.fixture(autouse=True)
    def _needs_thrml(self):
        pytest.importorskip("thrml")

    def test_samples_match_exact_boltzmann(self):
        J, b = _frustrated_graph()
        states, probs = exact_boltzmann(J, b, 0.7)

        samples = sample_with_thrml(J, b, 0.7, n_chains=2000, n_samples=40, n_warmup=50, steps_per_sample=2, seed=1)

        assert samples.shape == (2000, 40, 6)
        assert _total_variation(samples.reshape(-1, 6), states, probs) < 0.03

    def test_spins_are_plus_or_minus_one(self):
        J, b = _frustrated_graph()

        samples = sample_with_thrml(J, b, 0.7, n_chains=4, n_samples=3, n_warmup=2, steps_per_sample=1, seed=2)

        assert set(samples.unique().tolist()) <= {-1.0, 1.0}
