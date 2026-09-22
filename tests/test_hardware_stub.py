"""
Tests for the stand-in substrate driver (models/thermodynamic/interfaces/hardware_stub.py).

The stub is the driver every test and CI run uses in place of hardware. It must be
deterministic after reset, account simulated latency without sleeping by default,
and satisfy the same BaseSubstrateDriver contract as any real driver.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.interfaces.abstract_substrate import (  # noqa: E402
    BaseSubstrateDriver,
    SubstrateConfig,
)
from models.thermodynamic.interfaces.hardware_stub import StubSubstrateDriver  # noqa: E402
from models.thermodynamic.p_bit_emulator import ising_energy  # noqa: E402

FERRO_PAIR = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)


def _config(temperature: float = 1.0, substrate_type: str = "thermodynamic") -> SubstrateConfig:
    return SubstrateConfig(substrate_type=substrate_type, device_id=None, precision="stochastic", temperature=temperature)


def _driver(latency_s: float = 1e-4, temperature: float = 1.0) -> StubSubstrateDriver:
    return StubSubstrateDriver(_config(temperature), FERRO_PAIR, torch.zeros(2, dtype=torch.float64), latency_s, seed=3)


def run_through(driver: BaseSubstrateDriver, x: torch.Tensor, steps: int) -> torch.Tensor:
    """A caller that only knows the abstract contract."""
    return driver.decode(driver.execute(driver.encode(x), steps=steps))


class TestContract:
    def test_is_a_substrate_driver(self):
        assert isinstance(_driver(), BaseSubstrateDriver)

    def test_encode_maps_sign_to_spins(self):
        spins = _driver().encode(torch.tensor([[0.3, -0.2], [0.0, -1.0]]))

        assert spins.tolist() == [[1.0, -1.0], [1.0, -1.0]]

    def test_energy_is_the_mean_ising_energy(self):
        driver = _driver()
        state = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)

        expected = ising_energy(state, FERRO_PAIR, torch.zeros(2, dtype=torch.float64)).mean().item()

        assert driver.energy(state) == pytest.approx(expected)
        assert driver.energy(state) == pytest.approx(0.0)

    def test_execute_does_not_modify_its_input(self):
        driver = _driver()
        state = torch.tensor([[1.0, -1.0]], dtype=torch.float64)

        driver.execute(state, steps=5)

        assert state.tolist() == [[1.0, -1.0]]

    def test_works_through_the_abstract_interface(self):
        decoded = run_through(_driver(), torch.randn(4, 2), steps=3)

        assert decoded.shape == (4, 2)
        assert set(decoded.unique().tolist()) <= {-1.0, 1.0}


class TestDeterminism:
    def test_reset_reproduces_the_same_trajectory(self):
        driver = _driver()
        x = torch.randn(16, 2, generator=torch.Generator().manual_seed(0))

        first = run_through(driver, x, steps=10)
        driver.reset()
        second = run_through(driver, x, steps=10)

        assert torch.equal(first, second)

    def test_low_temperature_aligns_the_ferromagnetic_pair(self):
        driver = _driver(temperature=0.05)

        decoded = run_through(driver, torch.tensor([[0.5, -0.5]] * 32), steps=20)

        assert torch.all(decoded[:, 0] == decoded[:, 1])


class TestLatency:
    def test_latency_accumulates_per_step_without_sleeping(self):
        driver = _driver(latency_s=2e-3)

        driver.execute(driver.encode(torch.ones(1, 2)), steps=4)

        assert driver.elapsed_s == pytest.approx(8e-3)

    def test_reset_clears_elapsed_time(self):
        driver = _driver()
        driver.execute(driver.encode(torch.ones(1, 2)), steps=4)

        driver.reset()

        assert driver.elapsed_s == 0.0

    @pytest.mark.parametrize("bad", [5e-6, 2e-2])
    def test_latency_outside_ten_microseconds_to_ten_milliseconds_is_rejected(self, bad):
        with pytest.raises(ValueError, match="latency"):
            _driver(latency_s=bad)

    def test_non_thermodynamic_config_is_rejected(self):
        with pytest.raises(ValueError, match="thermodynamic"):
            StubSubstrateDriver(_config(substrate_type="gpu_emulation"), FERRO_PAIR, torch.zeros(2), 1e-4)
