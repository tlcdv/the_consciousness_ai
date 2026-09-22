"""
Stand-in thermodynamic driver for tests and CI.

It implements BaseSubstrateDriver on the p-bit emulator, so any code written
against the driver contract runs without hardware or a vendor SDK. It is
deterministic: reset restores the seed, so the same calls give the same states.

Latency is simulated, not measured. Each execute step adds latency_s to
elapsed_s; the stub sleeps only when sleep=True. elapsed_s says nothing about any
real device.
"""
from __future__ import annotations

import time
from typing import Optional

import torch

from models.thermodynamic.interfaces.abstract_substrate import BaseSubstrateDriver, SubstrateConfig
from models.thermodynamic.p_bit_emulator import BlockGibbsSampler, ising_energy

MIN_LATENCY_S = 1e-5
MAX_LATENCY_S = 1e-2


class StubSubstrateDriver(BaseSubstrateDriver):
    """P-bit emulator behind the driver contract, with simulated latency."""

    def __init__(
        self,
        config: SubstrateConfig,
        couplings: torch.Tensor,
        biases: torch.Tensor,
        latency_s: float,
        seed: int = 0,
        sleep: bool = False,
    ):
        super().__init__(config)
        if config.substrate_type != "thermodynamic":
            raise ValueError(f"the stub emulates a thermodynamic substrate, got {config.substrate_type!r}")
        if not MIN_LATENCY_S <= latency_s <= MAX_LATENCY_S:
            raise ValueError(f"latency_s must be within [{MIN_LATENCY_S}, {MAX_LATENCY_S}] s, got {latency_s}")
        self.couplings, self.biases = couplings, biases.to(couplings.dtype)
        self.latency_s, self.seed, self.sleep = latency_s, seed, sleep
        self.sampler: Optional[BlockGibbsSampler] = None
        self.elapsed_s = 0.0
        self.reset()

    def encode(self, continuous_state: torch.Tensor) -> torch.Tensor:
        """Sign of each value as a spin: x >= 0 -> +1, x < 0 -> -1."""
        return torch.where(continuous_state >= 0, 1.0, -1.0).to(self.couplings.dtype)

    def execute(self, substrate_state: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Run steps block Gibbs sweeps on a copy of the spins and add simulated latency."""
        spins = substrate_state.clone()
        for _ in range(steps):
            self.sampler.step(spins)
        self.elapsed_s += self.latency_s * steps
        if self.sleep:
            time.sleep(self.latency_s * steps)
        return spins

    def decode(self, substrate_state: torch.Tensor) -> torch.Tensor:
        return substrate_state.clone()

    def energy(self, substrate_state: torch.Tensor) -> float:
        """Mean Ising energy over the batch."""
        return ising_energy(substrate_state, self.couplings, self.biases).mean().item()

    def reset(self) -> None:
        self.sampler = BlockGibbsSampler(self.couplings, self.biases, 1.0 / self.config.temperature, seed=self.seed)
        self.elapsed_s = 0.0
