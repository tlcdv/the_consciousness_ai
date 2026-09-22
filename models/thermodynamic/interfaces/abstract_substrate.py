"""
The contract every computational substrate driver implements.

A driver converts a continuous tensor into the substrate's native form, runs the
substrate's own dynamics, converts back, and reports the energy of its state.
Substrates: GPU emulation (the current baseline), neuromorphic spikes, and
thermodynamic sampling. No driver here touches hardware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

SUBSTRATE_TYPES = ("gpu_emulation", "neuromorphic", "thermodynamic")
PRECISIONS = ("float32", "float16", "stochastic")


@dataclass
class SubstrateConfig:
    """Configuration for one substrate.

    temperature is the sampling temperature. Samplers use the dimensionless gain
    beta = 1 / temperature; it is not the physical temperature of any chip.
    """

    substrate_type: str
    device_id: Optional[str]
    precision: str
    temperature: float = 1.0
    time_step_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.substrate_type not in SUBSTRATE_TYPES:
            raise ValueError(f"substrate_type must be one of {SUBSTRATE_TYPES}, got {self.substrate_type!r}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.time_step_ms <= 0:
            raise ValueError(f"time_step_ms must be positive, got {self.time_step_ms}")


class BaseSubstrateDriver(ABC):
    """Abstract base class for all substrate drivers.

    Every substrate implements encode, execute, decode, energy and reset.
    """

    def __init__(self, config: SubstrateConfig):
        self.config = config

    @abstractmethod
    def encode(self, continuous_state: torch.Tensor) -> object:
        """Convert a continuous tensor to the substrate's native form.

        Neuromorphic: spike trains. Thermodynamic: p-bit configuration and couplings.
        """

    @abstractmethod
    def execute(self, substrate_state: object, steps: int = 1) -> object:
        """Run the substrate's native dynamics for the given number of steps.

        Neuromorphic: propagate spikes. Thermodynamic: block Gibbs sampling.
        """

    @abstractmethod
    def decode(self, substrate_state: object) -> torch.Tensor:
        """Convert the substrate's native form back to a continuous tensor."""

    @abstractmethod
    def energy(self, substrate_state: object) -> float:
        """Energy of the substrate state.

        Thermodynamic: H = -sum_{i<j} J_ij m_i m_j - sum_i b_i m_i.
        A driver that cannot compute its energy must raise, never return a default.
        """

    @abstractmethod
    def reset(self) -> None:
        """Return the substrate to its initial state."""
