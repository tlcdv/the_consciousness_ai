"""
Experimental substrate package: p-bit (Ising) emulation and free energy relaxation.

This package is isolated from production. The training entry point, the
action-selection core, and models/core/ must never import it; the guard is
tests/test_substrate_isolation.py.

Imports are lazy (PEP 562). Importing the package loads no submodule and no
optional simulator library; a name is loaded on first attribute access.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

_LAZY_NAMES: Dict[str, str] = {
    "BlockGibbsSampler": "p_bit_emulator",
    "exact_boltzmann": "p_bit_emulator",
    "greedy_coloring": "p_bit_emulator",
    "grid_couplings": "p_bit_emulator",
    "ising_energy": "p_bit_emulator",
    "LinearGaussianModel": "energy_minimization",
    "RelaxationTrajectory": "energy_minimization",
    "free_energy": "energy_minimization",
    "free_energy_gradient": "energy_minimization",
    "relax_free_energy": "energy_minimization",
    "stable_step_bound": "energy_minimization",
    "BaseSubstrateDriver": "interfaces.abstract_substrate",
    "SubstrateConfig": "interfaces.abstract_substrate",
    "StubSubstrateDriver": "interfaces.hardware_stub",
    "LIFBridgeLayer": "interfaces.snn_bridge",
    "atan_spike": "interfaces.snn_bridge",
    "latency_decode": "interfaces.snn_bridge",
    "latency_encode": "interfaces.snn_bridge",
    "phase_readout": "interfaces.snn_bridge",
    "phase_to_spike_time": "interfaces.snn_bridge",
    "rate_decode": "interfaces.snn_bridge",
    "rate_encode": "interfaces.snn_bridge",
    "spike_time_to_phase": "interfaces.snn_bridge",
    "export_lif_bridge_to_nir": "interfaces.nir_export",
    "lif_bridge_from_nir": "interfaces.nir_export",
    "nirtorch_node_map": "interfaces.nir_export",
    "sample_with_thrml": "interfaces.thrml_adapter",
    "thrml_available": "interfaces.thrml_adapter",
}

__all__ = sorted(_LAZY_NAMES)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{_LAZY_NAMES[name]}")
    return getattr(module, name)


def __dir__():
    return __all__
