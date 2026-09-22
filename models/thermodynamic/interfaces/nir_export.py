"""
Export and import of LIFBridgeLayer through the Neuromorphic Intermediate
Representation (NIR, Pedersen et al., Nature Communications 15, 8122, 2024).

The exported graph is input -> Affine -> LIF -> output. NIR's LIF is continuous
time, tau dv/dt = (v_leak - v) + r I, which is LIFBridgeLayer with r = 1 and
v_leak = v_rest; the Euler step dt is not part of the graph and is supplied again on
import. KuramotoLayer is not exported: NIR has no oscillator primitive.

nir and nirtorch are optional (Python 3.10+) and never in requirements.txt. Every
function imports them on call and raises ImportError when they are missing.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import nn

from models.thermodynamic.interfaces.snn_bridge import LIFBridgeLayer


def _require_nir():
    try:
        import nir
    except ImportError as missing:
        raise ImportError("nir is not installed; it needs Python 3.10+: pip install nir") from missing
    return nir


def export_lif_bridge_to_nir(layer: LIFBridgeLayer):
    """NIR graph input -> affine -> lif -> output with the layer's parameters."""
    nir = _require_nir()
    weight = layer.linear.weight.detach().double().numpy()
    n_out, n_in = weight.shape
    bias = layer.linear.bias.detach().double().numpy() if layer.linear.bias is not None else np.zeros(n_out)
    full = lambda value: np.full(n_out, value, dtype=np.float64)  # noqa: E731
    nodes = {
        "input": nir.Input(input_type={"input": np.array([n_in])}),
        "affine": nir.Affine(weight=weight, bias=bias),
        "lif": nir.LIF(
            tau=full(layer.tau_mem), r=full(1.0), v_leak=full(layer.v_rest),
            v_threshold=full(layer.v_threshold), v_reset=full(layer.v_reset),
        ),
        "output": nir.Output(output_type={"output": np.array([n_out])}),
    }
    edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
    return nir.NIRGraph(nodes=nodes, edges=edges)


def _single_value(name: str, values: np.ndarray) -> float:
    if not np.allclose(values, values.flat[0]):
        raise ValueError(f"LIFBridgeLayer needs one shared {name}; the graph has per-neuron values")
    return float(values.flat[0])


def lif_bridge_from_nir(graph, dt: float) -> LIFBridgeLayer:
    """Rebuild a LIFBridgeLayer from a graph written by export_lif_bridge_to_nir."""
    affine, lif = graph.nodes["affine"], graph.nodes["lif"]
    if not np.allclose(lif.r, 1.0):
        raise ValueError("LIFBridgeLayer assumes r = 1")
    n_out, n_in = affine.weight.shape
    layer = LIFBridgeLayer(
        n_in, n_out, tau_mem=_single_value("tau", lif.tau), v_threshold=_single_value("v_threshold", lif.v_threshold),
        dt=dt, v_rest=_single_value("v_leak", lif.v_leak), v_reset=_single_value("v_reset", lif.v_reset),
    )
    with torch.no_grad():
        layer.linear.weight.copy_(torch.from_numpy(affine.weight))
        layer.linear.bias.copy_(torch.from_numpy(affine.bias))
    return layer


class LIFStep(nn.Module):
    """One Euler step of a NIR LIF node per call, in nirtorch's stateful form.

    forward(current, state) returns (spikes, membrane); nirtorch threads the
    membrane from one call to the next through its state dictionary.
    """

    def __init__(self, node, dt: float):
        super().__init__()
        self.dt = dt
        for name in ("tau", "r", "v_leak", "v_threshold", "v_reset"):
            self.register_buffer(name, torch.as_tensor(np.asarray(getattr(node, name)), dtype=torch.float32))

    def forward(self, current: torch.Tensor, state: Optional[torch.Tensor] = None):
        v = self.v_leak.expand_as(current) if state is None else state
        v = v + self.dt / self.tau * ((self.v_leak - v) + self.r * current)
        spikes = (v >= self.v_threshold).to(current.dtype)
        return spikes, torch.where(spikes > 0, self.v_reset.expand_as(v), v)


def affine_to_linear(node) -> nn.Linear:
    """nn.Linear carrying a NIR Affine node's weight AND bias.

    nirtorch 2.6's default Affine map assigns the bias to module.weight.bias, so
    the loaded layer keeps its random initial bias. This map replaces it.
    """
    n_out, n_in = node.weight.shape
    module = nn.Linear(n_in, n_out, bias=True)
    with torch.no_grad():
        module.weight.copy_(torch.as_tensor(node.weight, dtype=module.weight.dtype))
        module.bias.copy_(torch.as_tensor(node.bias, dtype=module.bias.dtype))
    return module


def nirtorch_node_map(dt: float) -> Dict[type, Callable]:
    """node_map for nirtorch.nir_to_torch: LIF -> LIFStep, Affine -> affine_to_linear."""
    nir = _require_nir()
    return {nir.LIF: lambda node: LIFStep(node, dt), nir.Affine: affine_to_linear}
