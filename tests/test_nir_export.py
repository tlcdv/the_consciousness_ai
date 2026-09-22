"""
Tests for the NIR export of LIFBridgeLayer (models/thermodynamic/interfaces/nir_export.py).

nir and nirtorch need Python 3.10+ and are not in requirements.txt, so the export
tests skip where they are missing. Where present, three checks: the graph survives a
write and read through NIR's own file format; a layer rebuilt from the graph spikes
exactly like the original; and nirtorch loads the graph into a torch module that
also spikes exactly like the original.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.thermodynamic.interfaces.snn_bridge import LIFBridgeLayer  # noqa: E402

DT = 0.001


def _layer() -> LIFBridgeLayer:
    torch.manual_seed(0)
    layer = LIFBridgeLayer(4, 3, tau_mem=0.02, v_threshold=0.2, dt=DT, v_rest=0.0, v_reset=0.0)
    with torch.no_grad():
        layer.linear.weight.abs_()  # positive drive, so the layer actually spikes
    return layer


def _drive() -> torch.Tensor:
    return 2.0 * torch.rand(80, 2, 4, generator=torch.Generator().manual_seed(1))


@pytest.mark.skipif(importlib.util.find_spec("nir") is not None, reason="only meaningful where nir is missing")
def test_missing_nir_raises_instead_of_exporting():
    from models.thermodynamic.interfaces.nir_export import export_lif_bridge_to_nir

    with pytest.raises(ImportError, match="nir"):
        export_lif_bridge_to_nir(_layer())


class TestWithNir:
    @pytest.fixture(autouse=True)
    def _needs_nir(self):
        pytest.importorskip("nir")

    def test_graph_survives_nir_file_round_trip(self, tmp_path):
        import nir

        from models.thermodynamic.interfaces.nir_export import export_lif_bridge_to_nir

        graph = export_lif_bridge_to_nir(_layer())
        nir.write(tmp_path / "bridge.nir", graph)
        loaded = nir.read(tmp_path / "bridge.nir")

        assert sorted(loaded.nodes) == ["affine", "input", "lif", "output"]
        assert torch.allclose(torch.from_numpy(loaded.nodes["affine"].weight), _layer().linear.weight.double())

    def test_rebuilt_layer_spikes_exactly_like_the_original(self):
        from models.thermodynamic.interfaces.nir_export import export_lif_bridge_to_nir, lif_bridge_from_nir

        original = _layer()
        rebuilt = lif_bridge_from_nir(export_lif_bridge_to_nir(original), dt=DT)

        with torch.no_grad():
            assert torch.equal(rebuilt(_drive()), original(_drive()))

    def test_nirtorch_loads_a_module_that_spikes_like_the_original(self):
        nirtorch = pytest.importorskip("nirtorch")
        from models.thermodynamic.interfaces.nir_export import export_lif_bridge_to_nir, nirtorch_node_map

        original = _layer()
        module = nirtorch.nir_to_torch(export_lif_bridge_to_nir(original), nirtorch_node_map(DT))

        with torch.no_grad():
            state, loaded = None, []
            for step in _drive():
                spikes, state = module(step, state)
                loaded.append(spikes)
            loaded = torch.stack(loaded)
            expected = original(_drive())

        assert torch.equal(loaded, expected)
        assert expected.sum() > 10
