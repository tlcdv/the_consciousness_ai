"""Regression tests for the AKOrN natural-frequency term.

The legacy update computed the rotation with einsum 'ndd,bnd->bnd', which reads
only the diagonal of each skew-symmetric generator. That diagonal is zero, so
the oscillators had no intrinsic frequency. The fix is behind a default-off
switch so the legacy baseline stays bit-identical.
"""
import math

import torch

from models.core.global_workspace import GlobalWorkspace
from models.core.oscillatory_binding import KuramotoLayer, WorkspaceBindingSystem


def _layer_with_frequency(w: float, natural_frequency: bool) -> KuramotoLayer:
    """One 2-D oscillator with generator [[0, -w], [w, 0]] and no coupling."""
    layer = KuramotoLayer(num_oscillators=1, dimensions=2, dt=0.1,
                          natural_frequency=natural_frequency)
    with torch.no_grad():
        layer.coupling_weights.zero_()
        # forward() uses omega = P - P^T, so P = [[0, 0], [w, 0]] gives the generator.
        layer.natural_frequencies.copy_(torch.tensor([[[0.0, 0.0], [w, 0.0]]]))
    return layer


def _angle(phases: torch.Tensor) -> float:
    return math.atan2(phases[0, 0, 1].item(), phases[0, 0, 0].item())


def test_legacy_default_has_no_rotation():
    # Arrange: a nonzero generator, zero coupling, legacy default.
    layer = _layer_with_frequency(w=2.0, natural_frequency=False)
    start = torch.tensor([[[1.0, 0.0]]])

    # Act
    end, _ = layer(start, iterations=3)

    # Assert: the legacy update ignores the generator (pinned bug).
    assert torch.equal(end, start)


def test_fixed_rotation_matches_closed_form_angle():
    # Arrange: normalize(x + dt * omega x) rotates x by exactly atan(dt * w) in 2-D.
    w, dt, steps = 2.0, 0.1, 3
    layer = _layer_with_frequency(w=w, natural_frequency=True)
    start = torch.tensor([[[1.0, 0.0]]])

    # Act
    end, _ = layer(start, iterations=steps)

    # Assert
    expected = steps * math.atan(dt * w)
    assert math.isclose(_angle(end), expected, rel_tol=1e-5)
    assert math.isclose(end.norm().item(), 1.0, rel_tol=1e-6)


def test_fixed_rotation_is_matrix_vector_product_for_many_oscillators():
    # Arrange
    torch.manual_seed(0)
    layer = KuramotoLayer(num_oscillators=5, dimensions=3, natural_frequency=True)
    with torch.no_grad():
        layer.coupling_weights.zero_()
    x = layer.init_phases(batch_size=2)
    omega = layer.natural_frequencies - layer.natural_frequencies.transpose(-1, -2)

    # Act
    end, _ = layer(x, iterations=1)

    # Assert: one step equals normalize(x + dt * omega @ x) per oscillator.
    manual = x + layer.dt * torch.einsum('nde,bne->bnd', omega, x)
    manual = manual / manual.norm(dim=-1, keepdim=True)
    assert torch.allclose(end, manual, atol=1e-6)
    assert not torch.allclose(end, x)


def test_switch_reaches_layer_from_workspace_config():
    # Arrange
    config = {"binding_mechanism": "akorn", "akorn_natural_frequency": True}

    # Act
    workspace = GlobalWorkspace(config)

    # Assert
    assert workspace.binding_system.kuramoto.natural_frequency is True
    assert GlobalWorkspace({}).binding_system.kuramoto.natural_frequency is False


def test_binding_system_default_matches_legacy_layer():
    # Arrange: two systems with identical parameters and phases, default switch.
    torch.manual_seed(1)
    default_system = WorkspaceBindingSystem(num_modules=5)
    torch.manual_seed(1)
    explicit_legacy = WorkspaceBindingSystem(num_modules=5, natural_frequency=False)
    bids = {"vision": 0.7, "audio": 0.2, "memory": 0.4, "body": 0.1, "semantic": 0.3}

    # Act
    torch.manual_seed(2)
    bound_a, sync_a = default_system.bind_bids(bids)
    torch.manual_seed(2)
    bound_b, sync_b = explicit_legacy.bind_bids(bids)

    # Assert
    assert bound_a == bound_b
    assert sync_a == sync_b
