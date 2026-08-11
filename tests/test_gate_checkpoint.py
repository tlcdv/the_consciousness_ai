"""
The gate must survive a save/load round trip.

Written 2026-08-11. The gate trains inside `tectum_optimizer` alongside the tectum
(`train_rlhf.py`, `list(tectum.parameters()) + list(gate.parameters())`) but was never
written to disk: the only save wrote `tectum.state_dict()`. Every offline probe therefore
rebuilt the gate from random init and read an untrained `gate_feedback`, while claiming to
read the substrate the training loop reads.

These tests pin the three properties that failure needed:
  1. the sibling path convention is shared by writer and reader, so it cannot drift,
  2. a saved gate reloads to bit-identical outputs, and
  3. an unloaded gate is NOT bit-identical to a trained one, which is what makes the bug
     detectable rather than silent.

Test 3 is the failing null: without it, tests 1 and 2 would still pass if the gate had no
learnable state worth persisting.
"""
import os
import tempfile

import pytest
import torch

from models.core.consciousness_gating import ConsciousnessGate, gate_checkpoint_path


GATE_CONFIG = {
    "hidden_size": 256,
    "ablate_feedback": False,
    "use_self_vector": False,
    "self_vector_dim": 64,
}


def _make_gate(seed: int) -> ConsciousnessGate:
    torch.manual_seed(seed)
    return ConsciousnessGate(dict(GATE_CONFIG))


def _gate_outputs(gate: ConsciousnessGate, seed: int = 0) -> torch.Tensor:
    """Drive the gate once from a fixed input and return its 5 node values."""
    torch.manual_seed(seed)
    input_state = torch.randn(1, GATE_CONFIG["hidden_size"])
    gate.reset_episode()
    with torch.no_grad():
        gate(input_state)
    assert gate.prev_gate_values is not None, "gate did not record prev_gate_values"
    return gate.prev_gate_values.detach().clone()


def test_gate_checkpoint_path_is_a_sibling_of_the_tectum():
    """Writer and reader derive the path the same way, from one function."""
    tectum = "runs/x/tectum.pt"
    path = gate_checkpoint_path(tectum)

    assert path.endswith(".gate.pt")
    assert os.path.dirname(path) == os.path.dirname(tectum)
    # Must never collide with the tectum checkpoint: existing tectum.pt files stay
    # loadable byte for byte, which is why this is a sibling and not a merged key.
    assert path != tectum


def test_gate_checkpoint_path_handles_a_missing_extension():
    assert gate_checkpoint_path("ckpt").endswith(".gate.pt")


def test_saved_gate_reloads_to_identical_outputs():
    """The round trip a probe depends on."""
    trained = _make_gate(seed=1)
    # Stand in for training: perturb the weights away from init.
    with torch.no_grad():
        for param in trained.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    expected = _gate_outputs(trained)

    with tempfile.TemporaryDirectory() as tmp:
        path = gate_checkpoint_path(os.path.join(tmp, "tectum.pt"))
        torch.save(trained.state_dict(), path)

        reloaded = _make_gate(seed=99)  # different init on purpose
        reloaded.load_state_dict(torch.load(path, map_location="cpu"))

    torch.testing.assert_close(_gate_outputs(reloaded), expected, rtol=0, atol=0)


def test_an_unloaded_gate_differs_from_a_trained_one():
    """The FAILING NULL.

    This is the property the bug violated. If a freshly constructed gate happened to
    match a trained one, then never saving the gate would have cost nothing and the
    round-trip test above would prove nothing. It does not match, so an offline probe
    that skips loading is measuring a different system.
    """
    trained = _make_gate(seed=1)
    with torch.no_grad():
        for param in trained.parameters():
            param.add_(torch.randn_like(param) * 0.05)

    fresh = _make_gate(seed=99)

    assert not torch.allclose(
        _gate_outputs(fresh), _gate_outputs(trained), rtol=1e-3, atol=1e-3
    ), (
        "a randomly initialised gate reproduces a trained gate's outputs; if this ever "
        "holds, gate weights carry no information and the persistence fix is pointless"
    )


def test_gate_feedback_is_a_learnable_parameter_worth_saving():
    """`gate_feedback` carries the gate-to-gate recurrence an interventional TPM needs.

    Pinned because that layer is the reason persistence matters: it is the only path by
    which the gate state at t influences the gate state at t+1.
    """
    gate = _make_gate(seed=1)
    names = dict(gate.named_parameters())
    assert "gate_feedback.weight" in names, "gate_feedback is no longer a parameter"
    assert names["gate_feedback.weight"].requires_grad
    assert "gate_feedback.weight" in gate.state_dict(), (
        "gate_feedback is absent from state_dict, so saving the gate would not save "
        "the recurrence"
    )
