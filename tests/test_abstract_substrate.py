"""
Tests for the substrate driver contract
(models/thermodynamic/interfaces/abstract_substrate.py).
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


def _config(**overrides) -> SubstrateConfig:
    fields = dict(substrate_type="gpu_emulation", device_id=None, precision="float32")
    fields.update(overrides)
    return SubstrateConfig(**fields)


class _IdentityDriver(BaseSubstrateDriver):
    def encode(self, continuous_state):
        return continuous_state.clone()

    def execute(self, substrate_state, steps=1):
        return substrate_state

    def decode(self, substrate_state):
        return substrate_state

    def energy(self, substrate_state):
        return float((substrate_state ** 2).sum())

    def reset(self):
        return None


class TestSubstrateConfig:
    def test_defaults(self):
        config = _config()
        assert config.temperature == 1.0
        assert config.time_step_ms == 1.0

    @pytest.mark.parametrize("field,bad", [
        ("substrate_type", "quantum"),
        ("precision", "int8"),
        ("temperature", 0.0),
        ("time_step_ms", -1.0),
    ])
    def test_invalid_fields_are_rejected(self, field, bad):
        with pytest.raises(ValueError, match=field):
            _config(**{field: bad})


class TestBaseSubstrateDriver:
    def test_cannot_instantiate_the_abstract_class(self):
        with pytest.raises(TypeError):
            BaseSubstrateDriver(_config())

    def test_subclass_missing_a_method_cannot_be_instantiated(self):
        class Incomplete(BaseSubstrateDriver):
            def encode(self, continuous_state):
                return continuous_state

        with pytest.raises(TypeError):
            Incomplete(_config())

    def test_complete_subclass_round_trips(self):
        driver = _IdentityDriver(_config())
        x = torch.tensor([0.25, -1.0])

        decoded = driver.decode(driver.execute(driver.encode(x)))

        assert torch.equal(decoded, x)
        assert driver.energy(x) == pytest.approx(1.0625)
