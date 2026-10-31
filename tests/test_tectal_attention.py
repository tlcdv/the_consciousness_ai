"""The tectal attention gain (docs/decisions/2026_10_17_tectal_attention.md)."""
from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.tectal_attention import attend, attention_field


def training_parser():
    from scripts.training import train_rlhf
    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise RuntimeError("captured")

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(RuntimeError):
            train_rlhf.main()
    return captured["parser"]


class TestField:
    def test_ones_everywhere_and_the_gain_on_the_cells(self):
        field = attention_field([0, 17, 255], grid=16, gain=2.0)
        assert field.shape == (16, 16)
        assert field[0, 0] == 2.0 and field[1, 1] == 2.0 and field[15, 15] == 2.0
        assert field.sum() == pytest.approx(256 + 3)

    def test_no_cells_is_all_ones(self):
        assert np.array_equal(attention_field([], 16, 2.0), np.ones((16, 16)))

    def test_the_gain_cannot_be_below_one(self):
        with pytest.raises(ValueError):
            attention_field([1], 16, 0.5)


class TestAttend:
    def test_a_uniform_map_follows_the_closed_form(self):
        x = torch.full((1, 4, 16, 16), 0.5)
        y = attend(x, attention_field(list(range(32)), 16, 3.0))
        scale = 1.0 / (1.0 + (3.0 - 1.0) * 32 / 256)
        assert y[0, :, 0, 0].tolist() == pytest.approx([0.5 * 3.0 * scale] * 4)
        assert y[0, :, 15, 15].tolist() == pytest.approx([0.5 * scale] * 4)

    def test_the_total_absolute_activity_of_each_map_is_kept(self):
        x = torch.randn(2, 8, 16, 16, generator=torch.Generator().manual_seed(0))
        y = attend(x, attention_field([5, 6, 7, 21, 22], 16, 2.0))
        assert y.abs().sum(dim=(1, 2, 3)).tolist() == pytest.approx(x.abs().sum(dim=(1, 2, 3)).tolist(), rel=1e-5)

    def test_a_field_of_ones_returns_the_map_unchanged(self):
        x = torch.randn(1, 8, 16, 16, generator=torch.Generator().manual_seed(1))
        assert torch.equal(attend(x, attention_field([], 16, 2.0)), x)


class FakeTectum:
    def __init__(self, binding=True):
        self.topo_map = torch.nn.Identity()
        self.map_history = []
        self.map_binding = object() if binding else None


class TestWiring:
    def test_the_hook_applies_the_selection_of_the_previous_step(self):
        from scripts.training import train_rlhf
        tectum = FakeTectum()
        train_rlhf._wire_tectal_attention(tectum, gain=2.0, log_dir=None)
        x = torch.ones(1, 4, 16, 16)
        assert torch.equal(tectum.topo_map(x), x)
        tectum.map_history.append({"group": [0]})
        y = tectum.topo_map(x)
        assert float(y[0, 0, 0, 0] / y[0, 0, 15, 15]) == pytest.approx(2.0)
        assert tectum.tectal_attention == {"gain": 2.0, "steps": 2, "steps_with_gain": 1}

    def test_a_record_from_before_the_last_room_reset_is_not_used(self):
        from scripts.training import train_rlhf

        class Room:
            def reset(self, *args, **kwargs):
                return "frame", {}

        tectum, room = FakeTectum(), Room()
        train_rlhf._wire_tectal_attention(tectum, gain=2.0, log_dir=None, env=room)
        tectum.map_history.append({"group": [0]})
        assert room.reset() == ("frame", {})
        x = torch.ones(1, 4, 16, 16)
        assert torch.equal(tectum.topo_map(x), x)
        tectum.map_history.append({"group": [0]})
        assert not torch.equal(tectum.topo_map(x), x)

    def test_a_gain_below_one_is_refused(self):
        from scripts.training import train_rlhf
        with pytest.raises(ValueError):
            train_rlhf._wire_tectal_attention(FakeTectum(), gain=0.5, log_dir=None)

    def test_it_needs_the_map_binding(self):
        from scripts.training import train_rlhf
        with pytest.raises(ValueError):
            train_rlhf._wire_tectal_attention(FakeTectum(binding=False), gain=2.0, log_dir=None)

    def test_the_flags_are_off_by_default_and_the_help_names_the_rules(self):
        parser = training_parser()
        args = parser.parse_args([])
        assert args.tectal_attention == "off" and args.tectal_attention_gain == 2.0
        assert args.tectal_orienting_covert is False
        text = next(a.help for a in parser._actions if "--tectal-attention" in a.option_strings)
        assert "--map-binding" in text and "2026_10_17_tectal_attention" in text
        text = next(a.help for a in parser._actions if "--tectal-orienting-covert" in a.option_strings)
        assert "does not move" in text

    def test_covert_orienting_selects_a_target_but_gives_no_movement(self):
        from models.core.global_workspace import GlobalWorkspace
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        train_rlhf._wire_map_binding(tectum, workspace, None, "trimmed")

        class Policy:
            def select_action(self, state, emotion_arousal=0.5, rpe_cache=0.0, self_vector=None):
                return np.array([0.9, 0.9], dtype=np.float32), 0.25

        policy = Policy()
        train_rlhf._wire_tectal_orienting(policy, tectum, "group", seed=0, log_dir=None, idle="still",
                                          covert=True)
        frame = torch.zeros(1, 3, 224, 224)
        frame[0, :, 112 - 7:112 + 7, 200:224] = 1.0
        for _ in range(5):
            tectum(frame, torch.zeros(1, tectum.feature_dim, 2))
            action, value = policy.select_action("broadcast")
            assert value == 0.25 and np.array_equal(action, np.zeros(2, dtype=np.float32))
        assert len(tectum.tectal_motor_map.history) == 5
