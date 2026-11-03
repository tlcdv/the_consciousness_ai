"""--phi-control, which takes phi out of the agent (docs/decisions/2026_11_03_phi_out_of_the_agent.md).

The default is 'off' since 2026-11-03, by the owner's decision to drop IIT. 'on' reproduces every earlier
run, which a run-level md5 check in that decision document showed. These tests pin the rules of the flag
that do not need a training run.
"""
from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.training import train_rlhf


class TestExplorationScale:
    @pytest.mark.parametrize("phi, scale", [(0.0, 1.0), (0.05, 0.5), (0.1, 0.2), (0.5, 0.2)])
    def test_with_phi_on_the_legacy_rule_holds(self, phi, scale):
        assert train_rlhf.exploration_scale_from_phi(phi, phi_off=False) == pytest.approx(scale)

    @pytest.mark.parametrize("phi", [0.0, 0.05, 0.5, float("nan")])
    def test_with_phi_off_exploration_is_not_scaled(self, phi):
        assert train_rlhf.exploration_scale_from_phi(phi, phi_off=True) == 1.0

    def test_a_missing_phi_would_have_set_the_floor_with_the_legacy_rule(self):
        """Why phi off needs its own branch. max(0.2, nan) is 0.2, the strongest exploitation."""
        assert train_rlhf.exploration_scale_from_phi(float("nan"), phi_off=False) == 0.2


class TestNeutralMemoryValue:
    def test_it_is_below_the_memory_threshold_and_the_relevance_floor(self):
        """The memory the training loop builds stores to its index only at attention 0.7 or more."""
        from models.memory.memory_core import MemoryCore
        config = train_rlhf.build_config(argparse.Namespace(action_dim=2, lr=1e-3, episodes=1, max_steps=10))
        assert train_rlhf.PHI_OFF_ATTENTION < MemoryCore(config["memory"]).attention_threshold
        assert max(0.2, train_rlhf.PHI_OFF_ATTENTION) == 0.2  # the relevance rule of consolidation


class TestConfig:
    def test_the_default_is_off_and_on_is_passed_through(self):
        base = dict(action_dim=2, lr=1e-3, episodes=1, max_steps=10)
        assert train_rlhf.build_config(argparse.Namespace(**base))["phi_control"] == "off"
        assert train_rlhf.build_config(argparse.Namespace(phi_control="on", **base))["phi_control"] == "on"


class TestFlagRules:
    @staticmethod
    def parser():
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        return captured["parser"]

    def test_the_flag_is_off_by_default_and_names_its_decision(self):
        parser = self.parser()
        assert parser.parse_args([]).phi_control == "off"
        assert parser.parse_args(["--phi-control", "on"]).phi_control == "on"
        text = next(a.help for a in parser._actions if "--phi-control" in a.option_strings)
        assert "2026_11_03_phi_out_of_the_agent" in text

    def test_the_default_refuses_riiu_and_needs_no_pyphi(self):
        """A command with no --phi-control is now a phi-off command."""
        with pytest.raises(SystemExit):
            train_rlhf._reject_incompatible_flags(argparse.ArgumentParser(), argparse.Namespace(enable_riiu=True))
        train_rlhf._require_pyphi(argparse.ArgumentParser(), argparse.Namespace(), None)

    @pytest.mark.parametrize("other", ["enable_riiu", "phi_proxy"])
    def test_phi_off_refuses_riiu_and_the_proxy(self, other):
        args = argparse.Namespace(phi_control="off", **{other: True})
        with pytest.raises(SystemExit):
            train_rlhf._reject_incompatible_flags(argparse.ArgumentParser(), args)

    def test_phi_on_keeps_riiu_allowed(self):
        train_rlhf._reject_incompatible_flags(argparse.ArgumentParser(),
                                              argparse.Namespace(phi_control="on", enable_riiu=True))

    def test_phi_off_does_not_need_pyphi(self):
        train_rlhf._require_pyphi(argparse.ArgumentParser(), argparse.Namespace(phi_control="off"), None)

    def test_phi_on_still_needs_pyphi(self):
        with pytest.raises(SystemExit):
            train_rlhf._require_pyphi(argparse.ArgumentParser(), argparse.Namespace(phi_control="on"), None)
