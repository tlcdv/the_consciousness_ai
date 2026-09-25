"""
Which winner's tensor reaches the broadcast when two modules win ignition.

`_resolve_competition` returns the winners strongest first. The legacy
winner_take_all path then merges their payloads with `dict.update`, and every
payload the training loop builds is a dict with a "tensor" key
(scripts/training/train_rlhf.py, the `payloads` dict passed to settle). Each
later, weaker winner therefore overwrites the "tensor" and "source" of the one
before it: with two winners the policy reads the WEAKER winner's tensor, next
to the stronger winner's capsule poses, while the step CSV records the stronger
one as `bid_winner`.

`broadcast_merge="top_winner"` keeps the value a stronger winner wrote. It is
off by default, so every earlier run and test stays bit identical. These tests
pin the legacy defect as it is, the fixed behavior behind the flag, and the
default.
"""
from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.global_workspace import GlobalWorkspace


def _config(broadcast_merge: str | None = None,
            broadcast_mode: str = "winner_take_all") -> dict:
    config = {
        # Winners need a bound bid >= 0.8 * ignition_threshold = 0.24. The AKOrN
        # boost multiplies each bid by a factor in [1.0, 1.5], so vision 0.9 lands in
        # [0.9, 1.35] and audio 0.4 in [0.4, 0.6]: both win, and vision always ranks
        # first, whatever the random oscillator phases do.
        "ignition_threshold": 0.3,
        "ignition_gain": 5.0,
        "reverberation_alpha": 0.0,
        "workspace_dim": 16,
        "broadcast_mode": broadcast_mode,
        "num_modules": 4,
        "module_names": ["vision", "audio", "memory", "body"],
    }
    if broadcast_merge is not None:
        config["broadcast_merge"] = broadcast_merge
    return config


def _payloads() -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "vision": {"tensor": torch.randn(16, generator=g), "source": "tectum",
                   "capsule_poses": torch.randn(4, generator=g)},
        "audio": {"tensor": torch.randn(16, generator=g), "source": "audio",
                  "spectrum": torch.randn(4, generator=g)},
    }


def _compete(config: dict, bids: dict | None = None):
    torch.manual_seed(0)
    workspace = GlobalWorkspace(config)
    payloads = _payloads()
    broadcast, _ = workspace.run_competition(
        inputs={}, goal_vector=torch.zeros(3),
        bids=bids if bids is not None else {"vision": 0.9, "audio": 0.4},
        payloads=payloads,
    )
    return workspace, broadcast, payloads


def test_both_modules_win_and_vision_ranks_first():
    """Every test below assumes two winners with the stronger one first."""
    workspace, _, _ = _compete(_config())
    assert workspace.state.is_conscious
    assert workspace.state.winners == ["vision", "audio"]


def test_legacy_merge_broadcasts_the_weaker_winners_tensor():
    """The defect, pinned as the documented legacy behaviour."""
    _, broadcast, payloads = _compete(_config())
    assert torch.equal(broadcast["tensor"], payloads["audio"]["tensor"])
    assert broadcast["source"] == "audio"
    # The stronger winner's capsule poses ride along with the weaker tensor.
    assert torch.equal(broadcast["capsule_poses"], payloads["vision"]["capsule_poses"])


def _same_broadcast(a: dict, b: dict) -> bool:
    if list(a) != list(b):
        return False
    return all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k]
               for k in a)


def test_default_is_the_legacy_merge():
    workspace = GlobalWorkspace(_config())
    assert workspace.broadcast_merge == "legacy"
    _, default_broadcast, _ = _compete(_config())
    _, legacy_broadcast, _ = _compete(_config("legacy"))
    assert _same_broadcast(default_broadcast, legacy_broadcast)


def test_top_winner_merge_broadcasts_the_strongest_winners_tensor():
    _, broadcast, payloads = _compete(_config("top_winner"))
    assert torch.equal(broadcast["tensor"], payloads["vision"]["tensor"])
    assert broadcast["source"] == "tectum"
    assert torch.equal(broadcast["capsule_poses"], payloads["vision"]["capsule_poses"])
    # A key only the weaker winner carries is still merged in, as in legacy.
    assert torch.equal(broadcast["spectrum"], payloads["audio"]["spectrum"])


def test_top_winner_keeps_the_legacy_key_order_and_structured_payload():
    legacy_ws, legacy, _ = _compete(_config("legacy"))
    top_ws, top, _ = _compete(_config("top_winner"))
    assert list(top) == list(legacy)
    assert list(top_ws.state.broadcast_payload) == list(legacy_ws.state.broadcast_payload)
    for name, payload in legacy_ws.state.broadcast_payload.items():
        assert torch.equal(top_ws.state.broadcast_payload[name]["tensor"], payload["tensor"])


def test_single_winner_is_identical_under_both_merges():
    """With one winner nothing collides, so the flag changes nothing."""
    bids = {"vision": 0.9, "audio": 0.1}
    legacy_ws, legacy, _ = _compete(_config("legacy"), bids)
    _, top, _ = _compete(_config("top_winner"), bids)
    assert legacy_ws.state.winners == ["vision"]
    assert _same_broadcast(top, legacy)


def test_unknown_merge_value_raises():
    with pytest.raises(ValueError, match="broadcast_merge"):
        GlobalWorkspace(_config("strongest"))


def test_top_winner_with_attention_weighted_raises():
    """attention_weighted never runs the merge, so the flag would do nothing there."""
    with pytest.raises(ValueError, match="winner_take_all"):
        GlobalWorkspace(_config("top_winner", broadcast_mode="attention_weighted"))


# --- the training entry point carries the flag, the config key and the default ----

class _ParserCaptured(Exception):
    pass


def _train_rlhf_parser() -> argparse.ArgumentParser:
    """The argparse parser main() builds, captured before main() parses sys.argv."""
    from scripts.training import train_rlhf

    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise _ParserCaptured

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(_ParserCaptured):
            train_rlhf.main()
    return captured["parser"]


def test_training_flag_defaults_to_legacy_and_reaches_the_workspace_config():
    from scripts.training.train_rlhf import build_config

    parser = _train_rlhf_parser()
    default_args = parser.parse_args([])
    assert default_args.broadcast_merge == "legacy"
    assert build_config(default_args)["workspace"]["broadcast_merge"] == "legacy"

    flagged = parser.parse_args(["--broadcast-merge", "top_winner"])
    assert build_config(flagged)["workspace"]["broadcast_merge"] == "top_winner"


def test_build_config_without_the_attribute_keeps_legacy():
    """Callers that build a Namespace by hand (most tests) get the legacy merge."""
    from scripts.training.train_rlhf import build_config

    args = argparse.Namespace(action_dim=2, lr=1e-3, episodes=1, max_steps=10)
    assert build_config(args)["workspace"]["broadcast_merge"] == "legacy"
