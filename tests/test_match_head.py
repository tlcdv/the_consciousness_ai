"""Tests for the DMTS supervised match head (models/self_model/match_head.py)
and its wiring in train_rlhf.py.

The head is the documented "most direct" next step on the DMTS learning wall:
the match is supervised-decodable 0.845 from [current obs_map ; held sample] but
the Go/No-Go RL policy does not learn it from reward
(docs/results/rssm_working_memory_2026_06_12.md, Final localization). These tests
verify the head can decode a separable match signal (value test, not just shapes)
and that the wiring is default-off and gated to --env dmts --policy-input
obsmem-conv.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.self_model.match_head import MatchHead, AuxMatchHead
from scripts.training.train_rlhf import build_config, init_components

SPATIAL = (128, 16, 16)  # obsmem-conv tap: [current obs_map(64) ; held sample(64)]


# ── Shapes and basic contract ────────────────────────────────────────────────

def test_match_head_forward_shapes():
    head = MatchHead(SPATIAL, num_actions=5)
    flat = torch.randn(4, 128 * 16 * 16)
    spatial = torch.randn(4, 128, 16, 16)
    assert head(flat).shape == (4, 5)
    assert head(spatial).shape == (4, 5)


def test_match_head_predict_returns_valid_action():
    head = MatchHead(SPATIAL, num_actions=5)
    a = head.predict(torch.randn(1, 128 * 16 * 16))
    assert isinstance(a, int) and 0 <= a < 5


def test_aux_match_head_forward_shape():
    head = AuxMatchHead(context_dim=128, num_actions=5)
    assert head(torch.randn(4, 128)).shape == (4, 5)


# ── Value tests: the heads must actually decode a separable match signal ──────

def _separable_spatial_batch(labels: torch.Tensor) -> torch.Tensor:
    """Input where channel-block `k` is hot iff the label is `k` (5 classes over
    128 channels). A conv+linear head should learn this cleanly."""
    n = labels.shape[0]
    x = torch.randn(n, 128, 16, 16) * 0.1
    block = 128 // 5
    for i, lab in enumerate(labels):
        k = int(lab)
        x[i, k * block:(k + 1) * block] += 3.0
    return x


def test_match_head_learns_separable_mapping():
    """The conv head decodes a separable spatial mapping well above chance."""
    torch.manual_seed(0)
    head = MatchHead(SPATIAL, num_actions=5)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(300):
        labels = torch.randint(0, 5, (16,))
        logits = head(_separable_spatial_batch(labels))
        loss = head.loss(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    labels = torch.randint(0, 5, (128,))
    acc = (head(_separable_spatial_batch(labels)).argmax(1) == labels).float().mean().item()
    assert acc > 0.6, f"match head failed to learn separable mapping (acc={acc:.3f}, chance=0.2)"


def test_aux_match_head_learns_separable_mapping():
    """The linear aux head decodes a separable feature mapping well above chance."""
    torch.manual_seed(0)
    head = AuxMatchHead(context_dim=128, num_actions=5)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    def batch(labels):
        x = torch.randn(labels.shape[0], 128) * 0.1
        block = 128 // 5
        for i, lab in enumerate(labels):
            k = int(lab)
            x[i, k * block:(k + 1) * block] += 3.0
        return x

    for _ in range(300):
        labels = torch.randint(0, 5, (32,))
        logits = head(batch(labels))
        loss = head.loss(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    labels = torch.randint(0, 5, (128,))
    acc = (head(batch(labels)).argmax(1) == labels).float().mean().item()
    assert acc > 0.6, f"aux head failed to learn separable mapping (acc={acc:.3f}, chance=0.2)"


# ── Wiring: default-off and gated to dmts + obsmem-conv ───────────────────────

def _args(**overrides):
    base = argparse.Namespace(
        episodes=1, max_steps=10, action_dim=5, lr=1e-3, render=False,
        env="dmts", difficulty=0, log_dir="runs/_test", log_ei_every=0,
        enable_audio=False,
        ablate_memory_replay=False, ablate_consolidation_fix=False,
        ablate_rnd_zero_on_reward=False,
        gate_diversity_loss="off", gate_feedback="off",
        ablate_gate_diversity=False, ablate_gate_entropy=False,
        ablate_gate_feedback=False, ablate_pad_loop=False, ablate_bptt=False,
        phi_sample_every=5,
        enable_riiu=False, riiu_rank=16, riiu_window=64,
        riiu_source="broadcast", riiu_probe_all=False, seed=0,
        broadcast_mode="winner_take_all",
        attention_temperature=0.5, attention_floor=0.05,
        enable_mock_semantic=False, phi1_min_active_modules=0,
        policy="gonogo", policy_input="obsmem-conv",
        enable_match_head=False, match_head_mode="acting",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _config(**overrides):
    config = build_config(_args(**overrides))
    # main() overrides the action dim for discrete envs before init_components.
    if config["env"] == "dmts":
        config["action_selection"]["action_dim"] = 5
    return config


def _heads(components):
    return [c for c in components if isinstance(c, (MatchHead, AuxMatchHead))]


def test_match_head_disabled_by_default():
    components = init_components(_config())
    assert _heads(components) == []
    assert components[-1] is None  # match_optimizer


def test_match_head_acting_enabled_dmts_obsmem():
    components = init_components(_config(enable_match_head=True, match_head_mode="acting"))
    heads = _heads(components)
    assert len(heads) == 1 and isinstance(heads[0], MatchHead)
    assert components[-1] is not None  # match_optimizer present


def test_match_head_aux_enabled_uses_aux_head():
    components = init_components(_config(enable_match_head=True, match_head_mode="aux"))
    heads = _heads(components)
    assert len(heads) == 1 and isinstance(heads[0], AuxMatchHead)


def test_match_head_disabled_when_not_obsmem_conv():
    """Flag on but policy_input != obsmem-conv -> disabled with a warning."""
    components = init_components(
        _config(enable_match_head=True, policy_input="broadcast")
    )
    assert _heads(components) == []
    assert components[-1] is None


def test_match_head_disabled_when_not_dmts():
    """Flag on but env != dmts -> disabled with a warning."""
    components = init_components(
        _config(enable_match_head=True, env="dark_room", policy_input="obsmem-conv")
    )
    assert _heads(components) == []
    assert components[-1] is None
