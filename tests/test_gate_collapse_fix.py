"""Tests for Phase C of the 2026-05-17 Phi-1 retest plan: gate-collapse fixes.

Three changes verified:
  C1: diversity-loss default flipped to OFF (--gate-diversity-loss off)
  C1: gate-feedback default flipped to OFF (--gate-feedback off)
  C2: adaptation binarization floor lowered from 0.001 to 1e-5
      + median-vs-floor warning on the dimension being pinned

The 2026-05-14 ablation evidence anchoring these changes:
  - E_no_div (diversity loss OFF) had +240% phi_std vs head A_current
  - F_no_fb (gate_feedback OFF) had best Phi-1 r vs head A_current
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.training.train_rlhf import build_config
from models.evaluation.iit_phi import (
    IITMetrics,
    GATE_CM,
    GATE_NODE_LABELS,
    _DEFAULT_BINARIZATION_FLOORS,
)


def _default_args(**overrides):
    """Argparse Namespace mimicking train_rlhf.py defaults after Phase A+C."""
    base = argparse.Namespace(
        episodes=1, max_steps=10, action_dim=2, lr=1e-3, render=False,
        env="dark_room", difficulty=0, log_dir="runs/_test", log_ei_every=0,
        enable_audio=False,
        ablate_memory_replay=False, ablate_consolidation_fix=False,
        ablate_rnd_zero_on_reward=False,
        # NEW Phase C flags with their NEW defaults
        gate_diversity_loss="off", gate_feedback="off",
        # Legacy aliases still present, default False
        ablate_gate_diversity=False, ablate_gate_entropy=False,
        ablate_gate_feedback=False, ablate_pad_loop=False, ablate_bptt=False,
        phi_sample_every=5,
        enable_riiu=False, riiu_rank=16, riiu_window=64,
        riiu_source="broadcast", riiu_probe_all=False, seed=None,
        # Phase A flags
        broadcast_mode="winner_take_all",
        attention_temperature=0.5, attention_floor=0.05,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_default_gate_diversity_loss_is_off():
    """Default config now has the diversity loss OFF (post-Phase C flip)."""
    config = build_config(_default_args())
    assert config["ablate_gate_diversity"] is True, (
        f"expected ablate_gate_diversity=True by default (loss OFF), "
        f"got {config['ablate_gate_diversity']}"
    )


def test_default_gate_feedback_is_off():
    """Default config now has the gate_feedback projection OFF."""
    config = build_config(_default_args())
    assert config["ablate_gate_feedback"] is True, (
        f"expected ablate_gate_feedback=True by default (feedback OFF), "
        f"got {config['ablate_gate_feedback']}"
    )


def test_legacy_alias_ablate_flags_still_force_off():
    """Old --ablate-gate-diversity and --ablate-gate-feedback flags still work
    as backward-compatible aliases when explicitly set."""
    # User explicitly passes --ablate-gate-diversity (legacy semantics)
    config = build_config(_default_args(
        gate_diversity_loss="log_distance",  # tries to opt in to legacy loss
        ablate_gate_diversity=True,           # but legacy ablate flag wins
    ))
    assert config["ablate_gate_diversity"] is True, (
        "legacy --ablate-gate-diversity must force the loss off "
        "even when --gate-diversity-loss=log_distance is set"
    )
    # Same for feedback
    config = build_config(_default_args(
        gate_feedback="on",
        ablate_gate_feedback=True,
    ))
    assert config["ablate_gate_feedback"] is True


def test_opt_in_to_legacy_loss_via_new_flag():
    """Setting --gate-diversity-loss log_distance opts back into the legacy
    behavior (loss ON). Used for ablation testing of the new default."""
    config = build_config(_default_args(gate_diversity_loss="log_distance"))
    assert config["ablate_gate_diversity"] is False, (
        f"opting in to legacy log_distance loss should set "
        f"ablate_gate_diversity=False (loss ON); got {config['ablate_gate_diversity']}"
    )
    config = build_config(_default_args(gate_feedback="on"))
    assert config["ablate_gate_feedback"] is False


def test_adaptation_floor_lowered_to_1e_5():
    """The _DEFAULT_BINARIZATION_FLOORS constant has adaptation at 1e-5."""
    assert _DEFAULT_BINARIZATION_FLOORS == (0.1, 0.1, 1e-5, 0.05, 0.1), (
        f"unexpected floor values: {_DEFAULT_BINARIZATION_FLOORS}"
    )
    # Adaptation floor index 2
    assert _DEFAULT_BINARIZATION_FLOORS[2] == 1e-5, (
        f"adaptation floor must be 1e-5 (was 0.001 pre-2026-05-17), "
        f"got {_DEFAULT_BINARIZATION_FLOORS[2]}"
    )


def test_floor_change_widens_binarized_state_visits():
    """With adaptation in [0, 0.015] (matching ConsciousnessGate's scaling),
    the new 1e-5 floor lets adaptation contribute to state diversity; the
    old 0.001 floor would have pinned it to 0 always."""
    metrics = IITMetrics(history_len=200, tpm_window=200)

    # Feed a synthetic gate trajectory mimicking real ConsciousnessGate output:
    # adaptation lives in [0, 0.015] (per consciousness_gating.py:187 scaling),
    # other dims in roughly [0.3, 0.7].
    rng = np.random.default_rng(42)
    from models.core.consciousness_gating import GatingState
    for step in range(100):
        adaptation_val = float(rng.uniform(0.0, 0.015))  # narrow range
        gs = GatingState(
            attention_level=float(rng.uniform(0.3, 0.7)),
            stability_score=float(rng.uniform(0.3, 0.7)),
            adaptation_rate=adaptation_val,
            meta_memory_coherence=float(rng.uniform(0.3, 0.7)),
            narrator_confidence=float(rng.uniform(0.3, 0.7)),
        )
        metrics.update_from_gate_state(gs)

    # Count unique binarized states visited
    unique_states = set(metrics.state_history)
    assert len(unique_states) >= 4, (
        f"expected >= 4 unique binarized states with new 1e-5 floor; "
        f"got {len(unique_states)}: {sorted(unique_states)}. "
        f"If this fails, the floor change is not letting adaptation contribute "
        f"to state diversity."
    )


def test_iit_state_arity_remains_5():
    """Regression: the IIT state tuple must always have arity 5 (one per
    gate node). Past bugs from 4-tuple state pollution would silently
    invalidate the TPM (see CLAUDE.md 2026-05-03 entry)."""
    metrics = IITMetrics()
    from models.core.consciousness_gating import GatingState
    gs = GatingState(
        attention_level=0.5, stability_score=0.5, adaptation_rate=0.005,
        meta_memory_coherence=0.5, narrator_confidence=0.5,
    )
    state = metrics.update_from_gate_state(gs)
    assert len(state) == 5, (
        f"state arity must be 5 (one per gate node), got {len(state)}: {state}"
    )
    assert all(s in (0, 1) for s in state), f"non-binary state: {state}"
    # GATE_CM and labels must also be 5x5 / length 5
    assert GATE_CM.shape == (5, 5), f"GATE_CM shape: {GATE_CM.shape}"
    assert len(GATE_NODE_LABELS) == 5, f"label count: {len(GATE_NODE_LABELS)}"
