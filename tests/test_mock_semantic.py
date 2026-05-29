"""Tests for Phase D of the 2026-05-17 Phi-1 retest plan: MockSemanticModule
and the multi-modal Phi-1 pre-flight gate in train_rlhf.py.

The strategy of Phase D is to make Phi-1 PHYSICALLY testable on dark_room
(no env subclass; reuse the existing audio mixin and add MockSemantic) by
ensuring more than one active modality is bidding at non-zero magnitude.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.mock_semantic import MockSemanticModule
from scripts.training.train_rlhf import build_config, init_components


def _synthetic_obs(seed: int = 0, h: int = 224, w: int = 224) -> np.ndarray:
    """Build a deterministic synthetic frame mimicking dark_room layout: a
    dark background with a bright spot at a seed-determined location.
    Random-noise frames are too uniform to exercise the tile-stat features.
    """
    rng = np.random.default_rng(seed)
    obs = (rng.normal(20, 5, size=(h, w, 3))).clip(0, 255).astype(np.uint8)
    # Bright spot at a seed-dependent location
    cy = (h // 4) + (seed * 37) % (h // 2)
    cx = (w // 4) + (seed * 53) % (w // 2)
    r = 20
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    obs[mask] = [240, 240, 200]
    return obs


def test_mock_semantic_embedding_shape_and_determinism():
    """Embedding must be 1-D of shape [embedding_dim] and identical for
    the same input across instances built with the same seed."""
    obs = _synthetic_obs(seed=42)
    mod_a = MockSemanticModule(embedding_dim=1536, seed=7)
    mod_b = MockSemanticModule(embedding_dim=1536, seed=7)
    e1 = mod_a.embed(obs)
    e2 = mod_b.embed(obs)
    assert e1.shape == (1536,), f"expected [1536], got {tuple(e1.shape)}"
    assert torch.allclose(e1, e2), "embeddings should be deterministic across instances with same seed"


def test_mock_semantic_bid_floor_and_ceiling():
    """Bid must lie in [0.1, 1.0] for any input (floor ensures the channel
    participates in workspace competition; ceiling matches sigmoid bid range)."""
    mod = MockSemanticModule(embedding_dim=1536)
    for seed in range(10):
        obs = _synthetic_obs(seed=seed)
        emb = mod.embed(obs)
        bid = mod.bid_from_embedding(emb)
        assert 0.1 <= bid <= 1.0, (
            f"bid {bid} out of range [0.1, 1.0] for seed {seed}"
        )


def test_mock_semantic_varies_with_input():
    """Different observations should produce different embeddings (the whole
    point is to be a non-zero, input-dependent signal)."""
    mod = MockSemanticModule(embedding_dim=1536)
    obs_a = _synthetic_obs(seed=1)
    obs_b = _synthetic_obs(seed=2)
    emb_a = mod.embed(obs_a)
    emb_b = mod.embed(obs_b)
    cos = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0)
    assert cos.item() < 0.99, (
        f"embeddings for different obs should differ; got cosine {cos.item():.4f}"
    )


def _default_args(**overrides):
    """Argparse Namespace mimicking train_rlhf.py defaults after Phase A+C+D."""
    base = argparse.Namespace(
        episodes=1, max_steps=10, action_dim=2, lr=1e-3, render=False,
        env="dark_room", difficulty=0, log_dir="runs/_test", log_ei_every=0,
        enable_audio=False,
        ablate_memory_replay=False, ablate_consolidation_fix=False,
        ablate_rnd_zero_on_reward=False,
        gate_diversity_loss="off", gate_feedback="off",
        ablate_gate_diversity=False, ablate_gate_entropy=False,
        ablate_gate_feedback=False, ablate_pad_loop=False, ablate_bptt=False,
        phi_sample_every=5,
        enable_riiu=False, riiu_rank=16, riiu_window=64,
        riiu_source="broadcast", riiu_probe_all=False, seed=None,
        broadcast_mode="winner_take_all",
        attention_temperature=0.5, attention_floor=0.05,
        # Phase D defaults
        enable_mock_semantic=False,
        phi1_min_active_modules=0,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_init_components_with_mock_semantic_enabled():
    """init_components returns a MockSemanticModule instance when the flag is set."""
    config = build_config(_default_args(enable_mock_semantic=True))
    components = init_components(config)
    # Locate by type (position-independent so future additions to the returned
    # tuple do not break this test).
    mock_semantics = [c for c in components if isinstance(c, MockSemanticModule)]
    assert len(mock_semantics) == 1, (
        f"expected exactly one MockSemanticModule, got {len(mock_semantics)}; "
        f"component types: {[type(c).__name__ for c in components]}"
    )


def test_init_components_without_mock_semantic_returns_none():
    """Default config keeps mock_semantic disabled (no MockSemanticModule)."""
    config = build_config(_default_args())
    components = init_components(config)
    mock_semantics = [c for c in components if isinstance(c, MockSemanticModule)]
    assert mock_semantics == []
