"""Tests for the RIIU substrate-probe wiring added in Phase B of plan
let-s-plan-the-next-misty-parasol.md (2026-05-17).

Covers:
  1. --riiu-probe-all causes init_components to build three RIIUPhi instances
  2. RIIUPhi.push validates input dimensionality with ValueError
  3. The substrate-selection logic in compare_phi_pathways.load_metrics maps
     each --substrate value to the right CSV column
  4. --riiu-probe-all without --enable-audio emits a startup RuntimeWarning
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.evaluation.phi_riiu import RIIUPhi
from scripts.training.train_rlhf import build_config, init_components
from scripts.analysis.compare_phi_pathways import load_metrics, SUBSTRATE_COLUMN


def _default_args(**overrides):
    """Build an argparse.Namespace mimicking train_rlhf.py defaults."""
    base = argparse.Namespace(
        episodes=1,
        max_steps=10,
        action_dim=2,
        lr=1e-3,
        render=False,
        env="dark_room",
        difficulty=0,
        log_dir="runs/_test",
        log_ei_every=0,
        enable_audio=False,
        ablate_memory_replay=False,
        ablate_consolidation_fix=False,
        ablate_rnd_zero_on_reward=False,
        ablate_gate_diversity=False,
        ablate_gate_entropy=False,
        ablate_gate_feedback=False,
        ablate_pad_loop=False,
        ablate_bptt=False,
        phi_sample_every=5,
        enable_riiu=True,
        riiu_rank=4,
        riiu_window=16,
        riiu_source="broadcast",
        riiu_probe_all=False,
        seed=42,
        # Phase A defaults (added 2026-05-17)
        broadcast_mode="winner_take_all",
        attention_temperature=0.5,
        attention_floor=0.05,
        # Phase C defaults (added 2026-05-17)
        gate_diversity_loss="off",
        gate_feedback="off",
        # Phase D defaults (added 2026-05-17)
        enable_mock_semantic=False,
        phi1_min_active_modules=0,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_probe_all_instantiates_three_riiu():
    """With --riiu-probe-all, init_components returns a 3-entry RIIUPhi dict."""
    config = build_config(_default_args(riiu_probe_all=True))
    components = init_components(config)
    # Find the RIIUPhi-dict component by type (position-independent so future
    # additions to the returned tuple do not break this test).
    riiu_phis = next(
        (c for c in components if isinstance(c, dict)
         and c and all(isinstance(v, RIIUPhi) for v in c.values())),
        None,
    )
    assert riiu_phis is not None, (
        f"no RIIUPhi dict found in components; got types: "
        f"{[type(c).__name__ for c in components]}"
    )
    assert set(riiu_phis.keys()) == {"broadcast", "tectum", "audio"}, (
        f"unexpected keys: {set(riiu_phis.keys())}"
    )
    for name, instance in riiu_phis.items():
        assert isinstance(instance, RIIUPhi), (
            f"riiu_phis[{name!r}] is {type(instance)}, expected RIIUPhi"
        )
        assert instance.dim == config["workspace_dim"], (
            f"riiu_phis[{name!r}].dim = {instance.dim}, "
            f"expected workspace_dim = {config['workspace_dim']}"
        )


def test_substrate_dim_validation():
    """RIIUPhi.push raises ValueError on shape mismatch, accepts on match."""
    riiu = RIIUPhi(dim=256, rank=16, window=64, device="cpu")
    correct = torch.randn(256)
    wrong = torch.randn(128)
    # Correct dim: no exception
    riiu.push(correct)
    # Wrong dim: ValueError with the dim in the message
    with pytest.raises(ValueError, match="dim=256"):
        riiu.push(wrong)


def test_substrate_load_maps_to_explicit_column(tmp_path: Path):
    """load_metrics with --substrate tectum reads phi_riiu_tectum into phi_riiu."""
    # Build a synthetic metrics.csv with all 4 RIIU columns
    df = pd.DataFrame({
        "global_step": list(range(2000)),
        "phi": [1e-5] * 2000,
        "sync_r": [0.24] * 2000,
        "is_conscious": [0] * 2000,
        "reward": [0.0] * 2000,
        "broadcast_mag": [1.0] * 2000,
        "valence": [0.0] * 2000,
        "arousal": [0.0] * 2000,
        "dominance": [0.0] * 2000,
        "phi_method": ["pyphi"] * 2000,
        "phi_riiu": [1.0e-7] * 2000,             # would-be reward source
        "phi_riiu_broadcast": [1.0e-7] * 2000,    # equal to phi_riiu in this fixture
        "phi_riiu_tectum": [5.0e-4] * 2000,       # distinctively different
        "phi_riiu_audio": [0.0] * 2000,           # zero (audio off)
    })
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    df.to_csv(run_dir / "metrics.csv", index=False)

    # --substrate tectum should overwrite phi_riiu with phi_riiu_tectum values
    loaded = load_metrics(str(run_dir), substrate="tectum")
    assert (loaded["phi_riiu"] == 5.0e-4).all(), (
        "load_metrics(substrate='tectum') did not overwrite phi_riiu with the "
        "tectum column values"
    )
    # --substrate broadcast keeps the broadcast values (which equal the
    # legacy phi_riiu in this fixture)
    loaded_b = load_metrics(str(run_dir), substrate="broadcast")
    assert (loaded_b["phi_riiu"] == 1.0e-7).all()

    # Substrate column lookup table is consistent
    assert SUBSTRATE_COLUMN["broadcast"] == "phi_riiu_broadcast"
    assert SUBSTRATE_COLUMN["tectum"] == "phi_riiu_tectum"
    assert SUBSTRATE_COLUMN["audio"] == "phi_riiu_audio"


def test_audio_degenerate_warn_when_audio_disabled():
    """init_components with --riiu-probe-all but --enable-audio off must warn."""
    config = build_config(_default_args(riiu_probe_all=True, enable_audio=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        init_components(config)
    riiu_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "riiu-probe-all" in str(w.message).lower()
        and "audio" in str(w.message).lower()
    ]
    assert len(riiu_warnings) >= 1, (
        f"expected at least one RIIU audio-degenerate RuntimeWarning, "
        f"got {[str(w.message) for w in caught]}"
    )
