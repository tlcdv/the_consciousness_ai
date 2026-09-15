"""Collect a training step's internal values for the session recorder.

The training loop passes `locals()` once per step. These functions only READ
values the loop already computed; they run no forward pass, draw no random
number, and change no tensor, so metrics.csv is identical with recording on.

Tiers (see SessionRecorder):
    scalars  every step of every episode, into steps.jsonl
    vectors  every step of every episode, into vectors.npz
    maps     every step of the recorded episodes, into maps.npz
    full     every step of the first and last episode, float16, into full_*.npy
"""

from __future__ import annotations

import numbers
from typing import Optional

import numpy as np
import torch

LOSS_NAMES = (
    "pred_loss", "topo_loss", "control_loss", "recon_loss", "wm_recon_loss",
    "latent_id_loss", "contrastive_loss", "total_tectum_loss", "sync_loss",
    "gate_diversity_loss", "gate_loss", "rnd_loss", "wm_loss", "m_loss",
    "last_recon_loss", "last_match_loss",
)
REWARD_NAMES = ("env_reward", "reward_for_action_core", "intrinsic_bonus",
                "curiosity_score", "delta_phi", "reward_val")
VECTOR_NAMES = ("tectum_content", "broadcast", "audio_content", "audio_waveform",
                "gate_values_tensor", "sv_for_policy")


def tensor_or_none(value) -> Optional[np.ndarray]:
    """A float32 numpy copy of a tensor or array, or None."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def as_float(value) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.numel() == 1 else None
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def collect_losses(snapshot: dict) -> dict:
    """Every loss the step computed, as floats. A loss the step did not compute is absent."""
    found = {name: as_float(snapshot.get(name)) for name in LOSS_NAMES if name in snapshot}
    return {name: number for name, number in found.items() if number is not None}


def json_safe(value):
    """Numbers, strings, booleans, None and their dicts and lists. Arrays and tensors are left out."""
    if isinstance(value, dict):
        kept = {str(k): json_safe(v) for k, v in value.items()}
        return {k: v for k, v in kept.items() if v is not _DROP}
    if isinstance(value, (list, tuple)):
        return [item for item in (json_safe(v) for v in value) if item is not _DROP]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Number):
        return value.item() if hasattr(value, "item") else value
    return _DROP


_DROP = object()


def modules_with_parameters(named_objects: dict) -> dict:
    """Every nn.Module with parameters, including those held inside plain objects."""
    found = {}
    for name, obj in named_objects.items():
        if isinstance(obj, torch.nn.Module):
            candidates = {name: obj}
        else:
            candidates = {"%s.%s" % (name, attr): held for attr, held in vars(obj).items()
                          if isinstance(held, torch.nn.Module)} if obj is not None else {}
        found.update({k: m for k, m in candidates.items() if any(True for _ in m.parameters())})
    return found


def gradient_norm(module) -> Optional[float]:
    """L2 norm over every parameter gradient the module holds, or None when it holds none."""
    grads = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    if not grads:
        return None
    return float(torch.stack(grads).norm())


def reduce_kl_map(kl_map) -> Optional[np.ndarray]:
    """Sum the per-category KL over axis 2: [B, latents, categories, H, W] -> [B, latents, H, W]."""
    if kl_map is None:
        return None
    return tensor_or_none(kl_map.sum(dim=2))


def collect_scalars(snapshot: dict, workspace, tectum, gate, grad_modules: dict) -> dict:
    """The per-step values that fit in one JSON line.

    Over 20 lines because it is one flat mapping of named values; splitting it
    would scatter the record's schema across helpers.
    """
    settle = snapshot.get("settle_result")
    gate_state = getattr(gate, "state", None)
    return json_safe({
        "kl_div_pre_tanh": getattr(tectum, "_last_kl_div", None),
        "modulated_bids": getattr(workspace, "last_modulated_bids", None),
        "ignition": _ignition(workspace),
        "settle": None if settle is None else {
            "cycles_used": settle.cycles_used, "converged": settle.converged,
            "prediction_errors": [as_float(e) for e in settle.prediction_errors],
            "final_bids": settle.final_bids},
        "phi": {"value": as_float(snapshot.get("phi")), "method": snapshot.get("phi_method"),
                "riiu": as_float(snapshot.get("phi_riiu_val"))},
        "gate": None if gate_state is None else _gate_values(gate_state),
        "emotion": snapshot.get("emotion"), "emotion_post": snapshot.get("emotion_post"),
        "rewards": {name: as_float(snapshot.get(name)) for name in REWARD_NAMES},
        "value_estimate": as_float(snapshot.get("value")),
        "self_prediction": {"mse": as_float(snapshot.get("self_pred_mse")),
                            "skill": as_float(snapshot.get("self_pred_skill"))},
        "losses": collect_losses(snapshot),
        "gradient_norms": {name: gradient_norm(m) for name, m in grad_modules.items()},
        "info_after_step": snapshot.get("info"),
        "learned_valence": _learned_valence(snapshot.get("modulator")),
        "audio_surprise_error": getattr(getattr(snapshot.get("auditory_specialist"),
                                              "surprise", None), "last_error", None),
    })


def _learned_valence(modulator) -> Optional[dict]:
    learned = getattr(modulator, "learned_valence", None)
    if learned is None:
        return None
    return {"values": dict(learned.values), "td_error": learned.last_td_error}


def _ignition(workspace) -> dict:
    state = workspace.state
    return {"salience": getattr(state, "ignition_salience", None),
            "threshold": getattr(workspace, "ignition_threshold", None),
            "baseline": getattr(workspace, "_energy_baseline", None),
            "broadcast_strength": as_float(getattr(state, "broadcast_strength", None))}


def _gate_values(gate_state) -> dict:
    names = ("attention_level", "stability_score", "adaptation_rate",
             "meta_memory_coherence", "narrator_confidence")
    return {name: as_float(getattr(gate_state, name, None)) for name in names}


def collect_vectors(snapshot: dict, workspace, tectum, auditory_specialist) -> dict:
    """Per-step vectors small enough to keep for every episode."""
    vectors = {name: tensor_or_none(snapshot.get(name)) for name in VECTOR_NAMES}
    affect = snapshot.get("audio_affect")
    binding = workspace.binding_system
    vectors.update({
        "binding_phases": tensor_or_none(getattr(binding, "current_phases", None)),
        "capsule_poses": tensor_or_none(getattr(tectum, "_last_capsule_poses", None)),
        "capsule_activities": tensor_or_none(getattr(tectum, "_last_capsule_activities", None)),
        "acoustic_features": None if affect is None else tensor_or_none(affect.get("acoustic_features")),
        "audio_tonotopic": tensor_or_none(getattr(auditory_specialist, "_last_tonotopic", None)),
        "audio_spatial": _first_feature_channel(getattr(auditory_specialist, "_last_spatial", None)),
    })
    return {name: array for name, array in vectors.items() if array is not None}


def _first_feature_channel(spatial) -> Optional[np.ndarray]:
    """[B, feature_dim, 2] repeats one (azimuth, elevation) pair; keep [B, 2]."""
    return None if spatial is None else tensor_or_none(spatial[:, 0, :])


def collect_maps(snapshot: dict, tectum) -> dict:
    """Spatial maps, kept for the recorded episodes."""
    maps = {
        "obs_map": tensor_or_none(getattr(tectum, "_last_obs_map", None)),
        "h_state": tensor_or_none(getattr(tectum, "h_state", None)),
        "kl_map_per_latent": reduce_kl_map(getattr(tectum, "_last_kl_map", None)),
        "policy_state": tensor_or_none(snapshot.get("policy_state")),
    }
    return {name: array for name, array in maps.items() if array is not None}


def collect_full(tectum) -> dict:
    """The full world-model state, kept for the first and last episode."""
    full = {
        "z_state": getattr(tectum, "z_state", None),
        "prior_logits": getattr(tectum, "_last_prior_logits", None),
        "post_logits": getattr(tectum, "_last_post_logits", None),
        "kl_map": getattr(tectum, "_last_kl_map", None),
    }
    return {name: tensor_or_none(t) for name, t in full.items() if t is not None}
