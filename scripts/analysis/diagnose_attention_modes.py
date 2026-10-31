"""Diagnosis of the tectal attention gate. Written after the gate's data, and reported only.

docs/results/tectal_attention_2026_10.md. The gate FAILED in live runs after an offline check of the
same gain had passed. Three read-only parts, with no change of weights:

  arms     recorded runs, link against no link at the same steps. It reads the binding groups, the
           gain on the recorded map, the content, the recurrent state and the vision bid.
  effect   fixed weights and identical noise draws. It reads the change of `workspace_content` from
           the gain against its change from one step to the next, in evaluation and in training mode.
  offline  the offline check of docs/decisions/2026_10_17_tectal_attention.md with fixed weights, in
           evaluation mode (as it ran) and in training mode (the mode of the live runs).

Control: a gain of 1 leaves the map unchanged, so with identical noise draws its content must equal
the no-gain content exactly. `gain_effect` raises if it does not.

Usage:
    python -m scripts.analysis.diagnose_attention_modes --runs runs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from models.audio.auditory_specialist import AuditorySpecialist
from models.core.rhythm_binding import RhythmMapBinding, fresh_band_envelope
from models.core.sensory_tectum import SensoryTectum
from scripts.analysis.probe_tectal_attention import direction_score, selected_object
from scripts.training.train_rlhf import _wire_tectal_attention
from simulations.environments.simple_visual_env import SimpleVisualEnv

CONFIG = {"tectum_feature_dim": 64, "tectum_grid_size": 16, "workspace_dim": 256,
          "rssm_latent_mode": "continuous", "capsule_workspace_source": "all_levels",
          "vision_bid_reduction": "zscore"}
GAIN, ROOM_SEED = 1.5, 521
EFFECT_EPISODES, OFFLINE_EPISODES, STEPS, SETTLE = 2, 40, 60, 15
MODES = ("eval", "train")


class Cochlea:
    """The frozen band envelope of the room's sound, stored as the training loop stores it."""

    def __init__(self, specialist):
        self.specialist = specialist
        self.last_band_envelope, self.band_envelope_count = None, 0

    def hear(self, waveform) -> None:
        wave = torch.from_numpy(np.asarray(waveform)).float()[None].mean(1, keepdim=True)
        with torch.no_grad():
            hair = self.specialist.hair_cell(self.specialist.filterbank(wave))
        self.last_band_envelope = hair[0, :self.specialist.num_bands].mean(-1).double().numpy()
        self.band_envelope_count += 1


def load_tectum_state(weights: str) -> dict:
    """The saved tectum, without the recurrent state of the run's last step."""
    state = torch.load(weights, map_location="cpu", weights_only=False)["tectum"]
    return {key: tensor for key, tensor in state.items() if key not in ("h_state", "z_state")}


def build_tectum(state: dict, ear: Cochlea, mode: str) -> SensoryTectum:
    """A tectum with the saved weights, its own rhythm binder, in evaluation or training mode."""
    tectum = SensoryTectum(CONFIG)
    missing, unexpected = tectum.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("weights do not match: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    tectum.train(mode == "train")
    tectum.map_binding = RhythmMapBinding(fresh_band_envelope(ear), bands=ear.specialist.num_bands)
    return tectum


def build_room(seed: int) -> SimpleVisualEnv:
    """The covert room of the gate, seeded in its global stream, its audio and its decoy."""
    np.random.seed(seed)
    room = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4, report_collision=True,
                           view="agent_centered", agent_mark="ring", agent_colour=(217, 119, 87),
                           decoy_sound=True, decoy_level=(0.5, 3.0), decoy_visible=True, start_between=True,
                           pulse=True, sound_direction=False)
    room.seed_audio(seed)
    room.seed_decoy(seed)
    return room


def as_image(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1)[None]


def content_of(tectum: SensoryTectum, image: torch.Tensor, noise_seed: int) -> np.ndarray:
    """`workspace_content` of one forward pass, with the noise draws fixed by `noise_seed`."""
    torch.manual_seed(noise_seed)
    with torch.no_grad():
        content, _ = tectum(image, torch.zeros(1, 64, 2))
    return content.reshape(-1).numpy().astype(float)


def relative(change: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(change) / np.linalg.norm(reference))


def gain_effect(state: dict, specialist, mode: str) -> Dict[str, float]:
    """Mean relative change of the content from the gain and from one step to the next."""
    room, ear = build_room(ROOM_SEED), Cochlea(specialist)
    plain, gained, unit = (build_tectum(state, ear, mode) for _ in range(3))
    _wire_tectal_attention(gained, GAIN, None, env=room)
    _wire_tectal_attention(unit, 1.0, None, env=room)
    from_gain, from_step, previous = [], [], None
    for episode in range(EFFECT_EPISODES):
        frame, facts = room.reset()
        for t in range(STEPS):
            if "audio_waveform" in facts:
                ear.hear(facts["audio_waveform"])
            image, noise = as_image(frame), 1000 * episode + t
            base = content_of(plain, image, noise)
            boosted = content_of(gained, image, noise)
            if not np.array_equal(base, content_of(unit, image, noise)):
                raise RuntimeError("the control with gain 1 differs from no gain (%s mode)" % mode)
            if t > SETTLE:
                from_gain.append(relative(boosted - base, base))
                from_step.append(relative(base - previous, base))
            previous = base
            frame, _, _, _, facts = room.step(np.zeros(2, dtype=np.float32))
    return {"gain": float(np.mean(from_gain)), "step": float(np.mean(from_step))}


def selected_direction(tectum: SensoryTectum, offsets: dict) -> Optional[np.ndarray]:
    """Unit vector to the object nearer to the group of the previous step, or None without one."""
    if len(tectum.map_history) < 2:
        return None
    target = selected_object(tectum.map_history[-2].get("group", []), offsets)
    if target is None:
        return None
    vector = np.asarray(offsets["_truth_%s_offset" % target], dtype=float)
    return vector / np.linalg.norm(vector)


def offline_scores(state: dict, specialist, mode: str) -> Dict[str, float]:
    """The score of the offline check, no gain and gain, for one room sequence in one mode."""
    room, ear = build_room(ROOM_SEED), Cochlea(specialist)
    arms = {"no gain": build_tectum(state, ear, mode), "gain": build_tectum(state, ear, mode)}
    _wire_tectal_attention(arms["gain"], GAIN, None, env=room)
    rows = {name: ([], [], []) for name in arms}
    for episode in range(OFFLINE_EPISODES):
        frame, facts = room.reset()
        for t in range(STEPS):
            if "audio_waveform" in facts:
                ear.hear(facts["audio_waveform"])
            image = as_image(frame)
            offsets = {"_truth_light_offset": (room.light_pos - room.agent_pos).tolist(),
                       "_truth_decoy_offset": (room.decoy_pos - room.agent_pos).tolist()}
            for name, tectum in arms.items():
                content = content_of(tectum, image, 1000 * episode + t)
                label = selected_direction(tectum, offsets) if t > 0 else None
                if label is not None:
                    for part, row in zip(rows[name], (content, label, episode)):
                        part.append(row)
            frame, _, _, _, facts = room.step(np.zeros(2, dtype=np.float32))
    return {name: direction_score(*(np.array(part) for part in rows[name])) for name in arms}


def binding_groups(run: str) -> List[list]:
    with open(os.path.join(run, "map_binding.json"), encoding="utf-8") as handle:
        return [record.get("group", []) for record in json.load(handle)["history"]]


def vision_bids(run: str) -> np.ndarray:
    with open(os.path.join(run, "metrics.csv"), encoding="utf-8", newline="") as handle:
        return np.array([float(row["bid_vision"]) for row in csv.DictReader(handle)])


def last_episode_with_maps(link: str, nolink: str) -> str:
    """The last recorded episode that holds the maps in both arms."""
    def recorded(run: str) -> set:
        return {os.path.basename(os.path.dirname(path))
                for path in glob.glob(os.path.join(run, "episodes", "ep_*", "maps.npz"))}
    return sorted(recorded(link) & recorded(nolink))[-1]


def map_gain_ratio(link_map: np.ndarray, nolink_map: np.ndarray, cells: list) -> float:
    """(group cells / other cells) of the link map, over the same quantity of the no-link map."""
    mask = np.zeros(link_map.shape[-1] * link_map.shape[-2], dtype=bool)
    mask[cells] = True

    def contrast(one_map: np.ndarray) -> float:
        cell_size = np.abs(one_map).mean(axis=tuple(range(one_map.ndim - 2))).reshape(-1)
        return float(cell_size[mask].mean() / cell_size[~mask].mean())
    return contrast(link_map) / contrast(nolink_map)


def compare_arms(link: str, nolink: str) -> dict:
    """Link against no link, with the groups of the whole run and the maps of one recorded episode."""
    groups = binding_groups(link)
    other = binding_groups(nolink)
    episode = last_episode_with_maps(link, nolink)
    start = int(episode.split("_")[1]) * 200
    link_maps = np.load(os.path.join(link, "episodes", episode, "maps.npz"))
    nolink_maps = np.load(os.path.join(nolink, "episodes", episode, "maps.npz"))
    a = np.load(os.path.join(link, "episodes", episode, "vectors.npz"))["tectum_content"].reshape(200, -1)
    b = np.load(os.path.join(nolink, "episodes", episode, "vectors.npz"))["tectum_content"].reshape(200, -1)
    ratios = [map_gain_ratio(link_maps["obs_map"][t], nolink_maps["obs_map"][t], groups[start + t - 1])
              for t in range(1, 200) if groups[start + t - 1]]
    cosines = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    bids = {"link": vision_bids(link), "no link": vision_bids(nolink)}
    return {"episode": episode, "same_groups": sum(x == y for x, y in zip(groups, other)), "records": len(groups),
            "map_ratio": (float(np.min(ratios)), float(np.median(ratios)), float(np.max(ratios))),
            "cosine": (float(cosines.min()), float(cosines.mean())),
            "content_between_arms": float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1))),
            "content_step": float(np.mean(np.linalg.norm(b[1:] - b[:-1], axis=1) / np.linalg.norm(b[1:], axis=1))),
            "h_state_max": float(np.abs(link_maps["h_state"] - nolink_maps["h_state"]).max()),
            "bid_vision": {name: (float(v.min()), float(v.mean()), float(v.max())) for name, v in bids.items()},
            "bid_vision_max_difference": float(np.abs(bids["link"] - bids["no link"]).max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[801, 802, 803])
    parser.add_argument("--offline-weights", nargs="+",
                        default=["vg_learn_s601/episodes/ep_0039/weights.pt", "ta_nolink_s801/episodes/ep_0039/weights.pt"])
    args = parser.parse_args()
    specialist = AuditorySpecialist({})
    specialist.eval()
    for seed in args.seeds:
        link, nolink = (os.path.join(args.runs, "ta_%s_s%d" % (arm, seed)) for arm in ("link", "nolink"))
        print("arms s%d: %s" % (seed, json.dumps(compare_arms(link, nolink))), flush=True)
    for seed in args.seeds:
        state = load_tectum_state(os.path.join(args.runs, "ta_nolink_s%d" % seed, "episodes", "ep_0039", "weights.pt"))
        for mode in MODES:
            print("effect s%d %s mode: %s" % (seed, mode, json.dumps(gain_effect(state, specialist, mode))), flush=True)
    for weights in args.offline_weights:
        state = load_tectum_state(os.path.join(args.runs, weights))
        for mode in MODES:
            print("offline %s %s mode: %s" % (weights, mode, json.dumps(offline_scores(state, specialist, mode))), flush=True)


if __name__ == "__main__":
    main()
