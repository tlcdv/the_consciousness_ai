"""Why does the workspace content hardly depend on the tectum map? Read-only, no change of weights.

docs/results/tectal_attention_2026_10.md found that a gain of 1.5 on the map moves the content by at
most 4.25e-05 of its size. This probe finds where along the path the effect falls. Written and pushed
before its first run.

HYPOTHESIS, from reading the code and the saved weights, not yet measured. `squash` shrinks a short
vector roughly by the square of its length, it runs at every capsule level, and the routing matrices
keep their starting scale of about 0.01. So the poses are very short. `workspace_proj` adds a fixed
bias, so the content would be mostly that bias. `squash` keeps the relative change of a short vector,
so the relative change would fall at the projection and the absolute size at the capsules.

Part 1, layer table. Saved weights in training and in evaluation mode, identical noise draws in the
compared passes, room seed 521, 2 episodes of 60 steps, steps after 15. Arms on the output of
`topo_map`: none, the gain of 1.5 on the group cells, a random direction of the same norm as that gain's
change, and a gain of 1. Dark room runs use the binder's group of the previous step. DMTS checkpoints
have no binder, so they use a fixed central block of 36 cells and are a contrast only.

  Controls, which raise on failure. A gain of 1 equals no perturbation exactly at every tap. At
  `obs_map`, the random arm's relative change equals the gain arm's within 1 percent.

Part 2, offset share. Recorded vectors of the last episode with saved weights. Part 3, live decode.
Recorded vectors of the runs, whole episodes held out. Labels: the direction of the object with the
larger x offset, which needs no tone, and the tone band of the selected group.

PRE-STATED PREDICTIONS, each at 3 of 3 seeds of `vg_learn_s601..603` and `ta_nolink_s801..803`:
  P1  per-element rms falls at each of the three routing levels, in both modes.
  P2  |content - bias| / |content| <= 0.1, in both modes.
  P3  the largest fall of the gain's relative change along obs_map, state_tensor, primary, flat, content
      is the projection (flat to content), in evaluation mode.
Pre-stated at 3 of 3 seeds of `vg_learn_s601..603`, on the episodes with saved maps:
  P4  the direction score from `obs_map` >= 0.5 and from `tectum_content` <= 0.2.

WHAT EACH OUTCOME POINTS TO, for the later design review and not a design choice here:
  Largest fall at the RSSM: a route around the latent. At primary or routing: the capsule stage.
  At the projection: the scale of the poses against the bias.
  No fall of 100 times or more at any stage: only sampling noise, so the posterior mean in the path.

Usage:
    python -m scripts.analysis.probe_content_map_sensitivity --runs runs --output-json out.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from models.audio.auditory_specialist import AuditorySpecialist
from models.core.rhythm_binding import RhythmMapBinding, fresh_band_envelope
from models.core.sensory_tectum import SensoryTectum
from models.core.tectal_attention import attend, attention_field
from scripts.analysis.diagnose_attention_modes import CONFIG, Cochlea, as_image, build_room, relative
from scripts.analysis.probe_tectal_attention import direction_score, selected_object
from simulations.environments.dmts_env import DMTSEnv

GAIN, ROOM_SEED, EPISODES, STEPS, SETTLE = 1.5, 521, 2, 60, 15
MODES = ("eval", "train")
TAPS = ("obs_map", "posterior_mean", "z_t", "h_t", "state_tensor", "primary",
        "level_0", "level_1", "level_2", "flat", "content")
STAGES = (("rssm", "obs_map", "state_tensor"), ("primary", "state_tensor", "primary"),
          ("routing", "primary", "flat"), ("projection", "flat", "content"))
BLOCK = [row * 16 + column for row in range(6, 12) for column in range(5, 11)]
DARK_ROOM = tuple("ta_nolink_s%d" % s for s in (801, 802, 803)) + tuple("vg_learn_s%d" % s for s in (601, 602, 603))
DMTS = ("capfix_alllevels", "capfix_seed43", "capfix_seed44")
RIDGE_ALPHA, FOLDS, NULL_SHUFFLES = 100.0, 5, 20


def rms(tensor: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(tensor))))


def locate_fall(changes: Dict[str, float]) -> Tuple[str, float]:
    """The stage where the relative change falls most, and by what factor."""
    falls = {name: changes[start] / max(changes[end], 1e-30) for name, start, end in STAGES}
    worst = max(falls, key=falls.get)
    return worst, float(falls[worst])


def offset_share(vectors: np.ndarray, bias: np.ndarray) -> Dict[str, float]:
    """The fixed part of [steps, dim] vectors, and how close it lies to the projection bias."""
    mean = vectors.mean(axis=0)
    centred = vectors - mean
    singular = np.linalg.svd(centred, compute_uv=False) ** 2
    share = np.cumsum(singular) / max(singular.sum(), 1e-30)
    cosine = float(mean @ bias / max(np.linalg.norm(mean) * np.linalg.norm(bias), 1e-30))
    return {"mean_norm": float(np.linalg.norm(mean)), "centred_rms_norm": float(np.sqrt((centred ** 2).sum(1).mean())),
            "bias_norm": float(np.linalg.norm(bias)), "cosine_mean_bias": cosine,
            "mean_minus_bias_over_mean": float(np.linalg.norm(mean - bias) / max(np.linalg.norm(mean), 1e-30)),
            "pca_dims_95": int(np.searchsorted(share, 0.95) + 1)}


def right_object_direction(facts: dict) -> np.ndarray:
    """Unit vector to the object with the larger x offset. The two objects look alike, so this needs no tone."""
    light = np.asarray(facts["_truth_light_offset"], dtype=float)
    decoy = np.asarray(facts["_truth_decoy_offset"], dtype=float)
    pick = light if light[0] >= decoy[0] else decoy
    return pick / max(np.linalg.norm(pick), 1e-12)


def swapped_labels(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each episode gets the labels of another episode, cut or repeated to its own length."""
    episodes = np.unique(groups)
    shuffled = np.array(y, copy=True)
    for episode, source in zip(episodes, np.roll(rng.permutation(episodes), 1)):
        own, other = np.flatnonzero(groups == episode), y[groups == source]
        shuffled[own] = other[np.arange(len(own)) % len(other)]
    return shuffled


def tone_accuracy(x: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Held-out accuracy of a ridge read of a two-class label, whole episodes out."""
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("the tone label needs exactly 2 classes, got %s" % classes)
    target = np.where(labels == classes[1], 1.0, -1.0)[:, None]
    episodes = np.unique(groups)
    hits, total = 0, 0
    for k in range(FOLDS):
        test = np.isin(groups, episodes[k::FOLDS])
        if not test.any() or test.all():
            continue
        mean, scale = x[~test].mean(0), x[~test].std(0)
        scale = np.where(scale == 0, 1.0, scale)
        a, b = (x[~test] - mean) / scale, (x[test] - mean) / scale
        gram = a @ a.T + RIDGE_ALPHA * np.eye(len(a))
        centre = target[~test].mean()
        guess = b @ (a.T @ np.linalg.solve(gram, target[~test] - centre)) + centre
        hits += int(((guess > 0) == (target[test] > 0)).sum())
        total += int(test.sum())
    return hits / max(total, 1)


def build_tectum(state: dict, ear: Optional[Cochlea], mode: str) -> SensoryTectum:
    tectum = SensoryTectum(CONFIG)
    missing, unexpected = tectum.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("weights do not match: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    tectum.train(mode == "train")
    if ear is not None:
        tectum.map_binding = RhythmMapBinding(fresh_band_envelope(ear), bands=ear.specialist.num_bands)
    tectum.primary_tap, tectum.history_start = None, 0

    def keep_primary(module, inputs, output):
        tectum.primary_tap = output.detach()
    tectum.capsule_layer.primary.register_forward_hook(keep_primary)
    return tectum


def perturb(tectum: SensoryTectum, kind: str, gain: float) -> None:
    """A forward hook on `topo_map`, as the gain of `_wire_tectal_attention`, plus a random arm."""
    generator = torch.Generator().manual_seed(0)

    def hook(module, inputs, output):
        if tectum.map_binding is None:
            cells = BLOCK
        else:
            fresh = len(tectum.map_history) > tectum.history_start
            cells = tectum.map_history[-1].get("group", []) if fresh else []
        if not cells:
            return output
        attended = attend(output, attention_field(cells, output.shape[-1], gain))
        if kind == "gain":
            return attended
        noise = torch.randn(output.shape, generator=generator, dtype=output.dtype)
        return output + noise * (attended - output).norm() / noise.norm()
    tectum.topo_map.register_forward_hook(hook)


def read_taps(tectum: SensoryTectum, content: torch.Tensor) -> Dict[str, np.ndarray]:
    levels = [poses.reshape(-1) for poses in tectum.capsule_layer._level_poses]
    tensors = {"obs_map": tectum._last_obs_map, "posterior_mean": tectum._last_post_logits, "z_t": tectum.z_state,
               "h_t": tectum.h_state, "state_tensor": tectum._last_state_tensor, "primary": tectum.primary_tap,
               "flat": torch.cat(levels), "content": content}
    tensors.update({"level_%d" % i: poses for i, poses in enumerate(levels)})
    return {name: tensor.detach().reshape(-1).numpy().astype(np.float64) for name, tensor in tensors.items()}


def room_frames(arms: Dict[str, SensoryTectum], room, ear: Cochlea) -> Iterator[Tuple[int, int, np.ndarray]]:
    for episode in range(EPISODES):
        frame, facts = room.reset()
        for tectum in arms.values():
            tectum.history_start = len(tectum.map_history)
        for t in range(STEPS):
            if "audio_waveform" in facts:
                ear.hear(facts["audio_waveform"])
            yield episode, t, frame
            frame, _, _, _, facts = room.step(np.zeros(2, dtype=np.float32))


def dmts_frames() -> Iterator[Tuple[int, int, np.ndarray]]:
    room = DMTSEnv(num_trials=20)
    for episode in range(EPISODES):
        frame, _ = room.reset(seed=ROOM_SEED + episode)
        for t in range(STEPS):
            yield episode, t, frame
            frame, _, done, cut, _ = room.step(0)
            if done or cut:
                frame, _ = room.reset(seed=ROOM_SEED + episode + 100 + t)


def layer_table(state: dict, specialist, mode: str, dark_room: bool) -> dict:
    """Mean relative changes, rms and the bias share at each tap, with the two controls."""
    ear = Cochlea(specialist) if dark_room else None
    arms = {kind: build_tectum(state, ear, mode) for kind in ("plain", "gain", "random", "unit")}
    perturb(arms["gain"], "gain", GAIN)
    perturb(arms["random"], "random", GAIN)
    perturb(arms["unit"], "gain", 1.0)
    frames = room_frames(arms, build_room(ROOM_SEED), ear) if dark_room else dmts_frames()
    bias = arms["plain"].capsule_layer.workspace_proj.bias.detach().numpy().astype(np.float64)
    rows: Dict[str, Dict[str, list]] = {name: {"gain": [], "random": [], "step": [], "rms": []} for name in TAPS}
    moving, previous = [], None
    for episode, t, frame in frames:
        image, noise = as_image(frame), 1000 * episode + t
        taps = {}
        for kind, tectum in arms.items():
            torch.manual_seed(noise)
            with torch.no_grad():
                content, _ = tectum(image, torch.zeros(1, 64, 2))
            taps[kind] = read_taps(tectum, content)
        if not all(np.array_equal(taps["plain"][name], taps["unit"][name]) for name in TAPS):
            raise RuntimeError("the control with gain 1 differs from no perturbation (%s mode)" % mode)
        if t > SETTLE:
            base = taps["plain"]
            for name in TAPS:
                rows[name]["gain"].append(relative(taps["gain"][name] - base[name], base[name]))
                rows[name]["random"].append(relative(taps["random"][name] - base[name], base[name]))
                rows[name]["step"].append(relative(base[name] - previous[name], base[name]))
                rows[name]["rms"].append(rms(base[name]))
            moving.append(relative(base["content"] - bias, base["content"]))
        previous = taps["plain"]
    table = {name: {key: float(np.mean(cells)) for key, cells in row.items()} for name, row in rows.items()}
    start = table["obs_map"]
    if dark_room and abs(start["gain"] - start["random"]) > 0.01 * start["gain"]:
        raise RuntimeError("the random arm's size at obs_map differs from the gain arm's")
    return {"taps": table, "moving_share": float(np.mean(moving)),
            "fall": locate_fall({name: table[name]["gain"] for name in TAPS})}


def load_state(path: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("tectum", state)
    return {key: tensor for key, tensor in state.items() if key not in ("h_state", "z_state")}


def episode_folder(run: str, episode: int) -> str:
    return os.path.join(run, "episodes", "ep_%04d" % episode)


def recorded_samples(run: str, with_maps: bool) -> Dict[str, tuple]:
    """Per stream (x, direction labels, episodes), and the band of the selected group for the tone."""
    with open(os.path.join(run, "map_binding.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    streams: Dict[str, tuple] = {}
    episodes = sorted(os.listdir(os.path.join(run, "episodes")))
    for number, folder in enumerate(episodes):
        root = os.path.join(run, "episodes", folder)
        if with_maps and not os.path.exists(os.path.join(root, "maps.npz")):
            continue
        with open(os.path.join(root, "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle]
        tables = dict(np.load(os.path.join(root, "vectors.npz")))
        if with_maps:
            tables.update(dict(np.load(os.path.join(root, "maps.npz"))))
        names = ("tectum_content", "broadcast", "capsule_poses") + (("obs_map", "h_state") if with_maps else ())
        for name in names:
            where = {int(s): i for i, s in enumerate(tables[name + "__steps"])}
            x, y, g, band = streams.setdefault(name, ([], [], [], []))
            for t in range(1, len(steps)):
                if t not in where:
                    continue
                facts = steps[t]["internals"]["info_after_step"]
                record = history[number * 200 + t - 1]
                x.append(np.asarray(tables[name][where[t]], dtype=np.float32).ravel())
                y.append(right_object_direction(facts))
                g.append(number)
                band.append(record.get("band", -1) if selected_object(record.get("group", []), facts) else -1)
    return {name: tuple(np.array(part) for part in parts) for name, parts in streams.items()}


def decode_run(run: str, with_maps: bool) -> dict:
    """Direction score with a null of swapped episodes, and the tone accuracy with its majority share."""
    rng = np.random.default_rng(0)
    scores = {}
    for name, (x, y, groups, bands) in recorded_samples(run, with_maps).items():
        null = [direction_score(x, swapped_labels(y, groups, rng), groups) for _ in range(NULL_SHUFFLES)]
        scores[name] = {"direction": direction_score(x, y, groups), "null_p95": float(np.percentile(null, 95)),
                        "steps": int(len(y))}
        keep = bands >= 0
        found, counts = np.unique(bands[keep], return_counts=True)
        if len(found) >= 2:
            pair = found[np.argsort(counts)[-2:]]
            two = np.isin(bands, pair)
            scores[name]["tone"] = tone_accuracy(x[two], bands[two], groups[two])
            scores[name]["tone_majority"] = float(np.sort(counts)[-1] / np.sort(counts)[-2:].sum())
    return scores


def predictions(tables: dict, decodes: dict) -> Dict[str, bool]:
    """P1 to P4 from the measured tables."""
    p1, p2, p3 = True, True, True
    for run in DARK_ROOM:
        for mode in MODES:
            cell = tables[run][mode]
            levels = [cell["taps"][name]["rms"] for name in ("primary", "level_0", "level_1", "level_2")]
            p1 &= all(lower > upper for lower, upper in zip(levels, levels[1:]))
            p2 &= cell["moving_share"] <= 0.1
        p3 &= tables[run]["eval"]["fall"][0] == "projection"
    p4 = all(decodes[run]["maps"]["obs_map"]["direction"] >= 0.5 and
             decodes[run]["maps"]["tectum_content"]["direction"] <= 0.2 for run in DARK_ROOM if run.startswith("vg_"))
    return {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3), "P4": bool(p4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    specialist = AuditorySpecialist({})
    specialist.eval()
    report: dict = {"tables": {}, "offset": {}, "decode": {}}
    for run in DARK_ROOM + DMTS:
        dark = run in DARK_ROOM
        path = os.path.join(args.runs, run, "episodes", "ep_0039", "weights.pt") if dark else os.path.join(args.runs, run, "tectum.pt")
        state = load_state(path)
        report["tables"][run] = {mode: layer_table(state, specialist, mode, dark) for mode in MODES}
        print(run, {mode: (report["tables"][run][mode]["fall"], report["tables"][run][mode]["moving_share"]) for mode in MODES}, flush=True)
        if dark:
            vectors = np.load(os.path.join(episode_folder(os.path.join(args.runs, run), 39), "vectors.npz"))
            bias = state["capsule_layer.workspace_proj.bias"].numpy().astype(np.float64)
            report["offset"][run] = {name: offset_share(vectors[name].reshape(len(vectors[name]), -1).astype(np.float64), bias)
                                     for name in ("tectum_content", "broadcast")}
            print(run, "offset", report["offset"][run]["tectum_content"], flush=True)
    for run in DARK_ROOM:
        root = os.path.join(args.runs, run)
        report["decode"][run] = {"all": decode_run(root, False), "maps": decode_run(root, True)}
        print(run, "decode", json.dumps(report["decode"][run]), flush=True)
    report["predictions"] = predictions(report["tables"], report["decode"])
    print("PREDICTIONS:", report["predictions"], flush=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
