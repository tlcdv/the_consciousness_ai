"""Record training episodes so a session can be replayed.

The run folder is the session record. The ethics manifest is written first
(models/ethics/framework.py); this recorder adds `session.json` and, for EVERY
episode, `episodes/ep_NNNN/` with the light tier:

    frames.npz    the frames the agent received, uint8, one per step
    steps.jsonl   one line per step: which input each module received, the raw
                  bids, the bound bids, winner, ignition, sync_R, reward, action,
                  environment phase, and the interoceptive state
    vectors.npz   per step vectors such as tectum_content, broadcast and audio
                  (VECTOR_NAMES in session_capture.py)
    meta.json     step count, frame count, whether they match, and the tiers

The selected episodes (--record-episodes) add maps.npz and weights.pt, and the first
and last episodes add full tensors; SessionRecorder lists the tiers.

Recording which input each module received is the point. A module fed zeros or a
constant produces a constant bid, and a constant bid looks like a mechanism
failure unless the missing input is visible next to it
(docs/results/modality_starvation_2026_09.md).

The recorder only reads values the training loop already computed. It uses no
random state, so a run with recording on logs the same metrics as a run without.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # Python before 3.8
    import importlib_metadata

from models.ethics.framework import git_output
from scripts.training.session_capture import json_safe

RECORD_EVERY = 10
SILENCE_THRESHOLD = 1e-7  # the auditory specialist's own all-zero test
DEFAULT_MEMORY_BID = 0.1
# Heavy tiers are skipped below this much free disk space.
MIN_FREE_GB = 20.0


def parse_record_episodes(spec: str, total_episodes: int) -> frozenset:
    """Episodes to record: "default", "none", or a comma list of indices."""
    if spec == "none":
        return frozenset()
    if spec == "default":
        chosen = set(range(0, total_episodes, RECORD_EVERY)) | {total_episodes - 1}
        return frozenset(i for i in chosen if 0 <= i < total_episodes)
    try:
        chosen = {int(part) for part in spec.split(",") if part.strip()}
    except ValueError:
        raise ValueError(
            "--record-episodes takes 'default', 'none', or a comma list of "
            "episode indices; got %r" % spec)
    return frozenset(i for i in chosen if 0 <= i < total_episodes)


def audio_input_source(has_specialist: bool, waveform) -> str:
    """'sound', 'silent' (a waveform of zeros), or 'none' (no audio reached the module)."""
    if not has_specialist or waveform is None:
        return "none"
    if float(np.max(np.abs(waveform))) < SILENCE_THRESHOLD:
        return "silent"
    return "sound"


def memory_input_source(memory_bid: float) -> str:
    """'retrieved' when a stored experience raised the bid, else 'default'."""
    return "retrieved" if memory_bid > DEFAULT_MEMORY_BID else "default"


def body_input_source(has_self_model: bool, drive_off: bool) -> str:
    """Where the body bid comes from. None of these is sensory data."""
    if not has_self_model:
        return "constant"
    if drive_off:
        return "constant_by_ethics_framework"
    return "energy_counter"


def module_is_optimized(module, optimizers: Iterable) -> Optional[bool]:
    """True when any optimizer updates a parameter of the module. None if the module is absent."""
    if module is None:
        return None
    owned = {id(parameter) for parameter in module.parameters()}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if any(id(parameter) in owned for parameter in group["params"]):
                return True
    return False


def describe_modules(tectum, auditory_specialist, mock_semantic, has_self_model: bool,
                     drive_off: bool, optimizers: Iterable) -> dict:
    """Static facts per module for session.json: where its input comes from, and
    whether any optimizer trains it (None when the module is not a network)."""
    optimizers = list(optimizers)
    semantic_net = mock_semantic if hasattr(mock_semantic, "parameters") else None
    return {
        "vision": {"input": "environment frame",
                   "trained": module_is_optimized(tectum, optimizers)},
        "audio": {"enabled": auditory_specialist is not None,
                  "trained": module_is_optimized(auditory_specialist, optimizers + list(
                      getattr(auditory_specialist, "owned_optimizers", lambda: [])()))},
        "memory": {"input": "retrieval from stored experiences", "trained": None},
        "body": {"input": body_input_source(has_self_model, drive_off), "trained": None},
        "semantic": {"input": "mock embedding" if mock_semantic is not None else "none",
                     "trained": module_is_optimized(semantic_net, optimizers)},
    }


TIER_ORDER = ("light", "maps", "full", "weights")


class SessionRecorder:
    """Writes the session record for one training run.

    Tiers per episode (owner decision 2026-09-15):
        light    every episode: frames, steps.jsonl, vectors.npz
        maps     recorded episodes: maps.npz
        full     first and last episode: full_<name>.npy, float16, one row per step
        weights  recorded episodes: weights.pt with every trained module
    Heavy tiers are skipped, and the skip written to meta.json, when free disk space
    is below `min_free_gb`.
    """

    def __init__(self, log_dir: str, total_episodes: int, selected: Iterable[int],
                 run_facts: dict, module_facts: dict, config: dict, max_steps: int,
                 min_free_gb: float = MIN_FREE_GB):
        self.folder = Path(log_dir)
        self.selected = frozenset(selected)
        last = int(total_episodes) - 1
        self.full_episodes = frozenset(i for i in (0, last) if i >= 0)
        self.max_steps = int(max_steps)
        self.min_free_gb = float(min_free_gb)
        self._episode: Optional[_EpisodeBuffer] = None
        self.modules: dict = {}
        self._write_run_files(run_facts, module_facts, config)

    def attach_modules(self, modules: dict) -> None:
        """Name to nn.Module, used for gradient norms and the weights tier."""
        self.modules = dict(modules)

    def wants(self, tier: str) -> bool:
        """True when the current episode records this tier."""
        return self._episode is not None and tier in self._episode.tiers

    def _write_run_files(self, run_facts: dict, module_facts: dict, config: dict) -> None:
        _write_json(self.folder / "session.json", {
            "run": run_facts, "modules": module_facts,
            "recorded_episodes": sorted(self.selected),
            "full_tensor_episodes": sorted(self.full_episodes),
        })
        _write_json(self.folder / "config.json", json_safe(config))
        _write_json(self.folder / "environment.json", environment_facts())
        diff = git_output(["diff", "HEAD"])
        (self.folder / "git_diff.patch").write_text(diff or "", encoding="utf-8")

    def start_episode(self, episode_idx: int) -> None:
        wanted = [tier for tier, chosen in (
            ("maps", episode_idx in self.selected),
            ("full", episode_idx in self.full_episodes),
            ("weights", episode_idx in self.selected)) if chosen]
        skipped = [] if not wanted or self._disk_allows() else wanted
        tiers = ["light"] + [tier for tier in wanted if tier not in skipped]
        folder = self.folder / "episodes" / ("ep_%04d" % episode_idx)
        self._episode = _EpisodeBuffer(episode_idx, tiers, skipped, folder, self.max_steps)

    def _disk_allows(self) -> bool:
        self.folder.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(str(self.folder)).free / 1e9 >= self.min_free_gb

    def record_inputs(self, frame, input_source: dict, raw_bids: dict,
                      interoception: dict) -> None:
        """What the agent received this step, captured before the environment steps."""
        if self._episode is not None:
            self._episode.add_inputs(frame, {
                "input_source": dict(input_source), "raw_bids": _floats(raw_bids),
                "interoception": _floats(interoception)})

    def record_outcome(self, winner: str, ignited: bool, sync_r: float,
                       bound_bids: dict, reward: float, action, env_phase: str) -> None:
        """What happened this step. Pairs with the preceding record_inputs call."""
        if self._episode is not None:
            self._episode.add_outcome({
                "winner": winner, "ignited": bool(ignited), "sync_r": float(sync_r),
                "bound_bids": _floats(bound_bids), "reward": float(reward),
                "action": np.asarray(action, dtype=float).ravel().tolist(),
                "env_phase": env_phase})

    def record_extra(self, scalars: dict, vectors: dict, maps: dict, full: dict) -> None:
        """Internal values of the step just completed (see session_capture)."""
        if self._episode is not None:
            self._episode.add_extra(scalars, vectors, maps, full)

    def end_episode(self, modules: Optional[dict] = None) -> None:
        """Write the episode. `modules` defaults to the attached modules."""
        if self._episode is not None:
            self._episode.write(self.modules if modules is None else modules)
        self._episode = None


class _EpisodeBuffer:
    """One episode's recorded values, written by `write`."""

    def __init__(self, index: int, tiers: list, skipped: list, folder: Path, max_steps: int):
        self.index, self.tiers, self.skipped = index, tiers, skipped
        self.folder, self.max_steps = folder, max_steps
        self.frames, self.steps, self.pending = [], [], None
        self.pending_started = 0.0
        self.vectors, self.maps, self.full, self.full_rows = {}, {}, {}, 0

    def add_inputs(self, frame, fields: dict) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
        self.pending = dict(fields, step=len(self.frames) - 1)
        self.pending_started = time.perf_counter()

    def add_outcome(self, fields: dict) -> None:
        if self.pending is None:
            return
        self.pending.update(fields)
        self.steps.append(self.pending)
        self.pending = None

    def add_extra(self, scalars: dict, vectors: dict, maps: dict, full: dict) -> None:
        if not self.steps:
            return
        self.steps[-1]["internals"] = scalars
        # Wall time, for performance only: it varies between identical runs.
        self.steps[-1]["seconds_inputs_to_end"] = time.perf_counter() - self.pending_started
        step = len(self.steps) - 1
        _append_rows(self.vectors, vectors, step)
        if "maps" in self.tiers:
            _append_rows(self.maps, maps, step)
        if "full" in self.tiers and len(self.steps) <= self.max_steps:
            self._write_full_row(full, len(self.steps) - 1)

    def _write_full_row(self, full: dict, row: int) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, array in full.items():
            if name not in self.full:
                self.full[name] = np.lib.format.open_memmap(
                    str(self.folder / ("full_%s.npy" % name)), mode="w+",
                    dtype=np.float16, shape=(self.max_steps,) + array.shape)
            self.full[name][row] = array
        self.full_rows = row + 1

    def write(self, modules: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.frames:
            np.savez_compressed(self.folder / "frames.npz", frames=np.stack(self.frames))
        lines = "".join(json.dumps(step) + "\n" for step in self.steps)
        (self.folder / "steps.jsonl").write_text(lines, encoding="utf-8")
        _save_rows(self.folder / "vectors.npz", self.vectors)
        if "maps" in self.tiers:
            _save_rows(self.folder / "maps.npz", self.maps)
        for memmap in self.full.values():
            memmap.flush()
        self.full.clear()
        if "weights" in self.tiers:
            torch.save(_state_dicts(modules), str(self.folder / "weights.pt"))
        _write_json(self.folder / "meta.json", self._meta())

    def _meta(self) -> dict:
        meta = {"episode": self.index, "steps": len(self.steps), "frames": len(self.frames),
                "complete": len(self.steps) == len(self.frames),
                "tiers": [tier for tier in TIER_ORDER if tier in self.tiers],
                "skipped_tiers": [tier for tier in TIER_ORDER if tier in self.skipped]}
        if "full" in self.tiers:
            meta["full_rows"] = self.full_rows
        return meta


def _append_rows(store: dict, arrays: dict, step: int) -> None:
    for name, array in arrays.items():
        store.setdefault(name, []).append((step, np.asarray(array)))


def _save_rows(path: Path, store: dict) -> None:
    """Stack each value's rows and save the step number of every row as
    name__steps, because a value can be missing on some steps (no audio on step
    0). A value whose shape changed within the episode is saved row by row under
    name__step<N>, never silently dropped."""
    packed = {}
    for name, rows in store.items():
        steps = [step for step, _ in rows]
        arrays = [array for _, array in rows]
        if len({array.shape for array in arrays}) == 1:
            packed[name] = np.stack(arrays)
            packed[name + "__steps"] = np.asarray(steps, dtype=np.int32)
        else:
            packed.update({"%s__step%d" % (name, s): a for s, a in rows})
    np.savez_compressed(path, **packed)


def _state_dicts(modules: dict) -> dict:
    return {name: {key: tensor.detach().cpu() for key, tensor in module.state_dict().items()}
            for name, module in modules.items()}


def environment_facts() -> dict:
    """Software and hardware versions, read at run start."""
    facts = {"python": sys.version.split()[0], "platform": platform.platform(),
             "torch": torch.__version__, "numpy": np.__version__,
             "cuda_available": torch.cuda.is_available(),
             "cuda_version": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for package in ("brian2", "pyphi", "gymnasium", "scikit-learn"):
        try:
            facts[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            facts[package] = None
    return facts


def _write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def _floats(mapping: dict) -> dict:
    return {key: float(number) for key, number in mapping.items()}
