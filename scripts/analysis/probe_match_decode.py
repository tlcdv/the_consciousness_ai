"""Probe-conditions match decode for DMTS (reproducible companion to the
in-loop capture).

The 2026-06-14 "0.845" match-action decode was an uncommitted ad-hoc measurement.
This script reproduces it under PROBE conditions (untrained components, scripted
policy, no training, no_grad), capturing exactly what the offline localization saw:
the obsmem-conv policy_state `[current obs_map ; held sample mem_slot]` at the first
choice frame of each trial, one record per trial, plus the `target_position` label.

It saves an .npz that is decoded by the SAME protocol as the in-loop records
(scripts/analysis/decode_choice_records.py), so probe-vs-in-loop is apples-to-apples:

  probe ~0.845, in-loop ~chance  -> the TRAINING LOOP degrades the signal
  probe ~chance                  -> the 0.845 was never robust (probe artifact)

Run:
  python -m scripts.analysis.probe_match_decode --episodes 14 --seed 42 \
      --out runs/cap/probe_records.npz
  python -m scripts.analysis.decode_choice_records --records runs/cap/probe_records.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.probe_perception_decodability import _build_components, frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv
from models.self_model.working_memory_latch import ObsMapSampleMemory


def collect(episodes: int, seed: int):
    cfg, tectum, *_ = _build_components("dmts", action_dim=5, seed=seed, mock_semantic=False)
    env = DMTSEnv(num_trials=20)
    mem = ObsMapSampleMemory()
    X, y = [], []
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            tectum.reset_state(1)
            mem.reset()
            done, steps = False, 0
            while not done and steps < 4000:
                ph = info.get("phase")
                f = frame_to_tensor(obs, cfg["device"])
                tectum(f, torch.zeros(1, cfg["tectum_feature_dim"], 2, device=cfg["device"]))
                slot = mem.update(tectum._last_obs_map, f)
                if ph == "choice" and info.get("target_position") is not None:
                    # obsmem-conv policy_state: [current obs_map ; held sample]
                    ps = torch.cat([
                        tectum._last_obs_map.reshape(-1),
                        slot.reshape(-1),
                    ]).cpu().numpy().astype(np.float32)
                    X.append(ps)
                    y.append(int(info["target_position"]))
                a = 0 if ph != "choice" else int(rng.integers(1, env.num_choices + 1))
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
                steps += 1
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/cap/probe_records.npz")
    args = ap.parse_args()
    X, y = collect(args.episodes, args.seed)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, X=X, y=y)
    cls, cnt = np.unique(y, return_counts=True)
    print(f"saved {len(y)} probe-condition records to {args.out} "
          f"(classes={dict(zip(cls.tolist(), cnt.tolist()))})")


if __name__ == "__main__":
    main()
