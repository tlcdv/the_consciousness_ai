"""Does the GNW ignition gate signal carry ANY task-phase structure? (Track A2)

The 2026-07 signature assessment found ignition saturated (99.8 to 100 percent of steps
flagged conscious): `is_conscious == input_energy >= EMA(input_energy)` and the input
energy never dips below its own average once stable. The open diagnostic question is
whether the gate SIGNAL (input_energy = max bound bid) carries task-event structure at
all. If sample onset does not move it relative to the delay, no thresholding scheme on
this signal can make ignition task-selective, and the honest verdict is that the gate
needs different CONTENT, not a different threshold.

Forward-only probe, no training. Runs DMTS episodes through the standard pipeline
(reusing probe_perception_decodability's builders), records per step:
  - the DMTS phase (fixation / sample / delay / choice),
  - input_energy, recovered EXACTLY from consecutive EMA baselines
    (baseline_t = a*baseline_{t-1} + (1-a)*energy_t, a = 0.95, so
     energy_t = (baseline_t - a*baseline_{t-1}) / (1-a)),
  - salience_t = energy_t - baseline_t (what the sigmoid sees),
  - is_conscious, broadcast magnitude, vision bid.
Then prints per-phase stats and the sample-vs-delay contrast (the discriminating
structure ignition would need).

Usage:
    python -m scripts.analysis.probe_ignition_signal [--episodes 3] [--seed 42]
        [--load-tectum runs/latentid_trained/tectum.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_perception_decodability import (
    _build_components,
    _compute_broadcast,
)
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv

PHASES = ("fixation", "sample", "delay", "choice")


def collect(episodes: int, seed: int, load_tectum: str | None) -> dict:
    config, tectum, workspace, reentrant, self_model, memory, mock_sem = \
        _build_components("dmts", action_dim=5, seed=seed, mock_semantic=False,
                          load_tectum=load_tectum)
    env = DMTSEnv(num_trials=20)
    alpha = getattr(workspace, "baseline_alpha", 0.95)

    rows = {k: [] for k in ("phase", "energy", "salience", "is_conscious",
                            "broadcast_mag", "vision_bid", "sample_onset")}
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            if hasattr(tectum, "reset_state"):
                tectum.reset_state(1)
            done = False
            steps = 0
            prev_baseline = None
            prev_phase = None
            while not done and steps < 4000:
                phase = info.get("phase")
                frame = frame_to_tensor(obs, config["device"])
                audio = torch.zeros(1, config["tectum_feature_dim"], 2,
                                    device=config["device"])
                tectum_content, vision_bid = tectum(frame, audio)
                bc = _compute_broadcast(config, tectum, workspace, reentrant,
                                        self_model, memory, mock_sem,
                                        tectum_content, vision_bid, obs)
                baseline = workspace._energy_baseline
                if baseline is None:
                    prev_baseline, prev_phase = baseline, phase
                    obs, _, term, trunc, info = env.step(0)
                    done = term or trunc
                    steps += 1
                    continue
                if prev_baseline is None:
                    energy = baseline  # first step: baseline initialized to energy
                else:
                    energy = (baseline - alpha * prev_baseline) / (1.0 - alpha)
                rows["phase"].append(phase)
                rows["energy"].append(float(energy))
                rows["salience"].append(float(energy - baseline))
                rows["is_conscious"].append(int(workspace.state.is_conscious))
                rows["broadcast_mag"].append(
                    float(np.linalg.norm(bc)) if bc is not None else 0.0)
                rows["vision_bid"].append(float(vision_bid))
                rows["sample_onset"].append(
                    int(phase == "sample" and prev_phase != "sample"))
                prev_baseline, prev_phase = baseline, phase
                obs, _, term, trunc, info = env.step(0)
                done = term or trunc
                steps += 1
    return rows


def report(rows: dict, label: str) -> None:
    phase = np.asarray(rows["phase"])
    energy = np.asarray(rows["energy"])
    salience = np.asarray(rows["salience"])
    ic = np.asarray(rows["is_conscious"])
    bmag = np.asarray(rows["broadcast_mag"])

    print(f"\n===== {label} (n={len(energy)} steps) =====")
    print(f"{'phase':10s} {'n':>6s} {'energy mean':>12s} {'std':>10s} "
          f"{'salience>0':>11s} {'ignited':>8s} {'bmag mean':>10s}")
    for ph in PHASES:
        m = phase == ph
        if not m.any():
            continue
        print(f"{ph:10s} {int(m.sum()):>6d} {energy[m].mean():>12.6f} "
              f"{energy[m].std():>10.2e} {float((salience[m] > 0).mean()):>11.3f} "
              f"{float(ic[m].mean()):>8.3f} {bmag[m].mean():>10.4f}")

    # The contrast ignition would need: sample vs delay
    ms, md = phase == "sample", phase == "delay"
    if ms.any() and md.any():
        pooled = np.sqrt(0.5 * (energy[ms].var() + energy[md].var()))
        d = (energy[ms].mean() - energy[md].mean()) / pooled if pooled > 0 else 0.0
        print(f"sample-vs-delay energy effect size d = {d:+.3f} "
              f"(pooled std {pooled:.2e})")
        pooled_b = np.sqrt(0.5 * (bmag[ms].var() + bmag[md].var()))
        db = (bmag[ms].mean() - bmag[md].mean()) / pooled_b if pooled_b > 0 else 0.0
        print(f"sample-vs-delay broadcast_mag effect size d = {db:+.3f}")
    onset = np.asarray(rows["sample_onset"], dtype=bool)
    if onset.any():
        print(f"sample-ONSET steps: n={int(onset.sum())} "
              f"salience mean={salience[onset].mean():+.6f} "
              f">0 fraction={float((salience[onset] > 0).mean()):.3f} "
              f"(vs all-step salience mean {salience.mean():+.6f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-tectum", type=str, default=None)
    args = ap.parse_args()

    label = f"trained ({args.load_tectum})" if args.load_tectum else "untrained init"
    rows = collect(args.episodes, args.seed, args.load_tectum)
    report(rows, label)


if __name__ == "__main__":
    main()
