"""
Is stimulus content sustained in the tectum and transient in the workspace?

Vishne, Gerber, Knight and Deouell (2023, Cell Reports 42, 112752) showed images
for several durations to patients with intracranial electrodes. Content in the
ventral visual stream stayed decodable, with a stable code, for as long as the
image stayed on. Frontoparietal content appeared as a brief burst at onset and
did not track duration, without any report. The COGITATE collaboration adopted
the paradigm because IIT predicts the first pattern in posterior cortex and GNWT
predicts the second in prefrontal cortex.

The agent's matching question. The DMTS sample is shown for S steps with
S in {5, 10, 20}. At every step from 3 steps before onset to 10 steps after
offset, this probe records the tectum content vector and the workspace
broadcast, then builds a temporal generalization matrix per stage
(models/evaluation/temporal_generalization.py) with sample_shape as the label.

PRE-STATED GATE (written before the first run), per stage, at S = 20:
  SUSTAINED / TRANSIENT / MIXED from classify_dynamics over the sample window,
  with the threshold from label_shuffle_threshold (p95 of the per-permutation
  maximum, family-wise over steps).
  The Vishne pattern is REPLICATED only if the tectum is SUSTAINED and the
  broadcast is TRANSIENT at all 3 seeds. Any other combination at any seed is
  reported as it is, and the pattern is NOT replicated.
  Whether the decodable window grows with S is reported descriptively.

A broadcast that cannot be computed raises. It is never replaced by zeros.

Usage:
    python -m scripts.analysis.probe_sustained_vs_ignition --runs-dir runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.evaluation.temporal_generalization import (  # noqa: E402
    classify_dynamics,
    label_shuffle_threshold,
    temporal_generalization,
)
from scripts.analysis.probe_pci import _seed_everything  # noqa: E402
from scripts.analysis.probe_perception_decodability import (  # noqa: E402
    _build_components,
    _compute_broadcast,
)
from scripts.training.train_rlhf import frame_to_tensor  # noqa: E402
from simulations.environments.dmts_env import DMTSEnv  # noqa: E402

TRAINED = {42: "capfix_alllevels/tectum.pt",
           43: "capfix_seed43/tectum.pt",
           44: "capfix_seed44/tectum.pt"}
DURATIONS = (5, 10, 20)
PRE, POST = 3, 10


def _flat(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64).ravel()


def _step_states(comps, obs) -> tuple[np.ndarray, np.ndarray]:
    config, tectum, workspace, reentrant, self_model, memory, mock_sem = comps
    device = config["device"]
    frame = frame_to_tensor(obs, device)
    audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
    content, bid = tectum(frame, audio)
    broadcast = _compute_broadcast(config, tectum, workspace, reentrant, self_model,
                                   memory, mock_sem, content, bid, obs)
    if broadcast is None:
        raise RuntimeError("broadcast could not be computed; refusing to substitute zeros")
    return _flat(content), np.asarray(broadcast, dtype=np.float64).ravel()


def record_epochs(comps, seed: int, duration: int, trials: int):
    """Epochs [trials, PRE + duration + POST, D] per stage, and sample_shape labels."""
    env = DMTSEnv(num_trials=trials, sample_steps=duration, fixation_steps=PRE + 2,
                  min_delay=POST + 2, max_delay=POST + 2)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    tec, bro, phases, shapes = [], [], [], []
    with torch.no_grad():
        done = False
        while not done:
            t_vec, b_vec = _step_states(comps, obs)
            tec.append(t_vec); bro.append(b_vec)
            phases.append(info["phase"]); shapes.append(info["sample_shape"])
            action = int(rng.integers(1, 3)) if info["phase"] == "choice" else 0
            obs, _, done, _, info = env.step(action)
    return _epoch(np.array(tec), np.array(bro), phases, shapes, duration)


def _epoch(tec, bro, phases, shapes, duration):
    onsets = [i for i in range(1, len(phases))
              if phases[i] == "sample" and phases[i - 1] != "sample"]
    length = PRE + duration + POST
    keep = [o for o in onsets if o - PRE >= 0 and o - PRE + length <= len(phases)]
    t_ep = np.stack([tec[o - PRE:o - PRE + length] for o in keep])
    b_ep = np.stack([bro[o - PRE:o - PRE + length] for o in keep])
    labels = np.array([shapes[o] for o in keep])
    return t_ep, b_ep, labels


def analyse(stage: str, X: np.ndarray, y: np.ndarray, duration: int, seed: int) -> str:
    live = X.reshape(-1, X.shape[-1]).std(axis=0) > 0
    if not live.any():
        print(f"    {stage:<10} CONSTANT, nothing to decode")
        return "CONSTANT"
    X = X[:, :, live]
    acc = temporal_generalization(X, y, n_splits=5, seed=seed)
    threshold = label_shuffle_threshold(X, y, n_splits=5, n_perm=20, seed=seed)
    verdict = classify_dynamics(acc, threshold, onset=PRE, length=duration)
    diag = np.diag(acc)
    window = diag[PRE:PRE + duration]
    print(f"    {stage:<10} {verdict:<9} threshold {threshold:.3f}  "
          f"sample-window diag min {window.min():.3f} max {window.max():.3f}  "
          f"above threshold {int((window > threshold).sum())}/{duration}  "
          f"post-offset above {int((diag[PRE + duration:] > threshold).sum())}/{POST}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()

    replicated = {}
    for seed, rel in TRAINED.items():
        path = str(Path(args.runs_dir) / rel)
        if not Path(path).exists():
            raise SystemExit(f"missing checkpoint {path}")
        _seed_everything(seed)
        comps = _build_components("dmts", action_dim=5, seed=seed, mock_semantic=True,
                                  load_tectum=path, latent_mode="continuous",
                                  capsule_workspace_source="all_levels")
        print(f"seed {seed}  {path}")
        for duration in DURATIONS:
            t_ep, b_ep, y = record_epochs(comps, seed, duration, args.trials)
            print(f"  S = {duration}  trials {len(y)}  classes {len(set(y))}")
            v_t = analyse("tectum", t_ep, y, duration, seed)
            v_b = analyse("broadcast", b_ep, y, duration, seed)
            if duration == max(DURATIONS):
                replicated[seed] = (v_t, v_b)
    print("PRE-STATED GATE")
    ok = all(v == ("SUSTAINED", "TRANSIENT") for v in replicated.values())
    print(f"  per seed (tectum, broadcast): {replicated}")
    print("  REPLICATED at all 3 seeds." if ok else "  FAILED. The Vishne pattern is NOT replicated.")


if __name__ == "__main__":
    main()
