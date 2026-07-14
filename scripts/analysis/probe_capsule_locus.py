"""Where inside the capsule hierarchy does stimulus identity die? (Track B diagnosis)

The B1 continuous latent puts identity into z_state (the pre-capsule RSSM latent), but
capsule_poses and tectum_content are still at chance: identity is destroyed inside the
capsule stage (HierarchicalCapsuleComposition). This probe decodes the DMTS sample's
shape/color from each INTERNAL capsule level to localize the loss:

  state_tensor (z_state) -> primary_caps -> routing level 0 -> level 1 -> level 2 (final)

Read-only, forward-only, leakage-free (one record per trial). Requires a trained tectum
saved with a continuous latent (the mode where z_state carries identity), so the input to
the capsule stage is known to be identity-rich and any drop is the capsule stage's doing.

Usage:
    python -m scripts.analysis.probe_capsule_locus --load-tectum runs/b1_continuous/tectum.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from scripts.analysis.probe_perception_decodability import _build_components
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv


def decode(X, y, seed=0):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    chance = cnt.max() / len(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    lin = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    lin.fit(Xtr, ytr)
    lin_acc = lin.score(Xte, yte)
    k = min(80, Xtr.shape[0] - 1, Xtr.shape[1])
    mlp = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),
                        MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=seed))
    mlp.fit(Xtr, ytr)
    return lin_acc, mlp.score(Xte, yte), chance


def collect(episodes, seed, load_tectum, latent_mode):
    cfg, tectum, *_ = _build_components("dmts", action_dim=5, seed=seed,
                                        mock_semantic=False, load_tectum=load_tectum,
                                        latent_mode=latent_mode)
    cap = tectum.capsule_layer
    env = DMTSEnv(num_trials=20)
    taps = {"z_state": [], "primary_caps": []}
    # one entry per routing level, filled after we see the first forward
    level_taps = None
    ys = {"shape": [], "color": []}
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            tectum.reset_state(1)
            done, steps, cur = False, 0, None
            while not done and steps < 4000:
                ph = info.get("phase")
                frame = frame_to_tensor(obs, cfg["device"])
                audio = torch.zeros(1, cfg["tectum_feature_dim"], 2, device=cfg["device"])
                tectum(frame, audio)
                if ph == "sample":
                    state = tectum._last_state_tensor  # [1, C, H, W] = z_state input to capsules
                    primary = cap.primary(state)       # [1, num_primary, primary_dim]
                    levels = cap.get_all_level_poses()  # list of (poses, activities) per routing level
                    cur = {
                        "z_state": state.reshape(-1).cpu().numpy(),
                        "primary_caps": primary.reshape(-1).cpu().numpy(),
                        "levels": [p.reshape(-1).cpu().numpy() for p, _ in levels],
                        "shape": info["sample_shape"], "color": info["sample_color"],
                    }
                    nonlocal_levels = len(levels)
                    if level_taps is None:
                        level_taps = [[] for _ in range(nonlocal_levels)]
                elif ph == "choice" and cur is not None:
                    taps["z_state"].append(cur["z_state"])
                    taps["primary_caps"].append(cur["primary_caps"])
                    for i, lv in enumerate(cur["levels"]):
                        level_taps[i].append(lv)
                    ys["shape"].append(cur["shape"])
                    ys["color"].append(cur["color"])
                    cur = None
                a = 0 if ph != "choice" else int(rng.integers(1, env.num_choices + 1))
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
                steps += 1
    return taps, level_taps, ys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-tectum", type=str, default=None)
    ap.add_argument("--latent-mode", type=str, default="continuous",
                    choices=["discrete", "continuous"])
    args = ap.parse_args()

    print(f"capsule-locus probe: {args.load_tectum} | latent_mode={args.latent_mode}")
    taps, level_taps, ys = collect(args.episodes, args.seed, args.load_tectum, args.latent_mode)
    n = len(ys["shape"])
    print(f"leakage-free, n={n} trials\n")
    print(f"pipeline: z_state -> primary_caps -> routing levels (0..last=final capsule_poses)\n")

    rows = [("z_state", taps["z_state"]), ("primary_caps", taps["primary_caps"])]
    for i, lv in enumerate(level_taps):
        rows.append((f"routing_L{i}", lv))
    for name, X in rows:
        dim = len(X[0])
        for lab in ("shape", "color"):
            lin, mlp, ch = decode(X, ys[lab], seed=args.seed)
            print(f"  {name:14s} dim={dim:>7d} {lab:5s} linear={lin:.3f} pca+mlp={mlp:.3f} chance={ch:.3f}")


if __name__ == "__main__":
    main()
