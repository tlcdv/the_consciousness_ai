"""Localize WHERE stimulus identity dies in the tectum forward pipeline.

Prior results: obs_map decodes stimulus identity at ~1.0, but tectum_content (the
256-D content the workspace broadcasts and the policy reads) decodes at chance
(perception_decodability_2026_06_09.md), and a reconstruction objective FAILED to fix
it (tectum_reconstruction_2026_06_10.md, "architectural"). The 2026-06-15 match-head
diagnosis sharpened this: the agent's binding constraint is that the integration
collapse destroys the relational/identity information cognition needs.

This probe pins the LOCUS of that loss along the forward chain
  obs_map -> RSSM z_state -> capsule poses -> tectum_content (256-D)
by decoding the on-screen sample shape/color from each stage, leakage-free (one record
per trial). It answers the strategic fork:

  identity survives in capsule poses, dies only at tectum_content
      -> the loss is the final projection: a NARROW, fixable locus (R3 viable).
  identity already gone at z_state or capsule poses
      -> the RSSM / capsule routing destroys it: a DEEP problem (escalates to R1).

This does not dissolve the design tension (a low-D integrated workspace is partly BY
DESIGN in GNW); it locates the bottleneck so the next decision is informed.

Run:
  python -m scripts.analysis.probe_collapse_locus --episodes 14 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from scripts.analysis.probe_perception_decodability import (
    _build_components, frame_to_tensor, linear_decode,
)
from simulations.environments.dmts_env import DMTSEnv


def pca_mlp_decode(X, y, seed: int = 0, test_size: float = 0.3, npca: int = 80):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    chance = cnt.max() / len(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed, stratify=y)
    k = min(npca, Xtr.shape[0] - 1, Xtr.shape[1])
    clf = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),
                        MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                      random_state=seed))
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte), chance


def collect(episodes: int, seed: int, load_tectum: str | None = None,
            latent_mode: str = "discrete"):
    cfg, tectum, *_ = _build_components("dmts", action_dim=5, seed=seed,
                                        mock_semantic=False, load_tectum=load_tectum,
                                        latent_mode=latent_mode)
    env = DMTSEnv(num_trials=20)
    taps = {"obs_map": [], "z_state": [], "capsule_poses": [], "tectum_content": []}
    ys = {"shape": [], "color": []}
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            tectum.reset_state(1)
            cur = None
            done, steps = False, 0
            while not done and steps < 4000:
                ph = info.get("phase")
                f = frame_to_tensor(obs, cfg["device"])
                wc, _ = tectum(f, torch.zeros(1, cfg["tectum_feature_dim"], 2,
                                              device=cfg["device"]))
                if ph == "sample":
                    cur = {
                        "obs_map": tectum._last_obs_map.reshape(-1).cpu().numpy().astype(np.float32),
                        "z_state": tectum.z_state.reshape(-1).cpu().numpy().astype(np.float32),
                        "capsule_poses": tectum._last_capsule_poses.reshape(-1).cpu().numpy().astype(np.float32),
                        "tectum_content": wc.reshape(-1).cpu().numpy().astype(np.float32),
                        "shape": info["sample_shape"], "color": info["sample_color"],
                    }
                elif ph == "choice" and cur is not None:
                    # one record per trial: no within-trial leakage
                    for t in taps:
                        taps[t].append(cur[t])
                    ys["shape"].append(cur["shape"])
                    ys["color"].append(cur["color"])
                    cur = None
                a = 0 if ph != "choice" else int(rng.integers(1, env.num_choices + 1))
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
                steps += 1
    return taps, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-tectum", type=str, default=None,
                    help="Load a trained tectum state_dict (from train_rlhf "
                         "--save-tectum) before probing, to measure the TRAINED "
                         "RSSM/pipeline instead of the untrained init. Default None "
                         "keeps the original untrained-probe behavior bit-identical.")
    ap.add_argument("--latent-mode", type=str, default="discrete",
                    choices=["discrete", "continuous"],
                    help="RSSM latent mode to BUILD before loading (Path B1). Must "
                         "match the checkpoint: a --rssm-latent-mode continuous run "
                         "must be probed with --latent-mode continuous or the state "
                         "dict will not load (the continuous latent adds cont_logvar).")
    args = ap.parse_args()

    mode = f"TRAINED tectum ({args.load_tectum})" if args.load_tectum else "UNTRAINED init"
    print(f"collapse-locus probe mode: {mode} | latent_mode={args.latent_mode}")
    taps, ys = collect(args.episodes, args.seed, args.load_tectum, args.latent_mode)
    n = len(ys["shape"])
    print(f"LEAKAGE-FREE collapse-locus decode, n={n} trials (one record per trial)")
    print(f"pipeline order: obs_map -> z_state -> capsule_poses -> tectum_content\n")
    order = ["obs_map", "z_state", "capsule_poses", "tectum_content"]
    for tap in order:
        dim = len(taps[tap][0])
        for lab in ["shape", "color"]:
            lin = linear_decode(taps[tap], ys[lab], seed=args.seed)
            mlp_acc, chance = pca_mlp_decode(taps[tap], ys[lab], seed=args.seed)
            print(f"  {tap:16s} dim={dim:6d} {lab:6s} "
                  f"linear={lin['test_acc']:.3f} pca+mlp={mlp_acc:.3f} "
                  f"chance={lin['uniform_chance']:.3f}")
    print()
    print("READING: high at obs_map (control); where it drops to chance is the loss locus.")
    print("  drops only at tectum_content -> narrow final-projection fix (R3 viable)")
    print("  drops at z_state/capsule_poses -> RSSM/capsule routing is the culprit (deep, R1)")


if __name__ == "__main__":
    main()
