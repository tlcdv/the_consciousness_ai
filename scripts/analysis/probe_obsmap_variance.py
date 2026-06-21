"""Is stimulus identity a LOW-VARIANCE direction in obs_map?

This tests the leading hypothesis from the FAILED R1 reconstruction experiments
(collapse_locus_wmrecon_2026_06_21.md, collapse_locus_wmobs_2026_06_21.md): a
reconstruction objective on the RSSM latent trained its MSE down (15x for pixels, ~1300x
for obs_map) but never made the latent decode identity. The proposed mechanism:
reconstruction MSE is dominated by the high-variance, stimulus-INDEPENDENT structure of
the target, while the identity-discriminating direction is low-variance, so an
MSE-minimizing code (like a low-rank compression, or the discrete RSSM bottleneck)
discards identity without paying much MSE.

Direct test: PCA the obs_map across stimuli (one record per trial, leakage-free), then
decode stimulus shape/color from the TOP-k principal components (the highest-variance
subspace, which is what an MSE-optimal rank-k reconstruction keeps). If identity decodes
at chance from the top-k PCs that already capture most of the variance, and only appears
when many low-variance PCs are included, the hypothesis holds: MSE reconstruction keeps
the variance but loses the identity.

This is a read-only analysis (no training). obs_map's covariance structure is
encoder-determined and decodes identity at ~1.0 in every arm, so the result characterizes
what any MSE reconstruction of obs_map faces.

Run:
  python -m scripts.analysis.probe_obsmap_variance --seed 42 \
      --load-tectum runs/wmobs_trained/tectum.pt
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scripts.analysis.probe_collapse_locus import collect
from scripts.analysis.probe_perception_decodability import linear_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-tectum", type=str, default=None)
    args = ap.parse_args()

    taps, ys = collect(args.episodes, args.seed, args.load_tectum)
    X = np.asarray(taps["obs_map"], dtype=np.float32)  # [n, 16384]
    n = X.shape[0]
    mode = f"TRAINED ({args.load_tectum})" if args.load_tectum else "UNTRAINED init"
    print(f"obs_map variance/identity probe  mode={mode}  n={n} trials\n")

    # Standardize then PCA (the same preprocessing the decodability probe uses).
    Xs = StandardScaler().fit_transform(X)
    n_pc = min(200, n - 1, X.shape[1])
    pca = PCA(n_components=n_pc, random_state=args.seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = np.cumsum(pca.explained_variance_ratio_)

    print(f"{'top-k PCs':>10} {'cum.var':>8} {'shape_acc':>10} {'color_acc':>10}  "
          f"(chance ~ 0.167)")
    for k in [2, 5, 10, 20, 50, 100, n_pc]:
        if k > n_pc:
            continue
        sub = scores[:, :k]
        sh = linear_decode(sub, ys["shape"], seed=args.seed)["test_acc"]
        co = linear_decode(sub, ys["color"], seed=args.seed)["test_acc"]
        print(f"{k:>10d} {evr[k - 1]:>8.3f} {sh:>10.3f} {co:>10.3f}")

    print()
    print("READING: if identity decodes at chance from the top-k PCs that already capture")
    print("most of the variance, and only rises with many low-variance PCs, then identity")
    print("is a low-variance direction that an MSE reconstruction keeps the variance but")
    print("loses, which explains why the R1 reconstruction objectives FAILED.")


if __name__ == "__main__":
    main()
