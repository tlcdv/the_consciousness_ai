"""Offline decode of captured DMTS choice records: the decisive diagnostic for the
match-head FAILED result (docs/results/dmts_match_head_2026_06_15.md).

Single-sample and batched in-loop match heads both failed at chance behaviorally,
with the batched head plateauing at ~0.70 train accuracy. The offline probe got
0.845 (PCA-80 + MLP, held-out). This script decodes the LIVE training-loop choice
records (captured via `train_rlhf.py --capture-choice-records`) two ways:

  (a) PCA-80 + MLP : the exact protocol that produced 0.845. Tests whether the
      in-loop [obs;mem] records are clean and decodable.
  (b) the conv MatchHead architecture, trained offline with many epochs on a
      held-out split. Tests whether the conv arch can reach 0.845 given proper
      training (unlimited epochs, real validation), or whether it plateaus.

Reading:
  (a) ~0.845 and (b) ~0.845 -> records clean, conv fine -> the ONLINE training
      procedure (LR/batch schedule) was the in-loop bottleneck.
  (a) ~0.845 and (b) ~0.70  -> records clean, the CONV ARCHITECTURE is the
      bottleneck (e.g. AdaptiveAvgPool destroys choice-position info) -> use a
      PCA-bottleneck / MLP head.
  (a) ~chance               -> the in-loop ObsMapSampleMemory LATCH is degraded
      (the [obs;mem] does not determine the match) -> fix the latch, not the head.

Usage:
  python -m scripts.analysis.decode_choice_records --records runs/cap/choice_records.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="path to choice_records.npz")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--pca", type=int, default=80)
    ap.add_argument("--conv-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.records)
    X, y = d["X"], d["y"]
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    chance = counts.max() / n  # majority-class baseline
    print(f"records: n={n}, dim={X.shape[1]}, classes={dict(zip(classes.tolist(), counts.tolist()))}, "
          f"majority/chance={chance:.3f}")
    if n < 60:
        print("WARNING: very few records; decode accuracy will be noisy.")

    # Stratified train/test split.
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_frac, random_state=args.seed, stratify=y)

    # (a) PCA-80 + MLP (the 0.845 protocol), plus a linear baseline for context.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    npca = min(args.pca, Xtr.shape[0] - 1, Xtr.shape[1])
    mlp = make_pipeline(StandardScaler(), PCA(n_components=npca, random_state=args.seed),
                        MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                      random_state=args.seed))
    mlp.fit(Xtr, ytr)
    acc_mlp = mlp.score(Xte, yte)

    lin = make_pipeline(StandardScaler(), PCA(n_components=npca, random_state=args.seed),
                        LogisticRegression(max_iter=2000))
    lin.fit(Xtr, ytr)
    acc_lin = lin.score(Xte, yte)

    print(f"(a) PCA-{npca}+MLP   test acc: {acc_mlp:.3f}   (chance {chance:.3f})")
    print(f"    PCA-{npca}+linear test acc: {acc_lin:.3f}")

    # (b) the conv MatchHead, trained offline with held-out validation.
    import torch
    from models.self_model.match_head import MatchHead
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Records are flattened obsmem-conv policy_state -> reshape [N,128,16,16].
    spatial = (128, 16, 16)
    num_actions = int(classes.max()) + 1
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    yte_t = torch.tensor(yte, dtype=torch.long, device=dev)
    torch.manual_seed(args.seed)
    head = MatchHead(spatial, num_actions=num_actions).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best_test = 0.0
    final_train = 0.0
    ntr = Xtr_t.shape[0]
    for ep in range(args.conv_epochs):
        head.train()
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, 64):
            idx = perm[i:i + 64]
            logits = head(Xtr_t[idx])
            loss = head.loss(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            final_train = (head(Xtr_t).argmax(1) == ytr_t).float().mean().item()
            test = (head(Xte_t).argmax(1) == yte_t).float().mean().item()
        best_test = max(best_test, test)
    print(f"(b) conv MatchHead offline: final train acc {final_train:.3f}, best test acc {best_test:.3f}   (chance {chance:.3f})")

    print()
    print("READING:")
    if acc_mlp < chance + 0.1:
        print("  PCA+MLP at chance -> in-loop LATCH is degraded; fix the latch, not the head.")
    elif best_test < acc_mlp - 0.1:
        print("  PCA+MLP decodes but conv head does not -> CONV ARCHITECTURE is the bottleneck.")
    else:
        print("  Both decode -> records clean and conv capable -> ONLINE training procedure was the bottleneck.")


if __name__ == "__main__":
    main()
