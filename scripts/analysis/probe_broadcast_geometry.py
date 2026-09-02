"""
Is the 256-D broadcast alive, or does it point one way for a whole run?

The broadcast is the conscious content in this architecture. `workspace_state` keeps only
its LENGTH (`broadcast.norm()`), so its DIRECTION has never been recorded during a run. An
indirect estimate on 2026-08-17 inverted the memory bid formula and put the cosine between
the current broadcast and its best recent match at 0.99999976 to 1.0
(`docs/results/memory_retrieval_repair_2026_08.md`). That is a lead from a bid formula,
not a measurement of the broadcast.

This is the fork planning #11 raises, applied to the broadcast: are the dynamics genuinely
near-constant, or do they vary while the instruments reading them are blind?

A METHODOLOGICAL POINT, stated because it changes what can be claimed. The plan for this
probe proposed comparing the decodability of the RAW broadcast against the CENTERED
broadcast, expecting centering to reveal a signal hidden under a large constant offset.
That comparison is structurally impossible here: `grouped_decode` fits
`StandardScaler` then `LogisticRegression`, and the scaler already centers every feature
on the training split. A linear classifier on standardized features is invariant to a
constant offset, so raw and centered give the same number by construction.

So the two questions are separated:

  - DECODE answers whether the broadcast carries stimulus content at all. That is the
    (a) / (b) fork and it is what the gate reads.
  - GEOMETRY answers why an instrument reading raw magnitude or raw distance would see
    nothing even if the content is there. It is descriptive and is not gated.

PRE-STATED GATE, written before any number from these runs was read:

    (a) ALIVE BUT MIS-READ
        the broadcast decodes `sample_shape` above its own shuffled floor by at least
        0.10 at all 3 seeds. The content is present, and any instrument that reads only
        the broadcast's length or its raw pairwise distance is blind to it.

    (b) GENUINELY NEAR-CONSTANT
        it does not clear that margin at any seed. The dynamics really are still, and no
        instrument repair rescues this level. That is the answer which says stop
        repairing instruments and say so.

Traps this probe is built against, each of which cost a re-run earlier:

  - the split holds out whole TRIALS chosen at random, never a random split over steps,
    because consecutive steps are near-duplicates and the shuffled control cannot detect
    that leak
  - `sample_shape` is the label, never phase, because step index alone decodes
    sample-vs-delay at 1.0000 and any phase result is a clock reading
  - shuffling is done ACROSS TRIALS, because shape is constant within a trial
  - the effective sample is TRIALS, not steps

Read-only. Uses runs already on disk.

Run:
    python -m scripts.analysis.probe_broadcast_geometry
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_klmap_phase_information import grouped_decode

SEEDS = (42, 43, 44)
MARGIN = 0.10                 # pre-stated
PCA_VARIANCE_TARGET = 0.95


def load(seed: int):
    d = Path("runs") / f"bcast_s{seed}"
    rows = list(csv.DictReader(open(d / "metrics.csv", newline="")))
    if "env_sample_shape" not in rows[0]:
        raise SystemExit(f"{d}/metrics.csv has no labels. Stale run directory.")
    B = np.load(d / "broadcast.npy").astype(np.float64)
    if B.shape[0] != len(rows):
        raise SystemExit(f"{d}: broadcast.npy has {B.shape[0]} rows, csv has {len(rows)}")
    trial = [int(r["env_trial"]) for r in rows]
    # env_trial is per-EPISODE and repeats (0..3). A unique trial id needs the episode,
    # which is recoverable from where the counter resets.
    ep, eps = 0, []
    for i, v in enumerate(trial):
        if i and v < trial[i - 1]:
            ep += 1
        eps.append(ep)
    return {
        "B": B,
        "shape": np.array([r["env_sample_shape"] for r in rows]),
        "phase": np.array([r["env_phase"] for r in rows]),
        "trial": np.array([f"{e}:{t}" for e, t in zip(eps, trial)]),
    }


def main() -> None:
    data = {s: load(s) for s in SEEDS}

    print("=" * 84)
    print("GEOMETRY: how much of the broadcast is a fixed vector?")
    print("=" * 84)
    print("  Descriptive, NOT gated. Explains what a magnitude or distance reading sees.")
    print("")
    for s, d in data.items():
        B = d["B"]
        mu = B.mean(axis=0)
        C = B - mu
        mu_norm = float(np.linalg.norm(mu))
        c_rms = float(np.sqrt((C ** 2).sum(axis=1).mean()))
        num = (B[:-1] * B[1:]).sum(axis=1)
        den = np.linalg.norm(B[:-1], axis=1) * np.linalg.norm(B[1:], axis=1) + 1e-12
        step_cos = float(np.mean(num / den))
        sv = np.linalg.svd(C, compute_uv=False)
        var = sv ** 2
        k95 = int(np.searchsorted(np.cumsum(var) / var.sum(), PCA_VARIANCE_TARGET) + 1)
        dim_distinct = [len(np.unique(B[:, j])) for j in range(B.shape[1])]
        print(f"  seed {s}  n={len(B)}")
        print(f"    norm of mean vector        = {mu_norm:.6f}")
        print(f"    RMS of centered remainder  = {c_rms:.6f}")
        print(f"    varying over fixed ratio   = {c_rms / (mu_norm + 1e-12):.6f}")
        print(f"    mean consecutive cosine    = {step_cos:.9f}")
        print(f"    PCA dims for 95pc variance = {k95} of {B.shape[1]}")
        print(f"    per-dim distinct values    : min={min(dim_distinct)} "
              f"median={int(np.median(dim_distinct))} max={max(dim_distinct)}")

    print("")
    print("=" * 84)
    print("DECODE: does the broadcast carry stimulus identity?")
    print("=" * 84)
    print("  sample-phase steps, trial-grouped split, 6-class sample_shape.")
    print("")
    verdicts = []
    for s, d in data.items():
        m = d["phase"] == "sample"
        X, y, g = d["B"][m], d["shape"][m], d["trial"][m]
        rng = np.random.default_rng(s)
        uniq = np.unique(g)
        lab = {t: y[g == t][0] for t in uniq}
        perm = dict(zip(uniq, rng.permutation([lab[t] for t in uniq])))
        y_shuf = np.array([perm[t] for t in g])

        real = grouped_decode(X, y, g, s)
        shuf = grouped_decode(X, y_shuf, g, s)
        maj = real["majority"]
        margin = real["test_acc"] - shuf["test_acc"]
        ok = margin >= MARGIN and (real["test_acc"] - maj) >= MARGIN
        verdicts.append(ok)
        print(f"  seed {s}  n={int(m.sum())} steps, {len(uniq)} trials, "
              f"held-out={real['n_test']}  majority={maj:.4f}")
        print(f"    broadcast  acc={real['test_acc']:.4f}   over majority="
              f"{real['test_acc'] - maj:+.4f}")
        print(f"    SHUFFLED   acc={shuf['test_acc']:.4f}   margin over shuffle="
              f"{margin:+.4f}   {'PASS' if ok else 'FAIL'}")

    print("")
    print("=" * 84)
    print("PRE-STATED GATE")
    print("=" * 84)
    if all(verdicts):
        print("  VERDICT (a): ALIVE BUT MIS-READ, at all 3 seeds.")
        print("  The broadcast carries stimulus identity. Any instrument reading only")
        print("  its length, or raw pairwise distance, is blind to content that is there.")
    elif not any(verdicts):
        print("  VERDICT (b): GENUINELY NEAR-CONSTANT, at all 3 seeds.")
        print("  The broadcast does not carry stimulus identity above its own shuffled")
        print("  floor. The honest move is to stop repairing instruments at this level.")
    else:
        print(f"  VERDICT: seeds DISAGREE {verdicts}. Not decisive; report as such.")


if __name__ == "__main__":
    main()
