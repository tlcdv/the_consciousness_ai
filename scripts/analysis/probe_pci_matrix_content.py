"""
Does the response MATRIX carry the stimulus that the PCI scalar throws away?

`pci_content_2026_09.md` established that a gate-level PCI does not carry the
stimulus: eta-squared 0.060960 against a permutation null MEAN of 0.061989, with the
rssm control failing the same test. PCI compresses a 5-channel by 60-step binary
matrix into one number. This probe asks whether the matrix it compresses carries the
shape.

The question matters because this project already has the same shape of finding at
seven other sites: every VECTOR tested carries the stimulus and every SCALAR tested
does not. If the response matrix decodes the shape and the PCI scalar does not, PCI
becomes the eighth case of one mechanism rather than an eighth separate failure. If
the matrix does not decode either, the gate response genuinely carries nothing and
the scalar reduction is not the culprit.

THREE THINGS ARE TESTED, at the gate and at the rssm control:

  response      the CONTINUOUS causal response matrix, perturbed minus clean, over
                the response window. This is what exists before any thresholding.
  binary        the same matrix after `binarize_response`, which is exactly what the
                LZ count is taken over. The step PCI actually compresses.
  scalar        `pci` itself, re-derived here, as a consistency check against
                `pci_content_2026_09.md`.

PRE-STATED GATE, written before any value from this run was read.

    (a) THE MATRIX CARRIES IT, THE SCALAR DISCARDS IT
        cross-validated accuracy on `response` at the gate exceeds its own label
        permutation null p95, while the scalar does not clear its null. This is the
        project's existing vector-versus-scalar finding, at an eighth site.

    (b) NOTHING CARRIES IT
        neither `response` nor `binary` clears its null. The gate response is
        contentless as a matrix, and the scalar reduction is not what loses it.

    (c) BINARIZATION IS THE LOSS
        `response` clears its null and `binary` does not.

METHOD. `StandardScaler` then `LogisticRegression`, matching
`probe_perception_decodability._decode`. That helper uses a single train/test split,
which at n=81 leaves a test set of about 16 and is too noisy to read. Here the same
classifier is scored by stratified k-fold cross-validation instead, and the null is a
label permutation scored the same way. The change is stated rather than silent.

TRAPS THIS PROBE IS BUILT AGAINST, each of which has cost this project a re-run:

  - The impulse lands in the DELAY phase on every seed, verified over 200 seeds. The
    phase does not vary, so no result here can be a phase reading. This is checked
    and printed, not assumed.
  - Rule 8 is applied at the gate: only quiet-baseline probe seeds are admitted,
    because a reading from the noisy group cannot register at all.
  - The label is `sample_shape`, constant within a trial, and one probe seed is one
    trial. Permutation is across seeds.
  - Chance is printed as BOTH uniform (1/6) and the majority-class rate, because the
    classes are not perfectly balanced and the majority rate is the honest floor.

Read-only. No training. Two rollouts per probe seed, about 12 seconds each.

Run:
    python -m scripts.analysis.probe_pci_matrix_content --trials 120 \\
      --load-tectum runs/gate3_s42/tectum.pt --latent-mode continuous \\
      --capsule-workspace-source all_levels --var-floor 1e-6 \\
      --npz runs/_pci_content/mat120_s42.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.evaluation.perturbational_complexity import (  # noqa: E402
    binarize_response,
    compute_pci,
)
from scripts.analysis.probe_pci import _rollout  # noqa: E402
from scripts.analysis.probe_pci_content import label_for_seed, quiet_split  # noqa: E402

N_PERMUTATIONS = 200
SITES = ("gate", "rssm")


def collect(args):
    """One clean and one perturbed rollout per probe seed."""
    out = {s: {"resp": [], "binary": [], "pci": [], "base_sd": []} for s in SITES}
    seeds, shapes, phases = [], [], []
    n_steps = args.perturb_step + args.response_window

    for i in range(args.trials):
        seed = args.seed + i
        action_dim = 5 if args.env == "dmts" else 2
        rng = np.random.default_rng(seed)
        actions = np.where(rng.random(n_steps) < 0.0,
                           rng.integers(0, action_dim, size=n_steps), 0)
        common = dict(
            env_name=args.env, seed=seed, n_steps=n_steps, actions=actions,
            perturb_site="rssm", magnitude=args.magnitude, impulse_seed=seed + 7919,
            load_tectum=args.load_tectum, latent_mode=args.latent_mode,
            capsule_source=args.capsule_workspace_source,
        )
        rssm_c, gate_c, _ = _rollout(perturb_step=None, **common)
        rssm_p, gate_p, _ = _rollout(perturb_step=args.perturb_step, **common)

        for site, clean, pert in (("gate", gate_c, gate_p), ("rssm", rssm_c, rssm_p)):
            pre = clean[:, : args.perturb_step]
            div = float(np.abs(pert[:, : args.perturb_step] - pre).max())
            if div != 0.0:
                sys.exit("REFUSING: pre-impulse divergence %.3e at seed %d" % (div, seed))
            response = pert[:, args.perturb_step:] - clean[:, args.perturb_step:]
            binary = binarize_response(response, pre, threshold_sigma=3.0,
                                       var_floor=args.var_floor)
            res = compute_pci(response, pre, threshold_sigma=3.0,
                              var_floor=args.var_floor)
            out[site]["resp"].append(response)
            out[site]["binary"].append(binary.astype(np.float64))
            out[site]["pci"].append(res.pci)
            out[site]["base_sd"].append(float(np.median(pre.std(axis=1, ddof=1))))

        ph, sh = label_for_seed(seed, args.perturb_step)
        seeds.append(seed)
        shapes.append(sh)
        phases.append(ph)
        if (i + 1) % 10 == 0:
            print("  %d / %d seeds" % (i + 1, args.trials), flush=True)

    packed = {"seeds": np.array(seeds), "shapes": np.array(shapes),
              "phases": np.array(phases)}
    for s in SITES:
        packed["%s_resp" % s] = np.array(out[s]["resp"])
        packed["%s_binary" % s] = np.array(out[s]["binary"])
        packed["%s_pci" % s] = np.array(out[s]["pci"])
        packed["%s_base_sd" % s] = np.array(out[s]["base_sd"])
    return packed


def cv_accuracy(X, y, seed):
    """Stratified k-fold accuracy for StandardScaler + LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = max(2, min(5, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, X, y, cv=cv).mean()), n_splits


def test_block(name, X, y, classes, seed):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    keep = X.std(axis=0) > 0
    n_live = int(keep.sum())
    if n_live == 0:
        print("  %-9s ALL %d FEATURES CONSTANT. Nothing to decode." % (name, X.shape[1]))
        return
    X = X[:, keep]
    acc, n_splits = cv_accuracy(X, y, seed)
    rng = np.random.default_rng(seed)
    null = np.array([cv_accuracy(X, rng.permutation(y), seed)[0]
                     for _ in range(N_PERMUTATIONS)])
    p95, mean = float(np.percentile(null, 95)), float(null.mean())
    uniform = 1.0 / len(classes)
    majority = float(np.bincount(y).max() / len(y))
    verdict = "DECODES" if acc > p95 else "does not decode"
    print("  %-9s features %d of %d live   %d-fold CV" % (name, n_live, keep.size, n_splits))
    print("  %-9s acc=%.4f   null p95=%.4f   null mean=%.4f   uniform=%.4f  majority=%.4f"
          % ("", acc, p95, mean, uniform, majority))
    print("  %-9s %s" % ("", verdict))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="dmts", choices=["dmts", "dark_room"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--perturb-step", type=int, default=40)
    p.add_argument("--response-window", type=int, default=60)
    p.add_argument("--magnitude", type=float, default=1000.0)
    p.add_argument("--var-floor", type=float, default=1e-6)
    p.add_argument("--load-tectum", default=None)
    p.add_argument("--latent-mode", default="discrete",
                   choices=["discrete", "continuous"])
    p.add_argument("--capsule-workspace-source", default="final",
                   choices=["final", "all_levels"])
    p.add_argument("--npz", required=True)
    p.add_argument("--reuse", action="store_true",
                   help="Load --npz instead of running the rollouts")
    p.add_argument("--null-seed", type=int, default=20260913)
    args = p.parse_args()

    if args.reuse and os.path.exists(args.npz):
        print("Reusing %s" % args.npz)
        d = dict(np.load(args.npz, allow_pickle=True))
    else:
        print("Collecting %d probe seeds" % args.trials)
        d = collect(args)
        os.makedirs(os.path.dirname(os.path.abspath(args.npz)), exist_ok=True)
        np.savez_compressed(args.npz, **d)
        print("Wrote %s" % args.npz)

    phases = sorted(set(d["phases"].tolist()))
    print("\nPhase at the impulse step: %s" % phases)
    if len(phases) > 1:
        print("  WARNING: phase VARIES. Any result could be a phase reading. Do not cite.")
    else:
        print("  Constant, so no result here can be a phase reading.")

    shapes = d["shapes"]
    classes = sorted(set(shapes.tolist()))
    print("Shape classes: %s" % {c: int((shapes == c).sum()) for c in classes})

    for site in SITES:
        sds = d["%s_base_sd" % site]
        thresh, _, gap = quiet_split(sds)
        mask = sds < thresh
        print("\n" + "=" * 76)
        print("SITE: %s" % site)
        print("=" * 76)
        if thresh == float("inf"):
            print("  Rule 8 does not apply (largest baseline gap %.1fx). All %d admitted."
                  % (gap, int(mask.sum())))
        else:
            print("  Rule 8 split at %.4e (gap %.1fx): %d admitted, %d rejected"
                  % (thresh, gap, int(mask.sum()), int((~mask).sum())))
        y_labels = shapes[mask]
        if len(y_labels) < 30:
            print("  TOO FEW ADMITTED SEEDS. Reporting nothing here.")
            continue
        cls = sorted(set(y_labels.tolist()))
        y = np.array([cls.index(v) for v in y_labels])
        print("  admitted classes: %s" % {c: int((y_labels == c).sum()) for c in cls})
        print()
        test_block("response", d["%s_resp" % site][mask], y, cls, args.null_seed)
        print()
        test_block("binary", d["%s_binary" % site][mask], y, cls, args.null_seed)
        print()
        test_block("scalar", d["%s_pci" % site][mask].reshape(-1, 1), y, cls,
                   args.null_seed)

    print("\nThe pre-stated gate is in this module's docstring. Read it before citing.")


if __name__ == "__main__":
    main()
