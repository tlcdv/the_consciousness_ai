"""
Does a gate-level PCI carry the stimulus, or only the fact that something happened?

This is the clause that stands between PCI and TRUSTED. `pci_probe_seed_null_2026_09.md`
established that the gate on `gate3_s42` produces a replicated causal response, 16 of 20
probe seeds at 6.87 to 24.42 times its own baseline. A response that repeats is not yet a
response that MEANS anything. Nobody has asked whether it depends on what the system was
looking at.

DESIGN. Each probe seed gives one DMTS trial with one `sample_shape`, and the impulse at
step 40 lands in the DELAY phase on every seed without exception (verified over 200 seeds
before this probe was written). So the phase is constant and only the held shape varies.
That structurally removes the trap that cost the earlier content probes a re-run: step
index alone decodes sample-versus-delay at 1.0000, so any result that could be a phase
reading is worthless. Here there is no phase variation to read.

The label is `sample_shape`, 6 classes, recovered by stepping the environment exactly as
`probe_pci` steps it (action 0 throughout, reset to seed+1 on termination).

RULE 8 IS APPLIED. Only probe seeds in the quiet baseline group are admitted. The gate
baseline is bimodal with a 42x to 60x empty gap, and a reading from the noisy group
cannot register at all, so including those seeds would put a pile of structural zeros
into the test. The gap is found in the data and the admitted and rejected counts are
printed.

TWO QUANTITIES ARE TESTED, and the second is the stronger test:

  pci               what the instrument reports. Coarsely quantized: LZ complexity on a
                    5-channel by 60-step matrix took only the values 2, 4, 5 and 6, so
                    only four PCI values exist. A weak test, reported because it is what
                    would be cited.
  max_abs_response  the raw causal response, continuous, 3.6x spread across seeds. If the
                    gate carries the stimulus at all, it shows here first.

CONTROLS, both free because they come from the same runs:

  rssm              the control site. If the rssm carries content and the gate does not,
                    that separates the site from the measure.
  broadcast         exploratory.

PRE-STATED GATE, written before any value from this run was read.

    (a) CARRIES CONTENT
        eta-squared for 6-class `sample_shape` exceeds its own permutation null p95, on
        `pci` or on `max_abs_response`, at the gate.

    (b) DEGENERATE
        fewer than 3 distinct values among the admitted readings, so there is nothing for
        a content test to explain.

    (c) VARYING BUT CONTENTLESS
        clears (b) and does NOT clear its permutation null p95.

The non-degeneracy bar used elsewhere in this project (100 distinct values, modal share
under 1 percent) does NOT apply here and is not used. It was written for 8000-step logged
series. This probe has one independent reading per seed, so (b) above replaces it.

Shuffling is across SEEDS, because one seed is one trial and the shape is constant within
it. The effective sample is SEEDS, and the count is printed.

Read-only. Consumes a CSV already produced by `probe_pci`.

Run:
    python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 120 \\
      --magnitude 1000 --load-tectum runs/gate3_s42/tectum.pt \\
      --latent-mode continuous --capsule-workspace-source all_levels \\
      --var-floor 1e-6 --out runs/_pci_content/c120_s42.csv
    python -m scripts.analysis.probe_pci_content --csv runs/_pci_content/c120_s42.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.probe_bid_reduction_candidates import eta_squared  # noqa: E402
from simulations.environments.dmts_env import DMTSEnv  # noqa: E402

N_PERMUTATIONS = 2000
MIN_DISTINCT_READINGS = 3


def label_for_seed(seed: int, perturb_step: int) -> tuple[str, str]:
    """The (phase, sample_shape) the probe sees at `perturb_step`.

    Steps the environment exactly as probe_pci._rollout does: action 0 every step,
    reset to seed+1 on termination.
    """
    env = DMTSEnv(num_trials=20)
    _, info = env.reset(seed=seed)
    for _ in range(perturb_step):
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            _, info = env.reset(seed=seed + 1)
    return info.get("phase"), info.get("sample_shape")


# A site is only split when its baselines are genuinely bimodal. Without this the
# largest gap in a tight unimodal distribution is still the largest gap, and the
# split throws away most of the readings for no reason: on a 20-seed run it admitted
# 0 of 20 at the rssm, whose baseline sd is the same 2.443e-02 on every seed. Rule 8
# describes a 42x to 60x separation, so anything under 5x is not that phenomenon.
MIN_GAP_FOR_SPLIT = 5.0


def quiet_split(sds: np.ndarray) -> tuple[float, int, float]:
    """Rule 8. Returns (threshold, n_below, gap). An infinite threshold admits all."""
    s = np.sort(sds[sds > 0])
    if s.size < 3:
        return float("inf"), int(sds.size), 1.0
    gaps = s[1:] / s[:-1]
    k = int(np.argmax(gaps))
    gap = float(gaps[k])
    if gap < MIN_GAP_FOR_SPLIT:
        return float("inf"), int(sds.size), gap
    return float(np.sqrt(s[k] * s[k + 1])), k + 1, gap


def null_p95(x: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    """Permutation null, shuffling the shape label across seeds."""
    rng = np.random.default_rng(seed)
    vals = np.array([eta_squared(x, rng.permutation(groups))
                     for _ in range(N_PERMUTATIONS)])
    return float(np.percentile(vals, 95)), float(vals.mean())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="probe_pci output CSV")
    p.add_argument("--perturb-step", type=int, default=40)
    p.add_argument("--null-seed", type=int, default=20260913)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        sys.exit("empty CSV")

    bad = [r for r in rows if r.get("weights") != "trained"]
    if bad:
        sys.exit("REFUSING: %d rows are not from trained weights" % len(bad))

    seeds = sorted({int(r["seed"]) for r in rows})
    print("Readings: %d probe seeds, %s" % (len(seeds), args.csv))

    labels = {s: label_for_seed(s, args.perturb_step) for s in seeds}
    phases = {ph for ph, _ in labels.values()}
    print("Phase at the impulse step: %s" % sorted(phases))
    if len(phases) > 1:
        print("  WARNING: phase VARIES across seeds. Any result here could be a phase")
        print("  reading, which step index alone decodes at 1.0000. Do not cite it.")
    else:
        print("  Constant, so no result here can be a phase reading.")

    shapes = {}
    for s in seeds:
        shapes[s] = labels[s][1]
    counts = {}
    for v in shapes.values():
        counts[v] = counts.get(v, 0) + 1
    print("Shape classes: %s" % dict(sorted(counts.items())))

    for site in ("gate", "rssm", "broadcast"):
        sel = {int(r["seed"]): r for r in rows if r["read_site"] == site}
        if not sel:
            continue
        sds = np.array([float(sel[s]["median_baseline_sd"]) for s in seeds])
        thresh, n_below, gap = quiet_split(sds)
        admitted = [s for s in seeds if float(sel[s]["median_baseline_sd"]) < thresh]
        rejected = [s for s in seeds if s not in admitted]

        print("\n" + "=" * 76)
        print("SITE: %s" % site)
        print("=" * 76)
        if thresh == float("inf"):
            print("  Rule 8 does NOT apply: largest baseline gap is %.1fx, under the %.0fx"
                  " bar. All %d seeds admitted." % (gap, MIN_GAP_FOR_SPLIT, len(admitted)))
        else:
            print("  Rule 8 split at baseline sd %.4e (gap %.1fx): %d admitted, %d rejected"
                  % (thresh, gap, len(admitted), len(rejected)))
        if len(admitted) < 12:
            print("  TOO FEW ADMITTED SEEDS for a content test. Reporting nothing here.")
            continue

        g = np.array([shapes[s] for s in admitted])
        ac = {}
        for v in g:
            ac[v] = ac.get(v, 0) + 1
        print("  admitted shape classes: %s" % dict(sorted(ac.items())))

        for field in ("pci", "max_abs_response"):
            x = np.array([float(sel[s][field]) for s in admitted])
            distinct = len(np.unique(x))
            e = eta_squared(x, g)
            p95, mean = null_p95(x, g, args.null_seed)
            verdict = ("DEGENERATE (b)" if distinct < MIN_DISTINCT_READINGS
                       else "CARRIES CONTENT (a)" if e > p95
                       else "VARYING BUT CONTENTLESS (c)")
            print("\n  %-17s n=%d  distinct=%d  min=%.4e  max=%.4e"
                  % (field, len(x), distinct, x.min(), x.max()))
            print("  %-17s eta2=%.6f   null p95=%.6f   null mean=%.6f   (%d perms)"
                  % ("", e, p95, mean, N_PERMUTATIONS))
            print("  %-17s VERDICT: %s" % ("", verdict))

    print("\nThe pre-stated gate is in this module's docstring. Read it before citing.")


if __name__ == "__main__":
    main()
