"""
Does `phi` carry stimulus identity? Does `broadcast_mag`?

THIS IS THE TEST ON THE LAST INSTRUMENT STANDING. The project record names `phi` as the
one signal that ever moved under an intervention. EI is deprecated, CE 2.0 carries a
degeneracy confound, `is_conscious` is pinned at 1, PCI reads exactly 0.0 at the gate site
(`docs/results/pci_trained_2026_08.md`), sync_R reports the bids
(`docs/results/sync_r_content_2026_09.md`), and the vision bid is saturated. Nobody has
ever asked whether `phi` carries content.

`phi` is also the only quantity measured so far that passes the strict non-degeneracy bar
on its own: 1129 to 1267 distinct values with the modal value holding 0.2 to 0.4 percent
of steps, against sync_R's 85 percent.

`broadcast_mag` is tested in the same pass because it is the third component of
`ws_state = (broadcast_mag, phi, sync_r)` and it costs nothing extra. It is
`broadcast.norm()`, and the 256-D broadcast it summarizes decodes shape at 0.69 to 0.77
(`docs/results/broadcast_geometry_2026_08.md`). So this asks directly whether taking the
norm discards that content.

A COVERAGE TRAP THIS PROBE IS BUILT AGAINST. `phi_method` is "skipped" at 6400 of 8000
steps and "pyphi" at 1599. The logged `phi` at a skipped step is the previous computed
value carried forward. A content test over all steps would therefore measure a
step-function hold, which tracks the clock rather than the stimulus. The gated test uses
COMPUTED steps only. The all-steps number is printed beside it, marked as not gated.

PRE-STATED GATE, written before any value from these runs was read. Same three outcomes
as the sync_R probe, because the standing fork's two answers were not enough.

    (a) ALIVE AND CARRYING CONTENT
        eta-squared for 6-class `sample_shape` exceeds its own permutation null p95 at all
        3 seeds. The quantity carries the stimulus and any instrument that misses it is at
        fault.

    (b) GENUINELY NEAR-CONSTANT
        fewer than 100 distinct values, or standard deviation below 1e-6, at any seed.

    (c) VARYING BUT CONTENTLESS
        clears the non-degeneracy bar and does NOT clear its permutation null.

Traps carried over from the earlier probes, each of which cost a re-run:

  - `sample_shape` is the label, never phase, because step index alone decodes
    sample-vs-delay at 1.0000 and any phase result is a clock reading
  - shuffling is done ACROSS TRIALS, because shape is constant within a trial
  - the effective sample is TRIALS, not steps, and the trial count is printed

Read-only. No model loaded, no training. Uses runs already on disk.

Run:
    python -m scripts.analysis.probe_scalar_content
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_bid_reduction_candidates import (
    MAX_PINNED_FRACTION,
    MIN_DISTINCT,
    N_PERMUTATIONS,
    eta_squared,
    permutation_null,
)

SEEDS = (42, 43, 44)
DEAD_MAX_DISTINCT = 100
DEAD_MAX_STD = 1e-6


def load(seed: int) -> dict:
    run_dir = Path("runs") / f"bcast_s{seed}"
    rows = list(csv.DictReader(open(run_dir / "metrics.csv", newline="")))
    if "env_sample_shape" not in rows[0]:
        raise SystemExit(f"{run_dir}/metrics.csv has no labels. Stale run directory.")

    trial = [int(r["env_trial"]) for r in rows]
    episode, episodes = 0, []
    for i, v in enumerate(trial):
        if i and v < trial[i - 1]:
            episode += 1
        episodes.append(episode)

    return {
        "phi": np.array([float(r["phi"]) for r in rows]),
        "broadcast_mag": np.array([float(r["broadcast_mag"]) for r in rows]),
        "computed": np.array([r["phi_method"] == "pyphi" for r in rows]),
        "shape": np.array([r["env_sample_shape"] for r in rows]),
        "phase": np.array([r["env_phase"] for r in rows]),
        "trial": np.array([f"{e}:{t}" for e, t in zip(episodes, trial)]),
    }


def describe(name: str, x: np.ndarray) -> bool:
    values, counts = np.unique(x, return_counts=True)
    distinct = len(values)
    pinned = int(counts.max()) / len(x)
    std = float(x.std())
    dead = distinct < DEAD_MAX_DISTINCT or std < DEAD_MAX_STD
    strict = distinct >= MIN_DISTINCT and pinned <= MAX_PINNED_FRACTION
    print(f"    {name:<16} n={len(x):<6} distinct={distinct:<6} std={std:.3e} "
          f"modal share={pinned:.2%}  "
          f"strict={'PASS' if strict else 'FAIL'}  {'DEAD' if dead else 'not dead'}")
    return not dead


def content_test(x: np.ndarray, shapes: np.ndarray, trials: np.ndarray,
                 seed: int, label: str, gated: bool) -> bool:
    observed = eta_squared(x, shapes)
    null = permutation_null(x, shapes, trials, seed)
    p95 = float(np.percentile(null, 95))
    ok = observed > p95
    tag = "" if gated else "   [NOT GATED]"
    print(f"    {label:<28} n={len(x):<5} trials={len(np.unique(trials)):<4} "
          f"eta2={observed:.6f}  null mean={null.mean():.6f} p95={p95:.6f}  "
          f"{'PASS' if ok else 'FAIL'}{tag}")
    return ok


def main() -> None:
    data = {s: load(s) for s in SEEDS}

    print("=" * 88)
    print("1. NON-DEGENERACY")
    print("=" * 88)
    print(f"  Strict bar carried from probe_bid_reduction_candidates.py: >= {MIN_DISTINCT}")
    print(f"  distinct and <= {MAX_PINNED_FRACTION:.0%} at the modal value.")
    print(f"  DEAD bar: < {DEAD_MAX_DISTINCT} distinct or std < {DEAD_MAX_STD}.")
    print("")
    alive = {"phi": {}, "broadcast_mag": {}}
    for seed, d in data.items():
        print(f"  seed {seed}")
        alive["phi"][seed] = describe("phi (computed)", d["phi"][d["computed"]])
        describe("phi (all steps)", d["phi"])
        alive["broadcast_mag"][seed] = describe("broadcast_mag", d["broadcast_mag"])
        print(f"    phi computed at {int(d['computed'].sum())} of {len(d['phi'])} steps "
              f"({d['computed'].mean():.1%}); the rest carry the last value forward.")

    print("")
    print("=" * 88)
    print("2. CONTENT: DOES THE SCALAR TRACK WHICH SHAPE WAS SHOWN?")
    print("=" * 88)
    print("  sample-phase steps, 6-class sample_shape, eta-squared against a null of")
    print(f"  {N_PERMUTATIONS} shuffles ACROSS TRIALS.")
    print("")
    passed = {"phi": {}, "broadcast_mag": {}}
    for seed, d in data.items():
        sample = d["phase"] == "sample"
        print(f"  seed {seed}")

        gate_mask = sample & d["computed"]
        passed["phi"][seed] = content_test(
            d["phi"][gate_mask], d["shape"][gate_mask], d["trial"][gate_mask],
            seed, "phi, computed steps", gated=True)
        content_test(
            d["phi"][sample], d["shape"][sample], d["trial"][sample],
            seed, "phi, all sample steps", gated=False)
        passed["broadcast_mag"][seed] = content_test(
            d["broadcast_mag"][sample], d["shape"][sample], d["trial"][sample],
            seed, "broadcast_mag", gated=True)

    print("")
    print("=" * 88)
    print("PRE-STATED GATE")
    print("=" * 88)
    for name in ("phi", "broadcast_mag"):
        if not all(alive[name].values()):
            verdict = "(b) GENUINELY NEAR-CONSTANT"
        elif all(passed[name].values()):
            verdict = "(a) ALIVE AND CARRYING CONTENT, at all 3 seeds"
        elif not any(passed[name].values()):
            verdict = "(c) VARYING BUT CONTENTLESS, at all 3 seeds"
        else:
            verdict = f"SEEDS DISAGREE {passed[name]}. Not decisive; claim nothing."
        print(f"  {name:<16} {verdict}")


if __name__ == "__main__":
    main()
