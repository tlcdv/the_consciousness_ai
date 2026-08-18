"""
Is the workspace level alive and badly discretized, or genuinely near-constant?

This is the fork planning #11 calls the deepest question on the map, applied to the
workspace level, and it is the question planning #6 asks directly.

WHAT THE "WORKSPACE LEVEL" ACTUALLY IS. `train_rlhf.py:1895` builds
`ws_state = (broadcast_mag, phi, sync_r)`, a 3-tuple of scalars. It is NOT the
256-dimensional broadcast. `metrics_logger.py` then does two things to it:

    ws_flat     = sum(ws_state)                        # three scalars added together
    ws_discrete = discretize_continuous(ws_flat, 8)    # clips to [0,1], floor(x * 7)

So three quantities on unrelated scales are summed, and the sum is clipped into [0,1].
`broadcast_mag` is `broadcast.norm()`, the L2 norm of a 256-D vector, which has no reason
to lie in [0,1]. `phi` was measured at 1.2e-03 mean earlier today. `sync_r` is a Kuramoto
order parameter in [0,1].

That is the same shape of defect as the vision bid: incommensurable quantities summed,
then a saturating transform. Whether it is the ACTUAL cause here is what this measures.

PRE-STATED GATE, written before any value from these runs was read:

    (a) ALIVE, BADLY DISCRETIZED
        every component has more than 100 distinct raw values, AND the joint discretized
        state count is under 8, at all 3 seeds. The binning is blind and is repairable.

    (b) GENUINELY NEAR-CONSTANT
        any component has 10 or fewer distinct raw values, or a standard deviation below
        1e-6. No binning rescues that, and under #11 that is the answer that says stop
        repairing instruments.

    MIXED
        components disagree. Report per component and do not force one label.

The clipping diagnostic below is reported SEPARATELY and is not part of the gate. It was
derived from reading `discretize_continuous`, not from these numbers, so folding it into
the gate after the fact would be deciding the test on what it found.

Read-only. Uses metrics.csv from runs already on disk. No training, no model loaded.

Run:
    python -m scripts.analysis.probe_workspace_state_variance
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.evaluation.effective_information import discretize_continuous

RUNS = ("bidlog_s42", "bidlog_s43", "bidlog_s44",
        "memfix_s42", "memfix_s43", "memfix_s44")
COMPONENTS = ("broadcast_mag", "phi", "sync_r")
NUM_WORKSPACE_STATES = 8          # the pipeline default
MIN_DISTINCT_ALIVE = 100          # pre-stated
MAX_DISTINCT_DEAD = 10            # pre-stated
MIN_STD_DEAD = 1e-6               # pre-stated


def load(run: str) -> dict | None:
    path = Path("runs") / run / "metrics.csv"
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {c: np.array([float(r[c]) for r in rows], dtype=np.float64)
            for c in COMPONENTS}


def main() -> None:
    data = {r: load(r) for r in RUNS}
    data = {r: d for r, d in data.items() if d is not None}
    if not data:
        print("No runs found under runs/. Nothing to measure.")
        sys.exit(1)

    print("=" * 84)
    print("RAW COMPONENT VARIANCE, before any discretization")
    print("=" * 84)
    print(f"  Pre-stated: alive needs >{MIN_DISTINCT_ALIVE} distinct per component;")
    print(f"  dead is <={MAX_DISTINCT_DEAD} distinct or std<{MIN_STD_DEAD:g}.\n")
    per_run = {}
    for run, d in data.items():
        print(f"  {run}  n={len(d['phi'])}")
        flags = {}
        for c in COMPONENTS:
            a = d[c]
            dist = len(np.unique(a))
            alive = dist > MIN_DISTINCT_ALIVE
            dead = dist <= MAX_DISTINCT_DEAD or a.std() < MIN_STD_DEAD
            flags[c] = (alive, dead)
            print(f"    {c:<14} distinct={dist:>5}  mean={a.mean():.6e}  "
                  f"std={a.std():.6e}  min={a.min():.6e}  max={a.max():.6e}")
        per_run[run] = flags

    print("\n" + "=" * 84)
    print("WHAT THE PIPELINE ACTUALLY DISCRETIZES")
    print("=" * 84)
    print("  ws_flat = broadcast_mag + phi + sync_r, then discretize_continuous(., 8)\n")
    for run, d in data.items():
        flat = d["broadcast_mag"] + d["phi"] + d["sync_r"]
        disc = discretize_continuous(flat, NUM_WORKSPACE_STATES)
        states = len(np.unique(disc))
        print(f"  {run}  ws_flat distinct={len(np.unique(flat)):>5}  "
              f"min={flat.min():.4f} max={flat.max():.4f}  "
              f"-> discretized states={states} of {NUM_WORKSPACE_STATES}  "
              f"bins used={sorted(set(disc.tolist()))}")

    print("\n" + "=" * 84)
    print("CLIPPING DIAGNOSTIC (reported, NOT part of the gate)")
    print("=" * 84)
    print("  discretize_continuous clips to [0,1] before binning. broadcast_mag is an")
    print("  L2 norm over 256 dimensions and has no reason to lie in that range.\n")
    for run, d in data.items():
        flat = d["broadcast_mag"] + d["phi"] + d["sync_r"]
        above = float((flat > 1.0).mean())
        below = float((flat < 0.0).mean())
        print(f"  {run}  ws_flat above 1.0: {above:6.1%}   below 0.0: {below:6.1%}   "
              f"survives unclipped: {1.0 - above - below:6.1%}")

    print("\n" + "=" * 84)
    print("PRE-STATED GATE")
    print("=" * 84)
    verdicts = {}
    for c in COMPONENTS:
        all_alive = all(per_run[r][c][0] for r in per_run)
        any_dead = any(per_run[r][c][1] for r in per_run)
        verdicts[c] = "ALIVE" if all_alive and not any_dead else (
            "DEAD" if any_dead else "UNCLEAR")
        print(f"  {c:<14} {verdicts[c]}")

    labels = set(verdicts.values())
    print()
    if labels == {"ALIVE"}:
        print("  VERDICT (a): ALIVE, BADLY DISCRETIZED at every component.")
        print("  The raw signals vary. Whatever the discretized state count shows is a")
        print("  property of the binning, not of the dynamics.")
    elif "DEAD" in labels and len(labels) == 1:
        print("  VERDICT (b): GENUINELY NEAR-CONSTANT.")
        print("  No binning rescues this. Per #11 the honest move is to stop repairing")
        print("  instruments at this level and say so.")
    else:
        print(f"  VERDICT: MIXED {verdicts}.")
        print("  Components disagree, so no single label is applied. Read per component.")


if __name__ == "__main__":
    main()
