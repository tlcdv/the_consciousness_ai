"""
Does `emergent_complexity` stay cardinality-invariant once the TPM is ESTIMATED?

The 2026-08-01 cardinality scan (docs/results/ce2_state_space_scaling_2026_08.md)
killed `ce2_ratio`: on CONSTRUCTED block TPMs, `causal_emergence` swings from 0.857
at n=8 to 0.996 at n=243 with the macro structure held fixed, so the headline value
reads cardinality rather than structure. One channel survived that scan.
`emergent_complexity` held at exactly k-1 across every n tested, because it counts
singular values above the non-trivial mean and that count is a property of the macro
structure.

That result was on matrices built by hand. The training loop does not build matrices
by hand. It calls `compute_ce2_from_trajectories`, which calls `_build_tpm`, which
counts observed transitions and applies Laplace smoothing. This probe asks whether the
invariance survives that estimator.

WHY IT MIGHT NOT. Laplace smoothing adds one pseudo-count to every unseen transition.
On a large state space with sparse coverage, most of the matrix is smoothing rather
than data. That could manufacture singular values above gamma_star, or wash out real
ones, and the size of the effect scales with n because the number of unseen cells
scales with n^2. That would break the invariance exactly where it is needed, since the
two levels the project compares differ in n by a factor of 30.

PRE-STATED GATE (written before any number from this probe was read, per the
convention in docs/results/gate_binning_2026_07.md and the 2026-08-01 scan):

    PASS, invariant:  at fixed k and fixed transitions-per-state, the modal
                      `emergent_complexity` across seeds is IDENTICAL at n = 8, 32
                      and 243, for every coverage level tested.
    FAIL, confounded: the modal complexity differs across n at fixed k and fixed
                      coverage, so the surviving channel inherits the same cardinality
                      artifact as the headline value once estimation enters, and CE 2.0
                      has nothing left that is comparable across levels.

    Reported separately, NOT part of the pass/fail gate: whether the recovered value
    equals the constructed-matrix answer k-1. A channel can be cardinality-invariant
    while being biased by estimation, and those are different defects with different
    consequences. Invariance is what `ce2_ratio` needed; accuracy is what a within-level
    reading needs.

Complexity is an integer read off a stochastic sample, so this is a stochastic result
and it is run at 5 seeds. The ranges are published, not the best seed.

CONTROL. Every configuration is also scored on the CONSTRUCTED matrix it was sampled
from, using the same `compute_ce2_from_tpm` the 2026-08-01 scan used. That reproduces
the k-1 result inside this script rather than taking it from the earlier doc, so a
disagreement localises to the estimator rather than to the metric.

Run:
    python -m scripts.analysis.probe_ce2_complexity_estimation
"""
from __future__ import annotations

import numpy as np

from models.evaluation.causal_emergence_svd import (
    compute_ce2_from_tpm,
    compute_ce2_from_trajectories,
)

# The cardinalities the training loop actually scores: workspace 8, gate 243.
# 32 sits between them so a trend can be seen rather than only its endpoints.
STATE_COUNTS = [8, 32, 243]

# Macro structures. k blocks means k-1 expected complexity on a constructed matrix.
BLOCK_COUNTS = [2, 3, 4]

# Transitions sampled per state. 1 is the sparse regime where Laplace smoothing
# dominates; 100 is dense enough that it should not. The logger's own EI window is
# 10000 steps, which at n=243 is about 41 transitions per state, so the middle level
# is the one closest to production.
TRANSITIONS_PER_STATE = [1, 10, 100]

SEEDS = [42, 43, 44, 45, 46]


def uniform_block_tpm(n_states: int, n_blocks: int) -> np.ndarray:
    """k uniform blocks over n states, row-stochastic. Blocks may be uneven.

    Same generator as scripts/analysis/probe_ce2_state_space_scaling.py, duplicated
    deliberately: that probe is a finished, cited artifact and importing across probes
    would couple this result to any later edit of that one.
    """
    if n_blocks < 1 or n_states < n_blocks:
        raise ValueError(f"need 1 <= n_blocks <= n_states, got {n_blocks}, {n_states}")
    tpm = np.zeros((n_states, n_states), dtype=np.float64)
    for block in np.array_split(np.arange(n_states), n_blocks):
        tpm[np.ix_(block, block)] = 1.0 / len(block)
    return tpm


def leaky_block_tpm(n_states: int, n_blocks: int, leak: float) -> np.ndarray:
    """Block structure with a uniform leak, so the chain is IRREDUCIBLE.

    The pure block family is reducible: a walk starting inside one block never leaves
    it, so coverage is capped by block size no matter how long the trajectory runs.
    That confounds "the estimator inflates complexity" with "the walk could not reach
    most states". The leaky variant can reach every state, so running both separates
    the two.
    """
    uniform = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)
    return (1.0 - leak) * uniform_block_tpm(n_states, n_blocks) + leak * uniform


def sample_trajectory(tpm: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Markov walk over a row-stochastic TPM, returned as integer state indices."""
    n = tpm.shape[0]
    states = np.empty(length, dtype=np.int64)
    current = int(rng.integers(n))
    for step in range(length):
        states[step] = current
        current = int(rng.choice(n, p=tpm[current]))
    return states


def scan() -> list[dict]:
    """Score every (k, n, coverage, seed) on the estimated matrix and its source."""
    rows = []
    for n_blocks in BLOCK_COUNTS:
        for n_states in STATE_COUNTS:
            if n_states < n_blocks:
                continue
            source = uniform_block_tpm(n_states, n_blocks)
            constructed = compute_ce2_from_tpm(source)

            for per_state in TRANSITIONS_PER_STATE:
                length = per_state * n_states
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    traj = sample_trajectory(source, length, rng)
                    estimated = compute_ce2_from_trajectories([traj], n_states)
                    rows.append({
                        "k": n_blocks,
                        "n": n_states,
                        "per_state": per_state,
                        "seed": seed,
                        "distinct_visited": int(len(np.unique(traj))),
                        "complexity_constructed": constructed.emergent_complexity,
                        "complexity_estimated": estimated.emergent_complexity,
                        "ce_constructed": constructed.causal_emergence,
                        "ce_estimated": estimated.causal_emergence,
                        "gamma_star_estimated": estimated.gamma_star,
                    })
    return rows


def _modal(values: list[int]) -> int:
    """Most common value; ties resolve to the smallest, which is the conservative read."""
    return min(sorted(set(values), key=lambda v: (-values.count(v), v))[:1])


def main() -> None:
    rows = scan()

    print("=" * 78)
    print("CONTROL: constructed matrices must reproduce complexity == k-1")
    print("=" * 78)
    control_ok = True
    for n_blocks in BLOCK_COUNTS:
        for n_states in STATE_COUNTS:
            if n_states < n_blocks:
                continue
            got = next(r["complexity_constructed"] for r in rows
                       if r["k"] == n_blocks and r["n"] == n_states)
            expected = n_blocks - 1
            flag = "ok" if got == expected else "MISMATCH"
            if got != expected:
                control_ok = False
            print(f"  k={n_blocks} n={n_states:>3}  complexity={got}  expected={expected}  {flag}")
    if not control_ok:
        raise AssertionError(
            "CONTROL FAILED: constructed-matrix complexity does not equal k-1, so this "
            "probe cannot attribute anything to the estimator. Fix before reading on."
        )
    print("  control holds: the constructed-matrix result is reproduced here.\n")

    print("=" * 78)
    print("ESTIMATED matrices: modal complexity across 5 seeds (range in brackets)")
    print("=" * 78)
    verdict_rows = []
    for n_blocks in BLOCK_COUNTS:
        print(f"\n  k = {n_blocks}   (constructed answer: {n_blocks - 1})")
        header = "    per_state | " + " | ".join(f"n={n:>3}" for n in STATE_COUNTS)
        print(header)
        for per_state in TRANSITIONS_PER_STATE:
            cells, modes = [], []
            for n_states in STATE_COUNTS:
                if n_states < n_blocks:
                    cells.append("  -  ")
                    continue
                vals = [r["complexity_estimated"] for r in rows
                        if r["k"] == n_blocks and r["n"] == n_states
                        and r["per_state"] == per_state]
                mode = _modal(vals)
                modes.append(mode)
                cells.append(f"{mode:>2} [{min(vals)}-{max(vals)}]")
            print(f"    {per_state:>9} | " + " | ".join(cells))
            verdict_rows.append((n_blocks, per_state, modes))

    print("\n" + "=" * 78)
    print("PRE-STATED GATE")
    print("=" * 78)
    invariant = True
    for n_blocks, per_state, modes in verdict_rows:
        same = len(set(modes)) == 1
        if not same:
            invariant = False
            print(f"  FAIL  k={n_blocks} per_state={per_state}: complexity varies across n {modes}")
    if invariant:
        print("  PASS: modal complexity is IDENTICAL across n at every (k, coverage).")
        print("        emergent_complexity is cardinality-invariant under estimation.")
    else:
        print("\n  FAILED: emergent_complexity inherits the cardinality artifact once the")
        print("  matrix is estimated. No CE 2.0 channel is comparable across levels.")

    print("\n  Reported separately, not part of the gate: accuracy against k-1.")
    for n_blocks, per_state, modes in verdict_rows:
        exact = all(m == n_blocks - 1 for m in modes)
        if not exact:
            print(f"    k={n_blocks} per_state={per_state}: modal {modes}, "
                  f"constructed answer {n_blocks - 1}")

    print("\n" + "=" * 78)
    print("MECHANISM: is complexity counting macro structure, or visited states?")
    print("=" * 78)
    print("  Reported per k, because coverage depends on block size and averaging")
    print("  across k would hide that.\n")
    print("    k   n  per_state | visited/n  | complexity | gamma_star")
    for n_blocks in BLOCK_COUNTS:
        for n_states in STATE_COUNTS:
            for per_state in TRANSITIONS_PER_STATE:
                sel = [r for r in rows if r["k"] == n_blocks and r["n"] == n_states
                       and r["per_state"] == per_state]
                if not sel:
                    continue
                vis = int(np.mean([r["distinct_visited"] for r in sel]))
                cx = _modal([r["complexity_estimated"] for r in sel])
                gs = float(np.mean([r["gamma_star_estimated"] for r in sel]))
                print(f"    {n_blocks}  {n_states:>3}  {per_state:>9} | "
                      f"{vis:>4}/{n_states:<4} | {cx:>10} | {gs:.6f}")

    matched = [r for r in rows if abs(r["complexity_estimated"] - r["distinct_visited"]) <= 2]
    print(f"\n  complexity within 2 of the visited-state count: "
          f"{len(matched)}/{len(rows)} configurations")

    print("\n" + "=" * 78)
    print("IRREDUCIBLE CONTROL: same question on a chain that CAN reach every state")
    print("=" * 78)
    print("  The pure block family is reducible, so a walk is trapped in one block and")
    print("  coverage is capped by structure rather than by sample size. If complexity")
    print("  is still n-dependent here, partial coverage is not the whole explanation.\n")
    print("    k   n  per_state | visited/n  | complexity | expected k-1")
    for n_blocks in BLOCK_COUNTS:
        for n_states in STATE_COUNTS:
            source = leaky_block_tpm(n_states, n_blocks, leak=0.2)
            for per_state in TRANSITIONS_PER_STATE:
                cx, vis = [], []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    traj = sample_trajectory(source, per_state * n_states, rng)
                    cx.append(compute_ce2_from_trajectories(
                        [traj], n_states).emergent_complexity)
                    vis.append(len(np.unique(traj)))
                print(f"    {n_blocks}  {n_states:>3}  {per_state:>9} | "
                      f"{int(np.mean(vis)):>4}/{n_states:<4} | {_modal(cx):>10} | "
                      f"{n_blocks - 1}")


if __name__ == "__main__":
    main()
