# emergent_complexity FAILED under estimation: it counts sampling noise, not macro structure

**Result: FAILED against its pre-stated gate, and worse than the gate was written to
detect.** `emergent_complexity` is not cardinality-invariant once the transition matrix
is estimated from sampled trajectories. It is not measuring macro structure at all. On a
243-state chain with 2 blocks and **every state visited**, it reports 112 where the
constructed matrix reports 1.

This closes the last channel that survived the 2026-08-01 cardinality scan. No CE 2.0
channel is comparable across levels. Deterministic generators, 5 seeds, no training run,
no checkpoint. No indicator moves.

## Why this was checked

[`ce2_state_space_scaling_2026_08.md`](ce2_state_space_scaling_2026_08.md) killed
`ce2_ratio`: on constructed block matrices `causal_emergence` swings from 0.857143 at
n=8 to 0.995868 at n=243 with the macro structure held fixed. One channel survived that
scan. `emergent_complexity` held at exactly k-1 at every n, because it counts singular
values above the non-trivial mean and that count reads the macro structure.

That scan built its matrices by hand. The training loop does not. It calls
`compute_ce2_from_trajectories`, which calls `_build_tpm`, which counts observed
transitions and applies Laplace smoothing. This probe asks whether the invariance
survives the estimator the logger actually uses.

## Pre-stated gate

Written in `scripts/analysis/probe_ce2_complexity_estimation.py` before any number from
it was read.

> **PASS, invariant:** at fixed k and fixed transitions-per-state, the modal
> `emergent_complexity` across seeds is IDENTICAL at n = 8, 32 and 243, for every
> coverage level tested.
> **FAIL, confounded:** the modal complexity differs across n at fixed k and fixed
> coverage.

Accuracy against k-1 was pre-declared as reported-but-not-gated, because a channel can
be cardinality-invariant while being biased, and those are different defects.

## Result: FAILED, all nine configurations

Modal `emergent_complexity` across 5 seeds. The constructed-matrix answer is k-1.

| k | per_state | n=8 | n=32 | n=243 | constructed |
|---|-----------|-----|------|-------|-------------|
| 2 | 1   | 3 | 11 | 79 | 1 |
| 2 | 10  | 3 | 11 | 87 | 1 |
| 2 | 100 | 2 | 10 | 87 | 1 |
| 3 | 1   | 3 | 8  | 60 | 2 |
| 3 | 10  | 1 | 9  | 64 | 2 |
| 3 | 100 | 1 | 7  | 65 | 2 |
| 4 | 1   | 2 | 7  | 49 | 3 |
| 4 | 10  | 2 | 6  | 51 | 3 |
| 4 | 100 | 1 | 5  | 50 | 3 |

Nine of nine fail. The value grows with n at every fixed k and coverage.

**Control, verified inside the same script:** the constructed matrices reproduce
complexity = k-1 at every n tested. The script RAISES rather than prints if that fails,
so the comparison cannot be made against an unverified baseline.

## Partial coverage is NOT the explanation

The pure block family is reducible: a walk starting inside one block never leaves it, so
coverage is capped by block size however long the trajectory runs. That confounds "the
estimator inflates complexity" with "the walk could not reach most states".

The irreducible control separates them. A leaky variant (20 percent uniform mixing) can
reach every state, and its **constructed** matrices still give complexity = k-1 exactly
at every n, verified before use.

| k | n | per_state | visited | complexity | expected |
|---|---|-----------|---------|------------|----------|
| 2 | 243 | 10  | **243/243** | **112** | 1 |
| 2 | 243 | 100 | **243/243** | **111** | 1 |
| 3 | 243 | 10  | **243/243** | **111** | 2 |
| 3 | 243 | 100 | **243/243** | **108** | 2 |
| 2 | 32  | 100 | 32/32 | 11 | 1 |
| 3 | 32  | 100 | 32/32 | 9  | 2 |
| 2 | 8   | 100 | 8/8   | 1  | 1 |
| 3 | 8   | 100 | 8/8   | 2  | 2 |

At full coverage the inflation is unchanged. So the defect is the estimator, not the
sampling reach.

Note the last two rows: at n=8 with dense coverage the channel is correct. The failure is
a function of state-space size, which is precisely the property that made this channel
look worth keeping.

## Mechanism

`gamma_star` is the MEAN of the non-trivial singular values, and a significant scale is
one exceeding it. An estimated matrix carries one small singular value per direction of
sampling noise, and there are n-1 of them. Adding many near-zero values drags the mean
down, so far more values clear it.

Measured `gamma_star` on estimated matrices, averaged over seeds:

| n | k=2 | k=3 | k=4 |
|---|-----|-----|-----|
| 8   | 0.080898 | 0.089665 | 0.096758 |
| 32  | 0.036156 | 0.029880 | 0.028219 |
| 243 | 0.014223 | 0.010369 | 0.008300 |

It collapses by roughly an order of magnitude from n=8 to n=243, at every k. The
threshold that defines "significant" is itself a function of state count once the matrix
is estimated, so the count it produces cannot be compared across state counts.

This is a property of the definition meeting a finite sample. It is not a coding defect,
and the local implementation still matches the source on every checked point.

## What this settles

- **`emergent_complexity` is retired for cross-level use.** It fails the same
  comparability clause that killed `ce2_ratio`, by a different route.
- **No CE 2.0 channel is comparable across levels.** The headline value was confounded by
  cardinality on constructed matrices; the surviving channel is confounded by cardinality
  once estimated. Both roads are closed.
- **Within-level use is now also in question,** which this probe did not set out to test.
  Complexity at fixed n is stable across seeds, but its value tracks the state count
  rather than the k it is supposed to recover, so a within-level trend would need its own
  justification rather than inheriting one from this channel's earlier reputation.

## What this does NOT show

- Nothing here touches a trained checkpoint or a training run. It is a property of the
  metric and the estimator, measured on synthetic generators with known answers.
- It does not distinguish Laplace smoothing from ordinary finite-sample noise as the
  source of the extra singular values. Both add small values and both scale with n, and
  separating them was not needed to answer the question asked.
- It says nothing about whether an INTERVENTIONAL matrix would behave differently. That
  is the open estimator question and it is tracked separately.

## Reproduce

```
python -m scripts.analysis.probe_ce2_complexity_estimation
```

Deterministic given the seeds in the script (42 to 46). Runtime about two minutes.
