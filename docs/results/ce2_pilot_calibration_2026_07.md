# CE 2.0 calibration pilot: FAILED (gate and workspace values are frozen-input artifacts)

**Verdict: FAILED.** The CE 2.0 gate and workspace values measured on this pilot are
constant-trajectory artifacts, not signal. They are reproduced exactly by a frozen
single-state trajectory of the same window length. CE 2.0 does NOT escape the
degeneracy that suspended the EI-based claims; the defect is the discretized input,
not the metric.

Single seed (42), dark_room only, 500 episodes. Hypothesis-grade. No indicator moves
and no consciousness claim is made.

## Setup
`python -m scripts.training.train_rlhf --env dark_room --episodes 500 --max-steps 200
--seed 42 --log-ce2-every 10 --log-ei-every 10 --log-dir runs/ce2_pilot_seed42`

Window = 10 episodes x 200 steps = 2000 steps. Raw data: `runs/ce2_pilot_seed42/episodes.csv`
(501 rows = header + 500 episodes, verified from disk).

## The code path did fire (this is not a null run)
50 rows with CE 2.0 computed and 50 with EI computed, as expected at every-10 over
500 episodes. The "no RSSM latent transitions" wiring warning never fired.

## Evidence: the values are frozen-input artifacts
`ce2_gates` took exactly ONE distinct value across all 50 windows; `ce2_workspace`
took two (and was constant from episode 19 onward). Both equal the value a fully
frozen trajectory produces at the same window length:

| Level | Observed in pilot (constant) | Synthetic frozen trajectory, 2000 steps | Match |
|-------|------------------------------|------------------------------------------|-------|
| Gate, 243 states | 0.877884 | 0.877884 | exact |
| Workspace, 8 states | 0.662874 | 0.662874 | exact |

For comparison, EI on the same run was degenerate in the already-documented way:
`ei_gates` had one distinct value (0.027063) and `ei_gates_corr` was 0.0 in all 50
windows.

## Why a frozen input scores HIGH (the confound)
CE 2.0's `sigma_2 - gamma_star` measures the coarse-graining gain still AVAILABLE, and
a maximally degenerate microscale has the most available. So the metric RISES as the
trajectory degenerates. Measured over 2000 steps with 243 states:

| distinct states visited | 1 | 2 | 10 | 50 | 243 |
|---|---|---|---|---|---|
| CE 2.0 | 0.8779 | 0.7978 | 0.4329 | 0.1108 | 0.0148 |

The frozen value is a CEILING, not a floor, and it is length dependent (243 states:
0.287037 at 100 steps, 0.877884 at 2000, 0.948498 at 10000, 0.961161 at 100000).
Two consequences:
1. A higher CE 2.0 does not mean "more emergent"; it can mean "more frozen".
2. Subtracting it as a floor (the EI-style correction) would be wrong: it would clamp
   every real trajectory to zero. That approach was tried and rejected during this work.

Comparing a 243-state gate level against an 8-state workspace level is additionally
confounded by state-space size, the same class of artifact the 2026-07-02 assessment
found in the EI ratio.

## Consequences for the pre-registered CE 2.0 predictions
- **CE2-1 (onset):** 0 of 50 windows had `ce2_workspace > ce2_gates`. NOT falsified.
  The pilot reached only episode 500 and the prediction's window starts at 500, and
  more importantly the comparison is currently untestable because both sides are
  frozen-input artifacts. Threshold unchanged.
- **CE2-2 (magnitude band):** cannot be calibrated. `ce2_ratio` was 0.7551 in the last
  25 windows (min 0.7551, max 0.8148, mean 0.7563), a constant equal to the ratio of
  two artifacts. Fixing a band on this would band an artifact. Stays deferred.
- **CE2-4 (RSSM latent):** untested. `ce2_rssm` is the one channel that varies
  (min 0.391356, max 0.414624, mean 0.402487) with `ce2_complexity_rssm` pinned at 19,
  but no identity decodability was measured in this run to correlate against, and a
  ~6 percent spread on one seed cannot be called signal or noise.

## What this localizes
The blocker is the micro-level discretization: the 5 gate dimensions under fixed
tertile binning never leave one joint state, so the 243-state TPM carries no observed
transition structure. This matches the 2026-07-02 assessment (gate EI bit-identical in
every window, reproduced by a single-state trajectory) and the 2026-07-06 note that
"the gate discretization is the next distinct locus". CE 2.0 inherits it because no
metric can extract structure from a constant trajectory.

## Recommendation
Do NOT run the 3-seed CE 2.0 campaign. It would reproduce these artifacts three times.
The decisive next work is upstream: give the micro level a discretization that actually
transitions (adaptive or quantile binning per window rather than fixed tertiles, or a
gate signal with real variation). Re-run this pilot only after the gate trajectory
visits more than one state.

## Instrument changes made in response
- `frozen_trajectory_ce2_value(num_states, traj_len)`: the reference value a frozen
  input produces, for detecting this artifact. Explicitly NOT for subtraction.
- `trajectory_degeneracy(trajectories, num_states)`: reports distinct states visited,
  transition count, coverage, and a `degenerate` flag.
- `compute_and_log_ce2` now logs `ce2_gates_states` / `ce2_workspace_states` and emits
  a warning when a CE 2.0 value is computed on a degenerate trajectory.
- The degeneracy ordering and the two pilot constants are pinned as regression tests
  in `tests/test_causal_emergence_svd.py`.

## Retraction
An earlier reading in this work, that `ce2_gates = 0.878` showed CE 2.0 escaping the EI
floor, was taken from a single CSV row before the trajectory was examined. It is
retracted: that number is the frozen-input reference.

## Caveats
Single seed, dark_room only, 500 episodes, one window length (2000 steps). The gate
discretization was not swept. Nothing here evaluates CE 2.0 on a non-degenerate input,
so the instrument's usefulness remains untested rather than refuted.
