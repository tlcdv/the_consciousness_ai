# Instrument repair (2026-07): EI floor correction (A1) + ignition diagnosis (A2)

Track A of the forward roadmap (`docs/roadmap_next_2026_07.md`) repairs the degenerate
signature instruments the 2026-07 assessment
(`docs/results/signature_assessment_2026_07.md`) found. This document records A1, the
effective-information floor-bias correction. It is a math fix validated on existing run
data; no training was run for it.

## The problem (restated from the assessment)

`ei_gates` was bit-identical at 0.031178 in every EI window of all five assessed runs.
That value is the constant-trajectory Laplace floor: with Laplace smoothing, a trajectory
frozen in one joint state gives an EI that depends only on (state count, window length),
not on any causal structure. The reported ~12x "causal emergence ratio" was
0.373712 / 0.031178, the ratio of the 8-state floor to the 243-state floor. Because the
floor is HIGHER for fewer states, a frozen macro level (8 states) automatically "beats" a
frozen micro level (243 states). The instrument reported emergence under total degeneracy.

## The fix

`models/evaluation/effective_information.py`:

- `constant_trajectory_floor(num_states, traj_len)`: the closed-form EI of a frozen
  trajectory. Unit-tested to match `compute_effective_information([np.zeros(L)], N)` to 12
  decimals across N in {2, 8, 243} and L in {10, 200, 10000}, and to reproduce the two
  observed frozen values exactly (0.031178 at N=243, 0.373712 at N=8, both at L=10000).
- `corrected_effective_information(trajectories, num_states)`: raw EI minus that floor,
  clamped at 0. A frozen trajectory now reports exactly 0; genuine transition structure
  survives (a deterministic 4-cycle stays > 0.5).
- `compare_ei_levels` now also returns `ei_gates_corr`, `ei_workspace_corr`, `ratio_corr`,
  `emergent_corr`. The raw fields are unchanged, so pre-registered thresholds and
  historical csv values still refer to the same quantity.

The training logger (`scripts/training/metrics_logger.py`) writes three new episode
columns (`ei_gates_corr`, `ei_workspace_corr`, `ei_ratio_corr`) and TensorBoard scalars,
so runs from now on carry the corrected reading by default. The read-only
`scripts/analysis/report_signatures.py` recomputes the corrected values for older runs
from their logged raw values (valid because the floor depends only on state count and
window length). 8 new unit tests in `tests/test_effective_information.py`; the EI +
metrics-logger test files pass (34 tests).

## Corrected result on the five existing runs (window = 10000)

Floors at this window: gates (243 states) 0.031178, workspace (8 states) 0.373712.

| run | window | corr ei_gates | corr ei_workspace | corr ratio |
|---|---|---:|---:|---|
| collapse_trained | ep49 | 0.000000 | 0.006537 | undefined (micro frozen) |
| collapse_trained | ep99 | 0.000000 | 0.000000 | undefined (both frozen) |
| wmrecon_trained | ep49 | 0.000000 | 0.314822 | undefined (micro frozen) |
| wmrecon_trained | ep99 | 0.000000 | 0.495655 | undefined (micro frozen) |
| wmobs_trained | ep49 | 0.000000 | 0.109955 | undefined (micro frozen) |
| wmobs_trained | ep99 | 0.000000 | 0.569519 | undefined (micro frozen) |
| wmpredict_trained | ep49 | 0.000000 | 0.211353 | undefined (micro frozen) |
| wmpredict_trained | ep99 | 0.000000 | 0.000000 | undefined (both frozen) |
| wmpredict_trained2 | ep49 | 0.000000 | 0.068629 | undefined (micro frozen) |
| wmpredict_trained2 | ep99 | 0.000000 | 0.000000 | undefined (both frozen) |

## What the correction reveals (the honest, sharper finding)

The corrected reading is neither "12x emergence" nor "no signal". It splits the two levels:

1. **The gate (micro) level is genuinely frozen in every window of every run** (corrected
   EI 0.000000 throughout). The five gate dimensions never leave one joint tertile bin for
   entire 50-episode windows. The 12x number was never measuring gate causal structure; it
   was measuring the Laplace floor of a constant.
2. **The workspace (macro) level does carry real transition structure** in most windows
   (corrected EI up to 0.570, e.g. wmrecon ep99 0.496, wmobs ep99 0.570). This is genuine,
   above the floor, and it varies across training, so the macro level is not degenerate.

The consequence is that the causal-emergence RATIO is currently ill-posed, not because the
macro level is empty but because the micro denominator is a true zero. You cannot divide
real macro structure by a frozen micro level and call the quotient emergence. The fix that
makes the ratio meaningful is therefore not in the EI math (now correct) but in the gate
discretization: the micro level must actually transition. Candidates (not yet built, this
is the A1 output that feeds the next step):

- The gate activations are saturated into one tertile joint state; either the gates need
  non-saturated dynamics, or the tertile binning must resolve their actual variation
  (per-dimension adaptive thresholds risk the self-referential inflation the fixed-tertile
  choice was made to avoid, so any change needs the same guard).
- This is the same low-variation root as the perception collapse and the ignition
  saturation: the content driving the gates barely changes step to step.

## A2: ignition selectivity. VERDICT: not fixable at the gate; the signal carries no task structure

The assessment showed ignition saturated (99.8 to 100 percent of steps conscious;
`is_conscious == input_energy >= EMA(input_energy)` in
`models/core/global_workspace.py`). A2 asked whether any thresholding scheme could make
it selective. Diagnosis first, no mechanism built.

**Step 1 (zero compute, proxy).** Over the six existing runs, `broadcast_mag` dips below
its own alpha 0.95 EMA on 50 to 56 percent of steps in four runs (median relative dip
1e-3 to 3e-3, evenly spread over training), while the two post-fix wm-predict runs show
the logged gate going quiet only in the first ~3000 steps. So the workspace output
oscillates as millinoise around its mean; an EMA gate on such a signal is a coin flip,
not task selectivity. The proxy cannot settle the question because the gate reads
`max(bound_bids)`, which is not logged.

**Step 2 (instrumented forward probe, `scripts/analysis/probe_ignition_signal.py`).**
Forward-only DMTS episodes through the standard pipeline (no training), recovering the
exact gate signal from consecutive EMA baselines and tagging every step with the task
phase. Two arms, 5300 steps each (3 episodes, seed 42, 60 trials):

| arm | fixation energy | sample | delay | choice | sample-vs-delay d | ignited (all phases) |
|---|---:|---:|---:|---:|---:|---:|
| trained (latentid tectum) | 1.498533 | 1.498976 | 1.499055 | 1.498991 | -0.024 | 0.998 to 0.999 |
| untrained init | 1.499040 | 1.499632 | 1.499822 | 1.499955 | -0.056 | 0.998 to 1.000 |

The gate signal is phase-invariant to within |d| < 0.06 in both arms (within-phase std
~3e-3 on a ~1.499 signal that sits against its ~1.5 ceiling; the AKOrN-boosted winning
bid is effectively pinned). Salience is positive on ~99.8 percent of steps in EVERY
phase. Sample ONSET does produce a consistently positive salience (mean +0.00067,
60 of 60 onsets), but it is 0.03 percent of the signal scale and salience is positive
nearly everywhere, so it separates nothing. `broadcast_mag` is equally phase-invariant
(|d| <= 0.03).

**Verdict: FAILED as a fixable-instrument, honestly characterized.** No threshold, EMA,
or centering scheme on `max(bound_bids)` can produce task-selective ignition, because
the signal contains no task contrast to select on. The saturation is a CONTENT problem:
the module bids do not vary with what is on screen. This is the same root as the
perception collapse (identity-free content propagating to the workspace) and it moves
the fix out of the instrument and into perception (Track B). No cosmetic parameter
tuning was done, per the standing rule. The probe stays in the repo as the measurement
tool that will show when ignition BECOMES selectable (a future agent whose bids carry
task structure will show a nonzero sample-vs-delay d here first).

## Status move

A2 adds no status move either: ignition remains un-measurable as a selective signature on
this agent, now with the cause pinned to the phase-invariant gate signal rather than the
threshold logic. The EI instrument's math is corrected and the corrected reading is now
logged by default.
No Butlin indicator status is moved on the strength of this: the corrected EI shows the
emergence comparison remains un-measurable on the current agent (frozen micro level), which
is an honest negative, not progress from PARTIAL to IMPLEMENTED. The value delivered is
that a future agent with non-frozen gates will now be measured correctly instead of scored
by a floor artifact. Pre-registered EI thresholds in `docs/preregistered_predictions.md`
are not revised; they refer to the raw quantity and are left intact, with this document
recording that the raw quantity was floor-biased on these runs.
