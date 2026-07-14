# Instrument repair (2026-07): EI floor correction (Track A1)

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

## Status move

The EI instrument's math is corrected and the corrected reading is now logged by default.
No Butlin indicator status is moved on the strength of this: the corrected EI shows the
emergence comparison remains un-measurable on the current agent (frozen micro level), which
is an honest negative, not progress from PARTIAL to IMPLEMENTED. The value delivered is
that a future agent with non-frozen gates will now be measured correctly instead of scored
by a floor artifact. Pre-registered EI thresholds in `docs/preregistered_predictions.md`
are not revised; they refer to the raw quantity and are left intact, with this document
recording that the raw quantity was floor-biased on these runs.
