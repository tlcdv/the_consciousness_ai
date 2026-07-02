# Consciousness-signature assessment on existing runs (2026-07-02): the instruments are largely DEGENERATE

**DEGENERATE-first summary.** Before measuring consciousness signatures on the current
agent (the accepted direction after the perception-collapse and world-model chapters,
see [perception_collapse_synthesis_2026_06_21.md](perception_collapse_synthesis_2026_06_21.md)
and [wm_predict_stage1_2026_06_24.md](wm_predict_stage1_2026_06_24.md)), a zero-compute
pilot over five recent trained runs checked whether the signature instruments themselves
discriminate anything. Most do not:

1. The effective-information (EI) causal-emergence measurement sits at a frozen floor at
   BOTH levels, and the reported "emergence ratio" of ~12x is the ratio of two constant
   floors, reproduced exactly by a constant trajectory. It currently measures window
   length and state-space size, not causal structure.
2. GNW ignition is saturated: 99.79 to 100 percent of steps are flagged conscious in
   every run, including the two runs trained after the selective-ignition fix.
3. The binding signal sync_R is invariant across five different training objectives
   (0.2662 to 0.2666).
4. Five instrument families were inactive (all-zero) in all runs: RIIU phi, all five
   Levin metrics, self-prediction, plus the dominance channel of PAD.
5. The one objective-sensitive signature is phi: low magnitude everywhere, but ~3.6x
   lower in the two action-conditioned world-model runs. Single seed, hypothesis only.

Consequence: "measure the signatures" cannot proceed as a reporting exercise; on this
agent most instruments return constants. The honest output is this characterization plus
the per-instrument requirement for becoming measurable (below).

## Method

Read-only aggregation, no training: `scripts/analysis/report_signatures.py` over the five
DMTS 100-episode seed-42 runs from the 2026-06-21..24 sessions (`runs/collapse_trained`,
`runs/wmrecon_trained`, `runs/wmobs_trained`, `runs/wmpredict_trained`,
`runs/wmpredict_trained2`; 20000 metric steps each; identical metrics schema). The five
runs differ only in the tectum training objective (reward-only; + frame reconstruction;
+ obs_map reconstruction; + value-equivalent world model, two reward-head designs), which
gives a free cross-objective sensitivity axis. All numbers below are from the script
output produced this session. The first three runs were trained before the `b568c52`
selective-ignition fix, the two wm-predict runs after it.

## Finding 1: EI is frozen at both levels; the emergence ratio is a floor artifact

Observed in `episodes.csv` (EI computed on 50-episode windows, at episodes 49 and 99):

| run | ei_gates (ep49 / ep99) | ei_workspace (ep49 / ep99) | ratio (ep49 / ep99) |
|---|---|---|---|
| collapse_trained | 0.031178 / 0.031178 | 0.380249 / 0.373712 | 12.20 / 11.99 |
| wmrecon_trained | 0.031178 / 0.031178 | 0.688534 / 0.869367 | 22.08 / 27.88 |
| wmobs_trained | 0.031178 / 0.031178 | 0.483667 / 0.943231 | 15.51 / 30.25 |
| wmpredict_trained | 0.031178 / 0.031178 | 0.585065 / 0.373712 | 18.77 / 11.99 |
| wmpredict_trained2 | 0.031178 / 0.031178 | 0.442341 / 0.373712 | 14.19 / 11.99 |

`ei_gates` is bit-identical in all ten windows of all five runs. The diagnosis script
reproduces it exactly: a trajectory that occupies ONE joint state for the whole
10000-step window gives EI = 0.031178 for the 243-state gate space under the
Laplace-smoothed TPM (`models/evaluation/effective_information.py`). So the five gate
dimensions never leave one joint tertile bin for entire 50-episode windows, in every run.
The value is a pure function of (window length, state count).

The same holds at the macro level: a constant trajectory in the 8-state workspace space
gives EI = 0.373712 at window 10000, which is exactly the value observed in four of the
ten windows (three runs at ep99, and it is the minimum everywhere). The remaining windows
(0.38 to 0.94) are slightly above the floor, meaning a few bin transitions occurred.

Therefore the reported "causal emergence" ratio of ~12x at the frozen windows equals
0.373712 / 0.031178, the ratio of the two floors. Because the floor is higher for
smaller state spaces, a macro level with fewer bins than the micro level "beats" it
whenever both are frozen. The instrument is biased toward reporting emergence under
degeneracy. This also reinterprets the historical EI-1 verdict in
[preregistered_predictions.md](../preregistered_predictions.md) ("first emergence at
episode 49"): that observation is consistent with the floor artifact, not with detected
causal structure. The pre-registered EI predictions and thresholds are NOT revised; what
is recorded here is that the current gate/workspace discretization cannot test them.

To become measurable, the EI instrument needs a micro level that actually transitions
(the gate activations are saturated into one tertile joint state; either the gates need
non-saturated dynamics or the binning needs to resolve their actual variation) and a
bias treatment for the Laplace floor (for example, subtracting the constant-trajectory
baseline for the given window length and state count).

## Finding 2: ignition is saturated; the selective-ignition fix survives only as an early transient

Per-step `is_conscious` over 20000 steps:

| run | fraction ignited | quiet steps | where the quiet steps are |
|---|---|---|---|
| collapse_trained (pre-fix) | 1.0000 | 1 | step 0 only |
| wmrecon_trained (pre-fix) | 1.0000 | 1 | step 0 only |
| wmobs_trained (pre-fix) | 1.0000 | 1 | step 0 only |
| wmpredict_trained (post-fix) | 0.9979 | 42 | steps 149 to 2926, none after |
| wmpredict_trained2 (post-fix) | 0.9996 | 9 | steps 1034 to 2790, none after |

The post-fix selective gate (`models/core/global_workspace.py`: ignition fires when
input energy exceeds its own EMA baseline, alpha 0.95) does produce quiet steps, but
only during the first ~15 episodes; after step ~2900 neither run ever goes quiet again
for the remaining ~17000 steps. Mechanism: `is_conscious` is equivalent to
`input_energy >= EMA(input_energy)`, and once the input energy stabilizes or creeps
slowly upward it never dips below its own running average. The earlier "1.0 -> ~0.93"
observation came from a 3-episode diagnostic smoke (`runs/_ignition_smoke`: per-episode
ratios 1.0, 0.95, 0.925), i.e. the transient, not a steady state. Episode-level
`consciousness_ratio` in the full runs: pre-fix mean 0.99995 (min 0.995); post-fix means
0.9979 (min 0.94) and 0.99955 (min 0.975).

To become measurable, ignition needs an input-energy signal that actually dips below its
recent average on task-relevant structure. On DMTS the workspace input energy is
near-constant (Finding 3 in the 2026-06-21 session log; broadcast_mag CV 0.02 to 0.30
here, with the variation concentrated in early training), so the gate has nothing to
discriminate. This is the same low-variation root as the perception collapse: the
content flowing into the workspace barely changes.

## Finding 3: sync_R does not respond to the training objective

Mean sync_R across the five runs: 0.2666, 0.2666, 0.2666, 0.2665, 0.2662 (CV within each
run 0.046 to 0.050; range roughly 0.16 to 0.32). Five different training objectives left
the AKOrN binding synchrony statistically indistinguishable. Same seed and env in all
runs, so objective-insensitivity is the demonstrated claim (seed/env sensitivity was not
tested here). This extends the closed Phi-1 chapter's finding: binding synchrony is not
only uncorrelated with phi in training, it is unmoved by the training objective entirely.

## Finding 4: phi is the one objective-sensitive signature, and it is tiny

Phi is computed by pyphi every 5th step (3999 computed values per run; the csv carries
zeros on skipped steps, so all stats here use computed steps only):

| run | phi mean | phi max | CV |
|---|---|---|---|
| collapse_trained | 1.12e-3 | 5.96e-3 | 0.98 |
| wmrecon_trained | 1.16e-3 | 1.03e-2 | 1.1 |
| wmobs_trained | 1.09e-3 | 8.14e-3 | 1.2 |
| wmpredict_trained | 3.19e-4 | 2.94e-3 | 0.78 |
| wmpredict_trained2 | 3.06e-4 | 2.94e-3 | 0.88 |

Two clean groups: the three reconstruction-family runs sit at ~1.1e-3, the two
action-conditioned world-model runs at ~3.1e-4, a ~3.6x reduction. So phi DOES vary and
DOES respond to an architecture/objective change, unlike the other signals. Its absolute
magnitude stays near zero throughout. Single seed per config: the group difference is a
hypothesis, not a law; it is also a between-groups observation on runs that differ in
exactly one code path, which makes it the cheapest future sensitivity probe.

## Finding 5: inactive instruments

In all five runs, every value of `phi_riiu` (all four variants), all five `levin_*`
metrics, `self_pred_mse`, `self_pred_skill`, and the PAD `dominance` channel is exactly
zero. These modules are default-off (RIIU, Levin metrics, self-prediction) or
structurally unused (dominance) in the current training path. Valence and arousal do
vary (valence mean ~0.127, CV ~1.8; arousal mean ~0.539, CV ~0.12) and are essentially
identical across the five runs. Any signature assessment that lists RIIU/Levin/
self-prediction values for these runs would be reporting zeros from disabled code, not
measurements.

## What this means for the mission

Per the 2026-06-02 decision, success is judged by consciousness signatures. This pilot
shows the signature instruments themselves are mostly degenerate on the current agent:
frozen micro-states (EI), a saturated gate (ignition), an objective-invariant binding
signal (sync_R), and disabled modules (RIIU, Levin, self-prediction). The degeneracies
share the already-characterized root: the perception/workspace pathway produces
near-constant content, so metrics computed on that content are near-constant. Reporting
Butlin indicator coverage on top of these instruments without this characterization
would overstate what is measured. The rubric's structural claims (mechanisms present)
stand; the measured-signature claims now have this document as their honest baseline.

The substrate-independence test (preregistered section 13) remains
BLOCKED-AND-CHARACTERIZED on agent competence; nothing here revises its thresholds.

## Honest scope

- All five runs are DMTS, seed 42, 100 episodes, one machine. Objective-sensitivity
  claims (Findings 3 and 4) are within-seed comparisons; seed and env sensitivity were
  not tested. The floor identities in Finding 1 are exact mechanical reproductions and
  do not depend on seed.
- No fresh compute was run for this assessment (the gate for fresh runs, a signal that
  is both non-degenerate and in need of an env/seed contrast, was not met by any
  signal).
- The EI floor bias and the gate saturation are properties of the measurement given the
  agent's dynamics; the EI code implements its documented formula correctly.

Script: `scripts/analysis/report_signatures.py` (read-only; reruns reproduce every
number above from the run csvs).
