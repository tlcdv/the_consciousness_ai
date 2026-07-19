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

## A3: dormant-module micro-benchmark (RIIU, Levin, self-vector)

Roadmap Track A3: verify the three dormant measurement modules produce non-degenerate
signals on the current perception-fixed agent (`--rssm-latent-mode continuous`,
`--capsule-workspace-source all_levels`).

**Method.** Three DMTS episodes (200 steps each, seeds default/43/44), all three modules
enabled: `--enable-riiu --riiu-probe-all --enable-audio --enable-levin-metrics
--enable-self-vector`. The agent carried no task competence (reward -35.86, -37.16,
-40.66; trials_correct 3, 1, 0 out of the episode's trials, consistent with chance). The
goal was to check whether each module's logged csv column has non-zero variance, not to
solve DMTS. One 200-step episode per seed, so every number below is a within-episode
statistic on an UNTRAINED module, not a trained-agent signature.

**Results (3-seed aggregate).** Data from `runs/a3_microbench/metrics.csv`,
`runs/a3_seed43/metrics.csv`, `runs/a3_seed44/metrics.csv` (200 rows each):

| Signal | no_seed mean [cv] | seed_43 mean [cv] | seed_44 mean [cv] | Interpretation |
|--------|------------------:|------------------:|------------------:|----------------|
| `phi_riiu_broadcast` | 0.01316 [0.85] | 0.07195 [1.4] | 0.003591 [2.0] | **REPLICATED non-degenerate variance in all 3 seeds** |
| `phi_riiu_tectum` | 0.01316 [0.85] | 0.07468 [1.3] | 0.003885 [1.8] | Same pattern as broadcast |
| `phi_riiu_audio` | 0 [nan] | 0 [nan] | 0 [nan] | All zeros. DMTS has no audio input. |
| `levin_bioelectric_complexity` | 0.1744 [0.0003] | 0.1779 [0.0029] | 0.1679 [0.002] | Frozen (CV < 0.003 across seeds) |
| `levin_morphological_adaptation` | 9.3e-6 [0.48] | 0.0059 [3.3] | 0.0027 [4.8] | Near-zero magnitude, noisy |
| `levin_collective_intelligence` | 0.4964 [0.0002] | 0.5083 [0.0012] | 0.5 [0.0025] | Frozen at ~0.5 |
| `levin_goal_directed` | 0 [nan] | 0 [nan] | 0 [nan] | Confirmed zero (no goal embeddings) |
| `self_pred_mse` | 0.0033 [4.1] | 0.0051 [4.5] | 0.0017 [4.2] | High CV, tiny magnitude |
| `self_pred_skill` | -0.79 [-0.57] | -0.82 [-0.47] | -0.81 [-0.57] | Negative (below persistence baseline) |

The `self_pred_skill` CVs are negative because the mean is negative; the sign is kept
rather than dropped, since a negative CV is a reminder that the ratio is not a dispersion
measure when the mean crosses zero.

Note on `phi_riiu_broadcast` vs `phi_riiu_tectum`: in seed 42 the two columns are
BIT-IDENTICAL at every one of the 200 rows (both mean 0.0131583, cv 0.8513, max 0.04525).
That is expected, not a coincidence: the broadcast is built from the winning vision
payload, which IS tectum_content (`perception_decodability_2026_06_09.md`). The two
columns are therefore ONE signal, not two independent substrates, and must not be read as
mutual confirmation. They diverge slightly in seeds 43 and 44 only because a non-vision
module won some steps.

Episodes.csv: EI values all zero in all 3 seeds (200 steps too short for the 10000-step EI
window). `consciousness_ratio` = 1.0000, 0.9200, 0.8550 (seeds 42, 43, 44): NOT pinned at
1.0, unlike every run in the `signature_assessment_2026_07.md` set. `avg_phi` (pyphi) =
3.958e-03, 1.655e-03, 3.318e-04, a ~12x spread across seeds.

### A3: what this does and does not show

**What it shows.** The RIIU column is not frozen. It reports a varying value in all 3
seeds (CV 0.85 to 2.0) where the pyphi-on-gates instrument and the corrected micro EI
both read constants. RIIU measures a sliding-window SVD residual (the variance of the
activation after subtracting the dominant singular mode), NOT the 5 discrete gate values
that pyphi and the EI correction found frozen. So RIIU is the first phi-family column in
this project that is READABLE rather than pinned. That is an instrument-readiness result.

**What it does NOT show, stated plainly so it is not misread later.** This is NOT evidence
of integrated information in the agent:

1. The modules are UNTRAINED and randomly initialized. A random SVD residual varies
   because the input varies and the weights are random. Non-zero variance is the null
   expectation here, not a positive finding. Variance is not integration.
2. There is NO ablation and NO comparison arm. The project's own standard for an indicator
   move (set by the 2026-07-06 signature ablation, which used OFF/HALF/ON over the
   perception ladder across 3 seeds) is a CAUSAL response to a manipulation. Nothing was
   manipulated here.
3. The across-seed spread is ~20x (0.0036 to 0.072 broadcast mean). A quantity that moves
   20x with the random seed is not yet a characterized signature. The earlier draft of
   this section waved this off as "expected for untrained random initialization"; that
   reasoning cuts against the claim, not for it, because it concedes the variation is
   initialization noise.
4. Broadcast and tectum are one signal, not two (see the note above the fold).

**Therefore no Butlin indicator moves on the strength of A3.** RPT-2 stays PARTIAL. The
deliverable is a working instrument, not a measured signature. What A3 buys is that a
future ablation CAN now be read on the phi axis, where previously the column was frozen
and no manipulation could have registered.

### A3: the non-findings (honest characterizations)

- **Levin metrics** remain frozen because they compute on the broadcast/holon content, which
  is still near-constant (the perception fix moved identity into tectum_content but the
  broadcast magnitude itself remains ~1.135 with micro-variation). The Levin modules are
  not the bottleneck; content is. They will activate when the broadcast carries varied
  content (which requires the RL/credit-assignment wall, Track C1, to be solved).
- **Self-vector** self-prediction fails (skill < 0) for the same reason: the first-order
  features are too static. The predictor overfits to noise and underperforms the persistence
  baseline. This will improve when the broadcast carries richer dynamics.
- **Audio RIIU** requires an environment with actual audio input (not DMTS). Not a wiring
  defect.

### A3: recommended next step

RIIU is readable, which makes it a candidate AXIS for an ablation, not a result on its own.
The `phi_riiu_*` columns are already logged per-step to metrics.csv when `--enable-riiu` is
on. The decisive next experiment is the same design that produced the only replicated
signature this project has: run the existing OFF/HALF/ON perception ladder
(`signature_ablation_2026_07.md`) with `--enable-riiu` on, across >= 3 seeds, and ask
whether RIIU phi RESPONDS to the manipulation. A signal that varies with the seed but not
with the ablation is noise. Only a replicated ablation response would justify a status
move, and the gate should be pre-stated before the runs.

Do NOT use RIIU phi as a reward source (`--riiu-source broadcast`) yet. Optimizing an
instrument that has not been shown to track anything is how a metric gets gamed, and it
would destroy its value as a measurement.

Levin and self-vector remain OFF by default unless explicitly enabled for a content-rich
environment. No flag defaults change.

## Status moves (updated with A3)

- **Butlin RPT-2 (integrated information)**: NO MOVE. Stays PARTIAL. RIIU is readable
  where pyphi-on-gates is frozen, but a readable instrument on untrained modules with no
  ablation is not a measured signature. An earlier draft of this section moved RPT-2 to
  IMPLEMENTED; that was not supported by the data and is retracted here.
- **Butlin C1-GWT (global availability)**: no move, but the stated REASON is corrected.
  Ignition is NOT pinned at 1.0 in these runs (`consciousness_ratio` 1.0000, 0.9200,
  0.8550). An earlier draft asserted a flat 1.0 and reasoned from it; that was wrong.
  The honest position is that these three 200-step episodes are too short, and too
  confounded with the newly enabled modules, to say whether the `b568c52` selectivity is
  holding. It is an OPEN question, worth a read-only check against the saturated runs in
  `signature_assessment_2026_07.md`, not a settled negative.
- **Butlin EI (emergence)**: no move (micro level still frozen).
- **Levin intrinsic metrics**: no move (content-limited, not instrument-limited).
- **Self-model (PSM-1)**: no move (content-limited).

Pre-registered thresholds in `docs/preregistered_predictions.md` are not revised. RIIU phi
thresholds are not registered there (this is a new measurement pathway developed after
registration).
