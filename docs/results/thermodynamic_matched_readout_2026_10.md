# Readout trained on channel outputs: the encoding gain survives

**Result.** The variance-equalised encoding kept its advantage when the readout was trained on the channel's own noisy
outputs. The prediction that the advantage would shrink (Q2) was contradicted. This addresses the main reservation of
`docs/results/thermodynamic_encoding_2026_10.md`, which fitted every readout on clean vectors.

Probe and plan: `scripts/analysis/probe_thermodynamic_matched.py`. The regimes, predictions and report plan were committed
(`9c1f11b`) before the data run. A pilot on a discarded draw (model 44, draw 999) preceded it and added the Poisson channel to
the plan. No arm, prediction or report item was changed after the pilot. Data: the 30 cached recordings (models 42, 43, 44;
stimulus draws 300 to 309; 1200 trials per model). Raw output: `runs/thermo_fidelity/mat_*.json` and `matched_summary.json`
(local, gitignored). The earlier encoding study is the reference for the encodings (plain, pca0, half, full).

## Numbers

**Budget to reach 95% of the plain clean accuracy (p-bit channel).** T95* is the smallest dose (p-bit samples per dimension) at
which accuracy reaches 95% of the above-chance accuracy of the plain arm's clean readout. "none" means above 16384. CT is the
clean-trained readout and M is the matched-trained readout.

| Model | Stream | Plain CT / M | pca0 CT / M | half CT / M | full CT / M |
|---|---|---|---|---|---|
| 42 | tectum_content | none / none | 16384 / 4096 | 256 / 256 | 64 / 16 |
| 42 | workspace_broadcast | none / none | none / none | none / none | none / none |
| 42 | obs_map | 1024 / 1024 | 1024 / 1024 | 64 / 64 | 16 / 16 |
| 42 | z_state | none / none | none / none | 4096 / 1024 | 1024 / 256 |
| 43 | tectum_content | 16384 / 16384 | 1024 / 1024 | 64 / 64 | 16 / 16 |
| 43 | workspace_broadcast | none / none | none / none | none / none | none / none |
| 43 | obs_map | 1024 / 1024 | 1024 / 1024 | 16 / 64 | 16 / 16 |
| 43 | z_state | 16384 / 4096 | 16384 / 4096 | 1024 / 256 | 256 / 16 |
| 44 | tectum_content | none / none | none / none | none / none | 16384 / 1024 |
| 44 | workspace_broadcast | none / none | none / none | none / none | none / none |
| 44 | obs_map | 1024 / 1024 | 1024 / 1024 | 64 / 64 | 16 / 16 |
| 44 | z_state | none / none | none / none | none / none | 4096 / 4096 |

- Where the plain arm and the encoded arm both reach T95* (5 comparisons per arm and regime), the saving over plain is:
  clean-trained `half` 16 to 256 times and `full` 64 to 1024 times; matched `half` 16 to 256 times and `full` 64 to 1024 times.
  The ranges are the same in both regimes.
- The plain arm never reaches T95* in 7 of 12 entries, in either regime. `half` reaches it in 2 of those entries and `full` in 4,
  in both regimes.
- The Poisson channel gives the same picture (3 comparisons per arm: `half` 16 times, `full` 16 to 64 times, both regimes).
- The earlier encoding study reported 16 to 64 times for `half` and 64 to 256 times for `full`. This run reproduces its
  clean-trained arm with new noise draws and gets a wider range, because one entry moved by one rung of the dose ladder
  (model 43 `tectum_content`, plain: 4096 earlier, 16384 now). So the factors carry a one-rung (4 times) uncertainty.

**Accuracy at equal events (p-bit, matched-trained), paired differences with 95% trial bootstrap intervals, 12 entries:**
- `half` minus `plain`: above zero in 11 entries at dose 16, in 12 at dose 256, in 8 at dose 4096; below zero in none.
- `full` minus `plain`: above zero in 12 entries at dose 16, in 12 at dose 256, in 8 at dose 4096; below zero in none.
- `pca0` minus `plain` (the rotation control): above zero in 1, 2 and 2 entries, below zero in 0, 1 and 3 entries. A rotation alone
  does not produce the gain.

**Effect of matched training on its own (p-bit, matched minus clean-trained, 12 entries, intervals above or below zero):**

| Arm | Dose 16 | Dose 256 | Dose 4096 |
|---|---|---|---|
| plain | higher in 10, lower in 0 | higher in 8, lower in 3 | higher in 9, lower in 0 |
| pca0 | higher in 10, lower in 0 | higher in 4, lower in 4 | higher in 5, lower in 3 |
| half | higher in 5, lower in 3 | higher in 8, lower in 0 | higher in 7, lower in 0 |
| full | higher in 8, lower in 0 | higher in 7, lower in 1 | higher in 5, lower in 0 |

- Matched training helps the plain arm most where it was brittle: `workspace_broadcast` (model 42 accuracy at 256 samples 0.29 to
  0.46; model 43 0.37 to 0.53) and `z_state` (model 43 plain T95* 16384 to 4096; accuracy at 256 samples 0.35 to 0.55).
- It does not help everywhere. For `obs_map` the plain and `pca0` arms are slightly lower near the knee at 256 samples (plain: 0.69 to 0.64
  at model 42, 0.71 to 0.66 at model 44), and `half` is lower at 16 samples (0.71 to 0.67 at model 42, 0.72 to 0.68 at model 44).
- At dose 16, matched training was worse than clean-trained in 3 of 48 p-bit cells and in 11 of 48 Poisson cells. This confirms the
  data-limit prediction for sparse Poisson spikes (Q3).

**`workspace_broadcast`.** No arm reaches T95* in either regime, because the rotated arms have a lower clean ceiling (0.64 to 0.69
against 0.72 to 0.75). Under matched training the plain arm catches up at 4096 samples (accuracy 0.62 to 0.68 against 0.65 to 0.68 for
`half` and `full`). At low dose the encoded arms stay far ahead (accuracy at 16 samples: plain 0.27 to 0.33, `full` 0.63 to 0.65).

## Predictions

- **Q1, partly supported.** Matched training raised accuracy for the plain arm in most entries and most at low dose, but not uniformly
  (see `obs_map`), and the encoded arms gained less or mixed amounts.
- **Q2, contradicted.** The advantage did not shrink in budget terms. For some entries it grew, because the encoded arms gained from
  matched training as well (model 43 `z_state`, p-bit: plain 16384 to 4096, `full` 256 to 16).
- **Q3, supported for Poisson at low dose.** Matched training can be worse when the spikes are sparse and the training data small.
  The ceiling of the rotated `workspace_broadcast` arms stays.

## Read with care

1. **One readout family.** The readout is a ridge regression on sigmoid features. A nonlinear readout could change both regimes.
2. **Equal events, not equal cost.** The linear transform before the channel is not counted.
3. **Two noisy training copies per row.** More copies might help the matched readout. The Poisson result at low dose is data-limited.
4. **Rung noise.** Budget values move by one rung of the ladder between noise draws. Compare factors, not single rungs.
5. **Snapshots, one label, one phase.** The channel noise is independent per dimension. Sigmoid map, spike scale 0.2, shrinkage 1% are assumptions.
6. **Three checkpoints.** Intervals cover trial sampling only. Nothing is pooled across models.

Instruments stay UNPROVEN, and 0 are TRUSTED. This result says nothing about consciousness or Phi.

## What would change this, in order

1. **A cost for the transform.** Count the pre-processing operations next to the channel events, under stated assumptions.
2. **Sweep the equalisation strength** (exponent and shrinkage) on held-out draws, and test dropping tail axes.
3. **A noisy attractor memory across the delay phase.** This uses the dynamics and not only snapshots.
4. **The policy in the loop:** feed round-tripped vectors to the agent and measure task accuracy against budget, at least 3 seeds.
