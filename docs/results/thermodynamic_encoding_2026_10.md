# Variance-equalised encoding through a stochastic spike channel

**Result.** Equalising the variance of the principal axes before the channel cut the number of channel events needed to
reach a given accuracy by a large factor, at the same number of dimensions and the same events per decision. At 256 samples
per dimension both equalised arms had higher accuracy than plain in all 12 model and stream entries. The gain included `obs_map`,
where I predicted the smallest gain. For `workspace_broadcast` the encoded vectors have a lower clean ceiling (0.64 to 0.69
against 0.72 to 0.75), so no encoded arm reaches 90% of the plain clean accuracy there. The result is an exploratory finding about a design lever.
It is not yet a claim that the encoding is better in general, for the reasons under "Read with care".

Probe and plan: `scripts/analysis/probe_thermodynamic_encoding.py`. The arms, the predictions and the report plan were
committed (`dc17e39`) before the run. One pilot on a discarded draw (model 43, draw 999) preceded it, and the docstring
records what it showed. No arm, prediction or report item was changed after the pilot.

Data: the 30 cached recordings of the fidelity study (models 42, 43, 44, stimulus draws 300 to 309, 1200 trials per
model). Raw output: `runs/thermo_fidelity/enc_*.json` and `encoding_summary.json` (local, gitignored).

## What was compared

Four encodings of the same vector, all with 256 dimensions, so the same events per decision at equal dose T:
`plain` (the fidelity probe's encoding), `pca0` (rotation into the training principal axes, a control), `half` (axes scaled
by var^(-1/4)) and `full` (axes scaled by var^(-1/2), shrinkage 1% of the mean variance). Each arm is rescaled to the same mean
training variance per dimension. The transform is fitted on the training rows and uses no labels. The readout is a ridge
readout fitted on the clean training vectors of the same arm. Its penalty comes from nested cross validation over a grid
extended to 1e-5. The channels are the p-bit channel and the Poisson channel (spike scale 0.2, an assumption).

## Clean accuracy per arm

| Model | Stream | plain | pca0 | half | full |
|---|---|---|---|---|---|
| 42 | tectum_content | 0.454 | 0.508 | 0.505 | 0.490 |
| 42 | workspace_broadcast | 0.751 | 0.651 | 0.667 | 0.665 |
| 42 | obs_map | 0.759 | 0.759 | 0.752 | 0.752 |
| 42 | z_state | 0.769 | 0.769 | 0.767 | 0.762 |
| 43 | tectum_content | 0.627 | 0.695 | 0.685 | 0.675 |
| 43 | workspace_broadcast | 0.738 | 0.668 | 0.678 | 0.694 |
| 43 | obs_map | 0.754 | 0.754 | 0.762 | 0.762 |
| 43 | z_state | 0.771 | 0.771 | 0.770 | 0.770 |
| 44 | tectum_content | 0.424 | 0.476 | 0.493 | 0.484 |
| 44 | workspace_broadcast | 0.725 | 0.642 | 0.647 | 0.647 |
| 44 | obs_map | 0.763 | 0.763 | 0.766 | 0.766 |
| 44 | z_state | 0.722 | 0.722 | 0.742 | 0.739 |

- `plain` here uses the extended penalty grid, so it differs from the fidelity study for the entries that sat at the old
  grid edge. Model 44 `z_state` rose from 0.483 to 0.722, model 44 `tectum_content` from 0.335 to 0.424, and model 42
  `z_state` from 0.751 to 0.769. The other entries moved by 0.01 or less. So the fidelity study understated the class
  information at model 44, as its caveats said it might.
- For `workspace_broadcast` the rotated arms have a lower clean accuracy than plain (0.64 to 0.69 against 0.72 to 0.75).
  I did not find out why. The sigmoid acts on each coordinate separately, so a rotation is not neutral for the 256-D streams.

## Budget to reach 90% and 95% of the plain clean accuracy (p-bit channel)

T90* and T95* are the smallest doses at which accuracy reaches 90% and 95% of the above-chance accuracy of the plain arm's
clean readout. The value in brackets is the smallest dose at which the lower 95% bound reaches it. "none" means above
16384. Doses are p-bit samples per dimension.

| Model | Stream | Arm | T90* (lower) | T95* (lower) | Plain T95* / arm T95* |
|---|---|---|---|---|---|
| 42 | tectum_content | plain | 16384 (16384) | none (none) | - |
| 42 | tectum_content | pca0 | 4096 (16384) | 16384 (16384) | plain never reaches (>16384), arm at 16384 |
| 42 | tectum_content | half | 256 (256) | 256 (256) | plain never reaches (>16384), arm at 256 |
| 42 | tectum_content | full | 16 (64) | 64 (64) | plain never reaches (>16384), arm at 64 |
| 42 | workspace_broadcast | plain | none (none) | none (none) | - |
| 42 | workspace_broadcast | pca0 | none (none) | none (none) | - |
| 42 | workspace_broadcast | half | none (none) | none (none) | - |
| 42 | workspace_broadcast | full | none (none) | none (none) | - |
| 42 | obs_map | plain | 1024 (1024) | 1024 (1024) | - |
| 42 | obs_map | pca0 | 1024 (1024) | 1024 (1024) | 1 |
| 42 | obs_map | half | 16 (16) | 64 (64) | 16 |
| 42 | obs_map | full | 4 (4) | 16 (16) | 64 |
| 42 | z_state | plain | none (none) | none (none) | - |
| 42 | z_state | pca0 | none (none) | none (none) | - |
| 42 | z_state | half | 4096 (4096) | 4096 (16384) | plain never reaches (>16384), arm at 4096 |
| 42 | z_state | full | 1024 (1024) | 1024 (4096) | plain never reaches (>16384), arm at 1024 |
| 43 | tectum_content | plain | 4096 (4096) | 4096 (16384) | - |
| 43 | tectum_content | pca0 | 1024 (1024) | 4096 (4096) | 1 |
| 43 | tectum_content | half | 64 (64) | 64 (64) | 64 |
| 43 | tectum_content | full | 16 (16) | 16 (16) | 256 |
| 43 | workspace_broadcast | plain | none (none) | none (none) | - |
| 43 | workspace_broadcast | pca0 | none (none) | none (none) | - |
| 43 | workspace_broadcast | half | none (none) | none (none) | - |
| 43 | workspace_broadcast | full | none (none) | none (none) | - |
| 43 | obs_map | plain | 256 (256) | 1024 (1024) | - |
| 43 | obs_map | pca0 | 256 (256) | 1024 (1024) | 1 |
| 43 | obs_map | half | 16 (16) | 16 (16) | 64 |
| 43 | obs_map | full | 4 (4) | 16 (16) | 64 |
| 43 | z_state | plain | 16384 (16384) | 16384 (16384) | - |
| 43 | z_state | pca0 | 16384 (16384) | 16384 (16384) | 1 |
| 43 | z_state | half | 1024 (1024) | 1024 (1024) | 16 |
| 43 | z_state | full | 256 (256) | 256 (256) | 64 |
| 44 | tectum_content | plain | none (none) | none (none) | - |
| 44 | tectum_content | pca0 | none (none) | none (none) | - |
| 44 | tectum_content | half | none (none) | none (none) | - |
| 44 | tectum_content | full | 4096 (16384) | 16384 (16384) | plain never reaches (>16384), arm at 16384 |
| 44 | workspace_broadcast | plain | none (none) | none (none) | - |
| 44 | workspace_broadcast | pca0 | none (none) | none (none) | - |
| 44 | workspace_broadcast | half | none (none) | none (none) | - |
| 44 | workspace_broadcast | full | none (none) | none (none) | - |
| 44 | obs_map | plain | 256 (1024) | 1024 (1024) | - |
| 44 | obs_map | pca0 | 256 (1024) | 1024 (1024) | 1 |
| 44 | obs_map | half | 16 (16) | 64 (64) | 16 |
| 44 | obs_map | full | 4 (16) | 16 (16) | 64 |
| 44 | z_state | plain | none (none) | none (none) | - |
| 44 | z_state | pca0 | none (none) | none (none) | - |
| 44 | z_state | half | none (none) | none (none) | - |
| 44 | z_state | full | 4096 (4096) | 4096 (16384) | plain never reaches (>16384), arm at 4096 |

- Where both the plain arm and the arm reach T95*, `half` needs 16 to 64 times fewer events (5 comparisons) and `full` needs
  64 to 256 times fewer (5 comparisons).
- The plain arm never reaches T95* within 16384 samples for model 42 `tectum_content` and `z_state` and for model 44
  `tectum_content` and `z_state`. In those four entries `full` reaches it at 64, 1024, 16384 and 4096 samples (model 42
  `tectum_content`, model 42 `z_state`, model 44 `tectum_content`, model 44 `z_state`). `half` reaches it in the two model 42
  entries (256 and 4096 samples) and not at model 44.
- No arm reaches T90* for `workspace_broadcast`, because the encoded ceiling is below the plain clean accuracy.
- The Poisson channel shows the same ordering. For example `obs_map` T95* falls from 4096 (plain) to 256 (`half` and `full`)
  at all three models.

## Accuracy at equal events, with the difference to plain (p-bit channel)

Differences are paired over trials. Intervals are 95% bootstrap intervals over trials.

| Model | Stream | Dose | plain | pca0 | half (diff to plain, 95% CI) | full (diff to plain, 95% CI) |
|---|---|---|---|---|---|---|
| 42 | tectum_content | 16 | 0.19 | 0.18 | 0.32 (+0.13, +0.11 to +0.14) | 0.43 (+0.24, +0.22 to +0.26) |
| 42 | tectum_content | 256 | 0.26 | 0.26 | 0.47 (+0.20, +0.18 to +0.22) | 0.48 (+0.22, +0.20 to +0.24) |
| 42 | tectum_content | 4096 | 0.40 | 0.43 | 0.50 (+0.10, +0.08 to +0.12) | 0.49 (+0.09, +0.07 to +0.11) |
| 42 | workspace_broadcast | 16 | 0.20 | 0.25 | 0.54 (+0.34, +0.32 to +0.36) | 0.50 (+0.30, +0.28 to +0.32) |
| 42 | workspace_broadcast | 256 | 0.29 | 0.46 | 0.63 (+0.34, +0.31 to +0.36) | 0.53 (+0.24, +0.22 to +0.27) |
| 42 | workspace_broadcast | 4096 | 0.51 | 0.63 | 0.65 (+0.15, +0.12 to +0.17) | 0.58 (+0.08, +0.05 to +0.10) |
| 42 | obs_map | 16 | 0.32 | 0.32 | 0.72 (+0.40, +0.38 to +0.42) | 0.75 (+0.44, +0.42 to +0.46) |
| 42 | obs_map | 256 | 0.69 | 0.68 | 0.75 (+0.06, +0.06 to +0.07) | 0.75 (+0.07, +0.06 to +0.08) |
| 42 | obs_map | 4096 | 0.76 | 0.76 | 0.75 (-0.01, -0.02 to -0.00) | 0.75 (-0.01, -0.02 to -0.00) |
| 42 | z_state | 16 | 0.19 | 0.18 | 0.25 (+0.06, +0.05 to +0.08) | 0.38 (+0.19, +0.18 to +0.21) |
| 42 | z_state | 256 | 0.24 | 0.24 | 0.49 (+0.25, +0.23 to +0.27) | 0.65 (+0.41, +0.39 to +0.43) |
| 42 | z_state | 4096 | 0.50 | 0.49 | 0.74 (+0.25, +0.23 to +0.26) | 0.76 (+0.26, +0.25 to +0.28) |
| 43 | tectum_content | 16 | 0.21 | 0.23 | 0.54 (+0.33, +0.31 to +0.34) | 0.63 (+0.42, +0.40 to +0.44) |
| 43 | tectum_content | 256 | 0.42 | 0.45 | 0.68 (+0.26, +0.24 to +0.27) | 0.67 (+0.25, +0.23 to +0.27) |
| 43 | tectum_content | 4096 | 0.60 | 0.67 | 0.69 (+0.08, +0.07 to +0.10) | 0.67 (+0.07, +0.05 to +0.09) |
| 43 | workspace_broadcast | 16 | 0.21 | 0.31 | 0.54 (+0.33, +0.31 to +0.35) | 0.41 (+0.20, +0.18 to +0.23) |
| 43 | workspace_broadcast | 256 | 0.37 | 0.54 | 0.59 (+0.22, +0.20 to +0.25) | 0.44 (+0.08, +0.05 to +0.10) |
| 43 | workspace_broadcast | 4096 | 0.58 | 0.65 | 0.63 (+0.04, +0.02 to +0.06) | 0.50 (-0.09, -0.12 to -0.07) |
| 43 | obs_map | 16 | 0.37 | 0.37 | 0.73 (+0.36, +0.35 to +0.38) | 0.76 (+0.38, +0.37 to +0.40) |
| 43 | obs_map | 256 | 0.72 | 0.72 | 0.76 (+0.04, +0.03 to +0.05) | 0.76 (+0.04, +0.03 to +0.05) |
| 43 | obs_map | 4096 | 0.75 | 0.75 | 0.76 (+0.01, +0.00 to +0.02) | 0.76 (+0.01, +0.00 to +0.02) |
| 43 | z_state | 16 | 0.20 | 0.20 | 0.37 (+0.17, +0.16 to +0.19) | 0.58 (+0.38, +0.36 to +0.40) |
| 43 | z_state | 256 | 0.36 | 0.35 | 0.67 (+0.31, +0.29 to +0.33) | 0.76 (+0.40, +0.38 to +0.42) |
| 43 | z_state | 4096 | 0.69 | 0.69 | 0.77 (+0.08, +0.07 to +0.09) | 0.77 (+0.08, +0.07 to +0.09) |
| 44 | tectum_content | 16 | 0.17 | 0.17 | 0.17 (-0.00, -0.01 to +0.01) | 0.19 (+0.02, +0.01 to +0.03) |
| 44 | tectum_content | 256 | 0.16 | 0.16 | 0.18 (+0.02, +0.01 to +0.03) | 0.25 (+0.09, +0.08 to +0.10) |
| 44 | tectum_content | 4096 | 0.17 | 0.17 | 0.23 (+0.06, +0.05 to +0.07) | 0.40 (+0.23, +0.21 to +0.25) |
| 44 | workspace_broadcast | 16 | 0.19 | 0.25 | 0.48 (+0.29, +0.27 to +0.31) | 0.50 (+0.31, +0.29 to +0.34) |
| 44 | workspace_broadcast | 256 | 0.28 | 0.43 | 0.57 (+0.29, +0.27 to +0.31) | 0.53 (+0.24, +0.22 to +0.27) |
| 44 | workspace_broadcast | 4096 | 0.51 | 0.62 | 0.62 (+0.11, +0.09 to +0.14) | 0.56 (+0.06, +0.03 to +0.08) |
| 44 | obs_map | 16 | 0.33 | 0.33 | 0.72 (+0.39, +0.37 to +0.41) | 0.76 (+0.43, +0.41 to +0.44) |
| 44 | obs_map | 256 | 0.71 | 0.71 | 0.76 (+0.06, +0.05 to +0.07) | 0.76 (+0.06, +0.05 to +0.07) |
| 44 | obs_map | 4096 | 0.76 | 0.76 | 0.77 (+0.00, -0.00 to +0.01) | 0.77 (+0.00, -0.01 to +0.01) |
| 44 | z_state | 16 | 0.17 | 0.17 | 0.17 (+0.00, -0.01 to +0.01) | 0.21 (+0.04, +0.03 to +0.05) |
| 44 | z_state | 256 | 0.16 | 0.17 | 0.21 (+0.05, +0.04 to +0.06) | 0.38 (+0.22, +0.20 to +0.24) |
| 44 | z_state | 4096 | 0.19 | 0.19 | 0.42 (+0.23, +0.21 to +0.25) | 0.70 (+0.50, +0.48 to +0.53) |

Across the 12 model and stream entries, the interval for `half` minus `plain` lies above zero in 10 at dose 16, in 12 at
dose 256 and in 10 at dose 4096. For `full` the counts are 12, 12 and 9. At dose 4096 one `half` entry and two `full` entries are
below plain (`obs_map` model 42 by 0.01, and `workspace_broadcast` model 43 by 0.09 for `full`), because the plain arm has caught up there.

## Predictions

- **P1, supported.** The gain is largest where the class sits in low-variance directions: model 44 `z_state` (plain 0.19 at 4096 samples,
  `full` 0.70) and `tectum_content` (0.17 against 0.40), and model 42 `z_state` (0.50 against 0.74 to 0.76).
- **P2, contradicted.** `obs_map` gained as much as the others in budget terms: T95* for the p-bit channel fell from 1024 to 16 to 64.
  Its accuracy at 4096 samples was already at the ceiling, so the gain is in the low-dose region.
- **P3, partly supported.** For `obs_map` and `z_state` the `pca0` arm matches `plain` (their vectors are already principal scores), so
  the gain there is not a rotation effect. For `tectum_content` and `workspace_broadcast` the rotation alone changes the result
  (higher budget efficiency, lower broadcast ceiling), so part of the gain there may come from the rotation.
- **P4, supported.** `full` hurts `workspace_broadcast` at high dose (0.50 at 4096 samples for model 43, against 0.63 for `half`
  and 0.58 for `plain`).

## Read with care

1. **The readout is fitted on clean vectors, with a penalty chosen on clean data.** A very small penalty makes a readout
   brittle to decoder noise. The plain arm suffers most from this, because its class information sits in low-variance directions.
   A readout trained on channel outputs would raise the plain arm too, so the real gain may be smaller than reported. This is the
   most important open question, and it is the next experiment.
2. **Equal events, not equal cost.** The equalisation needs a linear transform of the vector before the channel (a 256 by 256 matrix
   product and a scaling). That cost is not counted here. The comparison counts channel events only.
3. **The transform is fitted on training data from the same task.** It uses the covariance of the vectors, not labels. A real
   substrate would need that covariance in advance.
4. **Snapshots and one label.** The vectors are recorded snapshots and not dynamics. The label is `sample_shape`. One phase (`sample`).
5. **Three checkpoints.** Intervals cover trial sampling only. Nothing is pooled across models.
6. **Assumptions.** The sigmoid map, the spike scale 0.2 and the shrinkage 1% are choices. I did not sweep them on the data.
7. **Binomial counts.** The channel uses exact binomial counts of the per-bin process. Tests check the mean and variance against
   the closed forms and against the per-bin implementation.

The instruments `thermodynamic_entropy`, `pbit_energy` and `fep_free_energy` stay UNPROVEN, and 0 instruments are TRUSTED. This
result says nothing about consciousness or Phi.

## What would change this, in order

1. **A readout trained on channel outputs** at each dose, for every arm. This tests whether the gain survives a fair readout.
2. **A cost for the transform.** Count the pre-processing operations next to the channel events, under stated assumptions.
3. **Sweep the equalisation strength** (the exponent and the shrinkage) with a held-out set of draws, to find whether `half` or `full` is
   better and whether tail axes should be dropped.
4. **A noisy attractor memory across the delay phase**, and then the policy in the loop.
