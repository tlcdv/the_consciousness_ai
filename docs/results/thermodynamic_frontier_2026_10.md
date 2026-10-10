# Rank-truncated equalised encodings: the events and energy frontier

**Result.** Keeping fewer principal axes shrinks the pre-processing cost, which the transform-cost study found to dominate total
energy. The class information of `obs_map`, `z_state` and `tectum_content` needs about 64 axes. With 64 equalised axes the ceiling matches
the full 256-axis encoding for `obs_map` and for `z_state` at models 42 and 43, and total energy falls about 4 times for any ratio of
multiply-add energy to channel-event energy of 100 or more. Below 64 axes the class is lost. `workspace_broadcast` behaves differently:
its ceiling is flat from 16 axes. For the raw streams, truncation raises the break-even ratio from 48 to 64 up to 210 to 280, which is still
well below the sourced reference ratio of about 2300.

Probe and plan: `scripts/analysis/probe_thermodynamic_frontier.py`. The arms, the cost rules, the predictions and the report plan were committed
(`824620c`) before the data run. One pilot on a discarded draw (model 43, draw 999) preceded it, and the docstring records that it contradicted
prediction F1. Data: the 30 cached recordings (models 42, 43, 44; stimulus draws 300 to 309; 1200 trials per model). p-bit channel, readout
trained on the channel's own noisy outputs. Raw output: `runs/thermo_fidelity/fro_*.json` and `frontier_summary.json` (local, gitignored).

## What was compared

`plain256` (the yardstick), `full{k}` (the top k axes, each scaled by var^(-1/2)) for k of 8, 16, 32, 64, 128 and 256, and `pca{k}`
(the top k axes, no scaling) for k of 16, 64 and 256. Events per decision are k x T. Multiply-adds follow the folding rules of the
transform-cost study, with the PCA projection to k axes costing d x k for the larger streams and the dense map 256 x k for the raw streams.
T95 is the smallest dose (p-bit samples per axis) at which accuracy reaches 95% of the above-chance accuracy of the plain256 arm's clean
readout. "none" means above 16384.

## Ceiling (clean accuracy) by number of axes

| Model | Stream | plain256 | full8 | full16 | full32 | full64 | full128 | full256 | pca16 | pca64 | pca256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | tectum_content | 0.45 | 0.19 | 0.26 | 0.36 | 0.46 | 0.48 | 0.49 | 0.26 | 0.46 | 0.51 |
| 42 | workspace_broadcast | 0.75 | 0.40 | 0.65 | 0.65 | 0.66 | 0.67 | 0.66 | 0.64 | 0.64 | 0.65 |
| 42 | obs_map | 0.76 | 0.22 | 0.33 | 0.48 | 0.75 | 0.75 | 0.75 | 0.33 | 0.76 | 0.76 |
| 42 | z_state | 0.77 | 0.21 | 0.27 | 0.42 | 0.77 | 0.77 | 0.76 | 0.28 | 0.77 | 0.77 |
| 43 | tectum_content | 0.63 | 0.22 | 0.29 | 0.44 | 0.65 | 0.68 | 0.67 | 0.29 | 0.65 | 0.69 |
| 43 | workspace_broadcast | 0.74 | 0.41 | 0.66 | 0.67 | 0.67 | 0.68 | 0.69 | 0.66 | 0.66 | 0.67 |
| 43 | obs_map | 0.75 | 0.23 | 0.31 | 0.49 | 0.76 | 0.76 | 0.76 | 0.32 | 0.76 | 0.75 |
| 43 | z_state | 0.77 | 0.20 | 0.31 | 0.46 | 0.77 | 0.77 | 0.77 | 0.31 | 0.77 | 0.77 |
| 44 | tectum_content | 0.42 | 0.17 | 0.21 | 0.33 | 0.43 | 0.47 | 0.48 | 0.21 | 0.44 | 0.48 |
| 44 | workspace_broadcast | 0.72 | 0.40 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.64 | 0.64 | 0.64 |
| 44 | obs_map | 0.76 | 0.25 | 0.33 | 0.48 | 0.77 | 0.76 | 0.77 | 0.33 | 0.76 | 0.76 |
| 44 | z_state | 0.72 | 0.18 | 0.23 | 0.37 | 0.68 | 0.74 | 0.74 | 0.22 | 0.67 | 0.72 |

- For `obs_map`, `z_state` and `tectum_content` the ceiling collapses below 64 axes (for `obs_map` 0.31 to 0.33 at 16 axes and 0.48 to 0.49 at
  32, against 0.75 to 0.77 at 64). At 64 axes `obs_map` matches full rank (0.754 to 0.765), and `z_state` matches it at models 42 and 43
  (0.766, 0.767). Model 44 `z_state` needs 128 axes (0.684 at 64, 0.741 at 128).
- `workspace_broadcast` is different. The ceiling is flat from 16 axes (0.646 to 0.664 at 16 axes, 0.647 to 0.694 at 256). Its class
  information sits in the top 16 axes. All encoded broadcast arms stay below the plain256 ceiling (0.725 to 0.751), as in the earlier studies, so no
  arm reaches the yardstick and no energy ratio can be reported for it.

## Budget to reach 95% of the plain256 clean accuracy (p-bit samples per axis)

| Model | Stream | plain256 T95 | full64 T95 | full128 T95 | full256 T95 | pca64 T95 | pca256 T95 |
|---|---|---|---|---|---|---|---|
| 42 | tectum_content | none | 256 | 64 | 64 | 16384 | 4096 |
| 42 | obs_map | 1024 | 64 | 16 | 16 | 4096 | 1024 |
| 42 | z_state | none | 1024 | 256 | 256 | none | none |
| 43 | tectum_content | 16384 | 64 | 64 | 16 | 16384 | 1024 |
| 43 | obs_map | 1024 | 16 | 16 | 16 | 4096 | 1024 |
| 43 | z_state | 4096 | 64 | 64 | 16 | 16384 | 4096 |
| 44 | tectum_content | none | 16384 | 4096 | 4096 | none | none |
| 44 | obs_map | 1024 | 64 | 16 | 16 | 4096 | 1024 |
| 44 | z_state | none | none | 16384 | 4096 | none | none |

At equal k, equalisation matters a great deal. At 64 axes `full64` reaches the target at 16 to 1024 samples (16384 for model 44 `tectum_content`) where `pca64` needs 4096 or
16384 or never reaches it (`obs_map`: 16 to 64 against 4096; `z_state` model 43: 64 against 16384; `tectum_content` models 42 and 43: 256 and
64 against 16384).

## Total-energy ratio against plain256 (E_plain256 / E_arm), at the rho given

rho is the energy of one multiply-add divided by the energy of one channel event. It is not measured here. 2300 is the labelled reference
ratio of the transform-cost study. Where plain256 does not reach the target, it is taken at 16384 samples, so the ratio is a lower bound.

| Model | Stream | Arm | rho 1 | rho 10 | rho 100 | rho 1000 | rho 2300 |
|---|---|---|---|---|---|---|---|
| 42 | tectum_content | full64 (lower bound) | 126.57 | 22.88 | 2.58 | 0.36 | 0.22 |
| 42 | tectum_content | full128 (lower bound) | 100.56 | 12.26 | 1.30 | 0.18 | 0.11 |
| 42 | tectum_content | full256 (lower bound) | 50.28 | 6.13 | 0.65 | 0.09 | 0.05 |
| 42 | obs_map | full64 | 7.51 | 4.37 | 4.04 | 4.00 | 4.00 |
| 42 | obs_map | full128 | 3.93 | 2.20 | 2.02 | 2.00 | 2.00 |
| 42 | obs_map | full256 | 1.96 | 1.10 | 1.01 | 1.00 | 1.00 |
| 42 | z_state | full64 (lower bound) | 7.53 | 4.37 | 4.04 | 4.00 | 4.00 |
| 42 | z_state | full128 (lower bound) | 3.94 | 2.20 | 2.02 | 2.00 | 2.00 |
| 42 | z_state | full256 (lower bound) | 1.97 | 1.10 | 1.01 | 1.00 | 1.00 |
| 43 | tectum_content | full64 | 201.12 | 24.52 | 2.60 | 0.36 | 0.22 |
| 43 | tectum_content | full128 | 100.56 | 12.26 | 1.30 | 0.18 | 0.11 |
| 43 | tectum_content | full256 | 58.96 | 6.24 | 0.65 | 0.09 | 0.05 |
| 43 | obs_map | full64 | 7.85 | 4.39 | 4.04 | 4.00 | 4.00 |
| 43 | obs_map | full128 | 3.93 | 2.20 | 2.02 | 2.00 | 2.00 |
| 43 | obs_map | full256 | 1.96 | 1.10 | 1.01 | 1.00 | 1.00 |
| 43 | z_state | full64 | 4.98 | 4.10 | 4.01 | 4.00 | 4.00 |
| 43 | z_state | full128 | 2.49 | 2.05 | 2.00 | 2.00 | 2.00 |
| 43 | z_state | full256 | 1.25 | 1.02 | 1.00 | 1.00 | 1.00 |
| 44 | tectum_content | full64 (lower bound) | 3.94 | 3.46 | 1.60 | 0.34 | 0.21 |
| 44 | tectum_content | full128 (lower bound) | 7.52 | 4.90 | 1.13 | 0.18 | 0.11 |
| 44 | tectum_content | full256 (lower bound) | 3.76 | 2.45 | 0.56 | 0.09 | 0.05 |
| 44 | obs_map | full64 | 7.51 | 4.37 | 4.04 | 4.00 | 4.00 |
| 44 | obs_map | full128 | 3.93 | 2.20 | 2.02 | 2.00 | 2.00 |
| 44 | obs_map | full256 | 1.96 | 1.10 | 1.01 | 1.00 | 1.00 |
| 44 | z_state | full128 (lower bound) | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| 44 | z_state | full256 (lower bound) | 1.60 | 1.07 | 1.01 | 1.00 | 1.00 |

- **Larger streams.** `full64` gives 4.0 from rho of 100 upward. This is the ratio of the projection costs (256 to 64 axes), because the
  projection dominates, and the events no longer matter. At rho of 1 the ratio is 5.0 to 7.9, and at 10 it is 4.1 to 4.4. At `full128` the ratio is 2.0.
  Model 44 `z_state` needs 128 axes, where the ratio is 2.0 as a lower bound.
- **Raw streams (`tectum_content`).** The break-even rho, where the ratio is 1, is 48 to 64 for 256 axes, 116 to 132 for 128 axes and 210 to 280 for
  64 axes (lower bounds where plain256 does not reach the target). At the reference ratio of 2300 every truncated arm still costs more than it
  saves (ratios 0.11 to 0.22). So truncation helps, but not enough to make the dense digital rotation pay at that reference.

## Predictions

- **F1, contradicted as written, supported in a modified form.** I predicted a small k of 16 to 32 would suffice for `obs_map`. It does not.
  The class needs about 64 axes. At 64 axes the energy saving is real (4 times).
- **F2, supported.** Ceilings fall with k for `tectum_content` (and for the other two large streams). For the raw streams a smaller k raises the
  break-even from about 64 to about 280. For `workspace_broadcast` the ceiling is flat from 16 axes.
- **F3, supported.** Model 44 `tectum_content` and `z_state` need 128 axes or more to reach the target.
- **F4, supported.** `full` is much cheaper in events than `pca` at the same k.

## Read with care

1. **The yardstick is the plain256 clean accuracy.** `workspace_broadcast` never reaches it. An observation that its encoded ceiling is
   flat from 16 axes uses a different yardstick, chosen after seeing the data. It is exploratory, and I did not test it with a gate.
2. **The 4 times saving is the ratio of the projection costs, 256 to 64 axes.** It does not depend on the events. A different dimension
   reduction, or a projection done in the substrate, would change it.
3. **Energy values rest on counted multiply-adds and an unmeasured ratio rho.** The reference ratio rests on a 45 nm digital figure and a
   modelled cell energy. Offline fitting and the energy to produce the vector are not counted.
4. **Lower bounds.** Where plain256 does not reach the target, the ratio is a lower bound at the 16384 cap.
5. **One readout family, snapshots, one label, one phase.** T95 values carry a one-rung (4 times) uncertainty between noise draws.
6. **Three checkpoints.** Intervals cover trial sampling only. Nothing is pooled across models.

Instruments stay UNPROVEN, and 0 are TRUSTED. This result says nothing about consciousness or Phi.

## What would change this, in order

1. **A noisy attractor memory across the delay phase.** This uses the dynamics of the task and not only snapshots.
2. **The policy in the loop:** feed round-tripped, truncated vectors to the agent and measure task accuracy against budget, at least 3 seeds.
3. **A projection in the substrate,** or a different dimension reduction, to test whether the 4 times ceiling on the saving moves.
4. **Sourced figures for a stated target** to replace the swept ratio rho.
