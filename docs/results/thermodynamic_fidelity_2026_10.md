# Spike channel fidelity: a dose response of the class readout

**What this run is.** No pass or fail was tested. The probe measures how much of the readable stimulus class
survives a stochastic spike channel as the budget (events per dimension) grows. The probe and its report plan are in
`scripts/analysis/probe_thermodynamic_fidelity.py`. Both were committed (`c3da446`) before the data run. A pilot on a
discarded draw widened two grids before the run. Nothing else changed after the pilot.

**Result 1, the reference. The earlier UNTESTABLE verdicts came from a mis-tuned readout.** The earlier probe fixed the
ridge penalty at 100. With a penalty chosen by nested cross validation, all four streams carry the 6-class stimulus
label at all three models (clean accuracy 0.33 to 0.77 against chance 0.167, lower 95% bounds 0.31 or higher).
Statements in the earlier result documents that a stream "does not carry the class" are withdrawn. Each of those
documents now has a correction note. Their pass and fail verdicts remain valid as records of that probe only.

**Result 2, the cost. The budget needed depends on the stream and on the model, and the p-bit channel is cheaper
than the Poisson channel.**
- `obs_map` is cheap. The p-bit channel retains 95% of the above-chance accuracy at 1024 samples per dimension at all
  three models. The Poisson channel needs about 4096 bins.
- `workspace_broadcast` is costly. The p-bit channel reaches 90% only at 16384 samples, at all three models. It reaches 95%
  at 16384 for model 43 by point estimate only (the lower bound does not), and not at the other two. The Poisson channel stays
  at 0.64 to 0.77 at 16384 bins.
- `tectum_content` and `z_state` differ by model. At models 42 and 43 the p-bit channel reaches 90% between 1024 and 16384
  samples. At model 44 it reaches only 0.47 and 0.51 at 16384 samples.

**Read this correctly.** The channel decoder is unbiased, so retention tends to 1 as the budget grows, in any stream.
Survival at infinite budget follows from statistics and is not a finding. The finding is the budget, and how it varies.

## Reference, step 1

Clean accuracy of the tuned ridge readout (5-fold cross validation grouped by trial, 10 stimulus draws of 120 trials
per model, 1200 trials). Legacy is the old probe's reference (penalty 100, fixed).

| Model | Stream | Clean accuracy (tuned) | 95% lower bound | Legacy (penalty 100) | Median selected penalty | Trials |
|---|---|---|---|---|---|---|
| 42 | tectum_content | 0.455 | 0.428 | 0.322 | 0.01 | 1200 |
| 42 | workspace_broadcast | 0.745 | 0.721 | 0.419 | 0.002 | 1200 |
| 42 | obs_map | 0.757 | 0.733 | 0.537 | 0.065 | 1200 |
| 42 | z_state | 0.751 | 0.726 | 0.345 | 0.001 | 1200 |
| 43 | tectum_content | 0.627 | 0.601 | 0.402 | 0.01 | 1200 |
| 43 | workspace_broadcast | 0.738 | 0.713 | 0.480 | 0.003 | 1200 |
| 43 | obs_map | 0.754 | 0.730 | 0.576 | 0.1 | 1200 |
| 43 | z_state | 0.765 | 0.741 | 0.470 | 0.003 | 1200 |
| 44 | tectum_content | 0.335 | 0.310 | 0.161 | 0.001 | 1200 |
| 44 | workspace_broadcast | 0.726 | 0.702 | 0.398 | 0.001 | 1200 |
| 44 | obs_map | 0.763 | 0.739 | 0.562 | 0.065 | 1200 |
| 44 | z_state | 0.483 | 0.457 | 0.174 | 0.001 | 1200 |

- `workspace_broadcast` at 0.73 to 0.75 matches the project's earlier 0.69 to 0.77
  (`docs/results/broadcast_geometry_2026_08.md`).
- `obs_map` here reaches 0.75 to 0.76. The project measured about 1.0 on the full 16384-D map
  (`docs/results/collapse_locus_trained_2026_06_21.md`). This probe uses a 4 by 4 pooled map reduced to 256 components.
  The difference is expected and I did not separate pooling from the other choices.
- The selected penalty sits at or next to the lower edge of the grid (0.001) for several entries
  (model 42 `z_state`, model 44 `tectum_content`, `workspace_broadcast`, `z_state`). Those clean accuracies may be understated,
  and the readout there is close to unregularised.

## Dose response, step 2

rho(T) = (A_T - 1/6) / (A_clean - 1/6): the share of above-chance accuracy retained after the channel, with the readout
fitted on clean vectors only. T is the number of bins (Poisson) or p-bit samples (p-bit) per dimension. T90 and T95 are
the smallest doses at which the point estimate reaches 0.90 and 0.95. The value in brackets is the smallest dose at which
the lower 95% bound reaches it. "none" means above 16384. Intervals come from 2000 bootstrap resamples of trials. For rho
between 0.2 and 0.95 the interval width (high minus low) had a median of 0.04 and a maximum of 0.12 (60 values).

| Model | Stream | Channel | rho at 4 | rho at 16 | rho at 64 | rho at 256 | rho at 1024 | rho at 4096 | rho at 16384 | T90 (lower bound) | T95 (lower bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | tectum_content | poisson | -0.01 | 0.02 | 0.04 | 0.09 | 0.24 | 0.44 | 0.71 | none (none) | none (none) |
| 42 | tectum_content | pbit | 0.03 | 0.08 | 0.15 | 0.38 | 0.65 | 0.86 | 0.97 | 16384 (16384) | 16384 (16384) |
| 42 | workspace_broadcast | poisson | 0.01 | 0.01 | 0.04 | 0.08 | 0.19 | 0.38 | 0.65 | none (none) | none (none) |
| 42 | workspace_broadcast | pbit | 0.02 | 0.06 | 0.14 | 0.30 | 0.55 | 0.81 | 0.93 | 16384 (16384) | none (none) |
| 42 | obs_map | poisson | 0.03 | 0.07 | 0.16 | 0.35 | 0.69 | 0.96 | 1.00 | 4096 (4096) | 4096 (16384) |
| 42 | obs_map | pbit | 0.11 | 0.25 | 0.54 | 0.88 | 0.99 | 1.00 | 1.00 | 1024 (1024) | 1024 (1024) |
| 42 | z_state | poisson | 0.01 | 0.01 | 0.01 | 0.05 | 0.10 | 0.24 | 0.51 | none (none) | none (none) |
| 42 | z_state | pbit | 0.02 | 0.04 | 0.08 | 0.17 | 0.38 | 0.72 | 0.96 | 16384 (16384) | 16384 (16384) |
| 43 | tectum_content | poisson | -0.01 | 0.04 | 0.07 | 0.16 | 0.34 | 0.65 | 0.89 | none (none) | none (none) |
| 43 | tectum_content | pbit | 0.06 | 0.11 | 0.26 | 0.56 | 0.84 | 0.95 | 0.98 | 4096 (4096) | 4096 (16384) |
| 43 | workspace_broadcast | poisson | 0.02 | 0.03 | 0.06 | 0.14 | 0.28 | 0.52 | 0.77 | none (none) | none (none) |
| 43 | workspace_broadcast | pbit | 0.05 | 0.11 | 0.22 | 0.42 | 0.70 | 0.89 | 0.95 | 16384 (16384) | 16384 (none) |
| 43 | obs_map | poisson | 0.04 | 0.10 | 0.21 | 0.47 | 0.80 | 0.98 | 0.99 | 4096 (4096) | 4096 (4096) |
| 43 | obs_map | pbit | 0.15 | 0.34 | 0.68 | 0.95 | 0.99 | 0.99 | 1.00 | 256 (256) | 1024 (1024) |
| 43 | z_state | poisson | 0.01 | 0.03 | 0.07 | 0.17 | 0.37 | 0.72 | 0.97 | 16384 (16384) | 16384 (16384) |
| 43 | z_state | pbit | 0.04 | 0.11 | 0.26 | 0.56 | 0.90 | 1.00 | 1.00 | 1024 (4096) | 4096 (4096) |
| 44 | tectum_content | poisson | -0.01 | 0.00 | 0.03 | 0.01 | 0.01 | 0.05 | 0.11 | none (none) | none (none) |
| 44 | tectum_content | pbit | 0.02 | 0.00 | 0.02 | 0.04 | 0.10 | 0.23 | 0.47 | none (none) | none (none) |
| 44 | workspace_broadcast | poisson | 0.01 | 0.01 | 0.03 | 0.09 | 0.18 | 0.37 | 0.64 | none (none) | none (none) |
| 44 | workspace_broadcast | pbit | 0.03 | 0.06 | 0.14 | 0.29 | 0.55 | 0.79 | 0.92 | 16384 (16384) | none (none) |
| 44 | obs_map | poisson | 0.03 | 0.06 | 0.16 | 0.39 | 0.73 | 0.96 | 0.99 | 4096 (4096) | 4096 (16384) |
| 44 | obs_map | pbit | 0.12 | 0.28 | 0.59 | 0.91 | 0.99 | 1.00 | 1.00 | 256 (1024) | 1024 (1024) |
| 44 | z_state | poisson | -0.01 | -0.01 | -0.01 | 0.01 | 0.00 | 0.05 | 0.12 | none (none) | none (none) |
| 44 | z_state | pbit | -0.01 | -0.01 | -0.00 | 0.03 | 0.10 | 0.23 | 0.51 | none (none) | none (none) |

- At the same dose the p-bit channel retains more than the Poisson channel. The variance formulas explain the size. At
  x = 0.5 the Poisson decoder has variance 2.25 / T and the p-bit decoder has variance 0.25 / T. So the p-bit channel needs
  about 9 times fewer events for equal noise. The observed ratio of T95 for `obs_map` is 4 (1024 against 4096).
  The Poisson code here uses a spike probability of 0.2 times x per bin, which is an assumption (F_DT).
- `z_state` and `tectum_content` at model 44 do not reach 90% within 16384 samples. The next section looks at why.

## Why the cost varies: a sensitivity check (exploratory)

One model (44), one stimulus draw (300), one repeat, fixed ridge penalties. Hypothesis grade. It was not part of the plan and
it is not a result to cite.

| Stream | Penalty | Clean accuracy | rho at 1024, 4096, 16384 (p-bit) |
|---|---|---|---|
| z_state | selected (0.001) | 0.503 | 0.11, 0.21, 0.44 |
| z_state | fixed 1 | 0.192 | not interpretable (clean is near chance) |
| z_state | fixed 10 | 0.208 | not interpretable |
| tectum_content | selected (0.001) | 0.305 | 0.04, 0.05, 0.37 |
| tectum_content | fixed 1 | 0.193 | not interpretable |
| tectum_content | fixed 10 | 0.208 | not interpretable |

At model 44 these two streams are readable only by a nearly unregularised readout. With a penalty of 1 the clean
accuracy falls to near chance. So the class information sits in directions of low variance, and a white-noise channel
removes low-variance directions first. This explains the large budget. It is a hypothesis for the whole picture. I did not
confirm it with a variance analysis. The project has seen low-variance identity directions before in another latent
(`docs/results/collapse_locus_wmobs_2026_06_21.md`).

## What this does and does not show

- It shows the class information of all four streams is carried through a stochastic spike channel at a measurable cost, and
  that the cost depends on the spectrum of the representation as much as on the channel.
- It does not show a substrate-independence claim. It shows that one readout of one label survives at a stated budget. It
  uses recorded snapshots of the vectors and not their dynamics. It uses one code (sigmoid of a normalised vector, a
  rate-like channel with independent noise per dimension) and a readout fitted on clean vectors only.
- The channel numbers are for that protocol. A readout trained on channel outputs, or an encoding that allocates gain to
  the informative directions, would change them. These two changes are the next experiments.
- Instruments stay UNPROVEN, and 0 are TRUSTED. This result says nothing about consciousness or Phi.

## Caveats

3 checkpoints, 10 stimulus draws each (seeds 300 to 309), one phase (`sample`), one label (`sample_shape`), one noise
repeat count (2). The ridge penalty sits at the grid edge for several entries. Intervals cover trial sampling only. They do
not cover checkpoint-to-checkpoint variation, so nothing is pooled across models. The Poisson spike probability scale
(0.2) and the sigmoid map are assumptions. No energy is derived. An energy per event would be an assumption and none
was measured.

## What would change this, in order

1. **Variance-equalised encoding.** Test whether giving each informative direction equal channel signal-to-noise (gain allocation)
   cuts the budget at equal events. This is a design lever, and the data are already cached (`runs/thermo_fidelity`).
2. **A readout trained on channel outputs** at each dose, to separate channel loss from readout mismatch.
3. **The delay phase** and a memory test: hold the sample in a noisy attractor memory across the 12-step delay.
4. **The policy in the loop:** feed round-tripped vectors to the agent and measure task accuracy against budget.
5. **Penalty grid edge:** extend the grid below 0.001 and rerun the reference for the entries at the edge.

## Correction, 2026-10-10 (penalty grid)

The reference table above used a penalty grid with a lower limit of 0.001, and several entries sat at that limit. A later run with
the grid extended to 1e-5 (`docs/results/thermodynamic_encoding_2026_10.md`, the plain arm) raised the clean accuracy of model 44
`z_state` from 0.483 to 0.722, of model 44 `tectum_content` from 0.335 to 0.424, and of model 42 `z_state` from 0.751 to 0.769.
The other entries moved by 0.01 or less. So the clean accuracies and the retention values of those three entries in this document
are understated, and their budgets describe a readout that was not at its optimum. The caveat above that said so is confirmed.
The ordering of the streams by cost is unchanged for the other entries.
