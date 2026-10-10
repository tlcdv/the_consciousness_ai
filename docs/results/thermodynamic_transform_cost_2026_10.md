# Cost of the equalising transform

**Result.** The event-level saving of the variance-equalised encoding (16 to 1024 times fewer channel events) does NOT carry over to
total energy in most settings once the multiply-adds of the pre-processing are counted. It carries over only where one multiply-add
costs less than about 60 channel events, or where the stream needs no extra multiply-adds and the channel is a large part of the cost.
This tempers the claim of the earlier results, which counted channel events only.

Probe and plan: `scripts/analysis/probe_thermodynamic_transform_cost.py`. The counting rules, the ratio sweep and the report plan were
committed (`ea5b611`) before the first run. The diagonal-only check was added after the first report and committed (`7c3cabc`) before it
ran, and its docstring says so. The inputs are the matched-readout study's T95* values (p-bit channel). No new channel was run for the
cost report. Raw output: `runs/thermo_fidelity/transform_cost.json` and `diag_summary.json` (local, gitignored).

## Counting rules in one paragraph

One decision uses 256 channel dimensions. The energy of a decision, in units of one channel event, is 256 x T + rho x M, where T is the
dose, M the online multiply-adds, and rho the energy of one multiply-add divided by the energy of one channel event. rho is not measured
here. It is swept from 1 to 10000. All of an arm's linear steps fold into one matrix. A raw 256-D stream (`tectum_content`,
`workspace_broadcast`) needs a dense 256 by 256 matrix for the equalised arms instead of a diagonal z score: 65280 extra multiply-adds.
A larger stream (`obs_map` at 1024 dimensions, `z_state` at 16384) is first projected to 256 principal axes by a matrix both arms need. The
equalising scale folds into the columns of that matrix, so the extra is zero. Online multiply-adds: plain 1792 for the raw streams, 263680
for `obs_map` and 4195840 for `z_state`. The equalised arms add 65280 for the raw streams and nothing for the larger ones.

## Result 1: the dense rotation of the raw streams

Break-even rho* = 256 x (T95* of plain minus T95* of the arm) / 65280. Above rho* the transform costs more than the events it saves.
Where plain never reaches T95*, T95* of plain is set to 16384 and rho* is a lower bound.

| Model | Stream | Arm | T95* plain | T95* arm | Break-even rho* |
|---|---|---|---|---|---|
| 42 | tectum_content | half | none | 256 | at least 63 |
| 42 | tectum_content | full | none | 16 | at least 64 |
| 43 | tectum_content | half | 16384 | 64 | 64 |
| 43 | tectum_content | full | 16384 | 16 | 64 |
| 44 | tectum_content | full | none | 1024 | at least 60 |

`workspace_broadcast` has no entry, because no arm reaches T95*. Net saving factor in total energy (plain divided by arm), same five entries:

| rho | 1 | 10 | 100 | 1000 | 10000 | 2300 (reference) |
|---|---|---|---|---|---|---|
| half or full, range | 13 to 59 | 4.5 to 6.2 | 0.63 to 0.65 | 0.09 | 0.03 | 0.05 |

At rho of 10 the transform still saves energy (4.5 to 6.2 times). At 100 the total energy rises by 1.5 times. At 1000 and above it rises by
more than 10 times. So for the raw streams the dense rotation pays only if a multiply-add is cheaper than about 60 channel events.

## Result 2: the larger streams

- The equalising scale is free online. The diagonal-only variant (scaling the PCA scores with no rotation) gave the same T95* as the
  rotated arms in all 6 model and stream entries, both training regimes and both arms (24 comparisons), and accuracy within 0.01 at
  doses 16, 256 and 4096. The first report had shown
  that the obs_map scores are not a signed permutation of the principal axes (diagonality 0.42; `z_state` 1.0; the raw streams 0.25). The
  diagonal-only check shows this does not matter for the result. I did not find out why the diagonality is low. Near-equal variances among
  the principal axes would explain it, and I did not test that.
- The common PCA projection dominates the energy. The channel events equal the digital cost only at rho of about 1 for `obs_map` at 1024
  samples and at rho of about 1 for `z_state` at 16384 samples. At larger rho the common cost is larger, and the channel saving
  disappears from the total. Net saving factors (matched regime): `obs_map` 1.9 to 2.0 at rho of 1, 1.1 at 10, 1.0 from 100. `z_state` 1.2 to 2.0
  at rho of 1, 1.0 to 1.1 at 10, 1.0 from 100 (the plain arm never reaches T95* at models 42 and 44, so those values are lower bounds).

## Reference scenario (one point, labelled)

- A 32-bit floating point multiply costs 3.7 pJ and an add 0.9 pJ in 45 nm (Horowitz, ISSCC 2014, as quoted in a
  [Frontiers parameter table](https://pmc.ncbi.nlm.nih.gov/articles/PMC8934428/table/T3)). A multiply-add is about 4.6 pJ.
- Extropic models a cell energy of about 2 fJ per Gibbs-sampler cell update ([arXiv:2510.23972](https://arxiv.org/html/2510.23972v2),
  section III). The paper presents it as a physical model and not a measurement, and it includes the random number generator, the bias,
  the clock and the communication.
- The ratio is about 2300. Newer process nodes or integer arithmetic would lower it. I did not verify an 8-bit integer figure, and a cell
  in a coupled machine may differ from an independent sample. The sweep from 1 to 10000 covers these cases, and the conclusions above
  hold across it.

## What this does and does not say

- The equalisation is a fidelity-per-event lever. It is a total-energy lever only where the pre-processing is cheap relative to the
  channel, for example a rotation done in memory or in analog, or a stream that needs no rotation and has a small common cost.
- The boundary of this accounting is the step from the recorded vector to the readout. It leaves out the energy to produce the vector
  (the agent's forward pass) and the offline fit of the covariance. A claim about the efficiency of thermodynamic hardware needs those.
- The channel event energy is assumed equal for the p-bit and Poisson channels, and the setup cost per dimension (setting the bias) is
  left out because it is the same for every arm.
- T95* values carry a one-rung (4 times) uncertainty between noise draws. The break-even values for the raw streams sit close to 60 to 64 because
  T95* of plain is at or near the 16384 cap, so they are stable to that uncertainty.

Instruments stay UNPROVEN, and 0 are TRUSTED. This result says nothing about consciousness or Phi.

## What would change this, in order

1. **Rank-truncated equalisation for the raw streams.** Keep the top k principal axes (for example 32 of 256). The extra falls from 256 x 256 to
   256 x k multiply-adds and may keep most of the gain.
2. **A measured rho** for a stated target (a digital process or an in-memory array), from sourced figures for both the multiply-add and the channel event.
3. **A sweep of the equalisation strength** on held-out draws.
4. **The system boundary:** add the energy to produce the vector, so the comparison covers the whole decision.
