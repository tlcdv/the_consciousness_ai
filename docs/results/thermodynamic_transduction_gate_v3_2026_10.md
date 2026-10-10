# Thermodynamic transduction, gate v3

**Verdict. PASSED on `workspace_broadcast` and `obs_map`. UNTESTABLE on `tectum_content` and `z_state`.
Read the PASS as narrow, for the reasons under "Why this PASS is narrow".**

| Stream | Gate v3 verdict | Note |
|---|---|---|
| workspace_broadcast | PASSED | K1 to K4 hold at all 3 seeds. The closest K2 margin is 0.17 points (seed 42). |
| obs_map | PASSED | K1 to K4 hold at all 3 seeds. At seed 42 the settled arm sits exactly on the K2 margin (gap -0.0500, 30 of 600 rows). |
| tectum_content | UNTESTABLE | K1 fails at model 44. The binarised readout stayed at chance there. |
| z_state | UNTESTABLE | K1 fails at model 44. K2 would also fail at models 42 and 43 (gaps -0.052, -0.065). |

What the PASS says. After Poisson spike transduction and p-bit settling, the 6-class stimulus label is
read out as well as it is from the noise-free binarised vector, within 5 accuracy points, at all 3 seeds,
in two streams. The control at beta 0.05 stayed at or below the null p95 in all 12 cells (0.148 to 0.185), so the result
needs the settling dynamics. What it does not say follows below.

This is the third run. Gate v1 (`thermodynamic_transduction_2026_10.md`) was UNTESTABLE on all four
streams. Gate v2 (`thermodynamic_transduction_gate_v2_2026_10.md`) FAILED on two and was UNTESTABLE on two.

Source. `runs/thermo_cost/gate_v3.json` (complete, 3 seeds, 100 permutations) and `gate_v3.log`, written on
2026-10-10. Raw output stays under `runs/`.

## What was run

| Item | Value |
|---|---|
| Models | `runs/capfix_alllevels`, `runs/capfix_seed43`, `runs/capfix_seed44` (`tectum.pt`), model seeds 42, 43, 44 |
| Stimulus sequences | recording seeds 142, 143, 144 (new). The three checkpoints are the same as in v1 and v2 |
| Rows | DMTS phase `sample`, 120 trials and 600 rows per seed, 6 classes (`sample_shape`) |
| Class counts | 142 `[85,105,85,120,90,115]`, 143 `[75,75,115,165,80,90]`, 144 `[115,80,100,105,90,110]` |
| Cross validation | 5 folds, grouped by trial |
| Memory | 6 class prototypes in 256 spins, sign of the class means in the ridge subspace, projection-rule couplings |
| Transduction and settling | sigmoid, Poisson rate code with 64 bins, threshold 0.5. Chromatic block Gibbs, beta 4.0, 30 sweeps, final sweep read |
| Null | 100 trial-level label shuffles, subspace refitted in each. Null means were 0.162 to 0.174 |
| Chance | 1/6 = 0.167 |

Gate v3 (K1 to K4, margin 0.05) is in the probe docstring and was committed as `e213c02` before any v3 value was read.
It was not changed after the data came in.

## Accuracy per stream and seed

Columns. ridge = linear readout (informational). quant = sign of the projected vector, no noise, no settling.
noisy = Poisson spikes without settling. settled = the full pipeline. control = settled at beta 0.05.
Nulls are the p95 over 100 shuffles. K2 holds if settled is at least quant minus 0.05.

| Stream | Model / stimulus | ridge | quant | noisy | settled | control | null p95 quant | null p95 settled | settled - quant | Gates failed |
|---|---|---|---|---|---|---|---|---|---|---|
| tectum_content | 42 / 142 | 0.353 | 0.328 | 0.335 | 0.303 | 0.160 | 0.227 | 0.227 | -0.025 | none |
| tectum_content | 43 / 143 | 0.325 | 0.320 | 0.312 | 0.270 | 0.148 | 0.234 | 0.242 | -0.050 | none |
| tectum_content | 44 / 144 | 0.152 | 0.143 | 0.158 | 0.165 | 0.167 | 0.242 | 0.205 | +0.022 | K1, K3 |
| workspace_broadcast | 42 / 142 | 0.388 | 0.322 | 0.348 | 0.273 | 0.185 | 0.234 | 0.217 | -0.048 | none |
| workspace_broadcast | 43 / 143 | 0.475 | 0.400 | 0.418 | 0.403 | 0.163 | 0.225 | 0.230 | +0.003 | none |
| workspace_broadcast | 44 / 144 | 0.375 | 0.358 | 0.352 | 0.327 | 0.162 | 0.242 | 0.227 | -0.032 | none |
| obs_map | 42 / 142 | 0.567 | 0.358 | 0.465 | 0.308 | 0.162 | 0.225 | 0.207 | -0.050 | none |
| obs_map | 43 / 143 | 0.558 | 0.300 | 0.422 | 0.282 | 0.152 | 0.225 | 0.207 | -0.018 | none |
| obs_map | 44 / 144 | 0.500 | 0.333 | 0.428 | 0.332 | 0.177 | 0.233 | 0.215 | -0.002 | none |
| z_state | 42 / 142 | 0.408 | 0.275 | 0.317 | 0.223 | 0.177 | 0.225 | 0.197 | -0.052 | K2 |
| z_state | 43 / 143 | 0.400 | 0.325 | 0.347 | 0.260 | 0.155 | 0.227 | 0.207 | -0.065 | K2 |
| z_state | 44 / 144 | 0.120 | 0.135 | 0.140 | 0.155 | 0.158 | 0.223 | 0.195 | +0.020 | K1, K3 |

## Why this PASS is narrow

1. **The design followed a failure.** Gate v3 was designed after the v2 result, on the same three checkpoints.
   Only the stimulus sequences are new. The margin of 0.05 was chosen after the v2 gaps between these arms
   (0.01 to 0.10) were seen. This is a follow-up with a fresh stimulus draw, not an independent replication of a prior claim.
2. **The reference is a weak one.** The comparison arm is the noise-free binarised readout. That arm is far below the
   ridge readout on `obs_map` (0.30 to 0.36 against 0.50 to 0.57) and below it on `workspace_broadcast` (0.32 to 0.40
   against 0.38 to 0.48). So "settled matches quant" means the p-bit stage adds little loss on top of binarisation.
   It does not mean the original vector survives. The representation loses most of its readable class information when
   it is reduced to one bit per dimension, and that loss is not tested by this gate.
3. **Margin edge cases.** `obs_map` at model 42 sits exactly on the margin (settled 185 correct, quant 215, of 600).
   `workspace_broadcast` at model 42 is 0.17 points inside it. `tectum_content` at model 43 also sits exactly on the
   margin (192 against 162 correct), in a stream that is UNTESTABLE anyway. A different draw could fail K2 at any of them.
4. **Poisson noise helped `obs_map`.** The noisy arm scored 0.095 to 0.122 above quant. Settling then removed that gain
   (settled minus noisy -0.157, -0.140, -0.097). So on `obs_map` the p-bit stage returns the state to the level of the
   noise-free binarised readout, and it does not improve on it. I did not investigate why noise helps the prototype readout.
5. **One configuration.** One beta, one sweep count, one memory rule, one phase (`sample`), 3 checkpoints, about 20 trials per class.
   Nothing was swept on checkpoint data.

## What the numbers say besides the gate

- Settled minus quant ranged from -0.065 to +0.022 in the 12 cells. Settled was below quant in 9 of 12 cells.
- In `workspace_broadcast`, noisy minus quant ranged from -0.007 to +0.027 and settled minus noisy from -0.075 to -0.015.
- `tectum_content` carries the class in the binarised readout at 2 of 3 stimulus draws (0.33 and 0.32 at models 42 and 43,
  chance at model 44). `z_state` does the same at models 42 and 43 and also sits at chance at model 44.
  Model 44 is also where `tectum_content` and `z_state` failed H1 in v2. The same cells fail across two stimulus
  draws, so the cause more likely belongs to that checkpoint and stream than to the stimulus sequence. I did not test why.
- `pbit_energy` at the last sweep stays at the stored-state value. That shows the chains reach stored states. It says
  nothing about the class.

## Quantities that are not findings

- `fep_free_energy` change varies with normalisation and component count and is not comparable across streams.
- The GPU to p-bit energy ratio in the JSON is arithmetic on assumed constants (1 W, 50 MHz, 6.7e10 FLOP per joule) and on
  256 sequential colour classes of the dense memory. It is not a measurement of any chip.

## Standing of the instruments

`thermodynamic_entropy`, `pbit_energy` and `fep_free_energy` stay UNPROVEN. The PASS concerns the class readout after
transduction, not these three quantities. None of them was shown to depend on the stimulus class. 0 instruments are TRUSTED.
No rubric entry or indicator changes. This result says nothing about consciousness or Phi.

## What would change this

1. **Replication on unused stimulus seeds with gate v3 unchanged** (for example recording seeds 242, 243, 244). The gate text is
   fixed, so this is a true replication of this result. It also settles the two edge cases. About 65 minutes of run time.
2. **Fresh checkpoints.** Three newly trained models, to separate the stimulus draw from the model. About 3 hours of serial training.
3. **A readout with more than one bit per dimension**, to test whether binarisation is the main loss.
4. **The `delay` phase** (`--phase delay`), for the stored-content question.
5. **Publication.** A public number needs at least the replication in item 1. Until then the page states no result.
