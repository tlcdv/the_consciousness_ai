# Thermodynamic transduction, gate v2

**Verdict. FAILED on two streams, UNTESTABLE on two. No stream PASSED.**

| Stream | Gate v2 verdict | Why |
|---|---|---|
| workspace_broadcast | FAILED | H2 fails at seed 44 by 0.005 (gap 0.105 against a margin of 0.10). H1, H3, H4 hold at all 3 seeds. |
| obs_map | FAILED | H2 fails at all 3 seeds (gaps 0.253, 0.212, 0.160). H1, H3, H4 hold at all 3 seeds. |
| tectum_content | UNTESTABLE | H1 fails at seed 44. The ridge readout stayed at chance there. |
| z_state | UNTESTABLE | H1 fails at seeds 42 and 44. |

This is the second run. The first (gate v1, `thermodynamic_transduction_2026_10.md`) was UNTESTABLE
on all four streams. Gate v2 made two of them testable. The substrate independence thesis is not
shown by this run. It is not refuted either, see "What the numbers say".

Source. `runs/thermo_cost/gate_v2_sample.json` (complete, 3 seeds) and `gate_v2_sample.log`,
written on 2026-10-10. Raw output stays under `runs/`.

## What was run

| Item | Value |
|---|---|
| Checkpoints, seed 42, 43, 44 | `runs/capfix_alllevels`, `runs/capfix_seed43`, `runs/capfix_seed44` (`tectum.pt`) |
| Task rows | DMTS, phase `sample`, 120 trials per seed, 600 rows per seed, 6 classes (`sample_shape`) |
| Class counts, seed 42 | `[95, 90, 100, 90, 145, 80]`. Seed 43 `[140, 85, 80, 95, 115, 85]`. Seed 44 `[105, 90, 130, 80, 90, 105]` |
| Cross validation | 5 folds, grouped by trial |
| Reference | ridge readout on the normalised vectors (lambda 100) |
| Memory | 6 class prototypes in 256 spins. Sign of the class means of the vectors projected onto the ridge subspace. Projection-rule couplings |
| Transduction | sigmoid, Poisson rate code with 64 bins, threshold 0.5 |
| Settling | chromatic block Gibbs, beta 4.0, 30 sweeps, read from the final sweep |
| Null | 100 trial-level label shuffles. The subspace is refitted in each shuffle |
| Chance | 1/6 = 0.167. Null means were 0.158 to 0.170 |

The gate (H1 to H4, margin 0.10) is in the probe docstring and was committed as `23e2c65` before any v2 value
was read. One change followed in `eebad10`, also before any v2 result existed. The first v2 launch stopped after
about 3 hours with a singular matrix and wrote no output. The projection rule now uses a pseudo-inverse, which
equals the inverse for independent prototypes. The projection rule itself was chosen on synthetic data, after the
v1 result and before any v2 value from a checkpoint.

## Accuracy per stream and seed

Columns. ridge = the gated reference. quant = sign of the projected vector, no noise, no settling.
noisy = Poisson spikes without settling. settled = the full pipeline. control = settled at beta 0.05.
Nulls are the p95 over 100 shuffles.

| Stream | Seed | ridge | quant | noisy | settled | control | null p95 ridge | null p95 settled | Gates failed |
|---|---|---|---|---|---|---|---|---|---|
| tectum_content | 42 | 0.283 | 0.248 | 0.233 | 0.210 | 0.157 | 0.227 | 0.232 | H3 |
| tectum_content | 43 | 0.360 | 0.378 | 0.362 | 0.327 | 0.190 | 0.245 | 0.213 | none |
| tectum_content | 44 | 0.167 | 0.175 | 0.158 | 0.167 | 0.183 | 0.242 | 0.218 | H1, H3 |
| workspace_broadcast | 42 | 0.453 | 0.355 | 0.370 | 0.368 | 0.147 | 0.242 | 0.220 | none |
| workspace_broadcast | 43 | 0.433 | 0.350 | 0.347 | 0.335 | 0.155 | 0.234 | 0.215 | none |
| workspace_broadcast | 44 | 0.392 | 0.308 | 0.305 | 0.287 | 0.180 | 0.225 | 0.208 | H2 |
| obs_map | 42 | 0.550 | 0.392 | 0.445 | 0.297 | 0.157 | 0.234 | 0.207 | H2 |
| obs_map | 43 | 0.550 | 0.350 | 0.478 | 0.338 | 0.160 | 0.233 | 0.203 | H2 |
| obs_map | 44 | 0.417 | 0.308 | 0.333 | 0.257 | 0.185 | 0.233 | 0.210 | H2 |
| z_state | 42 | 0.242 | 0.197 | 0.237 | 0.205 | 0.162 | 0.250 | 0.205 | H1, H3 |
| z_state | 43 | 0.458 | 0.283 | 0.392 | 0.298 | 0.150 | 0.234 | 0.198 | H2 |
| z_state | 44 | 0.168 | 0.132 | 0.155 | 0.143 | 0.147 | 0.233 | 0.200 | H1, H3 |

## What the numbers say

- Two streams carry the class in a form the ridge readout finds at every seed (workspace_broadcast 0.39 to 0.45,
  obs_map 0.42 to 0.55, against a null p95 near 0.23). `tectum_content` carries it at 2 of 3 seeds and `z_state` at 1 of 3.
- In the two testable streams the settled pipeline stayed above its own null at every seed (H3 holds). The class partly
  survives transduction and settling. It loses more than the pre-stated margin to the ridge readout, so the gate FAILED.
- The control (beta 0.05) fell to chance in all 12 cells (H4 holds). The settled result needs the p-bit dynamics.
- The workspace_broadcast failure is narrow. The gap at seed 44 is 0.105 and the margin is 0.10. The gate is not changed.
- The loss differs by stream. Gap ridge minus settled
  - workspace_broadcast 0.085, 0.098, 0.105. Gap noisy minus settled 0.002, 0.012, 0.018. Settling costs almost nothing here.
  - obs_map 0.253, 0.212, 0.160. Gap noisy minus settled 0.148, 0.140, 0.077. Settling costs 0.08 to 0.15 here.
- The gate compares different readout families. Ridge is a linear readout of 256 real numbers. The settled arm reads one bit per
  dimension through 6 prototypes. The `quant` arm (no noise, no settling) already lags ridge by 0.11 to 0.20 on obs_map and
  by 0.08 to 0.10 on workspace_broadcast. So part of the H2 gap comes from binarisation and the prototype readout, and not from
  the p-bit physics. This was not separated by a gate. It is a hypothesis.
- The pbit_energy at the last sweep was between -124.1 and -124.8 in every cell. With the projection rule a stored prototype
  has energy -(n - K) / 2 = -125 for n = 256 and K = 6 (a closed form). The chains reach stored states in all streams. Energy
  says nothing about whether the state is the right class.

## Quantities that are not findings

- `fep_free_energy` change ranged from -14.1 to -5219.6 across cells. It scales with the normalisation and the component count.
- The GPU to p-bit energy ratio in the JSON is arithmetic on assumed constants (1 W, 50 MHz, 6.7e10 FLOP per joule) and on 256
  sequential colour classes of the dense memory. It is not a measurement of any chip.

## Caveats

3 seeds, 120 trials each (about 20 per class), one phase (`sample`), one beta, one sweep count, one memory rule.
Nothing was swept on checkpoint data. Class counts differ between seeds because the seeds differ. Instruments stay UNPROVEN.
No indicator or rubric entry changes. The result says nothing about consciousness or Phi.
The projection rule and the discriminative subspace were chosen after the v1 failure. They are design choices made in response
to a negative result, not a replication.

## What would change this

1. A gate v3 that compares the settled arm with `quant` (same prototype readout, no noise, no settling). That isolates what the
   Poisson code and the settling add on top of binarisation. It would be chosen after seeing this result, so it is exploratory
   until a fresh seed set is run. Pre-state it in the probe docstring and use seeds not used here.
2. A readout with more than one bit per dimension (multi-level spikes or several p-bits per dimension), to test whether
   binarisation is the main loss.
3. The `delay` phase (`--phase delay`), once `sample` has a stream that passes.
4. More trials for `tectum_content` and `z_state`, where the ridge readout is near the null at one or two seeds.
