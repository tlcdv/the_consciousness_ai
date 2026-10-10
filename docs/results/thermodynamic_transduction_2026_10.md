# Thermodynamic transduction of recorded representations, first run

**Verdict. No transduction verdict was reached. All four streams are UNTESTABLE at all three seeds.**
The reference readout (nearest class centroid on the normalised vectors) did not rise above its
own label-permutation null at every seed for any stream. The pre-stated gate G1 therefore failed,
and the question "does the class survive p-bit transduction" was not asked of any stream.
This is not a PASS and it is not a measured failure of transduction. It says the readout in this
probe cannot see the class in these vectors at the stated sample size.

Source. `runs/thermo_cost/sample_phase.json` and `sample_phase.log`, written by
`scripts/analysis/probe_thermodynamic_cost.py` on 2026-10-10. Raw output stays under `runs/`.

## What was run

| Item | Value |
|---|---|
| Checkpoints, seed 42, 43, 44 | `runs/capfix_alllevels`, `runs/capfix_seed43`, `runs/capfix_seed44` (`tectum.pt`) |
| Task rows | DMTS, phase `sample`, 60 trials per seed, 300 rows per seed, 6 classes (`sample_shape`) |
| Class counts per seed | 42 `[45,45,70,35,60,45]`, 43 `[80,45,40,50,55,30]`, 44 `[70,60,55,30,35,50]` |
| Cross validation | 5 folds, grouped by trial |
| Memory | 6 class prototypes in 256 spins (Hebbian, zero diagonal) |
| Transduction | sigmoid, Poisson rate code with 64 bins, threshold 0.5 |
| Settling | chromatic block Gibbs, beta 4.0, 30 sweeps |
| Null | 100 trial-level label shuffles, full pipeline rerun for each |
| Chance | 1/6 = 0.167. The null means were 0.157 to 0.177 |

The gate was written in the probe docstring before this run (G1 reference above null p95, G2 settled
within 0.10 of reference, G3 settled above null p95, G4 control not above null p95). It was not
changed after the data came in. The ridge arm was added after a 3 permutation smoke run and
enters no gate. The smoke run is the only data seen before the gate was fixed in code.

## Accuracy per stream and seed

Columns. ref = nearest centroid, ridge = linear readout (informational), quant = sign of z,
noisy = Poisson spins without settling, settled = full pipeline, control = settled at beta 0.05.
Nulls are the p95 over 100 shuffles.

| Stream | Seed | ref | ridge | quant | noisy | settled | control | null p95 ref | null p95 settled | Gates failed |
|---|---|---|---|---|---|---|---|---|---|---|
| tectum_content | 42 | 0.173 | 0.233 | 0.217 | 0.177 | 0.123 | 0.197 | 0.260 | 0.244 | G1, G3 |
| tectum_content | 43 | 0.133 | 0.330 | 0.113 | 0.163 | 0.097 | 0.153 | 0.250 | 0.240 | G1, G3 |
| tectum_content | 44 | 0.067 | 0.153 | 0.067 | 0.070 | 0.233 | 0.190 | 0.267 | 0.234 | G1, G3 |
| workspace_broadcast | 42 | 0.307 | 0.290 | 0.307 | 0.250 | 0.223 | 0.130 | 0.250 | 0.240 | G3 |
| workspace_broadcast | 43 | 0.200 | 0.317 | 0.167 | 0.170 | 0.170 | 0.157 | 0.267 | 0.270 | G1, G3 |
| workspace_broadcast | 44 | 0.117 | 0.183 | 0.083 | 0.100 | 0.197 | 0.207 | 0.250 | 0.250 | G1, G3 |
| obs_map | 42 | 0.183 | 0.500 | 0.433 | 0.190 | 0.127 | 0.170 | 0.267 | 0.213 | G1, G3 |
| obs_map | 43 | 0.250 | 0.583 | 0.433 | 0.223 | 0.193 | 0.170 | 0.250 | 0.210 | G1, G3 |
| obs_map | 44 | 0.117 | 0.333 | 0.317 | 0.170 | 0.200 | 0.140 | 0.251 | 0.207 | G1, G3 |
| z_state | 42 | 0.167 | 0.333 | 0.227 | 0.147 | 0.150 | 0.163 | 0.284 | 0.200 | G1, G3 |
| z_state | 43 | 0.163 | 0.400 | 0.200 | 0.187 | 0.160 | 0.153 | 0.233 | 0.204 | G1, G3 |
| z_state | 44 | 0.067 | 0.183 | 0.260 | 0.133 | 0.160 | 0.167 | 0.237 | 0.220 | G1, G3 |

## What the numbers say and what they do not

- The full pipeline (`settled`) was at or below its null p95 for all 12 stream and seed cells.
  That holds even where the reference sees the class (workspace_broadcast seed 42, reference 0.307).
  At that cell settled was 0.223 against a null p95 of 0.240. That one cell is the only place where
  the question was askable, and the class did not survive there at one seed.
- The control (beta 0.05) stayed at or below the null p95 in every cell, so G4 held everywhere
  The instrument falls to chance when the p-bit physics is removed.
- Several reference accuracies are below chance (0.067 at tectum_content seed 44 and z_state seed 44).
  I did not investigate this. A likely cause is class means that do not transfer across trials, but
  I did not test that.
- The linear ridge readout reached 0.50, 0.58 and 0.33 on obs_map, and 0.33 to 0.40 on z_state at
  two seeds. Those values lie above the reference null, but ridge has no null of its own in this
  run, so they are not evidence yet. They suggest the class sits in obs_map and z_state in a form
  that a centroid readout misses.
- Quantizing alone (sign of z) scored 0.433 on obs_map at seeds 42 and 43, higher than the settled
  pipeline. The Poisson code and settling lowered accuracy there. No cause was tested.
- Class counts differ between the three seeds because the seeds differ. The three recordings are
  different stimulus sequences, and each seed pairs with its own checkpoint.

## Quantities that are not findings

- `fep_free_energy` change ranged from -13.7 to -2838.7 across cells. It scales with the
  normalisation and the number of components, so no value is comparable across streams.
- The GPU to p-bit energy ratio was 0.382 in every cell. It is arithmetic on assumed constants
  (1 W, 50 MHz, 6.7e10 FLOP per joule) and on 256 sequential colour classes of the dense memory.
  It is not a measurement and not a statement about any chip.
- `pbit_energy` fell during settling in every cell. That shows the sampler descended. It says
  nothing about the class.

## Caveats

3 seeds, 60 trials each, one phase (`sample`), one memory design, one beta, one sweep count,
nearest centroid as the gated reference. Nothing was swept. `delay` phase was not run.
Instruments stay UNPROVEN. No indicator or rubric entry changes.

## What would change this

1. A new pre-statement with a stronger gated reference (the ridge readout, with its own null), then
   a new run. The gate in this document must not be edited to fit the data.
2. More trials per class. 60 trials give 10 trials per class on average and about 8 per class in
   each training fold, which is thin for 256 dimensions.
3. A prototype memory built from a discriminative direction (for example class means of the ridge
   projection) so the p-bit readout uses the information the ridge readout finds.
4. The `delay` phase, once the `sample` phase has a testable stream.

## Correction, 2026-10-10

The ridge readout in this probe used a fixed penalty of 100 that was never tuned. A later run with a penalty chosen by nested
cross validation (`docs/results/thermodynamic_fidelity_2026_10.md`) found that all four streams carry the 6-class stimulus label
at all three models (clean accuracy 0.33 to 0.77, chance 0.167). So any statement in this document that a stream "does not
carry the class", or that the readout was "at chance" because of the stream, is withdrawn. It describes the readout of this
probe. The pass, fail and untestable verdicts above remain correct records of what this probe measured. They are not
statements about the representations. The gate series is superseded by the dose-response study.
