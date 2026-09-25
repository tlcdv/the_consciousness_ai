# FAILED: the tectum shows a consistent phase gradient, but at 2 of 3 seeds it survives without the carried state, so it is not evidence of a traveling wave

**The pre-stated gate reads PRESENT at all 3 trained seeds, and the control that separates
propagation from input layout rejects that reading at 2 of 3.** The phase gradient
directionality (PGD) of the tectum ConvGRU state is above its spatial-shuffle null at seeds
42, 43 and 44. When the recurrent state h is zeroed before every step, the gradient falls to
its null at seed 42, and stays above its null with more than half of its value at seeds 43 and
44. At those two seeds the gradient comes from the input path, not from activity carried across
the map. Seed 42 is the only case consistent with propagation, and one seed is a hypothesis.

## Question and instrument

Muller, Busch, Davis and Reynolds (2026, Neuron) describe traveling waves as a product of local
recurrent connections on a map with conduction delays. The tectum carries a 64 by 16 by 16
ConvGRU state with 3 by 3 kernels across environment steps
(`models/core/sensory_tectum.py`). The question was whether that state already propagates,
before any wave mechanism is built.

`models/evaluation/wave_detection.py` (new, 9 tests in `tests/test_wave_detection.py`).

- `project_channels` projects the [T, 64, 16, 16] state onto its first principal channel
  direction.
- `phase_field` takes the analytic phase per cell with no band filter, following the
  filtering caution in Muller et al. 2026, Box 2.
- `phase_gradient_directionality` gives |mean gradient| / mean |gradient| per step (Rubino,
  Robbins and Hatsopoulos 2006). It is 1 for a plane wave and near 0 for noise, spirals and
  target waves. The tests pin all four cases.
- `count_phase_singularities` counts 2 by 2 plaquettes with a 2 pi phase winding. One spiral
  gives exactly 1, and a plane wave gives 0.
- `measure_waves` compares both with a null that moves each cell's whole time series to a
  random grid position, and refuses a field with a constant cell.

Units are cells per step. No Hz.

## Runs

`scripts/analysis/probe_tectum_waves.py`, read-only, DMTS, 512 steps with random actions,
100 shuffle surrogates, the three continuous-latent all-level checkpoints
`runs/capfix_alllevels`, `capfix_seed43`, `capfix_seed44`, and an untrained tectum as the
control. 384 steps are used after dropping the Hilbert edge samples.

```
python -m scripts.analysis.probe_tectum_waves --env dmts --steps 512 --surrogates 100 --runs-dir <runs>
python -m scripts.analysis.probe_tectum_waves --env dmts --steps 512 --surrogates 100 --runs-dir <runs> --no-carry
```

### Carried state (the pre-stated gate)

| tectum | seed | PGD | null mean | null p95 | above | singularities (null) |
|---|---|---|---|---|---|---|
| untrained | 42 | 0.0520 | 0.0379 | 0.0627 | no | 0.0 (2.0) |
| untrained | 43 | 0.0718 | 0.0433 | 0.0658 | yes | 0.0 (5.0) |
| untrained | 44 | 0.2075 | 0.0371 | 0.0625 | yes | 0.0 (0.0) |
| trained | 42 | 0.6232 | 0.0450 | 0.0696 | yes | 0.0 (0.0) |
| trained | 43 | 0.2026 | 0.0426 | 0.0631 | yes | 1.5 (15.0) |
| trained | 44 | 0.6126 | 0.0395 | 0.0624 | yes | 0.0 (0.0) |

Gate output, PRESENT at all 3 seeds. The untrained control is above its null at 2 of 3 seeds.

### h zeroed before every step (the control)

The shuffle null keeps each cell's own time series, so a spatially smooth pattern of response
latencies passes it even when nothing travels across the map, for example a pattern set by the
stimulus layout or by zero padding at the grid border. The control zeros h, and keeps the
latent z, before every step. The RSSM builds h_t from the previous h and the previous z, and
the observation enters through z, so zeroing h alone keeps the input path and cuts the
multi-step carry. A first attempt zeroed both h and z through `reset_state`, and every one of
the 256 cells was then constant at every step, for trained and untrained tectums alike,
because that also removes the input. That attempt is recorded, not used.

The reading was written into the probe docstring before this run. The gradient comes from the
input if the trained PGD stays above its null with no carry and keeps more than half of its
carried value. It needs the carried state if it falls to its null.

| tectum | seed | PGD | null p95 | above | carried PGD | reading |
|---|---|---|---|---|---|---|
| untrained | 42 | 0.1133 | 0.0736 | yes | 0.0520 | |
| untrained | 43 | 0.1065 | 0.0751 | yes | 0.0718 | |
| untrained | 44 | 0.2434 | 0.0700 | yes | 0.2075 | |
| trained | 42 | 0.0487 | 0.0609 | no | 0.6232 | needs the carried state |
| trained | 43 | 0.1643 | 0.0563 | yes | 0.2026 | input |
| trained | 44 | 0.5349 | 0.0599 | yes | 0.6126 | input |

## What is established

- A consistent phase gradient across the tectum map, above a spatial-shuffle null, at 3 of 3
  trained seeds. This is a measured property of the state.
- At 2 of 3 trained seeds that gradient does not depend on the carried recurrent state. It is
  produced through the input path, so it is not a traveling wave in the sense of Muller et al.
- At 1 of 3 (seed 42) the gradient needs the carried state. Hypothesis only.
- Singularity counts are 0 or near 0 in every trained run, so there is no sign of rotating
  waves. This is descriptive and no gate rests on it.

## Caveats

- DMTS has a fixed central stimulus and a gray background, so the input layout is strongly
  structured in space. A moving-stimulus environment could give a different answer. Not run.
- One random-action rollout of 512 steps per checkpoint. The trained checkpoints are the three
  capfix seeds; other training configurations were not measured.
- The first principal channel direction is one projection of 64 channels. Other directions
  were not measured.
- The PGD index cannot see spiral or target waves. The singularity count is the only reading
  for those.

## Consequence for the planning fork

The measurement that was to come before building wave-lattice oscillators on the tectum is
negative at 2 of 3 seeds. It gives no support for the claim that the current tectum already
carries traveling waves, and it also shows that a phase gradient on this map can come from the
input layout alone. Any future wave mechanism must be judged with the h-zeroed control, not
with PGD against the shuffle null alone.
