# The gate cannot be woken by perturbation at any magnitude, and the floor debate was the wrong question

**The gate's causal response saturates at roughly 0.55 times its OWN spontaneous
fluctuation.** A hundredfold increase in impulse magnitude buys a twofold increase
in gate response. It never approaches the 3 sigma a detection requires, and it never
will.

**This is floor-independent**, which is why it settles a question that two days of
floor arguments could not. It compares the causal response to the site's own
fluctuation, a ratio that no threshold setting can change.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## The measurement that should have come first

Ratio of causal response to the site's own baseline fluctuation, at magnitude 1000,
across the three trained-gate checkpoints. A response must exceed 3 to register.

| Checkpoint | rssm (CONTROL) | gate (PRIMARY) | broadcast |
|---|---|---|---|
| `gate3_s42` | **45.59** | 0.26 | 0.83 |
| `gate3_s43` | **6.37** | 0.00 | 0.00 |
| `gate3_s44` | **6.68** | 0.01 | 0.01 |

The control clears the bar by 2 to 15 times. The gate reaches a quarter of it at
best. The impulse moves the gate by less than the gate moves on its own.

Nothing about variance floors, normalizations or thresholds changes this. The
quantity being detected is smaller than the quantity it must be distinguished from.

## The prediction, and its failure

The 2026-08 study found the rssm response scaled linearly with magnitude. If the
gate did too, a ratio of 0.26 at magnitude 1000 predicts 3.0 at roughly 11,500.

Tested on `gate3_s42`, 3 trials each:

| Magnitude | gate ratio | gate PCI | rssm active_fraction | rssm PCI |
|---|---|---|---|---|
| 1,000 | 0.27 | 0.0000 | 0.068 | 0.1654 |
| 10,000 | 0.51 | 0.0000 | 0.944 | 0.1509 |
| 100,000 | **0.55** | 0.0000 | 0.993 | **0.0227** |

**The gate does not scale. It saturates.** 100x the impulse gives 2x the response,
and the curve is flattening: 0.27 to 0.51 to 0.55. Extrapolating it to 3.0 is not
possible, because it is asymptoting well below 1.

The linearity reported in 2026-08 was a property of the rssm, not of the gate. The
gate numbers in that document (5.96e-08, 3.338e-07, 8.128e-06 at magnitudes 1, 100,
1000) were never linear either, and that was not noticed at the time.

## A second reading rule falls out: the control has an upper limit too

At magnitude 100,000 the rssm's `active_fraction` reaches 0.993 and its PCI collapses
from 0.165 to 0.023. That is the regime the module docstring names: when every entry
crosses threshold, source entropy goes to zero and so does PCI.

So the control is only valid inside a magnitude WINDOW. Below it the impulse does not
propagate; above it the response saturates and the measure inverts. Both ends are now
measured on this architecture:

- Too small: magnitude 1, response 156x below the noise floor (2026-08).
- Too large: magnitude 100,000, `active_fraction` 0.993 and PCI collapsing.
- Usable: magnitude 1000, `active_fraction` 0.068, PCI 0.165.

**Rule 7. Report `active_fraction` at the control with every PCI. A control above
roughly 0.9 means the impulse is too large and the number is meaningless.**

## What this establishes

- **The gate is unreachable by perturbation on this architecture.** Not "did not
  respond", not "below the floor": its response saturates at half its own noise, so
  no impulse can make it detectable.
- **The instrument is sound.** The control discriminates at every checkpoint and the
  determinism check passes at 0.0e+00 everywhere.
- **The floor question was secondary.** Two days of work on variance floors could
  not have resolved this, because the obstruction is a ratio, not a threshold.

## What this does NOT establish

- **It does not say the gate is causally disconnected.** The 2026-08 study showed
  the pathway EXISTS: perturbing `h_state` by magnitude 1000 changes
  `tectum_content` by 5.406e-04. The signal arrives attenuated, not absent.
- **It does not explain the saturation.** A saturating nonlinearity somewhere in the
  path is the obvious candidate, and `sensory_tectum.py:456` is a documented `tanh`
  saturation on a neighbouring path. That is a hypothesis and is NOT tested here.
- **It is one checkpoint for the sweep.** The ratio table is three checkpoints; the
  magnitude sweep is `gate3_s42` only.
- **It does not answer planning #15.** That needs the alternatives survey.

## Next

1. Find the saturation. Instrument the path from `h_state` to the gate and locate
   where a 100x input change becomes a 2x output change. `sensory_tectum.py:456` is
   the first place to look.
2. Repeat the magnitude sweep on `gate3_s43` and `gate3_s44` to confirm the
   saturation is not specific to one checkpoint.
3. Retire the floor question. Rules 1 to 7 stand, but the gate's readability is
   settled by the ratio, not by them.

## Reproduce

```
for M in 1000 10000 100000 ; do
  python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 3 \
    --magnitude $M --load-tectum runs/gate3_s42/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --out runs/_pci_multi/mag$M.csv
done
```

The ratio is `max_abs_response / median_baseline_sd`, both already columns in the
output CSV.
