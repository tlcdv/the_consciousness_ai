# The reading rule exists, and the gate responds on ONE checkpoint of three

**Scope first. The gate produces a measurable causal response on `gate3_s42` and on
neither of the other two trained checkpoints.** On `gate3_s42` it registers on 16 of
20 probe seeds at 6.87 to 24.42 times its own baseline. On `gate3_s43` and
`gate3_s44` it registers on 0 of 20, with responses of 0 to 2.384e-07, which is 0 to
8 units in the last place of float32. **Under the standing rule that a headline needs
three seeds, "the gate responds" is a statement about ONE checkpoint, replicated 16
times inside it, and not a statement about the architecture.**

**The reading rule the inventory falsifier asked for now exists.** The gate's
baseline fluctuation is BIMODAL across probe seeds, with an empty multiplicative gap
of 42.5x to 60.3x, and the four noisy seeds are THE SAME four on independently
trained checkpoints. A threshold placed in that gap selects which probe seeds can
carry a gate reading at all.

**The default variance floor silences the response that exists.** On the 16 quiet
seeds the gate baseline sd is 2.508e-06 to 2.521e-06, which is 40x BELOW
`DEFAULT_VAR_FLOOR` of 1e-4, so every gate channel is marked dead before measurement.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move. PCI stays UNPROVEN.

## Method

20 probe seeds (42 to 61) on each of the three trained-gate checkpoints. Magnitude
1000, the only magnitude rule 7 permits. Variance floor 1e-6, chosen earlier because
it admits the four live gate nodes and excludes the dead one. Every run: determinism
check passed at pre-impulse divergence 0.000e+00, every site `[trained]`, no
DISQUALIFIED block, no random-gate warning.

## The baseline is bimodal and the split is a property of the PROBE SEED

`gate3_s42`, gate baseline sd across 20 probe seeds:

| Group | Count | Baseline sd | Registers |
|---|---|---|---|
| Quiet | **16 of 20** | 2.508e-06 to 2.521e-06 | **16 of 16** |
| Noisy | 4 of 20 | 1.519e-04 to 2.120e-04 | 0 of 4 |

The largest multiplicative gap in the sorted baselines is **60.3x, between 2.521e-06
and 1.519e-04, with nothing inside it**. The quiet group spans 0.5 percent.

The same structure appears on `gate3_s44`: a 42.5x empty gap, 16 seeds below and 4
above. **The four noisy seeds are identical on both: 43, 44, 45 and 53.** On
`gate3_s43` the same four seeds are elevated as well, at 9.039e-05 to 1.255e-04
against a floor group at 5.487e-05, though the separation there is only 2.3x.

Two independently trained networks, the same four probe seeds noisy. **The noisy
regime is a property of the rollout the seed produces, not of the trained weights.**

## The response is stable while the baseline moves

`gate3_s42`, all 20 seeds:

| Quantity | Range | Spread |
|---|---|---|
| Gate response | 1.732e-05 to 6.154e-05 | 3.6x |
| Gate baseline sd | 2.508e-06 to 2.120e-04 | **84.5x** |

This is what the retracted saturation verdict mistook for a saturating response. The
response is not the unstable quantity.

## Per checkpoint

| Checkpoint | Control (rssm) | Gate registers | Gate response range |
|---|---|---|---|
| `gate3_s42` | 0.1640 sd 0.0100 | **16 of 20** | 1.732e-05 to 6.154e-05 |
| `gate3_s43` | 0.0487 sd 0.0041 | 0 of 20 | 0 to 1.192e-07 |
| `gate3_s44` | 0.0623 sd 0.0029 | 0 of 20 | 5.960e-08 to 2.384e-07 |

The control is alive on all three. On `gate3_s42` the gate's mean PCI is 0.1070
against a control of 0.1640, so where it responds it responds at a size comparable to
the control.

**The zeros on `gate3_s43` and `gate3_s44` are not a floor artefact.** Their
responses are 0 to 8 ulp of a float32 near 0.485, where one ulp is 2.9802e-08. No
threshold recovers a quantity at numerical resolution. Those two gates are genuinely
unreachable by this perturbation.

## Why this was never seen at the default floor

At `var_floor` 1e-4, `gate3_s42` reads 0.0000 on all 20 seeds, for two different
reasons that look identical in the output:

- the **16 quiet seeds** have a baseline of 2.5e-06, 40x below the floor, so every
  channel is marked dead and cannot register at any response size;
- the **4 noisy seeds** have a baseline above the floor, so their channels are live,
  but their response does not clear 3 sigma of that larger baseline.

This is reading rule 1 exactly: a zero carries two indistinguishable meanings. It is
now measured on 20 seeds rather than argued.

## Rule 8

**Rule 8. Select probe seeds by measured gate baseline sd before reading the gate.**
Run the clean rollout first, take the gate's median baseline sd, and read the gate
only on seeds in the quiet group. The groups are separated by a 42x to 60x empty gap
on every checkpoint measured, so the threshold is not a tuned parameter. Report how
many seeds were admitted and how many rejected. A gate reading averaged over both
groups describes neither, and the 0.0439 five-trial mean reported earlier is such an
average.

## A limit on what gate PCI can ever say

**Gate PCI is coarsely quantized.** The LZ complexity of a 5-channel by 60-step
binary matrix took only the values 2, 4, 5 and 6 across the 60 readings here.

> **CORRECTED by `pci_content_2026_09.md`.** Over 360 readings the gate LZ takes 2,
> 4, 5, 6, 7 and 8, giving SIX PCI values: 0.0, 0.109718, 0.137147, 0.164576,
> 0.192006 and 0.219435. "Four possible values" was true of this sample and wrong as
> a general statement. The coarseness itself stands: the step is 0.027429.

So a gate PCI can report "registers" or "does not register". **It cannot be ranked
finely, and differences below about 0.027 are not resolvable.** This is a property of
having 5 channels, not of the floor, and no floor setting changes it.

## What this establishes

- **The reading rule exists and is measured, not argued.** Rule 8, from a 42x to 60x
  empty gap reproduced on three checkpoints.
- **The gate on `gate3_s42` is causally reachable**, replicated on 16 of 20 probe
  seeds with a live control and a determinism check at 0.000e+00 on every run.
- **The gate on `gate3_s43` and `gate3_s44` is not**, and this is not a floor
  artefact because their responses are at float32 resolution.
- **The default floor hid a real response**, which confirms reading rules 1 and 2.

## What this does NOT establish

- **It is not a claim about the architecture.** One checkpoint of three. The other
  two are inert at 20 seeds each, and why they differ is not measured.
- **It does not make PCI TRUSTED.** Nothing here shows a gate PCI carries information
  about the STIMULUS, which is the content-sensitivity clause. A response that
  replicates is not yet a response that means anything. **ANSWERED the same day and
  it FAILED: `pci_content_2026_09.md`.** eta-squared 0.060960 at the gate's raw
  response against a null mean of 0.061989, so the measured value is below what
  random labels give on average. The control fails too.
- **It does not explain the two regimes.** What makes probe seeds 43, 44, 45 and 53
  produce a noisier baseline on every checkpoint is not known.
- **It does not lift rule 3.** The controls here are 0.1640, 0.0487 and 0.0623, a
  3.4x spread, so absolute PCI still cannot be compared across checkpoints.
- **It does not answer planning #15**, which also needs the alternatives survey.

## Next

1. **Content-test the gate PCI on `gate3_s42`.** It is the only site with a
   replicated response, so it is the only one where the content question can be
   asked. Same eta-squared against a permutation null used elsewhere.
2. **Find what makes seeds 43, 44, 45 and 53 noisy.** It is reproducible across
   checkpoints, so it is a property of the rollout and should be findable.
3. **Ask why one checkpoint of three has a reachable gate.** All three share one
   training configuration.

## Reproduce

```
for S in 42 43 44 ; do
  python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 20 \
    --magnitude 1000 --load-tectum runs/gate3_s$S/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --var-floor 1e-6 --out runs/_pci_null/n20_s$S.csv
done
```

About 4 minutes per checkpoint, serial. Read the per-trial rows, not the mean: the
mean over 20 seeds averages two regimes. Sort by `median_baseline_sd` and the gap is
visible on sight.
