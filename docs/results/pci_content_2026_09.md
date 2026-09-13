# FAILED. A gate-level PCI does not carry the stimulus, and neither does the control's

**FAILED. The gate's causal response does not depend on which shape the system is
holding.** eta-squared for 6-class `sample_shape` is 0.060960 at the gate's raw
response, against a permutation null whose p95 is 0.134895 and whose MEAN is
0.061989. **The measured value sits below the null mean.** Random labels explain more
of the variance, on average, than the true labels do.

**The control fails the same test.** The rssm reads eta-squared 0.078322 against a
null p95 of 0.088844. So this is not a fact about the gate site. It is a fact about
the measure.

This is outcome (c) of the pre-stated gate, VARYING BUT CONTENTLESS, at every site
and on both quantities tested.

**Consequence: PCI cannot become TRUSTED.** Content sensitivity is the clause this
was run to settle, and PCI fails it explicitly. PCI is the eighth scalar on this
architecture to vary convincingly and carry nothing.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## What was run

120 probe seeds on `gate3_s42`, the only checkpoint with a replicated gate response
(`pci_probe_seed_null_2026_09.md`). Magnitude 1000, variance floor 1e-6, determinism
check passed at pre-impulse divergence 0.000e+00 on every run, every site
`[trained]`, no DISQUALIFIED block.

The gate was pre-stated in `scripts/analysis/probe_pci_content.py` before any value
from this run was read.

**The phase trap is structurally excluded.** The impulse at step 40 lands in the
DELAY phase on all 120 seeds, verified over 200 seeds before the probe was written.
Step index alone decodes sample-versus-delay at 1.0000, so a result that could be a
phase reading is worthless. Here the phase does not vary, so there is nothing to read.
Only the held shape varies.

Shape classes across 120 seeds: circle 20, hexagon 23, pentagon 20, square 15, star
20, triangle 22.

**Rule 8 was applied.** At the gate, the baseline split at 1.9569e-05 with a 60.3x
gap: 81 seeds admitted, 39 rejected. At the rssm, rule 8 does not apply, because the
largest baseline gap is 1.0x, so all 120 were admitted.

## Result

Shuffling is across SEEDS, because one seed is one trial and the shape is constant
within it. 2000 permutations.

| Site | Quantity | n | eta-squared | null p95 | null mean | Verdict |
|---|---|---|---|---|---|---|
| **gate** | `pci` | 81 | 0.113845 | 0.131807 | 0.063459 | FAIL |
| **gate** | `max_abs_response` | 81 | **0.060960** | 0.134895 | **0.061989** | FAIL, below the null MEAN |
| rssm (control) | `pci` | 120 | 0.073675 | 0.091290 | 0.042115 | FAIL |
| rssm (control) | `max_abs_response` | 120 | 0.078322 | 0.088844 | 0.041803 | FAIL |
| broadcast | `pci` | 81 | 0.049327 | 0.140275 | 0.062697 | FAIL |
| broadcast | `max_abs_response` | 81 | 0.066475 | 0.138336 | 0.061859 | FAIL |

Neither quantity clears its null at any site. The raw response was included because it
is continuous and therefore the stronger test: 70 distinct values across 81 readings,
spanning 1.481e-05 to 7.135e-05. It is the one that lands exactly on its null mean.

**The gate `pci` value of 0.113845 is the closest any row comes**, at 86 percent of
its null p95. It does not clear the bar, and on a quantity with 6 possible values it
is the weaker of the two tests. It is recorded so that nobody re-derives it as a
near-miss and treats it as a result.

## The null behaves exactly as theory says

For k classes and n readings, eta-squared from random labels has an expected value of
(k-1)/(n-1). Measured against that:

| n | k | Predicted null mean | Measured null mean |
|---|---|---|---|
| 81 | 6 | 0.0625 | 0.0635, 0.0620, 0.0627, 0.0619 |
| 120 | 6 | 0.0420 | 0.0421, 0.0418 |

This is why the earlier 20-seed reading could not have decided anything: at n=16 the
null mean is 0.333 and the p95 is 0.62, so nothing short of an overwhelming effect
could clear it. The 120-seed run brings the bar down to about 0.13.

## A correction to the previous verdict

`pci_probe_seed_null_2026_09.md` said gate PCI has "four possible values", from LZ
complexity taking only 2, 4, 5 and 6. That was accurate for the 60 readings it had
and wrong as a general statement. Over 360 readings the gate LZ takes **2, 4, 5, 6, 7
and 8**, giving six PCI values: 0.0, 0.109718, 0.137147, 0.164576, 0.192006 and
0.219435.

The point that stands is unchanged and is why it was raised: gate PCI is coarsely
quantized, with a step of 0.027429, so it cannot be ranked finely. The rssm takes 14
values and the broadcast 43, so the coarseness is a property of having 5 channels.

## What this establishes

- **A gate-level PCI does not carry stimulus identity**, at 81 readings with the
  phase held constant and the baseline regime controlled.
- **Neither does the control's**, so the failure belongs to the measure and not to
  the site. A perturbation that propagates is not a perturbation that carries.
- **PCI fails the content clause**, which is the clause that stood between it and
  TRUSTED. It cannot be promoted on the present evidence.
- **The pattern now holds at 8 cases.** Every scalar reduction tested on this
  architecture varies and carries nothing, while the vectors carry the stimulus.

## What this does NOT establish

- **It does not say the gate response is an artefact.** It is real and replicated at
  16 of 20 and 81 of 120 seeds, with a live control and determinism at 0.000e+00. It
  signals that something happened. It does not signal what.
- **It does not test a vector-valued PCI.** The test here is on the scalar the
  instrument reports. Whether the 5-channel binary response MATRIX carries the shape
  is a different question and is not answered.
- **One checkpoint.** `gate3_s42` is the only one with a response to test.
- **One label.** `sample_shape`, 6 classes, during delay. Colour, size and the
  choice phase are not tested.
- **It does not retire PCI.** That is an owner decision. What it does is remove the
  path by which PCI could have become TRUSTED.

## Next

1. **Test the response MATRIX, not the scalar.** The binary 5-by-60 response matrix
   is what PCI compresses. If the matrix carries the shape and the scalar does not,
   that is the same finding this project already has at 7 other sites, and it would
   make 8 a pattern with a stated mechanism rather than a tally.
2. **Decide PCI's status.** It fails the content clause. The comparable measures that
   failed it were retired.
3. Colour and size as labels, which cost one analysis pass each on the same CSV.

## Reproduce

```
python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 120 \
  --magnitude 1000 --load-tectum runs/gate3_s42/tectum.pt \
  --latent-mode continuous --capsule-workspace-source all_levels \
  --var-floor 1e-6 --out runs/_pci_content/c120_s42.csv
python -m scripts.analysis.probe_pci_content --csv runs/_pci_content/c120_s42.csv
```

About 25 minutes for the probe, seconds for the analysis. The analysis refuses to run
on any CSV with a row that is not from trained weights.
