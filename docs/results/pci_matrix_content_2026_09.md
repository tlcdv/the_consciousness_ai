# FAILED, and NOT for the reason the other seven failed: the response matrix carries nothing either

**FAILED. The 5-by-60 response matrix that PCI compresses does not decode the
stimulus, so the scalar reduction is not what loses the content.** There is no
content in the perturbational response to lose.

**This SEPARATES PCI from the project's standing pattern rather than joining it.** At
seven other sites the vector carries the stimulus and the scalar reduction discards
it. Here the vector does not carry it either. PCI fails for a different reason, and
the hoped-for outcome (a) is refuted.

Pre-stated outcome (b), NOTHING CARRIES IT. The gate was written into
`scripts/analysis/probe_pci_matrix_content.py` before any value from this run was
read.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## What was run

120 probe seeds on `gate3_s42`, magnitude 1000, floor 1e-6. Pre-impulse divergence
was asserted to be exactly 0.0 at every seed and every site; the probe exits rather
than continue if it is not. The impulse lands in the DELAY phase on all 120 seeds, so
no result here can be a phase reading.

Rule 8 applied at the gate: 81 admitted, 39 rejected at a 60.2x gap. At the rssm rule
8 does not apply, largest gap 1.0x, so all 120 were admitted.

Three forms of the same response, decoded for 6-class `sample_shape` with
`StandardScaler` then `LogisticRegression`, scored by 5-fold stratified
cross-validation against a 200-permutation label null.

| Site | Form | Live features | CV accuracy | Null p95 | Uniform | Majority |
|---|---|---|---|---|---|---|
| gate | `response` (5x60) | 139 of 300 | 0.2096 | 0.2463 | 0.1667 | 0.2099 |
| gate | `binary` (5x60) | 13 of 300 | 0.1478 | 0.2471 | 0.1667 | 0.2099 |
| gate | `scalar` (pci) | 1 | 0.1838 | 0.2463 | 0.1667 | 0.2099 |
| rssm | `response` (64x60) | 3840 of 3840 | 0.1083 | 0.2333 | 0.1667 | 0.1917 |
| rssm | `binary` (64x60) | 419 of 3840 | 0.2000 | 0.2417 | 0.1667 | 0.1917 |
| rssm | `scalar` (pci) | 1 | 0.2083 | 0.2337 | 0.1667 | 0.1917 |

Nothing clears its null. The gate's continuous response reaches 0.2096 against a
majority-class rate of 0.2099, which is what a classifier that has learned nothing
returns.

## The power objection, and the retest that closes it

A negative from a model with 3840 features and 120 samples is weak evidence, and the
rssm row shows the overfitting directly: 0.1083, BELOW the 0.1667 uniform chance. So
the test was repeated with PCA in front of the classifier, which removes the
objection.

| Site | Form | Components | Accuracy | Null p95 |
|---|---|---|---|---|
| gate | `response` | 5 | 0.2228 | 0.2471 |
| gate | `response` | **10** | **0.2581** | **0.2463** |
| gate | `response` | 20 | 0.2213 | 0.2589 |
| gate | `binary` | 5 | 0.1971 | 0.2478 |
| gate | `binary` | 10 | 0.0868 | 0.2364 |
| rssm | `response` | 5 | 0.1583 | 0.2500 |
| rssm | `response` | 10 | 0.1333 | 0.2417 |
| rssm | `response` | 20 | 0.1333 | 0.2421 |
| rssm | `binary` | 5 | 0.1667 | 0.2337 |
| rssm | `binary` | 10 | 0.2083 | 0.2500 |
| rssm | `binary` | 20 | 0.1750 | 0.2500 |

**One cell of eleven crosses its p95, and it is NOT a finding.** The gate's continuous
response at 10 components reads 0.2581 against 0.2463. It does not hold at 5
components (0.2228) or at 20 (0.2213), and eleven comparisons at the 5 percent level
produce about 0.55 crossings by chance alone. A result that appears at exactly one of
three settings and vanishes on both sides of it is a multiple-comparison artefact.

It is written down here so that nobody re-derives it from the same data and reports it
as a near-miss or a weak positive. It is neither.

## What binarization does, separately from content

Binarization is not neutral, even though it is not where the content is lost. At the
gate it leaves **13 of 300 features varying, down from 139**. At the rssm it leaves
419 of 3840. So the thresholding step discards roughly 90 percent of the varying
entries before the LZ count is taken.

That is worth knowing about the instrument. It is not the explanation for the content
failure, because the 139-feature continuous matrix does not decode either.

## What this establishes

- **The causal response carries no stimulus identity, in any form tested**:
  continuous matrix, binary matrix, or scalar, at the primary site and at the control.
- **The scalar reduction is NOT the culprit for PCI.** This refutes the hypothesis
  this run was built to test, which was the useful outcome to rule out.
- **PCI's failure is different in kind from the other seven.** Elsewhere the content
  exists in the vector and the reduction throws it away. Here perturbing the recurrent
  state with random noise and watching the response says something about the
  dynamics at that moment, and nothing about what the network is currently holding.
- **Combined with `pci_content_2026_09.md`, the content clause is settled negatively
  by two independent methods**: eta-squared on the scalar, and decoding on the matrix.

## What this does NOT establish

- **It does not prove there is zero content.** A null decoding result bounds an
  effect, it does not exclude one. At n=81 with 6 classes this test would miss a weak
  effect. The claim is that no effect large enough to matter for an instrument is
  present.
- **One checkpoint.** `gate3_s42` is the only one with a response to test.
- **One label.** `sample_shape` during delay. Colour, size, and the choice phase are
  untested.
- **It does not say the perturbation is useless.** It propagates, it is replicable,
  and the control discriminates. It reports that something happened.
- **It does not retire PCI.** That is an owner decision.

## Next

1. **Decide PCI's status.** Its content clause is now settled negatively by two
   methods. The comparable measures that failed this clause were retired.
2. If PCI is kept for what it does measure, state in the inventory what that is: a
   propagation check on the dynamics, not a content measure, and not comparable to
   the human scale.
3. The remaining untested labels (colour, size) cost one analysis pass each on
   `runs/_pci_content/mat120_s42.npz`, which is on disk.

## Reproduce

```
python -m scripts.analysis.probe_pci_matrix_content --trials 120 \
  --load-tectum runs/gate3_s42/tectum.pt --latent-mode continuous \
  --capsule-workspace-source all_levels --var-floor 1e-6 \
  --npz runs/_pci_content/mat120_s42.npz
```

About 20 minutes to collect and 10 to score. Pass `--reuse` to re-score the saved
matrices without re-running the rollouts.
