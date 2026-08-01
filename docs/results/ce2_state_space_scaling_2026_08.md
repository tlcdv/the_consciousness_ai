# CE 2.0 across state-space sizes: FAILED, `ce2_ratio` is confounded by cardinality

**Verdict: FAILED.** CE 2.0 is not comparable across state-space sizes. Holding the
macro structure fixed and varying only the number of microstates moves CE 2.0 from
0.857143 at 8 states to 0.995868 at 243 states. `ce2_ratio` therefore divides two
quantities that live on different scales, and a ratio away from 1.0 can be produced
by state count alone, with no difference in macro structure at all.

This kills CE2-1 as written. The prediction reads `ce2_workspace > ce2_gates` off an
8-state level against a 243-state level, and the cardinality artifact alone pushes
that comparison to 0.860699 for structurally identical systems.

No indicator moves. No consciousness claim. This is an instrument verdict.

## Why no seeds are reported

This is deterministic linear algebra on constructed TPMs, not a sampled experiment.
There is no RNG anywhere in the probe and no run-to-run variation to average over.
Every number below is exact to 1e-9 against a closed form. The 3-seed rule exists
for stochastic measurements and does not apply here. Nothing in this document rests
on a trained checkpoint.

## Method

Hold the macro structure fixed at k equivalency classes, the structure CE 2.0 claims
to detect, and vary only n. A metric comparable across cardinality returns the same
value for the same macro structure at any n.

`scripts/analysis/probe_ce2_state_space_scaling.py`, run 2026-08-01. It calls the
existing `compute_ce2_from_tpm`; no metric code was reimplemented.

## Pre-stated gate

Written into the probe docstring before any number from it was read, following the
convention in [gate_binning_2026_07.md](gate_binning_2026_07.md):

> PASS, comparable: at fixed k, `|CE(n=243) - CE(n=8)| <= 0.05`.
> FAIL, confounded: at fixed k, `|CE(n=243) - CE(n=8)| > 0.05`.

The 0.05 tolerance was set against the effect CE2-1 must detect. The 2026-07 pilot
logged `ce2_ratio` at 0.7551, a claimed 24 percent departure from parity, so a
cardinality artifact of comparable size would be indistinguishable from the signal.

## Result: FAIL, by a wide margin

Uniform blocks, k=2 held fixed:

| n states | CE 2.0 | closed form | emergent complexity |
|---------:|-------:|------------:|--------------------:|
| 8 | 0.857143 | 0.857143 | 1 |
| 16 | 0.933333 | 0.933333 | 1 |
| 32 | 0.967742 | 0.967742 | 1 |
| 64 | 0.984127 | 0.984127 | 1 |
| 128 | 0.992126 | 0.992126 | 1 |
| 243 | 0.995868 | 0.995868 | 1 |

- spread across the two cardinalities the training loop divides: **0.138725**
- implied `ce2_ratio` for **identical** macro structure: **0.860699**
- gate: `0.138725 <= 0.05` is false, so **FAIL**

Same direction at k=3 (0.714286 to 0.991736) and k=4 (0.571429 to 0.987603), and in
a leaky variant softened toward all-to-all uniform, so this is not a knife-edge of
exactly block-diagonal matrices. Leaky k=2 at leak 0.1 runs 0.771429 to 0.896281,
and at leak 0.5 runs 0.428571 to 0.497934, both rising monotonically with n.

## The closed form, and why the metric behaves this way

For k uniform blocks over n states the TPM has k singular values equal to 1 and n-k
equal to 0. Discarding the trivial sigma_1 gives `gamma* = (k-1)/(n-1)` and
`sigma_2 = 1`, so

    CE(n, k) = 1 - (k-1)/(n-1)

Independent of block sizes. Every measured value above matches it exactly; the probe
raises rather than reports if any value departs by more than 1e-9. Two pre-existing
tests pin the same identity independently, asserting CE = 6/7 at (n=8, k=2) and
CE = 0.75 at (n=9, k=3).

The mechanism is visible in the formula. `gamma*` is a mean over n-1 values and so
falls as n grows, while `sigma_2` is a single raw singular value that does not. Their
difference therefore rises toward 1 with cardinality, for a macro structure that has
not changed.

## What the source says, checked independently

Both papers were read directly. Appendix S3 was reached in full via the arXiv HTML
rendering and read verbatim. The key statement below was re-fetched and confirmed
against the source a second time rather than accepted from a single reading.

- **Hoel 2025 S3 states no comparability or normalization condition for state
  count.** S3 never compares two systems of different cardinality. The only
  normalization in the paper is for path length, on a different quantity: "these
  values in turn can be normalized by log2(L)".
- **The parent method it adapts states the size dependence explicitly, and it points
  against the ratio.** Zhang et al. 2025 (Hoel's reference [22], arXiv:2402.15054v5),
  section "Normalization and Examples": "Since Gamma_alpha is size-dependent, we need
  to normalize them by dividing the size of P", defining gamma_alpha = Gamma_alpha/N,
  so that "comparisons between Markov chains with different sizes are more
  reasonable". Verified by direct fetch on 2026-08-01.
- **The dropped log2(n) term is a red herring here.** Hoel removes it deliberately
  ("the size term is rendered unnecessary"), it was never part of the SVD heuristic,
  and where it survives in the causal primitives it is a per-system denominator, not
  a cross-system normalizer. Restoring it would not make the ratio valid.
- **`gamma*` and `CE = sigma_2 - gamma*` match the source exactly.** No local
  departure was found in the definitions.

So the empirical result and the literature agree, by separate routes.

## What survives: emergent complexity is cardinality-invariant

`emergent_complexity` counts singular values above `gamma*`, which stays at k-1 at
every n tested (verified at n = 8, 32, 243 for k = 2, 3, 4). It reads the macro
structure and ignores the microstate count, which is exactly the property
`causal_emergence` lacks. If any CE 2.0 channel is worth keeping for cross-level
comparison, current evidence points at this one rather than at the headline value.

Not yet established: whether complexity stays invariant on empirically estimated,
partially degenerate TPMs. Everything above is on constructed matrices.

## What this does NOT show

- It does not show CE 2.0 is meaningless within a single level at a fixed state
  count. That question is open and is tracked separately.
- It does not rescue or condemn `ce2_gates` or `ce2_workspace` individually. It
  concerns their comparison.
- It does not test the trained tectum, any checkpoint, or any real trajectory. The
  degeneracy confound documented in
  [ce2_pilot_calibration_2026_07.md](ce2_pilot_calibration_2026_07.md) is a second,
  independent problem and is untouched here.
- It does not derive a corrected normalization. Dividing by N in Zhang's manner is a
  candidate, but it would be a local extension of the method and would need its own
  validation before any number from it could be quoted.

## Consequences

- **CE2-1 (onset)** cannot be read from `ce2_ratio` as written. It needs either a
  validated cross-cardinality normalization or a restatement that does not compare
  levels of different size.
- **CE2-2 (magnitude band)** stays deferred. Banding `ce2_ratio` would band an
  artifact, now for a second and independent reason.
- `ce2_ratio` and `ce2_emergent` remain logged. They should not be cited until the
  above is resolved. No change to their default-off status was made here.

## Changes

- `scripts/analysis/probe_ce2_state_space_scaling.py` (new): the scan, the closed
  form, and the pre-stated gate.
- `tests/test_causal_emergence_svd.py`: `TestStateSpaceSizeConfound`, 10 tests
  pinning the closed form at both cardinalities, the spread, the 0.860699 identical
  structure ratio, and the cardinality invariance of complexity.
- Suite: 908 passed, 5 skipped (was 898 passed, 5 skipped).
