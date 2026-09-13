# RPT-2's negative evidence was never eligible for acceptance under its own pre-registration

**RPT-2 is scored PARTIAL because "the in-training phi-binding coupling (Phi-1) FAILED
across 9 runs". The pre-registration that defined Phi-1 requires non-degenerate
variances before a negative result may be accepted, and 0 of the 9 runs met that
condition.** Every run recorded a `sync_R_std` at or below 0.019 against a
pre-registered floor of 0.02.

**RPT-2 does NOT become IMPLEMENTED.** Removing invalid negative evidence leaves no
evidence, not positive evidence. The status stays PARTIAL. What changes is the
RATIONALE, and the change matters because the two say different things about what
work is needed: "measured and failed" closes a question, "never validly measured"
does not.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## The pre-registration's own terms

`docs/preregistered_predictions.md`, line 598:

> Variance floors: phi_std > 0.01, sync_R_std > 0.02 (degenerate variance triggers a
> re-run, not threshold revision).

And line 599:

> Falsification: r < 0.15 on BOTH pathways WITH NON-DEGENERATE VARIANCES triggers
> acceptance of the negative result.

Acceptance of the negative is conditional on non-degenerate variances. That condition
was never met.

## All 9 runs, from the campaign's own summary table

Reproduced from `phi1_phaseBalt_2026_05_24.md`, which lists the whole campaign.

| Date | Architecture | Pathway | `phi_std` | clears 0.01 | `sync_R_std` | clears 0.02 |
|---|---|---|---|---|---|---|
| 2026-05-14 | AKOrN, 5 ablations | pyphi | varies | not stated | low | **no** |
| 2026-05-16 | AKOrN | RIIU | 5.55e-05 | no | low | **no** |
| 2026-05-17 | AKOrN | RIIU x3 | 7.13e-05 | no | low | **no** |
| 2026-05-17 | AKOrN + A+C+D | pyphi | 1.84e-03 | no | 0.019 | **no** |
| 2026-05-18 | AKOrN + A+C+D | pyphi | 2.05e-03 | no | 0.019 | **no** |
| 2026-05-18 | AKOrN + A+C+D | RIIU | 1.99e-02 | **yes** | 0.019 | **no** |
| 2026-05-19 | AKOrN + A+B+C+D | pyphi | 1.67e-03 | no | 0.016 | **no** |
| 2026-05-19 | AKOrN + A+B+C+D | RIIU | 8.22e-03 | no | 0.016 | **no** |
| 2026-05-24 | KomplexNet + A+C+D | pyphi | 1.93e-03 | no | 0.016 | **no** |

**0 of 9 clear both floors. 0 of 9 clear the `sync_R` floor. 1 of 9 clears the `phi`
floor.**

## The original verdicts said this. The summary did not.

This is not a discovery about the runs. The run documents recorded it honestly and at
the time:

- `phi1_retest_2026_05_17.md`: "variance below pre-registered floor: phi_std=1.843e-03
  (floor 0.01), sync_R_std=1.933e-02 (floor 0.02). Plan section 10 says: degenerate
  variance triggers a re-run, not threshold revision."
- `phi1_phaseB_2026_05_19.md`: the same, with its own numbers.
- `phi1_phaseBalt_2026_05_24.md`: "Strict mechanical verdict on the pre-registered
  pyphi pathway: RE-RUN (variances below floor)."

**The protocol's verdict on these runs is RE-RUN, not FAIL.** The error is in the
rubric's one-line summary of them, which reads "FAILED across 9 runs". The underlying
work was reported correctly and the summary drifted from it.

## Why the variance was always degenerate, which was not knowable at the time

The cause was established on 2026-09-02, more than three months after the campaign
ended, in `sync_r_content_2026_09.md`: the Kuramoto oscillators converge to full
synchrony and stay there, because nothing resets their phase. At full synchrony the
order parameter equals the mean of the module bids exactly, and the bids are
separately measured degenerate.

**Verified in the current code.** `reset_state` is defined at
`models/core/oscillatory_binding.py:149`. The only `reset_state` call in
`scripts/training/train_rlhf.py` is `tectum.reset_state(1)` at line 423, which is a
different object. The binding layer's phases are never reset during training.

So `sync_R_std` could not have cleared its floor in any training run of this
architecture. The campaign was testing a correlation against a quantity that the
architecture pins.

## A second, independent problem with the rationale

RPT-2's stated reason ends: "not yet demonstrated as a measured phi signature during
training". The rubric's own definition of IMPLEMENTED, settled 2026-08-11 and in the
same file, says:

> A measured signature during training is not necessary. It is one strong kind of
> evidence, not the definition. The source requires no training-time measurement.
> Demanding one is a stricter local standard, and a local standard must not be
> presented as the source's.

RPT-2's rationale applies exactly the standard that paragraph retired. The rationale
predates the settlement and was not revisited when the definition changed.

## What this does NOT establish

- **RPT-2 is not IMPLEMENTED.** Invalid negative evidence removed leaves nothing, not
  a positive. The rubric also states that presence of a mechanism is not sufficient,
  so "the mechanisms are implemented" does not carry it either.
- **It does not say binding and integration ARE coupled.** It says the project has not
  validly tested whether they are.
- **It does not rehabilitate `phi` or `sync_R`.** Both are separately measured
  contentless (`scalar_content_2026_09.md`, `sync_r_content_2026_09.md`). A valid
  re-test of Phi-1 would need a binding measure and an integration measure that each
  pass the content clause, and neither exists today.
- **It does not audit the other indicators.** GWT-1 is also flagged for review and is
  not examined here.

## What partial positive evidence exists, and what it covers

RPT-2 reads "organized, integrated perceptual representations". The two halves are
not equally supported.

| Half | Evidence | Status |
|---|---|---|
| organized | `obs_map` decodes 6-class shape at ~1.0; `kl_map` at 0.84 / 0.71 / 0.76; the 256-D broadcast at 0.76 / 0.69 / 0.77 | measured, 3 seeds |
| integrated | none valid | **not measured** |

The decoding evidence is real and was never weighed against RPT-2, because the score
was tied to the phi-binding test. It supports the "organized" half at 3 seeds. It says
nothing about "integrated".

## What the rationale should say

Status unchanged at PARTIAL. The rationale is replaced with what the evidence
supports: the organized half is measured, the integrated half has never been validly
measured, and the Phi-1 campaign did not test it because every run failed the
pre-registered variance precondition for a reason the architecture guarantees.

## The owner decision that remains

Whether the decoding evidence is sufficient for the "organized" half to count toward
RPT-2 is a scoring judgement and is NOT taken here. It is recorded so the fork is
visible. Moving RPT-2 would move the clock, so it stays with the owner.

## Next

1. A valid Phi-1 re-test needs a binding measure that varies and carries content.
   `sync_R` cannot be it while `reset_state` is never called. Calling it is a
   one-line change whose effect on the measure is unknown and would need its own run.
2. The same audit applied to GWT-1, which is separately flagged for review.
3. Decide the "organized" half question above.
