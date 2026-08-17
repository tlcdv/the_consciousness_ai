# FAILED: no weighted reduction of `kl_map` makes the vision bid track stimulus content

**Every candidate was rejected, at all 3 seeds, on a gate stated before any number was
read.** Four reductions were tested against the current `kl_map.sum()`. All four fix the
degeneracy: the bid stops being a constant. None of them makes the bid track which shape
was shown. Every candidate's variance-explained sits BELOW its own permutation null.

The model was not changed. No flag was added. The equal-weight sum stays.

Read-only probe on one trained checkpoint (`runs/gate_ckpt_s42`), 3 seeds, 40 episodes
each, 8000 steps per seed, dmts.

## Why this was attempted

[klmap_phase_information_2026_08.md](klmap_phase_information_2026_08.md) established that
the 262,144-element `kl_map` carries stimulus identity at close to its own ceiling, and
that `sensory_tectum.py:444` reduces it with an equal-weight sum whose output saturates
`tanh` to exactly 1.0. The question was whether a different, non-uniform reduction could
carry some of that content into the bid.

## The candidates, all unsupervised

No training signal for salience exists, so every candidate had to be computable from the
data itself. That constraint ruled out a learned weighting from the start.

| id | reduction | weighting |
|---|---|---|
| S0 | `tanh(kl_map.sum())` | current baseline, equal weight |
| S1 | `tanh(kl_map.mean())` | equal weight, de-saturates only |
| S2 | per-element running z, `sigmoid(mean(z))` | non-uniform |
| S3 | per-element running z, `tanh(mean(abs(z)))` | non-uniform |
| S4 | per-element running z, `sigmoid(mean(top 1 percent))` | non-uniform |

Running statistics are a causal EMA over past steps only, updated after the current step
is read. A full-sample statistic is not computable in a forward pass, so scoring against
one would have measured something unimplementable.

## Clause 1 and 2: every alternative fixes the degeneracy

Pre-stated: at least 100 distinct values, span at least 0.2, inside [0, 1], at most 1
percent of steps pinned at a bound. Values below are seed 42; seeds 43 and 44 agree to
three decimals.

| candidate | min | max | distinct | pinned | verdict |
|---|---|---|---|---|---|
| **S0 current** | 1.000000 | 1.000000 | **1** | **100.0%** | **FAIL** |
| S1 | 0.007967 | 0.868862 | 4652 | 0.0% | PASS |
| S2 | 0.000000 | 0.767283 | 7687 | 0.4% | PASS |
| S3 | 0.000000 | 1.000000 | 7925 | 0.5% | PASS |
| S4 | 0.500000 | 1.000000 | 7923 | 0.8% | PASS |

The current bid fails its own non-degeneracy check at 3 seeds, as a pass/fail in code
rather than as prose. That is clause 1 of the acceptance bar applied to the bid itself.

## Clause 3: FAILED. None of them tracks stimulus shape

Scored as eta-squared, the fraction of the bid's variance explained by `sample_shape`,
on sample-phase steps. The null shuffles shape labels ACROSS TRIALS, 100 times, because
shape is constant for all ~20 steps of a trial and shuffling steps would manufacture a
falsely low floor. The bar is the 95th percentile of each candidate's own null.

| seed | trials | S1 | S2 | S3 | S4 | null p95 |
|---|---|---|---|---|---|---|
| 42 | 115 | 0.0348 | 0.0350 | 0.0345 | 0.0381 | ~0.083 to 0.092 |
| 43 | 114 | 0.0208 | 0.0364 | 0.0298 | 0.0091 | ~0.079 to 0.092 |
| 44 | 114 | 0.0137 | 0.0260 | 0.0267 | 0.0412 | ~0.085 to 0.102 |

Every value is below its own null, at every seed. Shape explains roughly 1 to 4 percent
of each bid's variance where a RANDOM regrouping of trials explains about 9 percent.

The candidates are not merely failing to reach significance. They sit at the low end of
their own null distributions. The bids vary a great deal, up to 7925 distinct values, and
that variation is dominated by within-trial dynamics rather than by which stimulus is on
screen.

## What this settles

**A scalar bid cannot carry the content that `kl_map` holds, at least not by any of these
reductions.** The map decodes shape at 0.84 / 0.71 / 0.76 with 256 features. Reduced to
one number, that content does not survive, and the earlier result should not be read as
implying it would.

This was flagged before the run and the measurement confirms it: a bid is one dimension,
and stimulus identity reaches the workspace through the payload, not through the bid.

## What this does NOT settle

- **It does not show the bid is worthless.** A bid signalling "how surprising is this
  moment" could be useful without tracking shape. This probe scored content-tracking,
  because shape is the only clock-free task label available, and it did not score
  salience usefulness, for which no ground truth exists here.
- **It does not vindicate the current bid.** S0 fails clause 1 outright at 3 seeds. The
  degeneracy is real and every alternative removes it. What the alternatives do not buy
  is content.
- **It does not exhaust the space of reductions.** Four unsupervised candidates were
  tested. A learned weighting was excluded by design because there is no training signal
  for salience, and inventing a proxy objective is a much larger bet than this evidence
  supports.
- **Phase-tracking remains undemonstrated.** Step index alone decodes sample versus delay
  at 1.0000, so no bid change can be presented as producing task-driven selection.

## Two candidate formulations were corrected before the scoring run

Recorded because correcting candidates mid-investigation can hide selection.

A smoke run showed S2 using `tanh` on a signed z-score, which goes negative and violates
the bid range requirement, and S4 dividing a top-1-percent mean by the overall mean, which
lands far outside `tanh`'s responsive range and pinned it at 1.0. Both were corrected to
`sigmoid` of a z-score before the scoring run.

Both corrections fix violations of clause 1 and 2, which were stated in advance and are
independent of the outcome metric. Neither was motivated by how a candidate scored on
eta-squared. The distinction matters: tuning candidates on the outcome metric would be
selection and would invalidate the run.

## Consequence

Planning #22 asked whether to replace the saturated bid, and the owner chose the weighted
reduction over the cheap `mean()`. **On this evidence the weighted reduction does not earn
its cost.** It removes the degeneracy, and so does the one-line `mean()`, and neither
delivers the content that motivated preferring the weighted version.

What remains true and unaddressed: the vision bid is a constant, the workspace competition
never switches module, and both bear on GWT-1 and GWT-2.

## Reproduce

```
python -m scripts.analysis.probe_bid_reduction_candidates \
  --load-tectum runs/gate_ckpt_s42/tectum.pt --episodes 40
```
