# FAILED: the agent does not show the Vishne pattern. Its workspace broadcast holds a fixed code for the whole sample, and its tectum is sustained at only 1 of 3 seeds

**The pre-stated gate needed a SUSTAINED tectum and a TRANSIENT broadcast at all 3 seeds at
S = 20. The broadcast is SUSTAINED at 3 of 3 seeds, and the tectum is SUSTAINED at 1 of 3.**
The pattern Vishne, Gerber, Knight and Deouell (2023, Cell Reports 42, 112752) found in human
intracranial recordings, stable content in the ventral visual stream and a transient onset
burst in prefrontal cortex, is not present in the agent. The agent's layout is closer to the
reverse. The workspace broadcast carries one fixed code for as long as the sample is on.

## Question

Vishne et al. showed images for several durations and decoded their category over time. The
ventral-stream code was stable and lasted as long as the image. The frontoparietal code was a
short burst at onset, with no report required. COGITATE adopted the paradigm because it
separates an IIT-style posterior prediction from a GNWT-style prefrontal prediction. The
matching question for the agent is whether content is sustained in the tectum and transient in
the workspace broadcast.

## Instrument

`models/evaluation/temporal_generalization.py` (new, 4 tests in
`tests/test_temporal_generalization.py`). A logistic decoder is trained at step t and tested
at step t', with 5-fold cross-validation over trials, so a trial never sits in both folds.
The threshold is the p95 of the per-permutation maximum of the diagonal under labels shuffled
across trials, 20 permutations, so it controls the family of steps. `classify_dynamics` labels
a window SUSTAINED when at least 80 percent of its steps decode and a decoder from the second
window step carries to at least 80 percent of them, and TRANSIENT when the first 3 steps
decode and at most half of the rest do. Otherwise MIXED. Tests pin a stable code, a transient
code, shuffled labels at chance and both labels.

## Runs

`scripts/analysis/probe_sustained_vs_ignition.py`, read-only, DMTS with the sample shown for
S = 5, 10 and 20 steps, 60 trials per duration, 6-class `sample_shape`, epochs from 3 steps
before onset to 10 steps after offset, the capfix checkpoints (continuous latent, all-level
capsules), mock semantic on. A broadcast that cannot be computed raises; none did.

```
python -m scripts.analysis.probe_sustained_vs_ignition --runs-dir <runs> --trials 60
```

Uniform chance is 0.167.

| seed | S | stage | label | threshold | window diag min | window diag max | window steps above | after offset above |
|---|---|---|---|---|---|---|---|---|
| 42 | 5 | tectum | SUSTAINED | 0.284 | 0.333 | 0.383 | 5/5 | 0/10 |
| 42 | 5 | broadcast | SUSTAINED | 0.285 | 0.400 | 0.483 | 5/5 | 0/10 |
| 42 | 10 | tectum | MIXED | 0.301 | 0.300 | 0.400 | 9/10 | 0/10 |
| 42 | 10 | broadcast | SUSTAINED | 0.253 | 0.600 | 0.600 | 10/10 | 0/10 |
| 42 | 20 | tectum | MIXED | 0.300 | 0.267 | 0.333 | 8/20 | 0/10 |
| 42 | 20 | broadcast | SUSTAINED | 0.253 | 0.600 | 0.600 | 20/20 | 0/10 |
| 43 | 5 | tectum | SUSTAINED | 0.336 | 0.517 | 0.533 | 5/5 | 0/10 |
| 43 | 5 | broadcast | SUSTAINED | 0.303 | 0.400 | 0.400 | 5/5 | 0/10 |
| 43 | 10 | tectum | SUSTAINED | 0.317 | 0.517 | 0.583 | 10/10 | 0/10 |
| 43 | 10 | broadcast | SUSTAINED | 0.285 | 0.600 | 0.600 | 10/10 | 0/10 |
| 43 | 20 | tectum | SUSTAINED | 0.351 | 0.483 | 0.567 | 20/20 | 0/10 |
| 43 | 20 | broadcast | SUSTAINED | 0.285 | 0.600 | 0.600 | 20/20 | 0/10 |
| 44 | 5 | tectum | MIXED | 0.333 | 0.200 | 0.200 | 0/5 | 0/10 |
| 44 | 5 | broadcast | SUSTAINED | 0.286 | 0.433 | 0.433 | 5/5 | 0/10 |
| 44 | 10 | tectum | MIXED | 0.301 | 0.100 | 0.167 | 0/10 | 0/10 |
| 44 | 10 | broadcast | SUSTAINED | 0.302 | 0.550 | 0.550 | 10/10 | 0/10 |
| 44 | 20 | tectum | MIXED | 0.334 | 0.117 | 0.167 | 0/20 | 0/10 |
| 44 | 20 | broadcast | SUSTAINED | 0.302 | 0.550 | 0.550 | 20/20 | 0/10 |

Gate output at S = 20, per seed (tectum, broadcast): 42 (MIXED, SUSTAINED), 43 (SUSTAINED,
SUSTAINED), 44 (MIXED, SUSTAINED). FAILED.

## What is established

- The broadcast is SUSTAINED at every seed and every duration, 9 of 9. At S = 10 and S = 20
  its diagonal accuracy has the same value at every sample step (min equals max, 0.600, 0.600
  and 0.550 at S = 20), so the decoded broadcast code does not change while the sample is on.
- No stage decodes above threshold at any step after sample offset, at any seed or duration.
  Content does not outlast the stimulus in either stage under this single-step decoder.
- The tectum is SUSTAINED at seed 43 only. At seed 44 it decodes at or below uniform chance
  at every sample step.
- A transient, onset-only broadcast code, the signature GNWT predicts for the workspace, does
  not appear at any seed.

## Caveats

- 60 trials and 6 classes give about 10 trials per class and about 8 per class in each
  training fold, against feature dimensions in the hundreds. Power is low. The tectum
  decodability here is lower than in the perception-collapse work, which used a different
  protocol and far more samples, so the two are not compared.
- The threshold uses 20 permutations. The p95 of 20 values is coarse.
- The stimulus is a static DMTS shape. The Vishne stimuli were natural images of four
  categories, and no image content is shared.
- A flat, identical accuracy across the sample steps is what a latched workspace state would
  give. The probe does not test the mechanism, so that reading is a hypothesis.

## Consequence

The agent's workspace does not produce an ignition-style transient for content, the property
the GNWT side of the paradigm predicts. That bears on the ignition half of GWT-2 in
`docs/consciousness_indicators_butlin.md`, which the rubric already records as saturated. This
result does not change the rubric. It adds one measured fact against which any future ignition
claim can be checked.
