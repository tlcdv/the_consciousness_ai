# The vision bid throws away task content that the tectum already has

**The defect: `sensory_tectum.py:444` sums a 262,144-element prediction-error map to one
scalar, and that sum destroys stimulus identity the map carries.** The pre-sum map decodes
which of 6 shapes was shown at 0.84, 0.71, 0.76 against majority baselines of 0.29, 0.22,
0.19, with a shuffled floor at 0.18, 0.17, 0.20. The summed scalar, which is the only
thing the workspace bid ever sees, decodes task phase at chance.

The information is present, one line of arithmetic discards it, and the workspace
competition downstream is left with a constant. This is the loss point.

Read-only probe on one trained checkpoint (`runs/gate_ckpt_s42`), 3 seeds, 40 episodes
each, 8000 steps per seed, dmts. No model changed, no indicator re-scored.

## What was asked

The workspace-competition verdict
([workspace_competition_2026_08.md](workspace_competition_2026_08.md)) found the bid
pinned at exactly 1.0 and the summed KL unable to separate sample from delay. Summing
262,144 elements cancels structure, so the open question was whether the map before the
sum carries what the scalar does not.

## Fidelity, checked before anything else

`kl_map` is recomputed in the probe from the tectum's cached `_last_post_logits`,
`_last_prior_logits` and `rssm.cont_logvar`, reproducing `sensory_tectum.py:443` rather
than approximating it. The recomputed `kl_map.sum() / batch` is compared against the value
the tectum actually squashed:

| seed | n | worst relative error |
|---|---|---|
| 42 | 8000 | 0.000e+00 |
| 43 | 8000 | 0.000e+00 |
| 44 | 8000 | 0.000e+00 |

Exact at every step. Every number below is gated on this.

## The phase test is confounded, by construction, and was known to be before it ran

DMTS uses `fixation_steps=10` and `sample_steps=20`, so the sample-to-delay boundary sits
at a fixed step index of 30 in every trial. Sample versus delay is a deterministic
function of elapsed time. Measured: **step index alone, one feature, decodes it at
1.0000** at every seed and every split.

So the phase numbers cannot distinguish task content from a clock, and a recurrent RSSM
drifting with time is exactly what one expects. Reported for completeness, on the
trial-grouped split, as accuracy over the majority baseline:

| feature | dim | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| A `kl_map.sum()`, the bid input | 1 | +0.112 | +0.000 | +0.000 |
| B1 per-channel pooled map | 1024 | +0.421 | +0.410 | +0.472 |
| B2 per-position pooled map | 256 | +0.403 | +0.405 | +0.450 |
| C `obs_map` reference | 64 | +0.448 | +0.431 | +0.450 |
| D shuffled labels | 1024 | -0.009 | -0.045 | -0.003 |
| **T step index, clock** | **1** | **+0.478** | **+0.445** | **+0.487** |

The gap between A and B1 is the point: the map reaches 0.94 to 0.98 absolute where the
scalar sits at chance. What that gap means required a test a clock cannot pass.

## The clock-free test: stimulus identity

`sample_shape` is drawn at random per trial from 6 shapes (`dmts_env.py:193`), so it is
independent of step index. Decoded from the sample-phase steps only, on the
trial-grouped split, with space preserved because shape is a spatial property:

| seed | held-out trials | majority | B2 `kl_map` | C2 `obs_map` ceiling | D2 shuffled |
|---|---|---|---|---|---|
| 42 | 31 | 0.2873 | **0.8399** | 0.8430 | 0.1821 |
| 43 | 31 | 0.2196 | **0.7131** | 0.7131 | 0.1683 |
| 44 | 30 | 0.1893 | **0.7621** | 0.8463 | 0.2039 |

**`kl_map` matches or nearly matches its own upper reference at all 3 seeds**, and at seed
43 it equals it to four decimals. The shuffled floor sits at or below the majority
baseline everywhere, so this is not overfitting.

A clock cannot decode which shape was on screen. This is content.

## Two methodological failures on the way here, both mine

Recorded because the wrong numbers were persuasive and would have been published.

**1. A random train/test split leaks, and the shuffle control does not catch it.**
Consecutive steps in one phase are near-duplicates, and `sample_shape` is constant across
all ~20 steps of a trial, so a random split puts a near-twin of nearly every test sample
into training. Under that split identity read **1.0000** at all 3 seeds. Under whole-trial
holdout it reads 0.84, 0.71, 0.76.

The shuffled control cannot detect this and the reason is worth stating: shuffling gives
near-twins DIFFERENT labels, so memorizing stops paying; with real labels the twins share
a label and it pays. The control and the leak point in opposite directions. Only a
grouped split fixes it.

**2. The first grouped run was underpowered and used the wrong features, and reported a
false negative.** At 10 episodes the identity test had about 7 held-out trials for a
6-class problem, and the majority baseline swung between 0.22 and 0.67 across seeds.
Identity read -0.394, +0.112, +0.000 and looked like a failure to replicate. It was
noise. Separately, that run decoded identity from B1, which pools the spatial grid away,
and used an `obs_map` reference that pools it away too, so the ceiling scored -0.409. A
reference below chance means the reference is broken, not that the model lacks the
information.

Raising to 40 episodes (about 31 held-out trials) and preserving space fixed both.

A third fault was caught by an impossible-looking output rather than by review: the
grouped split originally took the last 30 percent of the sorted group list, and
`np.unique` sorts `"episode:trial"` lexicographically, so "held-out trials" were exactly
the trials of the last episodes. Trial grouping and episode grouping printed identical
numbers at all 3 seeds, which is what exposed it. Groups are now drawn at random.

## What this settles

**The bid formula is the loss point, not the representation.** The tectum's prediction
error retains stimulus identity at close to the ceiling set by its own input map. The sum
at `sensory_tectum.py:444` discards it before anything downstream can use it.

This is narrower and more useful than the earlier reading. The workspace competition is
not starved of information by a bottleneck somewhere upstream. It is starved by one
reduction, at a known line.

Not shown:

- **Phase-tracking selection is NOT demonstrated.** The phase decode is fully explained by
  a clock. What is demonstrated is that content survives to `kl_map`, which is the
  precondition for a content-driven bid, not the thing itself.
- **No bid was changed and none is proposed here.** A bid is a scalar, and a linear probe
  with 256 free weights is not a bid. What the result shows is that a weighted reduction
  of `kl_map` can preserve identity where the equal-weight reduction (the sum) does not.
  Where those weights would come from is an open design question.
- **A linear probe is a lower bound.** It shows a linear function suffices. It does not
  establish that the current architecture would learn one.
- **One checkpoint, 3 probe seeds.** This measures probe variance, not variance across
  independently trained systems.
- **Six-class accuracy on about 31 held-out trials** is roughly 5 trials per class. The
  effect is large and consistent, and the interval around each number is wide.

## Reproduce

```
python -m scripts.analysis.probe_klmap_phase_information \
  --load-tectum runs/gate_ckpt_s42/tectum.pt --episodes 40
```
