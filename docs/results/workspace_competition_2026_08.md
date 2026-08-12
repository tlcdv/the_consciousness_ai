# The workspace competition FAILED: one module wins 400 of 400 steps in every phase

**Result: DEGENERATE at 3 seeds, on the pre-stated gate.** Vision wins 100.0 percent of
steps in fixation, sample, delay and choice. The raw vision bid is exactly
`1.000000000` at every one of 1200 measured steps. The selection mechanism selects
nothing, and this bears directly on GWT-1 and GWT-2, both currently marked IMPLEMENTED.

The cause is not the one the investigation was built to find. The signal upstream of the
bid varies by a factor of 167. `torch.tanh` deletes all of it.

Read-only probe on one trained checkpoint (`runs/gate_ckpt_s42`), 3 seeds, 2 episodes
each, dmts. No indicator is re-scored here.

## The gate, stated before any number was read

- **DEGENERATE**: one module wins in >= 95 percent of steps in EVERY phase, delay
  included.
- **PHASE-TRACKING**: the modal winner differs between sample and delay.
- **WEAK**: the winner varies but does not track phase.

DMTS makes this decidable. During the delay there is no sample on screen and the task
depends on held information, so a workspace doing state-dependent attention should
plausibly favour memory there and vision during sample and choice. The question is not
"is vision too strong", which a visual task does not settle. It is "does the winner
track task phase".

## Result

Winner share, per phase, per seed. 400 steps per seed.

| seed | fixation | sample | delay | choice | mean 1st-2nd margin |
|---|---|---|---|---|---|
| 42 | vision 100.0% | vision 100.0% | vision 100.0% | vision 100.0% | 1.315 |
| 43 | vision 100.0% | vision 100.0% | vision 100.0% | vision 100.0% | 1.315 |
| 44 | vision 100.0% | vision 100.0% | vision 100.0% | vision 100.0% | 1.312 |

**DEGENERATE at all 3 seeds.**

## The competition is decided by arithmetic, before any module competes

`raw_bids["vision"]` is `min(1.0, vision_bid)` in both the probe and the training loop.
The other bids have hard ceilings written into `train_rlhf.py:1056-1073`: memory is
`min(0.6, 0.1 + retrieval_score * 0.5)` and body is a two-valued switch, `0.15` when
interoceptive energy is below 0.4 and `0.05` otherwise.

Measured, this session, 400 steps at each of 3 seeds:

| quantity | min | max | distinct |
|---|---|---|---|
| raw vision bid, tectum output | 1.000000000 | 1.000000000 | 1 |

The vision bid sits at the clamp ceiling at 100.0 percent of steps. The highest value any
other module can reach is 0.6. So the winner is determined by an inequality between two
constants, and no retrieval score and no interoceptive state can change it.

## The bids entering the competition are constants

Standard deviation of each raw bid across the run:

| module | sd, seed 42 | sd, seed 43 | sd, seed 44 |
|---|---|---|---|
| vision | 1.276e-02 | 7.415e-03 | 6.556e-03 |
| audio | 0 | 0 | 0 |
| memory | 0 | 0 | 0 |
| body | 0 | 0 | 0 |
| semantic | 0 | 0 | 0 |

Even vision's small non-zero variance is not a response to the stimulus. Vision takes
exactly two values, 1.0 and 0.95, and the 0.95 is a fixed decay factor:
`sensory_tectum.py:521` returns `current_bid * 0.95` when the reentrant loop hands back
an empty broadcast. The variance measures how often the broadcast was empty. It carries
no information about what was on screen.

## The intended cause discriminator could not run, and that is the finding

The plan proposed separating two causes by standardizing each module's bid across the run
and recomputing the winner. That asks which module is most surprised relative to its own
normal, which is what a salience competition should compare.

**The test is undefined here.** Standardization needs variance, and only one bid has any.
At every seed the probe reports `UNDEFINED. Only 1 bid(s) carry any variance: ['vision']`
and declines to name a winner.

This is recorded because the first version of the function did not decline. It pinned
zero-variance bids at 0.0 and returned a winner anyway, which made `max()` fall through to
alphabetical order and print `audio: 400` for a module whose bid is exactly 0.0 at every
step. Read at face value that would have said the winner moves under rescaling, which is
the signature of a scale artifact. It was a tie-break artifact. The function now returns
`None` and reports the per-module standard deviations instead.

So the cause is neither of the two the step was built to separate. The bids do not carry
a mis-scaled salience signal. They do not carry a salience signal.

## Where the signal goes

`sensory_tectum.py:456` is `bid = torch.tanh(kl_div)`. In float32, `tanh` returns exactly
1.0 once its input passes roughly 9.0.

The input, measured this session:

| seed | min | max | mean | sd |
|---|---|---|---|---|
| 42 | 2088.5 | 348221.5 | 344166.6 | 30370.4 |
| 43 | 2088.5 | 348213.9 | 344164.5 | 30370.2 |
| 44 | 2088.5 | 348205.0 | 344164.6 | 30370.2 |

The smallest value observed is already 232 times past saturation. The signal spans a
factor of 167 between its minimum and maximum with a standard deviation of about 30,370,
and every one of those values maps to the same output.

The reason the range is this large is `sensory_tectum.py:444`: in the continuous branch
`kl_div = kl_map.sum() / post_logits.shape[0]`, a sum over 1024 latent elements divided
only by the batch size. The quantity handed to `tanh` is a sum where `tanh` expects
something near unit scale.

## Rescaling the bid would NOT produce phase-tracking, and that is a scale-free result

Before recommending a fix, the fix has to be worth making. Any rescaling of the bid is a
monotone transform of the same pre-tanh KL, so it cannot create a phase difference the KL
does not already carry. Picking a scale and reporting who wins measures the scale, not the
model.

The scale-free question is whether the KL separates the phases. Mean pre-tanh KL:

| seed | fixation | sample | delay | choice | sample vs delay, Cohen's d |
|---|---|---|---|---|---|
| 42 | 325382.8 | 347399.6 | 347429.9 | 347603.4 | -0.061 |
| 43 | 325383.2 | 347336.4 | 347478.6 | 347601.3 | -0.293 |
| 44 | 325398.2 | 347378.2 | 347416.7 | 347619.5 | -0.078 |

Sample and delay differ by 30 to 142 out of about 347,400, which is 0.01 to 0.04 percent,
with an effect size below 0.3 at every seed.

Fixation is separable: 325,383 against 347,400, a 6.3 percent drop, consistent at all
three seeds. So the signal is not blind to everything. It distinguishes a blank screen
from a populated one, and it does not distinguish holding an item in memory from looking
at one.

**Consequence for the proposed fix.** Repairing the bid scale would change which module
wins, by an amount decided by whichever scale is chosen. It would not make the winner
track the task. The scale repair is not a fix for GWT-2 and should not be sold as one.

This agrees with the independent A2 content finding that the pre-tanh signal is
phase-invariant, reproduced here from a separate probe and separate code path.

## What this settles, and what it does not

**Settled.** The workspace competition never switches winner, at 3 seeds, in any phase.
The immediate cause is a bid pinned at the clamp ceiling by a saturated `tanh`. The
underlying signal varies but does not separate sample from delay.

Not shown:

- **This is not an indicator re-score.** GWT-1 and GWT-2 are both marked IMPLEMENTED and
  this is evidence bearing on them. Re-scoring is a separate decision and it moves the
  public clock, so it is not taken here.
- **Two of five modules were disabled.** `_build_components` hardcodes
  `enable_audio=False`, so audio and semantic bid exactly 0.0 and this is a three-module
  race. A four-module race is a stronger test of parallel competition. It does not change
  the arithmetic above: the disabled modules bid 0.0 against a ceiling of 1.0.
- **The probe substitutes two bids.** `probe_perception_decodability.py:231-237` hardcodes
  `memory: 0.1` and `body: 0.05` where the training loop computes them. The ceilings
  argument covers this: the computed values cannot exceed 0.6 and 0.15, both below a
  vision bid measured at 1.0 on every step.
- **One checkpoint.** Three probe seeds measure the probe's variance, not variance across
  independently trained systems.
- **No claim about what the correct bid scale is.** `kl_map.mean()` and an explicit
  division by the element count are both candidates. Neither is tested here, and the
  phase result above says neither would buy task selectivity.

## Reproduce

```
python -m scripts.analysis.probe_workspace_competition \
  --load-tectum runs/gate_ckpt_s42/tectum.pt
```
