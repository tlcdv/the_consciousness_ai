# FAILED: sync_R carries no stimulus identity. Once the oscillators converge it reports the bids

**sync_R does not track which shape was shown, at any of 3 seeds.** Its eta-squared for
6-class `sample_shape` is 0.0020, 0.0146 and 0.0182, against permutation nulls whose MEANS
are 0.0030, 0.0172 and 0.0318. Every observed value sits below its own null mean, let
alone the 95th percentile.

Under the standing fork this is neither (a) nor (b). It is a third answer the fork did not
have: **varying but contentless.**

The mechanism is arithmetic. The Kuramoto oscillators converge to full synchrony and stay
there, because nothing ever resets their phase. At full synchrony the order parameter
equals the mean of the module bids exactly, so sync_R stops measuring binding and starts
reporting the bid vector, which is separately measured degenerate.

3 seeds, 40 episodes, 8000 steps each, dmts, `--enable-audio --enable-mock-semantic`, from
`runs/bcast_s4{2,3,4}` already on disk. Plus one constructed input. No training was run.

## Two corrections to the work that led here

**The wrong class was quoted.** The planning note that proposed this audit quoted
`models/core/complex_binding.py:169-171`. That is the "komplex" path.
`global_workspace.py:143-154` defaults `binding_mechanism` to `"akorn"`, so the active
class is `WorkspaceBindingSystem` in `models/core/oscillatory_binding.py` and no number in
`runs/` came from `complex_binding.py`.

**The planned test was invalid and was not run.** The plan proposed grouping steps by the
logged bid tuple, to see whether sync_R is constant inside a group. `train_rlhf.py:1973`
logs `raw_bids`, but `global_workspace.py:210-217` lets `affective_modulator.modulate()`
REWRITE `bids` before `bind_bids` receives them at line 224. The logged bids are not the
amplitudes, so that grouping would charge the modulator's variation to phase state.

The data says so without any modelling. The order parameter has `sum(amplitudes) / N` as a
hard ceiling, and sync_R sits ABOVE the ceiling implied by the logged bids:

| seed | median logged bid sum / N | modal sync_r | excess |
|---|---|---|---|
| 42 | 0.449999988 | 0.450108000 | +1.080e-04 |
| 43 | 0.449999988 | 0.450108000 | +1.080e-04 |
| 44 | 0.449999988 | 0.450108000 | +1.080e-04 |

## The distinct-value count was a misleading summary, including mine

An earlier note recorded sync_r as taking 988 to 1245 distinct values and left it there.
That count is true and it hides the shape of the distribution.

| seed | distinct | std | modal value | share at the modal value |
|---|---|---|---|---|
| 42 | 1253 | 1.29e-02 | 0.450108000 | **48.5%** |
| 43 | 1002 | 9.06e-03 | 0.450108000 | **83.8%** |
| 44 | 991 | 8.38e-03 | 0.450108000 | **85.5%** |

**The modal value is identical to nine decimal places at all three seeds**, and it holds
the majority of steps. Against the strict non-degeneracy bar used on the vision-bid
candidates (>= 100 distinct, span >= 0.2, <= 1 percent pinned), sync_r FAILS at all three
seeds on the pinned clause. It clears the separate DEAD bar, so it is not near-constant.

The modal value is also phase-invariant. At seed 42 it occurs at 1519 of 2975 sample steps
and 1605 of 3215 delay steps, in proportion to how often each phase occurs.

## Why: the oscillators converge and nothing resets them

`oscillatory_binding.py:117-118`:

```
mean_field = einsum('bn,bnd->bd', amplitudes, current_phases) / N
sync_R     = ||mean_field||
```

`current_phases` are unit vectors, so `sum(amplitudes) / N` is reached exactly when every
oscillator points the same way.

`reset_state()` exists and is called in `scripts/demos/demo_akorn_binding.py` only. **It is
never called from the training loop.** Phases persist for a whole run.

Constructed input, the modal logged bid tuple `[1.0, 0.0, 0.1, 0.15, 1.0]`, ceiling
0.450000000, five iterations per step with persistent phases:

| step | sync_R | gap to ceiling |
|---|---|---|
| 1 | 0.348264754 | +1.017e-01 |
| 20 | 0.373813897 | +7.619e-02 |
| 100 | 0.446163923 | +3.836e-03 |
| 200 | 0.449979007 | +2.099e-05 |
| 400 | 0.450000018 | -1.788e-08 |

This is a deterministic result, so it has no seeds and claiming seed replication for it
would be false precision. Its independent check is a closed-form control in
`probe_sync_r_content.py` that RAISES if the converged value misses `sum/N` by more than
1e-6.

The episode statistics agree that there is no reset. Episode means have a standard
deviation of 1.9e-03 to 2.7e-03, which is 0.15 to 0.33 of the within-episode variation. A
per-episode random reinitialization would scatter them far more.

## What is established, and what is not

**Established.** sync_r fails the content test at 3 seeds. Its modal value is
seed-independent and phase-invariant and holds 48 to 86 percent of steps. The layer
converges to `sum(amplitudes)/N` under persistent phases, verified against a closed form.
`reset_state()` is never called from the training loop.

**NOT established: that the runs were sitting exactly at full synchrony.** That the modal
value is 0.450108 rather than 0.450000 is consistent with a modulated bid sum of 2.25054,
and the affect-modulated bids are NOT logged, so the identity cannot be checked. The
prediction is specific and cheap to test: log the bid vector that `bind_bids` actually
receives, and check whether `sync_r` equals its mean whenever the trajectory is at the
modal value. Until that is done the convergence account is a strong inference from a
constructed input, not a measurement of the runs.

**NOT established: anything about the 2026-07 marker.** `signature_assessment_2026_07.md:111`
reports mean sync_R of 0.2662 to 0.2666 across five training objectives. These runs give
0.460953, 0.452451 and 0.451857. Those are different flags and different modules enabled.
Cross-arm comparison under different flags is invalid here, so both are recorded and
neither is subtracted from the other. The only claim licensed: the marker was measured
under a configuration that is not the one in use, so it cannot be cited as current until
it is re-measured under matched flags.

## What this means for the indicator rubric

RPT-2's negative evidence in `docs/consciousness_indicators_butlin.md:250-252` is that
"mean sync_R is 0.2662 to 0.2666 across five different training objectives", read as
binding synchrony not responding to the training objective.

Once the oscillators converge, sync_R is the mean of the bid vector. So the invariance of
sync_R across those five arms is a statement about the BIDS in those arms, not about
binding. Whether the bids were in fact invariant across them cannot be checked: the bid
columns did not exist in 2026-07.

**No indicator is re-scored here.** The count stays 3 IMPLEMENTED, 11 PARTIAL, 14 total,
and the clock does not move. A re-score of RPT-2 is a decision for the owner, and it rests
on evidence this document does not have.

## Scoreboard for the standing fork

| level | verdict |
|---|---|
| gate | (a) alive, blind binning |
| workspace 3-tuple | (a) alive, blind binning |
| 256-D broadcast | (a) alive, mis-read |
| vision bid, 4 reductions | (c) varying, contentless |
| **sync_R** | **(c) varying, contentless** |

The fork's two answers are not enough. Two levels now vary and carry nothing, which is
neither "the instrument is blind" nor "the dynamics are still".

## Reproduce

```
python -m scripts.analysis.probe_sync_r_content
```
