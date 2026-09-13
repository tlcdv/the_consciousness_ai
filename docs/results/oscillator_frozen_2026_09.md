# The oscillatory binding layer is FROZEN: it emits one value forever

**Over 13,600 settled steps the Kuramoto binding layer produces exactly ONE value.**
`sync_R` is 0.450000 at every step, standard deviation 0.000e+00, one distinct value.
The per-module alignment spread is 1.314e-02 at every step, one distinct value.

**So the binding boost is the same multiplier for every module, and the binding cannot
change which module wins.** The boost is `1 + 0.5 * align`
(`oscillatory_binding.py:193-199`), described in the code as "an emergent property of
the oscillator dynamics". It is currently a constant.

Oscillatory binding is one of the six neurobiological features that guide this
architecture, and `docs/architecture.md:23` states its role as "Kuramoto oscillators
synchronize module representations. Modules that process related information
phase-lock into unified percepts." **Measured: they do not phase-lock selectively.
They are permanently locked, identically, whatever they process.**

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## The measurement

`scripts/analysis/probe_oscillator_reset.py`, read-only, on `runs/gate3_s42`. Two
arms, 70 episodes, 400 warm-up steps discarded, 13,600 settled steps analysed.

| Quantity | A: current behaviour | B: `reset_state()` per episode |
|---|---|---|
| `sync_R` distinct values | **1** | 5624 |
| `sync_R` range | 0.450000 to 0.450000 | 0.004858 to 0.450000 |
| `sync_R` sd | **0.000e+00** | 2.696e-02 |
| align spread distinct values | **1** | 13160 |
| align spread range | 1.314e-02 to 1.314e-02 | 2.073e-04 to 9.998e-01 |
| winner | vision 1.000 | vision 0.905, semantic 0.095 |
| silent | 0.000 | 0.327 |
| decodes `sample_shape` | nothing to decode | **no**, 0.1721 vs null p95 0.2221 |

## Why it is frozen

`reset_state` is defined at `oscillatory_binding.py:149` and is **never called** from
`scripts/training/train_rlhf.py`. The only `reset_state` there is
`tectum.reset_state(1)` at `:423`, a different object. The phases are initialized once
per process, lazily, only when `current_phases is None`
(`oscillatory_binding.py:170-171`), and carried forward across every step of every
episode. They converge and never leave.

This is the same root cause `sync_r_content_2026_09.md` identified for sync_R being a
readout of the bids. What is new here is the consequence for BINDING rather than for
the metric: once converged, the layer stops discriminating between modules at all.

## The reset unfreezes the layer, and what it produces carries NOTHING

**Tested and FAILED.** The alignment vector produced by the reset does not decode the
stimulus. 203 trials on `gate3_s42`, 6-class `sample_shape`, one reading per trial,
500-permutation null:

| Quantity | Value |
|---|---|
| CV accuracy | **0.1721** |
| Null p95 | 0.2221 |
| Null mean | 0.1799 |
| Uniform chance | 0.1667 |
| Majority class | 0.2020 |

**The measured accuracy is below the null MEAN and below the majority-class rate.** A
classifier that always guessed the commonest shape would do better. This is
pre-stated outcome (c), RELAXATION NOISE.

**A near-miss that did not survive more power, recorded so nobody re-derives it.** At
83 trials the same test read 0.2169 against a null p95 of 0.2419, which is 90 percent
of the way to the bar. Raising the sample to 203 trials did not push it over; it
collapsed to 0.1721. The first reading was noise, and it is exactly the kind of
almost-significant number that becomes a false finding if reported alone.

The control behaves as the gate predicted: in arm A the alignment is CONSTANT across
trials, so there is nothing to decode. That is outcome (b), and it confirms the test
is reading what it claims to read.

## What a per-episode reset does, stated carefully

It unfreezes the layer. It **repairs nothing**.

The variation in arm B is the **relaxation transient after each reset**. `sync_R` in
arm B still reaches 0.450000, the same ceiling, so within each episode it re-converges
to the frozen value. Resetting re-injects a transient every episode; it does not stop
the convergence.

**And the transient carries nothing**, measured above. So the reset converts a frozen
layer into a noisy one. Neither binds.

Arm B also costs ignition: the workspace is silent on 0.327 of steps against 0.000 in
arm A.

## All three checkpoints return identical numbers, and that is the point

`gate3_s42`, `gate3_s43` and `gate3_s44` return byte-identical values in both arms.

**This is NOT three-seed replication and must not be cited as such.** The oscillator's
only input is the bid vector. The vision bid is `tanh(kl_div)`, saturated at exactly
1.0 on every checkpoint, and the other four bids are constants. So the oscillator
receives the same input regardless of the trained weights, and the phase seed is
fixed. Treat it as ONE measurement.

It does establish something, though: **the binding layer is currently blind to
training.** Nothing the network learns reaches it, because the only channel into it is
a saturated constant.

## What this establishes

- The binding layer emits one value for 13,600 consecutive steps, on the measure that
  decides whether it discriminates.
- The binding boost is therefore a uniform 1.5x and changes no module's rank.
- The cause is a missing per-episode reset, verified in the current code.
- The saturated vision bid means the binding layer cannot see the trained weights.
- **Calling the reset does not repair it.** The layer varies, and what it emits does
  not decode the stimulus, at 203 trials with a 500-permutation null.

## What this does NOT establish

- **It does not show the oscillators are useless in principle.** It shows that as
  wired and as driven by saturated constant bids they neither vary nor carry content.
  A different input might behave differently, and that is untested.
- **It is offline replay, not training.** No claim about learning dynamics.
- **One measurement, not three seeds**, for the reason above.
- **One task**, DMTS, where audio is structurally silent.
- **It does not re-score RPT-2 or any other indicator.**

## Next

1. ~~Content-test the reset transient.~~ **DONE, and it FAILED.** See above. The reset
   is not a repair and should not become a flag on this evidence.
2. **The consolidated pattern is worth stating.** Representations carry the stimulus
   (`obs_map` ~1.0, `kl_map` 0.84/0.71/0.76, the 256-D broadcast 0.76/0.69/0.77).
   DYNAMICAL quantities do not: PCI's response matrix
   (`pci_matrix_content_2026_09.md`) and now the oscillator alignment. That is 2 cases
   and a hypothesis, not a law, but it predicts where not to look next.
3. `architecture.md:115` asks "does disrupting oscillatory synchronization degrade
   performance". That question is currently unanswerable, because synchronization is
   already maximal and constant. Fixing the freeze is a precondition for asking it.

## Reproduce

```
python -m scripts.analysis.probe_oscillator_reset --checkpoint runs/gate3_s42   --episodes 70 --content --permutations 500
```

About 20 minutes. Drop `--episodes 70 --content` for the 4-minute freeze check alone.
Note that at 30 episodes the content test read 0.2169 against a p95 of 0.2419, a
near-miss that reversed with more trials. Do not run it under-powered.
