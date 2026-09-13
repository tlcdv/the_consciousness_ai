# The oscillatory binding layer is FROZEN: it emits one value forever

**Over 1200 settled steps the Kuramoto binding layer produces exactly ONE value.**
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

`scripts/analysis/probe_oscillator_reset.py`, read-only, on `runs/gate3_s42`. 8
episodes, 400 warm-up steps discarded, 1200 settled steps analysed. Two arms.

| Quantity | A: current behaviour | B: `reset_state()` per episode |
|---|---|---|
| `sync_R` distinct values | **1** | 563 |
| `sync_R` range | 0.450000 to 0.450000 | 0.081518 to 0.450000 |
| `sync_R` sd | **0.000e+00** | 2.290e-02 |
| align spread distinct values | **1** | 1193 |
| align spread range | 1.314e-02 to 1.314e-02 | 1.625e-02 to 9.734e-01 |
| winner | vision 1.000 | vision 0.928, semantic 0.072 |
| silent | 0.000 | 0.313 |

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

## What a per-episode reset does, stated carefully

It unfreezes the layer. It is **not** established to repair anything.

The variation in arm B is the **relaxation transient after each reset**. `sync_R` in
arm B still reaches 0.450000, the same ceiling, so within each episode it re-converges
to the frozen value. Resetting re-injects a transient every episode; it does not stop
the convergence.

**Whether that transient constitutes meaningful binding, that is, whether modules
processing related information lock together, is NOT tested here.** It could be
relaxation noise. That is the next question and it is the same content question this
project asks of everything else.

Arm B also costs ignition: the workspace is silent on 0.313 of steps against 0.000 in
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

- The binding layer emits one value for 1200 consecutive steps, on the measure that
  decides whether it discriminates.
- The binding boost is therefore a uniform 1.5x and changes no module's rank.
- The cause is a missing per-episode reset, verified in the current code.
- The saturated vision bid means the binding layer cannot see the trained weights.

## What this does NOT establish

- **It does not show a reset improves anything.** It shows the layer stops being
  constant. The transient's meaning is untested.
- **It is offline replay, not training.** No claim about learning dynamics.
- **One measurement, not three seeds**, for the reason above.
- **One task**, DMTS, where audio is structurally silent.
- **It does not re-score RPT-2 or any other indicator.**

## Next

1. **Content-test the reset transient.** Does per-module alignment track which module
   is carrying the stimulus? This is the same eta-squared against a permutation null
   used elsewhere, and it decides whether a reset gives real binding or noise.
2. If it does, the reset becomes a default-off flag with a 3-seed training arm, and
   the bar is stated before the run.
3. `architecture.md:115` asks "does disrupting oscillatory synchronization degrade
   performance". That question is currently unanswerable, because synchronization is
   already maximal and constant. Fixing the freeze is a precondition for asking it.

## Reproduce

```
python -m scripts.analysis.probe_oscillator_reset --checkpoint runs/gate3_s42
```

About 4 minutes.
