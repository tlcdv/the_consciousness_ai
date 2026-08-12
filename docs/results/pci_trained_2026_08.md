# PCI on a trained checkpoint: the instrument works, the gate is causally inert

**Result: the gate shows ZERO causal response to perturbation, at 3 seeds and 2
magnitudes, with a working control.** PCI itself is not degenerate. It reads 0.0649
(sd 0.0036) at the control site and exactly 0.0000 at the primary site. The instrument
discriminates; the gate does not respond.

This is the first control-validated architectural finding on this project. Every prior
degeneracy result was confounded by a possible harness failure. This one is not.

One trained checkpoint (`runs/gate_ckpt_s42`), 3 probe seeds, 5 trials each, dmts. No
indicator moves.

## The default magnitude was 500x too small, and that is a harness fact

The first run FAILED at the control site: `rssm` PCI 0.0000 with the probe printing its
own "the impulse did not propagate, this is a harness or configuration failure, NOT a
finding about the agent". Per the standing rule, no PCI value is interpretable in that
state, so the harness was diagnosed before anything was read.

The response scales **perfectly linearly** with impulse magnitude, exactly 1.120e-03 per
unit:

| magnitude | rssm response | rssm PCI |
|---|---|---|
| 1 (default) | 1.120e-03 | 0.0000 |
| 10 | 1.120e-02 | 0.0000 |
| 100 | 1.120e-01 | 0.0155 |
| 1000 | 1.120e+00 | 0.0589 |

Baseline sd at the rssm is 1.750e-01 and the threshold is 3 sigma, so a response must
exceed 0.525 to register. At the default magnitude of 1.0 the response is 156x BELOW the
noise floor. The impulse propagates; it was simply far too small to be seen.

The perfect linearity over three orders of magnitude is itself worth recording. A system
with rich nonlinear dynamics would saturate or amplify. This one attenuates linearly, so
what PCI scores here is closer to a linear echo than to integrated dynamics.

## Result at 3 seeds

| site | role | magnitude 100 | magnitude 1000 |
|---|---|---|---|
| rssm | CONTROL | 0.0167, 0.0167, 0.0192 | 0.0657, 0.0645, 0.0645 |
| **gate** | **PRIMARY** | **0.0000, 0.0000, 0.0000** | **0.0000, 0.0000, 0.0000** |
| broadcast | exploratory | 0.0000, 0.0000, 0.0000 | 0.0000, 0.0000, 0.0007 |

Control mean at magnitude 1000: **0.0649, sd 0.0036** across seeds 42, 43, 44.

## The gate is decoupled by five orders of magnitude

The gate's raw response does not scale with the impulse the way the rssm's does:

| magnitude | rssm response | gate response | ratio |
|---|---|---|---|
| 1 | 1.120e-03 | 5.960e-08 | 19,000x |
| 100 | 1.120e-01 | 3.338e-07 | 336,000x |
| 1000 | 1.120e+00 | 8.128e-06 | 138,000x |

At magnitude 1, the gate response of 5.96e-08 is float noise. Even at magnitude 1000,
an impulse roughly 5,700 times the rssm's own baseline sd, the gate moves by 8e-06.

### Correction: the pathway EXISTS, the signal dies along it

An earlier draft of this document said "the gate does not receive the perturbation".
That was too strong, and a direct test corrects it.

Perturbing `tectum.h_state` by magnitude 1000 and re-running the SAME frame changes
`tectum_content` by **5.406e-04**, against a clean content scale of 6.162e-02. So the
recurrent state does influence the content pathway. The wiring is intact.

What the pathway does is attenuate. The cascade, from one magnitude-1000 impulse:

| stage | response |
|---|---|
| h_state (injection site) | 1.120e+00 |
| tectum_content | 5.406e-04 |
| gate | 8.128e-06 |

Roughly three orders of magnitude lost from the recurrent state to the content, and two
more from the content to the gate. The correct statement is that the perturbation reaches
the gate attenuated by about 137,000x, far below any threshold, not that it fails to
arrive.

That is a different and more useful finding than a disconnection: the architecture
connects these stages, and the connection transmits almost nothing.

This is consistent with the same day's interventional result, where forcing the gate's
own previous state across all 243 joint states moved its output by at most 4.6e-02 and
never out of one tertile bin. The gate is close to a constant function of everything
upstream of it.

### One observation recorded, not yet chased

In the same test, `vision_bid` read **exactly 1.000000** both before and after a
magnitude-1000 impulse. The ignition signal is `input_energy = max(bound_bids)`, which
the 2026-07 diagnosis found pinned near 1.499. A vision bid saturated at exactly 1.0
would be a direct contributor to that. Single observation on one frame, not a
measurement, and it is not pursued here.

## What this settles, and what it does not

**PCI is NOT retired.** It discriminates: 0.0649 at one site and 0.0000 at another, with
seed variance of 0.0036. That is an instrument producing different numbers for different
conditions, which is more than any other instrument on this project has done.

**The zero is a finding about the architecture, not the instrument.** The control rules
out the harness explanation that made every previous zero ambiguous.

Not shown:

- **The published human PCI scale does not transfer.** The ~0.31 conscious/unconscious
  cutoff must never be quoted against these numbers, and 0.0649 is not "low
  consciousness". It is a number on an uncalibrated local scale.
- **The readable values need a non-physiological impulse.** Magnitude 100 to 1000 against
  a baseline sd of 0.175 is not a perturbation of the operating regime. Whether a PCI
  measured there means anything about the system's normal dynamics is open.
- **One checkpoint, 3 probe seeds.** This measures PCI's own variance, not variance
  across independently trained systems. Three trained checkpoints would cost three
  training runs and are not done here.
- **The response matrices for a second, independently normalized reading (PCIst) were
  not persisted.** That data-retention requirement was not implemented before this run,
  so PCIst cannot be computed on these trials offline.

## Consequence

The instrument programme has produced one instrument that discriminates. Its reading on
the primary site is zero, and the zero is now trustworthy rather than ambiguous.

That relocates the question. The gate is not badly measured. It is causally isolated:
inert under perturbation from upstream, and nearly constant under intervention on its own
recurrence. Two independent probes on the same day agree on that.

## Reproduce

```
python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 5 --magnitude 1000 \
  --load-tectum runs/gate_ckpt_s42/tectum.pt \
  --latent-mode continuous --capsule-workspace-source all_levels
```
