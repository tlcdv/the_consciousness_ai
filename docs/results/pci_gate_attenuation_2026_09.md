# The attenuation is at ONE stage, and the saturation verdict published hours earlier is RETRACTED

**RETRACTION FIRST. `pci_gate_saturation_2026_09.md`, published earlier today, is
wrong in its headline and in its inventory row.** It said the gate's causal response
is 0.26 times the gate's own spontaneous fluctuation and saturates at 0.55, so "3.0
is unreachable at any magnitude". Both halves fail:

- **0.26 was a median across three probe seeds whose values are 11.53, 0.14 and
  0.26.** The spread is 84x and it is entirely in the DENOMINATOR. The responses are
  2.906e-05, 2.933e-05 and 4.023e-05, a spread of 1.4x. A median of three numbers
  that disagree by 84x summarises nothing, and the document did not say a median had
  been taken.
- **The bar of 3 IS reached.** At probe seed 42 on `gate3_s42` the ratio is 11.53 at
  magnitude 1000, and 5 of the 15 measured cells below exceed 3.
- **The saturation is not architectural.** For a 10x impulse increase the gate
  response grows 1.2x on `gate3_s42`, 2.0x on `gate3_s44` and 37.3x on `gate3_s43`.
  Three checkpoints, three behaviours.

**What replaces it: the entire attenuation is at one stage, `h_state` to
`tectum_content`, and nothing downstream of it attenuates at all.**

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move. PCI stays UNPROVEN.

## The stage table

New probe, `scripts/analysis/probe_gate_attenuation.py`. It instruments every stage
between the impulse and the gate's five nodes and reports the maximum absolute
clean-vs-perturbed difference in the response window. It uses the same impulse seed
offset as `probe_pci`, and its gate responses reproduce that probe's CSV exactly
(2.9057e-05, 2.9325e-05, 4.0233e-05, 5.9605e-08 at the four matching runs).

`gate3_s42`, probe seed 42:

| Stage | m=1000 | m=10000 | m=100000 | growth | expected |
|---|---|---|---|---|---|
| `h_state` | 1.1230e+00 | 1.1230e+01 | 1.1230e+02 | **100.00** | 100 |
| `tectum_content` | 2.2791e-03 | 4.1879e-03 | 3.9858e-03 | **1.75** | 100 |
| `vision_bid` | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0 | 100 |
| `broadcast` | 2.2791e-03 | 4.1879e-03 | 3.9858e-03 | 1.75 | 100 |
| `gate_in` | 2.2791e-03 | 4.1879e-03 | 3.9858e-03 | 1.75 | 100 |
| `enriched` | 2.2791e-03 | 4.1879e-03 | 3.9858e-03 | 1.75 | 100 |
| `logits` | 1.1645e-04 | 2.7695e-04 | 2.7586e-04 | 2.37 | 100 |
| `gate_state` | 2.9057e-05 | 6.9171e-05 | 6.8933e-05 | 2.37 | 100 |

Read the four middle rows. `tectum_content`, `broadcast`, `gate_in` and `enriched`
are IDENTICAL to every printed digit, at every magnitude, in all five runs. The
workspace, the reentrant settle, the truncation to `hidden_size` and the temporal
feedback projection add no attenuation whatever. `logits` and `gate_state` apply a
fixed factor of about 0.013, which moves by 1.4x while the input moves by 100x, so
that stage is linear too.

**One stage on the whole path is nonlinear, and it is the tectum forward that turns
`h_state` into `tectum_content`.** That is the answer to the question this work set
out to ask.

## The `tanh` at `sensory_tectum.py:456` transmits exactly nothing

`vision_bid` is **1.0000000000 at every step of every rollout, minimum equal to
maximum, sd 0.000e+00**, and its clean-vs-perturbed difference is exactly zero at
every magnitude. The bid is `torch.tanh(kl_div)` and it is fully saturated.

This was already recorded as the cause of degenerate workspace competition. It is now
measured on the PCI path as well: a hundred-thousand-fold impulse changes the vision
bid by zero. The bid is not an attenuating channel, it is an absent one.

It was also the stated first place to look for the saturation, and it is NOT the
cause: the content path carries the perturbation fine, and it is the tectum's
recurrent-to-output step that compresses it.

## The gate nodes are NOT saturated either

Measured node means over 40 clean steps on `gate3_s42`:

| Node | mean | sd (probe seed 42) | sd (probe seed 43) |
|---|---|---|---|
| `attention_level` | 4.976352e-01 | 6.5188e-06 | 2.3325e-04 |
| `stability_score` | 4.846665e-01 | 2.5200e-06 | 2.1202e-04 |
| `adaptation_rate` | 8.172921e-03 | 1.0671e-08 | 9.6147e-08 |
| `meta_memory_coherence` | 4.854634e-01 | 8.5097e-06 | 1.2234e-04 |
| `narrator_confidence` | 4.934433e-01 | 7.7088e-07 | 2.6004e-04 |

Every node sits at 0.485 to 0.498, the steepest part of the sigmoid. The sigmoid
output stage was the obvious suspect and it is cleared.

## The real obstruction: the denominator moves 84x with the PROBE seed

This is what the retracted document mistook for saturation.

`gate3_s42`, one checkpoint, five trials at a 1e-6 floor. Trials differ only in probe
seed:

| Trial | Probe seed | gate baseline sd | gate response | ratio | PCI @ 1e-6 |
|---|---|---|---|---|---|
| 0 | 42 | **2.520e-06** | 2.906e-05 | **11.53** | 0.1097 |
| 1 | 43 | 2.120e-04 | 2.933e-05 | 0.14 | 0.0000 |
| 2 | 44 | 1.519e-04 | 4.023e-05 | 0.26 | 0.0000 |
| 3 | 45 | 2.120e-04 | 3.469e-05 | 0.16 | 0.0000 |
| 4 | 46 | **2.518e-06** | 1.961e-05 | **7.78** | 0.1097 |

The response varies by 2.1x. The baseline varies by 84x. The two trials that register
are exactly the two whose baseline is quiet, and that is the whole of the effect.

**Where the extra baseline variance enters is now located too.** The
`tectum_content` baseline sd is the same on every probe seed, 1.0843e-04 and
1.0837e-04. The `broadcast` baseline sd is 1.0843e-04 at probe seed 42, 5.0012e-03 at
seed 43 and 3.5828e-03 at seed 44. So on the noisy seeds the broadcast carries 33x to
46x more spontaneous fluctuation than the tectum content it transmits, while passing
the causal response through unchanged.

**Something other than the tectum injects variance into the broadcast, and that is
what buries the gate response.** Candidates inside `_compute_broadcast` are the
oscillatory phases in `reentrant.settle` (recorded as never reset, so they persist
and converge), the affective modulation of the bids, and the interoceptive state.
**None of these is tested. This is a location, not a mechanism.**

## Every measured cell

Gate ratio, response divided by the gate's own baseline sd. 3 is needed to register.

| Checkpoint | Probe seed | m=1000 | m=10000 | m=100000 |
|---|---|---|---|---|
| `gate3_s42` | 42 | **11.53** | **27.45** | **27.35** |
| `gate3_s42` | 43 | 0.14 | 0.22 | 0.25 |
| `gate3_s42` | 44 | 0.26 | 0.46 | 0.53 |
| `gate3_s43` | 42 | 0.00 | 0.04 | 1.54 |
| `gate3_s44` | 42 | 0.15 | **12.63** | **25.10** |

Two cautions that apply to this table, both of which changed a conclusion:

**Rule 7 disqualifies the last two columns.** At magnitude 10000 the control's
`active_fraction` is already 0.9359 to 0.9492, and at 100000 it is 0.9810 to 0.9995.
Rule 7 says above roughly 0.9 the impulse is too large and the reading is
meaningless. The retracted document stated that rule and then drew its conclusion
from the two columns the rule excludes. **At m=1000, the only valid column, exactly
one of these five runs registers.**

**Some responses are at float32 resolution.** Gate node values are float32 near
0.485, where one unit in the last place is 2.9802e-08. The m=1000 responses on
`gate3_s43` and `gate3_s44` are 5.9605e-08, which is 2 ulp. Growth factors computed
from a 2 ulp starting value mean nothing, so the per-checkpoint growth above is taken
from m=10000 upward.

## What this establishes

- **The attenuation is at `h_state` to `tectum_content` and nowhere else.** Verified
  at 5 run configurations and 3 magnitudes, 15 cases.
- **`sensory_tectum.py:456` transmits zero**, confirmed by direct measurement on this
  path. It was the stated first suspect and it is not the cause of the compression.
- **The gate sigmoids are in their linear region** and are not the cause either.
- **The gate DOES produce a detectable response on `gate3_s42` at 2 of 5 probe
  seeds**, 11.53 and 7.78 times its own baseline, and the default variance floor of
  1e-4 silences both because that floor is 40x above the gate's baseline sd of
  2.5e-06. **This vindicates reading rules 1 and 2 and retracts the claim that the
  floor question was secondary.**

## What this does NOT establish

- **It does not explain the compression.** It locates it inside one forward. Which
  operation inside `sensory_tectum.forward` compresses a 100x change in `h_state` is
  NOT measured here.
- **It does not explain the broadcast's seed-dependent variance.** Three candidates
  are named above and none is tested.
- **It does not make PCI readable at the gate.** One probe seed in five registering
  is not a reading rule. Rule 6 still applies: the gate needs a null, and now there is
  a precise reason why, which is that the denominator is the unstable quantity.
- **It does not promote anything.** No indicator moves.
- **It does not answer planning #15**, which also needs the alternatives survey.

## Next

1. **Build the null (rule 6), and build it over PROBE SEEDS.** The quantity that
   needs a distribution is the gate baseline, not the response. Run the probe with no
   impulse across 20 probe seeds and record the gate baseline sd distribution. That
   distribution decides whether a reading is usable on that seed.
2. **Find what injects variance into the broadcast.** Log the broadcast baseline sd
   with `reentrant.settle` held fixed, then with the affective modulation off. This
   separates the three named candidates and costs one read-only run each.
3. **Locate the compression inside `sensory_tectum.forward`.** Same instrumentation
   method, applied to the operations between `h_state` and the returned content.

## Reproduce

```
for CFG in "42 gate3_s42" "43 gate3_s42" "44 gate3_s42" "42 gate3_s43" "42 gate3_s44" ; do
  set -- $CFG
  python -m scripts.analysis.probe_gate_attenuation --env dmts --seed $1 \
    --magnitudes 1000 10000 100000 --load-tectum runs/$2/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --out runs/_pci_multi/att_$2_p$1.csv
done
```

About 17 seconds per configuration. Read the FULL output, including the growth column
and the baseline sd column. The growth column is the answer and the baseline sd
column is why the earlier verdict was wrong.
