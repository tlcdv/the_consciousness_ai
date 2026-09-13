# PCI reading rules: a zero is uninterpretable without the variance floor

**The default variance floor is ABOVE the gate substrate's own fluctuation, so at
the default setting the gate cannot register a response no matter how large it is.**
`DEFAULT_VAR_FLOOR` is 1e-4. Channels at or below it are "treated as inactive and
never mark significant". The gate nodes sit at 4.1e-05 to 6.9e-05
(`instrument_inventory.md`, measured 2026-09-02), entirely below that line.

**This partly supersedes `pci_three_trained_gates_2026_09.md`**, published earlier
today, which reported the gate as inert at three trained checkpoints. That reading
is correct at the default floor and uninformative as a statement about the
architecture, because the floor sits above the substrate.

**UPDATED LATER THE SAME DAY. The gate DOES respond, on some probe seeds.** An
attempt to fix the floor produced readings that cannot be separated from noise, and
that part stands. What changed is the reason. Measured in
`pci_gate_attenuation_2026_09.md`: on `gate3_s42` at probe seed 42 the gate response
is 11.53 times the gate's own baseline fluctuation, which is a detection by any
threshold, and the default floor silences it. On probe seeds 43, 44 and 45 the same
checkpoint gives 0.14, 0.26 and 0.16, which is no detection at all. The gate has two
regimes selected by the probe seed, and averaging across them reports neither. See
rule 6 and the section that follows it.

## The test

Same three checkpoints, same probe seed, same magnitude, same flags. Only
`--var-floor` changed. The flag was added for this test; the default is unchanged
and reproduces the earlier numbers exactly, which was checked first.

| Checkpoint | gate @ 1e-4 (default) | gate @ 1e-6 | gate raw max response |
|---|---|---|---|
| `gate3_s42` | 0.0000 | **0.0439** (2 of 5 trials at 0.1097) | 4.023e-05 |
| `gate3_s43` | 0.0000 | 0.0000 (0 of 5) | 1.192e-07 |
| `gate3_s44` | 0.0000 | 0.0000 (0 of 5) | 1.192e-07 |

Per-trial at `gate3_s42`: 0.109718, 0.0, 0.0, 0.0, 0.109718.

1e-6 was chosen because it admits the four live gate nodes while still excluding the
dead one, `gate_adaptation`, at std 1.043e-07.

## What this changes and what it does not

**CORRECTED LATER THE SAME DAY. Read the section "A per-site floor was tried and it
FAILED" below before using anything here.** The 0.0439 at `gate3_s42` was first
written up as a real response that the default floor had hidden. It is not. Measured
over 40 synthetic seeds, a lowered floor returns a non-zero from PURE NOISE in 29 of
40 cases, mean 0.083, max 0.192. 0.0439 and 0.0603 both sit inside that range.

What survives:

**The default floor cannot measure the gate.** It sits above the gate's own
fluctuation, so a zero there carries no information either way. That part stands.

**On two checkpoints of three the gate response is 1.192e-07**, roughly 100x BELOW
its own baseline fluctuation. That is not a borderline reading; nothing could detect
it, and no floor recovers it.

**Nothing here says the gate responds.** The lowered-floor readings are
indistinguishable from the noise the lowered floor admits. See rule 6.

## The reading rules

The inventory falsifier asks for "the reading rules that replace the human scale".
These are them, stated from the evidence above.

**Rule 1. A PCI of 0.0000 is uninterpretable on its own.** It carries two
indistinguishable meanings: the site did not respond, or the site's channels were
excluded before measurement. Always report `var_floor` beside it, and the site's own
`median_baseline_sd`. If the floor is at or above that sd, the zero is a property of
the instrument, not of the system.

**Rule 2. The floor must be set below the substrate being measured.** The default
1e-4 was chosen for the rssm, whose baseline sd is 2.4e-02 to 1.8e-01, three to four
orders of magnitude above it. The same floor applied to the gate, at 1e-05 to
1e-04, disqualifies the substrate. One floor cannot serve sites that differ by four
orders of magnitude.

**Rule 3. Never compare absolute PCI across checkpoints.** The control spans 3.5x
across seeds of one configuration, 0.0471 to 0.1631
(`pci_three_trained_gates_2026_09.md`). Within one checkpoint, comparing sites is
sound because they share the impulse and the network. Across checkpoints it is not.

**Rule 4. Cite `pci`, never `pci_casali`, and neither against the human 0.31
cutoff.** The Casali normalization divides by the observed activity level and
inflates with sparsity, exactly as the module docstring predicts: active_fraction
0.1846 gives 1.5x, 0.0682 gives 2.8x, 0.0443 gives 3.9x, 0.0040 gives 35.4x. Values
above 0.31 appear at sites whose local reading is noise.

**Rule 5. A reading is void unless the control is alive AND every site reports
`weights=trained`.** Both are now printed and recorded per row.

## What these rules do NOT provide

They do not provide an absolute scale. There is still no value of `pci` on this
architecture that means "integrated" rather than "not integrated", and nothing here
supplies one. The rules make a reading INTERPRETABLE, not comparable to human data.

That is the honest limit, and it is why PCI stays UNPROVEN rather than becoming
TRUSTED on the strength of this document. The falsifier asked for reading rules and
these are they; whether they are sufficient for TRUSTED is an owner decision, and
rule 3 in particular constrains the instrument severely.

## A per-site floor was tried and it FAILED

Implemented and tested the same day, `--var-floor-mode relative`, which derives the
floor from each site's own channels instead of one absolute value. It does not work,
and the reason is worth more than the fix would have been.

It behaves correctly on the three things it was designed for:

| Check | Result |
|---|---|
| Default unchanged | `gate3_s43` reproduces control 0.0471, gate 0.0000, exactly |
| Admits the live gate nodes | 4 of 5 register; `gate_adaptation` stays excluded |
| Per-site floors | rssm ~1e-3, gate ~1e-6, broadcast ~1e-5, recorded per row |

And it fails the one that decides whether it is usable. Over 40 synthetic seeds with
the response drawn at exactly the baseline scale, that is pure noise:

| Response | `absolute` non-zero | `relative` non-zero |
|---|---|---|
| 1.0 sigma (noise) | 0 of 40 | **29 of 40, mean 0.083, max 0.192** |
| 8.0 sigma (signal) | 0 of 40 | 40 of 40, mean 0.997 |

**The real gate reading this mode produced, `gate3_s42` at 0.0603, sits inside that
noise range.** So it cannot be cited. The `noise_eps` guard does not help: it sits
near 1e-9 while the gate channels are at 1e-5 to 1e-7.

The mode is kept, default off, with the defect pinned by a test that asserts the
failure rather than hiding it (`TestVarianceFloorMode` in
`tests/test_perturbational_complexity.py`).

## What this actually establishes, which is more useful than the fix

**Neither floor works at the gate.** `absolute` silences a real substrate;
`relative` admits its noise. The gate's fluctuation is too close to numerical noise
for any fixed threshold to separate signal from it.

That is a statement about the SUBSTRATE, not the instrument. PCI is not
misconfigured at the gate. It is being asked to resolve a difference smaller than
the noise it sits in.

**Rule 6, added.** A gate-level PCI is not readable from a threshold at all. It
needs a NULL: score the probe with no impulse, or with the impulse applied at a
shuffled step, and use that distribution as the floor. Anything inside the null is
not a response. This is the control this project already uses for content
(eta-squared against a permutation null shuffling labels across trials), and it is
the same discipline applied to a different measure.

Until that null exists, no gate-level PCI may be cited under any floor setting. The
existing zeros at the absolute floor are not evidence of inertness, and any non-zero
at a lower floor is not evidence of a response.

## Rule 7, and a correction that was itself retracted

**Rule 7. Report `active_fraction` at the control with every PCI.** Above roughly
0.9 the impulse is too large: at magnitude 100,000 the rssm reaches 0.993 and its
PCI collapses from 0.165 to 0.023, because when every entry crosses threshold the
source entropy goes to zero. The control is valid only inside a magnitude window,
and both ends are now measured. This rule stands.

**An earlier version of this section said rules 1, 2 and 6 treated a thresholding
problem that did not exist, on the strength of
`pci_gate_saturation_2026_09.md`. THAT DOCUMENT IS RETRACTED and this section with
it.** Measured in `pci_gate_attenuation_2026_09.md`: on `gate3_s42` at probe seed 42
the gate's causal response is 11.53 times its own baseline fluctuation, a clear
detection, and the default floor of 1e-4 silences it because that floor sits 40x
above the gate's baseline sd of 2.520e-06. **Rules 1 and 2 were right. The floor is
exactly the obstruction they said it was.**

## What the attenuation study adds to rule 6

Rule 6 asked for a null. The new evidence says precisely which quantity needs one.

The gate's RESPONSE is stable: 2.906e-05, 2.933e-05, 4.023e-05, 3.469e-05,
1.961e-05 across five probe seeds on one checkpoint, a spread of 2.1x. The gate's
BASELINE moves by 84x over the same five seeds: 2.520e-06, 2.120e-04, 1.519e-04,
2.120e-04, 2.518e-06. The two trials that register at a 1e-6 floor are exactly the
two with a quiet baseline.

The extra variance is located. `tectum_content` has the same baseline sd on every
probe seed (1.0843e-04, 1.0837e-04), while the `broadcast` it feeds reads 1.0843e-04
at probe seed 42 and 5.0012e-03 at probe seed 43, which is 46x larger. Something
between the tectum and the broadcast injects seed-dependent variance while passing
the causal response through unchanged. Candidates are the persistent oscillatory
phases in `reentrant.settle`, the affective modulation of the bids, and the
interoceptive state. None is tested.

**So rule 6's null must be taken over PROBE SEEDS, not over trials of one seed.** A
mean across probe seeds, which is what `--trials` produces, averages a detectable
regime and an undetectable one and reports a number that describes neither. The
0.0439 reported earlier in this document is such a mean and must not be cited.

## Next

0. **Build the null over PROBE SEEDS, on the BASELINE.** Run the probe with no
   impulse across about 20 probe seeds and record the distribution of the gate's
   baseline sd, not of its PCI. That distribution decides which seeds can carry a
   gate reading at all. This supersedes items 1 and 2 and is cheap, because it is
   the same probe with one argument changed.

1. Re-run the three checkpoints at 1e-6 across more probe seeds, to settle whether
   the two-regime split holds at the same 2-in-5 rate.
2. Find what injects seed-dependent variance into the broadcast while the tectum
   content it carries stays at a constant 1.084e-04. Three candidates are named in
   `pci_gate_attenuation_2026_09.md` and none is tested.
3. Re-examine any earlier verdict that read a gate-level zero at the default floor.
   The masking applies to all of them.

## Reproduce

```
for S in 42 43 44 ; do
  python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 5 \
    --magnitude 1000 --load-tectum runs/gate3_s$S/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --var-floor 1e-6 --out runs/_pci_multi/vf6_5t_s$S.csv
done
```

Omit `--var-floor` to reproduce the default-floor readings.
