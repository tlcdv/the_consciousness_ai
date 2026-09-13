# PCI reading rules: a zero is uninterpretable without the variance floor

**The default variance floor is ABOVE the gate substrate's own fluctuation, so at
the default setting the gate cannot register a response no matter how large it is.**
`DEFAULT_VAR_FLOOR` is 1e-4. Channels at or below it are "treated as inactive and
never mark significant". The gate nodes sit at 4.1e-05 to 6.9e-05
(`instrument_inventory.md`, measured 2026-09-02), entirely below that line.

**This partly supersedes `pci_three_trained_gates_2026_09.md`**, published earlier
today, which reported the gate as inert at three trained checkpoints. That reading
is correct at the default floor and incomplete as a statement about the
architecture.

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

**The gate is not uniformly inert.** On one checkpoint of three it responds,
intermittently, and the default floor hid that. The masking is real.

**On two checkpoints of three it genuinely does not respond.** Their raw gate
response is 1.192e-07, which is float noise. No floor setting recovers a signal that
is not there, and lowering the floor did not manufacture one, which is itself a
useful negative control on this test.

**One checkpoint of three, in two trials of five, is a HYPOTHESIS.** It is not
evidence that the gate responds. It is evidence that the previous measurement could
not have detected it either way.

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

## Next

1. Re-run the three checkpoints at 1e-6 with more trials and more probe seeds, to
   settle whether `gate3_s42`'s two-in-five response replicates.
2. Decide whether the probe should set the floor PER SITE from each site's own
   baseline sd, rather than one global value. That would make rule 2 automatic
   instead of a thing a reader must remember.
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
