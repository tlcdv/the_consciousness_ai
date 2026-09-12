# PCI across trained checkpoints: the gate is inert on all four, and the two normalizations disagree

**KILL. The gate shows ZERO causal response on four independently trained
checkpoints, with a working control on every one.** PCI reads exactly 0.0000 at the
primary site at every checkpoint, while the control reads 0.0552 to 0.0657. The
earlier single-checkpoint result generalises.

**Second result, not anticipated: `pci` and `pci_casali` disagree by up to 5.5x on
the same data.** On `capfix_seed44` the control reads 0.0552 under the local
normalization and 0.3065 under Casali, and the broadcast reads 0.0007 against
0.7248. The published human cutoff is near 0.31. Reading the Casali column against
the human scale on that checkpoint would produce a consciousness-level claim from a
signal the local normalization calls noise.

Read from the CSVs in `runs/_pci_multi/`, 16 rows each, not from exit codes.
Pre-registered before the runs; the gate stated below is the one that was written
first. No indicator moves and the clock does not move.

## What gap this closes

`docs/instrument_inventory.md` listed PCI as UNPROVEN with a two-part falsifier:

> Run on more than one checkpoint, and state the reading rules that replace the
> human scale.

Part one is now done. Part two is still open, and the second result above makes it
sharper rather than easier.

The earlier work (`pci_trained_2026_08.md`) used 3 PROBE seeds on ONE trained
checkpoint. Probe seeds vary the rollout, not the learned weights.

## Design

Identical flags on every run. Only `--load-tectum` varied. Serial, one at a time.

```
python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 5 \
  --magnitude 1000 --load-tectum runs/<CKPT>/tectum.pt \
  --latent-mode continuous --capsule-workspace-source all_levels
```

`capfix_alllevels`, `capfix_seed43` and `capfix_seed44` share a training
configuration and differ only in training seed, so the comparison among them is
clean. `gate_ckpt_s42` is the published checkpoint, re-run as a harness check.

The perception fix was already ON in the earlier work: its recorded command carries
`--latent-mode continuous --capsule-workspace-source all_levels`. So "does the
perception fix wake the gate" was already answered NO, and was not re-asked.

## The harness check passed first

Run 0 recovered the published numbers before anything new was read.

| Quantity | Published 2026-08 | This run |
|---|---|---|
| rssm control, seed 42 | 0.0657 | 0.0657 |
| gate | 0.0000 | 0.0000 |

Pre-impulse divergence 0.0e+00 on every trial of every run, so the two rollouts are
bit-identical up to the impulse and everything after it is the impulse and nothing
else.

## Result

| Checkpoint | rssm (CONTROL) | **gate (PRIMARY)** | broadcast |
|---|---|---|---|
| `gate_ckpt_s42` | 0.0657 sd 0.0053 | **0.0000** | 0.0000 |
| `capfix_alllevels` | 0.0645 sd 0.0012 | **0.0000** | 0.0000 |
| `capfix_seed43` | 0.0558 sd 0.0034 | **0.0000** | 0.0000 |
| `capfix_seed44` | 0.0552 sd 0.0030 | **0.0000** | 0.0007 sd 0.0014 |

Every control is non-zero, so no run is void and the zero at the gate is a fact
about the architecture rather than a broken probe. Control range across four
checkpoints is 0.0552 to 0.0657, a spread of 0.0105.

The gate's LZ complexity is exactly 2.0 at every checkpoint, the floor for a flat
signal. Its largest absolute response anywhere is 1.32e-05 against a control
response of 1.29e+00, five orders of magnitude apart.

## The normalization disagreement

Both columns are computed from the same response, in the same run.

| Checkpoint | site | `pci` | `pci_casali` | ratio |
|---|---|---|---|---|
| `capfix_alllevels` | rssm | 0.0645 | 0.0815 | 1.3x |
| `capfix_seed43` | rssm | 0.0558 | 0.0566 | 1.0x |
| **`capfix_seed44`** | **rssm** | **0.0552** | **0.3065** | **5.5x** |
| `gate_ckpt_s42` | rssm | 0.0657 | 0.0779 | 1.2x |
| **`capfix_seed44`** | **broadcast** | **0.0007** | **0.7248** | **1000x** |

Three checkpoints agree within 1.3x. The fourth does not, in both of its live sites.
`capfix_seed44` is also the checkpoint with by far the lowest `active_fraction`,
0.0274 against 0.2381 to 0.5668 elsewhere. The Casali normalization divides by a
source-entropy term, so a response concentrated in very few active channels inflates
it. That is a plausible mechanism and it is NOT demonstrated here; it is the next
thing to test.

**Operational rule until that is settled: cite `pci`, never `pci_casali`, and never
either against the human 0.31 scale.** One checkpoint in four already lands at 0.3065
under Casali at a site whose local reading is 0.0552.

## A spread the single-checkpoint study could not show

`active_fraction` at the control varies by a factor of 20 across checkpoints, while
`pci` varies by a factor of 1.2.

| Checkpoint | active_fraction | pci |
|---|---|---|
| `capfix_seed44` | 0.0274 | 0.0552 |
| `capfix_alllevels` | 0.2381 | 0.0645 |
| `gate_ckpt_s42` | 0.2739 | 0.0657 |
| `capfix_seed43` | 0.5668 | 0.0558 |

Independently trained networks spread the causal response very differently and score
almost the same. Any reading rule that treats `pci` as a summary of response
structure has to account for this, because the summary is stable while the structure
underneath it is not.

## What this establishes

- The gate is causally inert to perturbation across four trained checkpoints, three
  of them independently trained from one configuration.
- PCI discriminates: control separates from primary at every checkpoint.
- The two normalizations are not interchangeable, and the published human scale is
  unsafe against either.

## What this does NOT establish

- **Not a survey of architectures.** Three of the four checkpoints share one training
  configuration. This says the gate is inert across SEEDS of that configuration.
- **It does not answer planning #15.** That issue is blocked on the PCI evidence and
  on an alternatives survey. This supplies the first. The survey is still missing.
- **It does not supply the reading rules.** PCI stays UNPROVEN. Part one of its
  falsifier is met; part two is open and is now harder than it looked.
- **It says nothing about the C1 competence wall.**

## Reproduce

```
for CKPT in gate_ckpt_s42 capfix_alllevels capfix_seed43 capfix_seed44 ; do
  python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 5 \
    --magnitude 1000 --load-tectum runs/$CKPT/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --out runs/_pci_multi/$CKPT.csv
done
```

## Next

1. Test the source-entropy mechanism for the Casali divergence. Cheap: the columns
   are already logged, so it is an analysis of the existing CSVs plus one targeted
   probe, not a new run.
2. Write the reading rules, or record that PCI cannot be read without them.
3. Run PCI on a checkpoint from a DIFFERENT training configuration, which is the
   part this study deliberately did not cover.
