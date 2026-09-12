# PCI across trained checkpoints: the study FAILED its own design, and the falsifier cannot be met from disk

**FAILED. Three of the four checkpoints ran with a RANDOMLY INITIALISED gate, so
their gate readings say nothing about a trained system.** The probe printed
`WARNING: gate is randomly initialised (no sibling gate checkpoint); gate-level
readings below are not from the trained system` on every trial of those three runs,
ten times in total. The first version of this document reported them as a result
anyway, because the console output was filtered with a grep that did not include
that warning.

**Consequence: the inventory falsifier for PCI is NOT met, and cannot be met from
the checkpoints on disk.** Exactly one checkpoint in `runs/` has a trained gate:
`runs/gate_ckpt_s42`. Gate weights were not saved before 2026-08-11, so every
earlier checkpoint is a trained tectum with an untrained gate. Meeting the
falsifier needs NEW training runs, not another read-only probe.

Superseded claim, retracted: an earlier version of this file said "the gate shows
ZERO causal response on four independently trained checkpoints". That is false.
One checkpoint, not four.

## What is still valid

The tectum IS loaded from `tectum.pt` on all four checkpoints, so readings at the
`rssm` site are from trained weights in every run. Those stand.

| Checkpoint | rssm (CONTROL) | trained gate? | gate reading usable? |
|---|---|---|---|
| `gate_ckpt_s42` | 0.0657 sd 0.0053 | YES | yes |
| `capfix_alllevels` | 0.0645 sd 0.0012 | no | NO |
| `capfix_seed43` | 0.0558 sd 0.0034 | no | NO |
| `capfix_seed44` | 0.0552 sd 0.0030 | no | NO |

Three things survive.

**1. The harness reproduces.** Run 0 recovered the published control of 0.0657
against a published 0.0657, and its gate reading of 0.0000 on the one checkpoint
that has a trained gate. Pre-impulse divergence was 0.0e+00 on every trial of every
run, so the rollouts are bit-identical up to the impulse.

**2. The control is stable across four trained tecta**: 0.0552 to 0.0657, a spread
of 0.0105. This is a genuine multi-checkpoint reading, because the tectum is loaded
in all four. It says the perturbation propagates comparably in independently trained
recurrent states.

**3. The Casali divergence is confirmed with numbers.** This was PREDICTED, not
discovered. The module docstring of `perturbational_complexity.py` already states
that `pci_casali` diverges for sparse responses, gives a worked example returning
1.32, says "Do not rank conditions by it", and says the 0.31 cutoff "must never be
quoted against these values". What is new is the size on real trained tecta:

| Checkpoint | site | `pci` | `pci_casali` | ratio | active_fraction |
|---|---|---|---|---|---|
| `capfix_alllevels` | rssm | 0.0645 | 0.0815 | 1.3x | 0.2381 |
| `capfix_seed43` | rssm | 0.0558 | 0.0566 | 1.0x | 0.5668 |
| **`capfix_seed44`** | **rssm** | **0.0552** | **0.3065** | **5.6x** | **0.0274** |
| `gate_ckpt_s42` | rssm | 0.0657 | 0.0779 | 1.2x | 0.2739 |

The checkpoint that diverges is the one with by far the lowest `active_fraction`,
which is exactly the sparse regime the docstring names. One checkpoint in four lands
at 0.3065, numerically on the published human cutoff of 0.31, at a site whose local
reading is 0.0552.

**Operational rule: cite `pci`, never `pci_casali`, and neither against the human
0.31 scale.**

## What was fixed as a result

The module docstring says `source_entropy` and `active_fraction` are reported "so
the sparse regime is visible rather than hidden". The probe's console summary
printed neither `pci_casali` nor any divergence warning, so the sparse regime was
visible only to someone who opened the CSV. `probe_pci.py` now prints `pci_casali`
in the summary line and warns when it exceeds `pci` by 2x or more, naming the
mechanism and the citation rule. Verified firing at 5.6x and 1000.6x on
`capfix_seed44`.

## The lesson, recorded because it is the point

The probe was not wrong and the guard was not missing. The probe printed the
disqualifying warning ten times. It was filtered out of the console by a grep
written to extract the result lines, and the CSV columns that would have shown it
do not include gate provenance.

Two things follow, and the second is the useful one:

1. Never filter a probe's output down to the lines that carry the answer. The lines
   that disqualify the answer do not look like the answer.
2. A run's output CSV should carry whether each read site came from trained weights.
   `runs/_pci_multi/*.csv` has 15 columns and none of them records that the gate was
   random, so the CSVs alone cannot distinguish a valid gate reading from an invalid
   one. That is a gap worth closing before the next attempt.

## What it would take to meet the falsifier

Train at least two more checkpoints with the current code, which saves
`tectum.gate.pt`, then re-run this study. That is a training job, roughly 32 s per
episode on dmts, and it is heavy and serial. It is no longer a free read-only probe,
and the inventory should say so.

## What this does NOT establish

- **Nothing new about the gate.** The only valid gate reading here reproduces
  `pci_trained_2026_08.md` on the same checkpoint.
- **It does not answer planning #15**, which is blocked on this evidence and on an
  alternatives survey. It supplies neither.
- **It does not supply the reading rules.** PCI stays UNPROVEN on both halves of its
  falsifier.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## Reproduce

```
for CKPT in gate_ckpt_s42 capfix_alllevels capfix_seed43 capfix_seed44 ; do
  python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 5 \
    --magnitude 1000 --load-tectum runs/$CKPT/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --out runs/_pci_multi/$CKPT.csv
done
```

Read the FULL output. Three of these four print a gate-provenance warning that
disqualifies their gate readings.
