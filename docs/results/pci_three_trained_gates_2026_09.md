# PCI at three trained gates: the gate is inert, and the control is checkpoint-sensitive

> **PARTLY SUPERSEDED the same day by `pci_reading_rules_2026_09.md`.** The gate
> readings below were taken at the DEFAULT variance floor of 1e-4, which is above the
> gate substrate's own fluctuation of 1e-05 to 1e-04, so the gate could not register
> a response at any size. At a floor of 1e-6 the gate reads 0.0439 on `gate3_s42`,
> two trials of five, while `gate3_s43` and `gate3_s44` stay at exactly 0.0000 with a
> raw response of 1.192e-07. "Inert at three trained gates" holds at the default
> floor; as a statement about the architecture it holds at two of three. Everything
> else in this document stands, including the control spread and the Casali table.


**The gate shows ZERO causal response at three independently trained checkpoints,
each with a TRAINED gate, with a working control on every one.** PCI reads exactly
0.0000 at the primary site at all three seeds. The control reads 0.0471 to 0.1631.
Determinism passed at 0.0e+00 on every trial. No site was disqualified.

**This meets the first half of the inventory falsifier for PCI**, which the
2026-09-12 attempt did not: that attempt used three checkpoints whose gates were
randomly initialised. These three were trained for this purpose with the current
code, which saves `tectum.gate.pt`.

PCI remains UNPROVEN. The second half of its falsifier, the reading rules, is open,
and the control spread below makes it harder rather than easier.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## Result

Three checkpoints, 100 episodes each, dmts, seeds 42/43/44, trained with
`--rssm-latent-mode continuous --capsule-workspace-source all_levels`. Probed at
5 trials, magnitude 1000, probe seed 42.

| Checkpoint | rssm (CONTROL) | **gate (PRIMARY)** | broadcast | all sites trained? |
|---|---|---|---|---|
| `gate3_s42` | 0.1631 sd 0.0058 | **0.0000** | 0.0232 sd 0.0220 | yes |
| `gate3_s43` | 0.0471 sd 0.0023 | **0.0000** | 0.0000 | yes |
| `gate3_s44` | 0.0626 sd 0.0023 | **0.0000** | 0.0000 | yes |

Every row of every CSV carries `weights=trained` at all three sites, so none of
these readings is the random-gate artefact that voided the previous attempt.

## The gate result

Zero at three trained gates, with the control alive at all three. The 2026-08
single-checkpoint finding generalises to independently trained networks. The
perturbation reaches the recurrent state and does not reach the gate.

This is the strongest negative result the instrument set has produced, because it is
the only one with a control that separates from the primary at every seed.

## The broadcast flicker is a HYPOTHESIS, not a result

`gate3_s42` reads 0.0232 at the broadcast, sd 0.0220, min 0.0063, max 0.0625. The
other two seeds read exactly 0.0000.

**One seed of three. Under the standing rule that is a hypothesis and nothing more.**
It is recorded because it is the first non-zero broadcast reading on a trained gate
checkpoint, and because the older checkpoints read 0.0000 there. It may be a
property of that one training run.

It must not be cited as evidence that the broadcast responds. To become a result it
needs to replicate at two more seeds.

## The control is checkpoint-sensitive, and that is new

| Study | Control range | Ratio |
|---|---|---|
| 2026-09-12, four older checkpoints | 0.0552 to 0.0657 | 1.2x |
| This study, three fresh checkpoints | 0.0471 to 0.1631 | **3.5x** |

The earlier study's tight band suggested the control was stable across trained
networks. Across these three it is not. `gate3_s42` reads nearly three times
`gate3_s43` at the same site, same probe seed, same magnitude, same flags.

This bears directly on the missing reading rules. A control whose value moves by 3.5x
across seeds of one configuration cannot serve as a fixed reference, and any reading
rule that assumes a stable control is wrong.

## Casali divergence, again and larger

The warning added on 2026-09-12 fired on its own, without anyone looking for it:

| Checkpoint | site | `pci` | `pci_casali` | ratio | active_fraction |
|---|---|---|---|---|---|
| `gate3_s42` | rssm | 0.1631 | 0.4544 | 2.8x | 0.0682 |
| `gate3_s42` | broadcast | 0.0232 | 0.8211 | **35.4x** | 0.0040 |
| `gate3_s43` | rssm | 0.0471 | 0.1832 | 3.9x | 0.0443 |
| `gate3_s44` | rssm | 0.0626 | 0.0912 | 1.5x | 0.1846 |

The pattern predicted by the module docstring holds cleanly: the lower the
`active_fraction`, the larger the divergence. 0.1846 gives 1.5x, 0.0682 gives 2.8x,
0.0443 gives 3.9x, 0.0040 gives 35.4x. Two of these values exceed the published human
0.31 cutoff at sites whose local reading is far below it.

**Cite `pci`. Never `pci_casali`. Never either against the human scale.**

## Methods defect in this study, recorded

The three training runs were launched without `--log-dir`, which defaults to `runs`.
All three therefore wrote `runs/episodes.csv`, `runs/metrics.csv` and
`runs/env_episodes.csv`, each overwriting the last. Only seed 44's episode log
survives, at 101 rows covering episodes 0 to 99.

Completion of the other two is established from the code rather than from their
logs: the episode loop is at `train_rlhf.py:2639` and the checkpoint save is at
2773, 134 lines after it, so a checkpoint exists only if the loop finished. All six
files were written, all six md5 hashes are distinct, and each run took about 87
minutes.

That is sufficient for this study, which needs trained distinct checkpoints and not
training curves. It is not sufficient in general. **Any future training run must pass
`--log-dir runs/<name>` alongside `--save-tectum`**, and the example command in the
run-experiment protocol omits it.

## What this does NOT establish

- **Not a survey of architectures.** All three share one training configuration.
  This says the gate is inert across SEEDS of that configuration.
- **The broadcast flicker is not a finding.** One seed of three.
- **It does not supply the reading rules.** PCI stays UNPROVEN, and the 3.5x control
  spread makes that half harder.
- **It does not answer planning #15**, which also needs the alternatives survey.

## Reproduce

```
for S in 42 43 44 ; do
  python -m scripts.training.train_rlhf --env dmts --episodes 100 --max-steps 500 \
    --seed $S --rssm-latent-mode continuous --capsule-workspace-source all_levels \
    --save-tectum runs/gate3_s$S/tectum.pt --log-dir runs/gate3_s$S
  python -m scripts.analysis.probe_pci --env dmts --seed 42 --trials 5 \
    --magnitude 1000 --load-tectum runs/gate3_s$S/tectum.pt \
    --latent-mode continuous --capsule-workspace-source all_levels \
    --out runs/_pci_multi/gate3_s$S.csv
done
```

Training is about 87 minutes per seed and runs serially. Read the FULL probe output.
Check that every site prints `[trained]` and that no DISQUALIFIED block appears.
