# Phi out of the agent. A default-off flag first

Date: 2026-11-03. Owner decision of 2026-11-02: take phi, the integrated information of IIT, out of
the agent, in steps. First a flag that is off by default, then a check, then the owner's decision about
the default. The public text and the logo follow that decision.

## Why

- The hypothesis that phi tracks binding (Phi-1) FAILED across 9 runs and is closed. **Corrected
  2026-11-03.** Phi-1 was never confirmed in those 9 runs, but none of them met the variance floors the
  pre-registration requires before a negative counts, so each run's verdict was RE-RUN, and the chapter
  was closed (`docs/consciousness_indicators_butlin.md`, RPT-2 row, corrected 2026-09-13).
- The phi instrument carries no stimulus content (`docs/instrument_inventory.md`).
- The owner's new direction drops IIT as a measure.
- Phi still acts on the agent. In `runs/vg_learn_s601` (8000 steps, read on 2026-11-02) pyphi computed phi
  at 1599 steps, its largest value was 0.0083 and its mean 0.00093. So it lowered exploration by at most
  8.3 percent (scale 0.917 to 1.0).

## What phi reaches today, and what `--phi-control off` gives instead

| Reader | `on`, the default, unchanged | `off` |
|---|---|---|
| pyphi on the gate states, every `--phi-sample-every` steps | computed | not called. Phi is recorded as NaN with `phi_method` "off", never as a made-up 0.0 |
| Exploration, `scripts/training/train_rlhf.py` near the phi-weighted exploration comment | max(0.2, 1 - 10 phi) on arousal | 1.0 |
| Intrinsic reward | 0.5 times the change of phi | no phi term. The curiosity term stays |
| Memory attention and priority, and the action core's attention | phi | 0.0 |
| Self-model attention | the phi of the step before | 0.0 |
| Workspace state of the instrument logs | (length, phi, sync) | (length, sync) |

Why 0.0 for memory. Memory stores to its index only at an attention of 0.7 or more
(`MemoryConfig.attention_threshold`), and consolidation gives each experience a relevance of
max(0.2, priority). Phi stayed below both, so every experience already had the relevance 0.2 and none
reached the index. The value 0.0 keeps that. The flag refuses `--enable-riiu`, which adds a reward from
another integrated-information measure, and `--phi-proxy`, which chooses a stand-in phi. With `off` a run
does not need pyphi. The ConsciousnessGate still runs, and its states are still recorded.

## Checks, stated before their runs

All runs use Python 3.12.10 and the Gate B7 command, the recorded argv of `runs/f2_py312_s73`, with the
changes named below.

- **I1, the default is unchanged.** `--episodes 2 --seed 1298`, once from a checkout of main at a1dae12
  and once from this branch with no new flag. The two `metrics.csv` files have the same md5. If they
  differ, the change stops here.
- **I2, the off path runs.** The same command on this branch with `--phi-control off`. The run ends with
  2 rows in `episodes.csv`, and every recorded step has `phi_method` "off".
- **P, how much the behaviour changes, reported without a verdict.** `--episodes 10 --seed 1299`, once
  with the default and once with `--phi-control off`, from this branch. The report gives the first step at which
  the recorded actions differ, and the total reward of each episode with the mean of the last 5. One seed is
  a hypothesis. It sets no default.

Seeds 1298 and 1299 are discarded seeds that no run folder uses. The cost is about 4 minutes for I1 and
I2 and about 17 minutes for P, at the 45 to 51 seconds per episode measured on 2026-11-02.

## Decision for the owner after the checks

Whether `--phi-control off` becomes the default. A new default changes every training number from then
on. Runs made before it stay reproducible with `--phi-control on`.

## Owner decision, 2026-11-03

`--phi-control off` is the default from 2026-11-03, because the project drops Integrated Information
Theory as a whole. The phi code stays, reachable only with `--phi-control on`, so every earlier run can
still be reproduced, and pyphi is needed only then. Texts that present phi or IIT as the project's method
or measure change. Dated records keep their text and get a short dated note. The website, its
architecture diagram and the logo follow the same decision.
