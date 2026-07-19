# Path B stage 2 contrastive: integration defect found by audit (2026-07-19)

FAILED FIRST: the contrastive objective shipped in commit `e54aa86` was NON-FUNCTIONAL in
every code path. It could not have produced a valid result. No experimental verdict was
ever drawn from it, so nothing downstream is contaminated, but the "smoke test confirmed"
note recorded when it landed was not evidence of anything.

This document records what was broken, how it passed 10 green unit tests, and what is
fixed. It is a tooling verdict, not a consciousness result. No Butlin indicator moves.

## What was broken

### 1. The loss crashed for 2 or more negatives, and computed wrong math for exactly 1

`RSSMContrastiveHead.forward` returns `[1, D]` for a single latent (it unsqueezes a batch
dim). The training loop built its bank of per-trial sample averages from those outputs, so
every bank entry was `[1, D]` rather than `[D]`. Consequences:

- `torch.stack(neg_avgs)` produced `[K, 1, D]`, which is 3-D, so the `negatives.dim() == 2`
  broadcast branch in `loss()` was skipped.
- `torch.bmm` then received batch dim `K` against an anchor batch dim of 1.

Reproduced in isolation (`proj_dim=8`):

| K negatives | neg shape | result |
|---|---|---|
| 1 | `[1, 1, 8]` | runs, but WRONG value |
| 2 | `[2, 1, 8]` | `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [5, 128]...` |
| 5 | `[5, 1, 8]` | same crash |

The K=1 case did not crash only because both batch dims happened to be 1. It was still
wrong: the `[1, 1, D]` positive broadcast into an ELEMENTWISE product instead of a dot
product, so `pos_sim` had shape `[1, D]` instead of `[1]`. With all three vectors
identical, InfoNCE must return `ln(1 + K)`. Measured: **8.7504** where the correct value is
**0.6931** (`ln 2`).

### 2. The choice-phase anchor was labeled with the next step's phase

The aggregation block reads `info` before `env.step()` (correctly paired with the latent),
but the loss block ran after `env.step()` reassigned `info`. Replaying the real
`DMTSEnv` phase stream for 400 steps:

```
OLD, true phase of the latent it trained on: {'choice': 119, 'delay': 5}
NEW, true phase of the latent it trains on:  {'choice': 123}
```

5 of 124 anchors (~4%, one per trial, at the delay-to-choice transition) were blank-screen
DELAY latents trained as if they showed the stimulus. That directly contradicts the design
premise that the anchor and positive both carry the sample identity.

### 3. Two flags did not do what their help text said

- `--latent-contrastive-train-every` was stored in config and printed in a log line but
  never read anywhere in the training loop. Its help text described batching behavior that
  does not exist. REMOVED.
- The contrastive weight was unsettable: the code read `latent_contrastive["weight"]`, but
  `build_config` never wrote that key and no CLI flag existed, so it was pinned at 0.1.
  Every sibling head (`control_repr`, `recon`, `wm_recon`, `latent_id`) has a weight knob.
  This mattered: the B0 ceiling test only became airtight after a 1/10/100 weight sweep
  ruled out a weak-gradient artifact, and that sweep was impossible here. ADDED as
  `--latent-contrastive-weight`.
- `--enable-latent-contrastive` help claimed the objective trains "only the head, not the
  RSSM backbone". The code sums the loss into `total_tectum_loss`, which backprops into
  the RSSM. Verified empirically: `anchor.requires_grad` is True on 2/2 firings. The help
  text was wrong; the code and the commit message were right. Help CORRECTED.

### 4. `--help` crashed entirely (pre-existing, unrelated to this commit)

An unescaped `%` in the `--policy-input` help ("99% decodability") was parsed by argparse
as a `% d` conversion against its params dict:
`TypeError: %d format: a number is required, not dict`. So `python -m
scripts.training.train_rlhf --help` raised instead of printing, making every flag
undiscoverable. Escaped to `99%%`. Confirmed present before this commit by stashing.

## Why 10 green unit tests missed all of it

`tests/test_rssm_contrastive.py` exercised only well-formed tensors: `[B, D]` anchors with
`[B, K, D]` or `[K, D]` negatives. Those are the shapes the head's docstring promises. The
training loop never passes them. No test called the head the way the caller actually calls
it, so a crashing integration shipped with a green suite.

The recorded "DMTS smoke test confirmed at 60 steps" gave false confidence for a
structural reason: at 60 steps roughly one trial completes, the bank never reaches
`min_pairs`, the loss never fires, and the broken branch is never entered. **A smoke test
that does not reach the code path under test confirms nothing.**

## The remaining design problem (NOT fixed, owner decision)

Even with the shapes corrected, the objective is inert at the project's standard settings.
`_c_bank` is initialized inside `run_episode`, so it resets every episode, while the head
and its optimizer persist. Only ~2 to 3 DMTS trials complete in a 200-step episode, and
`min_pairs=4` requires that many completed sample phases within ONE episode.

Measured (seed 42, continuous latent):

| config | contrastive firings |
|---|---|
| 3 episodes x 200 steps (standard) | **0** |
| 1 episode x 400 steps | 2 |
| 1 episode x 600 steps | 2 |

So a standard 100-episode DMTS run would apply the objective **zero** times while
appearing to run fine. Three ways out, none taken here because each changes what the
experiment means: lower `min_pairs` (weakens InfoNCE, which wants many negatives), persist
the bank across episodes (introduces cross-episode negatives), or raise `--max-steps`
(changes the task regime).

Interim guard added instead: `run_episode` now logs the firing count, and emits a WARNING
naming the bank size and `min_pairs` when the count is zero. A silent zero-gradient run
looks identical to a genuine FAILED verdict and must never be reported as one.

## Verification

- Full suite: **784 passed, 4 skipped** (was 778/4; 6 regression tests added).
- New tests exercise the real calling convention: bank entries at K in {1, 2, 5, 16}, the
  `[D]` bank contract, and the `ln(1 + K)` identity that pins the math.
- Baseline bit-identity with the flag OFF: `metrics.csv` md5 `75c9357787a1f87e02651d84c76b88fa`
  identical before and after the change (DMTS, seed 42, 60 steps).
- Gradient reaches the RSSM: `anchor.requires_grad` True on 2/2 firings.
- Loss value ~1.55 on an untrained head, near the `ln 4 = 1.386` chance floor for 3
  negatives, which is the expected starting point.

## Status

No consciousness claim, no indicator move. Path B stage 2 remains UNTESTED as an
objective: it has never actually run. The flag stays default off. Before any verdict is
drawn from it, the episode-length/bank-scope problem above must be resolved and the weight
swept, per the B0 precedent.
