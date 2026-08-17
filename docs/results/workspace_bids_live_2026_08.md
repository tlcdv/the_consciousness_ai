# FAILED: four of five workspace modules emit a literal constant, measured live at 3 seeds

**The specialists do not compete.** In a live training run, at 3 seeds, 8000 steps each,
with audio and semantic explicitly enabled, four of the five modules emit a single
constant bid and the fifth takes two values. Vision wins 99.0 to 99.8 percent of steps,
and in the last 2000 steps of every seed it wins 2000 of 2000.

GWT-1 is scored on "multiple specialized systems operating in parallel". They run in
parallel. They do not compete. This is the first time the workspace competition has been
recorded during training.

## Why this run exists

Every previous statement about the non-vision bids was read from code, not measured.
Offline probes could not measure them: `_compute_broadcast` in
`probe_perception_decodability.py:231-237` substitutes the literals `0.1` and `0.05` for
the `memory` and `body` bids, so the live values had never been observed at all. The
workspace-competition verdict
([workspace_competition_2026_08.md](workspace_competition_2026_08.md)) flagged this as its
own main limitation.

The step CSV now logs the five raw bids and the winner
(`scripts/training/metrics_logger.py`, `models/core/global_workspace.py`). Instrumentation
only; nothing branches on it.

## Configuration chosen to favour the claim

`--enable-audio --enable-mock-semantic`, on top of the perception fix
(`--rssm-latent-mode continuous --capsule-workspace-source all_levels`). In the default
config audio and semantic bid exactly 0.0, so a default run could not separate "these
modules cannot compete" from "these modules were switched off". This is the configuration
most favourable to GWT-1, which is the right one for a claim being tested for failure.

## Result: the bids, exact values from the 9-decimal CSV

| module | distinct | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| vision | **1** | `1.000000000` x8000 | same | same |
| audio | **1** | `0.000000000` x8000 | same | same |
| memory | **1** | `0.100000000` x8000 | same | same |
| body | 2 | `0.15` x7885, `0.05` x115 | `0.15` x7866, `0.05` x134 | `0.15` x7885, `0.05` x115 |
| semantic | 5 | sd 4.54e-08 | sd 4.47e-08 | sd 4.27e-08 |

Vision and audio have a standard deviation of exactly 0. Memory's is 1.388e-17, which is
float representation noise on a constant. Semantic's five values all lie inside
`[0.999999762, 1.000000000]`, a span of 2.4e-07.

## Winners

| seed | vision | semantic | no ignition | last 2000 steps |
|---|---|---|---|---|
| 42 | 7922 (99.0%) | 58 (0.7%) | 20 (0.2%) | vision 2000 of 2000 |
| 43 | 7985 (99.8%) | 1 (0.0%) | 14 (0.2%) | vision 2000 of 2000 |
| 44 | 7918 (99.0%) | 53 (0.7%) | 29 (0.4%) | vision 2000 of 2000 |

**The competition degrades as training proceeds.** Every semantic win and every
non-ignition step falls early in the run. By the final 2000 steps the winner is vision at
every step, at every seed.

Semantic's wins are not selection. When semantic wins, its mean bid is 0.999999942 against
vision at 1.000000000. The margin is about 3e-08, a tie-break in the eighth decimal
resolved after modulation and binding.

## Three findings that were not predicted from reading the code

**1. The memory bid never leaves its floor.** `train_rlhf.py:1056-1073` computes
`memory_bid = min(0.6, 0.1 + retrieval_score * 0.5)`, starting at 0.1. It reads exactly
`0.100000000` at 8000 of 8000 steps, at all 3 seeds. The retrieval branch never fires
once. Earlier analysis treated memory's 0.6 cap as the reason it could not outbid vision.
The cap is irrelevant: memory never approaches it.

**2. Audio bids 0.0 even with `--enable-audio`.** The auditory specialist is constructed
(`train_rlhf.py:516-518`), and DMTS emits no sound, so it processes silence and bids zero
at every step. Enabling the flag does not produce a competitor on a visual task. Any
future count of "active modules" on DMTS must exclude audio regardless of the flag.

**3. Semantic saturates too, independently.** It sits at 1.0 to seven decimal places. This
is a SECOND saturation with a different cause from the vision `tanh` at
`sensory_tectum.py:456`. Two of the five bidders are pinned at the top of their range by
unrelated mechanisms.

## What this settles

The workspace competition is not close. It is not noisy. Four of five inputs are
constants, so there is nothing for a selection mechanism to select between, and the
outcome is fixed before any competition runs.

This also closes the harness question the previous verdict left open. That verdict argued
from code that memory and body could not exceed a vision bid of 1.0 given caps of 0.6 and
0.15. The argument was correct and the premise was weaker than the conclusion deserved.
The live values are now measured and they are lower than those caps, not higher.

## What this does NOT settle

- **No indicator is re-scored here.** GWT-1 is marked IMPLEMENTED and this is evidence
  against the competition half of it. Re-scoring changes the public coverage count and is
  an owner decision. The rubric records the evidence and the status is unchanged.
- **This is DMTS only.** Audio bidding zero is a property of a silent visual task, not
  proof that the auditory specialist cannot bid. A task with sound would test that and
  has not been run.
- **It does not show the modules are incapable of competing.** It shows that in this
  configuration, on this task, at 3 seeds, they do not. Whether a different bid
  construction would change that is the subject of
  [bid_reduction_candidates_2026_08.md](bid_reduction_candidates_2026_08.md), which
  FAILED for the vision bid.
- **`body` is not a salience signal.** It is a two-valued switch on interoceptive energy
  and it sits at 0.15 on 98.3 to 98.6 percent of steps. It varies, and it does not vary
  with anything about the stimulus.

## Reproduce

```
python -m scripts.training.train_rlhf --env dmts --episodes 40 --seed 42 \
  --rssm-latent-mode continuous --capsule-workspace-source all_levels \
  --enable-audio --enable-mock-semantic --log-dir runs/bidlog_s42
```

Then read `bid_vision`, `bid_audio`, `bid_memory`, `bid_body`, `bid_semantic` and
`bid_winner` from `runs/bidlog_s42/metrics.csv`.
