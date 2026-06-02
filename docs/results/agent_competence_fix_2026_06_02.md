# P5 fix attempt: confirm the representation bottleneck, then try to fix it

**Date:** 2026-06-02
**Builds on:** `docs/results/agent_competence_diagnosis_2026_06_01.md` (the policy is
not the bottleneck; a plain A2C and the Go/No-Go core tie on the broadcast at ~15,
both ~6x below DQN-on-pixels at 92). This session ran the confirmation the
diagnosis called for, then attempted the fix. All numbers loaded from disk
in-session (120 episodes x 100 steps, dark_room, seed 42, `--phi-sample-every 5`).

## Headline

- **Step 1 (confirmation): the broadcast representation is the bottleneck,
  learner-independent.** PASSED its purpose.
- **Step 2 (fix attempt): the action-conditioned forward objective did NOT raise
  competence. FAILED.** Reward is unchanged (14.06 ON vs 14.80 OFF, within noise,
  marginally lower). The representation is the bottleneck, but this particular way
  of shaping it does not fix it.

## Step 1 - confirmation (DQN on the broadcast, learner held constant)

The diagnosis compared two ON-policy learners on the broadcast against an
OFF-policy DQN on pixels, leaving learner family as a confound. This runs the same
off-policy learner (DQN) on the broadcast.

| learner | input | mean | first30 | last30 | max | positive |
|---------|-------|------|---------|--------|-----|----------|
| Go/No-Go | broadcast | 14.80 | 16.88 | 16.00 | 100.0 | 31/120 |
| A2C (standard) | broadcast | 15.45 | 17.55 | 14.18 | 100.0 | 58/120 |
| **DQN** | **broadcast** | **14.65** | 19.47 | 12.97 | 94.95 | 57/120 |
| DQN | pixels (baseline) | **92.00** | 52.72 | 92.00 | - | - |

Three different learners on the broadcast all land at ~15; the same DQN learner on
pixels reaches 92. Holding the learner family constant isolates the input: the
broadcast discards control-relevant information that the pixels retain. The
bottleneck is the representation, not the policy and not the on/off-policy
distinction.

(An early reading of ~65 reward at episode 28 of the DQN-broadcast run was
variance, not a learning trend: first30 19.47 -> last30 12.97, no upward slope, max
94.95 reached only occasionally.)

## Step 2 - fix attempt (action-conditioned forward objective shapes the tectum)

`--enable-control-repr` adds a `ControlRepresentationHead`: an MLP that predicts the
next observation (downsampled to 8x8x3) from the current tectum content + the
action taken, trained on the tectum_optimizer path alongside the reward predictor.
The gradient flows into the tectum through the content, intended to shape it to
encode action consequences (controllable dynamics). The objective is confirmed
active in the ON run (init_components instantiates the head with the flag on, None
with it off; verified in-session).

Go/No-Go policy, control-repr ON vs OFF, all else identical:

| arm | mean | first30 | last30 | max | positive |
|-----|------|---------|--------|-----|----------|
| control-repr ON | **14.06** | 15.83 | 15.90 | 100.0 | 30/120 |
| OFF (gonogo) | **14.80** | 16.88 | 16.00 | 100.0 | 31/120 |

Internal dynamics barely move either: phi 0.0011 (both), sync_R 0.247 (both),
broadcast_mag 0.89 ON vs 0.79 OFF.

**Verdict: FAILED.** The control objective did not raise reward (14.06 vs 14.80,
within noise) and did not meaningfully change the broadcast dynamics. Shaping
tectum content with a next-observation forward model, at weight 1.0, on this single
seed, on dark_room, does not make the broadcast more controllable in a way the
policy can use.

## Why it likely failed (hypotheses, not facts)

1. **The strongest structural one: the policy does not consume tectum content.** The
   policy reads the DETACHED, post-GNW broadcast (train_rlhf.py:735-747). The
   control objective shapes `tectum_content`, which is upstream of GNW competition,
   reentrant settling, attention-weighted fusion, and the detach. A better tectum
   content does not survive that bottleneck to reach the policy input. The flat
   broadcast_mag/phi/sync_R deltas are consistent with this: the objective changed
   the tectum a little, the policy-facing broadcast almost not at all.
2. The forward objective competes with the reward predictor + TDANN topographic +
   gate-diversity losses at weight 1.0; it may not dominate, or predicting raw
   next-observation pixels may pull the tectum toward visual reconstruction rather
   than a control-relevant abstraction.
3. Single seed. But the result is a clean null (slightly negative), not a
   borderline positive, so more seeds are unlikely to flip it to a meaningful gain.

## Honest limitations

- The control-repr loss was not logged to metrics.csv, so the forward model's loss
  trajectory (did it learn to predict) is not shown from disk. The head is
  instantiated, included in the backward, and stepped by its own optimizer, so it
  trained; the magnitude of that training is unmeasured. Logging it is a cheap
  next-step.
- Single seed (42), 120 episodes. The robust claims are the RELATIVE ones (DQN on
  broadcast ties the on-policy learners and sits far below DQN on pixels; control-
  repr ON ties OFF).

## What this changes

The diagnosis is now confirmed twice over: the broadcast representation, not the
policy or the learner family, is the competence bottleneck. The first fix attempt
(shaping tectum content with a forward objective) does not work, most plausibly
because the policy consumes the post-GNW detached broadcast, not the tectum content
the objective shapes.

## Next directions (gated; none auto-merged, >= 3 seeds before any default flip)

1. **Target the actual policy input.** Apply the control objective (or the policy
   gradient itself) to the broadcast the policy consumes, not upstream tectum
   content. This requires resolving the in-place detach obstacle (train_rlhf.py:725-734),
   e.g. an off-graph forward model trained on stored (broadcast, action,
   next_broadcast) tuples, decoupled from the tectum optimizer step.
2. **Log the control-repr loss** and tune its weight before concluding the
   objective form is wrong.
3. **Reconsider whether the GNW broadcast is the right policy input at all.** DQN on
   pixels reaches 92; whatever survives the tectum -> capsule pooling -> GNW
   competition -> detach pipeline plateaus at ~15. The lossy stage is between pixels
   and broadcast.

## Reproducibility

```bash
# Step 1 confirmation
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env dark_room \
  --episodes 120 --max-steps 100 --policy dqn --seed 42 --phi-sample-every 5 \
  --log-dir runs/p5_dqn_broadcast
# Step 2 fix attempt (ON); OFF is runs/p5_probe/gonogo from the diagnosis
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env dark_room \
  --episodes 120 --max-steps 100 --seed 42 --phi-sample-every 5 \
  --enable-control-repr --log-dir runs/p5_cr_on
```
