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
- **Localization (PROVISIONAL, confounded vs the pixel baseline): relative taps
  tie; absolute front-end claim is not established.** DQN reward by tap: pixels
  92.00 -> obs_map 17.14 (pre RSSM/capsule) -> tectum_content 15.93 (post capsule)
  -> broadcast 14.65 (post GNW). The comparison AMONG the three taps is fair (shared
  setup) and shows the capsule collapse and GNW add only a small further loss, so
  they are not the lever. But the comparison TO pixels (92) is confounded on three
  axes (see "Confound caveat" below), so "the front-end specifically loses the
  signal" is NOT established by this probe. What holds: the policy is not the
  bottleneck, and nothing on the broadcast beats ~15 under these short,
  high-exploration runs.

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

## Localization follow-up (same day): WHERE is the signal lost?

The fix failed most plausibly because the policy reads the post-GNW broadcast while
the objective shaped tectum content upstream. Before building a broadcast-targeted
fix, localize the lossy stage by running the SAME DQN learner on the pre-GNW tap
(`--policy-input tectum`) vs the post-GNW broadcast.

| tap | DQN reward | source |
|-----|------------|--------|
| pixels | **92.00** (last-100; first-100 was 52.72) | runs_baseline/baseline_dark_room.csv, 1000 ep |
| tectum_content (pre-GNW) | **15.93** | runs/p5_dqn_tectum, 120 ep mean |
| broadcast (post-GNW) | **14.65** | runs/p5_dqn_broadcast, 120 ep mean |

(Window caveat: the pixels baseline is the last-100 of a 1000-ep run; the two taps
are 120-ep means. Even the pixels FIRST-100, 52.72, is 3x above either tap, so the
gap holds regardless of window.)

**Verdict: the control-relevant signal is lost in pixels -> tectum_content, NOT in
the GNW.** tectum_content (15.93) ties the broadcast (14.65); the GNW competition,
reentrant settling, fusion, and detach stage costs almost nothing. The lossy stage
is the tectum ENCODER: retinotopic conv -> RSSM -> capsule composition -> workspace
projection, which collapses the spatially-rich frame into a 256-D vector a learner
cannot control from.

This overturns the "next direction" the Step-2 section proposed (target the
post-GNW broadcast): that fix would have inherited the same ceiling, because
tectum_content is already lossy. Localizing first prevented building it. It also
explains why the Step-2 control objective (which shaped tectum_content via a
next-observation forward model) failed: the encoder's pooling discards the
control-relevant spatial structure, and a next-observation prediction head did not
recover it.

### Second localization (same day): is the capsule collapse the lever?

The first localization said the loss is before the GNW. The natural next
hypothesis was the capsule composition / workspace projection collapsing the
spatial grid into 256-D. Tested it with a third tap (`--policy-input spatial`):
the same DQN reads the topographic `obs_map` (post retinotopic-encoder + inverse-
effectiveness fusion, PRE RSSM, PRE capsule; 16x16x64 = 16384-D, flattened).

| tap | DQN reward | stage |
|-----|------------|-------|
| pixels | **92.00** last-100 (105.31 overall, 1000 ep) | raw input |
| obs_map (spatial) | **17.14** (first30 19.34, last30 22.50) | post encoder + fusion |
| tectum_content | 15.93 | post capsule collapse |
| broadcast | 14.65 | post GNW |

**Verdict: the loss is at the FIRST encoding stage, pixels -> obs_map.** Even the
earliest spatial representation, before RSSM and before the capsule collapse, is
already at ~17, ~5x below pixels. The capsule collapse (obs_map 17.14 ->
tectum_content 15.93) and the GNW (-> broadcast 14.65) each add only a small
further loss. This RULES OUT the capsule/RSSM-collapse hypothesis as the lever: the
bulk of the 92 -> 17 drop happens in the retinotopic encoder + 16x16 fusion
front-end, not downstream.

(obs_map does show a small upward trend, first30 19.34 -> last30 22.50, that the
post-collapse taps lack, so spatial structure is marginally more learnable; but the
ceiling is still ~20, not ~92.)

## Confound caveat (added 2026-06-02 on re-examination)

A self-review found the tap-vs-pixels comparison is confounded on three axes. The
"DQN learner held constant" framing is only true AMONG the three taps (obs_map,
tectum_content, broadcast), not between the taps and the pixel baseline. Verified:

1. **Exploration schedule.** `DQNPolicy` decays epsilon over 50,000 steps; the tap
   runs are 12,000 steps (confirmed: 12,000 metric rows each), so epsilon ended at
   ~0.772 - the probe DQNs took ~77% random actions even at the end of training. The
   pixel baseline (`train_baseline_dqn.py`) decays over 500 EPISODES and spent its
   last ~500 of 1000 episodes at epsilon=0.05 (exploiting). The probes were mostly
   random; the baseline mostly exploited.
2. **Network architecture.** The pixel baseline uses a CNN; the tap DQNs use an MLP
   over a flattened vector. For the spatial obs_map tap (64x16x16) this discards the
   spatial structure a CNN would exploit - penalizing the very tap meant to test
   spatial richness.
3. **Training budget.** 120 episodes (taps) vs 1000 (pixels). obs_map was still
   rising (first30 19.34 -> last30 22.50), consistent with undertraining, not a
   ceiling.

What this means: the RELATIVE result (capsule collapse and GNW add little; policy is
not the bottleneck, corroborated by the properly-trained Go/No-Go and A2C) stands.
The ABSOLUTE claim ("the perceptual front-end is where the control signal is lost,
vs pixels") is NOT established - all three confounds push the taps below pixels
independent of representation content. To establish it would require a de-confounded
re-run (match the epsilon schedule, use a CNN over the spatial tap, equal episodes).
That re-run was judged low-value because reading #2 (below) retires control reward as
a target, so the localization no longer drives a build decision; it is available if
the absolute claim is ever needed.

## Resolved picture and honest readings

The chain (pixels 92 -> obs_map 17 -> tectum_content 16 -> broadcast 15) is
SUGGESTIVE of a perceptual-front-end bottleneck, but per the confound caveat above
the pixels-vs-taps step is not a clean measurement. The defensible conclusion is
narrower: the downstream consciousness machinery (RSSM, capsule hierarchy, GNW
competition, reentrant, detach) is NOT the primary cost (relative taps tie), and no
broadcast-side learner exceeds ~15 in these runs. Whether the front-end is the
specific lossy stage versus pixels is provisional pending a de-confounded re-run.

Two honest readings, not yet decided:
1. **Under-resourced front-end.** The encoder is 16x16 spatial, trained for
   reward-prediction + TDANN topography, not control. A control-trained or
   higher-resolution front-end might recover reward. This is a large change to a
   deliberately biological component (DINOv2 / retinotopic + topographic loss).
2. **Inherent to the design.** The architecture's perception is built for biological
   and topographic fidelity and for feeding a low-dimensional integrated workspace,
   not pixel-precise control. On a task that rewards precise localization
   (dark_room), a ~15-20 ceiling is the cost of that design. dark_room reward is not
   the project's success metric; the consciousness signatures are. Under this
   reading the agent underperforming a pixel DQN is expected, not a defect.

The localization is solid (numbers from disk). Which reading holds is the open
question, and it is a design/scope decision for the project owner, not a bug to
fix blindly.

**Next directions (gated; >= 3 seeds before any default flip, none auto-merged):**
1. If pursuing reading 1: probe a higher-resolution / control-trained front-end
   (e.g. larger grid, or let a control objective reach the encoder) and measure DQN
   reward recovery. Large change; measure before committing.
2. If accepting reading 2: stop optimizing dark_room reward as a target, document
   that the architecture trades control performance for its biological/integration
   properties, and judge the agent by the consciousness signatures instead.

## Decision (2026-06-02): reading #2 adopted

Reading #2 is adopted. dark_room control reward is retired as a target metric; success
is judged by consciousness signatures, formalized as the indicator rubric in
[`consciousness_indicators_butlin.md`](../consciousness_indicators_butlin.md). Full
rationale, the honest "perception must support the consciousness tasks" caveat, and the
research-backed forward path (Butlin indicators for evaluation; active inference for a
biologically principled front-end) are in
[`decisions/2026_06_02_competence_reading_2.md`](../decisions/2026_06_02_competence_reading_2.md).

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
# Localization: same DQN learner on the pre-GNW tap
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env dark_room \
  --episodes 120 --max-steps 100 --policy dqn --policy-input tectum --seed 42 \
  --phi-sample-every 5 --log-dir runs/p5_dqn_tectum
```
