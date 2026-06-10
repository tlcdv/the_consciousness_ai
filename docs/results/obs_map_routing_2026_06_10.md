# Routing the policy at the decodable obs_map (2026-06-10): NOT CONFIRMED, the cause is spatial processing + control gradient

Path 1 from the 2026-06-10 reconstruction follow-up. The reconstruction experiment
(`tectum_reconstruction_2026_06_10.md`) showed the `obs_map -> tectum_content`
collapse is architectural and not repaired by reconstruction pressure. The cheaper
workaround: skip the collapse, route the policy at the decodable `obs_map` directly,
and ask whether competence recovers. It does not.

## Enabling fix

Two latent bugs blocked `--policy-input spatial` on the real Go/No-Go policy (it only
ran with the `--policy dqn` diagnostic): `ActionSelectionCore` sized its PFC from
`workspace_dim` not `policy_input_dim` (crash), and the spatial tap was not detached
(backprop corrupted the tectum graph through the tectum optimizer's in-place steps).
Both fixed in commit `e8dabb8` (2 new tests). The flag now works with the project's
actual policy.

## Method

dark_room, seed 42, 100 episodes, 200 max-steps, `--phi-sample-every 5`. The only
difference between arms is `--policy-input`: `broadcast` (the post-GNW 256-D content,
shown at chance by the decodability probe) vs `spatial` (the 16384-D flattened
topographic obs_map, shown perfectly decodable for stimulus identity). dark_room is
the cleanest filter: purely spatial (find the light), the light position is in
obs_map, no working-memory or competence-floor confound. Reward is retired as a
target metric (2026-06-02) but kept as a behavioral sanity signal, used here as a
mechanism test, not a target chase.

## Results (all loaded from disk this session)

| arm | first-30 | last-30 | mean | positive episodes |
|-----|---------:|--------:|-----:|------------------:|
| broadcast (collapsed, at chance) | 35.14 | 24.93 | 29.47 | 30 |
| spatial (decodable obs_map) | 14.30 | 25.47 | 12.75 | 38 |

Reference (DQN baselines, verified from disk this session):

| config | learner | encoder | reward |
|--------|---------|---------|-------:|
| pixels | DQN | CNN | 92.00 (last-100) |
| broadcast | DQN | flat MLP | 14.65 mean / 12.97 last-30 |
| obs_map (spatial) | DQN | flat MLP | 17.14 mean / 22.50 last-30 |
| tectum_content | DQN | flat MLP | 15.93 mean / 17.99 last-30 |

## Verdict: NOT CONFIRMED

Routing the policy at the decodable obs_map did not recover competence. Both arms
converge to ~25 last-30; neither approaches DQN-on-pixels (92.00). The spatial arm
rises (14 -> 25) and finds the light slightly more often (38 vs 30 positive
episodes), but its 16384-D input layer makes it start worse and it plateaus at the
same ~25 as broadcast. There is no jump toward 92 in either arm. Single seed, so a
hypothesis; the robust signal is the absence of any move toward 92, consistent with
the multi-condition pattern below.

## The cause (airtight across two experiments and two learner families)

Every flat readout of every representation plateaus in the 13-29 band; the only
configuration that reaches 92 is a CNN over pixels:

- flat MLP over broadcast: DQN 14.65, Go/No-Go 29.47
- flat MLP over obs_map: DQN 17.14, Go/No-Go 12.75
- flat MLP over tectum_content: DQN 15.93
- CNN over pixels: DQN 92.00

The differentiator is not the representation (obs_map is decodable and spatially
structured [64, 16, 16]) and not the learner (DQN and Go/No-Go land in the same
band). It is the **convolutional / spatial processing**. A flat policy cannot turn a
spatially-structured input into control, even when that input is perfectly decodable
for identity by a linear probe. Identity-decodability (a linear category readout) is
not the same as control-usability (extract the light's position and map it to
movement); the latter needs spatial processing.

Two structural facts complete the picture:

1. The tectum is trained only by auxiliary objectives (reward-prediction MSE + TDANN
   topographic loss, plus the optional reconstruction term that also FAILED). The
   **control objective (policy gradient) never reaches the tectum**: the policy reads
   a detached `policy_state` (broadcast/tectum/spatial are all detached before the
   PFC). So perception is never shaped to be controllable. This is the 2026-06-01
   diagnosis restated structurally ("the broadcast is never optimized to be a
   controllable state").
2. The reconstruction experiment showed the tectum collapse discards stimulus
   identity and resists identity-preserving pressure applied at its output.

Combined: competence needs (a) spatial processing in the perception -> action
pathway and (b) the control objective to shape that pathway. The DQN baseline has
both (a CNN trained end-to-end by the control loss). The biological architecture's
tectum/capsule pathway is supposed to provide (a), but it collapses identity, and (b)
is absent because the policy is detached from perception.

## Engineering directions (keep the biological goal; a fork for the owner)

The DQN's CNN-in-the-policy is biologically wrong (the basal ganglia do not do raw
vision). The biologically faithful fixes put the spatial processing and the control
gradient in the perception pathway:

1. **End-to-end control gradient into the tectum.** Stop detaching the policy input
   so the policy gradient shapes the tectum (the deeper capsule layers learn
   action-relevant spatial composites). Requires coordinating the tectum optimizer's
   in-place updates with the policy backward (the reason the detach exists). Highest
   value, highest risk.
2. **Spatial reader in front of the policy.** Give the policy a small convolutional
   reader over the topographic obs_map [64, 16, 16] instead of a flat GRU over the
   flattened vector. Cheaper, isolates whether spatial processing of obs_map recovers
   competence (the decisive diagnostic for direction 1). Less biologically deep but
   the cleanest next confirmation.
3. **Repair the bottleneck (path 2) with a control objective.** Combine: preserve
   spatial structure through the tectum AND train it by the control loss, not just
   reward-MSE/TDANN.

Recommended next step: direction 2 as a cheap decisive diagnostic (does a conv reader
over obs_map reach toward 92?), then direction 1 as the biological fix if it confirms.
All gated, default-off, FAILED-first, >= 3 seeds before any default change.

## Caveats

- Single seed, dark_room only. The flat-vs-CNN comparison mixes input source (pixels
  vs obs_map) with encoder type (CNN vs flat) for the 92 datapoint; direction 2
  (CNN over obs_map) is exactly the experiment that de-confounds it.
- dark_room reward is retired as a target; used here as a mechanism signal, which is
  legitimate for "can the policy exploit decodable spatial input".

## Direction-1 follow-up (same day): conv reader over obs_map, also NOT a recovery

Implemented `--policy-input spatial-conv` (commit `92843d4`): a convolutional
front-end on the policy's PFC that reshapes the flattened obs_map to [C, H, W],
convolves it, and is trained by the control gradient (the obs_map input stays
detached, so no gradient flows into the tectum). This adds the two ingredients the
diagnosis named (spatial processing + control gradient) on the decodable obs_map.

Result (dark_room, seed 42, 100 ep, all from disk):

| arm | first-30 | last-30 | mean | positive eps |
|-----|---------:|--------:|-----:|-------------:|
| broadcast (flat, at chance) | 35.14 | 24.93 | 29.47 | 30 |
| spatial (flat reader) | 14.30 | 25.47 | 12.75 | 38 |
| spatial-conv (conv reader) | 14.09 | 30.56 | 17.96 | 39 |
| DQN-on-pixels (CNN) | - | - | 92.00 (last-100) | - |

Spatial-conv is the best of the three on last-30 (30.56 vs 25.47 flat-spatial vs
24.93 broadcast) and finds the light most often (39 positive episodes). So spatial
processing helps a little (conv beats flat on the same obs_map). But it does not
recover competence: still ~3x below DQN-on-pixels, in the same 25-30 band. Single
seed.

Reading: the conv result removes "no spatial processing" as the *sole* cause but
leaves a large residual gap. The biological front-end (retinotopic encoder + IE
fusion + RSSM/capsule) appears lossy for the pixel-precise spatial information that
dark_room navigation rewards, even at the obs_map stage that decodes object identity
perfectly. A conv over obs_map cannot recover position information the front-end
already blurred.

## Strategic reframe (the honest consequence)

dark_room reward is retired as a target (2026-06-02 reading #2): the architecture is
biology-first and trades pixel-precise control for integration properties, so an
agent underperforming a pixel-CNN on dark_room is expected, not a defect. The goal is
consciousness signatures on the consciousness-demanding tasks (DMTS/WCST), measured
by the Butlin indicator rubric and the pre-registered substrate-independence test
(preregistered_predictions.md section 13).

Crucially, DMTS/WCST do not need pixel-precise spatial control. They need stimulus
identity (which obs_map decodes at 1.000, per perception_decodability_2026_06_09.md),
plus working memory (DMTS delay) and rule inference (WCST). So the dark_room control
deficit may be orthogonal to whether the agent can enter the DMTS/WCST regimes. The
goal-aligned next test is therefore not more dark_room control work (a retired
metric), but whether a policy with obs_map access (spatial-conv) can enter a
consciousness-demanding regime where perception is not the bottleneck: WCST, where
the card is on-screen at decision and obs_map decodes it at 1.000. Measured by regime
entry (rule changes reached, trials correct), broadcast vs spatial-conv.

## Reproduce

```
export PYPHI_WELCOME_OFF=yes
python -m scripts.training.train_rlhf --env dark_room --episodes 100 --max-steps 200 \
    --seed 42 --policy-input broadcast --phi-sample-every 5 --log-dir runs/p1_broadcast
python -m scripts.training.train_rlhf --env dark_room --episodes 100 --max-steps 200 \
    --seed 42 --policy-input spatial --phi-sample-every 5 --log-dir runs/p1_spatial
```
