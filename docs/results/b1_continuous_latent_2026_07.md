# Continuous RSSM latent (Path B1, 2026-07): PASS. The discrete latent WAS the wall; the collapse moves downstream to the capsule stage

**PASS, with a precise scope.** Replacing the RSSM's discrete gumbel-softmax categorical
latent with a continuous Gaussian latent of the same shape makes the RSSM latent z_state
carry stimulus identity for the first time in the project's history: shape decodes at
0.881 (linear) / 0.714 (pca+mlp) and color at 0.976 / 0.774, against chance 0.167 and an
obs_map control at ~1.0. Every discrete-latent arm ever run (reward, frame reconstruction,
obs_map reconstruction, value-equivalent world model, direct supervision) left z_state at
chance (0.11 to 0.25). The discrete latent was the wall.

Two honest qualifications, both from disk this session:

1. **The identity is structural, not taught.** In the supervised run the latent-id CE
   stayed at the chance floor (Q4 3.65 vs 2 ln 6 = 3.5835), yet z_state decodes at ~0.88.
   A reward-only control (continuous latent, NO `--enable-latent-id`) decodes z_state
   identity just as well (shape 0.976 / 0.726, color 1.000 / 0.845). So the continuous
   latent PRESERVES the obs_map identity through the posterior net by construction; no
   identity-specific objective is needed. This is stronger and cleaner than the ceiling
   test was designed to find: the fix is the representation, not the objective.

2. **The collapse did not disappear, it moved one stage downstream.** With identity now in
   z_state, capsule_poses and tectum_content are STILL at chance in both continuous runs.
   The identity is destroyed at the capsule routing (`HierarchicalCapsuleComposition`),
   which is a NEW, later locus. Perception is not "fixed"; the wall moved from the RSSM to
   the capsule stage.

## The mechanism

`models/core/sensory_tectum.py`, `--rssm-latent-mode continuous` (default `discrete`,
baseline bit-identical, verified: identical state-dict keys, params, and forward output
when off). The continuous mode reuses the existing prior/posterior conv nets as the
Gaussian MEAN and adds one shared per-channel log-variance parameter (`cont_logvar`), so
z_t keeps its `[B, categories, classes, grid, grid]` shape and every downstream stage is
untouched. Training samples by reparameterization; eval returns the mean (deterministic,
so the probe reads a fixed latent). The categorical KL surprise bid becomes the Gaussian
KL between posterior and prior means at the shared variance.

Why it works: the discrete gumbel-softmax draws a hard one-hot per cell, quantizing the
continuous obs_map features into a categorical code that discards the low-variance
identity direction (the 2026-06-21 PCA finding: identity is a low-variance direction of
obs_map). A continuous latent whose mean is a conv of `cat([h_t, obs_map])` passes that
direction through instead of quantizing it away.

## Result (chance = 0.167, n = 280 trials, one record per trial)

| stage | discrete (any arm) | continuous + latent-id | continuous reward-only |
|-------|---:|---:|---:|
| obs_map shape (lin/mlp) | 0.99 / 0.98 | 0.988 / 1.000 | 0.988 / 0.988 |
| obs_map color | 1.00 / 0.98 | 1.000 / 1.000 | 1.000 / 0.976 |
| **z_state shape** | **0.11 to 0.25** | **0.881 / 0.714** | **0.976 / 0.726** |
| **z_state color** | **0.11 to 0.25** | **0.976 / 0.774** | **1.000 / 0.845** |
| capsule_poses shape | chance | 0.190 / 0.190 | 0.226 / 0.190 |
| capsule_poses color | chance | 0.226 / 0.226 | 0.179 / 0.179 |
| tectum_content shape | chance | 0.190 / 0.190 | 0.179 / 0.226 |
| tectum_content color | chance | 0.226 / 0.226 | 0.095 / 0.119 |

## What did NOT change

- **Task reward is flat and negative** in both continuous runs (first-10 to last-10:
  -35.32 to -35.40 supervised, -35.20 to -35.06 reward-only). The agent still does not
  learn DMTS. The z_state identity win is representational only; the RL / credit-assignment
  wall (2026-06-14/15: the policy did not learn DMTS even with identity available) is a
  separate blocker and is untouched here.
- phi (~9.8e-4 continuous vs ~1.1e-3 discrete) and sync_R (0.254 vs 0.267) are close to the
  discrete values; the continuous latent does not by itself move the other signatures.

## Replication: 3 seeds, reward-only (2026-07-06)

The single-seed result was replicated at two further seeds, continuous latent, reward-only
(no identity supervision), DMTS 100 ep each. z_state decodes identity well above chance in
every seed; capsule_poses and tectum_content stay at chance in every seed.

| seed | z_state shape (lin/mlp) | z_state color (lin/mlp) | capsule_poses (shape/color) | tectum_content |
|-----:|:-----------------------:|:-----------------------:|:---------------------------:|:--------------:|
| 42 (reward-only) | 0.976 / 0.726 | 1.000 / 0.845 | chance | chance |
| 43 | 0.988 / 0.976 | 1.000 / 0.988 | chance | chance |
| 44 | 0.988 / 0.988 | 0.988 / 0.952 | chance | chance |

obs_map control ~1.0 in all three. The z_state identity decode is now a REPLICATED result
(3/3 seeds, all far above the 0.167 chance line), not a single-seed hypothesis. The
downstream capsule collapse is equally consistent (3/3 seeds at chance), which makes the
capsule stage the well-supported next locus. The continuous mode stays default-off (the
default change would need a task-competence justification, which this does not yet provide:
reward is still flat, below).

## Honest scope

- **Three seeds (42, 43, 44), single env (DMTS), one machine.** The z_state identity result
  is replicated. What is NOT established: any task-reward benefit (flat, below), and any
  effect in other envs (WCST, dark_room) or at larger scale.
- The linear z_state probe on a 262144-D tap can overfit, but the pca+mlp probe (80 PCs +
  MLP, held-out test) also decodes at 0.71 to 0.85, well above chance, so the signal is
  real, not probe overfit.
- The two continuous runs differ only in the latent-id flag; the near-identical z_state
  decodes are the evidence that supervision is not the cause.
- All numbers loaded from `runs/b1_continuous` and `runs/b1_continuous_rewardonly`
  (episodes.csv, metrics.csv, probe_output.txt) this session.

## Downstream diagnosis: identity dies at the FINAL routing layer (2026-07-06)

With identity now in z_state, a read-only probe inside the capsule hierarchy
(`scripts/analysis/probe_capsule_locus.py`, continuous seed-42 checkpoint, n=280 trials,
leakage-free) decodes the sample shape/color from each internal capsule level. Identity is
carried most of the way and collapses at exactly one layer:

| capsule level (hierarchy) | dim | shape (lin/mlp) | color (lin/mlp) |
|---|---:|---:|---:|
| z_state (input to capsules) | 278528 | 0.583 / 0.595 | 1.000 / 0.738 |
| primary_caps (stride-2 conv) | 4096 | 0.452 / 0.738 | 0.952 / 0.845 |
| routing L0 (16 caps, 12-D) | 192 | 0.345 / 0.690 | 0.798 / 0.869 |
| routing L1 (8 caps, 16-D) | 128 | 0.357 / 0.571 | 0.714 / 0.821 |
| **routing L2 (4 caps, 16-D, FINAL)** | **64** | **0.190 / 0.190** | **0.226 / 0.226** |

Chance ~0.193 (shape) / 0.221 (color). Identity survives the stride-2 primary conv and the
first two routing levels (pca+mlp shape 0.69 to 0.74, color 0.82 to 0.87, all well above
chance), then drops to chance ONLY at the final routing layer, the squeeze from 8 output
capsules (128-D) to 4 output capsules (64-D). The `capsule_poses` and `tectum_content` the
workspace reads are the output of that final layer.

This is precise: the loss is not the whole capsule stack and not mere dimensionality (64-D
can hold 36 identity classes, and the 128-D level one step up still decodes it). It is the
final dynamic-routing-by-agreement step into 4 capsules. Candidate mechanisms for the next
sub-question (not yet tested): the softmax coupling over only 4 outputs plus the agreement
updates average identity away; the squash saturation at the top; or the 4-capsule count is
simply too few to keep 36 identities separable after routing. Fix directions (each a
default-off experiment, none built yet): widen the final level (more output caps or higher
output_dim), read workspace content from routing L1 (8 caps) where identity is still
present, or alter the final routing (fewer agreement iterations / skip connection).

## Capsule fix: identity now reaches tectum_content (2026-07-06)

Acting on the diagnosis, a default-off `--capsule-workspace-source all_levels` projects
workspace_content from the concatenation of every routing level (where identity survives)
instead of only the final 4-capsule level. The returned final capsule_poses and the
structured payloads are unchanged; only what the workspace/policy reads (tectum_content) is
enriched. Baseline bit-identical when off (verified: default == final, identical state-dict
and forward; all_levels changes only `workspace_proj.weight` width). Run: continuous latent
+ all_levels, DMTS 100 ep seed 42 (`runs/capfix_alllevels`).

| tap | continuous (final source) | continuous + all_levels |
|-----|--------------------------:|------------------------:|
| obs_map shape / color | ~1.0 | 0.988/0.976 , 1.000/0.988 |
| z_state shape / color | 0.88/0.71 , 0.98/0.77 | 0.988/0.762 , 1.000/0.869 |
| capsule_poses (final) | chance | shape 0.167/0.238 , color 0.452/0.321 |
| **tectum_content shape** | **chance (~0.19)** | **0.524 / 0.833** |
| **tectum_content color** | **chance (~0.23)** | **0.988 / 0.976** |

For the first time, tectum_content, the 256-D content the policy and the global workspace
read, decodes stimulus identity (pca+mlp shape 0.833, color 0.976, chance 0.167). The full
perception chain is now intact end to end: obs_map -> continuous z_state -> (all capsule
levels) -> tectum_content. As designed, the final capsule_poses stays near chance (all_levels
does not change the final level, only the projection source), so structured payloads that
read capsule_poses are unaffected.

Honest limits: single seed (42); the mode stays default-off. Task reward is still flat and
negative (first-10 -39.09, last-10 -37.88): the agent does not learn DMTS. The perception
chain is complete, but making identity AVAILABLE to the policy did not by itself produce
learning, which is exactly the separate RL / credit-assignment wall (Track C1, next).

## The new fork (owner decision)

The RSSM wall is broken and replicated (3 seeds); the loss is now pinned to the final
capsule routing layer. Status of the threads:

1. **Replication: DONE** (3/3 seeds above; z_state identity confirmed reward-only).
2. **Downstream locus: DIAGNOSED** (final routing layer, above). The next step is a
   default-off capsule experiment to carry identity into tectum_content: widen the final
   routing level, read workspace content from routing L1 (where identity is still present),
   or change the final routing. This is a real architectural bet and a new build, so it is
   an owner decision, not auto-started.
3. **The RL learning wall (Track C1)** remains independent and still gates task competence
   and the section-13 test regardless of how far identity is carried. All perception work
   so far leaves task reward flat: carrying identity to tectum_content is necessary for the
   policy to USE it, but the 2026-06-14/15 result warns it may not be sufficient.
