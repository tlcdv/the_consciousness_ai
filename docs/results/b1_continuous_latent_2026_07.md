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

## Honest scope

- **Single seed (42), single env (DMTS), one machine.** This is a strong hypothesis, not a
  law. Per the project rule, >= 3 seeds are required before any default change; the
  continuous mode stays default-off until then.
- The linear z_state probe on a 262144-D tap can overfit, but the pca+mlp probe (80 PCs +
  MLP, held-out test) also decodes at 0.71 to 0.85, well above chance, so the signal is
  real, not probe overfit.
- The two continuous runs differ only in the latent-id flag; the near-identical z_state
  decodes are the evidence that supervision is not the cause.
- All numbers loaded from `runs/b1_continuous` and `runs/b1_continuous_rewardonly`
  (episodes.csv, metrics.csv, probe_output.txt) this session.

## The new fork (owner decision)

The RSSM wall is broken; two threads open, neither auto-started:

1. **Replicate first (cheap, recommended before building on it):** run the continuous
   reward-only config at 2 more seeds and re-probe z_state, to move the single-seed
   hypothesis to a >= 3-seed result before any default change or downstream work.
2. **Chase the identity downstream:** the collapse now sits at the capsule routing
   (`HierarchicalCapsuleComposition`, `models/core/capsule_composition.py`). A collapse
   probe already localizes it; the question is whether the capsule dynamic-routing
   discards identity the way the discrete latent did, and whether a capsule-level change
   preserves it into tectum_content (what the policy and workspace actually read).

The RL learning wall (Track C1) remains independent and still gates task competence and
the section-13 test regardless of how far identity is carried.
