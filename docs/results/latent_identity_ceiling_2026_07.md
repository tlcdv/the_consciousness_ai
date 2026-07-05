# Latent identity ceiling test (2026-07-05): FAILED. The discrete RSSM latent does not carry identity even under direct supervision

**FAILED.** A supervised cross-entropy on the DMTS sample's shape and color, applied
directly to the pre-capsule RSSM latent with privileged environment labels (the maximum
identity pressure any objective can exert), did not make the latent identity-bearing.
Two independent readouts agree:

1. **In training, the supervised loss never left the chance floor.** Over 1407 labeled
   sample-phase steps across 100 episodes, the CE went from 3.838 (first quarter) to
   3.613 (last quarter); the chance floor for two independent 6-way labels is
   2 ln 6 = 3.584. In-training accuracy stayed at 0.165 to 0.197 against a 0.167 chance
   rate (last-50 events: loss 3.465, accuracy 0.230).
2. **Offline, the leakage-free collapse-locus probe on the trained checkpoint decodes
   z_state at chance.** obs_map control: shape 0.988, color 1.000. z_state: shape
   0.214/0.238 (linear/mlp), color 0.226/0.250, inside the 0.13 to 0.25 noise band every
   prior arm produced. capsule_poses and tectum_content at chance. Nothing moved toward
   the obs_map ceiling.

This closes the question the test was built to answer. Every previous objective at this
locus (reward MSE, frame reconstruction, obs_map reconstruction, the value-equivalent
reward/continue/KL world model) was variance-weighted and could in principle be excused
for ignoring a low-variance identity direction. Direct supervision has no such excuse:
the gradient explicitly demands identity separation, flows into the RSSM (posterior and
prior nets, the ConvGRU, and back through obs_map into the encoder), and still fails.
The discrete gumbel-softmax categorical latent, or its optimization path, is the wall.

Consequence for the escalation ladder: a label-free contrastive objective (InfoNCE and
relatives) through this same discrete latent is NOT motivated; it exerts strictly weaker
identity pressure than the supervision that just failed. The remaining escalation is the
continuous / higher-capacity latent change, which is an architectural decision for the
user, not a default.

## Method

- Head: `models/core/rssm_identity.py` (`RSSMIdentityHead`), 1x1 conv reduce + small MLP
  trunk + two linear class heads (6 shapes, 6 colors). Input is the grad-bearing
  pre-capsule state tensor `cat([h_t, z_flat])` cached by the tectum, the same locus and
  caching mechanism the wm-recon heads used.
- Flag: `--enable-latent-id` in `scripts/training/train_rlhf.py`, default off, baseline
  bit-identical when off. DMTS only; disabled with a warning alongside
  `--enable-wm-predict` (which replaces the single-step tectum optimizer block it lives
  in). Labels are the env's `sample_shape` / `sample_color` from the pre-step info,
  sample phase only. The CE joins the tectum loss sum, so its gradient trains the RSSM,
  not just the head.
- Run: DMTS, 100 episodes, seed 42, `--policy-input obsmem-conv`, `--save-tectum`
  (`runs/latentid_trained`, 100 episode rows verified on disk). Mirrors
  `runs/collapse_trained` (reward-only) exactly except the flag.
- Probe: `scripts/analysis/probe_collapse_locus.py --load-tectum
  runs/latentid_trained/tectum.pt`, leakage-free, one record per trial, n=280 trials.
- Unit tests: `tests/test_rssm_identity.py` (7 tests), including a frozen-head value
  test that the CE falls when the latent is free to move, and a gradient-reaches-latent
  test. Full suite 747 passed, 4 skipped.

## Result (chance = 0.167, 6 classes; prior columns from collapse_locus_wmobs_2026_06_21.md)

| stage | reward-only (lin/mlp) | R1-pixels (lin/mlp) | R1-obs_map (lin/mlp) | latent-id (lin/mlp) |
|-------|----------------------:|--------------------:|---------------------:|--------------------:|
| obs_map shape | 0.988 / 0.976 | 0.988 / 0.988 | 0.988 / 0.988 | 0.988 / 0.988 |
| obs_map color | 1.000 / 0.988 | 1.000 / 0.988 | 1.000 / 0.976 | 1.000 / 0.976 |
| z_state shape | 0.226 / 0.131 | 0.155 / 0.179 | 0.167 / 0.238 | 0.214 / 0.238 |
| z_state color | 0.226 / 0.119 | 0.226 / 0.202 | 0.226 / 0.238 | 0.226 / 0.250 |
| capsule_poses shape | 0.143 / 0.167 | 0.179 / 0.250 | 0.179 / 0.226 | 0.179 / 0.143 |
| capsule_poses color | 0.179 / 0.179 | 0.131 / 0.179 | 0.190 / 0.143 | 0.143 / 0.143 |
| tectum_content shape | 0.143 / 0.143 | 0.083 / 0.190 | 0.107 / 0.155 | 0.107 / 0.155 |
| tectum_content color | 0.179 / 0.179 | 0.167 / 0.190 | 0.167 / 0.190 | 0.202 / 0.202 |

The latent-id column is indistinguishable from every other arm at every post-obs_map
stage. The z_state mlp values (0.238/0.250) match the R1-obs_map arm's 0.238 values,
which the 2026-06-21 verdict already characterized as within-noise for this probe's
test-set size; the linear values sit at 0.214/0.226 with reward-only at 0.226/0.226.

Behavior: reward flat (first-10 mean -35.63, last-10 mean -35.52); the agent did not
learn DMTS, matching every prior arm, so the comparison is controlled.

## Why this is a FAILED verdict and not "inconclusive head"

The pre-stated KILL gate assumed the head's CE would train down while the latent stayed
undecodable. What happened is stronger: the CE itself never left the chance floor. Two
pieces of evidence exclude a defective head or a blocked gradient path:

1. The frozen-head unit value test: optimizing only the latent under the fixed head
   drives the CE down (200 steps, loss strictly lower). The head expresses the task.
2. Precedent on the identical gradient path: the wm-recon heads trained THROUGH the same
   `_last_state_tensor` cache and the same tectum loss sum, and their losses fell 15x
   (pixels) and ~1300x (obs_map) in the same 100-episode regime. The path trains what
   the latent can express.

So the head could not be trained because the latent, as produced by the RSSM from real
observations, never came to separate identities, which is exactly the property under
test. Chance-floor CE plus chance-level offline decode is the failure the ceiling test
was designed to expose.

## Honest scope

- Single seed (42), single env (DMTS), one machine, batch-1 online training with 1407
  labeled events. A larger loss weight, higher LR, or more episodes were not swept; the
  wm-recon precedent (losses falling orders of magnitude in the same regime) argues
  against sample count as the binding factor, but this was not ablated.
- The z_state mlp 0.238/0.250 values are nominally above chance in isolation; they
  match the noise band documented across prior arms and show no consistency with the
  linear probe or across stages. Treated as noise, consistent with prior verdicts.
- All numbers in this document were loaded from `runs/latentid_trained` (metrics.csv,
  episodes.csv, probe_output.txt) in the session that produced them.

## Standing conclusions after this test

- The perception collapse is now robust to reward, reconstruction (two targets),
  value-equivalent prediction, AND direct supervision at the locus. The discrete
  gumbel-softmax RSSM latent is the confirmed wall.
- The `--enable-latent-id` mechanism stays in the repo, default off, as a documented
  negative result alongside the recon and wm-predict flags.
- Next fork (user decision, not a default): replace the discrete latent with a
  continuous / higher-capacity latent (an architectural bet touching the RSSM core), or
  stand on the characterization and the signature-instrument findings
  (`signature_assessment_2026_07.md`). No label-free identity objective on the current
  latent is worth running.
