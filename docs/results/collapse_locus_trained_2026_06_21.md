# Collapse-locus re-probe on a TRAINED tectum (2026-06-21): diagnosis CONFIRMED

**Verdict: CONFIRMED. Stimulus identity still dies at the RSSM step on a TRAINED
tectum, exactly as on the untrained init. The 2026-06-16 finding was NOT an untrained
artifact.** obs_map decodes identity at ~1.0 both untrained and trained; z_state and
everything downstream sit at chance in both conditions. The reward-prediction RSSM
objective does not put stimulus identity into z_state, so training it does not close
the gap. The cheap narrow projection fix (R3) stays ruled out; R1 (a world-model
reconstruction objective on z_state) now has a confirmed, precise target.

All numbers below were loaded from disk this session
(`runs/collapse_trained/probe_untrained.txt`, `runs/collapse_trained/probe_trained.txt`,
`runs/collapse_trained/episodes.csv`).

## Method

Confirmatory A/B of the leakage-free collapse-locus probe
(`scripts/analysis/probe_collapse_locus.py`, one record per trial, DMTS, seed 42,
n=280). The only difference between arms is the tectum weights:

- **untrained**: the original probe (untrained init).
- **trained**: same probe with the new `--load-tectum runs/collapse_trained/tectum.pt`
  arg (default None keeps the untrained path bit-identical; the threading change is the
  only code edit this session, +10/-3 lines).

"Trained" = a plain DMTS run, 100 episodes, seed 42
(`python -m scripts.training.train_rlhf --env dmts --episodes 100 --seed 42
--phi-sample-every 5 --save-tectum runs/collapse_trained/tectum.pt`). Plain defaults, so
the tectum optimizer block (reward-predictor MSE + TDANN) is exactly the objective the
diagnosis names. No `--enable-recon`, no `--enable-match-head`, no `--policy-input`
override (none of those touch the tectum optimizer).

Honest characterization of "trained": the agent did NOT learn DMTS. total_reward was
flat and negative (first-10 episodes -11.38, last-10 -12.38, min -16.68, max -7.60), as
in every prior DMTS run. So "trained" means the tectum optimizer ran ~4000 steps on a
near-flat reward signal, not that the agent gained task competence. That is the correct
test: the diagnosis is about the tectum's training OBJECTIVE, which runs regardless of
RL competence.

## Result (A/B, chance = 0.167, 6 classes)

| stage | untrained shape (lin/mlp) | trained shape (lin/mlp) | untrained color (lin/mlp) | trained color (lin/mlp) |
|-------|--------------------------:|------------------------:|--------------------------:|------------------------:|
| obs_map (16384-D) | 0.988 / 0.988 | **0.988 / 0.976** | 1.000 / 1.000 | **1.000 / 0.988** |
| z_state RSSM (262144-D) | 0.214 / 0.179 | **0.226 / 0.131** | 0.226 / 0.119 | **0.226 / 0.119** |
| capsule_poses (64-D) | 0.131 / 0.119 | **0.143 / 0.167** | 0.131 / 0.167 | **0.179 / 0.179** |
| tectum_content (256-D) | 0.190 / 0.190 | **0.143 / 0.143** | 0.226 / 0.226 | **0.179 / 0.179** |

Trained is within noise of untrained at every stage. Training changes essentially
nothing about identity decodability along the pipeline.

The z_state linear numbers (0.21 to 0.23) sit marginally above uniform chance, but the
stronger PCA+MLP decoder lands at or below chance (0.119 to 0.131) in both arms, so
there is no real identity signal in z_state; the small linear excess is majority-class
noise, present untrained and trained alike. obs_map is the only stage that carries
identity, and it carries it near-perfectly in both arms.

## Decision-rule outcome (stated before running, the cleanest branch)

Per the plan's decision rule, this is outcome 1: **trained z_state still at chance AND
trained obs_map still high (~1.0).** obs_map preserves identity, the RSSM discards it,
and training the RSSM on reward does not fix it. The reward-only RSSM objective is
confirmed as the locus of the loss. R1 (reconstruct from z_state via a world-model
objective so z_state is forced to encode identity) has a confirmed, precise target.

## Reconciliation with the match-head 06-16 finding (no contradiction)

The match-head A/B (`dmts_match_head_2026_06_15.md`, Update 2026-06-16) found that
training the tectum DEGRADES obs_map's MATCH content (frozen 0.746 vs trained 0.458).
This probe finds training PRESERVES obs_map IDENTITY (~1.0 in both arms). These are
different quantities and both hold:

- **Identity** (decode this stimulus's shape/color from obs_map) is a per-stimulus
  property the retinotopic encoder retains through training.
- **Match** (decode which choice equals the held sample from `[obs_map ; held sample]`)
  is a relational property across two stimuli; tectum training corrupts it.

So "tectum training degrades obs_map" is true for relational/match content, not for
per-stimulus identity. The collapse-locus identity story (identity dies at the RSSM,
not at the encoder) holds cleanly in both trained and untrained conditions.

## Honest caveats

- Single seed (42). This is a confirmatory diagnostic, not a default change, so a
  single seed is acceptable, but it is reported as a confirmation within seed 42, not a
  law. The effect is large and clean (1.0 -> chance at the RSSM, both arms), so a
  multi-seed sweep is unlikely to overturn it, but it has not been run.
- "Trained" is the reward-predictor + TDANN objective on a flat-reward task. The
  reward-predictor had little signal to give, so the strongest reading is "this
  objective does not place identity in z_state," which is exactly the point.
- This probe confirms the loss persists under the CURRENT objective. It does NOT show
  that a RECONSTRUCTION-trained RSSM would preserve identity. That is the untested R1
  question: whether forcing z_state to reconstruct obs_map (or frozen DINOv2 features,
  per `docs/generative_world_models_perception.md`) makes z_state encode identity. R1
  must be built and measured FAILED-first, >=3 seeds, before any claim.
- The GNW bottleneck is partly BY DESIGN (a low-capacity integrated workspace). R1 must
  deliberately choose how much identity to preserve vs compress; the tension may not be
  fully resolvable.

## Strategic implication

- **R3 (narrow projection/readout fix): stays RULED OUT.** The loss is upstream of
  capsule_poses and tectum_content, at the RSSM, trained and untrained.
- **R1 (world-model ELBO from z_state): confirmed target.** The reward-only RSSM
  objective demonstrably fails to encode identity even after training. The principled
  fix is to add a reconstruction objective at the RSSM latent. This is genuinely
  different from the FAILED 2026-06-10 reconstruction, which decoded from the
  post-collapse tectum_content. Design inputs in `docs/active_inference_unification.md`
  and `docs/generative_world_models_perception.md` (the reconstruction target can be
  frozen DINOv2 patch features, not raw pixels).
- **R2 (measure as-is): now has a sharp, confirmed finding to report.** The
  architecture's world model is trained on reward, not reconstruction, and the
  integration pathway discards task-relevant identity at the RSSM, trained and
  untrained.

Decision (R1 vs R2) is the user's call, with this confirmed table in hand.
