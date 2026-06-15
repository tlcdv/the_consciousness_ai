# Collapse-locus probe (2026-06-16): stimulus identity dies at the RSSM

**Verdict: the information loss is DEEP, at the RSSM step (`obs_map -> z_state`), not
at the capsule routing and not at the final 256-D projection.** The cheap narrow fix
(repair the projection / readout) is ruled out. The locus is a core architecture
component, with a precise and principled fix. All numbers loaded from disk
(`runs/collapse_locus.txt`) in the session that wrote this doc.

## Method

`scripts/analysis/probe_collapse_locus.py`: leakage-free (one record per trial) decode
of the on-screen sample identity (shape, color) from each stage of the tectum forward
chain, under probe conditions (untrained components, scripted policy, no_grad). DMTS,
14 episodes, n=280, seed 42. Decoders: linear and PCA-80+MLP. Reuses
`_build_components`, `frame_to_tensor`, `linear_decode` from
`probe_perception_decodability.py`.

## Result

Pipeline order: `obs_map -> RSSM z_state -> capsule poses -> tectum_content (256-D)`.
Chance = 0.167 (6 classes).

| stage | dim | shape (lin / mlp) | color (lin / mlp) |
|-------|----:|------------------:|------------------:|
| obs_map | 16384 | 0.99 / 0.98 | 1.00 / 1.00 |
| z_state (RSSM latent) | 262144 | 0.18 / 0.14 | 0.23 / 0.14 |
| capsule_poses | 64 | 0.13 / 0.10 | 0.24 / 0.18 |
| tectum_content | 256 | 0.19 / 0.19 | 0.23 / 0.23 |

Identity is ~perfect in `obs_map` and at chance from `z_state` onward. The drop happens
at the very first stage after `obs_map`: the RSSM step. The capsule poses and the final
projection operate on already-identity-free input; they are downstream victims, not the
cause.

## Why (architectural diagnosis)

The RSSM (DreamerV3-style world model) in this project is trained to predict REWARD
(reward-predictor + BPTT through `tectum_content`), NOT to reconstruct observations. A
world model trained on reward has no pressure to retain stimulus identity, so it
compresses `obs_map` into a reward-relevant latent that discards it. This also explains
why the 2026-06-10 reconstruction head FAILED
(`tectum_reconstruction_2026_06_10.md`): it reconstructed from `tectum_content` (AFTER
the collapse), not from the RSSM latent (where the loss occurs).

## Honest caveats

- Untrained components, single seed (42). The localization (1.00 -> chance at the RSSM)
  is large and clean, but whether a RECONSTRUCTION-trained RSSM would preserve identity
  is untested here. That is exactly the R1 fix to evaluate.
- The strategic conclusion does not depend on the caveat: the loss is upstream of the
  capsules and the projection, so the narrow projection/readout fix (R3) is ruled out
  regardless of whether the RSSM is trained or not.
- This probe locates the loss; it does not dissolve the design tension. A low-capacity
  integrated workspace is partly BY DESIGN in GNW (consciousness as a limited-capacity
  bottleneck). The fix has to deliberately choose how much to preserve vs compress.

## Strategic implication (feeds the R1/R2/R3 decision)

- **R3 (cheap narrow fix): RULED OUT.** The loss is not at the final projection.
- **R1 (principled redesign): now has a PRECISE target.** Train the RSSM as a real
  generative world model, reconstruct `obs_map` (or the frame) from `z_state` via the
  ELBO (active-inference stage 1, `active_inference_unification.md`, roadmap Phase 6).
  This is DIFFERENT from the failed 2026-06-10 reconstruction (which decoded from the
  post-collapse `tectum_content`); reconstructing from the RSSM latent is the standard
  world-model objective that forces `z_state` to encode identity.
- **R2 (measure as-is): gets a sharp, honest finding.** The architecture's world model
  is not trained as a world model; the integration pathway discards task-relevant
  identity at the RSSM. That is a concrete, reportable result about the current system.

## Confirmatory next step (before committing to R1)

Probe the TRAINED RSSM `z_state`: load a trained tectum into the probe (or add
`--load-tectum`) and re-decode. If the trained `z_state` is also at chance, the
reward-only objective is confirmed as the cause and the world-model ELBO is the fix. If
the trained `z_state` recovers identity, the loss was an untrained artifact and the
picture changes. Single cheap run, gated, FAILED-first, before any R1 build.
