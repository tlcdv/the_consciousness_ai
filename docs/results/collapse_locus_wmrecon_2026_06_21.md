# R1 world-model reconstruction (wm-recon) validation (2026-06-21): FAILED

**FAILED.** Adding a reconstruction objective on the RSSM latent (--enable-wm-recon:
rebuild the downsampled frame from cat([h_t, z_flat]), the collapse locus) did NOT make
the RSSM latent or anything downstream decode stimulus identity. z_state, capsule_poses,
and the policy-relevant tectum_content all stay at chance (~0.167), statistically
indistinguishable from the reward-only and untrained arms. obs_map stays at ~1.0 in all
three. The objective genuinely trained (recon_loss fell ~15x), so this is not an
untrained-head confound: reconstruction pressure at the locus did not put identity into
the latent.

All numbers loaded from disk this session
(`runs/wmrecon_trained/probe_wmrecon.txt`, `runs/collapse_trained/probe_trained.txt`,
`runs/collapse_trained/probe_untrained.txt`, `runs/wmrecon_trained/{episodes,metrics}.csv`).

## What R1 is (and how it differs from the FAILED 2026-06-10 recon)

R1 (active-inference stage 1; `models/core/rssm_reconstruction.py`,
`--enable-wm-recon`, commit 3f34294) reconstructs the current downsampled frame from the
PRE-CAPSULE RSSM latent cat([h_t, z_flat]). The 2026-06-10 reconstruction
(`tectum_reconstruction_2026_06_10.md`) reconstructed from the POST-COLLAPSE
tectum_content and failed because identity was already gone before tectum_content. The
collapse-locus probe (confirmed on a trained tectum 2026-06-21,
`collapse_locus_trained_2026_06_21.md`) localizes the loss to the RSSM step, so R1
sources the reconstruction at the locus. That was the untested hypothesis. It FAILED too.

## Method

Trained a tectum with --enable-wm-recon (plain DMTS, 100 episodes, seed 42,
`runs/wmrecon_trained/tectum.pt`), then re-ran the leakage-free collapse-locus probe
(`scripts/analysis/probe_collapse_locus.py --load-tectum ...`, one record per trial,
n=280, seed 42). Compared against the reward-only trained arm
(`runs/collapse_trained/tectum.pt`, 2026-06-21) and the untrained arm, all same probe,
same seed.

Run characterization (from disk):
- recon_loss TRAINED DOWN: first 1.56e-2 -> last 1.07e-3, min 3.97e-5 (~15x reduction
  over 100 episodes). The reconstruction objective ran and the head learned to
  reconstruct. A "failed" verdict is NOT a "the head never trained" confound.
- Reward flat and negative (first-10 -11.12, last-10 -11.82), as in every DMTS run. The
  agent did not learn DMTS; "trained" means the tectum + recon objective ran ~4000
  optimizer steps. Same as the reward-only arm, so the comparison is controlled.

## Result (chance = 0.167, 6 classes)

| stage | untrained (lin/mlp) | reward-only (lin/mlp) | wm-recon R1 (lin/mlp) |
|-------|--------------------:|----------------------:|----------------------:|
| obs_map shape | 0.988 / 0.988 | 0.988 / 0.976 | 0.988 / 0.988 |
| obs_map color | 1.000 / 1.000 | 1.000 / 0.988 | 1.000 / 0.988 |
| z_state shape | 0.214 / 0.179 | 0.226 / 0.131 | 0.155 / 0.179 |
| z_state color | 0.226 / 0.119 | 0.226 / 0.119 | 0.226 / 0.202 |
| capsule_poses shape | 0.131 / 0.119 | 0.143 / 0.167 | 0.179 / 0.250 |
| capsule_poses color | 0.131 / 0.167 | 0.179 / 0.179 | 0.131 / 0.179 |
| tectum_content shape | 0.190 / 0.190 | 0.143 / 0.143 | 0.083 / 0.190 |
| tectum_content color | 0.226 / 0.226 | 0.179 / 0.179 | 0.167 / 0.190 |

The three arms are indistinguishable at every post-RSSM stage: obs_map ~1.0, everything
from z_state on at chance. The two largest wm-recon numbers (capsule_poses shape
pca+mlp 0.250, z_state color pca+mlp 0.202) are within noise of chance: the n=84 test
set gives wide variance, the linear counterparts are at chance, and the other label
(color/shape) is at chance, so there is no consistent rise. Against obs_map's 0.99-1.0,
no stage recovered identity.

## Verdict

**FAILED.** R1-as-built (reconstruct the raw frame from the discrete RSSM latent) does
not repair the collapse. The reconstruction objective trained (recon_loss -15x) but did
not make the latent encode stimulus identity. R3 stays ruled out (the loss is upstream
of the projection); R1-pixels is now also ruled out for this latent and target.

## Why the objective trained but identity did not appear (hypotheses, not asserted)

1. **Reconstruction does not imply identity (leading explanation).** recon_loss falling
   ~15x proves the decoder can minimize foreground-weighted reconstruction error, NOT
   that the latent linearly or non-linearly encodes which-shape/which-color. On
   low-diversity, sparse DMTS stimuli, a decoder can minimize the loss with an average /
   blurry reconstruction (centered blob of the mean color) that does not require the
   latent to carry identity. The probe (PCA+MLP, non-linear) reads identity off the
   latent at chance, so whatever the latent encodes, it is not recoverable identity.
   This is the same lesson as 2026-06-10 (reconstruction is architecturally
   insufficient), now shown at the locus, not just post-collapse.
2. **The discrete categorical RSSM latent may lack the capacity / gradient.** z_t is a
   32x32 categorical sampled with gumbel-softmax straight-through. STE gradients through
   a hard discrete bottleneck are weak and may not carry enough signal to encode fine
   stimulus identity under reconstruction pressure.
3. **The pixel/frame target may be the wrong target.** DINO-WM
   (`docs/generative_world_models_perception.md`) predicts FROZEN DINOv2 patch features
   precisely because they preserve identity by construction; raw pixels invite
   average-reconstruction collapse. Training here uses the conv-fallback encoder (no
   frozen DINOv2), so the DINO-feature target was not available and was not tested.

## Honest caveats

- Single seed (42). A confirmatory negative, not a law, but the effect is clean (all
  post-RSSM stages at chance in all three arms, large gap to obs_map).
- The trained recon head is not saved (--save-tectum saves only the tectum), so the
  reconstruction the head learned (faithful vs average-blob) was not inspected. The
  average-reconstruction explanation is a hypothesis, not a confirmed mechanism. The
  headline (the latent does not decode identity) does not depend on it.
- "Trained" is the recon + reward-MSE + TDANN objective on a flat-reward DMTS task.

## What this leaves (the user's call, FAILED-first)

- **R1-pixels: FAILED.** Reconstructing the raw frame from the discrete RSSM latent does
  not repair the collapse.
- **DINO-feature target (option A, untested):** predict frozen DINOv2 patch features
  from the latent instead of pixels. Needs frozen DINOv2 enabled (the project defaults
  to the conv fallback). This is the design doc's preferred, identity-preserving target
  and the natural next R1 variant, but it is a larger change (enable + freeze DINOv2,
  changing the encoder from the diagnosis baseline) and must clear the same FAILED-first,
  >=3-seed bar. No promise it works.
- **The discrete latent itself may be the limit.** A continuous or higher-capacity RSSM
  latent might encode identity where the categorical one does not; also a larger
  architectural change.
- **R2 (measure as-is):** the honest off-ramp. Report that the perception collapse is
  robust to a reconstruction objective at the locus, run the section-13
  substrate-independence test + Butlin rubric on the current agent, and document that
  causal efficacy is blocked by the collapse.

The R1 mechanism (`--enable-wm-recon`) stays in the codebase, default off, as a
documented negative result, like the 2026-06-10 recon head.
