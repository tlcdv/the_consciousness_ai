# R1 obs_map reconstruction target validation (2026-06-21): FAILED (reconstruction family exhausted)

**FAILED.** Reconstructing the DENSE, identity-rich obs_map feature map from the RSSM
latent (instead of the sparse pixel frame) did NOT make the latent decode stimulus
identity, even though the obs_map reconstruction trained to near-perfect fidelity
(MSE 1.97e-4, ~1300x reduction). z_state, capsule_poses, and tectum_content all stay at
chance. This REFUTES the "sparse pixel target was the problem" hypothesis from
`collapse_locus_wmrecon_2026_06_21.md`: reconstruction fidelity does not imply
identity-decodability of the latent, and this now holds for BOTH a sparse pixel target
(R1-pixels) and a dense identity-rich feature-map target (R1-obs_map). The
reconstruction-family R1 is exhausted; a third reconstruction target (DINOv2 features)
would be the same MSE-reconstruction family and is not worth running.

All numbers loaded from disk this session (`runs/wmobs_trained/probe_wmobs.txt`,
`runs/wmobs_trained/{episodes,metrics}.csv`, compared to
`runs/wmrecon_trained/probe_wmrecon.txt` and `runs/collapse_trained/probe_*.txt`).

## What changed from R1-pixels

R1-pixels reconstructed the downsampled RGB frame. R1-obs_map (--wm-recon-target obs_map,
commit 3afaf33) reconstructs the obs_map feature map (decodes identity at ~1.0, the
DINO-WM-style content-map target without a frozen DINOv2). A dense target was meant to
resist the average-collapse a sparse pixel target invites. The source (RSSM latent
cat([h_t, z_flat])) and everything else are identical.

## Method

Trained a tectum with --enable-wm-recon --wm-recon-target obs_map (plain DMTS, 100
episodes, seed 42, `runs/wmobs_trained/tectum.pt`), then re-ran the leakage-free
collapse-locus probe (--load-tectum, n=280, seed 42).

Run characterization (from disk):
- obs_map recon_loss (plain MSE on the feature map) TRAINED DOWN HARD: first 2.54e-1 ->
  last 1.97e-4, min 8.35e-6 (~1300x). The latent reconstructs obs_map to near-perfect
  fidelity. This is a much stronger reconstruction than R1-pixels (~15x). Not an
  untrained-head confound.
- Reward flat (first-10 -11.47, last-10 -11.08). Agent did not learn DMTS, same as every
  prior arm; the comparison is controlled.

## Result (chance = 0.167, 6 classes)

| stage | reward-only (lin/mlp) | R1-pixels (lin/mlp) | R1-obs_map (lin/mlp) |
|-------|----------------------:|--------------------:|---------------------:|
| obs_map shape | 0.988 / 0.976 | 0.988 / 0.988 | 0.988 / 0.988 |
| obs_map color | 1.000 / 0.988 | 1.000 / 0.988 | 1.000 / 0.976 |
| z_state shape | 0.226 / 0.131 | 0.155 / 0.179 | 0.167 / 0.238 |
| z_state color | 0.226 / 0.119 | 0.226 / 0.202 | 0.226 / 0.238 |
| capsule_poses shape | 0.143 / 0.167 | 0.179 / 0.250 | 0.179 / 0.226 |
| capsule_poses color | 0.179 / 0.179 | 0.131 / 0.179 | 0.190 / 0.143 |
| tectum_content shape | 0.143 / 0.143 | 0.083 / 0.190 | 0.107 / 0.155 |
| tectum_content color | 0.179 / 0.179 | 0.167 / 0.190 | 0.167 / 0.190 |

R1-obs_map is indistinguishable from R1-pixels and reward-only at every post-RSSM stage:
obs_map ~1.0, z_state / capsule_poses / tectum_content at chance. The two largest
R1-obs_map values (z_state mlp 0.238 on both labels) are not a consistent signal: the
linear z_state shape is exactly at chance (0.167), and across the literature of these
probes an isolated PCA+MLP value ~0.24 on n=84 test is within noise. Nothing recovered
toward obs_map's 0.99-1.0.

## Verdict

**FAILED, and the reconstruction-family R1 is exhausted.** Two reconstruction targets at
the RSSM latent, one sparse (pixels) and one dense and identity-rich (obs_map), both
trained their reconstruction loss down (15x and 1300x) and neither made the latent decode
identity. R3 was already ruled out. R1-via-reconstruction is now ruled out for this
latent, independent of target richness.

## Why (the key finding, CONFIRMED)

**Reconstruction fidelity does not imply identity-decodability, because identity is a
low-variance direction in obs_map.** The latent reconstructs obs_map at MSE ~2e-4 yet a
PCA-80+MLP probe reads identity off the latent at chance. The mechanism was confirmed
with a direct read-only analysis (`scripts/analysis/probe_obsmap_variance.py`, on the
wmobs trained tectum, seed 42, n=280, numbers from `runs/wmobs_trained/obsmap_variance.txt`):
decode shape/color from the top-k principal components of obs_map (the highest-variance
subspace, which is what an MSE-optimal rank-k reconstruction keeps).

| top-k PCs | cum. variance | shape acc | color acc |
|----------:|--------------:|----------:|----------:|
| 2 | 0.595 | 0.179 (chance) | 0.679 |
| 5 | 0.766 | 0.333 | 0.810 |
| 10 | 0.880 | 0.393 | 0.964 |
| 20 | 0.963 | 0.512 | 1.000 |
| 50 | 0.999 | 0.976 | 1.000 |

The top-20 PCs capture 96.3% of obs_map's variance but decode SHAPE at only 0.512; shape
only fully decodes (0.976) once ~50 PCs are included, reaching 99.9% variance. So the
shape-discriminating signal lives in the low-variance tail (the PCs between 96.3% and
99.9% cumulative variance). An MSE reconstruction that captures the high-variance bulk
(low MSE) discards exactly that low-variance identity direction, and the discrete RSSM
latent (gumbel-softmax STE) has no pressure from MSE to keep it. Color is more
high-variance (decodes 0.679 from the top-2 PCs), but shape is clearly low-variance. This
is why both a sparse pixel target and a dense obs_map target fail the same way, and why a
third MSE-reconstruction target (e.g. DINOv2 features) would fail identically.

## Honest caveats

- Single seed (42). A clean, large effect (all post-RSSM stages at chance across three
  arms), but a confirmation within seed 42, not a law.
- "Trained" is the recon + reward-MSE + TDANN objective on a flat-reward DMTS task.
- The low-variance-identity mechanism is now CONFIRMED by the PCA analysis above (shape
  needs ~50 PCs / 99.9% variance to decode; the top-20 PCs / 96.3% variance give only
  0.512). The headline (reconstruction does not repair the collapse, two targets) holds
  regardless, and the mechanism explains why and predicts a third MSE target would fail.
- The PCA analysis is single-seed (42) and on the wmobs trained tectum, but obs_map's
  covariance structure is encoder-determined and decodes identity at ~1.0 in every arm,
  so it characterizes what any MSE reconstruction of obs_map faces.

## What this leaves (the user's call, FAILED-first)

The reconstruction approach to R1 is done. Do NOT iterate more reconstruction targets.
Two materially different paths remain:

- **(b) A latent and objective that DIRECTLY pressure identity, not reconstruction MSE.**
  Candidates: a continuous / higher-capacity RSSM latent (addresses the discrete
  bottleneck) PAIRED with an objective that does not average over variance, e.g. a
  contrastive or InfoNCE objective on the latent (pull same-stimulus latents together,
  push different-stimulus apart), or a classification auxiliary. This is a real
  architectural bet with uncertain payoff and no guarantee, and a contrastive/classification
  objective drifts toward supervised-identity territory.
- **(c) R2 off-ramp (recommended).** We have now shown the perception collapse is robust
  to reconstruction at its source (two targets). Per reading #2 (judge by consciousness
  signatures, not DMTS competence), run the section-13 substrate-independence test +
  Butlin rubric on the CURRENT agent and report honestly that causal efficacy on
  consciousness-demanding tasks is blocked by the perception collapse, with the
  collapse now characterized (reconstruction-robust). This also respects the
  2026-06-14/15 finding that even WITH identity available, the RL did not learn DMTS (the
  second wall), so a successful perception fix may not unblock DMTS anyway.

The --enable-wm-recon mechanism (both targets) stays in the codebase, default off, as a
documented negative result.
