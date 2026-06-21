# Generative World Models for the Perception Frontier

*Companion to [active_inference_unification.md](active_inference_unification.md) and
directly relevant to the diagnosis in
[results/collapse_locus_2026_06_16.md](results/collapse_locus_2026_06_16.md). This doc
gathers the external architectures that bear on the project's current perception
bottleneck and frames them as design options for the R1 fix. Original synthesis with
citations, no reproduction of the papers.*

## The problem this addresses (from our own results)

The collapse-locus probe localized where task-relevant stimulus identity is lost in the
perception pathway: `obs_map` decodes shape and color near 1.0, but `z_state` (the RSSM
latent), `capsule_poses`, and `tectum_content` are all at chance. Identity dies at the
`obs_map -> z_state` step. The leading explanation is that the RSSM is trained to predict
reward, not to reconstruct observations, so it has no pressure to keep identity. The
named fix (roadmap R1) is to train the RSSM as a real generative world model so the
latent has to encode what the input contains.

The literature below is exactly that: world models whose objective forces the latent to
preserve content. The key design question it answers is **what the reconstruction target
should be**, which is the choice the R1 build has to make.

## Option A: predict frozen DINOv2 patch features (DINO-WM)

Zhou, G., Pan, H., LeCun, Y. & Pinto, L. (2024), "DINO-WM: World Models on Pre-trained
Visual Features enable Zero-shot Planning," arXiv:2411.04983 (ICML 2025).
Project page: https://dino-wm.github.io/

- **What it does:** Builds a world model by predicting future DINOv2 patch features from
  current features and actions, with no pixel reconstruction. Because DINOv2 patch tokens
  keep spatial structure, the predicted latent keeps identity, and the model supports
  planning by optimizing actions toward goal features.
- **Why it fits this project specifically:** The tectum already encodes frames with
  DINOv2 (`models/core/sensory_tectum.py`). So a DINO-WM-style objective, predict the
  next-step DINOv2 patch features, is close to a drop-in target for the RSSM. It avoids
  building and training a pixel decoder, and it preserves identity by construction
  because the target is the same patch-feature space that already decodes at ~1.0 in the
  probe.
- **Relation to the R1 plan:** [active_inference_unification.md](active_inference_unification.md)
  proposes an RSSM observation decoder plus an ELBO reconstruction term. DINO-WM is the
  same idea with a different, cheaper, identity-preserving target: reconstruct or predict
  the frozen DINOv2 feature map rather than raw pixels. It is worth evaluating as option
  A against pixel-ELBO as option B.
- **Honest caveat:** DINO-WM is a planning method trained on offline trajectories. The
  integration and compute cost on a batch-1, latency-bound setup is unproven here, and it
  must be tested FAILED-first, gated, default-off, with at least three seeds before any
  default change. Verified against the primary source.

## Option B: object-centric / slot world models

OC-STORM: Zhang, W., Jelley, A., McInroe, T., Storkey, A. & Wang, G. (2025),
arXiv:2501.16443. Foundation: Locatello et al. (2020), "Object-Centric Learning with Slot
Attention," arXiv:2006.15055.

- **What it does:** Represents the world as a set of per-object slots that keep identity
  under occlusion and interaction, and learns dynamics over those slots. Reported to
  improve sample-efficient RL in visually complex environments.
- **Why it fits:** Slot identity is the property the collapse probe found missing, and it
  is the same property DMTS feature-binding and working memory need. Slots are also a
  content-level form of binding, so this connects the perception fix to the binding
  pillar.
- **Honest caveat:** heavier to integrate than DINO-WM, and the fast-moving cluster's
  exact paper-to-identifier mapping should be re-confirmed at build time.

## Option C: the predictive-coding / active-inference implementation toolkit

Ororbia et al. (2024), "A Review of Neuroscience-Inspired Machine Learning,"
arXiv:2403.18929. Salvatori et al. (2024), "Predictive Coding Networks and Inference
Learning: Tutorial and Survey," arXiv:2407.04117.

- **What it is:** Implementation references for predictive-coding and free-energy
  objectives, which the Phase 6 plan needs and currently lacks (it cites Friston and Rao
  theory but no build recipe).
- **Why it fits:** Whichever reconstruction target is chosen (A or B), the training
  objective is a variational free-energy / predictive-coding loss. These give concrete
  recipes and a map of options, including backprop-free variants.

## How this plugs into the existing harness

The validation tools the R1 fix needs already exist, so an option can be tested cheaply:

- `scripts/analysis/probe_collapse_locus.py` and
  `scripts/analysis/probe_perception_decodability.py` decode identity per stage. After
  training a world model with a content-preserving objective, re-run the probe: if
  `z_state` now decodes identity, the objective worked.
- `models/core/sensory_tectum.py` is where the RSSM and its objective live.
- The decision rule is the project's existing one: trained `z_state` still at chance
  means the reward-only objective is confirmed and the world-model objective is the fix;
  trained `z_state` recovering identity changes the picture.

## Honest framing

This doc lists options, it does not pick one, and it makes no claim that any of them
moves a consciousness signature. The project has a record of well-motivated additions
that did not move a metric. The value here is narrow and concrete: the perception
collapse has a named cause (reward-only RSSM objective) and these are the published,
on-core ways to give the latent a content-preserving objective, with DINO-WM being the
closest fit to the existing DINOv2 encoder. Everything is gated, default-off, baseline
bit-identical, and tested FAILED-first.

## References

- Zhou, G., Pan, H., LeCun, Y. & Pinto, L. (2024). DINO-WM. arXiv:2411.04983.
- Zhang, W. et al. (2025). Object-centric world models (OC-STORM). arXiv:2501.16443.
- Locatello, F. et al. (2020). Object-Centric Learning with Slot Attention.
  arXiv:2006.15055.
- Ororbia, A. et al. (2024). A Review of Neuroscience-Inspired Machine Learning.
  arXiv:2403.18929.
- Salvatori, T. et al. (2024). Predictive Coding Networks and Inference Learning.
  arXiv:2407.04117.
- See also [active_inference_unification.md](active_inference_unification.md),
  [results/collapse_locus_2026_06_16.md](results/collapse_locus_2026_06_16.md),
  [watanabe_generative_model_approach.md](watanabe_generative_model_approach.md).
