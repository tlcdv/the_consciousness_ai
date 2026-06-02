# Active inference: unifying the training objective (design draft)

**Status: DESIGN DRAFT. Not implemented. Gated, probe-first, like every experimental
direction in this project.** This document records the case for, and a concrete
mapping of, replacing the current stack of hand-wired training objectives with a
single expected-free-energy (EFE) principle. It is the elaboration of roadmap Phase 6
Section 6.1, elevated by the 2026-06-02 coherence review.

## The problem this addresses (coherence, not just a result)

The training loop (`scripts/training/train_rlhf.py`) currently optimizes ~6 distinct,
partly reward-coupled objectives that were each added to fix a separate symptom:

| Current objective | Where | What it is for |
|-------------------|-------|----------------|
| reward-predictor MSE | tectum aux block | give the tectum a gradient (predict env reward from content) |
| TDANN topographic loss | tectum aux block | enforce spatial self-organization |
| RND curiosity bonus | step reward | exploration via prediction error on the broadcast |
| phi-delta intrinsic reward | step reward | reward increases in integrated information |
| `sync_R x reward` workspace optimizer | per 10 steps | make binding synchrony reward-correlated |
| policy loss (Go/No-Go actor-critic) | per episode | action selection |

Two problems:

1. **It is incoherent with the project's own stated philosophy.** The README says the
   agent "explores to reduce prediction error (anxiety), not to accumulate external
   reward," yet four of the six objectives key off external reward. Reading #2
   (judge by consciousness signatures, not reward) sharpens the contradiction: the
   evaluation is signature-based while the learning is substantially reward-based.

2. **It is the opposite of elegant.** Six hand-tuned terms with separate
   coefficients, several added reactively, is accretion. It is hard to reason about,
   hard to ablate cleanly (the 2026-05 ablation campaign showed how entangled they
   are), and not biologically principled as a whole even though individual pieces are.

## The principle: minimize expected free energy

Active inference (Friston et al. 2023, *Active Inference*, MIT Press; Rao et al.,
Active Predictive Coding, Neural Computation 2024; Deep Active Inference,
[arXiv:2505.19867](https://arxiv.org/abs/2505.19867)) casts perception, learning, and
action as one objective: minimize (expected) free energy.

- **Perception / learning = variational free energy.** Minimize prediction error
  between the generative world model and observations. The RSSM in `SensoryTectum`
  already optimizes a DreamerV3-style ELBO; this is variational free energy in all
  but name. The front-end (retinotopic encoder + fusion) would be trained as part of
  that generative objective, not by a separate ad-hoc reward-MSE.
- **Action = expected free energy (EFE)**, which decomposes into:
  - **pragmatic value** - reach preferred observations (a *prior over outcomes*, where
    reward enters as a preference, not as the learning signal); and
  - **epistemic value** - information gain / uncertainty reduction (exploration).

## What it concretely replaces

| Current objective | Under active inference |
|-------------------|------------------------|
| reward-predictor MSE + TDANN | folded into the generative model's variational free energy (the world model predicts observations; the front-end is trained by that, optionally keeping TDANN as a biological topographic prior on the latent) |
| RND curiosity bonus | the **epistemic value** term of EFE (information gain) - principled, not a bolt-on |
| phi-delta intrinsic reward | dropped as a training driver; phi remains a **measured signature**, not an injected reward (removes a circularity) |
| `sync_R x reward` optimizer | binding remains, but driven by the generative objective rather than reward-correlation; or kept as a separate biological prior, evaluated explicitly |
| policy loss + external reward | action selected to **minimize EFE** over imagined RSSM rollouts; reward becomes a preference prior (pragmatic value) |

Net: roughly six hand-tuned terms collapse toward two (variational free energy for
perception/learning; expected free energy for action), with reward demoted from "the
learning signal" to "a preference prior." That is the elegance gain, and it makes the
learning **intrinsic** as the README already claims.

## Why this is biologically coherent (does not abandon the strategy)

Active inference is a neuroscientific theory of cortical function (predictive coding /
free-energy principle). It is squarely inside the project's Feinberg-Mallatt /
Rouleau-Levin grounding (Rouleau-Levin theme 1 = predictive modelling; theme 8 =
coarse-graining; §6.1 explicitly names active inference). It maps to Butlin indicator
**PP-1** (predictive-coding input modules), which the rubric currently rates PARTIAL;
a principled free-energy front-end is the path to move PP-1 to IMPLEMENTED. It uses
the RSSM the project already has, so it is an objective change, not a new substrate.

## Falsifiable success criteria (judge by signatures, per reading #2)

1. **PP-1 indicator** moves PARTIAL -> IMPLEMENTED with the front-end trained by a
   free-energy objective (qualitative, but checkable against the rubric).
2. **Perception-supports-the-tasks prerequisite** (roadmap Phase 5): does the
   EFE-trained front-end let the agent ENTER the DMTS/WCST diagnostic regimes that are
   currently flat? This is the concrete, measurable payoff and the gate for whether
   the unification helps at all.
3. **Coherence audit:** the number of hand-tuned loss coefficients in
   `train_rlhf.py` drops materially; the learning loop no longer keys the bulk of its
   signal off external reward.

## Honest risks and scope

- EFE action selection (planning over imagined rollouts) is **computationally
  expensive** and historically finicky; the cited deep-AIF work exists precisely
  because naive EFE planning struggles at scale. On this CPU-bound laptop, full EFE
  planning may be impractical; a bootstrapped/amortized variant may be needed.
- This is a **significant re-architecture of the training loop**, not a probe. It
  should be staged: first amortize the existing RSSM ELBO + an epistemic term to
  replace RND (small, testable), then demote reward to a preference prior, then add
  EFE action selection. Each stage behind a default-off flag, measured before the
  next.
- **No promise it produces consciousness or even better signatures.** The honest
  claim is narrower: it is more coherent, more biologically principled, more elegant,
  and it makes the learning match the stated philosophy. Whether that moves the
  consciousness signatures is an empirical question to be tested FAILED-first, not
  assumed. The project has a pattern (Phi-1) of architectures that are theoretically
  motivated but do not produce the hoped signature; this must be held to the same
  evidential bar.

## Relationship to current work

- It does not reopen Phi-1 and does not change the biological architecture's
  components.
- It is the principled replacement for the ad-hoc front-end training that the
  2026-06-02 competence localization flagged (provisionally) as a bottleneck; the
  unification case rests on coherence/elegance and PP-1, not on that single
  confounded probe.
- Sequenced after the Phase 5 perception-supports-the-tasks check: if perception
  blocks the consciousness tasks, this becomes Phase-5 enabling work; otherwise it is
  Phase 6 as scheduled.
