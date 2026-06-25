# Website update notes: v1.3.0 (2026-06-24)

This document contains ready-to-port content for theconsciousness.ai. Apply the changes
in venturaEffect/the_consciousness_ai_page. The research repo itself is not modified by
this document.

---

## Version bump

Update **v1.2.0 to v1.3.0** everywhere the version string appears on the site.

---

## Factual correction (CRITICAL)

The website currently lists **"PPO reinforcement learning"** as the learning algorithm.
This is incorrect. The actual policy learner is a **Go/No-Go basal-ganglia model**,
implemented in `models/self_model/action_selection_core.py`. PPO, A2C, and DQN exist
only as **comparison baselines** (`scripts/training/train_baseline_dqn.py`,
`models/self_model/standard_actor_critic.py`). This error should be corrected on every
page where it appears.

Correct phrasing: "The agent learns via a biologically grounded Go/No-Go basal-ganglia
model. PPO, A2C, and DQN are trained separately as comparison baselines."

---

## Test count

Update from **"529 tests"** (or whatever is shown) to **737 tests passing**.

---

## Current project status (honest, as of v1.3.0)

Paste or adapt the following for the site's status / progress section:

> **Phase 5: Dynamic Self-Representation and Perception Research**
>
> v1.3.0 completes a major research arc spanning the entire spring 2026 campaign.
> 737 tests pass across the full architecture. Key findings this cycle:
>
> **What was established:**
>
> - The Phi-1 binding hypothesis was tested exhaustively across **9 runs, 4 binding
>   architectures (AKOrN, AKOrN+content-level binding, KomplexNet, RIIU), and 2 phi
>   formulations**. The pre-registered prediction (phi correlates r > 0.4 with binding
>   sync_R during training) was not achieved in any configuration. The strongest result
>   was the 2026-05-24 KomplexNet run: RIIU phi and sync_R show a significant
>   **inverse** correlation (r = -0.11, p < 10^-100), a real mechanistic finding. When
>   oscillator phases align (high sync_R), module content vectors cluster near +1,
>   yielding low SVD-residual phi; when they desync, content spans [-1, +1] and phi
>   rises. The in-training Phi-1 specific prediction is exhausted. The 2026-02-21
>   3-condition synthetic test demonstrating phi monotonicity with binding strength
>   still stands. The mission continues.
>
> - A leakage-free probe series characterized the **perception collapse**: the RSSM
>   step discards stimulus identity. The topographic obs_map (pre-RSSM) decodes shape
>   and color at ~100%; the RSSM latent and everything downstream are at chance. The
>   collapse is architectural: stimulus identity is a low-variance direction in the
>   feature space, and MSE objectives are dominated by high-variance structure. Two
>   reconstruction targets (raw frame, dense feature map) were tested and both failed
>   to repair the collapse.
>
> - A **value-equivalent world-model objective** (inspired by DreamerV3 / MuZero) was
>   built and tested: action-conditioned RSSM, balanced-KL training, reward and
>   continuation prediction from the RSSM latent, with BPTT through the full task
>   delay. No observation decoder (structurally avoiding the reconstruction trap). Both
>   configurations trained their losses down but the RSSM recurrent state did not
>   retain working memory across the delay. The discrete gumbel-softmax categorical
>   latent is the current bottleneck.
>
> - Phase 5 self-representation work established a **dynamic self-vector** (predicting
>   the agent's own first-order features one step ahead, default off). Validated on
>   navigation (+0.35 skill rising with training). Gating the ConsciousnessGate on the
>   self-vector did not improve WCST performance at single seed.
>
> - Dormant **Levin consciousness metrics** (bioelectric complexity, morphological
>   adaptation, collective intelligence, basal cognition) were activated and verified
>   as non-trivially varying with input, behind `--enable-levin-metrics`.
>
> **What remains open:**
>
> - Continuous or higher-capacity RSSM latent paired with a contrastive/InfoNCE
>   identity-pressuring objective: the remaining lever for perception repair.
> - Pre-registered substrate-independence test
>   (docs/preregistered_predictions.md section 13): does the consciousness agent beat a
>   DQN on DMTS/WCST self-monitoring trials? Currently blocked on agent competence.
> - Active-inference unification: consolidating the 6+ separate training objectives
>   into one free-energy principle (roadmap Phase 6).

---

## Technical models and components (correct this section on all pages)

Replace any stale or incorrect model list with:

**Perception:**
- Retinotopic encoder: DINOv2-B/14 (frozen, used when weights are available) or a
  4-layer strided conv fallback (default in CI and local runs without downloaded weights)
- Topographic obs_map: 16x16x64 spatial map with inverse-effectiveness fusion of vision
  and audio cues
- Qwen2-VL-7B (optional): semantic pathway, requires model download

**World model:**
- DreamerV3-style RSSM (Recurrent State Space Model) with gumbel-softmax categorical
  latent, action-conditioning, and optional balanced-KL training
- Value-equivalent world-model objective (`--enable-wm-predict`, default off): reward
  and continuation prediction from the RSSM latent with per-trial BPTT

**Workspace and binding:**
- AKOrN oscillatory binding (Kuramoto-on-N-sphere, default)
- KomplexNet binding (`--binding-mechanism komplex`, default off)
- 4-level hierarchical capsule composition with intra-hierarchy reentrant feedback
- Global Neuronal Workspace (GNW) with attention-weighted broadcast fusion

**Learning:**
- Go/No-Go basal-ganglia policy (the primary learner, `action_selection_core.py`)
- RND curiosity module on the GNW broadcast
- Memory consolidation with phi-prioritized replay
- Comparison baselines: vanilla DQN, standard A2C (not the primary learner)

**Auditory:**
- Gammatone filterbank (64 ERB bands, frozen) + inner hair cell model
- Tonotopic encoder (trainable 1D conv), auditory specialist in GNW competition
- Acoustic affect extraction (6 features -> PAD + paralinguistic class)

**Consciousness metrics (diagnostic, default off unless stated):**
- IIT Phi via pyphi 1.2+ on the 5-node ConsciousnessGate subsystem
- RIIU Auto-Phi on the broadcast (`--enable-riiu`, default off)
- Effective Information at gate vs workspace level (Hoel framework)
- Levin metrics: bioelectric complexity, morphological adaptation, collective
  intelligence, basal cognition (`--enable-levin-metrics`, default off)
- Dynamic self-vector skill (`--enable-self-vector`, default off)

**Environments:**
- Dark Room (2D navigation, light finding)
- Navigation (2x2 room grid, fog of war, battery, colored goals)
- DMTS (Delayed Match-to-Sample, 72 stimuli, configurable delay 15-40 steps)
- WCST (Wisconsin Card Sort analog, 3 feature dimensions, perseverative error tracking)

---

## /acm/ page specific

The /acm/ page describes the system. Key corrections needed:

1. Replace "PPO reinforcement learning" with "Go/No-Go basal-ganglia learning".
2. Add a brief note that the DMTS and WCST environments implement the key tasks from
   animal consciousness research (working memory, cognitive flexibility, meta-cognition).
3. The Phi/IIT section: note the in-training binding correlation hypothesis was tested
   extensively (9 runs) and did not reach the predicted r > 0.4; the 3-condition
   synthetic test of phi monotonicity stands; the mission continues via other signatures.

---

## /about/ page specific

Update the project stage from v1.2.0 to v1.3.0 and adjust any "in progress" items that
were completed this cycle (Phase 5 self-representation mechanics built and tested;
perception collapse characterized; world-model Stage 1 built and FAILED with honest
record).

---

## Links to add

Result docs worth linking from the website if the site references them:
- [Perception collapse synthesis](https://github.com/tlcdv/the_consciousness_ai/blob/main/docs/results/perception_collapse_synthesis_2026_06_21.md)
- [KomplexNet binding verdict](https://github.com/tlcdv/the_consciousness_ai/blob/main/docs/results/phi1_phaseBalt_2026_05_24.md)
- [World-model Stage 1 verdict](https://github.com/tlcdv/the_consciousness_ai/blob/main/docs/results/wm_predict_stage1_2026_06_24.md)
- [Butlin consciousness indicators rubric](https://github.com/tlcdv/the_consciousness_ai/blob/main/docs/consciousness_indicators_butlin.md)
- [Preregistered predictions](https://github.com/tlcdv/the_consciousness_ai/blob/main/docs/preregistered_predictions.md)
