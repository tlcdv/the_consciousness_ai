# Development Roadmap

This roadmap outlines the planned development phases for The Consciousness AI.

## Guiding Principles

- **Iterative Development:** Build and test components incrementally.
- **Emergence-Focused:** Design for emergent properties rather than explicit programming of consciousness.
- **Theory-Grounded:** Integrate insights from neuroscience and AI consciousness research (GNW, IIT, Feinberg-Mallatt).
- **Ethical Alignment:** Continuously ensure adherence to Asimov's Laws and responsible AI principles.

---

## Phase 1: Foundational Setup & Core Modules (Complete)

- **Goal:** Establish the basic project structure, core AI model integrations, and initial simulation environment.
- **Key AI Models:**
  - Qwen2-VL-7B for vision-language understanding (replaced PaLM-E and VideoLLaMA3).
  - DINOv2-B/14 (frozen) for spatially faithful retinotopic encoding.
  - Cochlear auditory pipeline (gammatone filterbank, inner hair cell model, tonotopic encoder, acoustic affect extraction).
- **Deliverables:**
  - Core repository structure.
  - [`ConsciousnessCore`](../models/core/consciousness_core.py) with GNW, gating, and ethics filter.
  - [`EmotionalProcessingCore`](../models/emotion/emotional_processing.py) with full PAD model.
  - [`EmotionalMemoryCore`](../models/memory/emotional_memory_core.py) with FAISS vector index.

## Phase 2: Core Architecture & Biologically Grounded Integration (Complete)

- **Goal:** Implement the six Feinberg-Mallatt neurobiological features as computational mechanisms.
- **Deliverables:**
  - **Oscillatory binding:** AKOrN Kuramoto oscillators integrated into GNW (replaces hardcoded synchrony multiplier).
  - **Sensory Tectum:** Topographic spatial maps with DreamerV3 RSSM world model, inverse effectiveness fusion (Stein & Meredith 1993).
  - **Reentrant processing:** 5-10 adaptive convergence cycles with predictive coding.
  - **Affective modulator:** Parallel valence field + arousal-threshold coupling (emotion modulates, does not compete).
  - **Self-model:** Body schema, interoceptive state, capability model.
  - **PAD homeostatic reward:** `Rtotal = Rext + lambda1*DeltaValence - lambda2*(Arousal - target)^2 + lambda3*Dominance`.
  - **Effective Information:** Hoel's PNAS 2013 framework for causal emergence detection.

## Phase 3: Compositional Deepening & Validation (Complete)

- **Goal:** Deepen hierarchical structure, wire embodiment-affect loop, and add biological validation.
- **Deliverables:**
  - **Capsule network hierarchy:** 4-level routing (Sabour 2017) between tectum and workspace.
  - **Multi-level reentrant feedback:** Top-down prediction errors between capsule levels.
  - **IIT Phi rewrite:** Causal gate states (attention, stability, adaptation, coherence, confidence) as subsystem nodes. Adaptive binarization, geometric proxy.
  - **Brian2 validation:** Parameter translation from AKOrN to spiking Kuramoto network, synchronization curve comparison.
  - **Trimodal tectum:** Somatosensory channel (body schema projected onto spatial grid via IE fusion).
  - **Embodiment-affect loop:** Interoceptive PAD generation (energy/fatigue/damage drive valence signals).
  - **Isomorphic visual mapping:** RetinotopicEncoder (DINOv2), TDANN topographic loss (Margalit 2024), inverse effectiveness.
  - **Ethics filter:** AsimovComplianceFilter with three-law evaluation pipeline and world model trajectory prediction.
  - **Dark Room training:** Full working training loop exercising the complete cognitive pipeline.
  - **Navigation environment:** Multi-room grid with fog of war, colored goals, battery system.
  - Python 3.10+ type annotations across 111 files.

## Phase 3.5: Alignment Fixes & Consciousness-Demanding Environments (Complete)

- **Goal:** Fix structural gaps that allow trivial solutions to pass as consciousness, and build environments that genuinely require the consciousness machinery.
- **Deliverables:**
  - **Two-stage emotion appraisal:** Reflex layer (surprise + reward delta, pre-workspace) + appraisal layer (phenomenological state, post-broadcast). Replaces brightness lookup that bypassed the entire affective architecture.
  - **Capsule broadcast payloads:** Structured capsule poses and activities preserved through GNW broadcast, so downstream consumers access compositional hierarchy instead of flattened scalars.
  - **Consciousness monitor fix:** Removed circular `progress_factor` that made consciousness level increase with step count by construction.
  - **DMTS environment:** 4-phase delayed match-to-sample (fixation, sample, delay, choice). 72 unique stimuli (6 shapes x 6 colors x 2 sizes), configurable distractor overlap (0-3 shared features), 15-40 step blank delay. Requires working memory and feature binding.
  - **WCST environment:** Wisconsin Card Sort analog with hidden rule changes after consecutive correct sorts. Tests meta-cognition, inhibition, and hypothesis testing.
  - **Shared stimulus renderer:** Pure numpy polygon rasterization (no pygame dependency). 6 shapes, 6 colors, card rendering.
  - **DQN baseline:** Vanilla DQN agent (CNN + MLP, replay buffer, target network) for controlled comparison on the same environments.
  - **Metrics logger integration:** ConsciousnessMetricsLogger wired into training loop with EI computation and insight moment detection.
  - **Memory consolidation:** Relevance decay, cosine-similarity merge, low-relevance pruning, replay batch selection.

## Phase 4: Narrative Engine & Social Interactions (Current Focus)

- **Goal:** Enable LLM-backed narrative reasoning, pre-register consciousness predictions, and expand social evaluation.
- **Deliverables:**
  - ~~`NarrativeEngine` V1 with LLM-backed generation and coherence tracking.~~ DONE.
  - ~~Pre-registered Phi/EI predictions tied to specific training milestones.~~ DONE. See `docs/preregistered_predictions.md`.
  - ~~Operational definition of "insight moments" for empirical testing.~~ DONE. See `docs/preregistered_predictions.md` section 3.
  - Consciousness indicator-property test suite expansion.
  - ~~Rename `QualiaState` to defensible terminology.~~ DONE.
  - ~~Two-stage emotion appraisal (reflex + post-broadcast appraisal).~~ DONE.
  - ~~Capsule structured payloads wired through GNW broadcast.~~ DONE.
  - ~~Consciousness monitor circular progress_factor removed.~~ DONE.
  - ~~Consciousness-demanding environments (DMTS, WCST).~~ DONE.
  - ~~DQN baseline agent for controlled comparison.~~ DONE.
  - ~~Metrics logger wired into training loop.~~ DONE.
  - ~~Memory consolidation cycle (relevance decay, cosine merge, pruning).~~ DONE.
  - ~~Navigation environment (multi-room, fog of war, battery).~~ DONE.
  - ~~Semantic pathway (Qwen2-VL embeddings as 5th workspace oscillator).~~ DONE.
  - ~~Documentation synchronization across all docs.~~ DONE 2026-05-17 (six docs updated to reflect 2026-05-14 pyphi ablation and 2026-05-16 RIIU empirical results).

### Status 2026-05-17

Pre-registered Phi-1 prediction (Pearson r > 0.4 between phi and AKOrN sync_R during training) FAILED in the 2026-05-14 pyphi ablation campaign (best r=+0.089 across 5 architectural variants, `docs/results/ablation_2026_05_14.md`) and the 2026-05-16 RIIU single-seed run on broadcast substrate (full-run r=+0.075, transient peak +0.267, `docs/results/riiu_compare_2026_05_16.md`). Per the decision protocol in `docs/preregistered_predictions.md` section 5, this is outcome 4 (fundamental redesign needed).

The 2026-05-17 follow-up tested the substrate hypothesis with a parallel RIIUPhi probe (broadcast, tectum, audio simultaneously, single seed, 50 episodes; see `docs/results/riiu_substrate_probe_2026_05_17.md`). Outcome: NO WINNER. Tectum and broadcast substrates produced bit-identical phi (max |diff| = 0.0 across 9000 rows) because the broadcast tensor IS the tectum payload when the vision module wins workspace competition. Audio was degenerate without `--enable-audio`. The plan's Phase C 3-seed verification was SKIPPED per decision gate.

Phi-1 stands FAILED across pathways (pyphi, RIIU) AND substrates (broadcast, tectum, audio). The project proceeds to Phase 5 (Dynamic Self-Representation & Meta-Cognition). The RIIU code path remains available behind `--enable-riiu` as a diagnostic.

### Status 2026-05-18: Phi-1 chapter closed

After the 2026-05-17 substrate probe found NO WINNER, a deeper diagnosis surfaced FIVE structural failure modes in the architecture (`models/core/global_workspace.py:217-231` winner-take-all; `models/core/oscillatory_binding.py:140-191` phase-not-content binding; `models/core/reentrant_processor.py:121-128` bids-only feedback; gate-state collapse to 2-3 of 32 binarized states; dark_room single-modality bottleneck). A new pre-registration (section 10 of `docs/preregistered_predictions.md`) tested whether fixing these would let Phi-1 emerge.

Phases A (attention-weighted fusion, commit `967fe2a`), C (gate-collapse fixes, commit `fafd581`), D (mock semantic + audio + pre-flight, commit `42fe78b`), E (pre-registration, commit `7227104`), and the wiring fixes (commit `d0318ff`) were implemented. F1 smoke passed. F2 single-seed 200-episode run produced pyphi r = -0.038. Option 3 dual-pathway run (Option 3 of the verdict decision tree) added the RIIU pathway and produced **pyphi r = -0.062, RIIU r = -0.005 (NOT significant)**. The 2026-05-16 transient r = +0.267 peak does NOT replicate; the corresponding rolling window in Option 3 shows r = +0.023.

Across 7 independent runs spanning 2 architectures (OLD pre-2026-05-17, NEW post-2026-05-17) and 2 phi formulations (pyphi gate-state TPM, RIIU broadcast SVD), no run achieves the pre-registered r > 0.4 threshold or even the partial r > 0.15 on the full-run statistic. Verdict doc: `docs/results/phi1_retest_dual_pathway_2026_05_18.md`.

The architectural improvements produced measurably better dynamics (28x phi mean, 10x RIIU phi variance, comparable reward) but did not produce the predicted binding-phi coupling. The Phi-1 chapter for the current binding+phi+gate architecture is closed. The project enters Phase 5.

### Status 2026-05-24: Phi-1 escalation chain exhausted (KomplexNet inverse finding)

After the 2026-05-18 closure, the pre-registered architectural escalation chain (sections 10-12 of `docs/preregistered_predictions.md`) was run to completion. Phase B (AKOrN content-level cross-attention, 2026-05-19) and Phase B-alt (KomplexNet complex-valued binding via `--binding-mechanism komplex`, 2026-05-24) were each pre-registered and tested at 200 episodes, seed 42.

Across 9 runs spanning 4 architectures (AKOrN; AKOrN+A+C+D; AKOrN+A+B+C+D; KomplexNet+A+C+D) and 2 phi formulations (pyphi gate-state TPM; RIIU broadcast SVD), no run reaches the pre-registered r > 0.4 in the predicted positive direction, nor the partial r > 0.15. The KomplexNet RIIU pathway produced the campaign's first substantively significant phi-binding correlation: r = -0.1116, p = 2.5e-108, n = 39000. It is in the INVERSE direction. RIIU phi standard deviation was 19.65x the pyphi standard deviation (best of campaign), and task reward was also best of campaign (mean 11.76, 64/200 positive episodes). Verdict doc: `docs/results/phi1_phaseBalt_2026_05_24.md`; pre-registration record: section 12.

Mechanistic reading: when phases align (high sync_R), KomplexNet content factors cluster near +1, compressing representational variance and lowering RIIU phi; desynchronization raises variance and raises phi. In this architectural family, oscillatory binding and IIT-style integrated information are mechanistically opposed, not coupled. This is a positive empirical constraint on theory, not a null result.

Decision (Option A of the verdict doc): narrow the public claim about the in-training Phi-1 prediction specifically, and proceed to Phase 5. What is exhausted is one in-training measurement choice. What stands: the Functionalist Emergentism thesis, the Feinberg-Mallatt biological grounding, the 2026-02-21 3-condition synthetic test (phi monotonicity with binding on a controlled stimulus), and the other implemented consciousness signatures (EI causal emergence, DMTS/WCST behavioral integration, phenomenological mapping, insight detection). The pre-registered r > 0.4 threshold and sign are not revised retroactively. No further Phi-1 ablations or binding architectures are pre-registered against Phi-1.

The KomplexNet implementation (`models/core/complex_binding.py`, `--binding-mechanism komplex`) stays in the codebase default-off as the empirical basis for the inverse finding. RIIU stays behind `--enable-riiu` as a diagnostic.

### Status 2026-06-02: agent competence localized; success metric set to consciousness signatures

The P5 competence thread localized the agent's low dark_room control reward (~15 vs a
vanilla DQN on raw pixels at 92) to the perceptual front-end. Holding the DQN learner
constant across the taps, reward is: pixels 92.00 -> obs_map 17.14 (pre RSSM/capsule)
-> tectum_content 15.93 (post capsule) -> broadcast 14.65 (post GNW). Defensible
finding: the capsule collapse, the GNW, and the policy are NOT the cost (the taps tie;
Go/No-Go and A2C tie on the broadcast). The pixels-vs-taps jump is CONFOUNDED
(exploration schedule, MLP-vs-CNN, 120-vs-1000 episodes; see the Confound caveat in
the results doc), so "the front-end specifically is the lossy stage" is provisional,
not established. Numbers on disk:
[`docs/results/agent_competence_fix_2026_06_02.md`](results/agent_competence_fix_2026_06_02.md).

Decision (adopted, [`docs/decisions/2026_06_02_competence_reading_2.md`](decisions/2026_06_02_competence_reading_2.md)):
the architecture is biology-first; dark_room control reward is the cost of that design,
not the success metric. **Success is judged by consciousness signatures**, formalized as
the indicator-properties rubric in
[`docs/consciousness_indicators_butlin.md`](consciousness_indicators_butlin.md) (Butlin,
Long et al. 2023, arXiv:2308.08708; TiCS 2025), together with the project's own metrics
(IIT phi, EI causal emergence, Levin metrics, phenomenological mapping, insight
detection, DMTS/WCST behavioral integration). Control reward is retired as a target and
kept only as a behavioral sanity signal.

Honest competence bar (not "declare victory"): the agent must still perceive well enough
to ENTER the diagnostic regimes of the consciousness-demanding tasks, or the signatures
cannot be measured. This becomes a Phase 5 prerequisite below.

The biological strategy, components, and roadmap are unchanged. The active-inference
direction (Phase 6 below) is elevated: the localization gives it a measured motivation,
since the front-end is lossy precisely because it is trained ad-hoc (reward-MSE + TDANN
topography) rather than by a principled free-energy / prediction-error objective.

## Phase 5: Dynamic Self-Representation & Meta-Cognition

- **Goal:** Implement a dynamic, learned self-model and explore meta-cognitive capabilities.
- **Evaluation (2026-06-02):** Phase 5 progress is judged by the indicator rubric
  ([`consciousness_indicators_butlin.md`](consciousness_indicators_butlin.md)), not
  control reward. The self-vector loop and metacognition target HOT-2 / HOT-3; the
  substrate-independence test targets measurable agency and integration on the
  consciousness tasks. Progress = moving an indicator from PARTIAL to IMPLEMENTED with a
  measured signature.
- **Prerequisite (2026-06-02): verify perception supports the consciousness tasks.** A
  cheap check before the signature experiments: can the agent perceive the DMTS sample /
  WCST card enough to enter the task's diagnostic regime (DMTS/WCST are currently flat)?
  If not, the front-end is the blocker, and the active-inference reframing below becomes
  Phase-5 enabling work rather than Phase-6 polish. Pursued probe-first and biologically;
  never by reverting to a non-biological control encoder.
- **Theoretical grounding:** This phase is the project's first concrete integration of [Rouleau & Levin (2026)](rouleau_levin_substrate_independence.md) ("Brains and where else?", *Phil. Trans. R. Soc. A* 384: 20250082). Their theme #4 (meta-representations), the aneurocentric formulations of Higher-Order Theory, self-organizing meta-representational theory, and self-comes-to-mind theory, and Levin's "computational boundary of a self" framing are the explicit targets the deliverables below operationalise.
- **Deliverables:**
  - **Dynamic Self-Representation Module:** Learned "self-vector" loop within `ConsciousnessCore` and `SelfRepresentationCore` as per Higher-Order theories. Evaluated against the aneurocentric HOT formulation in [`rouleau_levin_substrate_independence.md`](rouleau_levin_substrate_independence.md) §3 theme 4.
  - Reflective prompt templates and mechanisms for meta-cognitive evaluation.
  - Enhanced `ConsciousnessGating` informed by the dynamic self-model.
  - **Activate dormant Levin modules (Rouleau-Levin §4):** Wire `LevinConsciousnessEvaluator` (`models/evaluation/levin_consciousness_metrics.py`) into `models/evaluation/consciousness_monitor.py`, and wire `BioelectricSignalingNetwork` (`models/self_model/bioelectric_signaling.py`) and `holonic_intelligence.py` into the agent forward pass. These modules already exist but are currently unreferenced outside `tests/test_levin_consciousness_metrics.py`.
  - **Substrate-independence falsification test (Rouleau-Levin §5 point 3):** Validate that `collective_intelligence` and `goal_directed_behavior` from `LevinConsciousnessMetrics` rise during DMTS / WCST trials that require self-monitoring, and do **not** rise on the DQN baseline on the same environments.
  - **Computational boundary of self (Rouleau-Levin §5 point 2):** Implement a Markov-blanket-style causal-closure detector that identifies, at each timestep, which gates and which environmental variables are inside the self-model's predictive-causal boundary. This complements the EI macro-vs-micro test already in `models/evaluation/effective_information.py`.
  - **Eight-themes coverage audit (Rouleau-Levin §3 table):** Confirm that each of the 8 aneurocentric themes Rouleau & Levin distil has at least one logged metric in the training run, and that theme 4 (meta-representations) becomes empirically detectable above baseline once the dynamic self-vector trains. Candidate Phase 5 pre-registered prediction (to be added to `docs/preregistered_predictions.md` once the metric definitions are finalised).

### Metzinger MPE additions (2026-06-07, gated / design)

Integration of Metzinger's *The Elephant and the Blind* (2024), the Minimal
Phenomenal Experience framework. See [`metzinger_phenomenal_self_model.md`](metzinger_phenomenal_self_model.md).
Metzinger reframes the self-model work: the *computational self-model built by
self-prediction* is already the project's `SelfVectorModule` + RSSM mechanism, and
MPE points at a more minimal, *nonegoic* target (a model of the agent's own
epistemic space) prior to the self. The deliverables below are design-only and
behind the usual default-off gate; no default behavior changes, and no new metric
is built without clearing a "genuinely new falsifiable signature versus repackaging
existing quantities" bar.

  - **Existence-bias ablation (ethics checkpoint, concrete code lead):** A
    default-off `--ablate-existence-bias` flag that zeros or attenuates the
    survival-linked drives (interoceptive negative-valence terms, the homeostatic
    arousal/dominance reward terms, optionally Asimov Law 3 self-preservation), so a
    "no existence-bias" configuration can be run and its consciousness signatures
    compared. This operationalizes the *Bewusstseinskultur* ethics: Metzinger argues
    against building a craving-for-existence (*bhava-taṇhā*) into conscious machines.
    An ablation, not a claim about suffering; FAILED-first; three or more seeds
    before any conclusion. Tracked in [`ethics_framework.md`](ethics_framework.md).
  - **Epistemic-space / pure-awareness signature (design only):** A candidate
    nonegoic MPE signature. Must be distinguishable from RND, prediction error, and
    RSSM latent entropy. Build only against a pre-registered, falsifiable prediction
    tied to a task manipulation (for example, it should rise during the DMTS delay
    where the agent holds knowledge without input, and not for a reactive DQN).
  - **Transparency / opacity mechanism (design sketch):** Enforce content/vehicle
    separation in the self-vector to gate/policy pathway (downstream modules consume
    self-vector content without gradient access to its construction), with an opacity
    mode that exposes the vehicle for metacognitive readout. Addresses the standing
    transparency gap in the self-model; needs its own design pass.
  - **Skeptic discipline (adopted now, in docs):** the project explicitly adopts
    Metzinger's C/E/M-fallacies. A signature is an engineering metric, never an
    existence proof. No new code; this is a stance applied to how every result is
    reported.

## Phase 6: Creative Simulation & Advanced Evaluation

- **Goal:** Introduce mechanisms for creative simulation and refine advanced consciousness metrics.
- **Deliverables:**
  - **Creative Imagination Buffer:** Generate and evaluate novel mental simulations, selecting based on Phi or GNW ignition.
  - Reward-shaping hooks based on creative outputs.
  - Advanced IIT metrics (CES visualization).
  - **Active inference reframing (Rouleau-Levin §6.1; elevated 2026-06-02):** Replace or complement the ad-hoc front-end training (reward-MSE + TDANN topography) and the RSSM ELBO + reward objective with an explicit expected-free-energy / prediction-error objective along the lines of Friston et al. (2023) *Active Inference* (MIT Press), Rao et al. (Active Predictive Coding, Neural Computation 2024, MIT Press), and recent deep-active-inference world-model agents ([arXiv:2505.19867](https://arxiv.org/abs/2505.19867)). **Motivation (2026-06-02):** the perception is trained by an ad-hoc mix of auxiliary losses (reward-MSE + TDANN topography), and the broader training loop stacks ~6 hand-wired, partly reward-coupled objectives that contradict the project's stated "intrinsic motivation, not external reward" philosophy. An action-coupled expected-free-energy objective unifies most of them (predictive perception, exploration in place of RND, world-modeling, action selection) under one biologically principled principle, and resolves the reward-vs-signatures incoherence. The competence localization is consistent with this (the front-end may be a bottleneck) but is provisional; the unification case rests on coherence/elegance, not on that single confounded probe. If the Phase 5 perception prerequisite fails, this work moves earlier (Phase-5 enabling). See [`active_inference_unification.md`](active_inference_unification.md). Rationale and integration target in [`rouleau_levin_substrate_independence.md`](rouleau_levin_substrate_independence.md) §6.1.

## Phase 7: Peer Consciousness & Robustness

- **Goal:** Explore inter-agent awareness and conduct comprehensive system validation.
- **Deliverables:**
  - **Peer-Consciousness Probes:** Two consciousness agents interact and model each other's internal states.
  - Comprehensive ethical review and safety testing.
  - Long-term stability and learning assessments.

## Future Directions

- Full perceptual loop integration with robotics or advanced VR sensors.
- Subjective-report alignment (RLHF) so the agent's language faithfully mirrors internal states.
- Continuous refinement of the `AsimovComplianceFilter` and ethical governance.
- Development of `Consciousness-Metric.md` as a living spec for external contributors.
