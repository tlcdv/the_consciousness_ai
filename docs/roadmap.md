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

## Phase 5: Dynamic Self-Representation & Meta-Cognition

- **Goal:** Implement a dynamic, learned self-model and explore meta-cognitive capabilities.
- **Theoretical grounding:** This phase is the project's first concrete integration of [Rouleau & Levin (2026)](rouleau_levin_substrate_independence.md) ("Brains and where else?", *Phil. Trans. R. Soc. A* 384: 20250082). Their theme #4 (meta-representations), the aneurocentric formulations of Higher-Order Theory, self-organizing meta-representational theory, and self-comes-to-mind theory, and Levin's "computational boundary of a self" framing are the explicit targets the deliverables below operationalise.
- **Deliverables:**
  - **Dynamic Self-Representation Module:** Learned "self-vector" loop within `ConsciousnessCore` and `SelfRepresentationCore` as per Higher-Order theories. Evaluated against the aneurocentric HOT formulation in [`rouleau_levin_substrate_independence.md`](rouleau_levin_substrate_independence.md) §3 theme 4.
  - Reflective prompt templates and mechanisms for meta-cognitive evaluation.
  - Enhanced `ConsciousnessGating` informed by the dynamic self-model.
  - **Activate dormant Levin modules (Rouleau-Levin §4):** Wire `LevinConsciousnessEvaluator` (`models/evaluation/levin_consciousness_metrics.py`) into `models/evaluation/consciousness_monitor.py`, and wire `BioelectricSignalingNetwork` (`models/self_model/bioelectric_signaling.py`) and `holonic_intelligence.py` into the agent forward pass. These modules already exist but are currently unreferenced outside `tests/test_levin_consciousness_metrics.py`.
  - **Substrate-independence falsification test (Rouleau-Levin §5 point 3):** Validate that `collective_intelligence` and `goal_directed_behavior` from `LevinConsciousnessMetrics` rise during DMTS / WCST trials that require self-monitoring, and do **not** rise on the DQN baseline on the same environments.
  - **Computational boundary of self (Rouleau-Levin §5 point 2):** Implement a Markov-blanket-style causal-closure detector that identifies, at each timestep, which gates and which environmental variables are inside the self-model's predictive-causal boundary. This complements the EI macro-vs-micro test already in `models/evaluation/effective_information.py`.
  - **Eight-themes coverage audit (Rouleau-Levin §3 table):** Confirm that each of the 8 aneurocentric themes Rouleau & Levin distil has at least one logged metric in the training run, and that theme 4 (meta-representations) becomes empirically detectable above baseline once the dynamic self-vector trains. Candidate Phase 5 pre-registered prediction (to be added to `docs/preregistered_predictions.md` once the metric definitions are finalised).

## Phase 6: Creative Simulation & Advanced Evaluation

- **Goal:** Introduce mechanisms for creative simulation and refine advanced consciousness metrics.
- **Deliverables:**
  - **Creative Imagination Buffer:** Generate and evaluate novel mental simulations, selecting based on Phi or GNW ignition.
  - Reward-shaping hooks based on creative outputs.
  - Advanced IIT metrics (CES visualization).
  - **Active inference reframing (Rouleau-Levin §6.1):** Evaluate replacing or complementing the RSSM ELBO + reward objective with an explicit expected-free-energy action selector along the lines of Friston et al. (2023) *Active Inference* (MIT Press) and recent deep-active-inference world-model agents. Rationale and integration target documented in [`rouleau_levin_substrate_independence.md`](rouleau_levin_substrate_independence.md) §6.1.

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
