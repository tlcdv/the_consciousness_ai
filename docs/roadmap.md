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
  - Faster-Whisper for speech recognition and transcription.
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
  - Python 3.10+ type annotations across 111 files.

## Phase 4: Narrative Engine & Social Interactions (Current Focus)

- **Goal:** Enable LLM-backed narrative reasoning, pre-register consciousness predictions, and expand social evaluation.
- **Deliverables:**
  - ~~`NarrativeEngine` V1 with LLM-backed generation and coherence tracking.~~ DONE.
  - ~~Pre-registered Phi/EI predictions tied to specific training milestones.~~ DONE. See `docs/preregistered_predictions.md`.
  - ~~Operational definition of "insight moments" for empirical testing.~~ DONE. See `docs/preregistered_predictions.md` section 3.
  - Consciousness indicator-property test suite expansion.
  - ~~Rename `QualiaState` to defensible terminology.~~ DONE.
  - Documentation synchronization across all docs.

## Phase 5: Dynamic Self-Representation & Meta-Cognition

- **Goal:** Implement a dynamic, learned self-model and explore meta-cognitive capabilities.
- **Deliverables:**
  - **Dynamic Self-Representation Module:** Learned "self-vector" loop within `ConsciousnessCore` and `SelfRepresentationCore` as per Higher-Order theories.
  - Reflective prompt templates and mechanisms for meta-cognitive evaluation.
  - Enhanced `ConsciousnessGating` informed by the dynamic self-model.

## Phase 6: Creative Simulation & Advanced Evaluation

- **Goal:** Introduce mechanisms for creative simulation and refine advanced consciousness metrics.
- **Deliverables:**
  - **Creative Imagination Buffer:** Generate and evaluate novel mental simulations, selecting based on Phi or GNW ignition.
  - Reward-shaping hooks based on creative outputs.
  - Advanced IIT metrics (CES visualization).

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
