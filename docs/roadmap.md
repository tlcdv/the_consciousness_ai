# Development Roadmap

This roadmap outlines the planned development phases for the The Consciousness AI.

## Guiding Principles

- **Iterative Development:** Build and test components incrementally.
- **Emergence-Focused:** Design for emergent properties rather than explicit programming of consciousness.
- **Theory-Grounded:** Integrate insights from neuroscience and AI consciousness research (e.g., GNW, IIT, Higher-Order Theories).
- **Ethical Alignment:** Continuously ensure adherence to Asimov's Laws and responsible AI principles.

---

## Phase 1: Foundational Setup & Core Modules (Completed/Ongoing)

- **Goal:** Establish the basic project structure, core AI model integrations, and initial simulation environment.
- **Key AI Models:**
  - LLaMA 3.3 for narrative construction.
  - palme (open-source PaLM-E) for vision-language understanding.
  - Whisper v3 for speech recognition and transcription.
- **Deliverables:**
  - Core repository structure.
  - Initial integration of LLaMA 3.3, palme, Whisper v3.
  - Basic Unreal Engine 5 simulation setup.
  - [`ConsciousnessCore`](../models/core/consciousness_core.py) V0.1 (stub).
  - [`EmotionalProcessingCore`](../models/emotion/emotional_processing.py) V0.1 (stub).
  - [`EmotionalMemoryCore`](../models/memory/emotional_memory_core.py) V0.1 (stub with Pinecone).

## Phase 2: Initial Integration & Basic Emotional Loop (Current Focus - S1: Instrumentation)

- **Goal:** Implement a basic emotional processing loop and memory storage within simple simulations. Develop instrumentation for metric collection.
- **Deliverables:**
  - Basic emotional processing loop.
  - Initial `EmotionalMemoryCore` integration.
  - Simple survival scenarios in Unreal Engine.
  - **Instrumentation Layer:** Develop `scripts/logging/metrics_logger.py` to capture hidden-state tensors from key modules, a prerequisite for advanced consciousness metrics. (Corresponds to Sprint S-1 from Watanabe notes)

## Phase 3: Enhanced Perception, Memory & Basic Self-Model (S2: Φ* & Ignition)

- **Goal:** Integrate advanced perception, refine memory, and introduce a basic self-representation. Implement initial consciousness metrics.
- **Deliverables:**
  - Integration of VideoLLaMA3 and Whisper.
  - More complex emotional memory indexing.
  - Initial version of `SelfRepresentationCore`.
  - **Φ* Calculator & Ignition Detector:**
    Implement initial versions of `models/evaluation/iit_phi.py` (for Φ*) and `models/evaluation/gnw_metrics.py` (for GNW Ignition Index). Develop a basic dashboard (e.g., Grafana, or a custom `consciousness_dashboard.py`) for these metrics. (Corresponds to Sprint S-2 from Watanabe notes)

## Phase 4: Advanced Social Interactions & Narrative Engine (S3: Indicator Tests)

- **Goal:** Enable more complex social interactions, integrate narrative reasoning, and expand behavioral evaluation.
- **Deliverables:**
  - Complex social scenarios in Unreal Engine.
  - `NarrativeEngine` V1 for contextual reasoning.
  - **Indicator-Property Test Suite:** Develop `models/evaluation/consciousness_metrics.py` with a substantial set of tests from the AI consciousness rubric. (Corresponds to Sprint S-3 from Watanabe notes)
  - Refinement of emotional RL based on social feedback.

## Phase 5: Dynamic Self-Representation & Meta-Cognition (S4: Self-Representation)

- **Goal:** Implement a dynamic, learned self-model and explore meta-cognitive capabilities.
- **Deliverables:**
  - **Dynamic Self-Representation Module:** Implement the learned "self-vector" loop within `ConsciousnessCore` and `SelfRepresentationCore` as per Higher-Order theories.
  - Reflective prompt templates and mechanisms for meta-cognitive evaluation.
  - Enhanced `ConsciousnessGating` informed by the dynamic self-model. (Corresponds to Sprint S-4 from Watanabe notes)

## Phase 6: Creative Simulation & Advanced Evaluation (S5: Imagination Buffer)

- **Goal:** Introduce mechanisms for creative simulation and refine advanced consciousness metrics.
- **Deliverables:**
  - **Creative Imagination Buffer:** Implement the "Imagination Buffer" for generating and evaluating novel mental simulations, potentially selecting based on Φ or GNW ignition.
  - Reward-shaping hooks based on creative outputs.
  - Advanced IIT metrics (e.g., CES visualization). (Corresponds to Sprint S-5 from Watanabe notes)

## Phase 7: Peer Consciousness & Robustness (S6: Peer Probes)

- **Goal:** Explore inter-agent awareness and conduct comprehensive system validation.
- **Deliverables:**
  - **Peer-Consciousness Probes:** Scenarios where two consciousness agents interact and attempt to model/estimate each other's (simulated) internal states. (Corresponds to Sprint S-6 from Watanabe notes)
  - Comprehensive ethical review and safety testing.
  - Long-term stability and learning assessments.

## Future Directions

- Full perceptual loop integration with robotics or advanced VR sensors.
- Subjective-report alignment processes (e.g., RLHF) so the agent’s language faithfully mirrors internal states.
- Exploration of more advanced AI consciousness theories and their computational correlates.
- Continuous refinement of the `AsimovComplianceFilter` and ethical governance.
- Development of `Consciousness-Metric.md` as a living spec for external contributors.
- Discussion on licensing for any developed datasets (e.g., indicator-property dataset).
