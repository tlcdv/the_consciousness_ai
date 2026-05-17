# Architecture of The Consciousness AI

## System Overview

The system architecture fosters emergent consciousness through orchestrated specialized components. Awareness is not programmed directly. Instead, the system creates conditions where awareness emerges as the optimal strategy for maintaining Emotional Homeostasis in a complex environment.

### Core Architectural Pillars

1.  **Perception (The Senses):**
    *   **Vision (Spatial):** DINOv2-B/14 (frozen). Provides spatially faithful patch tokens with direct retinotopic correspondence for the Sensory Tectum spatial pathway.
    *   **Vision (Semantic):** Qwen2-VL-7B (4-bit quantized, optional). Provides high level scene understanding via the SemanticPathway, competing in the workspace as the 5th oscillator.
    *   **Audio:** Cochlear inspired auditory pipeline. Gammatone filterbank (64 ERB bands, frozen) decomposes waveforms into frequency channels, inner hair cell model extracts envelope and temporal fine structure, tonotopic encoder (trainable) produces spatial features for tectum integration, acoustic affect extraction maps 6 spectral features to PAD emotional state.
    *   **Somatosensory:** Body schema projected onto the tectum spatial grid via learned linear mapping for proprioceptive integration.
    *   **Fusion:** Trimodal inverse effectiveness fusion in the TopographicMap (Stein & Meredith 1993). TDANN topographic loss (Margalit 2024) enforces spatial self-organization.

2.  **Reinforcement and Emotion (The Drives):**
    *   **Emotional Homeostasis:** The agent maintains internal state variables: Valence (Satisfaction), Arousal (Anxiety), and Dominance (Control).
    *   **Affective Modulator:** Emotion does not compete for workspace access. Instead, it modulates bids via a valence field and adjusts the GNW ignition threshold via arousal coupling. Interoceptive drives (energy, fatigue, damage) generate additional PAD signals.
    *   **Emotional Reward Shaping:** `Rtotal = Rext + lambda1*DeltaValence - lambda2*(Arousal - Arousal_target)^2 + lambda3*Dominance`. Homeostatic arousal term penalizes deviation from optimal arousal, not arousal itself.

3.  **Consciousness (The Workspace):**
    *   **Global Workspace (GNW):** A central information bottleneck. Specialist modules (vision, audio, memory, body, semantic) compete for broadcast access via sigmoid ignition.
    *   **Oscillatory Binding (AKOrN):** Kuramoto oscillators synchronize module representations. Modules that process related information phase-lock into unified percepts.
    *   **Capsule Composition:** 4-level nested hierarchy (Sabour 2017) between tectum and workspace. Multi-level reentrant feedback: higher levels send predictions down, lower levels send errors up.
    *   **Reentrant Processing:** 5-10 adaptive convergence cycles with predictive coding. The settled state after convergence IS the conscious content.
    *   **Integrated Information (Phi):** Measured via ConsciousnessGate states (attention, stability, adaptation, coherence, confidence) as the IIT subsystem. Sliding-window TPM tracks real-time dynamics.
    *   **Effective Information (EI):** Hoel's framework (PNAS 2013) for causal emergence detection. Tertile quantile binning captures gate trajectory structure.

4.  **Simulation (The Body):**
    *   **Dark Room:** Built in Gymnasium environment. The agent starts in darkness and must find a light source to reduce prediction error.
    *   **Navigation:** Multi-room grid with fog of war, colored goals with varying rewards, battery system.
    *   **DMTS (Delayed Match-to-Sample):** Gold standard consciousness task. Requires working memory across a blank delay period, feature binding, and selective attention.
    *   **WCST (Wisconsin Card Sort):** Tests meta-cognition, cognitive flexibility, and hypothesis testing. Hidden rule changes without warning.
    *   **DQN Baseline:** Vanilla DQN for controlled comparison on the same environments.

---

## The Loop of Emergence

Consciousness emerges from the interaction of these feedback loops:

1.  **Perception-Emotion Loop:** Sensory input drives reflex emotion (surprise, reward delta). Arousal and valence modulate workspace ignition threshold and bid values.

2.  **Emotion-Memory Loop:** High attention/consciousness gates memory storage. Retrieved memories compete for workspace access alongside sensory modules.

3.  **Workspace Competition Loop:** Modules submit bids, AKOrN binding synchronizes related representations, sigmoid ignition selects winners, broadcast content reaches all modules.

4.  **Reentrant Loop:** Broadcast feeds back to specialists (top-down prediction). Specialists update bids (bottom-up error). 5-10 cycles until convergence.

5.  **Consciousness-Action Loop:** Phi-weighted exploration scales noise inversely with integration level (high phi = exploit, low phi = explore) as the intended mechanism. As of 2026-05-17 the phi signal (both pyphi and RIIU pathways) is too small in absolute scale to drive measurable exploration changes; phi mean sits near 1e-4 to 3e-4. Gate reconstruction loss trains gate networks to predict broadcast dynamics. See `docs/preregistered_predictions.md` sections 7-8 for the empirical state of Phi-driven behavior.

6.  **Action and Outcome:** Basal ganglia model (Go/No-Go pathways) selects actions. Emotional reward shaping reinforces behaviors that maintain homeostasis.

---

## Component Structure

### 1. `models/core/`
*   **`global_workspace.py`**: Central workspace. Runs competition, AKOrN binding, sigmoid ignition, and reverberation.
*   **`sensory_tectum.py`**: Topographic spatial maps with DreamerV3 RSSM, trimodal inverse effectiveness fusion, capsule composition.
*   **`reentrant_processor.py`**: 5-10 adaptive convergence cycles wrapping GNW competition.
*   **`oscillatory_binding.py`**: AKOrN Kuramoto oscillatory neurons (ICLR 2025).
*   **`capsule_composition.py`**: 4-level hierarchical capsule routing with multi-level reentrant feedback.
*   **`retinotopic_encoder.py`**: DINOv2-B/14 frozen backbone with conv stack fallback.
*   **`semantic_pathway.py`**: Qwen2-VL embeddings as 5th workspace oscillator.
*   **`consciousness_core.py`**: Orchestrates the full system. AsimovComplianceFilter ethics.
*   **`consciousness_gating.py`**: 5-node gate subsystem (attention, stability, adaptation, coherence, confidence). Broadcast predictor for reconstruction loss.

### 2. `models/emotion/`
*   **`affective_modulator.py`**: Parallel modulator. Valence field, arousal-threshold coupling, interoceptive PAD generation.
*   **`reward_shaping.py`**: Homeostatic reward formula with PAD terms.
*   **`emotional_processing.py`**: PAD emotional processing core with EMA smoothing.

### 3. `models/evaluation/`
*   **`iit_phi.py`**: IIT Phi from causal gate states. Sliding-window TPM, adaptive binarization, geometric proxy.
*   **`effective_information.py`**: Hoel's EI framework for causal emergence.
*   **`consciousness_monitor.py`**: Metric-only consciousness evaluation (no circular progress factor).

### 4. `models/audio/`
*   **`gammatone_filterbank.py`**: 64 ERB-scale gammatone filters (frozen).
*   **`hair_cell_model.py`**: Half-wave rectification + temporal smoothing.
*   **`tonotopic_encoder.py`**: Trainable 1D conv stack preserving frequency-to-position mapping.
*   **`auditory_specialist.py`**: Chains all audio modules into a workspace competitor.
*   **`spatial_audio.py`**: ITD/ILD binaural localization.
*   **`audio_affect_extractor.py`**: 6 features to PAD + paralinguistic classification.

### 5. `models/self_model/`
*   **`action_selection_core.py`**: Basal ganglia Go/No-Go pathways with dopamine modulation.
*   **`self_representation_core.py`**: Body schema, interoceptive state, capability model.

### 6. `models/memory/`
*   **`memory_core.py`**: Experience storage and retrieval with emotional context.
*   **`emotional_memory_core.py`**: FAISS-backed emotional memory index.

### 7. `scripts/training/`
*   **`train_rlhf.py`**: Full cognitive loop training. Exercises all components.
*   **`train_baseline_dqn.py`**: Vanilla DQN for controlled comparison.
*   **`metrics_logger.py`**: TensorBoard + CSV logging with EI computation and insight detection.

---

## Scientific Validation

We validate emergence by observing specific dynamics:

1.  **Anticipatory Behavior:** Does the agent act to prevent future anxiety, implying a mental model of time?
2.  **Insight (Phi Spikes):** Do spikes in Integrated Information correlate with the agent solving novel problems?
3.  **Homeostasis:** Does the agent autonomously maintain a stable internal emotional state without explicit hard-coded rules?
4.  **Causal Emergence:** Does EI(workspace) exceed EI(gates), indicating the whole carries information the parts do not?
5.  **Binding Necessity:** Does disrupting oscillatory synchronization degrade performance on consciousness-demanding tasks (DMTS, WCST)?
