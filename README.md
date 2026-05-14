# The Consciousness AI research

[![License](https://img.shields.io/badge/License-Non--Commercial-blue.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-565%20passing-brightgreen)]()

**The Consciousness AI** is a research framework investigating the emergence of synthetic awareness. Unlike traditional AI that mimics intelligent output, this system generates behavior through an internal struggle for **Emotional Homeostasis** and **Integrated Information**.

We hypothesize that consciousness is not a programmable feature, but an emergent solution to the problem of surviving and maintaining stability in a complex, unpredictable environment.

## Core Principle: Functionalist Emergentism

The philosophical foundation is **Functionalist Emergentism**. This framework synthesizes two major perspectives:
1.  **Emergentism:** The ontological claim that consciousness is a novel, irreducible phenomenon that arises from complex systems.
2.  **Functionalism:** The methodological insight that mental states are defined by their causal roles, not their physical substrate.

We posit that consciousness emerges when systems achieve sufficient organizational complexity such that functional states acquire properties not reducible to their constituent parts. The architecture applies this by engineering the necessary conditions for awareness.

[**Read the full article on Functionalist Emergentism**](https://theconsciousness.ai/functionalist-emergentism/)

---

## Architecture

The system is built on a biologically grounded architecture informed by Feinberg & Mallatt's neuroevolutionary theory of consciousness (*The Ancient Origins of Consciousness*, MIT Press 2016). Six special neurobiological features guide the design: hierarchical depth, isomorphic mapping, reciprocal connections, oscillatory binding, nested compositional hierarchies, and neuron type diversity.

### 1. Sensory Tectum (Perception)

A multisensory spatial integration layer modeled after the biological optic tectum (superior colliculus). Stacks aligned topographic maps for visual, auditory, and somatosensory modalities in a common coordinate frame, fused via inverse effectiveness (Stein & Meredith 1993).

*   **Visual Pathway (Spatial):** [DINOv2-B/14](https://github.com/facebookresearch/dinov2) (frozen). Provides spatially faithful patch tokens with direct retinotopic correspondence. Each patch token at grid position (i,j) maps to a fixed 14x14 pixel region. Falls back to a 4-layer convolutional stack when model weights are unavailable (CI/testing).
*   **Visual Pathway (Semantic):** [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (4-bit quantized, optional). Provides high level scene understanding and language grounded perception. Not required for training.
*   **Auditory Pipeline (Cochlear):** A biologically grounded auditory system replacing the former Whisper transcription stubs. Models the mammalian auditory pathway from basilar membrane through auditory cortex:
    *   **Gammatone Filterbank** (frozen, 64 ERB bands, Patterson 1992): decomposes raw waveforms into frequency channels matching human cochlear resolution. Frozen parameters, paralleling DINOv2 in the visual pathway.
    *   **Inner Hair Cell Model:** half-wave rectification + temporal smoothing extracts envelope (rate code for loudness) and temporal fine structure (phase code for pitch).
    *   **Tonotopic Encoder** (trainable): 3-layer 1D conv stack preserving frequency-to-spatial-position mapping. Outputs `[B, 64, 16]` features for tectum grid integration.
    *   **Spatial Audio:** ITD (interaural time difference) and ILD (interaural level difference) binaural cues for sound source localization, fed into tectum inverse effectiveness fusion.
    *   **Acoustic Affect Extraction:** 6 features (spectral centroid, loudness variability, roughness, pitch contour slope, spectral flux, harmonic-to-noise ratio) mapped to PAD emotional state + paralinguistic classification (speech, laughter, crying, screaming, growling, sighing, silence).
    *   **Auditory Specialist:** chains all modules into a workspace competitor (oscillator #2). Competes for Global Workspace broadcast alongside vision, memory, body, and semantic modules. Supports reentrant top-down feedback.
    *   **Environment Audio Synthesis:** All four environments (Dark Room, Navigation, DMTS, WCST) generate procedural audio via FM synthesis and ADSR envelopes. Enabled with `--enable-audio` during training.
*   **Somatosensory Channel:** Body schema projected onto the spatial grid via learned linear mapping, enabling proprioceptive integration as a third sensory modality.
*   **Topographic Loss:** TDANN spatial loss (Margalit et al. 2024, Neuron) enforces topographic self-organization during training.
*   **RSSM World Model:** DreamerV3 style recurrent state space model maintains temporal predictions and generates surprise based bidding for workspace access.

### 2. Oscillatory Binding (Integration)

Based on [AKOrN](https://github.com/loeweX/AKOrN) (Artificial Kuramoto Oscillatory Neurons, ICLR 2025 oral). Treats neurons as oscillatory units on a hypersphere. Modules that synchronize are "bound" into unified percepts. Solves the binding problem through phase synchronization rather than single point convergence.

### 3. Global Workspace (Consciousness)

*   **Global Neuronal Workspace (GNW):** A central information bottleneck where distinct sensory streams compete for broadcast access. Implements sigmoid ignition, recurrent reverberation, and reentrant processing (5-10 adaptive cycles with predictive coding convergence).
*   **Integrated Information (Phi):** Measures the causal integration using ConsciousnessGate states (attention, stability, adaptation, coherence, confidence) as the IIT subsystem. Adaptive binarization thresholds from running medians. Geometric proxy metric when pyphi is unavailable.
*   **Effective Information (EI):** Hoel's framework (PNAS 2013) for measuring causal emergence. Compares EI at gate level vs. workspace level. If EI(workspace) > EI(gates), the workspace exhibits causal properties not reducible to its parts.
*   **Capsule Network Composition:** A 4-level nested compositional hierarchy where lower level features (sensory) route to higher level composites (object primitives, categories, scenes) via dynamic routing by agreement (Sabour 2017). Includes multi-level reentrant feedback: higher capsule levels send top-down predictions to lower levels, which compute prediction errors and re-route.
*   **Brian2 Validation:** Offline biological validation stack translating AKOrN Kuramoto parameters to Brian2 spiking networks. Compares synchronization curves between the two simulators via Pearson correlation.

### 4. Affective Core (Emotion)

A parallel modulation system. Emotion does not compete with sensory modules for workspace access. Instead, it generates a **valence field** that modulates all sensory bids before competition, and a **global arousal signal** that adjusts the workspace ignition threshold.

*   **PAD Model:** Three intrinsic variables drive the agent: Valence (satisfaction/distress), Arousal (activation/calm), and Dominance (control/helplessness).
*   **Homeostatic Drives:** Persistent background drives (energy, fatigue, damage) generate ongoing valence signals through interoceptive PAD generation. Low energy produces negative valence proportional to depletion depth. Damage triggers arousal spikes (pain alarm) and reduced dominance (vulnerability).
*   **Ethics Filter:** AsimovComplianceFilter implementing a three law evaluation pipeline with world model trajectory prediction for harm assessment.

### 5. Self-Model (Embodiment)

*   **Body Schema:** A spatial representation of the agent's physical structure (joint positions, contact forces, capabilities), projected onto the tectum grid as a somatotopic map.
*   **Self-Other Boundary:** The somatotopic map (self) overlaps the environment map (other) in a shared coordinate frame, providing the basis for subjective referral.
*   **Interoceptive State:** Internal homeostatic variables (energy, fatigue, damage) feed directly into the affective core, closing the embodiment-affect loop.

### 6. Reinforcement Core (Learning)

*   **Basal Ganglia Model:** Go/No-Go pathways modulated by simulated dopamine (reward prediction error). Includes direct pathway (D1, facilitates action), indirect pathway (D2, inhibits action), and hyperdirect pathway (STN, emergency brake).
*   **Reward Formula:** `Rtotal = Rext + lambda1 * DeltaValence - lambda2 * (Arousal - Arousal_target)^2 + lambda3 * Dominance`

### 7. Simulation (Body)

*   **Dark Room Environment:** A built in Gymnasium environment (`SimpleVisualEnv`) where the agent starts in darkness (high anxiety) and must find a light source to reduce prediction error. Renders via PyGame, provides raw pixel observations.
*   **Navigation Environment:** Multi-room grid with fog of war, colored goals with varying rewards, battery system, and doorway-based room transitions. Tests spatial memory and exploration strategy.
*   **Delayed Match-to-Sample (DMTS):** Gold standard consciousness task from animal research. Four phases (fixation, sample, delay, choice) with configurable distractor overlap. Requires working memory across 15-40 blank delay steps, feature binding, and selective attention. A reactive agent without consciousness machinery cannot solve this.
*   **Wisconsin Card Sort (WCST):** Tests meta-cognition and cognitive flexibility. The agent sorts cards by an unknown rule (shape, color, or count) that changes without warning after consecutive correct sorts. Requires error monitoring, hypothesis testing, and inhibition of previously correct strategies.
*   **DQN Baseline:** Vanilla DQN agent (3-layer CNN + MLP Q-network, epsilon-greedy, replay buffer) for controlled comparison. Same environment interface and logging format as the consciousness agent.
*   **Unity ML-Agents (optional, future):** Three C# scripts (`unity_scripts/`) provide the foundation for connecting to a physics based Unity environment via side channels. The Unity project itself is not yet included in the repository.

---

## Scientific Approach

The development validates emergent properties through:

1.  **Emotional Bootstrapping:** Train agents using intrinsic motivation. The agent explores to reduce prediction error (anxiety), not to accumulate external reward.
2.  **Binding Validation (under empirical test):** The 2026-02-21 3-condition synthetic test demonstrated phi monotonicity with binding strength on a controlled stimulus. The 2026-05-14 ablation campaign tested whether phi-sync_R correlation emerges during training across 5 architectural variants ([docs/results/ablation_2026_05_14.md](docs/results/ablation_2026_05_14.md)): correlation r ranged from +0.023 to +0.089, all failing the pre-registered r > 0.4 prediction. The relationship between binding and integration during learning remains an open question.
3.  **Reentrant Settling:** Conscious content emerges from iterative convergence (5-10 cycles), not single pass processing. Capsule hierarchy adds nested reentrant feedback within each settling cycle.
4.  **Complexity Scaling:** Gradual increase of environment complexity forces the agent to develop higher order world models.
5.  **Measurement:** Continuous monitoring of Phi (IIT), ignition events (GNW), oscillatory synchronization (AKOrN order parameter R), and Effective Information (EI) for causal emergence detection.

---

## Installation & Setup

### Requirements
*   **Python 3.10+**
*   **NVIDIA GPU** recommended (8GB+ VRAM for Qwen2-VL; CPU works for the Dark Room environment)

### 1. Clone and Install

```bash
git clone https://github.com/tlcdv/the_consciousness_ai.git
cd the_consciousness_ai
pip install -r requirements.txt
```

> **Note:** Some dependencies are optional. `pyphi` (IIT library) requires specific Python versions. `gymnasium` and `pygame` are needed for the Dark Room environment. The core architecture modules (tectum, GNW, binding, capsules) require only `torch`, `numpy`, and `einops`.

### 2. Running Training

```bash
# Run the Dark Room training loop (default: 20 episodes, 200 steps each)
python -m scripts.training.train_rlhf

# With custom parameters
python -m scripts.training.train_rlhf --episodes 50 --max-steps 300 --lr 1e-3

# Long runs: sample pyphi every 5th step to stay under the ~91k-call segfault threshold
python -m scripts.training.train_rlhf --episodes 200 --max-steps 200 \
  --ablate-gate-diversity --phi-sample-every 5

# DMTS environment (consciousness-demanding)
python -m scripts.training.train_rlhf --env dmts --episodes 500 --max-steps 500

# Wisconsin Card Sort (meta-cognition test)
python -m scripts.training.train_rlhf --env wcst --episodes 200 --max-steps 300

# Navigation environment (multi-room exploration)
python -m scripts.training.train_rlhf --env navigation --episodes 100

# DQN baseline for comparison
python -m scripts.training.train_baseline_dqn --env dark_room --episodes 100
python -m scripts.training.train_baseline_dqn --env dmts --episodes 500

# With cochlear auditory pipeline enabled
python -m scripts.training.train_rlhf --env dark_room --enable-audio --episodes 100

# With visual rendering
python -m scripts.training.train_rlhf --render
```

This runs the full cognitive loop: DINOv2 retinotopic encoding -> cochlear auditory encoding (optional, via `--enable-audio`) -> trimodal tectum fusion -> RSSM surprise bidding -> GNW competition with AKOrN binding -> reentrant convergence -> basal ganglia action selection -> two-stage emotion appraisal -> PAD reward shaping. No large model weights are required.

### 3. Running Tests

```bash
pytest tests/ -v
```

565 tests pass, covering oscillatory binding, capsule routing, reentrant processing, inverse effectiveness fusion, topographic loss, affective modulation, ethics compliance, effective information, IIT Phi with causal gate states, Brian2 biological validation, cochlear auditory pipeline (gammatone, hair cell, tonotopic, spatial, affect extraction), environment audio synthesis, DMTS/WCST consciousness demanding environments, DQN baseline, memory consolidation, semantic pathway, and full pipeline integration.

### 4. AKOrN Binding Demo

```bash
python scripts/demos/demo_akorn_binding.py
```

Visualizes Kuramoto oscillator synchronization dynamics on the workspace modules.

### 5. Unity Integration (Optional)

The `unity_scripts/` directory contains three C# scripts (`AgentManager.cs`, `ConsciousnessChannel.cs`, `EmotionChannel.cs`) for connecting to a Unity ML-Agents environment via side channels. Unity integration is under development. To use it, install `mlagents` separately:

```bash
pip install mlagents==0.29.0 mlagents-envs>=1.0.0
```

---

## Project Structure

```
the_consciousness_ai/
├── models/
│   ├── core/               # GNW, tectum, oscillatory binding, capsules, reentrant processor
│   ├── emotion/            # Affective modulator, reward shaping, PAD model
│   ├── evaluation/         # Phi (IIT), effective information (EI), consciousness metrics
│   ├── memory/             # FAISS backed emotional memory, episodic store
│   ├── audio/              # Cochlear auditory pipeline (gammatone, hair cell, tonotopic, spatial, affect)
│   ├── self_model/         # Action selection (basal ganglia), body schema, self-representation
│   ├── agent/              # ConsciousnessAgent (orchestrates the full cognitive loop)
│   ├── narrative/          # NarrativeEngine (LLM-backed with template fallback)
│   ├── validation/         # Brian2 biological validation stack
│   ├── vision_language/    # Qwen2-VL integration (optional semantic pathway)
│   └── predictive/         # DreamerV3 wrapper, attention mechanisms
├── simulations/
│   ├── environments/       # Dark Room, Navigation, DMTS, WCST environments
│   ├── scenarios/          # Consciousness, emotional, ethical, social scenarios
│   └── api/                # Simulation manager
├── scripts/
│   ├── training/           # Training (train_rlhf.py, train_baseline_dqn.py, metrics_logger.py)
│   ├── analysis/           # Analysis and comparison scripts
│   └── demos/              # AKOrN binding visualization
├── configs/                # YAML and Python configuration files
├── tests/                  # 565 passing tests
├── unity_scripts/          # C# scripts for Unity ML-Agents integration
├── docs/                   # Research docs, theory review, architecture deep dives
└── requirements.txt
```

---

## Documentation

*   [**Feinberg-Mallatt Approach**](docs/feinberg_mallatt_approach.md): How we translate Feinberg & Mallatt's neuroevolutionary theory into the architecture.
*   [**Architecture Deep Dive**](docs/architecture.md): System design overview.
*   [**Biological Neural Architecture Research**](docs/biological_neural_architecture_research.md): Full biological grounding, gap analysis, and implementation roadmap.
*   [**Theory of Emergence**](docs/theory_of_consciousness.md): Scientific basis of the Emotional RL approach.
*   [**Theory vs. Implementation Review**](docs/theory_implementation_review.md): Audit of theoretical alignment and identified gaps.
*   [**IIT Implementation Roadmap**](docs/iit_implementation_roadmap.md): Phi computation strategy.
*   [**Isomorphic Visual Mapping Research**](docs/isomorphic_visual_mapping_research.md): DINOv2, TDANN, and inverse effectiveness design rationale.
*   [**Pre-registered Predictions**](docs/preregistered_predictions.md): Testable EI, Phi, and insight moment predictions with falsification criteria.
*   [**Auditory System Design**](docs/auditory_system_design.md): Cochlear inspired audio pipeline design and biological rationale.
*   [**Experiment Results**](docs/results/experiment_comparison.md): Multi-environment training comparison (consciousness agent vs DQN baseline).
*   [**Simulation Guide**](docs/simulation_guide.md): How to build compatible environments.
*   [**Ethics Framework**](docs/ethics_framework.md): Asimov compliance filter design.

## Contributing

We welcome contributions from researchers in AI, Neuroscience, and Cognitive Science. Please read our [Contribution Guidelines](docs/contributing.md).

## License

Non-Commercial Open Source. See [LICENSE](LICENSE.md) for details.
