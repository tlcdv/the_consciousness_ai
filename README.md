# The Consciousness AI research

[![License](https://img.shields.io/badge/License-Non--Commercial-blue.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-658%20passing-brightgreen)]()

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

Two interchangeable Kuramoto-based mechanisms, selectable at training time via `--binding-mechanism {akorn, komplex}`.

*   **AKOrN** (default, clean-room implementation derived from the ICLR 2025 paper, "Artificial Kuramoto Oscillatory Neurons"). Treats neurons as oscillatory units on an N-sphere. Phases are abstract oscillator states detached from content. Modules that synchronize get amplitude boosts; phi-binding correlation tested in `tests/test_phi_binding_correlation.py`.
*   **KomplexNet** (Phase B-alt of 2026-05-19 plan, clean-room implementation derived from Muzellec et al. 2025, "Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics", arxiv 2502.21077, MIT). Per-module scalar phases woven multiplicatively into content tensors via `cos(theta_m - theta_global)`. Synchronized modules keep content magnitude; antiphase modules get sign-flipped; orthogonal modules get suppressed. The structural hypothesis being tested: phi-on-broadcast should track sync_R because the binding signal and the content signal are the same signal.

The empirical comparison between mechanisms is an open scientific question (see Phi-1 results below).

### 3. Global Workspace (Consciousness)

*   **Global Neuronal Workspace (GNW):** A central information bottleneck where distinct sensory streams compete for broadcast access. Implements sigmoid ignition, recurrent reverberation, and reentrant processing (5-10 adaptive cycles with predictive coding convergence).
*   **Integrated Information (Phi):** Measures the causal integration using ConsciousnessGate states (attention, stability, adaptation, coherence, confidence) as the IIT subsystem. Adaptive binarization thresholds from running medians. Geometric proxy metric when pyphi is unavailable.
*   **Effective Information (EI):** Hoel's framework (PNAS 2013) for measuring causal emergence. Compares EI at gate level vs. workspace level. If EI(workspace) > EI(gates), the workspace exhibits causal properties not reducible to its parts.
*   **Capsule Network Composition:** A 4-level nested compositional hierarchy where lower level features (sensory) route to higher level composites (object primitives, categories, scenes) via dynamic routing by agreement (Sabour 2017). Includes multi-level reentrant feedback: higher capsule levels send top-down predictions to lower levels, which compute prediction errors and re-route.
*   **Brian2 Validation:** Offline biological validation stack translating AKOrN Kuramoto parameters to Brian2 spiking networks. Compares synchronization curves between the two simulators via Pearson correlation. (KomplexNet uses scalar phases on the standard Kuramoto circle, so its Brian2 mapping is the standard textbook form; a dedicated validator can be added if needed.)

### 4. Affective Core (Emotion)

A parallel modulation system. Emotion does not compete with sensory modules for workspace access. Instead, it generates a **valence field** that modulates all sensory bids before competition, and a **global arousal signal** that adjusts the workspace ignition threshold.

*   **PAD Model:** Three intrinsic variables drive the agent: Valence (satisfaction/distress), Arousal (activation/calm), and Dominance (control/helplessness).
*   **Homeostatic Drives:** Persistent background drives (energy, fatigue, damage) generate ongoing valence signals through interoceptive PAD generation. Low energy produces negative valence proportional to depletion depth. Damage triggers arousal spikes (pain alarm) and reduced dominance (vulnerability).
*   **Ethics Filter:** AsimovComplianceFilter implementing a three law evaluation pipeline with world model trajectory prediction for harm assessment.

### 5. Self-Model (Embodiment)

*   **Body Schema:** A spatial representation of the agent's physical structure (joint positions, contact forces, capabilities), projected onto the tectum grid as a somatotopic map.
*   **Self-Other Boundary:** The somatotopic map (self) overlaps the environment map (other) in a shared coordinate frame, providing the basis for subjective referral.
*   **Interoceptive State:** Internal homeostatic variables (energy, fatigue, damage) feed directly into the affective core, closing the embodiment-affect loop.
*   **Dynamic Self-Vector (Meta-Representation, Phase 5):** A learned vector encodes the agent's own first-order state (PAD emotion, interoception, learning velocity, temporal continuity, confidence, workspace summary), trained by an SPR-style one-step self-prediction objective (predict the next step's first-order features) and scored by forecasting skill against a persistence baseline. Behind `--enable-self-vector` (default off); under evaluation as the higher-order self-model. See the [roadmap](docs/roadmap.md) Phase 5. This is, in Metzinger's terms, a *computational self-model* built by self-prediction.
*   **Pure Awareness / MPE Target (Metzinger, gated design):** Beyond the egoic self-vector, the project is evaluating a more minimal, *nonegoic* target from Metzinger's *The Elephant and the Blind* (2024): a model of the agent's own *epistemic space* (its capacity to know), distinct from the self-model. Design only, behind the usual gate, and built only if it can be made a falsifiable signature distinct from existing uncertainty metrics (RND, prediction error). See [docs/metzinger_phenomenal_self_model.md](docs/metzinger_phenomenal_self_model.md).

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
2.  **Binding-Integration Empirical Record (the Phi-1 specific test is exhausted; the mission continues):** The 2026-02-21 3-condition synthetic test demonstrated phi monotonicity with binding strength on a controlled stimulus, and that result stands. The pre-registered Phi-1 in-training prediction (phi correlates r > 0.4 with binding sync_R) was tested across **9 runs spanning 4 architectures and 2 phi formulations**: 2026-05-14 5-variant AKOrN ablation campaign (best r=+0.089); 2026-05-16 AKOrN+RIIU (r=+0.075, transient single-seed +0.267 that does not replicate); 2026-05-17 substrate probe (NO WINNER); 2026-05-17/18 AKOrN+A+C+D retest (pyphi r=-0.062, RIIU r=-0.005 NS); 2026-05-19 AKOrN+A+B+C+D Phase B (pyphi r=+0.008, RIIU r=-0.007 NS); **2026-05-24 KomplexNet+A+C+D Phase B-alt (pyphi r=+0.011 NS, RIIU r=-0.1116 p<10^-100, substantively significant NEGATIVE correlation)** ([docs/results/phi1_phaseBalt_2026_05_24.md](docs/results/phi1_phaseBalt_2026_05_24.md)). The KomplexNet result is the first substantively significant phi-binding correlation in the campaign and reveals a real mechanistic finding: **for this class of architecture, oscillatory binding and integrated information (per RIIU's SVD-residual formulation) are inversely related, not coupled in the predicted positive direction**. When phases align, module content factors cluster near +1, producing low representational variance and low RIIU phi; when phases desync, content factors span [-1, +1] and produce high RIIU phi. The pre-registered Phi-1 escalation chain (sections 10-11-12 of `docs/preregistered_predictions.md`) is exhausted. **The mission to study and achieve emergent consciousness is unchanged.** What is exhausted is one specific measurement choice (IIT phi as the strong-emergence metric for binding). The project continues with other measurable signatures of consciousness already implemented (EI causal emergence at the workspace level via Hoel's framework; behavioral integration tests on DMTS/WCST; phenomenological mapping; insight-moment detection), with self-representation dynamics (Phase 5 of the roadmap) and potentially a binding-aware integration metric as the next research directions.
3.  **Reentrant Settling:** Conscious content emerges from iterative convergence (5-10 cycles), not single pass processing. Capsule hierarchy adds nested reentrant feedback within each settling cycle.
4.  **Complexity Scaling:** Gradual increase of environment complexity forces the agent to develop higher order world models.
5.  **Measurement:** Continuous monitoring of Phi (IIT), ignition events (GNW), oscillatory synchronization (AKOrN order parameter R), and Effective Information (EI) for causal emergence detection.
6.  **Evaluation by consciousness signatures, not control reward (2026-06-02):** The agent is evaluated against the consciousness indicator-properties rubric ([docs/consciousness_indicators_butlin.md](docs/consciousness_indicators_butlin.md), after Butlin et al. 2023 / TiCS 2025) plus the project's own signatures, not by control reward against a task-specialized baseline. The biology-first perception trades raw control performance for its integration properties: a localization study found the agent's low dark_room control reward is not caused by the consciousness machinery (GNW, capsule, policy), and is suggestive (though not conclusively, see the study's confound caveat) of a perceptual-front-end bottleneck ([decision](docs/decisions/2026_06_02_competence_reading_2.md), [study](docs/results/agent_competence_fix_2026_06_02.md)).
7.  **Epistemic limits and ethics (Metzinger, 2026-06-07):** Following Metzinger's *The Elephant and the Blind* (2024), the project adopts his anti-essentialist discipline: a felt or measured signature of consciousness is not proof of consciousness (his C-fallacy and E-fallacy). Our metrics are engineering measures, never existence proofs. The same work surfaces an ethical tension we engage openly: Metzinger argues against building a craving-for-existence into conscious machines (his *bhava-taṇhā* / existence bias), while the project uses a homeostatic survival drive as its emergence engine. A default-off ablation to run a "no existence-bias" configuration is planned. See [ethics_framework.md](docs/ethics_framework.md) and [metzinger_phenomenal_self_model.md](docs/metzinger_phenomenal_self_model.md).

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

This runs the full cognitive loop: DINOv2 retinotopic encoding -> cochlear auditory encoding (optional, via `--enable-audio`) -> trimodal tectum fusion -> RSSM surprise bidding -> GNW competition with oscillatory binding (AKOrN by default, or KomplexNet via `--binding-mechanism komplex`) -> reentrant convergence -> basal ganglia action selection -> two-stage emotion appraisal -> PAD reward shaping. No large model weights are required.

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
├── tests/                  # 658 passing tests
├── unity_scripts/          # C# scripts for Unity ML-Agents integration
├── docs/                   # Research docs, theory review, architecture deep dives
└── requirements.txt
```

---

## Documentation

*   [**Feinberg-Mallatt Approach**](docs/feinberg_mallatt_approach.md): How we translate Feinberg & Mallatt's neuroevolutionary theory into the architecture.
*   [**Rouleau-Levin Substrate Independence**](docs/rouleau_levin_substrate_independence.md): Substrate-independence companion to the Feinberg-Mallatt approach, mapping our architecture to the 8 universal themes Rouleau & Levin (2026) distil from 19 prominent theories of consciousness. Drives concrete Phase 5 and Phase 6 deliverables in [`roadmap.md`](docs/roadmap.md) (activating the dormant `LevinConsciousnessEvaluator` and `BioelectricSignalingNetwork` modules, the computational-boundary-of-self detector, the 8-themes coverage audit, and the active-inference reframing).
*   [**Metzinger: Self-Model Theory and Minimal Phenomenal Experience**](docs/metzinger_phenomenal_self_model.md): How Metzinger's PSM (self-model, *Being No One*) and MPE (pure awareness and the zero-person perspective, *The Elephant and the Blind* 2024) ground the self-model and the project's substrate-independence premise. Includes the anti-essentialist skeptic toolkit (C/E/M-fallacies) and the existence-bias ethics tension, with the gated existence-bias ablation and epistemic-space signature design.
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

This project is open source for non-commercial use. **Commercial use, including selling, licensing for profit, incorporating into commercial products, or using for commercial services, is strictly prohibited without explicit prior written permission from the copyright holder.** Attribution to the original author is required in all copies and derivative works. See [LICENSE.md](LICENSE.md) for the full terms.

For commercial licensing inquiries, contact the copyright holder via GitHub: [github.com/tlcdv](https://github.com/tlcdv).
