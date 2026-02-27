# The Consciousness AI

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Unity](https://img.shields.io/badge/Unity-ML--Agents-black)](https://unity.com/products/machine-learning-agents)

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

The system is built on a biologically grounded architecture informed by Feinberg & Mallatt's neuroevolutionary theory of consciousness. Six special neurobiological features guide the design: hierarchical depth, isomorphic mapping, reciprocal connections, oscillatory binding, nested compositional hierarchies, and neuron type diversity.

### 1. Sensory Tectum (Perception)

A multisensory spatial integration layer modeled after the biological optic tectum. Stacks aligned topographic maps for different sensory modalities in a common coordinate frame.

*   **Visual Pathway (Semantic):** [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (4-bit quantized). Processes visual streams and provides semantic scene understanding.
*   **Visual Pathway (Spatial):** V-JEPA / DreamerV3 RSSM world model. Maintains a structured latent representation preserving spatial, temporal, and causal relationships. This is the computational analog of biological topographic mapping.
*   **Auditory Cortex:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper). Real-time transcription of environmental audio.

### 2. Oscillatory Binding (Integration)

Based on [AKOrN](https://github.com/loeweX/AKOrN) (Artificial Kuramoto Oscillatory Neurons, ICLR 2025 oral). Treats neurons as oscillatory units on a hypersphere. Modules that synchronize are "bound" into unified percepts. Solves the binding problem through phase synchronization rather than single-point convergence.

### 3. Global Workspace (Consciousness)

*   **Global Neuronal Workspace (GNW):** A central information bottleneck where distinct sensory streams compete for broadcast access. Implements sigmoid ignition, recurrent reverberation, and reentrant processing (5-10 adaptive cycles with predictive coding convergence).
*   **Integrated Information (Phi):** Measures the causal integration of the workspace state. High Phi indicates a moment where the agent has fused disparate sensory data into a unified experience.
*   **Capsule Network Composition:** A nested compositional hierarchy where lower-level features (sensory) route to higher-level composites (objects, scenes) via dynamic routing by agreement, preserving part-whole relationships.

### 4. Affective Core (Emotion)

A parallel modulation system. Emotion does not compete with sensory modules for workspace access. Instead, it generates a **valence field** that modulates all sensory bids before competition, and a **global arousal signal** that adjusts the workspace ignition threshold.

*   **PAD Model:** Three intrinsic variables drive the agent: Valence (satisfaction/distress), Arousal (activation/calm), and Dominance (control/helplessness).
*   **Homeostatic Drives:** Persistent background drives (energy, safety, curiosity) generate ongoing valence signals even without external stimuli.

### 5. Self-Model (Embodiment)

*   **Body Schema:** A spatial representation of the agent's physical structure (joint positions, contact forces, capabilities).
*   **Self-Other Boundary:** The somatotopic map (self) overlaps the environment map (other) in a shared coordinate frame, providing the basis for subjective referral.
*   **Interoceptive State:** Internal homeostatic variables (energy, damage, arousal) feed into the affective core.

### 6. Reinforcement Core (Learning)

*   **Actor-Critic (PPO):** Emotionally shaped rewards. The agent is rewarded not just for task success, but for maintaining emotional homeostasis.
*   **Reward Formula:** `Rtotal = Rext + lambda1 * DeltaValence - lambda2 * (Arousal - Arousal_target)^2 + lambda3 * Dominance`

### 7. Simulation (Body)

*   **Unity ML-Agents:** The agent inhabits a physics-based Unity environment.
*   **Side Channels:** Bidirectional data streams to visualize the agent's internal state (Phi levels, oscillatory binding, emotional PAD) in real-time within the Unity HUD.

---

## Scientific Approach

The development validates emergent properties through:

1.  **Emotional Bootstrapping:** Train agents using intrinsic motivation. The agent explores to reduce prediction error (anxiety), not to accumulate external reward.
2.  **Binding Validation:** Phi measurement must correlate with oscillatory binding state (validated via 3-condition test: unbound, partial, full binding).
3.  **Reentrant Settling:** Conscious content emerges from iterative convergence (5-10 cycles), not single-pass processing.
4.  **Complexity Scaling:** Gradual increase of environment complexity forces the agent to develop higher-order world models.
5.  **Measurement:** Continuous monitoring of Phi (IIT), ignition events (GNW), and oscillatory synchronization (AKOrN order parameter R).

---

## Installation & Setup

### Requirements
*   **Python 3.10+**
*   **Unity 2022.3+** (LTS)
*   **NVIDIA GPU** (8GB+ VRAM recommended for Qwen2-VL)

### 1. Python Environment
```bash
git clone https://github.com/tlcdv/the_consciousness_ai.git
cd the_consciousness_ai
pip install -r requirements.txt
```

### 2. Unity Environment
1.  Open the `unity_project/` folder in Unity Hub.
2.  Install the **ML-Agents** package from the Package Manager.
3.  Drag the scripts from `unity_scripts/` (AgentManager.cs, etc.) onto your Agent GameObject.

### 3. Running a Simulation
```bash
# Start the Python Brain
python scripts/training/train_rlhf.py --env_id "ConsciousnessLab"
```
*Then press **Play** in the Unity Editor.*

---

## Documentation

*   [**Feinberg-Mallatt Approach**](docs/feinberg_mallatt_approach.md): How we translate Feinberg & Mallatt's neuroevolutionary theory into our artificial consciousness architecture.
*   [**Architecture Deep Dive**](docs/architecture.md): System design overview.
*   [**Biological Neural Architecture Research**](docs/biological_neural_architecture_research.md): Full biological grounding, gap analysis, and implementation roadmap.
*   [**Theory of Emergence**](docs/theory_of_consciousness.md): Scientific basis of the Emotional RL approach.
*   [**Theory vs. Implementation Review**](docs/theory_implementation_review.md): Audit of theoretical alignment and identified gaps.
*   [**IIT Implementation Roadmap**](docs/iit_implementation_roadmap.md): Phi computation strategy.
*   [**Simulation Guide**](docs/simulation_guide.md): How to build compatible Unity environments.

## Contributing

We welcome contributions from researchers in AI, Neuroscience, and Cognitive Science. Please read our [Contribution Guidelines](docs/contributing.md).

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
