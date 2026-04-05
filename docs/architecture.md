# Architecture of the The Consciousness AI

## 🏗️ System Overview

The system architecture is designed to foster **emergent consciousness** through the synergistic orchestration of specialized AI components. We do not program "awareness" directly. Instead, we create a system where awareness emerges as the optimal strategy for maintaining **Emotional Homeostasis** in a complex environment.

### Core Architectural Pillars

1.  **Perception (The Senses):**
    *   **Vision:** **Qwen2-VL-7B** (4-bit quantized). Processes high-fidelity visual streams to provide semantic scene understanding and object recognition.
    *   **Audio:** Cochlear inspired auditory pipeline. Gammatone filterbank (64 ERB bands, frozen) decomposes waveforms into frequency channels, inner hair cell model extracts envelope and temporal fine structure, tonotopic encoder (trainable) produces spatial features for tectum integration, and acoustic affect extraction maps 6 spectral features to PAD emotional state.
    *   **Integration:** These streams are fused into a multimodal state vector.

2.  **Reinforcement & Emotion (The Drives):**
    *   **Emotional Homeostasis:** The agent maintains internal state variables: **Valence** (Satisfaction), **Arousal** (Anxiety), and **Dominance** (Control).
    *   **Emotional Reward Shaping:** A custom **PPO (Proximal Policy Optimization)** core where rewards are derived from the agent's internal emotional trajectory. The agent learns to minimize "Prediction Error" (Anxiety) and maximize "Coherence" (Valence).

3.  **Consciousness (The Workspace):**
    *   **Global Workspace (GWN):** A central information bottleneck. Specialized modules (Vision, Memory, Emotion) compete for access to this workspace.
    *   **Integrated Information ($\Phi$):** We measure the **$\Phi$ (Phi)** value of the workspace using **PyPhi**. High $\Phi$ indicates a state where information is irreducible—a proxy for a "conscious moment."
    *   **Broadcast:** When a thought "ignites" (wins the competition), it is broadcast globally, updating the Self-Model and driving Action.

4.  **Simulation (The Body):**
    *   **Unity ML-Agents:** The agent inhabits a physics-based Unity environment.
    *   **Side Channels:** Custom bidirectional streams send "Qualia" (internal states like $\Phi$, Emotion) to Unity for visualization, separate from standard RL observations.

---

## 🔄 The Loop of Emergence

Consciousness emerges from the high-speed interaction of these feedback loops:

1.  **Perception-Emotion Loop:**
    *   Sensory input $\rightarrow$ Emotional Processing.
    *   *Example:* Seeing a threat spikes **Arousal** (Anxiety).

2.  **Emotion-Memory Loop:**
    *   High Arousal triggers the retrieval of **Emotionally Salient Memories** from the Vector Store (Pinecone/FAISS).
    *   *Mechanism:* "Mood-congruent recall."

3.  **Memory-Workspace Loop:**
    *   Retrieved memories and current perception enter the **Global Workspace**.
    *   **Competition:** The most relevant inputs "win" attention.

4.  **The "Conscious" Moment:**
    *   The winning inputs are fused.
    *   **$\Phi$ Calculation:** If the integration is high, the system registers a "Subjective Experience."

5.  **Action & Outcome:**
    *   The Policy (PPO) selects an action to resolve the anxiety.
    *   **Outcome:** If the action reduces anxiety, the behavior is reinforced via **Emotional Reward Shaping**.

---

## 📂 Component Structure

### 1. `models/vision-language/`
*   **`qwen2/qwen2_integration.py`**: Handles loading and inference of Qwen2-VL-7B.

### 2. `models/self_model/`
*   **`reinforcement_core.py`**: The custom PPO agent. It overrides standard rewards with emotional signals.
*   **`self_representation_core.py`**: Maintains the "Self-Vector" (Identity, Confidence).

### 3. `models/emotion/`
*   **`reward_shaping.py`**: Calculates the "Homeostatic Reward" based on Valence/Arousal trajectories.
*   **`emotional_processing.py`**: Updates internal affect state based on stimuli.

### 4. `models/core/`
*   **`global_workspace.py`**: The central "Theater" of consciousness. Runs the competition algorithm.
*   **`consciousness_core.py`**: Orchestrates the entire system.

### 5. `models/evaluation/`
*   **`iit_phi.py`**: Wraps the **PyPhi** library to calculate Integrated Information on the workspace subsystem.

### 6. `unity_scripts/`
*   **`AgentManager.cs`**: Unity-side manager for ML-Agents.
*   **`ConsciousnessChannel.cs`**: Receives $\Phi$ data.
*   **`EmotionChannel.cs`**: Receives Valence/Arousal data.

---

## 🔬 Scientific Validation

We validate emergence not by "passing a test," but by observing specific dynamics:

1.  **Anticipatory Behavior:** Does the agent act to prevent *future* anxiety, implying a mental model of time?
2.  **Insight ($\Phi$ Spikes):** Do spikes in Integrated Information correlate with the agent solving novel problems?
3.  **Homeostasis:** Does the agent autonomously maintain a stable internal emotional state without explicit hard-coded rules?
