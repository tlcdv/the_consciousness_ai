# Processing Pipeline

This document outlines the typical flow of information and processing within The Consciousness AI during a single perception-action cycle.

## Core Loop

1.  **Perception:**
    *   Raw sensor data (video, audio, text) from the simulation environment (Unreal Engine 5 via [`SimulationManager`](simulations/api/simulation_manager.py)) is received.
    *   Specialized models process this data into structured representations. DINOv2 retinotopic encoder handles spatial vision, Qwen2-VL provides semantic understanding, and the cochlear auditory pipeline (gammatone filterbank, inner hair cell model, tonotopic encoder) processes audio into frequency decomposed features and affect signals.

2.  **State Update & Integration (`ConsciousnessCore`):**
    *   The [`ConsciousnessCore`](models/core/consciousness_core.py) receives the processed perceptual information.
    *   **Emotional Processing:** The [`EmotionalProcessingCore`](models/emotion/emotional_processing.py) analyzes the input and current context to update the agent's internal emotional state (valence, arousal, specific emotions).
    *   **Memory Retrieval:** Relevant memories are retrieved from the [`EmotionalMemoryCore`](models/memory/emotional_memory_core.py) based on perceptual cues and emotional state.
    *   **Self-Model Update:** The [`SelfRepresentationCore`](models/self_model/self_representation_core.py) updates its model of the agent's status.
    *   **World Model Update:** The predictive world model (DreamerV3) updates its internal state based on the new observation.
    *   **Gating/Attention:** The [`ConsciousnessGating`](models/core/consciousness_gating.py) mechanism selects the most salient information for further processing.

3.  **Action Selection (`ConsciousnessCore`):**
    *   Based on the integrated state (perception, emotion, memory, goals, self-model), the core generates a potential action using its planning or policy components (potentially involving [`NarrativeEngine`](models/narrative/narrative_engine.py)).
    *   **Ethical Filtering:** The proposed action is evaluated by the `AsimovComplianceFilter`.
    *   If compliant, the action is approved. If not, a safe fallback action is chosen.

4.  **Action Execution:**
    *   The approved action is sent to the [`SimulationManager`](simulations/api/simulation_manager.py) or [`ACEConsciousAgent`](models/ace_core/ace_agent.py) to be executed in the simulation environment (Unreal Engine 5, NVIDIA ACE).

5.  **Learning & Memory Update:**
    *   **Outcome Observation:** The results of the action (new state, rewards, social feedback) are observed from the simulation.
    *   **Emotional Reward Shaping:** The [`EmotionalRewardShaper`](models/evaluation/consciousness_development.py) calculates a reward based on task success, ethical compliance, and crucially, the change in the agent's internal emotional state resulting from the action's outcome. *This step is vital for learning empathetic behaviors, as actions leading to positive internal states (potentially from positive social outcomes) are reinforced.*
    *   **Experience Storage:** The complete transition (state, action, reward, next_state, emotional_state) is stored in the [`EmotionalMemoryCore`](models/memory/emotional_memory_core.py) / experience replay buffer.
    *   **Model Training:** The RL agent (DreamerV3) and potentially other adaptable components (e.g., via PEFT) are updated using the stored experiences and rewards.
    *   **Memory Consolidation:** Background processes ([`MemoryConsolidationManager`](models/memory/consolidation.py)) may consolidate important experiences into long-term memory.

6.  **Monitoring:**
    *   The [`ConsciousnessMonitor`](models/evaluation/consciousness_monitor.py) periodically calculates and logs consciousness metrics based on the system's state and activity.

This loop repeats continuously, allowing the agent to perceive, feel, decide, act, and learn from its experiences within the simulated world.

// ... (Optional: Add diagrams or more detailed sub-flows) ...
