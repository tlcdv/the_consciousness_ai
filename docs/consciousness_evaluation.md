# Consciousness Evaluation

This document outlines the metrics and procedures for evaluating consciousness in the system, drawing from contemporary AI research and computational theories of consciousness (e.g., GNW, IIT, Higher-Order Theories). Metrics are primarily logged by the [`ConsciousnessMonitor`](../models/evaluation/consciousness_monitor.py) which calls specialized modules. A `metrics_logger.py` (planned for `scripts/logging/`) will be crucial for capturing necessary hidden-state tensors.

## Quantitative Metrics (Levels of Consciousness)

1. **Integrated Information (Φ and Φ*)**
    * **Module:** [`models/evaluation/iit_phi.py`](../models/evaluation/iit_phi.py) (Planned)
    * **Description:** Measure the complexity and integration of information processing.
        * **Φ:** Theoretical measure from IIT.
        * **Φ* (Mismatched Decoding):** A practical, GPU-friendly approximation (e.g., Oizumi et al.). Calculation: `phi = phi_star(z_t, z_t_minus1, partition=P)`, where `z_t` are concatenated hidden states. The implementation might reuse attention tensors from modules like `ace_core/ace_agent.py`.
    * **Output:** Scalar value for Φ/Φ*.

2. **Global Neuronal Workspace (GNW) Ignition Index**
    * **Module:** [`models/evaluation/gnw_metrics.py`](../models/evaluation/gnw_metrics.py) (Planned)
    * **Description:** Detects non-linear surges in workspace activations, indicating information becoming globally available. Mark an event when Δactivation > γ across ≥ k modules.
    * **Output:** Count or frequency of ignition events.

3. **Global Availability Latency**
   * **Module:** [`models/evaluation/gnw_metrics.py`](../models/evaluation/gnw_metrics.py) (Planned)
   * **Description:** Log wall-clock delay between a sensory event and its first reuse by another module, indicating speed of information propagation through the global workspace.
   * **Output:** Latency measurements.

4. **Global Workspace Ignition**

    (This seems like a duplicate of GNW Ignition Index - consider merging or clarifying if distinct)
   -Track how often broadcast events pass an ignition threshold.

5. **Perturbational Complexity Index (PCI)**
   * Apply external perturbations (memory wipes, random inputs) and measure system recovery time.

6. **Self-Awareness Score**
   * Query the self-model ([`SelfRepresentationCore`](../models/self_model/self_representation_core.py)) about internal states (e.g., simulated emotions, confidence), its identity as an AI, knowledge boundaries, or current operational context.
   * Count error corrections or track introspective queries.
   * Performance in mirror-test inspired simulation paradigms.

7. **Meta-Cognitive Capacity Metrics**
    * Track confidence levels reported by `SelfRepresentationCore` or `ConsciousnessCore` before and after key decisions.
    * Log instances of self-correction in planning or reasoning.
    * Evaluate performance on tasks requiring assessment of own knowledge (e.g., "Can you answer X?" followed by the actual attempt).

8. **Social Awareness Metrics (ToM-inspired)**
    * Performance in simulation scenarios designed to test understanding of other agents' goals or (simulated) beliefs based on their behavior.
    * Ability to adapt communication or actions based on the perceived emotional state of other agents.

9. **Situational Awareness Metrics**
    * Response accuracy and timeliness to critical environmental changes.
    * Ability to identify and adhere to context-specific rules or safety parameters within a simulation.
    * Correct identification of its operational mode (e.g., specific simulation type, test vs. "deployment" context).

## Qualitative Metrics (Content & Structure of Consciousness)

1. **IIT Cause-Effect Structure**
   * **Module:** [`models/evaluation/iit_phi.py`](../models/evaluation/iit_phi.py) (Planned)
   * **Description:** Beyond scalar Φ, analyze the cause-effect structure (CES) graph representing the quality of conscious experience.
   * **Output:** CES graph, potentially visualized (e.g., using NetworkX in a planned `consciousness_dashboard.py`).

2. **Self-Report Coherence**
   * **Module:** Integrated within `ConsciousnessMonitor` or a dedicated evaluation script.
   * **Description:** Compare the agent's language self-reports (e.g., about its focus or feelings) with its internal attention weights or `EmotionalProcessingCore` state.
   * **Output:** Coherence score (e.g., BLEU, coverage, or custom similarity metric).

## Behavioral & Capability-Based Evaluation

1. **Indicator-Property Rubric for AI Consciousness**
    * **Module:** [`models/evaluation/consciousness_metrics.py`](../models/evaluation/consciousness_metrics.py) (Planned)
    * **Description:** A suite of (e.g., 14) concrete behavioral tests derived from established indicator properties for AI consciousness. These can serve as unit/integration tests for specific conscious-like capabilities.
    * **Output:** Pass/fail or scores for each indicator.

Refer to [models/evaluation/consciousness_monitor.py](../models/evaluation/consciousness_monitor.py) for implementation details. The `ConsciousnessMonitor` class is responsible for orchestrating the calculation and tracking of these (and potentially other) metrics to provide an ongoing assessment of the system's state and emerging functional awareness.

### Addressing Evaluation Challenges

We acknowledge challenges in evaluating AI awareness, such as normative ambiguity and benchmark contamination (Li et al., 2025). Our approach attempts to mitigate some of these by:

* Focusing on functional, observable behaviors within diverse, custom-built simulation scenarios in Unreal Engine.
* Continuously refining metrics based on empirical results and theoretical advancements.
* Emphasizing internal consistency checks and behavioral diversity over single benchmark scores.
* Maintaining transparency in evaluation methodologies and limitations.

---
