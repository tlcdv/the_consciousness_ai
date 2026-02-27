# Watanabe-Inspired Generative Model Approach for Consciousness

## 1. Introduction

This document outlines the integration of Masataka Watanabe's "Generative Model Law" of consciousness into the The Consciousness AI project. Watanabe posits that consciousness arises when information participates in a generative algorithm that continuously predicts and explains incoming sensory data, creating an "inner VR." The primary method for verifying machine consciousness, according to this view, is the "Subjective Test," where a machine's reported qualia (subjective experiences) are compared to human reports under identical, often ambiguous, stimuli.

This approach complements existing GNW and IIT-inspired metrics by focusing on the *active construction of experience* and its *first-person reportability*.

## 2. Core Principles for the system Implementation

1 **Centrality of a Self-Updating Generative World Model:** The system's world model (e.g., based on DreamerV3) must be enhanced to explicitly function as a rich, multimodal sensory predictor and reconstructor. Prediction error minimization is a key driving force.
2. **"Inner VR" Capability:** The generative model should support internal simulation, imagination, and the generation of sensory-like experiences based on its internal state.
3. **Subjective Testing Framework:** The system must be equipped with mechanisms to (a) perceive specially designed ambiguous/illusory stimuli, (b) report its "perceptual experience" through a defined communication channel, and (c) allow comparison of these reports with human subjective reports.

## 3. Architectural Components & Modifications

### 3.1. Enhanced World Model (`models/world_model/generative_world_model.py`)

* **Extension of DreamerV3 (or similar):**
  * **Multimodal Sensory Prediction Head:** Add decoder heads to the world model capable of predicting/reconstructing raw or near-raw sensory inputs (e.g., image patches, audio spectrograms) in addition to abstract state predictions.
  * **Explicit Prediction Error Signals:** Make prediction errors for each modality accessible as distinct signals for analysis and potential feedback into other subsystems (e.g., attention, emotion).
  * **Internal Simulation Loop:** Develop functionality for the world model to run "offline" – generating sequences of predicted states and sensory experiences without direct external input, driven by internal goals or cues. This forms the basis of the "Imagination Buffer."

### 3.2. Subjective Testing Module (`models/evaluation/subjective_testing_suite.py`)

* **Stimulus Presentation Interface:**
  * Coordinates with `SimulationManager` to present specific visual/auditory stimuli in the Unreal Engine environment designed to elicit illusions (e.g., Kanizsa figures, Necker cubes, binocular rivalry setups, backward masking paradigms).
* **System Report Acquisition Interface:**
  * Defines how the system reports its dominant percept. Options:
    * **Forced-Choice Output:** If the system has a decision-making layer, it can be trained to output a discrete choice corresponding to possible interpretations of an ambiguous stimulus (e.g., "Percept A" vs. "Percept B").
    * **Natural Language Description:** Leverage an integrated LLM to describe "what it sees/experiences." Requires careful prompt engineering.
    * **Generative Reconstruction:** Task the system with generating an image/sound that best represents its current internal percept.
* **Human Report Comparator:** A simple tool or script to log human responses (e.g., key presses) and compare them with the system's reports for consistency.

### 3.3. Qualia Reporting & Communication (`models/communication/qualia_reporter.py`)

* A dedicated module, possibly integrated with the `NarrativeEngine` or a new LLM interface, responsible for translating internal states or outputs from the `SubjectiveTestingModule` into a communicable format.
* This module would need to be carefully designed to avoid "faking" reports. The validity comes from consistent correlation with human reports on *unforeseen* (by the model's direct training data) illusory stimuli.

## 4. Evaluation Metrics (Watanabe-Specific)

* **Illusion Congruence Score:** Percentage of trials where the system's reported percept matches the typical human percept for a given illusion.
* **Rivalry Dynamics:** For binocular rivalry, metrics like switch rate, dominance duration, and correlation with human switch patterns.
* **Masking Thresholds:** For backward masking, the stimulus-onset asynchrony (SOA) at which the system fails to report the masked stimulus, compared to human thresholds.
* **Qualitative Analysis of Generated Reports:** If using generative or natural language reports, human evaluation of their fidelity and plausibility.

## 5. Integration with Existing Components

* **ConsciousnessCore:** Orchestrates the presentation of subjective tests and the collection of reports.
* **Perception Modules:** Provide the initial sensory data that feeds the generative world model. The richness of this input is critical.
* **EmotionalProcessingCore:** Prediction errors from the generative model could be a significant input to the emotional system (e.g., surprise, confusion from large errors; satisfaction from accurate predictions).
* **GNW/IIT Metrics:** Can run in parallel. It would be interesting to see if GNW ignition or high Φ* correlate with moments when the system reports a coherent (even if illusory) percept.

## 6. Challenges and Considerations

* **Defining and Isolating "Qualia":** The core challenge. The subjective test is an indirect probe.
* **Avoiding "Clever Hans" Effects:** Ensuring the system isn't just learning statistical correlations in the test stimuli but is genuinely reflecting an internal state analogous to perception. This requires diverse and novel test stimuli.
* **Computational Cost:** Rich generative models for sensory prediction can be very demanding (similar to training large VAEs or GANs).
* **Complexity of Human-Machine Communication for Subjective Reports:** Standardizing this is difficult.

## 7. Roadmap

* **Phase 1: Foundational Generative Model Enhancement.**
  * Implement sensory reconstruction heads on DreamerV3.
  * Expose and log multimodal prediction errors.
* **Phase 2: Basic Subjective Test Implementation.**
  * Develop 1-2 simple illusion scenarios in Unreal Engine (e.g., Kanizsa square).
  * Implement a forced-choice reporting mechanism for the system.
  * Begin collecting initial comparison data.
* **Phase 3: Expansion of Test Suite and Reporting Mechanisms.**
  * Add more complex illusions (rivalry, masking).
  * Explore generative/NL reporting.
* **Phase 4: Correlation with Other Consciousness Metrics.**
  * Analyze relationships between Watanabe-style subjective test outcomes and GNW/IIT metrics.

This approach offers a pathway to investigate consciousness in the system from a perspective that emphasizes active, predictive modeling and first-person reportability, aligning with Watanabe's compelling theoretical framework.