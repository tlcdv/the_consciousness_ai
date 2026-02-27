# Datasets Used in The Consciousness AI

This document provides a detailed overview of the datasets used in the system project, their applications, and licensing details.

---

## Emotion Recognition & Social Interaction Datasets

### 1. **GoEmotions**

- **Description**: A large-scale dataset for fine-grained emotion classification from text. Useful for understanding emotional tone in text interactions.
- **License**: [Apache 2.0 License](https://github.com/google-research/google-research/blob/master/LICENSE)
- **Application**: Training text-based emotion classifiers.
- **Link**: [GoEmotions GitHub](https://github.com/google-research/google-research/tree/master/goemotions)

### 2. **MELD (Multimodal EmotionLines Dataset)**

- **Description**: Multimodal dataset featuring audio, visual, and textual dialogues annotated for emotions and sentiment. Based on the TV show Friends.
- **License**: Available for research/commercial use (verify specific terms).
- **Application**: Training multimodal emotion recognition models, understanding emotion in dialogue context.
- **Link**: [MELD Dataset GitHub](https://github.com/declare-lab/MELD)

### 3. **HEU Emotion**

- **Description**: Dataset containing video clips with emotional annotations, including facial expressions and speech.
- **License**: Available for research/commercial use (verify specific terms).
- **Application**: Training video and speech-based emotion recognition models.
- **Link**: [HEU Emotion Dataset](https://arxiv.org/abs/2007.12519)

### 4. **RAMAS (Real-world Affective Measurement in dyAdic and Small group interactions)**

- **Description**: Rich multimodal dataset (video, audio, physiology) capturing naturalistic social interactions in small groups.
- **License**: Requires application/agreement for access.
- **Application**: Studying group dynamics, empathy, social signal processing, and complex emotional expressions in interaction.
- **Link**: [RAMAS Project](https://ramas-project.github.io/)

### 5. **MSP-IMPROV**

- **Description**: Multimodal dataset of acted emotional interactions between pairs of actors. Includes audio, video, and motion capture.
- **License**: Available for research (check specific terms for commercial use).
- **Application**: Training models for dyadic interaction analysis, emotion expression recognition.
- **Link**: [MSP-IMPROV Dataset](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)

### 6. **IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)**

- **Description**: Widely used multimodal dataset of dyadic interactions between actors, featuring audio, video, motion capture, and self-reported emotion labels.
- **License**: Requires license agreement for access.
- **Application**: Benchmark for multimodal emotion recognition, sentiment analysis, and interaction modeling.
- **Link**: [IEMOCAP Dataset](https://sail.usc.edu/iemocap/)

---

## Simulation and Interaction Datasets

### 7. **AI Habitat Datasets (Gibson, Matterport3D)**

- **Description**: Photorealistic 3D scans of real-world indoor environments, used within the AI Habitat simulation platform.
- **License**: Various (often research-focused, check specific dataset licenses).
- **Application**: Training agents for navigation, interaction, and potentially social simulation in realistic 3D environments.
- **Link**: [AI Habitat Datasets](https://aihabitat.org/datasets/)

### 8. **VirtualHome**

- **Description**: Simulates daily activities in household environments, providing action scripts and environment states.
- **License**: MIT License.
- **Application**: Training agents for task planning, complex interactions with objects, understanding household activities, and potentially simulating agent needs/goals.
- **Link**: [VirtualHome Project](http://virtual-home.org/)

---

## Usage Guidelines

1. Ensure compliance with the licensing terms of each dataset when integrating into the project.
2. Preprocess datasets according to the requirements of the system's training and testing pipelines.
3. Document the preprocessing steps in `/docs/preprocessing.md`.

---

## Suggestions for New Datasets

If you discover a dataset that could improve the system's capabilities, please follow the contribution process outlined in the [CONTRIBUTING.md](../CONTRIBUTING.md) file.

We welcome:

- Emotion datasets covering underrepresented modalities or scenarios.
- Simulation datasets enhancing interaction complexity.
- Multimodal datasets with innovative applications.

---

## Dataset Contributions

Dataset origins are noted in their respective sections. Community suggestions and integrations are welcome via the contribution process.

Thank you for supporting the growth of the system!
