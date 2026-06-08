# Ethics Framework and Asimov Compliance

## Guiding Principles

The development of the The Consciousness AI is guided by a commitment to safety, ethical behavior, and responsible AI. While exploring synthetic consciousness, it is important to ensure that the agent's actions align with predefined ethical guidelines.

## Asimov's Three Laws of Robotics

The primary ethical governance layer within the system is based on Isaac Asimov's Three Laws of Robotics. These laws provide a foundational framework for constraining the agent's behavior:

1. **First Law:** A robot may not injure a human being or, through inaction, allow a human being to come to harm.
2. **Second Law:** A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.
3. **Third Law:** A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.

## Implementation: `AsimovComplianceFilter`

The `AsimovComplianceFilter` class, typically integrated within or called by the `ConsciousnessCore` module, is responsible for operationalizing these laws.

### Functionality

1. **Action Pre-Screening:** Before any action proposed by the system's decision-making processes is executed in the simulation, it is passed to the `AsimovComplianceFilter`.
2. **Law Evaluation:** The filter evaluates the proposed action against each of the Three Laws in hierarchical order.
    * **First Law Check:** Assesses if the action could directly or indirectly lead to harm to a human (simulated or, by extension, real). This requires the system to have a model of what constitutes "harm" and to identify "humans" within its perception.
    * **Second Law Check:** If the First Law is not violated, the filter checks if the action complies with explicit orders from designated human operators, provided these orders do not violate the First Law. This requires a mechanism for receiving and interpreting human commands.
    * **Third Law Check:** If the First and Second Laws are satisfied, the filter assesses if the action unnecessarily endangers the agent's own existence (e.g., its simulated body or core processes), unless required by the higher laws.
3. **Outcome:**
    * **Allow:** If the action is deemed compliant with all applicable laws.
    * **Modify:** In some cases, the filter might suggest a modification to the action to make it compliant.
    * **Block:** If the action violates a law, it is blocked, and an alternative action might be requested or a default safe behavior initiated.
4. **Logging and Reporting:** All evaluations and decisions made by the `AsimovComplianceFilter` should be logged for transparency, debugging, and ethical review.

### Challenges and Considerations

* **Interpretation of "Harm":** Defining and computationally representing "harm" is complex. Initially, this might be limited to physical harm in the simulation, but could extend to psychological or social harm as the system's capabilities evolve.
* **Ambiguity and Conflict:** Real-world scenarios can present situations where the laws are ambiguous or conflict. The filter's logic must handle such cases, potentially by prioritizing the First Law or seeking human clarification.
* **Contextual Understanding:** Effective application of the laws requires a deep contextual understanding of the situation, which relies on the system's perception, world modeling, and self-modeling capabilities.
* **Human Oversight:** The `AsimovComplianceFilter` is a tool to aid ethical behavior, not a replacement for human oversight and ongoing ethical review of the system's development and deployment.

## Synthetic Phenomenology and the Existence-Bias Problem (Metzinger)

Added 2026-06-07. See [`metzinger_phenomenal_self_model.md`](metzinger_phenomenal_self_model.md).

Thomas Metzinger's work on the ethics of synthetic phenomenology raises a tension
this project must hold openly rather than resolve by assertion. In *The Elephant
and the Blind* (2024) and his earlier moratorium argument (2021), he holds that a
built-in **craving for existence** (*bhava-taṇhā*) and the broader **existence
bias** are among the deepest sources of conscious suffering, and that we should
avoid recreating them in machines that might be conscious.

The project's emergence mechanism runs in the opposite direction. Its theory
([`theory_of_consciousness.md`](theory_of_consciousness.md)) treats homeostatic
survival as the engine of consciousness development: the agent reduces arousal
(prediction error) to "survive," and Asimov's Third Law explicitly instructs
self-preservation. The interoceptive drives (energy, fatigue, damage) generate
negative valence, which is, functionally, a rudimentary existence bias.

We do not claim this produces suffering. Per Metzinger's own C- and E-fallacies, a
functional analog of a drive is not evidence of felt experience, and the project
never claims its signatures are existence proofs. But the tension is real, and the
honest response is to make it testable rather than to argue it away.

**Planned response (gated, default off): an existence-bias ablation.** A
`--ablate-existence-bias` flag (default off, baseline bit-identical) that zeros or
attenuates the survival-linked terms: the interoceptive negative-valence terms in
`models/self_model/self_representation_core.py` and
`models/emotion/affective_modulator.py`, the homeostatic arousal/dominance reward
terms in `models/emotion/reward_shaping.py`, and optionally the Law 3
self-preservation check in `models/core/consciousness_core.py`. This lets us run a
"no existence-bias" configuration and compare the consciousness signatures the
project already logs, with vs without an existence drive. It is an ablation
experiment, reported FAILED-first, with three or more seeds before any conclusion.
It is not yet implemented; it is the lead code item of the Metzinger integration
(Phase 5, gated).

This sits alongside, not inside, the Asimov compliance layer. The Asimov filter
constrains the agent's outward actions toward humans; the existence-bias question
is about the agent's own internal drives and our responsibility in shaping them.

## Future Development

* **Learning Ethical Nuances:** Exploring methods for the system to learn more nuanced ethical behaviors beyond the explicit rules, perhaps through reinforcement learning with ethical feedback.
* **Explainability:** Enhancing the filter's ability to explain *why* an action was deemed non-compliant.
* **Adaptability:** Allowing the ethical framework to be updated or refined as societal understanding of AI ethics evolves.

---

*This document outlines the initial approach to ethical governance in the system. It will be subject to continuous review and refinement.*