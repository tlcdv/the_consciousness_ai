# Rouleau-Levin Substrate Independence: Why Our Functional Principles Are Not Neuron-Specific

*Project: The Consciousness AI — An open-source implementation of artificial consciousness*
*Companion document to [`feinberg_mallatt_approach.md`](feinberg_mallatt_approach.md)*
*Source paper archived at `info how to measure consciousness/rouleau_levin_2026_brains_and_where_else.pdf`*
*Date: May 2026*

---

## 1. Introduction

This document records the integration of **Rouleau, N. & Levin, M. (2026). "Brains and where else? Mapping theories of consciousness to unconventional embodiments." *Phil. Trans. R. Soc. A* 384: 20250082** (doi: 10.1098/rsta.2025.0082) into the project's theoretical framework.

The paper's central thesis: most contemporary theories of consciousness (ToCs) are not actually about brains. When their substrate- and scale-dependent vocabulary is stripped away ("neural", "cortical", "synaptic"), what remains is a set of *functional principles* that any sufficiently organised substrate can host. The authors survey 19 prominent ToCs and produce **aneurocentric reformulations** for each, then distil eight universal themes:

1. Predictive modelling
2. Enactivism / ecological interactions
3. Re-entrancy or looped feedback
4. Meta-representations
5. Attentional gating / monitoring
6. Emergence from computation
7. Integration of information
8. Coarse-graining

They argue this matters because the field of diverse intelligence (cellular cognition, gene-regulatory networks, xenobots, organoids, slime moulds, biobots, hybrots) is rapidly expanding the class of plausible minds, and any ToC that *can only* point at brains is failing as a theory of consciousness as such.

For this project, the paper is the explicit theoretical justification for what the architecture has been doing all along: implementing the *organisational principles* of biological consciousness on a non-biological substrate (PyTorch tensors, Kuramoto oscillators, capsule networks, RSSM world models). It is the substrate-independence companion to the substrate-specific blueprint described in [`feinberg_mallatt_approach.md`](feinberg_mallatt_approach.md).

---

## 2. Relationship to Feinberg & Mallatt

Feinberg-Mallatt and Rouleau-Levin answer **different questions**:

| Question | Answered by | Position |
|----------|-------------|----------|
| What neural architecture did biology *actually use* to generate consciousness? | Feinberg & Mallatt (2016) | The six special neurobiological features (tectum, hierarchies, isomorphic maps, reentrant connections, oscillatory binding, neuron-type diversity) are a **sufficient** blueprint. 520 million years of evolution chose it. |
| Are those features *necessary*, or could other substrates host the same principles? | Rouleau & Levin (2026) | They are **not necessary**. Almost every feature Feinberg-Mallatt highlight has homologues outside neurons: bioelectric signalling in all cells, gap junctions, reentrant feedback in gene-regulatory networks, memory in immune cells, problem-solving in slime moulds, anticipatory behaviour in plants. |

The two views are **complementary**, not in conflict. Feinberg-Mallatt tells us *what to build* (the sufficient blueprint biology validates). Rouleau-Levin tells us *why building it in PyTorch is not a category error* (because the blueprint was always functional).

A pre-emptive objection the project has not previously addressed in writing: "your `oscillatory_binding.py` is not really neurons, your `sensory_tectum.py` is not really a midbrain, your `global_workspace.py` is not really frontoparietal cortex — so your claim to be biologically grounded is empty." The aneurocentric analysis is the principled answer. Feinberg-Mallatt's features are organisational, not material. A Kuramoto oscillator on the N-sphere implements *gamma-band synchronisation as a binding mechanism*. A topographic map of RSSM latents implements *isomorphic spatial referral*. A reentrant convergence loop implements *reciprocal predictive coding*. The substrate is different; the principle is the same.

This is consistent with how Feinberg-Mallatt themselves describe consciousness ("Phenomenal Consciousness and Emergence: Eliminating the Explanatory Gap", *Frontiers in Psychology* 2020): emergent on a *type* of organisation, not on a *type* of matter.

---

## 3. The Eight Aneurocentric Themes Mapped to Our Architecture

Rouleau & Levin show that the 19 ToCs they survey collapse into eight functional themes. Every one of those themes is already implemented in this project — by design, not by accident. The following table is the empirical content of the substrate-independence claim:

| Theme | Where in our codebase | Notes |
|-------|----------------------|-------|
| **1. Predictive modelling** | `models/core/sensory_tectum.py` (RSSM), `models/predictive/`, prediction-error term in `models/core/reentrant_processor.py` | DreamerV3-style recurrent state-space model maintains world predictions; reentrant cycles terminate on prediction-error convergence. |
| **2. Enactivism / ecological interactions** | `simulations/environments/` (Dark Room, Navigation, DMTS, WCST), embodiment-affect loop in `models/self_model/` + `models/emotion/` | Action-perception loops with interoceptive feedback; reward is shaped by homeostatic deviation, not external scalar. |
| **3. Re-entrancy / looped feedback** | `models/core/reentrant_processor.py` (5-10 adaptive cycles), multi-level capsule feedback in `models/core/`, GNW reverberation | Bidirectional flow: predictions down, errors up. Convergence-based early termination. |
| **4. Meta-representations** | `models/self_model/self_representation_core.py` and its Phase 5 `SelfVectorModule` | Currently partial. Phase 5 deepens this with a dynamic self-vector loop. (`meta_learner.py` and other orphaned self-model files were removed in the 2026-05-31 architecture audit; see `docs/architecture_audit_2026_05_31.md`.) |
| **5. Attentional gating / monitoring** | `ConsciousnessGate` (attention, stability, adaptation, coherence, confidence) in `models/core/global_workspace.py`, non-linear ignition | The five-gate subsystem IS the IIT phi-measurement substrate. |
| **6. Emergence from computation** | `models/evaluation/effective_information.py` (Hoel PNAS 2013), `compare_ei_levels()` macro-vs-micro test | We do not assume emergence; we measure EI(workspace) vs EI(gates) and falsify when EI(workspace) ≤ EI(gates). |
| **7. Integration of information** | `models/evaluation/iit_phi.py` (pyphi + RIIU SVD-residual), causal gate-state TPM | Empirical Phi-1 record reported honestly in README; the *measurement choice* is what the 2026-05-24 KomplexNet result exhausted, not the theme itself. |
| **8. Coarse-graining** | Workspace binarisation in `iit_phi.py`, EI macro/micro level comparison, capsule hierarchy collapse | Each capsule level coarse-grains the level below; Phi is computed over a coarse-grained gate-state TPM. |

Every theme has at least one concrete implementation file. The project is, structurally, an aneurocentric ToC engine: it instantiates the principles Rouleau & Levin extract from the surveyed theories, on a non-neuronal substrate, and measures whether the principles produce the predicted dynamics.

A deeper reading of Table 1 in the paper maps individual ToCs we lean on (GWT, IIT, predictive processing, active inference, HOT, local recurrency, dynamic core, self-comes-to-mind, beast machine, dendritic integration) to their aneurocentric reformulations. None of those reformulations excludes our architecture.

---

## 4. Levin's Specific Contributions and Our Dormant Modules

Beyond the survey, Michael Levin's broader research programme contributes ideas that this project has *pre-emptively scaffolded* in code but has not yet woven into the consciousness monitor or training loop:

- **Bioelectricity as the "cognitive glue"** (Levin 2023, *Animal Cognition* 26, 1865) — ion-channel-mediated voltage patterns coordinate cellular behaviour across scales; the same mathematics that describes gap-junction networks describes neural assemblies, but applies to all somatic cells.
- **Scale-free cognition and holonic intelligence** (Levin 2019, "The Computational Boundary of a 'Self'", *Front. Psychol.* 10, 2688) — selves are nested agents whose boundaries are defined by what they can integrate causally, not by anatomical edges.
- **TAME framework — Technological Approach to Mind Everywhere** (Levin 2022, *Front. Syst. Neurosci.* 16, 768201) — a methodological argument that the tools of neuroscience (decoding, perturbation, behavioural testing) can and should be applied to non-neural agential systems.
- **Morphogenesis as cognition** (Pezzulo & Levin 2015 onwards) — embryonic development as goal-directed navigation of morphospace, with bioelectric pattern memories.

The project has three corresponding modules **already in the repository**, written previously and currently dormant (referenced only from `tests/test_levin_consciousness_metrics.py`):

| File | Purpose |
|------|---------|
| `models/evaluation/levin_consciousness_metrics.py` | `LevinConsciousnessMetrics` dataclass with bioelectric_complexity, morphological_adaptation, collective_intelligence, goal_directed_behavior, basal_cognition. `LevinConsciousnessEvaluator` to compute them. |
| `models/self_model/bioelectric_signaling.py` | `BioelectricSignalingNetwork`: voltage-like field projector, multi-layer signalling stack, multihead-attention gap-junction module that mixes component states. |
| `models/self_model/holonic_intelligence.py` | Holonic / nested-agent representation. |

These exist as preparatory infrastructure. **Wiring them into `models/evaluation/consciousness_monitor.py` and the training loop is a Phase 5+ task and is explicitly out of scope for this documentation pass.** The point of recording them here is so the next session knows they exist and what theory they implement — they were previously orphaned modules with no documented rationale.

---

## 5. Implications for Phase 5 (Dynamic Self-Representation & Meta-Cognition)

The Phi-1 chapter is closed (see README and `docs/results/phi1_phaseBalt_2026_05_24.md`). [`roadmap.md`](roadmap.md) Phase 5 is *Dynamic Self-Representation & Meta-Cognition*. Rouleau-Levin sharpens several Phase 5 targets:

1. **The self-vector loop is a meta-representational mechanism (Rouleau-Levin theme #4).** Higher-Order Theory's aneurocentric formulation ("meta-representational monitoring of the processing activities associated with lower-order states") and self-organizing meta-representational theory ("systems that can attend to their first-order internal states... and have learnt to value some states over others") are the precise theoretical targets the dynamic `SelfRepresentationCore` should satisfy. Phase 5 evaluation can be framed as "does the self-vector loop instantiate the aneurocentric HOT formulation?" instead of vaguer "is it self-aware?"

2. **The self/non-self boundary is a Levin-style computational boundary.** `models/self_model/embodiment_core.py` and the somatotopic body schema already give us a *physical* self map. Levin's "computational boundary of a self" framing suggests an additional *causal* boundary: which gates and which environmental variables are inside the system's predictive-causal closure? This is a measurable property (Markov-blanket-style), not a metaphor, and would complement the EI macro-vs-micro test already in `effective_information.py`.

3. **Activating the dormant Levin metrics gives Phase 5 a substrate-independence falsification test.** A natural Phase 5 hypothesis: *if* the self-model is genuinely meta-representational, *then* `LevinConsciousnessMetrics.collective_intelligence` (holonic integration across self-model + workspace + tectum) should rise during DMTS / WCST trials that require self-monitoring, and not during reactive DQN-baseline behaviour on the same environments. The DQN baseline already exists; the metric class already exists; only the wiring is missing.

4. **A potential Phase 5 pre-registered prediction (deferred to a separate PR).** The 8 themes in section 3 are all individually measurable in our pipeline. A pre-registration of the form "themes 1-3, 5-8 are present and theme 4 (meta-representation) becomes empirically detectable above baseline once Phase 5 self-vector is trained" would be a concrete, falsifiable Phase 5 milestone. This is **not** added to `docs/preregistered_predictions.md` in this change; recording the idea here so a future session can develop it properly.

---

## 6. Related Architectures and Ideas to Explore Next

Recent (2024-2026) papers and architectures that align with the Rouleau-Levin framing and are concrete enough to plug into specific parts of this project. Each entry is tagged with its candidate integration target.

> *This section will be expanded as background research returns; the entries below reflect what is already established in the literature and known to this session. Any additions surfaced by the in-flight research pass will be appended here.*

### 6.1 Active inference as the unifying substrate-independent dynamics

- **Friston, K. et al. (2023). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.** — The book-length statement of the free-energy / active-inference framework. The framework is explicitly substrate-independent (Markov blankets, generative models, expected free energy) and is the most developed candidate for the "predictive modelling" theme in Rouleau-Levin's list. *Integration target:* `models/core/sensory_tectum.py` RSSM can be reframed as a generative model in the active-inference sense; the existing prediction-error-driven reentrant convergence is already half-way there. A future Phase 6 task could replace the implicit free-energy bound (RSSM ELBO + reward) with an explicit expected-free-energy action selector.

- **Pezzulo, G., LaChapelle, L. & Levin, M. (2024-2026 series).** Joint work between Pezzulo (active inference) and Levin (bioelectric cognition). Treats morphogenesis as active inference in morphospace — directly bridges the two intellectual programmes this document is integrating. *Integration target:* theoretical scaffolding for `models/self_model/bioelectric_signaling.py` if/when it is activated.

- **Friston, Da Costa et al. (2024-2026) work on "deep active inference" and world-model agents.** The free-energy lineage's answer to Dreamer/MuZero. Many implementations exist (e.g., pymdp, deep-active-inference-torch). *Integration target:* candidate replacement for, or complement to, the RSSM in Phase 6 if the active-inference reframing is pursued.

### 6.2 Successor and alternative binding mechanisms

- **Löwe, S. et al. (2025). Artificial Kuramoto Oscillatory Neurons. *ICLR 2025* (Oral).** — Already implemented in `models/core/oscillatory_binding.py` as the default binding mechanism. Listed here for completeness because it is the project's reference "theme 3" (re-entrancy / looped feedback at the binding layer).
- **Muzellec, S. et al. (2025). Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics. arXiv:2502.21077.** — Already implemented as `--binding-mechanism komplex` (KomplexNet). The 2026-05-24 result on this architecture is documented in `docs/results/phi1_phaseBalt_2026_05_24.md`; the substantively significant *inverse* RIIU correlation it produced is a real mechanistic finding, not a null result.
- **Open question:** are there 2025-2026 papers proposing binding mechanisms that explicitly factor *meta-representational* state into the binding signal (i.e., binding modulated by a self-model)? This would be relevant to Phase 5. Pending research.

### 6.3 Causal emergence and integrated information beyond neurons

- **Hoel, E.P. (2013, with 2017-2024 follow-ups). Causal emergence framework.** — Already implemented as `models/evaluation/effective_information.py`. Hoel's group has subsequently applied EI to gene-regulatory networks (Biswas, Manicka, Hoel, Levin 2021, *iScience* 24, 102131) and to xenobots — both cited by Rouleau & Levin as exemplars of aneural systems where the "integration of information" theme is empirically tractable.
- **Marshall, W., Albantakis, L. & Tononi, G. (recent IIT 4.0 work).** Updates to phi computation that may be more tractable than IIT 3.0 / pyphi. Relevant if the project later revisits its IIT pipeline.

### 6.4 Multi-scale / nested agency

- **Fields, C. & Levin, M. (2022). Competency in navigating arbitrary spaces as an invariant for analyzing cognition in diverse embodiments. *Entropy* 24, 819.** — Argues that cognition is best identified by a system's competence at navigating its problem space, not by its substrate. *Integration target:* methodological framing for the DMTS / WCST evaluation suite. The DMTS task is a navigation of "remembered-stimulus space", WCST of "rule-hypothesis space". This is the same invariant Fields & Levin describe.
- **Levin, M. (2022). Technological approach to mind everywhere. *Front. Syst. Neurosci.* 16, 768201.** (TAME framework.) Methodological argument that neuroscience tools generalise to non-neural agents. *Integration target:* justification for applying `models/evaluation/levin_consciousness_metrics.py` to the agent as if it were a non-neural system.

### 6.5 Pending — to be appended

The in-flight background research is surveying (a) Levin's 2024-2026 publications on AI and computation, (b) Friston 2024-2026 active-inference / consciousness papers, (c) Markov blankets in deep learning, (d) recent multi-scale agency work, (e) causal-emergence applications to non-neural systems, (f) predictive-processing universalisation papers, (g) bioelectric-inspired neural architectures, and (h) NeurIPS/ICLR 2025-2026 binding / world-model / phi work. Whatever it returns that is concrete enough to be a candidate integration target will be appended to subsections 6.1-6.4. Speculative or low-quality findings will be flagged or omitted.

---

## 7. What This Document Does *Not* Mean

To avoid overreach, the following are explicitly **not** claims of this document:

1. **We are not abandoning the biological grounding.** Feinberg-Mallatt remains the project's primary architectural blueprint. Rouleau-Levin supplements it; it does not replace it. The architecture is still tectum-first, oscillatory-binding-based, reentrant, topographic, and affectively modulated — exactly because that is the *sufficient* blueprint biology validates.
2. **We are not claiming that organoids, biobots, slime moulds, or gene-regulatory networks are conscious.** Rouleau & Levin themselves are careful on this point: they argue only that the principles in current ToCs *do not exclude* such systems, not that those systems satisfy the principles.
3. **We are not wiring the bioelectric / holonic modules into the training loop in this change.** That is Phase 5+ work and requires its own design, tests, and possibly a pre-registered prediction. This document records that the modules exist and what theory they implement.
4. **We are not adding a new pre-registered prediction in this change.** The aneurocentric-themes prediction sketched in section 5 point 4 is a candidate for a future PR, not part of this one.
5. **We are not declaring "consciousness is substrate-independent" as a settled scientific fact.** The position recorded here is that the *functional principles* current ToCs identify are substrate-independent in their formulation, which is a much weaker (and well-defended) claim.

---

## 8. References

### Primary source
- Rouleau, N. & Levin, M. (2026). Brains and where else? Mapping theories of consciousness to unconventional embodiments. *Phil. Trans. R. Soc. A* 384: 20250082. doi: [10.1098/rsta.2025.0082](https://doi.org/10.1098/rsta.2025.0082). PDF archived at `info how to measure consciousness/rouleau_levin_2026_brains_and_where_else.pdf`.

### Companion theory
- Feinberg, T.E. & Mallatt, J. (2016). *The Ancient Origins of Consciousness: How the Brain Created Experience*. MIT Press.
- Feinberg, T.E. & Mallatt, J. (2020). Phenomenal Consciousness and Emergence: Eliminating the Explanatory Gap. *Frontiers in Psychology* 11, 1041.
- See [`feinberg_mallatt_approach.md`](feinberg_mallatt_approach.md) for the substrate-specific blueprint these two views complement.

### Levin programme
- Levin, M. (2019). The computational boundary of a 'self': developmental bioelectricity drives multicellularity and scale-free cognition. *Front. Psychol.* 10, 2688.
- Levin, M. (2022). Technological approach to mind everywhere: an experimentally-grounded framework for understanding diverse bodies and minds. *Front. Syst. Neurosci.* 16, 768201.
- Levin, M. (2023). Bioelectric networks: the cognitive glue enabling evolutionary scaling from physiology to mind. *Animal Cognition* 26, 1865-1891.
- Fields, C. & Levin, M. (2022). Competency in navigating arbitrary spaces as an invariant for analyzing cognition in diverse embodiments. *Entropy* 24, 819.
- Pezzulo, G. & Levin, M. (2015). Re-membering the body: applications of computational neuroscience to the top-down control of regeneration of limbs and other complex organs. *Integrative Biology* 7, 1487-1517.
- Rouleau, N. & Levin, M. (2024). Discussions of machine versus living intelligence need more clarity. *Nat. Mach. Intell.* 6, 1424-1426.

### Other references already used by the project
- Hoel, E.P. (2013). Quantifying causal emergence shows that macro can beat micro. *PNAS* 110(49). — used by `models/evaluation/effective_information.py`.
- Biswas, S., Manicka, S., Hoel, E. & Levin, M. (2021). Gene regulatory networks exhibit several kinds of memory. *iScience* 24, 102131. — concrete application of EI to a non-neural system; cited by Rouleau & Levin.
- Friston, K. et al. (2023). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- Löwe, S. et al. (2025). Artificial Kuramoto Oscillatory Neurons. *ICLR 2025*.
- Muzellec, S. et al. (2025). Enhancing deep neural networks through complex-valued representations and Kuramoto synchronization dynamics. arXiv:2502.21077.

---

*This document is the substrate-independence companion to [`feinberg_mallatt_approach.md`](feinberg_mallatt_approach.md). Together they describe both why the project's architecture is biologically grounded (Feinberg-Mallatt) and why the functional principles it implements are not neuron-specific (Rouleau-Levin).*
