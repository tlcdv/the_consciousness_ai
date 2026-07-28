# Aligned external knowledge resources

*Compiled 2026-06-21. A curated reading map of external work that complements this
project's core and is useful for its live frontiers. Every entry sits on the existing
spine (tectum-first neuroevolution, affective homeostasis, oscillatory binding,
generative world models, IIT/GNW measurement). Work from a different approach was
excluded on purpose (see the last section).*

## How to read this

- Ordered by **current build priority**, not by theoretical tidiness. The two live
  frontiers come first: the generative-world-model / perception collapse
  ([collapse_locus_2026_06_16.md](results/collapse_locus_2026_06_16.md)) and the DMTS
  learning / credit-assignment wall
  ([dmts_match_head_2026_06_15.md](results/dmts_match_head_2026_06_15.md)).
- Each entry has a **Tier**, a **why aligned**, and a **plugs in** (the module, doc, or
  roadmap phase it touches).
- Tiers are honest, not flattering. **Tier 1**: load-bearing for a live frontier.
  **Tier 2**: fills a real grounding gap the project is thin on. **Tier 3**: solid
  complement or framing, lower marginal value right now.
- A signature on this map is an input to study, never a claim that it works. This
  project has a record (Phi-1, KomplexNet, match-head) of well-motivated additions that
  did not move a metric. Each item here is held to the same FAILED-first bar.
- Companion docs that already cover the spine:
  [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md),
  [rouleau_levin_substrate_independence.md](rouleau_levin_substrate_independence.md),
  [metzinger_phenomenal_self_model.md](metzinger_phenomenal_self_model.md),
  [damasio_self_hierarchy.md](damasio_self_hierarchy.md),
  [watanabe_generative_model_approach.md](watanabe_generative_model_approach.md),
  [active_inference_unification.md](active_inference_unification.md),
  [consciousness_indicators_butlin.md](consciousness_indicators_butlin.md).

---

## 1. Generative world models and the perception frontier (Tier 1)

The project's own diagnosis: the reward-trained RSSM destroys stimulus identity at the
`obs_map -> z_state` step. These are the literature for the named fix (a real generative
world model that keeps identity). Dedicated doc:
[generative_world_models_perception.md](generative_world_models_perception.md).

### 1.1 DINO-WM. Zhou, Pan, LeCun & Pinto (2024), "World Models on Pre-trained Visual Features enable Zero-shot Planning"
arXiv:2411.04983 (ICML 2025). Project page: https://dino-wm.github.io/

- **Why aligned (highest single value):** Builds a world model by predicting **frozen
  DINOv2 patch features**, with no pixel reconstruction, which keeps spatial structure
  and supports planning. The project already encodes the tectum with DINOv2, so this is
  a direct, published recipe for the exact problem the collapse probe localized. It is
  an alternative or complement to the ELBO-reconstruction path in
  [active_inference_unification.md](active_inference_unification.md).
- **Plugs in:** `models/core/sensory_tectum.py` (RSSM), active-inference stage 1, the
  perception-decodability probes in `scripts/analysis/`.
- **Honest caveat:** it is a planning method that needs offline trajectories. The
  integration and compute cost on a batch-1, latency-bound setup is real and unproven
  here. Verified against the primary source this session.

### 1.2 Object-centric / slot-based world models for RL (2024-2025)
Representative: OC-STORM, Zhang, Jelley, McInroe, Storkey & Wang (2025), arXiv:2501.16443;
plus the slot-stability / interaction-aware line. Foundation: Locatello et al. (2020),
"Object-Centric Learning with Slot Attention," arXiv:2006.15055.

- **Why aligned:** World models whose latents are **per-object slots that hold identity
  under occlusion and interaction**. That is the property the collapse probe found
  missing, and it is the same property DMTS feature-binding and working memory need.
  Slots are also a content-level cousin of the project's binding pillar (a bound object
  is a slot).
- **Plugs in:** the RSSM redesign, DMTS, the binding pillar.
- **Honest caveat:** heavier to integrate than DINO-WM. The exact arXiv-id to title
  mapping in this fast-moving cluster should be re-checked at build time (a search
  snippet mislabeled one ID during compilation).

### 1.3 Predictive-coding and active-inference implementation toolkit
Ororbia, Mali, Kohan, Millidge & Salvatori (2024), "A Review of Neuroscience-Inspired
Machine Learning," arXiv:2403.18929. Salvatori et al. (2024), "Predictive Coding Networks
and Inference Learning: Tutorial and Survey," arXiv:2407.04117.

- **Why aligned:** The engineering references the Phase 6 active-inference plan is
  missing. [active_inference_unification.md](active_inference_unification.md) cites
  Friston and Rao theory but no implementation survey. These give concrete PC training
  recipes and a map of backprop-free options.
- **Plugs in:** Phase 6, the EFE / variational-free-energy front-end.

---

## 2. Learning, credit assignment and working memory (Tier 1, the DMTS wall)

The project's other live blocker: the agent cannot learn the DMTS match even when the
information is present and offline-decodable, and the conv match-head cannot express the
non-local comparison. These are biologically grounded answers (basal ganglia, PFC,
episodic memory), not generic RL engineering.

### 2.1 Zambaldi et al. (2019), "Deep Reinforcement Learning with Relational Inductive Biases"
ICLR 2019; "Relational Deep Reinforcement Learning," arXiv:1806.01830.

- **Why aligned (most targeted to the match-head failure):** Uses self-attention to
  reason over entities and their **relations**, which is exactly the "is sample ==
  choice" non-local comparison the conv head could not represent
  ([dmts_match_head_2026_06_15.md](results/dmts_match_head_2026_06_15.md) found a global
  PCA+MLP could decode the match at ~0.74-0.85 but the local conv could not). A
  relational / attention read-out is the principled replacement.
- **Plugs in:** the match read-out, the policy input over `[obs ; mem]`, the capsule /
  workspace relational structure.

### 2.2 Hung et al. (2019), "Optimizing agent behavior over long time scales by transporting value"
Nature Communications 10:5223; arXiv:1810.06721. Temporal Value Transport (TVT).

- **Why aligned:** Uses an **episodic memory** to credit distant past actions for later
  reward, built for delayed-reward memory tasks. DMTS is precisely that shape (reward
  arrives after a 15-40 step blank delay). The project already has an episodic
  emotional memory, so the mechanism has a home.
- **Plugs in:** `models/memory/`, the DMTS reward path, credit assignment across the
  delay.

### 2.3 Raposo et al. (2021), "Synthetic Returns for Long-Term Credit Assignment"
arXiv:2102.12425.

- **Why aligned:** Learns an explicit association between past events and future reward
  as a credit-assignment proxy, improving on TVT for states that only partially predict
  a distant reward. The lighter-weight option for the same DMTS credit-assignment wall.
- **Plugs in:** the DMTS reward path, an auxiliary credit-assignment head.

### 2.4 Wang et al. (2018), "Prefrontal cortex as a meta-reinforcement learning system"
Nature Neuroscience 21:860-868.

- **Why aligned:** Dopamine trains the PFC to act as its own learning system, that is,
  **meta-RL / learning to learn**. This is the biological theory behind the project's
  PFC + basal-ganglia Go/No-Go model and is directly on point for WCST (rule switching,
  meta-cognition) and for why a single slow learner stalls on these tasks.
- **Plugs in:** `models/self_model/` action selection, WCST, the dopamine credit model.

### 2.5 Botvinick, Wang, Dabney, Miller & Kurth-Nelson (2020), "Deep Reinforcement Learning and Its Neuroscientific Implications"
Neuron 107:603-616; arXiv:2007.03750.

- **Why aligned:** The bridge between deep RL and neuroscience, with representation
  learning as a central theme. The project's competence diagnosis concluded the
  bottleneck is the **representation** the policy reads, which is exactly this paper's
  subject. Good orientation for the whole learning-frontier effort.
- **Plugs in:** framing for `docs/results/agent_competence_*`, the representation work.

### 2.6 Biologically grounded working memory for DMS (attractor models)
Representative: recurrent attractor models of delayed-match-to-sample (storage +
comparator units, fixed-point attractors per sample); "A working memory model based on
recurrent neural networks using reinforcement learning" (2024, PMC11564475).

- **Why aligned:** Models the **persistent-activity / attractor** substrate of working
  memory that the project leans on GNW reverberation to provide. Relevant to whether the
  DMTS sample should be held by an attractor or by an explicit gated memory slot.
- **Plugs in:** GNW reverberation, the DMTS working-memory mechanism, the self-model
  working-memory latch.

---

## 3. Affective consciousness, interoception and homeostatic grounding (Tier 2)

The project's stated engine is emergence from emotional homeostasis, but its affective
citations are sparse (Mehrabian PAD, Keramati-Gutkin). These supply the neuroscience the
engine actually models and connect it to the active-inference plan.

### 3.1 Solms (2021), *The Hidden Spring*; Solms & Friston (2018), "How and why consciousness arises"
Norton (book). *J. Consciousness Studies* 25(5-6):202-238 (paper).

- **Why aligned (load-bearing for coherence):** Argues consciousness is affective,
  brainstem-generated, a form of homeostasis, and formalized as **free-energy
  minimization** (decreases and increases in expected uncertainty felt as pleasure and
  unpleasure). This is the project's engine stated as theory, and it is the missing link
  between the Affective Core and the Phase 6 active-inference plan. Both citations
  verified this session.
- **Plugs in:** README Core Principle, `models/emotion/`,
  [active_inference_unification.md](active_inference_unification.md). Dedicated doc:
  [affective_consciousness_solms_panksepp.md](affective_consciousness_solms_panksepp.md).

### 3.2 Panksepp (1998), *Affective Neuroscience*; Panksepp & Biven (2012), *The Archaeology of Mind*
Oxford University Press; Norton.

- **Why aligned:** The subcortical-circuit grounding (PAG, hypothalamus, and the seven
  primary-process systems) for the limbic affective track Feinberg-Mallatt describe and
  the PAD modulator implements. The **SEEKING system** is the biological referent for
  the project's curiosity / RND exploration drive, and FEAR / RAGE for the THREAT
  modules.
- **Plugs in:** `models/emotion/affective_modulator.py` (APPROACH / THREAT sets), reward
  shaping, RND curiosity.

### 3.3 Seth (2021), *Being You*; Seth & Critchley (2013); Seth & Tsakiris (2018), "Being a Beast Machine"
Faber/Dutton (book). "Extending predictive processing to the body: emotion as
interoceptive inference." "Being a beast machine: the somatic basis of selfhood," *TiCS*
22(11):969-981.

- **Why aligned:** Emotion and selfhood as **interoceptive predictive inference**, the
  predictive-processing form of the project's interoceptive PAD loop and self-model.
  "Beast machine" grounds selfhood in bodily regulation, which is the embodiment-affect
  loop stated as theory.
- **Plugs in:** `models/self_model/`, interoceptive PAD generation, Phase 5 self-vector,
  Phase 6 active inference.

---

## 4. Evolutionary and biological origins of consciousness (Tier 2, tectum-first)

### 4.1 Merker (2007), "Consciousness without a cerebral cortex: A challenge for neuroscience and medicine"
*Behavioral and Brain Sciences* 30:63-81 (PMID 17475053).

- **Why aligned:** The strongest **independent** case that consciousness is constituted
  in the midbrain / upper brainstem, not the cortex, from decortication and
  hydranencephaly evidence. The architecture rests this cornerstone on Feinberg-Mallatt
  alone today. A second independent line strengthens the foundation at low cost.
- **Plugs in:** [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md) section 2.1,
  `models/core/sensory_tectum.py`. Dedicated doc:
  [merker_subcortical_consciousness.md](merker_subcortical_consciousness.md).

### 4.2 Ginsburg & Jablonka (2019), *The Evolution of the Sensitive Soul*; Birch, Ginsburg & Jablonka (2020), UAL primer
MIT Press; "Unlimited Associative Learning and the origins of consciousness," *Biology &
Philosophy* 35:56.

- **Why aligned:** The other major evolutionary-origins program, compatible with
  Feinberg-Mallatt (Cambrian, vertebrate-first), with a **behavioral marker**: Unlimited
  Associative Learning (UAL). UAL is a candidate operational test alongside DMTS / WCST.
- **Plugs in:** [preregistered_predictions.md](preregistered_predictions.md), the
  consciousness-demanding environments, the indicator rubric.

### 4.3 Feinberg & Mallatt (2018), *Consciousness Demystified*
MIT Press.

- **Why aligned:** The shorter, more recent synthesis from the project's anchor authors.
  Worth citing so the project tracks their latest statement, not only the 2016 volume.
- **Plugs in:** [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md) references.

---

## 5. Global workspace, self-monitoring and higher-order processing in deep learning (Tier 2 / 3)

### 5.1 Goyal, Bengio et al. (2021), "Coordination Among Neural Modules Through a Shared Global Workspace"
arXiv:2103.01197 (ICLR 2022).

- **Why aligned:** The deep-learning instantiation of GNW, modules competing through a
  shared bottleneck with limited broadcast slots. A reference design for what
  `models/core/global_workspace.py` builds by hand.
- **Plugs in:** `models/core/global_workspace.py`, the semantic pathway, the reentrant
  loop.

### 5.2 Dehaene, Lau & Kouider (2017), "What is consciousness, and could machines have it?"
*Science* 358:486-492.

- **Why aligned:** Defines **C1 (global availability)** and **C2 (self-monitoring /
  metacognition)** as the two computational targets for machine consciousness, which the
  project implements (GNW broadcast as C1, self-vector plus confidence gate as C2). The
  cleanest statement of what the architecture is trying to build.
- **Plugs in:** README Scientific Approach, Phase 5 self-model, the indicator rubric.

### 5.3 Aru, Suzuki & Larkum (2020), "Cellular Mechanisms of Conscious Processing"; Dendritic Integration Theory
*Trends in Cognitive Sciences* 24(10):814-825; Aru et al., Dendritic Integration Theory
(2020/2023).

- **Why aligned:** Apical-dendrite "context vs content" amplification is the cellular
  form of the project's reentrant top-down (context) vs bottom-up (sensory
  prediction-error) loop, and of the ConsciousnessGate's coupling of state and content.
- **Plugs in:** `models/core/reentrant_processor.py`,
  `models/core/consciousness_gating.py`.
- **Honest caveat:** translating apical amplification to the gate is a design analogy,
  not a recipe.

### 5.4 Tani (2016), *Exploring Robotic Minds*; Tani & White (2022), "Cognitive neurorobotics and self in the shared world"
Oxford University Press; *Adaptive Behavior*.

- **Why aligned:** Emergence of minimal and narrative self via prediction-error
  minimization in embodied agents, the robotic analog of the project's self-vector plus
  active-inference plan.
- **Plugs in:** Phase 5 self-model, Phase 6 active inference.

---

## 6. Binding and part-whole composition (Tier 3, deprioritized while Phi-1 is closed)

The Phi-1 in-training binding-integration prediction is exhausted across 9 runs. Binding
work is lower priority than perception now, but these are the on-core references if it
resumes.

### 6.1 Greff, van Steenkiste & Schmidhuber (2020), "On the Binding Problem in Artificial Neural Networks"
arXiv:2012.05208.

- **Why aligned:** The canonical framing of the exact problem the binding pillar exists
  to solve (segregation, representation, composition of object-like entities). Gives a
  principled vocabulary for the AKOrN / KomplexNet / capsule work and the DMTS
  feature-binding failure.
- **Plugs in:** [biological_neural_architecture_research.md](biological_neural_architecture_research.md),
  the binding-mechanism docs.

### 6.2 Lowe et al., the synchrony-binding line (same author as AKOrN)
"Complex-Valued Autoencoders for Object Discovery" (2022, TMLR); "Rotating Features for
Object Discovery" (NeurIPS 2023, arXiv:2306.00600).

- **Why aligned:** Binding by phase interference in complex / rotating features, the
  precursor and a scalable alternative to KomplexNet's `cos(theta)` weave. A
  content-level binding mechanism to compare against AKOrN and KomplexNet.
- **Plugs in:** the `--binding-mechanism` family, `models/core/oscillatory_binding.py`.

### 6.3 Alamia, Muzellec, Serre & VanRullen (2025), "GASPnet: Global Agreement to Synchronize Phases"
arXiv:2507.16674.

- **Why aligned:** Kuramoto phase synchronization fused with Transformer attention
  (key-query couplings drive the oscillators). Already named as the project's Phase B-alt
  candidate, the paper is now published, so it is the concrete next binding mechanism to
  A/B against KomplexNet. Authors and mechanism verified this session.
- **Plugs in:** binding-mechanism selection, the successor to the closed Phi-1 line.

### 6.4 Hinton (2021/2023), "How to represent part-whole hierarchies in a neural network" (GLOM)
arXiv:2102.12627; *Neural Computation* 35(3):413-452.

- **Why aligned:** "Islands of agreement" combine binding-by-agreement and part-whole
  composition in one idea, a conceptual sibling of the capsule hierarchy plus reentrant
  settling.
- **Plugs in:** `models/core/capsule_composition.py`, the reentrant processor.
- **Honest caveat:** GLOM is an "imaginary system" Hinton never fully implemented. It is
  inspiration, not a drop-in.

---

## 7. Measurement, IIT updates and AI-consciousness assessment (Tier 3, framing)

### 7.1 Cogitate Consortium (2025), "Adversarial testing of global neuronal workspace and integrated information theories of consciousness"
*Nature* 642:133-142 (s41586-025-08888-1).

- **Why aligned:** The large pre-registered IIT-vs-GNW experiment (n=256, fMRI/MEG/iEEG).
  Both theories took hits. A mature model for how to frame the project's own
  pre-registered, FAILED-first Phi results, and direct context for the combined IIT+GNW
  measurement strategy.
- **Plugs in:** [preregistered_predictions.md](preregistered_predictions.md),
  [iit_implementation_roadmap.md](iit_implementation_roadmap.md).

### 7.2 Albantakis, Tononi et al. (2023), "Integrated Information Theory (IIT) 4.0"
*PLoS Computational Biology* 19(10):e1011465; arXiv:2212.14787.

- **Why aligned:** The current IIT formalism (intrinsic-difference measure, refined
  postulates). The project computes phi via pyphi and gate states, so IIT 4.0 is the
  up-to-date target for that pathway.
- **Plugs in:** `models/evaluation/iit_phi.py`,
  [iit_implementation_roadmap.md](iit_implementation_roadmap.md).
- **Honest caveat:** lower priority while the Phi-1 measurement line is closed.

### 7.3 Pennartz et al. (2019), indicators of consciousness; Pennartz (2022), neurorepresentationalism
"Indicators and Criteria of Consciousness in Animals and Intelligent Machines: An
Inside-Out Approach," *Front. Syst. Neurosci.* 13:25. "What is neurorepresentationalism?"
*Behavioural Brain Research* 432:113969.

- **Why aligned:** A second, multimodal-representation-based indicator rubric for AI
  consciousness to triangulate against Butlin et al. Its five hallmarks (multimodal
  richness, situatedness, unity, dynamics, intentionality) map onto the trimodal tectum
  plus GNW unity.
- **Plugs in:** [consciousness_indicators_butlin.md](consciousness_indicators_butlin.md)
  (add a Pennartz cross-check).

---

## 8. Thalamic gating and perturbational measurement (added 2026-07-28)

*Three papers on thalamic contributions to human consciousness, reviewed and audited
against this architecture in [thalamic_gating_evidence.md](thalamic_gating_evidence.md).
Read that doc first: it carries the alignment verdict, the seven file-level gaps, and a
binding constraint on translating any frequency band into this system.*

### 8.1 Fang, Dang, Ping, Wang, Zhao, Zhao, Li & Zhang (2024), "Human intralaminar and medial thalamic nuclei transiently gate conscious perception through the thalamocortical loop" (Tier 1)
bioRxiv 2024.04.02.587714. **Preprint, not peer reviewed.**

- **Why aligned (strongest external support for the mission thesis):** Simultaneous depth
  recordings from 197 thalamic and 213 prefrontal sites during a near threshold Gabor task
  with the motor response matched across the awareness contrast. Intralaminar (CM, Pf) and
  medial (MDm) nuclei show earlier and stronger consciousness related activity than the
  ventral nuclei; low frequency phase locking rises within thalamus first, then thalamus to
  cortex, then within cortex; directed phase flow runs thalamus to cortex. The authors
  conclude the gate for conscious *contents* is subcortical and defend it on the same
  evolutionary grounds as [feinberg_mallatt_approach.md](feinberg_mallatt_approach.md).
  This is the first external, intracranial support the project's subcortical bet has.
- **Plugs in:** `models/core/global_workspace.py` (the gate's ownership question), the
  near threshold protocol for `simulations/environments/dmts_env.py`, the divergence onset
  probe.
- **Honest caveat:** five patients, preprint, and its frequency band results do not
  translate to this system (no millisecond clock). It licenses a measurement programme, not
  a thalamus module.

### 8.2 Koch, Massimini, Boly & Tononi (2016), "Neural correlates of consciousness: progress and problems" (Tier 1 for measurement)
*Nature Reviews Neuroscience* 17:307-321, doi:10.1038/nrn.2016.22. Peer reviewed.

- **Why aligned:** Reports gamma synchrony and the P3b as failed markers and PCI as the one
  measure that separates conscious from unconscious states at the single participant level
  across sleep, three anaesthetics and disorders of consciousness. Its stated reason PCI
  works where spontaneous measures fail (it evaluates deterministic responses to
  perturbation, so it is insensitive to random or locally generated patterns) names this
  project's own documented failure mode for EI, CE 2.0 and phi. It also independently
  corroborates the closed Phi-1 chapter and the invariant `sync_R`.
- **Plugs in:** `models/evaluation/perturbational_complexity.py`,
  `models/evaluation/consciousness_metrics.py` (which currently holds a PCI placeholder
  returning random numbers), [consciousness_indicators_butlin.md](consciousness_indicators_butlin.md).
- **Primary source for the measure:** Casali et al. (2013), *Sci. Transl. Med.* 5:198ra105.

### 8.3 Chowdhury, Kaufmann, Schreiner, Koeglsperger, Mehrkens, Remi, Vollmar & Staudigl (2025), "Thalamic oscillations distinguish natural states of consciousness in humans" (Tier 3)
bioRxiv 2025.01.28.635248. **Preprint, not peer reviewed.**

- **Why aligned:** A fast thalamic oscillation present in wakefulness and REM and absent in
  NREM, whose bursts track rapid eye movements and whose detection probability rises with
  proximity to the Central Thalamus. Evidence that one thalamic circuit switches oscillatory
  regime with neuromodulatory tone, and that the regime discriminates states of
  consciousness.
- **Plugs in:** nothing on the current build path, deliberately. This system has no brain
  states, no neuromodulatory tone that changes global dynamics, and no cheap experiment that
  could kill a sleep state machine before it was built. Tier 3 and recorded as evidence
  only; see section 4 of [thalamic_gating_evidence.md](thalamic_gating_evidence.md).

---

## Deliberately excluded (related but off-core)

To keep the map aligned with the project's biological-emergentist thesis, these were left
out on purpose:

- **LLM-as-consciousness probes** and "is GPT conscious" introspection benchmarks. A
  language-model-centric substrate philosophy, not the embodied biological emergence
  thesis. Chalmers (2023), "Could a Large Language Model Be Conscious?" is worth one
  contextual read but is not an architecture input.
- **Pure panpsychism or philosophy of mind** with no computational bridge.
- **Symbolic / GOFAI cognitive-architecture consciousness** (SOAR, ACT-R style). A
  different approach.
- **Generic agentic-LLM "world model" work** not grounded in predictive coding or a
  generative model.

---

## Verification status (honesty note)

Primary sources fetched and confirmed during compilation: DINO-WM (arXiv:2411.04983,
authors and no-reconstruction claim), GASPnet (arXiv:2507.16674, authors and
attention-plus-Kuramoto mechanism), the object-centric WM cluster claim (OC-STORM,
arXiv:2501.16443), and Solms & Friston (2018, *J. Consciousness Studies* 25(5-6):202-238).
The remaining citations were sourced from search with consistent venue and identifier
data but were not each individually fetched. Before any of these is cited inside code or
a results doc, fetch the primary link and confirm the exact venue, year, and identifier.
The one known issue to fix at that pass: the object-centric cluster had an
arXiv-id-to-title mismatch in a search snippet, so confirm each paper's title against its
id.
