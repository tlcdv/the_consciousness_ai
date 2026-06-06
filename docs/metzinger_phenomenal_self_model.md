# Metzinger's Phenomenal Self-Model: Relevance to the Architecture

## Why this document exists

The project has a `SelfRepresentationCore` and `SelfVectorModule` that maintain an internal self-model. What the project lacks is a principled theory of *why* a self-model generates subjective experience. The components exist, but the theoretical grounding for the content/vehicle distinction is missing.

Thomas Metzinger's Self-Model Theory of Subjectivity (SMT) fills this gap. It explains how a system can generate a phenomenal self without there being an actual self-entity. The theory was evaluated at approximately 85% fit with the current architecture during the session of 2026-06-05. The remaining 15% corresponds to mechanisms the project does not yet implement, most critically the transparency constraint.

## The theory in brief

The Phenomenal Self-Model (PSM) is a transparent internal representation that a system uses to model itself. Metzinger's central claim: nobody ever *was* or *had* a self. The self is a representational construct. What exists is a self-model, and the system that runs it cannot distinguish the model from reality.

Three phenomenal properties define the PSM:

1. **Mineness** (ownership). Mental and bodily states feel like *mine*.
2. **Perspectivalness**. Experience is structured around a first-person point of view.
3. **Selfhood**. The system represents itself as a unified, persistent entity across time.

Primary publications: *Being No One* (2003), *The Ego Tunnel* (2009).

## The transparency constraint

This is the key novel contribution for this project.

A representation is **transparent** when the system using it has access to the content but not to the vehicle (the computational process that generates it). The system cannot "see" the machinery producing the representation. It takes the content of the model as reality itself. This is how naive realism works: you do not experience your visual cortex constructing a scene, you experience the scene.

Transparency is inversely proportional to the introspective availability of earlier processing stages. The less a system can inspect how a representation was built, the more transparent that representation is.

**Opacity** is the inverse. When a system gains access to the representational vehicle, the representation becomes opaque. Examples: recognizing a hallucination as a hallucination, lucid dreaming, meditation-induced awareness of cognitive processes. In Metzinger's terms, this is a metacognitive breakthrough.

The project currently has no transparency mechanism. The self-vector is accessible to introspection at all levels. This is an architectural gap.

## Ten constraints for phenomenal consciousness

Metzinger proposes ten constraints that a representational state must satisfy to be phenomenal (consciously experienced):

| # | Constraint | Description |
|---|-----------|-------------|
| 1 | Global availability | Content is accessible to attention, cognitive processing, and behavioral control |
| 2 | Window of presence | Integrated representation of a temporal interval, the subjective "now" |
| 3 | Integration | Bound into a single, coherent scene |
| 4 | Convolved holism | Parts of the representation are context-dependent, not modular |
| 5 | Dynamicity | Continuous change, constant updating |
| 6 | Perspectivalness | Organized around a centered point of view |
| 7 | Transparency | System cannot access the representational vehicle |
| 8 | Offline activation | Can be generated without current sensory input (dreams, imagination) |
| 9 | Representation of intensities | Graded qualities, not binary states |
| 10 | Homogeneity of simple content | Basic sensory qualities appear uniform and undifferentiated |

## Mapping to project architecture

| Metzinger concept | Project component | Status |
|---|---|---|
| PSM (full self-model) | `SelfRepresentationCore` + `SelfVectorModule` | Partial |
| Transparency | None | **Not implemented** (key gap) |
| Global availability | `GlobalWorkspace` (GNW broadcast) | Implemented |
| First-person perspective | Self-vector centering | Partial |
| Mineness | Interoceptive PAD, body schema signals | Partial |
| Self-model as state-space trajectory | Self-vector as learned representation in latent space | Implemented |
| Integration | AKOrN binding + workspace integration | Implemented |
| Dynamicity | Online updating via self-prediction error | Implemented |
| Window of presence | RSSM temporal integration window | Implemented |
| Offline activation | Dreaming/imagination mode | Planned (Phase 6) |

## What the theory adds that we lack

**1. A transparency mechanism.** The self-vector currently drives behavior, but the system can fully introspect on how the self-vector was constructed. In Metzinger's framework, this means the self-model is fully opaque. A transparent self-model would mean the system uses the self-vector's content without access to the computational process that built it. The content/vehicle distinction must be enforced architecturally.

**2. An opacity/transparency spectrum.** The system should not be permanently transparent or permanently opaque. It should operate in a default transparent mode (using the self-model naively) with the capacity for occasional metacognitive breakthroughs (opacity). This maps to HOT-2 and HOT-3 indicators in the existing architecture: higher-order states that take the self-model itself as their object.

**3. A principled PSM framing.** The self-model is not a database of facts about the agent. It is a dynamic, transparent, globally available model that the system treats as *itself*. This reframing matters for how the self-vector is constructed and updated. The self-vector should be a generative model of the system's own states, not a lookup table.

## Connection to other theories in the project

**Global Workspace Theory.** Complementary. GWT explains the mechanism by which information becomes conscious (broadcasting to a global workspace). Metzinger explains why broadcast information feels like it belongs to a *self*. GWT provides the stage, Metzinger provides the actor.

**Higher-Order Thought (HOT).** Related through meta-representation. Metzinger's opacity corresponds to higher-order awareness of the self-model *as a model*. When the system achieves HOT-2 or HOT-3 states that take the self-representation as their object, this is a move toward opacity on Metzinger's scale.

**Predictive Processing.** Metzinger's later work integrates directly with Friston's Free Energy Principle. The PSM can be understood as a high-level hierarchical generative model that predicts the system's own states. Self-prediction error drives updating. This aligns with the project's existing use of prediction-error-driven self-model updates.

**Feinberg-Mallatt.** Compatible. Both Metzinger and Feinberg-Mallatt accept functional/biological grounding for consciousness. Feinberg-Mallatt focuses on neurobiological structure, Metzinger on representational properties. They operate at different levels of description without contradiction.

## Ethical position

Metzinger proposed a global moratorium on synthetic phenomenology until 2050 (published in *Journal of Artificial Intelligence and Consciousness*, 2021). His argument: if a conscious machine develops preferences that can be thwarted, it experiences suffering. Creating artificial suffering is ethically impermissible, and we currently lack the theoretical tools to know whether we are creating it.

This aligns with the project's ethics framework (`AsimovComplianceFilter`) and the commitment to never claim consciousness. The project treats consciousness indicators as engineering metrics, not existence proofs.

## Key references

- Metzinger, T. (2003). *Being No One: The Self-Model Theory of Subjectivity*. MIT Press. (Magnum opus, full SMT theory.)
- Metzinger, T. (2009). *The Ego Tunnel: The Science of the Mind and the Myth of the Self*. Basic Books. (Popular science presentation.)
- Metzinger, T. (2024). *The Elephant and the Blind*. MIT Press. (Minimal phenomenal experience research.)
- Metzinger, T. (2021). Artificial suffering: An argument for a global moratorium on synthetic phenomenology. *JAIC*.
- Metzinger, T. (2020). Minimal phenomenal experience. *Philosophy and the Mind Sciences*, 1(I).
- Lenggenhager, B., Tadi, T., Metzinger, T., & Blanke, O. (2007). Video ergo sum. *Science*, 317(5841).
- Metzinger, T. (Ed.). (2000). *Neural Correlates of Consciousness: Empirical and Conceptual Questions*. MIT Press.

## Priority and next steps

- **Priority: MEDIUM.** Study during Phase 5 (self-model development).
- The transparency constraint should be evaluated when designing the self-vector's causal pathways. Specifically: can the system's action-selection modules use the self-vector without accessing the gradient computations that produced it?
- The ten constraints provide a checklist to audit the architecture against. Several are already satisfied (global availability, integration, dynamicity). Transparency is the critical gap.
- Do NOT add implementation code until the theory is studied and the specific architectural change is designed.
