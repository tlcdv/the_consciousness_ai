# Damasio's Self Hierarchy: Relevance to the Architecture

## Why this document exists

Damasio's three-level self hierarchy (proto-self, core self, autobiographical self) provides a biologically grounded developmental model for how consciousness emerges from body-brain interaction. The project already partially implements proto-self functionality (`interoceptive_to_pad` in `affective_modulator.py`) but without theoretical grounding. Evaluated at ~75% fit, with one important caveat (session 2026-06-05).

## The caveat: substrate tension

Damasio insists consciousness REQUIRES a biological substrate. A living body with genuine homeostasis. This directly conflicts with the project's Functionalist Emergentism, which holds that consciousness can emerge from any sufficiently organized substrate.

The project's position (supported by Rouleau and Levin 2026): Damasio's functional principles can be computationally re-instantiated even if the biological substrate is absent. Damasio would disagree. He considers the body constitutive of mind, not merely causal.

This tension is acknowledged, not resolved. We extract the functional architecture while remaining honest about the substrate disagreement.

## The three-level hierarchy

### Proto-Self (Level 1)

Non-conscious. Neural patterns mapping the body's internal state moment-by-moment.

- Generates "primordial feelings": wordless, raw sensations of existence.
- Brain regions: brainstem, hypothalamus, insular cortex.
- Computational requirements: continuous body-state monitoring, homeostatic variable tracking, baseline vital signs vector, real-time interoceptive mapping.
- Requires NO memory, no object recognition. Pure real-time mapping.

The proto-self is not a "self" in any reflective sense. It is a dynamic map of the body that updates continuously, providing the organism's basic orientation: alive, stable, threatened, depleted.

### Core Self (Level 2. Core Consciousness)

Emerges when the proto-self interacts with an object. The brain maps an object AND simultaneously maps the changes in the proto-self caused by that object. This creates a transient "pulse" of core consciousness, bound to the here-and-now.

```
core_consciousness = f(proto_self_state, object_representation, delta_proto_self)
```

The critical operation is `delta_proto_self`. The brain detects how the body-map shifted in response to the object. This second-order mapping is what produces the feeling of knowing.

Core consciousness does NOT require long-term memory, language, or complex reasoning. It is regenerated each processing cycle. Animals with brainstems but minimal cortex still exhibit core consciousness.

### Autobiographical Self (Level 3. Extended Consciousness)

The self extended in time. Integrates past memories and future goals with the present core self.

- Requires: episodic memory, narrative engine, persistent self-model, goal/value system, temporal projection.
- Creates coherent narrative identity across sessions.
- Uniquely human in full form, though some mammals show precursors.

### Hierarchical dependence principle

Lower levels are load-bearing. If compromised, higher levels CANNOT function.

- Brainstem damage destroys all consciousness (proto-self gone, everything collapses).
- Cortical damage may impair autobiographical self while leaving core consciousness intact.
- This is not graceful degradation. It is strict hierarchical dependency.

Architectural implication: the project must enforce this. If interoceptive signals fail, self-model functions should degrade proportionally.

## Somatic Marker Hypothesis

Physiological signals (heart rate, skin conductance, gut responses) associated with past emotional experiences bias future decisions before conscious deliberation.

Two loops:

| Loop | Mechanism | Speed |
|------|-----------|-------|
| **Body Loop** | Actual bodily changes are sensed and mapped back to brain | Slow, high-fidelity |
| **As-If Body Loop** | Brain SIMULATES bodily states without physical change | Fast, approximate |

The as-if body loop enables rapid "what if" testing. The brain generates a predicted body response to evaluate options without waiting for actual physiological changes.

This is directly relevant to the project. The agent can simulate body states without having a physical body, making the as-if loop the primary (and only) available mechanism.

## Mapping to project architecture

| Damasio Concept | Project Component | Status |
|----------------|-------------------|--------|
| Proto-self | `interoceptive_to_pad()` in `affective_modulator.py` (energy/fatigue/damage → PAD) | PARTIAL |
| Core self (delta computation) | Workspace broadcast changes internal state (PAD changes from two-stage appraisal) | IMPLICIT, not formalized |
| Autobiographical self | `EmotionalMemoryCore` + `NarrativeEngine` + `SelfRepresentationCore` | PARTIAL |
| Body Loop | Interoceptive PAD → affective modulation → bid modification | IMPLEMENTED |
| As-If Body Loop | RSSM world model predicts future states | PARTIAL |
| Homeostatic drive | Battery system in `NavigationEnv`, energy/fatigue in body schema | IMPLEMENTED |
| Feelings (conscious readout) | `PhenomenologicalMapper` | PARTIAL |

## What the theory adds that we lack

1. **Explicit core self mechanism.** A second-order mapping that detects how the proto-self CHANGES when processing each object. Currently this delta computation is implicit in the workspace cycle. It should be an explicit, measurable operation.

2. **Principled hierarchical dependence.** If interoceptive signals fail, higher self-model functions should degrade. Currently not enforced. Each layer operates independently.

3. **Emotion/feeling separation.** Emotions are automatic programs (bodily responses). Feelings are the conscious perception of those programs. Currently conflated in PAD. The PAD vector mixes automatic valence computation with what should be a separate conscious readout.

4. **Formalized as-if body loop.** RSSM performs something functionally similar (predicting future states), but it is not framed as somatic simulation. Connecting RSSM predictions to body-state forecasting would ground the world model in interoceptive terms.

## Connection to other theories in the project

**GWT.** Complementary. GWT provides architecture (workspace broadcasting). Damasio provides content (proto-self bodily states are primary inputs competing for workspace access). The two theories address different layers of the same system.

**Predictive Processing.** Highly compatible. The as-if body loop IS predictive processing for body states. Interoceptive inference maps directly to proto-self maintenance. Active inference extends this to homeostatic regulation.

**Feinberg-Mallatt.** Compatible. Both emphasize subcortical origins of consciousness (tectum in Feinberg-Mallatt, brainstem in Damasio). Feinberg-Mallatt's "primary consciousness" aligns with Damasio's core consciousness.

**IIT.** Orthogonal. IIT measures integration structure (Φ). Damasio provides biological content and developmental sequence. They address different aspects and do not conflict.

**Metzinger.** Complementary tension. Both argue the "self" is constructed. Damasio says it is a real biological construct grounded in homeostasis. Metzinger says it is a transparent self-model, phenomenally real but ontologically an illusion. Both agree the body is central to self-construction.

## Key references

- *Descartes' Error* (1994). Somatic Marker Hypothesis.
- *The Feeling of What Happens* (1999). Three-level self hierarchy.
- *Self Comes to Mind* (2010). Brain constructs conscious mind.
- *The Strange Order of Things* (2018). Homeostasis and culture.
- *Feeling and Knowing* (2021). Consciousness starts with feeling.
- "Homeostatic Feelings and the Emergence of Consciousness" (Damasio and Damasio, *J Cognitive Neuroscience* 2024).

## Priority and next steps

**Priority: MEDIUM-HIGH.** The proto-self → core self mapping should be formalized during Phase 5.

Concrete actions:
- Enforce hierarchical dependence: if interoception is disabled, self-model quality should measurably degrade.
- Structure the emotion/feeling distinction into how `PhenomenologicalMapper` relates to PAD.
- Frame RSSM body-state predictions as the as-if body loop.
- Do NOT add implementation code until the specific architectural change is designed and reviewed.
