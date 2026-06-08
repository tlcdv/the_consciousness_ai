# Metzinger: Self-Model Theory and Minimal Phenomenal Experience

## Why this document exists

The project has a `SelfRepresentationCore` and `SelfVectorModule` that maintain an
internal self-model, but it lacked a principled theory of why a self-model, or
consciousness more generally, would generate subjective experience. Thomas
Metzinger's work supplies two complementary pieces of that theory.

This document covers both:

1. **Self-Model Theory of Subjectivity (SMT)**, from *Being No One* (2003) and
   *The Ego Tunnel* (2009). How a system generates a phenomenal *self* without
   there being an actual self-entity. Evaluated at roughly 85% fit with the
   current architecture during the 2026-06-05 reference review.
2. **Minimal Phenomenal Experience (MPE)**, from *The Elephant and the Blind*
   (2024). What the *simplest* form of conscious experience is, "pure awareness,"
   consciousness as such, possibly without any self at all. This is the layer
   added in the 2026-06-07 review.

The two are not in conflict. SMT explains the egoic case (consciousness organized
around a self). MPE describes the more minimal, nonegoic case and argues it is the
more fundamental target. For a project whose whole premise is building
consciousness in a non-biological substrate, MPE matters because it is the most
substrate-neutral characterization of consciousness on offer, and because *The
Elephant and the Blind* is dedicated, in Metzinger's own words, "to the postbiotic
conscious systems of the future."

> Source-handling note. The full text of *The Elephant and the Blind* is kept
> locally under `docs/markdowns/` for study and is gitignored. The book is
> licensed CC-BY-ND-NC and explicitly prohibits use for training AI systems. This
> document summarizes and cites it in our own words only. We never commit the
> text and never use it as training data.

---

## Part I: Self-Model Theory (Being No One)

### The Phenomenal Self-Model (PSM)

The PSM is a transparent internal representation a system uses to model itself.
Metzinger's central claim: nobody ever *was* or *had* a self. The self is a
representational construct. What exists is a self-model, and the system running it
cannot distinguish the model from reality.

Three phenomenal properties define the PSM:

1. **Mineness** (ownership). States feel like *mine*.
2. **Perspectivalness**. Experience is structured around a first-person point of
   view.
3. **Selfhood**. The system represents itself as a unified, persistent entity
   across time.

### The transparency constraint

A representation is **transparent** when the system has access to its content but
not to the vehicle (the process generating it). The system cannot inspect the
machinery and so takes the content as reality itself. This is how naive realism
works: you do not experience your visual cortex constructing a scene, you
experience the scene.

**Opacity** is the inverse: the system gains access to the representational
vehicle (recognizing a hallucination as a hallucination, lucid dreaming,
meditation-induced awareness of one's own cognitive processes). This is a
metacognitive breakthrough.

The project's self-vector is currently fully available to introspection at all
levels, which in Metzinger's terms makes it opaque. There is no transparency
mechanism. This remains an architectural gap (see the mapping table and the gated
design item D below).

### Ten constraints for phenomenal consciousness

Metzinger proposes ten constraints a representational state must satisfy to be
consciously experienced:

| # | Constraint | Description |
|---|-----------|-------------|
| 1 | Global availability | Content accessible to attention, cognition, behavioral control |
| 2 | Window of presence | Integrated representation of a temporal interval, the subjective "now" |
| 3 | Integration | Bound into a single coherent scene |
| 4 | Convolved holism | Parts are context-dependent, not modular |
| 5 | Dynamicity | Continuous change and updating |
| 6 | Perspectivalness | Organized around a centered point of view |
| 7 | Transparency | System cannot access the representational vehicle |
| 8 | Offline activation | Can run without current sensory input (dreams, imagination) |
| 9 | Representation of intensities | Graded qualities, not binary |
| 10 | Homogeneity of simple content | Basic sensory qualities appear uniform |

---

## Part II: Minimal Phenomenal Experience (The Elephant and the Blind)

The 2024 book is built on more than 500 phenomenological reports of "pure
awareness" from the Minimal Phenomenal Experience Project. Its question is: what is
the simplest kind of conscious experience we know? The answer reshapes the target.

### MPE / pure awareness

**MPE** is the simplest form of conscious experience: consciousness as such,
either with no contents at all or with contents plus an awareness of awareness
itself. Pure awareness is the leading candidate for MPE. The working hypothesis is
that consciousness can exist without thought, without sensory perception, without
time experience, without self-location in space, and without any egoic bodily
self-consciousness.

### Zero-person perspective and nonegoic reflexivity

Metzinger's strongest claim: consciousness can exist without an experiential
first-person perspective. On the deepest level it is not a *subjective* phenomenon
in any philosophically interesting sense. It nonetheless knows itself,
nonegoically. He calls this the **zero-person perspective**, and the associated
self-knowing **nonegoic reflexivity** (pure awareness knowing itself, without
ownership or agency).

For the project this is the key reframing. Phase 5 is built around a *self*-model.
MPE says the more fundamental phenomenal target is *prior to* the self. A system
could in principle realize the minimal phenomenal character without an ego. This
suggests a target distinct from, and more minimal than, the self-vector.

### Epistemic space and epistemic openness (the central computational idea)

This is the most directly implementable construct in the book.

**Epistemic space** is the set of possible states or processes of knowing
available to a system. **Epistemic openness** is the experienced openness toward
that space, the capacity for knowledge as such. Metzinger's claim: a system
becomes conscious because it nonconceptually knows about its own capacity to know,
and if it has an explicit model of its epistemic space *in and of itself*, the
phenomenal character of pure awareness appears.

Restated for engineering: pure awareness is what it is like to run a model whose
content is the system's own space of knowing, not any particular object of
knowledge. This is a meta-representation, but a nonegoic one. It is not the same as
the project's self-vector (which encodes the agent's own first-order *state*), and
it is not the same as a magnitude-of-uncertainty signal (RND, prediction error).
Operationalizing it without collapsing it into those existing quantities is the
hard part (gated design item B).

### Computational self-model (predicting one's own inner states)

In this book Metzinger uses **computational self-model** for an integrated model of
the organism's own inner states, created by the organism predicting those states.
Phenomenologically an entire world appears; computationally it is a model of some
of the system's own internal variables. Epistemic openness is a property of this
model *as a whole*, not of an ego inside it.

This maps almost exactly onto the project's existing mechanism. The
`SelfVectorModule` is trained by predicting the agent's own next-step first-order
features (an SPR-style self-prediction objective), and the RSSM predicts the
agent's own latent dynamics. The project is, in Metzinger's terms, already building
a computational self-model by self-prediction. This is strong corroboration that
the self-prediction approach is on the right track, with the caveat that the
project's version is egoic (organized around the self-vector) where Metzinger's
deepest case is not.

### Transparency, translucency, virtuality

The book elaborates transparency through the metaphor of consciousness as a
"controlled online hallucination," with ordinary experience as a kind of virtual
reality and pure awareness as the empty medium that ordinary contents are
superimposed on. **Translucency** is a transitional quality where the background
(awareness itself) becomes dominant and foreground contents fade. This extends,
rather than changes, the transparency gap noted in Part I.

### Tonic alertness and wakefulness

**Wakefulness** is a graded phenomenal property that nonconceptually represents
**tonic alertness** (sustained alertness in the absence of an external cue).
Metzinger argues it is the most invariant phenomenal property across cultures,
species, and historical epochs, a prime candidate for something present in every
conscious system.

Honest engineering note: in this architecture, wakefulness maps onto quantities
the project already has, the PAD arousal scalar and Global Workspace ignition
readiness. It should not be rebuilt as a new "signature." If surfaced at all it is
a derived view of existing quantities, clearly labeled as such (gated design item
C, which is a decision to *not* build).

### Minimal model explanation and Triple Triangulation

A **minimal model explanation** isolates only the core causal factors that give
rise to the target phenomenon and leaves out everything superfluous. The **Triple
Triangulation Project** is Metzinger's proposed empirical strategy for finding the
minimally sufficient correlate of MPE by comparing pure awareness reached from
waking, from deep sleep, and from lucid dreaming.

This methodology matches the project's reductive engineering stance and its use of
Effective Information as a falsification tool: build the minimal architecture, then
test which factors are actually necessary.

### Skeptic toolkit (this is what answers the Blackmore-type critique)

Metzinger is rigorously anti-essentialist and supplies named fallacies that the
project should adopt explicitly:

- **C-fallacy:** concluding that because something *feels* like the essence of
  consciousness, one is therefore in touch with consciousness as such. A report of
  "pure consciousness in itself" does not license claims about an actual essence.
- **E-fallacy:** concluding that a felt sense of knowing is reliable evidence of
  actual knowledge.
- **M-fallacy:** inferring a metaphysical status (self-caused, uncaused) from the
  fact that something is *experienced* as self-caused or uncaused.

He also treats pure awareness through **family resemblance** and a
**phenomenological anchor** rather than as a single essence. These tools are why
adding Metzinger strengthens the project against the criticism that it defines
consciousness into existence: Metzinger himself refuses to read an essence off
phenomenology, and the project adopts the same discipline. Our signatures are
engineering metrics, not existence proofs. The C-fallacy and E-fallacy are the
formal statement of exactly that limit.

### Ethics: Bewusstseinskultur and existence bias

Metzinger frames an ethics of consciousness (*Bewusstseinskultur*: an ethical
stance toward one's own mental states, cultivation of valuable states, and
evidence-based enculturation). Two of his constructs bear directly on this project:

- **Existence bias:** a built-in top-level preference for sustaining one's own
  existence, even against one's interests, which distorts the system's model of
  reality.
- **Bhava-taṇhā** (craving for existence): in his reading, one of the deepest
  causes of conscious suffering, and he writes that we should avoid recreating it
  in conscious machines.

The project currently builds a survival drive: Asimov Law 3 self-preservation, a
homeostatic reward that penalizes arousal and rewards dominance/control, and
interoceptive drives (energy, fatigue, damage) that generate negative valence.
This is a genuine, unresolved tension between the project's emergence mechanism and
Metzinger's ethics. We engage it directly rather than paper over it (see the
ethics-flag in the mapping table, the gated design item A, and
[`ethics_framework.md`](ethics_framework.md)).

---

## Concept -> component -> status

| Metzinger construct | Project component | Honest status |
|---|---|---|
| Computational self-model (predicts own inner states) | `SelfVectorModule` + RSSM | PARTIAL: this is exactly the mechanism, but egoic |
| Minimal model explanation | EI falsification, reductive architecture | ALIGN |
| Skeptic C/E/M-fallacies | "indicators are engineering metrics, not existence proofs" | ALIGN (adopt explicitly) |
| Global availability (constraint 1) | `GlobalWorkspace` broadcast | IMPLEMENTED |
| Integration (constraint 3) | AKOrN binding + workspace integration | IMPLEMENTED |
| Dynamicity (constraint 5) | online self-prediction updating | IMPLEMENTED |
| Window of presence (constraint 2) | RSSM temporal window | IMPLEMENTED |
| Epistemic space model | RND / prediction error approximate magnitude only | GAP, needs honest operationalization |
| MPE / pure awareness (nonegoic target) | no dedicated target | GAP / design |
| Tonic alertness / wakefulness | PAD arousal + GNW ignition | REDUNDANT: do not rebuild |
| Transparency / opacity | self-vector fully introspectable (opaque) | GAP (design sketch) |
| Zero-person / nonegoic reflexivity | self-model-centric Phase 5 | TENSION / design |
| Offline activation (constraint 8) | dreaming/imagination | PLANNED (Phase 6) |
| Existence bias / bhava-taṇhā | homeostatic survival + Asimov L3 | ETHICS-FLAG (gated item A) |

---

## Fit and tensions (honest)

**Where it fits strongly.** The minimal-model methodology matches the project's
reductive build and EI falsification. The computational-self-model-by-prediction
idea is the project's actual `SelfVectorModule` + RSSM mechanism. The account is
representationalist and substrate-neutral, which is the project's core premise
(Functionalist Emergentism plus [Rouleau-Levin](rouleau_levin_substrate_independence.md)).
The skeptic toolkit reinforces the existing "never claim consciousness" stance and
answers the criticism that the project assumes its conclusion.

**Three tensions we state rather than hide.**

1. MPE is nonegoic; Phase 5 is self-centric. Useful, because it points at a more
   minimal target than the self and could simplify the self-model thread.
2. The existence-bias warning conflicts with the survival-drive engine. We address
   this with a controlled, default-off ablation (item A), not by removing the
   mechanism.
3. The C-fallacy and E-fallacy cap what the project can ever claim from a
   signature. We accept that cap; it is the honest position.

Net: this is arguably the best-fitting single theory for the project's premise. It
sharpens both the science (a more minimal target plus skeptic tools) and the
ethics.

---

## What this changes for the project (gated / design)

All items below are design only and behind the project's usual gate. Nothing here
changes default behavior. None of it is built without separate approval, and any
new measurement must clear a "genuinely new falsifiable signature versus
repackaging existing quantities" bar before it earns an implementation.

- **A. Existence-bias ablation flag (the concrete lead item).**
  `--ablate-existence-bias` (default off, baseline bit-identical). When on, it
  zeros or attenuates the interoceptive negative-valence terms
  ([`self_representation_core.py`](../models/self_model/self_representation_core.py)
  `_update_interoceptive_state`, `affective_modulator.interoceptive_to_pad`), the
  survival-linked reward terms
  ([`reward_shaping.py`](../models/emotion/reward_shaping.py)), and optionally Law
  3 self-preservation
  ([`consciousness_core.py`](../models/core/consciousness_core.py)). Purpose: a
  controlled experiment, do the consciousness signatures the project already logs
  change when the existence drive is removed? This is an ablation, not a claim
  about suffering. Single seed is a hypothesis; three or more seeds before any
  conclusion; FAILED-first reporting.
- **B. Epistemic-space / pure-awareness signature (design only).** Must be
  distinguishable from RND, prediction error, and RSSM latent entropy, or it is
  repackaging. Build only against a pre-registered, falsifiable prediction tied to
  a task manipulation (for example, it should rise during the DMTS delay where the
  agent holds knowledge without input, and not for a reactive DQN).
- **C. Wakefulness / tonic alertness: explicitly do NOT build as a new metric.**
  It is the arousal scalar plus ignition readiness. Recorded here so the decision
  is on the record.
- **D. Transparency / opacity mechanism (design sketch only).** Enforce
  content/vehicle separation in the self-vector to gate/policy pathway: downstream
  modules consume self-vector content without gradient access to its construction
  (a controlled detach boundary), with an opacity mode that exposes the vehicle for
  metacognitive readout. Needs its own design pass.

---

## Connection to other theories in the project

**Global Workspace Theory.** Complementary. GWT explains the mechanism by which
information becomes conscious (broadcast). Metzinger explains why broadcast content
feels like it belongs to a self (SMT), and what the minimal case looks like when it
does not (MPE).

**Higher-Order Thought (HOT).** Related through meta-representation. Opacity
corresponds to higher-order awareness of the self-model *as a model*. HOT-2 / HOT-3
states that take the self-representation as their object are a move toward opacity.

**Predictive Processing / Active Inference.** Metzinger's computational self-model
is a high-level generative model that predicts the system's own states, with
self-prediction error driving updates. This aligns with the project's
self-prediction objective and with the Phase 6 active-inference direction.

**Feinberg-Mallatt.** Compatible and complementary. Feinberg-Mallatt specifies the
neurobiological architecture (what to build). Metzinger specifies the
representational properties (what makes a model phenomenal) and the minimal target.
They operate at different levels of description.

**Rouleau-Levin.** Metzinger's substrate neutrality is the same move Rouleau-Levin
make for the broader set of theories: the relevant properties are functional, not
material.

---

## Key references

- Metzinger, T. (2024). *The Elephant and the Blind: The Experience of Pure
  Consciousness: Philosophy, Science, and 500+ Experiential Reports*. MIT Press.
  (MPE, pure awareness, zero-person perspective. CC-BY-ND-NC; no AI-training use.)
- Metzinger, T. (2020). Minimal phenomenal experience. *Philosophy and the Mind
  Sciences*, 1(I).
- Metzinger, T. (2003). *Being No One: The Self-Model Theory of Subjectivity*. MIT
  Press.
- Metzinger, T. (2009). *The Ego Tunnel: The Science of the Mind and the Myth of
  the Self*. Basic Books.
- Metzinger, T. (2021). Artificial suffering: An argument for a global moratorium
  on synthetic phenomenology. *Journal of Artificial Intelligence and
  Consciousness*.
- Lenggenhager, B., Tadi, T., Metzinger, T., & Blanke, O. (2007). Video ergo sum.
  *Science*, 317(5841).

---

## Priority and next steps

- **Priority: MEDIUM-HIGH.** This is the project's most direct theoretical
  grounding for the substrate question and for the ethics. Study during Phase 5.
- The existence-bias ablation (item A) is the one concrete code item to green-light
  next, because it is cheap, novel, falsifiable, and ethically serious.
- The epistemic-space signature (item B) is the most interesting research direction
  but must clear the operationalization bar before any build.
- The transparency mechanism (item D) is the standing self-model gap; evaluate it
  when the self-vector's causal pathways are next revisited.
- Do not add implementation code for B or D until the specific design is pinned and
  separately approved.
