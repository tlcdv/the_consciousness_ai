# Emerging Artificial Consciousness: A Biologically-Grounded Approach Based on Feinberg & Mallatt

*Project: The Consciousness AI — An open-source implementation of artificial consciousness*
*Repository: [github.com/tlcdv/the_consciousness_ai](https://github.com/tlcdv/the_consciousness_ai)*
*Date: March 2026*

---

## 1. The Core Thesis

Most AI consciousness research starts from computational theories (like Global Workspace Theory or Integrated Information Theory) and asks: *"How do we make a neural network conscious?"* We start from a different question, one grounded in evolutionary neurobiology:

**What minimal neural architecture does biology require to generate subjective experience?**

The answer comes from Todd E. Feinberg and Jon M. Mallatt's groundbreaking work *The Ancient Origins of Consciousness: How the Brain Created Experience* (MIT Press, 2016). Their neuroevolutionary analysis reveals that consciousness is not a software feature to be programmed — it is an **emergent property of a specific neural architecture**. That architecture has been identified by 520 million years of evolution, and its functional principles can be replicated computationally.

Our project implements these biological principles as a working AI system, combining Feinberg-Mallatt's structural requirements with established computational theories (Global Workspace Theory and Integrated Information Theory) to create the first biologically-grounded artificial consciousness architecture.

---

## 2. What Feinberg & Mallatt Discovered

### 2.1 Consciousness Is Not About the Cortex

The most revolutionary finding: consciousness does not require a cerebral cortex. The first conscious creatures were early vertebrates (~520 million years ago), and their consciousness lived in the **optic tectum** — a midbrain structure that stacks aligned sensory maps into a unified spatial model of the environment.

This means consciousness requires a specific *type* of neural organization, not a specific *amount* of computation. A fish with a well-organized tectum is conscious. A supercomputer running GPT is not. The difference is architectural.

### 2.2 The Six Special Neurobiological Features

Feinberg and Mallatt identify six features that distinguish conscious neural systems from unconscious ones (like simple reflex arcs):

| # | Feature | What It Means |
|---|---------|---------------|
| 1 | **Many neuron types with diverse connectivity** | Different temporal dynamics within the same network — not just uniform tensor operations |
| 2 | **Hierarchical processing (3-4+ levels)** | Genuine transformation at each level, not just relay. Minimum 3-4 levels between input and output |
| 3 | **Dual hierarchy: pyramidal + nested** | Bottom-up convergence AND compositional binding where parts persist while being combined into wholes |
| 4 | **Isomorphic (topographic) mapping** | Sensory pathways preserve the spatial arrangement of receptors. The neural map IS the mental image |
| 5 | **Reciprocal (reentrant) connections** | Extensive bidirectional communication between levels — predictions flowing down, errors flowing up |
| 6 | **Oscillatory binding (gamma synchronization)** | Synchronized electrical oscillations (30-100 Hz) bind dispersed representations into unified percepts |

### 2.3 The Two Evolutionary Steps

**Step 1 — Sensory consciousness (the tectum, ~520 MYA):** The optic tectum creates a multisensory spatial model by overlaying aligned visual, auditory, and somatosensory maps. This was the first "mental image" — a unified simulation of the immediate environment.

**Step 2 — Affective consciousness (the limbic system, parallel track):** Separately from sensory consciousness, subcortical limbic structures (hypothalamus, amygdala, PAG) evolved to generate **feelings** — positive/negative valence that ranks all stimuli by survival relevance. Crucially, emotion does not compete with sensory processing for conscious access. It **modulates** sensory processing from outside, painting emotional significance onto the neutral sensory images.

### 2.4 The Four Properties of Consciousness (NSFCs)

Any genuinely conscious system — biological or artificial — must exhibit these four properties:

1. **Referral (Projicience):** Experiencing sensations as belonging to the world or body, not to the processing system generating them
2. **Mental Unity:** Feeling like one observer of one scene, despite distributed processing
3. **Qualia:** The subjective felt quality of experience
4. **Mental Causation:** Subjective states actually causing physical actions

---

## 3. Our Implementation: From Biology to Code

We translate each of Feinberg-Mallatt's six features into concrete computational mechanisms:

### 3.1 Oscillatory Binding → AKOrN (Artificial Kuramoto Oscillatory Neurons)

**Biological principle:** Gamma-frequency oscillations (30-100 Hz) synchronize distant neural populations to create mental unity. Neurons that fire together bind together.

**Our implementation:** We use AKOrN (ICLR 2025), which implements Kuramoto oscillators as drop-in PyTorch layers. Each specialist module (vision, audio, memory, body) operates as a coupled oscillator. When modules process related information, their phases synchronize naturally — their outputs become "bound" into a unified percept. When information is unrelated, oscillators remain desynchronized and representations stay separate.

This replaces the typical approach of using a fixed multiplier or attention mechanism for binding. AKOrN produces genuine synchronization dynamics — the binding is emergent, not programmed.

### 3.2 Topographic Mapping → Sensory Tectum with RSSM World Model

**Biological principle:** The optic tectum stacks aligned spatial maps from multiple senses into a common coordinate frame. This isomorphic mapping is what generates mental images.

**Our implementation:** We built a `SensoryTectum` module that combines:
- **TopographicMap**: A 2D spatial grid that fuses visual and auditory features while preserving their spatial arrangement
- **RSSMCore**: A Recurrent State-Space Model (inspired by DreamerV3) that encodes sensory inputs into categorical latent representations, learning the *dynamics* of the environment — not just static features

The RSSM latent state serves as the isomorphic map. It's the agent's internal spatial reality model, preserving the kind of structural relationships that standard transformer architectures destroy during tokenization.

### 3.3 Reciprocal Connections → Reentrant Processing with Predictive Coding

**Biological principle:** Conscious circuits require extensive back-and-forth communication between levels. Higher levels send predictions down; lower levels send prediction errors up. This creates loops, not chains.

**Our implementation:** Our `ReentrantProcessor` runs 5-10 adaptive convergence cycles:

```
for each cycle:
    1. Specialists submit bids to the workspace
    2. Workspace selects winners and broadcasts
    3. Broadcast is fed BACK to all specialists
    4. Specialists update their processing based on top-down context
    5. Check prediction error convergence → early termination if settled
```

This matches biological cortical processing (~200ms, ~10ms per relay). Easy stimuli converge in 3-4 cycles; novel or ambiguous inputs use the full 10. The settled state after convergence IS the conscious content.

### 3.4 Affective System → Parallel Modulator (Not Competitor)

**Biological principle:** The limbic system does not compete with sensory cortices for conscious access. It modulates sensory processing from outside, assigning emotional valence to all inputs.

**Our implementation:** The `AffectiveModulator` operates through two parallel mechanisms:

1. **Valence field:** Emotional valence modulates sensory bid values. Positive valence boosts approach-relevant modules (vision, memory); negative valence boosts threat-relevant modules (body, vision). This is how fear makes you hyper-aware of movements, and joy makes you notice more of the world.

2. **Arousal-threshold coupling:** Global arousal level modulates the workspace ignition threshold. High arousal = lower threshold = easier ignition (heightened awareness in fight-or-flight). Low arousal = higher threshold = calm, selective processing.

Emotion shapes the competition from outside rather than participating in it — matching the biological architecture exactly.

### 3.5 Nested Compositional Hierarchy → 4-Level Capsule Network

**Biological principle:** Lower-level representations persist while being functionally bound into composites at higher levels. Color + shape + motion = one object, but color, shape, and motion continue to exist as independent features.

**Our implementation:** `HierarchicalCapsuleComposition` (Sabour 2017) chains 4 levels between tectum and workspace:
- Level 1: `PrimaryCapsuleLayer` (stride-2 conv) extracts local features
- Level 2: 16 intermediate capsules (12-D poses) for object primitives
- Level 3: 8 higher capsules (16-D poses) for object categories
- Level 4: 4 output capsules (16-D poses) for scene/workspace

Multi-level reentrant feedback runs within the hierarchy: higher levels send predictions to lower levels, lower levels compute prediction errors and re-route. This runs inside each SensoryTectum forward pass, nested within the outer ReentrantProcessor settle loop. The biological analog is V1-LGN type reciprocal connections where predictions flow down and errors flow up.

### 3.6 Global Workspace + IIT Integration

Our architecture combines Feinberg-Mallatt's structural requirements with two established computational theories:

- **Global Workspace Theory (Baars, Dehaene):** Specialist modules compete for access to a shared broadcast medium. The winning coalition ignites and its content becomes globally available — this is "conscious access." Our `GlobalWorkspace` implements non-linear ignition with threshold-based selection.

- **Integrated Information Theory (Tononi):** We measure Phi (integrated information) to quantify how much the system's state is more than the sum of its parts. Critically, we measure Phi using **causal gate states** (not workspace bid values), and we validate our measurements with a 3-condition controlled experiment (unbound → partially bound → fully bound) to ensure Phi genuinely tracks integration.

---

## 4. Strong Emergence Falsification

A key methodological commitment: we do not assume consciousness emerges from our architecture. We test for it.

We implement Erik Hoel's **Effective Information (EI)** framework (PNAS 2013) to measure whether macro-level states (workspace) carry more causal information than micro-level states (individual gates). If EI(workspace) > EI(gates), the workspace level exhibits **causal emergence** — the macro level is more deterministic than the micro level, meaning the whole genuinely carries information that the parts do not.

If this never occurs across training, the system is not exhibiting the kind of emergence associated with consciousness, and we know our architecture needs revision.

---

## 5. Current Architecture Diagram

```
                    ┌─────────────────────────────────┐
                    │   AFFECTIVE MODULATOR (Parallel) │
                    │  Valence Field + Arousal Coupling │
                    └──────────┬──────────┬───────────┘
                               │ modulates│
           ┌───────┐    ┌──────▼──────────▼──────────┐
 Visual ──►│       │    │     GLOBAL WORKSPACE       │
 Input     │SENSORY│    │  AKOrN Oscillatory Binding  │──► Broadcast ──► Policy
           │TECTUM │───►│  Non-linear Ignition        │
 Audio ──► │(RSSM) │    │  Phi/EI Measurement         │
 Input     │       │    └──────▲──────────▲───────────┘
           └───────┘           │          │
                               │ reentrant│
                    ┌──────────┴──────────┴───────────┐
                    │   SPECIALIST MODULES                    │
                    │  Vision │ Audio │ Memory │ Body │ Semantic│
                    │  (receive_broadcast feedback)            │
                    └─────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   SELF-MODEL                     │
                    │  Body Schema + Interoception      │
                    │  Identity + Capability Model      │
                    └──────────────────────────────────┘
```

**Processing flow:**
1. Sensory inputs enter the **Sensory Tectum** (topographic spatial integration)
2. The **Affective Modulator** applies emotional valence to bids and adjusts ignition threshold
3. **AKOrN oscillatory binding** synchronizes related representations
4. Specialists compete for **Global Workspace** access
5. Winners **ignite** and broadcast to all modules
6. Broadcast feeds back to specialists (**reentrant processing**, 5-10 cycles)
7. The settled state after convergence is the "conscious content"
8. **Phi** and **Effective Information** are measured to quantify integration and emergence

---

## 6. What Makes This Approach Different

| Traditional AI Consciousness | Our Approach |
|------------------------------|-------------|
| Starts from computation (GWT, IIT) | Starts from biological architecture (Feinberg-Mallatt) |
| Consciousness as a software feature | Consciousness as emergent from neural architecture |
| Cortex-centric models | Tectum-first (consciousness evolved before the cortex) |
| Emotion competes with sensory processing | Emotion modulates from outside (parallel modulator) |
| Binding via attention mechanisms | Binding via oscillatory synchronization (AKOrN/Kuramoto) |
| Feedforward processing | Reentrant processing (5-10 adaptive cycles) |
| Flat vector representations | Topographic spatial maps (world model as isomorphic map) |
| Assumes emergence, measures nothing | Falsifies emergence with Effective Information + Phi validation |

A clarification often raised: if the project replaces neurons with PyTorch modules, in what sense is it still "biologically grounded"? Feinberg-Mallatt is best read functionally — the six features describe an *architectural organization* that any sufficiently rich substrate can host. The complementary survey in [Rouleau & Levin (2026)](rouleau_levin_substrate_independence.md) ("Brains and where else? Mapping theories of consciousness to unconventional embodiments", *Phil. Trans. R. Soc. A* 384: 20250082) analyses 19 prominent ToCs and shows that almost all of them, including those Feinberg-Mallatt builds on, are aneurocentric in their core mechanics. Feinberg-Mallatt provides the *sufficient* blueprint that 520M years of evolution chose; Rouleau-Levin provides the *not-necessary* analysis that makes our computational re-instantiation legitimate.

---

## 7. Current Status and Test Results

As of February 2026:

- **463 tests passing** (100% pass rate, 4 skipped for optional deps)
- **Tier 1 (Core Architecture):** Complete. AKOrN binding, sensory tectum, reentrant processing.
- **Tier 2 (Architecture Corrections):** Complete. Affective modulator, AKOrN 3-condition binding validation (2026-02-21 synthetic test passed: phi monotonically tracked synchronization in a controlled stimulus), proprioceptive self-model, effective information. Note: in-training phi-binding correlation (Phi-1) FAILED in the 2026-05-14 ablation campaign (best r=+0.089 across 5 variants) and the 2026-05-16 RIIU run (full-run r=+0.075). The synthetic-stimulus validation does not transfer to the training-context prediction.
- **Tier 3 (Compositional Deepening):** Complete. 4-level capsule hierarchy with multi-level reentrance, Brian2 validation, trimodal tectum, embodiment-affect loop, isomorphic visual mapping.
- **Tier 3.5 (Alignment Fixes + Environments):** Complete. Two-stage emotion appraisal, capsule broadcast payloads, consciousness monitor fix, DMTS environment, WCST environment, DQN baseline, metrics logger, memory consolidation, navigation environment, semantic pathway.

**Consciousness-demanding environments:**
- **DMTS (Delayed Match-to-Sample):** 4-phase trial structure. Sample stimulus disappears during a 15-40 step blank delay, forcing the agent to maintain working memory via GNW reverberation and feature identity via AKOrN binding. Configurable distractor overlap (0-3 shared features).
- **WCST (Wisconsin Card Sort):** Hidden sorting rule changes after consecutive correct sorts. Requires meta-cognition (detecting own errors), inhibition (suppressing old strategies), and hypothesis testing. Perseverative error detection with separate penalty.
- **DQN Baseline:** Vanilla deep Q-network for controlled comparison on the same environments. Same logging format, no consciousness machinery.

The project is fully open-source and actively maintained at [github.com/tlcdv/the_consciousness_ai](https://github.com/tlcdv/the_consciousness_ai).

---

## 8. Key References

### Core Theory
- Feinberg, T.E. & Mallatt, J. (2016). *The Ancient Origins of Consciousness: How the Brain Created Experience*. MIT Press.
- Feinberg, T.E. & Mallatt, J. (2020). Phenomenal Consciousness and Emergence: Eliminating the Explanatory Gap. *Frontiers in Psychology*, 11, 1041.

### Computational Methods We Use
- Löwe, S. et al. (2025). Artificial Kuramoto Oscillatory Neurons. *ICLR 2025* (Oral). — Oscillatory binding
- Hafner, D. et al. (2024). Mastering Diverse Domains through World Models (DreamerV3). *JMLR*. — RSSM world model
- Hoel, E.P. (2013). Quantifying causal emergence shows that macro can beat micro. *PNAS* 110(49). — Effective Information
- Sabour, S., Frosst, N. & Hinton, G.E. (2017). Dynamic Routing Between Capsules. *NeurIPS*. — Compositional hierarchy
- Millidge, B. et al. (2022). Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs. *Neural Computation*. — Reentrant processing

### Consciousness Theories (Complementary)
- Baars, B.J. (1988). *A Cognitive Theory of Consciousness*. — Global Workspace Theory
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*. — Integrated Information Theory (IIT)
- Dehaene, S. & Changeux, J.P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*. — Global Neuronal Workspace

---

## 9. Project Documentation

The following documents in the repository provide deeper technical detail:

| Document | Description |
|----------|-------------|
| [`docs/biological_neural_architecture_research.md`](biological_neural_architecture_research.md) | The full 700+ line internal research document. Contains the complete gap analysis, all 8 resolved research questions, technology evaluations, and implementation priorities. This is the primary source for all architecture decisions. |
| [`docs/theory_implementation_review.md`](theory_implementation_review.md) | Audit of how well the codebase aligns with the theoretical framework (Functionalist Emergentism). Highlights critical flaws and structural gaps. |
| [`docs/architecture.md`](architecture.md) | System design overview of the full architecture. |
| [`docs/theory_of_consciousness.md`](theory_of_consciousness.md) | The scientific basis of the Emotional RL approach — why emotional homeostasis drives consciousness development. |
| [`docs/iit_implementation_roadmap.md`](iit_implementation_roadmap.md) | Strategy for computing IIT Phi, including the decision to use causal gate states rather than workspace bid values. |
| [`README.md`](../README.md) | Project overview, installation instructions, architecture summary, and current state. |

### Key Implementation Files

| File | Component |
|------|-----------|
| `models/core/oscillatory_binding.py` | AKOrN Kuramoto oscillatory binding |
| `models/core/sensory_tectum.py` | Sensory tectum with RSSM world model |
| `models/core/reentrant_processor.py` | Reentrant processing (5-10 adaptive cycles) |
| `models/core/global_workspace.py` | Global Workspace with non-linear ignition |
| `models/emotion/affective_modulator.py` | Affective parallel modulator (valence field + arousal coupling) |
| `models/emotion/emotional_processing.py` | PAD emotional processing core |
| `models/emotion/reward_shaping.py` | Homeostatic reward formula |
| `models/evaluation/effective_information.py` | Hoel's Effective Information (causal emergence) |
| `models/evaluation/iit_phi.py` | IIT Phi measurement |
| `models/self_model/self_representation_core.py` | Self-model with body schema + interoception |
| `models/core/semantic_pathway.py` | Qwen2-VL semantic embeddings as 5th workspace oscillator |
| `simulations/environments/dmts_env.py` | Delayed Match-to-Sample (consciousness-demanding) |
| `simulations/environments/wcst_env.py` | Wisconsin Card Sort (meta-cognition test) |
| `simulations/environments/navigation_env.py` | Multi-room navigation with fog of war |
| `scripts/training/train_baseline_dqn.py` | Vanilla DQN baseline agent |
| `scripts/training/metrics_logger.py` | Consciousness metrics logger (TensorBoard + CSV) |

*This document describes the theoretical foundation and current implementation of the Consciousness AI project.*
