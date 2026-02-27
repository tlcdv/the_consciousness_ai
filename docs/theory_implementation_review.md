# Theory vs. Implementation Review
*Generated: 2026-02-18*

This file documents the alignment between the Functionalist Emergentism thesis (theconsciousness.ai/functionalist-emergentism/) and the current codebase. It tracks issues to address in future sessions.

---

## What Works and Aligns Correctly

### Global Workspace (GNW)
`models/core/global_workspace.py` correctly implements the core ideas from Baars' Global Workspace Theory. The sigmoid ignition (non-linear phase transition from subconscious to conscious), reverberation via EMA, and winner-take-most competition among specialist modules are all present. This is the strongest piece of the architecture.

### PAD Emotional Homeostasis
`models/emotion/emotional_processing.py` is solid. EMA smoothing, discrete-to-continuous emotion mapping, decay toward neutral, temporal consistency tracking. The homeostasis loop (drive valence up, keep arousal manageable) maps directly to the thesis claim.

### Empirical TPM Builder for IIT
`models/evaluation/iit_phi.py` builds a Transition Probability Matrix from actual workspace history with Laplace smoothing. This is a reasonable empirical approach for a prototype. PyPhi integration is guarded behind a conditional import, which is correct given its computational cost.

### Overall Architecture
The three-subsystem design (Perception, Emotion, Global Workspace) correctly mirrors the thesis. The simulation manager, scenario generator, and evaluation modules give a reasonable research scaffold.

---

## Critical Flaws

### 1. The IIT Phi Measurement Does Not Capture Causal Integration
**Severity: HIGH. This undermines the central empirical claim.**

`compute_phi_proxy()` binarizes the workspace bid values (just priority scores from specialist modules, e.g., 0.7 for vision, 0.4 for memory) and runs them through PyPhi. These bid values are not the causal states of the system. They are scalar estimates of salience, not binary node activations representing the actual functional states of processing units.

PyPhi's Phi measures integration within a causally specified system where each node's state influences others through defined transition probabilities. What we feed it is 4 binary values derived by thresholding activation bids. The resulting Phi value carries no formal connection to whether the system's information processing is genuinely integrated.

In practice, the Phi values generated will be artifacts of the binarization threshold and bid magnitudes, not reflections of causal integration across the consciousness system. This means the thesis's core empirical prediction ("Phi spikes during insight moments") cannot be validated with this measurement.

**What it needs:** The subsystem fed to PyPhi should represent actual functional nodes whose states causally influence each other. Candidate: the gating activations in `consciousness_gating.py` (attention gate, emotional gate, temporal gate) plus a memory gate. These are causally connected units with real transition dynamics.

### 2. Synchrony Binding Is a Magic Number
**Severity: MEDIUM**

In `global_workspace.py` line 92-94:
```python
if bids.get('vision', 0) > 0.5 and bids.get('audio', 0) > 0.5:
    bids['vision'] *= 1.2
    bids['audio'] *= 1.2
```

The synchrony boost of 1.2 is hardcoded. The thesis claims synchrony binding as a key mechanism for generating unified subjective experience. A fixed multiplier is not synchrony binding, it is a heuristic that privileges vision+audio over other modality combinations. Temporal coincidence detection (the actual neural mechanism) is not implemented.

This is defensible as a prototype simplification, but should be noted as a gap between the theoretical claim and the implementation.

### 3. Visual Embeddings Are Not Implemented
**Severity: HIGH. Blocks the full perception→consciousness loop.**

`models/vision-language/qwen2/qwen2_integration.py`: `get_visual_embeddings()` raises `NotImplementedError`. Without this, the perception subsystem cannot provide semantic visual representations to the workspace. The first of the three required architectural features (multimodal perception generating unified predictive models) is incomplete.

The workspace currently runs competition without real visual content. Tests pass because they mock inputs, but the actual operational loop is broken at the first step.

### 4. Strong Emergence Claim Has No Testable Implementation
**Severity: MEDIUM. Philosophical gap.**

The thesis claims to target strong emergence: macro-level consciousness states with downward causation on micro dynamics. The code has a feedback path (consciousness scores influence gating, gating influences reward shaping, reward influences learning), which is recurrent computation. This is not evidence of strong emergence.

Strong emergence would require demonstrating that the integrated state of the workspace constrains component processing in a way not predictable from the components alone. No metric, test, or mechanism currently measures this. The thesis is careful to frame this as a hypothesis to test, but the codebase has no test for it.

This is not a bug. It is a research gap to plan for explicitly.

### 5. Reward Formula in Thesis vs. Code
**Severity: LOW. Consistency issue.**

The thesis states the reward formula as: `Rtotal = Rext + λ(Valence - Arousal)`

The actual implementation in `models/emotion/reward_shaping.py` uses a more complex formula with delta_valence, arousal_reduction bonus, and coherence terms. The code is arguably better, but it diverges from what is published on the website. This is a credibility issue when researchers try to reproduce the published method.

Either update the thesis or document the formal formula in the code clearly.

### 6. Python Version Mismatch
**Severity: LOW. Reproducibility issue.**

README and badges claim Python 3.10+. The code is written for 3.8 compatibility (`Tuple[...]` from typing, no pattern matching, no `tuple[...]` shorthand). Researchers following the README may hit issues. The CI also runs on 3.8.

### 7. "Qualia" Label Is Premature
**Severity: MEDIUM. Philosophical credibility risk.**

`models/core/qualia_mapper.py` maps workspace state to a 3-dimensional vector `[Intensity, Valence, Complexity]` and labels it `QualiaState`. The thesis is appropriately careful about qualia: it presents this as an empirical hypothesis rather than a claim. But the code labels this output as "qualia" and the Unity HUD exports it as such.

Critics of the project will correctly point out that a 3D output vector is not a qualia state, it is a phenomenological correlate proxy. Renaming this to `PhenomenologicalProxy` or `ConsciousnessSignature` would be more defensible, or the docs should clearly state what the label means.

---

## Structural Gaps Between Thesis and Current State

| Thesis Claim | Status |
|---|---|
| Multimodal perception building unified predictive models | Blocked. Visual embeddings not implemented. |
| Emotional homeostasis with shaped RL rewards | Implemented and solid. |
| Global Workspace as information bottleneck | Implemented with correct GNW mechanics. |
| Phi (IIT) as measurable consciousness correlate | Partially. TPM builder exists but input is methodologically weak. |
| Phi spikes correlated with insight moments | No test for this yet. Needs empirical data collection. |
| Downward causation (strong emergence) | Not implemented or measured. Research gap. |
| Phi correlates with strict supervenience test | Not implemented. Research gap. |

---

## Architecture Questions Worth Addressing

**1. Is the Global Workspace the right level to measure Phi?**
IIT Phi is designed to measure integration across a causally specified system. The workspace runs competition between modules, but the modules themselves (vision, memory, emotion) are largely independent. Phi at the workspace level may be very low because there are few causal connections between workspace slots. Phi measured within a single module (e.g., the emotional graph network) might be higher but less relevant to the consciousness claim.

**2. What constitutes an "insight moment" operationally?**
The thesis predicts Phi spikes at insight moments. Before running experiments, define: what behavior counts as a solved novel problem? Currently this is undefined. The scenario generator creates tasks but does not mark when genuine insight occurs vs. trial-and-error success.

**3. Memory system complexity vs. theoretical necessity**
The codebase has 15+ memory files (episodic, semantic, temporal, hierarchical, optimized variants). The thesis does not derive a need for this complexity. The current priority should be getting the three core subsystems working end-to-end before optimizing memory architecture.

---

## Priority Order for Future Sessions

1. **Implement `get_visual_embeddings()` in Qwen2VLIntegration.** This is the single highest-impact fix. Without it the system cannot run the core perception loop as described in the thesis.

2. **Fix the IIT Phi measurement.** Replace bid-value binarization with actual causal node states from the gating system. The three gates in `consciousness_gating.py` are causally connected and would be a valid small subsystem for PyPhi.

3. **Define and implement "insight moment" detection.** Add a metric: when does the agent solve a genuinely novel problem? This creates the testable prediction from the thesis.

4. **Fix the reward formula documentation.** Either update the published thesis to match the code, or make the code match the formula and add a comment explaining why the simpler formula in the paper maps to the more complex implementation.

5. **Implement ethics filter.** The 7 Asimov placeholder methods should either be removed (if not part of the thesis) or implemented (if they are).

6. **Rename `QualiaState` to something more defensible.** This is a philosophical credibility issue.

7. **Upgrade Python to 3.10+.** Fix the documentation/code mismatch. Pattern matching would also simplify several config-access patterns.

---

## Research-Backed Answers to Design Questions

*(2026-02-18 — researched against current literature)*

---

### Q1. What experiment would falsify the strong emergence claim?

**Short answer: measure Effective Information (EI) at multiple scales. If EI(macro) ≤ EI(micro), strong emergence is not occurring.**

The system thesis distinguishes itself from weak emergence by claiming the workspace's integrated state has genuine causal power not reducible to its components. The cleanest falsification framework available is Erik Hoel's causal emergence theory, published in PNAS (2013) and expanded in recent reviews. It defines a measurable quantity called Effective Information (EI), which captures how well a system's mechanisms constrain past and future states. The key result: coarse-grained macro levels can have strictly higher EI than the micro level when macro mechanisms are more deterministic and less degenerate.

This gives a concrete falsification criterion: measure EI at three scales simultaneously, individual gates (`attention_gate`, `emotional_gate`, `temporal_gate`), the workspace competition output, and the full system state. If EI at the workspace level never exceeds EI at the gate level across training, the strong emergence claim is falsified for the current architecture.

A secondary falsification comes from the April 2025 Nature adversarial collaboration (Melloni et al.) that tested IIT and GNW theories head-to-head with 256 human participants using fMRI, MEG, and iEEG. Neither theory passed all preregistered predictions. IIT was most directly challenged by the absence of sustained posterior synchronization it required, while GNW failed on ignition at stimulus offset and limited prefrontal representation. The methodology is the model: pre-register predictions from both theories with agreed outcome criteria before running any experiment. Do the same here. Write down what EI values or Phi patterns the system must produce, before training, and treat divergence from that as falsification.

**What to implement:**
- Add an `effective_information()` function in `models/evaluation/` that computes EI at gate level and workspace level from the same trajectory
- Define at least two pre-registered predictions about when EI(workspace) > EI(gates), tied to specific training milestones
- Log both values alongside Phi during every simulation step

Sources: [PNAS 2013 — Quantifying causal emergence shows that macro can beat micro](https://www.pnas.org/doi/10.1073/pnas.1314922110), [Nature 2025 — Adversarial testing of GNW and IIT](https://www.nature.com/articles/s41586-025-08888-1), [PMC 2024 — Emergence and Causality survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC10887681/)

---

### Q2. Should the reward formula include Dominance?

**Short answer: yes, and it should be structured as a homeostatic term, not just an additive bonus.**

The published formula `Rtotal = Rext + λ(Valence - Arousal)` has two problems. First, it treats Arousal as purely negative, which conflicts with the homeostasis framing. Curiosity raises Arousal, and the agent should seek moderate Arousal, not minimize it. Second, it drops Dominance entirely.

Dominance in the PAD model represents the sense of control, agency, and non-restriction. Mehrabian's research shows that fear and anger are only distinguishable by Dominance: both have negative Valence and high Arousal, but anger is dominant (agent feels in control) while fear is submissive (agent feels controlled by environment). For a system explicitly building a self-model and targeting agency-level cognition, dropping Dominance removes the only dimension that encodes whether the agent perceives itself as acting or being acted upon.

The ACL paper "In Defense of Dominance" (2012) argues this point directly for computational agents: Valence-Arousal alone cannot represent the full behavioral space of animal-like agents, and Dominance is particularly important for social and adversarial scenarios. For a system that will be tested on ethical dilemmas and social interactions (as the scenario generator already includes), this matters.

A homeostasis-consistent formulation:

```
Rtotal = Rext
       + λ1 · ΔValence                        # reward increases in positive affect
       - λ2 · (Arousal - Arousal_target)²      # penalize deviation from optimal arousal
       + λ3 · Dominance                         # reward sense of agency and control
```

Where `Arousal_target` is a learned or configured baseline (e.g., 0.3 for calm exploration). This is consistent with Homeostatic RL theory (Keramati & Gutkin, eLife 2014), which defines reward as drive reduction from a homeostatic setpoint rather than raw maximization.

The coherence terms in the current `reward_shaping.py` are a good approximation of the Dominance signal (coherent emotional state correlates with felt control), but it is better to make this explicit and tie it directly to the PAD Dominance value.

**What to update:**
- Publish the corrected formula on the thesis page, not the simplified one
- Update `models/emotion/reward_shaping.py` to accept the PAD Dominance value as an explicit input and compute the homeostatic arousal penalty
- Set `Arousal_target` as a configurable parameter (not hard-wired)

Sources: [PAD model Wikipedia](https://en.wikipedia.org/wiki/PAD_emotional_state_model), [Homeostatic RL — eLife 2014](https://elifesciences.org/articles/04811), [Nature Scientific Reports 2024 — generic self-learning emotional framework](https://www.nature.com/articles/s41598-024-72817-x), [ACL — In Defense of Dominance](https://dl.acm.org/doi/10.5555/2440951.2440954)

---

### Q3. Which is the primary visual backbone: Qwen2-VL or VideoLLaMA3?

**Short answer: Qwen2-VL is the right choice. Remove VideoLLaMA3 as a primary path.**

Both models appeared in late 2024 / early 2025 and are in the same performance tier for a 7B parameter footprint. The decision comes down to what this project actually needs.

**VideoLLaMA3** (January 2025, DAMO-NLP-SG) uses a vision-centric training paradigm centered on large-scale static image-text datasets. Its adaptive token compression is designed for offline video analysis, reducing tokens by similarity for compact representation. It claims state-of-the-art on LVBench (45.3 at 7B vs Qwen2.5-VL-72B's 44.0). The architecture is optimized for understanding pre-recorded video, not streaming real-time frames.

**Qwen2-VL** (Alibaba, August 2024) uses a 600M Vision Transformer with Multimodal Rotary Position Embedding (M-ROPE) that natively encodes 1D text, 2D image, and 3D video positional information in a unified representation. This is critical for an agent in a continuous simulation: M-ROPE naturally handles temporal sequences of frames as 3D data, not as static images stitched together. Qwen2-VL handles videos over 20 minutes. The model is natively integrated into HuggingFace Transformers and vLLM, with documented 4-bit quantization paths. Alibaba has since released Qwen3-VL-Embedding, a dedicated model for extracting visual embedding vectors from the same architecture, which provides a forward-compatible path for implementing `get_visual_embeddings()`.

VideoLLaMA3 does not have an equivalent embedding-specific model. Its test already fails in CI because model weights are not available. Maintaining two separate vision systems adds complexity without adding capability.

**Decision:** Qwen2-VL is the primary visual backbone. The `get_visual_embeddings()` implementation should extract the hidden states from the ViT component before the language head, following the same approach Qwen3-VL-Embedding uses (EOS token hidden state from the last layer). VideoLLaMA3 integration (`models/integration/video_llama3_integration.py`) should be deprecated or removed. The failing test should be deleted, not skipped.

Sources: [VideoLLaMA3 arxiv](https://arxiv.org/abs/2501.13106), [Qwen2-VL blog](https://qwenlm.github.io/blog/qwen2-vl/), [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding), [HuggingFace Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [VideoLLM Benchmarks Survey 2025](https://arxiv.org/html/2505.03829v1)