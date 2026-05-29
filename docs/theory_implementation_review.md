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

### 1. ~~The IIT Phi Measurement Does Not Capture Causal Integration~~ RESOLVED (2026-03-17)
**Severity: HIGH. Originally undermined the central empirical claim.**

**Resolution:** `compute_phi_from_gate_state()` now uses the 5 ConsciousnessGate nodes (attention, stability, adaptation, coherence, confidence) as the IIT subsystem. These nodes have genuine causal dependencies. Adaptive binarization thresholds from running medians replace the hardcoded 0.5. `GATE_CM` connectivity matrix (5x5) encodes the causal graph for PyPhi. Geometric proxy metric (determinism x integration) available when pyphi is not installed. 38 tests validate the rewrite.

### 2. ~~Synchrony Binding Is a Magic Number~~ RESOLVED (2026-02-27)
**Severity: MEDIUM. Originally a heuristic gap.**

**Resolution:** AKOrN (Artificial Kuramoto Oscillatory Neurons, ICLR 2025) replaced the hardcoded 1.2 multiplier. Each specialist module operates as a coupled oscillator on a hypersphere. Binding emerges from phase synchronization dynamics, validated by a 3-condition controlled experiment (unbound/partial/full) showing Phi monotonically tracks synchronization.

### 3. ~~Visual Embeddings Are Not Implemented~~ RESOLVED (2026-03-15)
**Severity: HIGH. Originally blocked the perception-to-consciousness loop.**

**Resolution:** `get_visual_embeddings()` in `models/vision_language/qwen2/qwen2_integration.py` extracts features from the ViT encoder's visual tower before the language head. Returns mean-pooled `(1536,)` or spatial grid `(1536, H, W)`. Graceful degradation to zero tensors when model weights unavailable. Wired into `ConsciousnessAgent.step()` for the full cognitive cycle. The spatial pathway uses DINOv2-B/14 (RetinotopicEncoder) for the tectum, while Qwen2-VL provides the cortical semantic pathway.

### 4. ~~Strong Emergence Claim Has No Testable Implementation~~ PARTIALLY RESOLVED (2026-02-27)
**Severity: MEDIUM. Originally a research gap, now has tooling but not yet experimentally validated.**

**Resolution:** `compute_effective_information()` in `models/evaluation/effective_information.py` implements Hoel's PNAS 2013 framework. `compare_ei_levels()` compares EI at gate vs. workspace level. If EI(workspace) > EI(gates), the workspace exhibits causal emergence. 11 tests cover deterministic/random/partial TPMs, level comparison, and discretization. What remains: pre-registering specific predictions about when EI(workspace) > EI(gates) during training, and running the actual experiments.

**2026-05-17 update:** pre-registered predictions defined in `docs/preregistered_predictions.md` were tested across two campaigns (pyphi ablation 2026-05-14, RIIU 2026-05-16). All four testable predictions (EI-1, Phi-1, IM-1, IM-3) FAILED. EI-3 (EI/reward correlation) remains INCONCLUSIVE pending higher phi variance.

**2026-05-24 update:** the pre-registered architectural escalation chain (`docs/preregistered_predictions.md` sections 10-12) was run to completion. Phase B (AKOrN content-level cross-attention) and Phase B-alt (KomplexNet complex-valued binding) were each pre-registered and tested at 200 episodes. Across 9 runs spanning 4 architectures and 2 phi formulations (pyphi, RIIU), no run reaches the pre-registered r > 0.4 in the predicted positive direction, nor the partial r > 0.15. The in-training Phi-1 prediction stands FAILED. The KomplexNet RIIU run produced the campaign's first substantively significant phi-binding correlation, r = -0.11 (p < 1e-100, n = 39000), in the INVERSE direction: tight binding compresses representational variance, which lowers integrated information. The pre-registered threshold and sign are not revised. What is exhausted is one in-training measurement choice. The 2026-02-21 3-condition synthetic test of phi monotonicity with binding on a controlled stimulus still stands, and the project's mission proceeds via other signatures and Phase 5. See `docs/preregistered_predictions.md` section 12 and `docs/results/phi1_phaseBalt_2026_05_24.md`.

### 5. ~~Reward Formula in Thesis vs. Code~~ RESOLVED (2026-02-27)
**Severity: LOW. Originally a consistency issue.**

**Resolution:** `compute_emotional_reward()` in `models/emotion/reward_shaping.py` now implements the corrected homeostatic formula: `Rtotal = Rext + lambda1*DeltaValence - lambda2*(Arousal - Arousal_target)^2 + lambda3*Dominance`. The published thesis formula was updated to match. `Arousal_target` is configurable.

### 6. ~~Python Version Mismatch~~ RESOLVED (2026-03-18)
**Severity: LOW. Originally a reproducibility issue.**

**Resolution:** 602 type annotations migrated across 111 files to Python 3.10+ syntax (`list[X]`, `dict[K, V]`, `X | None`). `from __future__ import annotations` added to 93 files for backward compatibility. CI runs on 3.10+.

### 7. "Qualia" Label Is Premature
**Severity: MEDIUM. Philosophical credibility risk.**
**Status: RESOLVED (2026-03-18).** Renamed `QualiaState` to `PhenomenologicalState` and `QualiaMapper` to `PhenomenologicalMapper` in `models/core/qualia_mapper.py`. Docstrings now explicitly frame the 3D vector as "an empirical correlate proxy, not a claim about actual qualia." Backward compatibility aliases preserved for any external references.

### 8. ~~Consciousness Monitor Uses Circular Progress Factor~~ RESOLVED (2026-03-22)
**Severity: HIGH. Consciousness level increased with step count by construction, not from actual metrics.**

**Resolution:** Removed `progress_factor = min(1.0, self.step_count / 50.0)` from `evaluate_development()` and `evaluate_state()` in `models/evaluation/consciousness_monitor.py`. Consciousness level, emotional awareness, and memory coherence are now computed purely from actual metrics (attention, valence, arousal), not elapsed time.

### 9. ~~Capsule Structure Lost in GNW Broadcast~~ RESOLVED (2026-03-22)
**Severity: MEDIUM. The entire capsule compositional hierarchy was flattened to a scalar bid before workspace competition.**

**Resolution:** Added `broadcast_payload: dict[str, Any] | None` field to `WorkspaceState`. When tectum wins competition, the broadcast preserves structured capsule poses and activities from `get_capsule_payload()`. Downstream consumers (action selection, memory storage) can access compositional hierarchy instead of just a flat vector.

### 10. ~~Emotion System Bypassed by Brightness Lookup~~ RESOLVED (2026-03-22)
**Severity: HIGH. The entire affective modulator architecture was never exercised during training.**

**Resolution:** Replaced `evaluate_reflex_emotion()` (3-line pixel brightness to PAD mapping) with two-stage `evaluate_emotion()`:
1. **Reflex layer** (pre-workspace): computes valence from reward prediction error and arousal from tectum surprise bid
2. **Appraisal layer** (post-broadcast): uses PhenomenologicalMapper on workspace broadcast content to modulate valence and compute dominance

### 11. ~~Environments Trivially Solvable Without Consciousness~~ RESOLVED (2026-03-22)
**Severity: HIGH. Dark Room could be solved by a simple gradient follower. No task required working memory, binding, or meta-cognition.**

**Resolution:** Added two consciousness-demanding environments:
- **DMTS (Delayed Match-to-Sample):** Sample disappears during 15-40 step blank delay. Requires GNW reverberation for working memory, AKOrN binding for feature identity, and selective attention for distractor filtering.
- **WCST (Wisconsin Card Sort):** Hidden rule changes without warning. Requires meta-cognition (detecting own performance drop), inhibition (suppressing old rule), and hypothesis testing (trying each dimension systematically).
- **DQN baseline** added for controlled comparison on the same environments with the same logging format.

---

## Structural Gaps Between Thesis and Current State

| Thesis Claim | Status |
|---|---|
| Multimodal perception building unified predictive models | **Complete.** Dual-stream: DINOv2 spatial + Qwen2-VL semantic. Trimodal tectum fusion. |
| Emotional homeostasis with shaped RL rewards | **Complete.** Full PAD with homeostatic arousal term + Dominance. |
| Global Workspace as information bottleneck | **Complete.** GNW with AKOrN binding, reentrant processing, capsule hierarchy. |
| Phi (IIT) as measurable consciousness correlate | **Tested and FAILED (in-training Phi-1).** Causal gate-state pipeline implemented (38 tests). The in-training Phi-1 prediction (phi correlates r > 0.4 with binding sync_R during training) FAILED across 9 runs / 4 architectures / 2 phi formulations (pyphi + RIIU). Best positive r = +0.089; KomplexNet RIIU produced a substantively significant INVERSE r = -0.11 (p < 1e-100). The 2026-02-21 3-condition synthetic test of phi monotonicity with binding still stands; what is exhausted is the in-training measurement choice. See `docs/preregistered_predictions.md` sections 7-12 and `docs/results/phi1_phaseBalt_2026_05_24.md`. |
| Phi spikes correlated with insight moments | **FAILED first test.** IM-1 verdict 2026-05-14: insight phi mean 1e-04, threshold 5e-04. See `docs/preregistered_predictions.md` section 7. |
| Downward causation (strong emergence) | **Tooling complete.** EI function implemented. Needs experimental validation. |
| Environments require consciousness machinery | **Complete.** DMTS (working memory), WCST (meta-cognition), DQN baseline for comparison. |
| Emotion drives workspace dynamics | **Complete.** Two-stage appraisal (reflex + post-broadcast). Consciousness monitor metric-only. |
| Capsule structure accessible post-broadcast | **Complete.** Structured payloads preserved through GNW broadcast. |
| Phi correlates with strict supervenience test | **Closed for the binding+phi+gate architecture.** The prerequisite in-training Phi-1 prediction is exhausted at 9 runs (sections 7-12). The project pursues emergent consciousness via other measurable signatures (EI causal emergence, DMTS/WCST behavioral integration, phenomenological mapping) and Phase 5 self-representation dynamics, not the binding-phi correlation. |

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

*Updated 2026-03-18. Items 1-2 and 4-7 from original list resolved. Remaining priorities:*

1. ~~**Pre-register Phi/EI predictions.**~~ DONE. See `docs/preregistered_predictions.md`. Defines 3 EI predictions, 3 Phi predictions, 3 insight moment predictions, and a decision protocol.

2. ~~**Define "insight moment" operationally.**~~ DONE. Four-criterion definition in `docs/preregistered_predictions.md` section 3: novel state-action pair, reward jump, first-attempt success, high workspace activity.

3. ~~**NarrativeEngine V1.**~~ DONE. LLM-backed generation via HuggingFace transformers with graceful degradation to template fallback. Coherence tracking via keyword overlap. `NarrativeResult` dataclass with text, coherence score, method, and emotional context.

4. ~~**Rename `QualiaState` to defensible terminology.**~~ DONE. Renamed to `PhenomenologicalState` / `PhenomenologicalMapper`.

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