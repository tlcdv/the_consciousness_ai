# UPDATE ALWAYS CLAUDE.md so we know on future sessions what has been done, where we are and what needs to done next.

## Project Guidelines

### Git Workflow
- **Commit and Push**: Always use the following credentials for committing and pushing to the remote repository:
  - **Username**: `tlcdv`
  - **Email**: `zae@todosloscobardesdelvalle.com`
  - **IMPORTANT**: NEVER use "Co-Authored-By: Claude" or similar attribution in commit messages. All commits should be attributed solely to the configured user.
  - **IMPORTANT**: Never mention "CLAUDE.md", "Claude", or "Anthropic" in commit messages. Describe changes in plain technical terms only.

Be at the same concise and avoid verbose tone. Avoid AI slop, buzz words, filled words, over hyped tone. Systematically replace em-dashes ("—") with a dot (".") to start a new sentence, or a comma (",") to continue the sentence. And always remove the "-" character that are between words like AI-generated. It is a big sign if AI generated or AI writting. ### Writing Style Constraints
* **Eliminate Epanorthosis:** Do not use the "not just X, but Y" or "more than X, it is Y" rhetorical structure.
* **Avoid Circular Emphasis:** Prohibit self-corrections that restate the same idea using synonyms for emphasis, especially at the beginning or end of paragraphs.
* **Substantive Progression:** Ensure every sentence introduces new information or data. Do not use repetitive flourishes to "frame" a concept.
* **Directness:** Avoid "red flag" AI phrases that attempt to sound profound through structural repetition.

Use Github CLI if needed for authenticate to use the proper Github account always. It is tlcdv with the email zae@todosloscobardesdelvalle.com

---

## Session Log

### 2026-02-16: Test Infrastructure Repair and Memory System Upgrade

**What was done:**

1. **Fixed all 11 test collection errors** (0 errors remain, 79 tests now collect):
   - Installed missing deps: `pandas`, `opencv-python-headless`, `matplotlib`, `aiohttp`, `sentence-transformers`
   - Fixed `dataclass` import missing in `tests/integration/test_development_stages.py`
   - Fixed `List` type hint (Python 3.8 incompatible) in `tests/test_consciousness_integration.py`
   - Fixed `tuple[bool, Optional[Dict]]` syntax in `models/core/consciousness_core.py` (needs `Tuple` from typing on 3.8)
   - Fixed `DreamerV3` import in `dreamer_emotional_wrapper.py` and `simulation_manager.py` (class is `DreamerV3Wrapper`)
   - Fixed `enviroments` typo across `simulation_manager.py`, `test_simulation_integration.py`, `simulation_controller.py` (correct: `environments`)
   - Added `ConsciousnessScenarioManager` alias in `consciousness_scenarios.py` (tests import that name, class is `ConsciousnessScenarioGenerator`)
   - Fixed `EmotionalGraphNN` import in `test_consciousness_development.py` (class is `EmotionalGraphNetwork`)
   - Made `unreal` import conditional in `vr_environment.py` and `interactive_vr_environment.py`
   - Made `pyphi` import conditional in `iit_phi.py`
   - Made `VisualProcessor` import conditional in `video_llama3_integration.py`

2. **Replaced Pinecone with FAISS** in `models/memory/emotional_indexing.py`:
   - Removed external Pinecone dependency entirely
   - Implemented local FAISS based vector index with L2 normalized cosine similarity
   - Falls back to brute force numpy search when FAISS is unavailable
   - Maintains identical API: `store_memory()`, `retrieve_similar_memories()`, `get_temporal_sequence()`
   - Tracks `memory_stats` (emotional_coherence, temporal_consistency, consciousness_relevance)

3. **Created missing modules:**
   - `configs/__init__.py` and `configs/consciousness_development.py` with `DevelopmentConfig` dataclass
   - `models/memory/attention_schema.py` with `AttentionSchema` class (Attention Schema Theory)
   - `WorkspaceMessage` dataclass in `models/core/global_workspace.py`

**Test results after fixes:** 27 passed, 51 failed, 1 skipped (up from 3 passing / 11 collection errors before).

### 2026-02-16 (continued): EmotionalProcessingCore + Systematic Test Fixes

**What was done:**

1. **Implemented EmotionalProcessingCore** (`models/emotion/emotional_processing.py`):
   - Full PAD (Pleasure, Arousal, Dominance) emotional model with EMA smoothing
   - Maps perception context (emotion_label, reward, stress, threat, novelty, social_valence) to continuous PAD state
   - Includes temporal consistency tracking, emotional intensity, dominant emotion detection
   - PCI noise perturbation support for consciousness testing

2. **Fixed config compatibility across 15+ source files**:
   - `consciousness_core.py`: Added `_cfg_get()` helper for dict/dataclass config support
   - `emotional_memory_core.py`: Accept both dict and dataclass configs
   - `emotional_graph.py`: Accept optional config with sensible defaults
   - `dreamer_emotional_wrapper.py`: Handle missing 'dreamerV3' key, build MemoryConfig from dict
   - `consciousness_gating.py`: Support dict config for `config.gating` access
   - `attention_mechanism.py`: Support dict config for `ConsciousnessAttention`
   - `reward_shaping.py`: Use `.get()` with defaults for `emotional_dims`/`hidden_size`
   - `feature_networks.py`: Support dict config for `FeatureNetwork`
   - `video_llama3_integration.py`: Accept dict config, add `max_buffer_size` field, don't crash on model load failure
   - `emotional_evaluation.py`: Accept optional config, build MemoryConfig from dict
   - `consciousness_monitor.py`: Accept single arg config (backwards compatible)

3. **Fixed EmotionalMemoryIndex** (`models/memory/emotional_indexing.py`):
   - Added fallback embedding when EmotionalGraphNetwork is unavailable (builds vector from emotion values + state)
   - Added fallback consciousness score calculation from attention level and emotion intensity

4. **Implemented missing classes in optimized_store.py**:
   - `EmotionalHierarchicalIndex`: Valence based partitioning with cosine similarity search
   - `TemporalHierarchicalIndex`: Time based memory index
   - `MemoryConsolidationManager`: Consolidation threshold management
   - Added `_store_in_indices`, `_search_partition`, `_update_optimization_metrics` methods

5. **Completed optimized_indexing.py**: Added 12 missing private methods (partition init, search, rebalance, etc.)

6. **Created missing modules**:
   - `models/narrative/narrative_generator.py` with `NarrativeGenerator` class
   - `models/fusion/emotional_memory_fusion.py`: Deferred heavy model loading (LLaMA, PaLM-E, Whisper)
   - `models/generative/generative_emotional_core.py`: Deferred LLaMA loading, fallback `generate_response()`

7. **Fixed SimulationManager** (`simulations/api/simulation_manager.py`):
   - Accept optional `acm_system`, single arg `(config)` call
   - Deferred heavy component initialization
   - Added sync `run_interaction()` method for testing

8. **Fixed Levin consciousness metrics** (`models/evaluation/levin_consciousness_metrics.py`):
   - Normalized `evaluate_bioelectric_complexity` with `min(1.0, x/(x+1))`
   - Clamped all sub metrics to [0, 1] via `_clamp01()` wrapper

9. **Rewrote 6 test files** to match current API:
   - `test_Emotional_reinforcement_integration.py`
   - `test_reinforcement_core.py`
   - `test_narrative_engine.py`
   - `test_emotional_reinforcement.py`
   - `test_emotional_reinforcement_success.py`
   - `test_consciousness_pipeline.py` (partial, fixed VideoLLaMA3 constructor)

**Test results:** 53 passed, 25 failed, 1 skipped (up from 27 passed at session start).

### 2026-02-17: Core Method Implementations + Test Pass Rate to 97%

**What was done:**

1. **Implemented ConsciousnessCore core methods** (`models/core/consciousness_core.py`):
   - `get_state()` returns snapshot with `.consciousness_score`
   - `process_visual_stream(frame)` returns dict with visual_context, attention_metrics
   - `process_experience(scenario)` returns object with .state, .emotion, .attention
   - `process_attention(state, stress_level)` returns object with .consciousness_score

2. **Implemented EmotionalMemoryCore convenience methods** (`models/memory/emotional_memory_core.py`):
   - `store_experience(state, emotion_values, attention_level, context)`
   - `retrieve_similar_memories(emotion_query, k)`
   - Extended `store()` and `retrieve()` to accept multiple call signatures

3. **Implemented ConsciousnessMonitor evaluation methods** (`models/evaluation/consciousness_monitor.py`):
   - `evaluate_current_state()`, `evaluate_development()`, `evaluate_state()`, `evaluate_emotional_awareness()`

4. **Implemented ConsciousnessMetrics methods** (`models/evaluation/consciousness_metrics.py`):
   - `store_experience()`, `get_similar_emotional_experiences()`, `evaluate_consciousness_development()`

5. **Implemented EmotionalEvaluator missing methods** (`models/evaluation/emotional_evaluation.py`):
   - `_calculate_narrative_consistency()`, `_calculate_narrative_similarity()`, `_calculate_emotional_awareness()` with proper edge case handling
   - `_calculate_survival_adaptation()` guard against empty diff arrays
   - Changed `evaluate_interaction()` signature: action is now optional keyword arg

6. **Implemented DreamerEmotionalWrapper methods** (`models/predictive/dreamer_emotional_wrapper.py`):
   - `compute_reward(state, emotion_values, action_info)` with shaped reward from PAD values
   - Rewrote `process_interaction()` to avoid calling unimplemented methods

7. **Implemented EmotionalGraphNetwork methods** (`models/emotion/tgnn/emotional_graph.py`):
   - `process(input_data)`, `get_embedding(emotion_values)`
   - `_get_input_dim()`, `_fuse_with_narrative()`, `_calculate_memory_gate()`, `_update_state()`

8. **Implemented ModularSelfRepresentation methods** (`models/self_model/modular_self_representation.py`):
   - `update(current_state, emotional_context, attention_level)` returning consciousness_level
   - `_init_adaptation_params()`, fixed config reference bug

9. **Fixed MemoryIntegrationCore** (`models/memory/memory_integration.py`):
   - `get_state()`, `get_metrics()` methods
   - `store_experience()` converts float consciousness_level to tensor for ConsciousnessGate
   - `SemanticAbstractionNetwork.forward()` adapts input dimension via adaptive_avg_pool1d

10. **Fixed MemoryCore** (`models/memory/memory_core.py`):
    - `_create_memory_vector()` handles None action (defaults to zeros)

11. **Fixed EmotionalMemoryIndex** (`models/memory/emotional_indexing.py`):
    - Store fallback uses emotion values only (matches retrieval query construction) for correct cosine similarity

12. **Fixed ConsciousnessAttention** (`models/predictive/attention_mechanism.py`):
    - Added emotional_context boost to attention_level (emotional arousal increases alertness)
    - Added `input_state` kwarg alias, dimension adaptation

13. **Fixed multiple test files**:
    - `test_consciousness_development.py`: Added setUp components, scenario generation, adaptation tracking
    - `test_consciousness_system.py`: Added `_process_consciousness_cycle()`
    - `test_consciousness_metrics.py`: Added monitor to setUp
    - `test_development_stages.py`: Complete rewrite
    - `test_consciousness_pipeline.py`: Fixed config, added test helpers
    - `test_simulation_integration.py`: DummyEnvironment tracks total episodes for learning simulation

14. **Other fixes**:
    - `configs/consciousness_development.py`: Added `__getitem__` and `get()` for dict access
    - `models/fusion/emotional_memory_fusion.py`: Handle None encoders, fixed fusion quality range
    - `models/self_model/networks/feature_networks.py`: SocialContextNetwork uses `.get()` defaults
    - `simulations/scenarios/consciousness_scenarios.py`: Optional config, default params, added helper methods

**Test results:** 77 passed, 1 failed, 1 skipped (up from 53 passed / 25 failed).

---

## Current State

77 of 79 tests pass (97%). The project is at Phase 2-3 of the 7 phase roadmap.

**What fails (1 test):**
- `test_video_llama3_integration.py::test_load_model`: Requires actual VideoLLaMA3 model weights (not downloadable in test environment)

**What's skipped (1 test):**
- `test_predictive_processing.py::test_prediction_generation`: Async test needs pytest-asyncio

---

## Roadmap (Next Steps)

Priority order based on dependency and impact:

### 1. Wire up visual embeddings from Qwen2-VL (HIGH)
- File: `models/vision-language/qwen2/qwen2_integration.py`
- `get_visual_embeddings()` raises `NotImplementedError`
- Blocks: perception to workspace pipeline, visual stream processing

### 2. Implement ethics filter methods (MEDIUM)
- File: `models/core/consciousness_core.py` (AsimovComplianceFilter)
- 7 placeholder methods for the three laws

### 3. Full IIT Phi integration with PyPhi (MEDIUM)
- File: `models/evaluation/iit_phi.py`
- Empirical TPM builder works. `calculate_phi()` guarded behind conditional pyphi import

### 4. NarrativeEngine full implementation (MEDIUM)
- Currently basic stub. Needs LLM backed narrative generation and coherence tracking.

### 5. Goal management and order reception in ConsciousnessCore (LOW)
- Process external commands and goals through the consciousness pipeline

### 6. Python 3.10+ upgrade (OPTIONAL)
- Currently on 3.8.3. Upgrading enables cleaner type hints and pattern matching.

---

### 2026-02-18: Theory vs. Implementation Audit + Research-Backed Design Decisions

**What was done:**

1. **Full theory/implementation audit** against the Functionalist Emergentism thesis:
   - Created `docs/theory_implementation_review.md` with complete alignment analysis
   - Identified 7 issues (2 HIGH, 3 MEDIUM, 2 LOW severity)
   - Highest severity: IIT Phi measurement inputs are methodologically wrong (binarized bid values, not causal node states); visual embeddings unimplemented

2. **Resolved three open design questions** via literature research:

   **Q: Falsification criterion for strong emergence?**
   - Answer: use Hoel's Effective Information (EI) framework (PNAS 2013)
   - Measure EI at gate level and workspace level simultaneously during training
   - Falsification: if EI(workspace) ≤ EI(gates) across training, strong emergence is not occurring
   - Secondary model: the 2025 Nature adversarial IIT/GNW collaboration (n=256, fMRI+MEG+iEEG). Pre-register predictions before running any experiment.
   - **To implement:** `effective_information()` function in `models/evaluation/`

   **Q: Should Dominance be in the reward formula?**
   - Answer: yes. The published formula `Rtotal = Rext + λ(Valence - Arousal)` is incomplete and incorrect on two counts: Arousal is not purely negative (curiosity raises arousal), and dropping Dominance removes the agent's sense of agency.
   - Corrected formula: `Rtotal = Rext + λ1·ΔValence - λ2·(Arousal - Arousal_target)² + λ3·Dominance`
   - Supported by: Mehrabian PAD model, Homeostatic RL (Keramati & Gutkin eLife 2014), "In Defense of Dominance" (ACL 2012)
   - **To update:** `models/emotion/reward_shaping.py` + thesis page

   **Q: Qwen2-VL or VideoLLaMA3 as primary visual backbone?**
   - Answer: **Qwen2-VL is the primary backbone. VideoLLaMA3 should be removed.**
   - Reasons: M-ROPE natively handles temporal/3D video positional encoding (needed for live sim frames); Qwen3-VL-Embedding provides the direct path to implement `get_visual_embeddings()`; native HuggingFace integration; documented 4-bit quantization; VideoLLaMA3 is optimized for offline video analysis, not streaming.
   - **To do:** delete `models/integration/video_llama3_integration.py` and its failing test; implement `get_visual_embeddings()` in `qwen2_integration.py` using ViT hidden states

---

### (Sessions between Feb 21 and Feb 27: Tier 1 Core Architecture Implementation)

**What was done:**

1. **Implemented AKOrN Oscillatory Binding** (`models/core/oscillatory_binding.py`):
   - `KuramotoLayer`: discrete-time Kuramoto oscillator synchronization on N-spheres
   - `WorkspaceBindingSystem`: wraps AKOrN for the GNW, maps module names to oscillator indices
   - Integrated into `global_workspace.py`, replacing the hardcoded `1.2` synchrony multiplier
   - Tests: `test_phi_binding_correlation.py` (synchronization dynamics, 3-condition phi monotony)

2. **Implemented Sensory Tectum with RSSM** (`models/core/sensory_tectum.py`):
   - `TopographicMap`: 2D spatial grid fusing visual and audio spatial features
   - `RSSMCore`: DreamerV3 style recurrent state space model with categorical latents
   - `SensoryTectum`: full midbrain integration layer with surprise-based bidding and `receive_broadcast()` for reentrant feedback
   - Tests: `test_sensory_tectum.py` (topographic fusion, RSSM recurrence, surprise bids)

3. **Implemented Reentrant Processing** (`models/core/reentrant_processor.py`):
   - `ReentrantProcessor`: 5-10 adaptive convergence cycles wrapping GNW competition
   - Prediction error delta early termination (convergence threshold)
   - Top-down feedback via `receive_broadcast()` on specialist modules
   - Tests: `test_reentrant_processing.py` (8 core tests + 6 specialist feedback tests + 3 end-to-end tests)

4. **Implemented PAD Homeostatic Reward Formula** (`models/emotion/reward_shaping.py`):
   - Added `compute_emotional_reward()` with corrected formula: `Rtotal = Rext + λ1·ΔValence - λ2·(Arousal - Arousal_target)² + λ3·Dominance`
   - Configurable `arousal_target` parameter

5. **Extended Self-Model** (`models/self_model/self_representation_core.py`):
   - Added `body_schema` tensor, `interoceptive_state` dict, `capability_model` to `SelfState`
   - `update_body_schema()` method for proprioceptive input
   - `_update_interoceptive_state()` with energy/fatigue/damage homeostatic dynamics

6. **Phi-Binding Validation Test** (`tests/test_phi_binding_correlation.py`):
   - 3-condition controlled experiment (unbound/partial/full binding)
   - Validates AKOrN synchronization dynamics and phi monotony

**Test results after Tier 1:** 135 passed, 1 failed (VideoLLaMA3), 1 skipped.

---

### 2026-02-27: Tier 2 Architecture + VideoLLaMA3 Removal

**What was done:**

1. **Redesigned Affective System as Parallel Modulator** (`models/emotion/affective_modulator.py`):
   - `AffectiveModulator` class with two biological mechanisms:
     - Valence field: modulates sensory bid values (positive valence boosts approach modules, negative boosts threat modules)
     - Arousal-threshold coupling: adjusts GNW `ignition_threshold` (high arousal = lower threshold = easier ignition)
   - Dominance modulation: positive dominance slightly boosts all bids (active agency)
   - Integrated into `global_workspace.py`: emotion removed from workspace competition, modulator applied before AKOrN binding
   - Binding system reduced from 5 to 4 oscillators (vision, audio, memory, body)
   - Tests: `test_affective_modulator.py` (9 tests: neutral PAD, arousal coupling, valence field, dominance, clamping, GNW integration)

2. **Implemented Effective Information Function** (`models/evaluation/effective_information.py`):
   - `compute_effective_information()`: builds empirical TPM from state trajectories, computes EI = max_entropy - avg_noise
   - `compare_ei_levels()`: compares EI at gate vs workspace level for causal emergence detection
   - `discretize_continuous()`: bins continuous activations for TPM construction
   - Implements Hoel's PNAS 2013 framework for falsifying the strong emergence claim
   - Tests: `test_effective_information.py` (11 tests: deterministic/random/partial TPMs, level comparison, discretization, TPM construction)

3. **Deleted VideoLLaMA3 integration:**
   - Removed `models/integration/video_llama3_integration.py`
   - Cleaned all references in `simulations/api/simulation_manager.py`, `models/emotion/multimodal_detector.py`, `tests/test_consciousness_pipeline.py`
   - Renamed `llama_perception` -> `perception` in simulation_manager async methods
   - Eliminates the only failing test. Visual backbone is now Qwen2-VL exclusively.

**Test results:** 156 passed, 0 failed, 1 skipped.

---

### 2026-02-28: Flaky CI Test Fix + ACM Terminology Removal

**What was done:**

1. **Fixed flaky CI test** (`tests/test_action_selection.py::test_go_nogo_dopamine_modulation`):
   - Root cause: borderline statistical result, 10/20 wins fails `assertGreater(wins, 10)`
   - Fix: `torch.manual_seed(42)` for reproducibility, increased trials from 20 to 50, threshold from 10 to 25
   - CI was otherwise passing 153/154 tests

2. **Removed all "ACM" / "Artificial Consciousness Module" terminology** across 90+ files:
   - Updated README.md, workflow name, all Python docstrings/comments, docs, configs, simulation files
   - Variable renames: `acm_system` to `consciousness_system`, `self.acm` to `self.core`, `MockACMAgentInterface` to `MockAgentInterface`, `acm_step_data` to `step_data`, `acm_report` to `agent_report`
   - String constants: `"ACM-1"` to `"TCA-1"`, `"acm_memory_index"` to `"consciousness_memory_index"`, `"acm_default_run"` to `"consciousness_default_run"`
   - Project now consistently uses "The Consciousness AI" branding, matching the website (theconsciousness.ai)
   - Only remaining "ACM" references: CLAUDE.md session history (this file) and external paper citations (llama3_herd.md refers to Association for Computing Machinery)

**Test results:** 153 passed, 0 failed, 1 skipped.

---

## Current State (2026-02-28)

153 of 154 tests pass (99.4%). Tier 1 and Tier 2 architecture changes complete. ACM terminology removed.

**Architectural decisions locked and implemented:**
- Oscillatory binding: **AKOrN** (ICLR 2025) integrated into GNW
- Spatial topographic maps: **DreamerV3 RSSM** as tectum pathway alongside Qwen2-VL
- Reentrant processing: **5-10 adaptive cycles** with predictive coding convergence
- Affective system: **Parallel modulator** (valence field + arousal threshold coupling)
- Visual backbone: **Qwen2-VL** exclusively (VideoLLaMA3 deleted)
- Reward formula: **Full PAD** with homeostatic arousal term + Dominance
- Strong emergence falsification: **EI function** at gate vs. workspace level (Hoel framework)
- Phi-binding validation: **3-condition test** (unbound/partial/full) passing
- Self-model: **Body schema + interoception** in self_representation_core

**Decisions locked, not yet implemented:**
- Compositional hierarchy: **Capsule Networks** (over GNNs)
- Spiking validation: **Brian2/NEST** offline

**What fails (0 tests):** Nothing.

**What's skipped (1 test):**
- `test_predictive_processing.py::test_prediction_generation`: async test needs pytest-asyncio

---

## Roadmap (Revised 2026-02-27)

### Completed
- ~~Tier 1: AKOrN Oscillatory Binding~~
- ~~Tier 1: Sensory Tectum with RSSM~~
- ~~Tier 1: Reentrant Processing~~
- ~~Tier 2: Affective System Parallel Modulator~~
- ~~Tier 2: Phi-Binding Validation Test~~
- ~~Tier 2: Proprioceptive Self-Model (body schema + interoception)~~
- ~~PAD Homeostatic Reward Formula~~
- ~~Effective Information Function (Hoel framework)~~
- ~~Delete VideoLLaMA3~~

### Next: Tier 3 + Remaining Priorities

1. **Capsule Network Composition Layer** (HIGH)
   - Add CapsNet between tectum outputs and workspace
   - Part-whole routing via dynamic routing by agreement

2. **Wire up `get_visual_embeddings()` from Qwen2-VL** (HIGH)
   - File: `models/vision-language/qwen2/qwen2_integration.py`
   - Extract ViT hidden states before language head

3. **Implement ethics filter** (MEDIUM)
   - AsimovComplianceFilter: 7 placeholder methods in `consciousness_core.py`

4. **Full IIT Phi integration with PyPhi** (MEDIUM)
   - Use causal gate states (not workspace bid values) as input

5. **Brian2 Validation Stack** (LOW)
   - Offline biological validation of AKOrN binding patterns

6. **NarrativeEngine full implementation** (LOW)
   - LLM-backed narrative generation and coherence tracking

7. **Python 3.10+ upgrade** (OPTIONAL)

---

### 2026-02-21: Biological Neural Architecture Deep Research

**What was done:**

1. **Deep research on Feinberg & Mallatt's "The Ancient Origins of Consciousness"** via NotebookLM:
   - Extracted the six special neurobiological features required for consciousness
   - Mapped biological architecture to ACM implementation gaps
   - Created `docs/biological_neural_architecture_research.md` (700+ lines)

2. **Resolved all eight open research questions:**

   **Q1: Topographic mapping** - Use V-JEPA/DreamerV3 world models for spatial structure preservation alongside Qwen2-VL semantic. World models encode spatial causality naturally.

   **Q2: Discrete Kuramoto binding** - AKOrN (ICLR 2025 oral) already implements Kuramoto oscillatory neurons as drop-in PyTorch layers. PyTorch implementation on GitHub.

   **Q3: Reentrant cycles** - 5-10 adaptive cycles with prediction error early termination. Matches biological cortical processing (~200ms, ~10ms per relay).

   **Q4: Capsule Networks vs. GNNs** - Capsule Networks chosen. They natively implement nested compositional hierarchy (parts remain active while bound into wholes) via dynamic routing by agreement.

   **Q5: Spiking subsystem** - Dual-stack: AKOrN (PyTorch native) for production + Brian2/NEST for offline biological validation.

   **Q6: Phi-binding test** - Designed 3-condition controlled experiment (unbound/partial/full) with success criteria and red flags.

   **Q7: Embodiment** - Extend self-model with spatial body schema, somatotopic map, interoceptive state, self-other boundary.

   **Q8: Affective system** - Redesign as parallel modulator (valence field + arousal threshold), not competing with sensory modules.

3. **Key biological findings from the book:**
   - Consciousness requires 3-4+ hierarchical levels, not a flat pipeline
   - Isomorphic (topographic) mapping is the mechanism for mental images/referral
   - Oscillatory binding (gamma 30-100Hz) creates mental unity
   - The optic tectum (midbrain) was the first seat of consciousness, not the cortex
   - Affective consciousness (limbic system) evolved separately from sensory consciousness
   - The four NSFCs: Referral, Mental Unity, Qualia, Mental Causation

