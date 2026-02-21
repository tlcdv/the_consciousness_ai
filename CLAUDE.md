# UPDATE ALWAYS CLAUDE.md so we know on future sessions what has been done, where we are and what needs to done next.

## Project Guidelines

### Git Workflow
- **Commit and Push**: Always use the following credentials for committing and pushing to the remote repository:
  - **Username**: `tlcdv`
  - **Email**: `zae@todosloscobardesdelvalle.com`
  - **IMPORTANT**: NEVER use "Co-Authored-By: Claude" or similar attribution in commit messages. All commits should be attributed solely to the configured user.

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

## Current State (2026-02-21)

77 of 79 tests pass (97%). Architecture being redesigned based on deep biological research.

**Architectural decisions locked:**
- Oscillatory binding: **AKOrN** (ICLR 2025) replaces hardcoded synchrony multiplier
- Spatial topographic maps: **V-JEPA / DreamerV3 RSSM** as tectum pathway alongside Qwen2-VL
- Compositional hierarchy: **Capsule Networks** (over GNNs)
- Reentrant processing: **5-10 adaptive cycles** with predictive coding convergence
- Spiking validation: **Brian2/NEST** offline, AKOrN in production
- Affective system: **Parallel modulator** (not competing with sensory modules)
- Visual backbone: Qwen2-VL (VideoLLaMA3 deprecated)
- Reward formula: Full PAD with homeostatic arousal term + Dominance
- Strong emergence falsification: EI at gate vs. workspace level (Hoel framework)
- IIT input: must use causal gate states, not workspace bid values

**What fails (1 test):**
- `test_video_llama3_integration.py::test_load_model`. To be deleted when VideoLLaMA3 is removed.

**What's skipped (1 test):**
- `test_predictive_processing.py::test_prediction_generation`: async test needs pytest-asyncio

---

## Roadmap (Revised 2026-02-21)

Based on deep biological research (Feinberg & Mallatt's neural architecture of consciousness) and resolved design questions. Full research document: `docs/biological_neural_architecture_research.md`.

### Tier 1: Core Architecture Changes

1. **Implement AKOrN Oscillatory Binding** (HIGHEST PRIORITY)
   - File: `models/core/oscillatory_binding.py` (NEW)
   - Integrate AKOrN (Artificial Kuramoto Oscillatory Neurons) into workspace layers
   - Replace the hardcoded `1.2` synchrony multiplier in `global_workspace.py`
   - Tech: PyTorch native, available on GitHub

2. **Implement Sensory Tectum with V-JEPA/RSSM**
   - File: `models/core/sensory_tectum.py` (NEW)
   - Add V-JEPA or DreamerV3 RSSM as spatial pathway (topographic mapping)
   - Keep Qwen2-VL for semantic processing
   - Create aligned multisensory maps in latent space

3. **Add Reentrant Processing**
   - File: `models/core/global_workspace.py` (MODIFY) + all specialist modules
   - Add `receive_broadcast(context)` to every specialist module
   - Implement adaptive 5-10 cycle convergence loop with prediction error delta early termination
   - This replaces the single-pass workspace competition

### Tier 2: Architecture Corrections

4. **Redesign Affective System as Parallel Modulator**
   - Remove emotion from workspace competition
   - Implement valence field modulation on sensory bids
   - Add global arousal -> ignition threshold coupling

5. **Implement Phi-Binding Validation Test**
   - File: `tests/test_phi_binding_correlation.py` (NEW)
   - Three conditions: unbound, partially bound, fully bound
   - Must validate before trusting ANY Phi measurements

6. **Add Proprioceptive Self-Model**
   - File: `models/self_model/self_representation_core.py` (MODIFY)
   - Add body schema tensor, somatotopic map, interoceptive state
   - Implement self-other boundary in shared coordinate frame

### Tier 3: Compositional Deepening

7. **Capsule Network Composition Layer**
   - Add CapsNet between tectum outputs and workspace
   - Part-whole routing via dynamic routing by agreement

8. **Brian2 Validation Stack**
   - Offline biological validation of AKOrN binding patterns
   - Compare against biological gamma oscillation models

### Previous Priorities (still relevant)
- Wire up `get_visual_embeddings()` from Qwen2-VL
- Implement ethics filter (AsimovComplianceFilter)
- Full IIT Phi integration with PyPhi
- Delete VideoLLaMA3 integration
- Python 3.10+ upgrade

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

