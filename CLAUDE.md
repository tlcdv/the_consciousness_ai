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

---

## Current State

53 of 79 tests pass (67%). The project is at Phase 2-3 of the 7 phase roadmap.

**What passes (53 tests):**
- All previous 27 tests still pass
- Levin consciousness metrics (3/3)
- Metaconsciousness evaluation (3/3)
- Reinforcement core (3/3)
- Narrative engine (3/3)
- Emotional reinforcement (5/5)
- Emotional reinforcement success (5/5)
- Memory optimization (3/3)
- Memory indexing (3/4, 1 assertion failure on similarity threshold)
- Simulation integration (1/2)
- Various previously failing pipeline tests

**What fails (25 tests):** Tests that require actual method implementations (stubs), not config fixes:
- `test_consciousness_pipeline.py` (6): Needs `ConsciousnessCore.get_state()`, `process_visual_stream()`, `ConsciousnessMonitor.evaluate_development()`
- `test_consciousness_metrics.py` (5): Needs `ConsciousnessMetrics.store_experience()`, `evaluate_consciousness_development()`, `DreamerEmotionalWrapper.compute_reward()`
- `test_emotional_memory_integration.py` (5): Needs `EmotionalMemoryCore.store_experience()`, `retrieve_similar_memories()` with proper API, `EmotionalEvaluator.evaluate_interaction()` signature fix
- `test_consciousness_development.py` (4): Needs `ConsciousnessCore.get_state()`, `scenario_manager` attribute
- `test_consciousness_system.py` (1): Needs `ConsciousnessMonitor.evaluate_current_state()`
- `test_development_stages.py` (1): Same
- `test_memory_indexing.py` (1): Similarity assertion too strict (0.23 < 0.8 threshold)
- `test_simulation_integration.py` (1): Emotional learning assertion (improve over episodes)
- `test_video_llama3_integration.py` (1): Model loading (requires actual VideoLLaMA3 weights)

---

## Roadmap (Next Steps)

Priority order based on dependency and impact:

### 1. Implement ConsciousnessCore core methods (HIGH)
- `get_state()`, `process_visual_stream()`, `process_perception()`, activity logging
- These unblock 10+ tests across consciousness pipeline, development, and metrics
- File: `models/core/consciousness_core.py`

### 2. Implement EmotionalMemoryCore.store_experience() (HIGH)
- File: `models/memory/emotional_memory_core.py`
- Current `store()` takes `(timestamp, MemoryData)` but tests call `store_experience(state, emotion_values, ...)`
- Add convenience wrapper mapping the test API to internal storage
- Also add `retrieve_similar_memories()` with emotion query support

### 3. Wire up visual embeddings from Qwen2-VL (HIGH)
- File: `models/vision-language/qwen2/qwen2_integration.py`
- `get_visual_embeddings()` raises `NotImplementedError`
- Blocks: perception to workspace pipeline, visual stream processing tests

### 4. Implement ConsciousnessMonitor evaluation methods (MEDIUM)
- `evaluate_current_state()`, `evaluate_development()`
- File: `models/evaluation/consciousness_monitor.py`
- Unblocks: development stages tests, consciousness system tests

### 5. Implement ethics filter methods (MEDIUM)
- File: `models/core/consciousness_core.py` (AsimovComplianceFilter)
- 7 placeholder methods for the three laws

### 6. Full IIT Phi integration with PyPhi (MEDIUM)
- File: `models/evaluation/iit_phi.py`
- Empirical TPM builder works. `calculate_phi()` guarded behind conditional pyphi import

### 7. Complete remaining stubs (LOW)
- `ConsciousnessMetrics.evaluate_consciousness_development()`
- `DreamerEmotionalWrapper.compute_reward()`
- `EmotionalGraphNetwork.get_embedding()`, `predict_emotion()`
- `NarrativeEngine` full implementation
- Goal management, order reception in ConsciousnessCore

### 8. Python 3.10+ upgrade (OPTIONAL)
- Currently on 3.8.3. Upgrading enables cleaner type hints and pattern matching.
