# Repository map: where everything lives

The single navigation index for this repository. It answers "where is X" for both
contributors and automated tooling. For the conceptual design (pillars, loops, validation)
see [architecture.md](architecture.md); for the reading map of external work see
[aligned_external_resources.md](aligned_external_resources.md). This file is repository
structure only, kept current from the actual tree.

## Top-level directories

| Path | Purpose |
|------|---------|
| `models/core/` | The consciousness pipeline: workspace, tectum, binding, capsules, reentrant processing, gating, world-model objectives |
| `models/emotion/` | Affective core: PAD modulator, homeostatic reward shaping, emotional processing |
| `models/evaluation/` | Metrics and signatures: IIT Phi, Effective Information, Levin metrics, consciousness monitors |
| `models/audio/` | Cochlear auditory pipeline: gammatone filterbank, hair cell, tonotopic encoder, spatial audio, acoustic affect |
| `models/self_model/` | Action selection (basal ganglia), self-representation, working memory, policy variants |
| `models/memory/` | Episodic and emotional memory stores and indices |
| `models/predictive/` | DreamerV3 wrappers and predictive/attention helpers |
| `models/vision_language/` | Qwen2-VL semantic pathway and vision-language helpers |
| `models/narrative/` | Narrative engine (LLM-backed with template fallback) |
| `models/validation/` | Brian2 spiking-network validation of the binding dynamics |
| `models/world_model/` | Standalone generative world-model module |
| `models/perception/`, `models/fusion/`, `models/generative/` | Perception interface, multimodal fusion, generative helpers |
| `simulations/environments/` | Gymnasium environments: Dark Room, Navigation, DMTS, WCST, plus audio mixin and stimulus renderer |
| `simulations/scenarios/` | Scenario generators (consciousness, emotional, ethical, social) |
| `simulations/api/` | Simulation manager |
| `scripts/training/` | Training entry points and the metrics logger |
| `scripts/analysis/` | Probes, diagnostics, and experiment analysis scripts |
| `scripts/demos/` | Visual demos (AKOrN binding) |
| `docs/` | Theory, architecture, roadmap, and design docs |
| `docs/results/` | Dated experiment verdicts (tracked) |
| `tests/` | Test suite (pytest) |
| `configs/` | YAML and Python configuration |
| `unity_scripts/` | C# side-channel scripts for the optional Unity ML-Agents integration |
| `runs/` | Raw training output (gitignored, not in the repo) |

## Core architecture, by pipeline stage (models/core unless noted)

### Perception (the senses)
- `retinotopic_encoder.py` -- DINOv2-B/14 frozen backbone with a conv-stack fallback; retinotopic patch tokens.
- `sensory_tectum.py` -- topographic multisensory map, DreamerV3 RSSM, trimodal inverse-effectiveness fusion, capsule composition. Central perception module.
- `topographic_loss.py` -- TDANN spatial self-organization loss.
- `visual_tectum_projection.py` -- projection helper for the tectum grid.
- `semantic_pathway.py` -- Qwen2-VL embeddings as the 5th workspace competitor.
- `mock_semantic.py` -- deterministic stand-in semantic module (no Qwen2-VL weights needed).
- `models/audio/` -- the auditory pathway: `gammatone_filterbank.py`, `hair_cell_model.py`, `tonotopic_encoder.py`, `spatial_audio.py`, `audio_affect_extractor.py`, `auditory_specialist.py` (the workspace competitor).

### Binding (integration)
- `oscillatory_binding.py` -- AKOrN Kuramoto oscillatory binding (default mechanism).
- `complex_binding.py` -- KomplexNet-style content binding (opt-in, `--binding-mechanism komplex`).
- `binding_attention.py` -- phase-coherence-modulated cross-attention (content-level binding experiment).

### Workspace (consciousness)
- `global_workspace.py` -- the GNW: competition, ignition, reverberation, broadcast, binding integration.
- `capsule_composition.py` -- 4-level hierarchical capsule routing with multi-level reentrant feedback.
- `reentrant_processor.py` -- 5-10 adaptive convergence cycles wrapping the competition.
- `consciousness_gating.py` -- the 5-node gate subsystem (attention, stability, adaptation, coherence, confidence) used as the IIT substrate.
- `gating_components.py` -- gate building blocks.
- `qualia_mapper.py` -- phenomenological-state mapping from workspace content.
- `consciousness_core.py` -- orchestration shell and the AsimovComplianceFilter ethics pipeline (not the training loop).

### World-model objectives (perception frontier, mostly opt-in / default-off)
- `world_model_objective.py` -- world-model objective helpers.
- `rssm_reconstruction.py` -- RSSM latent reconstruction head (R1 perception-fix experiment).
- `tectum_reconstruction.py` -- reconstruction from tectum content (documented negative result).
- `control_representation.py` -- action-conditioned forward model on tectum content (documented negative result).
- `rnd_curiosity.py` -- Random Network Distillation curiosity on the broadcast.

### Emotion (the drives, models/emotion)
- `affective_modulator.py` -- parallel valence field + arousal-threshold coupling + interoceptive PAD generation.
- `reward_shaping.py` -- homeostatic PAD reward formula.
- `emotional_processing.py` -- PAD core with EMA smoothing.
- `tgnn/emotional_graph.py` -- emotional graph network.

### Self-model and action (models/self_model)
- `action_selection_core.py` -- basal-ganglia Go/No-Go pathways with dopamine modulation (default policy).
- `standard_actor_critic.py`, `dqn_policy.py` -- alternative policies for controlled comparison.
- `self_representation_core.py` -- body schema, interoceptive state, capability model, the dynamic self-vector.
- `working_memory_latch.py` -- gated working-memory capture for DMTS (RSSM latch and the obs_map sample memory).
- `match_head.py` -- DMTS match read-out head (experiment).
- `bioelectric_signaling.py`, `holonic_intelligence.py` -- Levin-framework modules (activated as measurement functions).
- `embodiment_core.py`, `self_representation_interface.py`, `networks/feature_networks.py` -- embodiment and feature helpers.

### Metrics and signatures (models/evaluation)
- `iit_phi.py` -- IIT Phi from causal gate states (pyphi + geometric proxy).
- `phi_riiu.py` -- RIIU sliding-window SVD-residual phi (diagnostic, opt-in).
- `effective_information.py` -- Hoel's EI for causal emergence (DEPRECATED 2026-07; see `causal_emergence_svd.py`).
- `causal_emergence_svd.py` -- Causal Emergence 2.0 SVD heuristic (Hoel 2025, arXiv:2503.13395v3); successor to EI, logged via `--log-ce2-every`.
- `causal_emergence.py` -- causal-emergence helpers.
- `perturbational_complexity.py` -- PCI_LZ (Casali et al. 2013), the perturbational
  integration-and-differentiation measure; unlike EI/CE 2.0 it supplies its own variation
  instead of reading a frozen spontaneous trajectory. Driven by `scripts/analysis/probe_pci.py`.
- `coupling_measures.py` -- PLV, phase transfer entropy and phase-amplitude coupling on
  step-indexed signals. Units are CYCLES PER STEP with no Hz reading; see the module
  docstring. Driven by `scripts/analysis/probe_workspace_ordering.py`.
- `levin_consciousness_metrics.py` -- Levin-framework consciousness metrics.
- `consciousness_monitor.py` -- metric-only consciousness evaluation.
- `gnw_metrics.py`, `subjective_testing_suite.py` -- workspace metrics and subjective-test scaffolding.
- Additional evaluation modules exist for development tracking, emotional and memory metrics, and dashboards.

### Memory (models/memory)
- `memory_core.py` -- experience storage and retrieval with emotional context.
- `emotional_memory_core.py`, `emotional_indexing.py` -- FAISS-backed emotional memory index.
- `optimized_store.py`, `optimized_indexing.py` -- memory consolidation manager and hierarchical indices.
- `memory_integration.py`, `memory_interface.py`, `attention_schema.py` -- integration, interface, attention-schema stub.

### World model and predictive (models/predictive, models/world_model)
- `predictive/dreamerv3_wrapper.py`, `predictive/dreamer_emotional_wrapper.py` -- DreamerV3 wrappers.
- `predictive/attention_mechanism.py`, `predictive/emotional_predictor.py`, `predictive/world_model_interface.py` -- predictive helpers.
- `world_model/generative_world_model.py` -- standalone generative world model.

## Simulation

- `simulations/environments/simple_visual_env.py` -- Dark Room (find the light).
- `simulations/environments/navigation_env.py` -- multi-room navigation with fog of war.
- `simulations/environments/dmts_env.py` -- Delayed Match-to-Sample (working memory).
- `simulations/environments/wcst_env.py` -- Wisconsin Card Sort (meta-cognition).
- `simulations/environments/audio_mixin.py` -- procedural audio for the environments.
- `simulations/environments/_stimulus_renderer.py` -- shared numpy stimulus rasterizer.
- `simulations/environments/vr_environment.py`, `interactive_vr_environment.py` -- optional VR scaffolding.
- `simulations/api/simulation_manager.py` -- simulation manager. `simulations/scenarios/` -- scenario generators.

## Scripts

### Training (scripts/training)
- `train_rlhf.py` -- the main cognitive-loop training entry point; exercises the full pipeline. Most CLI flags live here.
- `train_baseline_dqn.py` -- vanilla DQN baseline for controlled comparison.
- `metrics_logger.py` -- TensorBoard + CSV logging, EI computation, insight detection.
- `train_emotion_classifier.py`, `train_vision_model.py` -- auxiliary trainers.

### Analysis and probes (scripts/analysis)
- Perception probes: `probe_collapse_locus.py`, `probe_perception_decodability.py`, `probe_obsmap_variance.py`.
- Working-memory probes: `probe_wm_leakage_free.py`, `probe_match_decode.py`, `decode_choice_records.py`.
- Experiment analysis: `analyze_experiment.py`, `analyze_phi1_retest.py`, `compare_experiments.py`, `compare_phi_pathways.py`, `ablation_report.py`, `aggregate_seeds.py`.
- Phi diagnostics: `diagnose_phi_in_training.py`, `diagnose_phi_zero*.py`, `diagnose_levin_variance.py`, `inspect_verify_run.py`.
- `scripts/demos/demo_akorn_binding.py` -- AKOrN synchronization visualization.

## Documentation index (docs/)

### Theory and grounding
- `feinberg_mallatt_approach.md` -- tectum-first neuroevolutionary blueprint (primary theory).
- `merker_subcortical_consciousness.md` -- independent tectum-first support.
- `rouleau_levin_substrate_independence.md` -- substrate-independence, the 8 aneurocentric themes.
- `metzinger_phenomenal_self_model.md` -- self-model theory and minimal phenomenal experience.
- `damasio_self_hierarchy.md` -- proto/core/autobiographical self hierarchy.
- `affective_consciousness_solms_panksepp.md` -- affective homeostasis grounding for the emergence engine.
- `watanabe_generative_model_approach.md` -- generative-model law and subjective testing.
- `active_inference_unification.md` -- the free-energy unification design for the training objective.
- `generative_world_models_perception.md` -- external world-model options for the perception frontier.
- `biological_neural_architecture_research.md` -- full biological grounding and gap analysis.
- `theory_of_consciousness.md`, `theory_implementation_review.md` -- theory basis and alignment audit.
- `consciousness_indicators_butlin.md` -- the indicator-property evaluation rubric.

### Design, roadmap, and reference
- `architecture.md` -- system design overview (pillars, loops, validation).
- `roadmap.md` -- the authoritative roadmap and phase status.
- `preregistered_predictions.md` -- pre-registered EI/Phi/insight predictions and verdicts.
- `iit_implementation_roadmap.md` -- Phi computation strategy.
- `isomorphic_visual_mapping_research.md`, `auditory_system_design.md`, `levin_metrics_grounding.md` -- component design docs.
- `aligned_external_resources.md` -- curated reading map of aligned external work.
- `ethics_framework.md`, `simulation_guide.md`, `memory_system.md`, `pipeline.md`, `installation.md`, `datasets.md`, `contributing.md` -- supporting docs.
- `decisions/` -- dated decision records (license and design decisions).

### Experiment results (docs/results/)
Dated verdict documents for each experiment (ablation campaigns, Phi-1 retests, perception
collapse, DMTS match head, RIIU comparisons, agent competence, and more). These are the
tracked record of what was run and what it showed. Raw run output stays in `runs/`
(gitignored).

## Where experiments live

- Verdicts and analysis: `docs/results/` (tracked, one dated markdown per experiment).
- Raw training output (metrics.csv, episodes.csv, checkpoints): `runs/` (gitignored, local only).
- To reproduce, the training command is in the relevant `docs/results/` verdict and uses
  `scripts/training/train_rlhf.py` with the flags recorded there.
