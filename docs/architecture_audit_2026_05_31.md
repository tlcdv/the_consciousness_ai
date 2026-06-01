# Architecture coherence audit (2026-05-31)

Read-only audit prompted by a direct question: is the architecture coherent, is
there a simpler/more elegant way, is everything legit. Every claim below is tied
to a file/line or a disk-loaded number. No code was changed to produce this.

## Headline verdict

- **Integrity: legit.** The empirical record is honest (FAILED-first results,
  numbers from disk, 649 tests passing, all commits authored as `tlcdv`, no
  fabrication). Nothing in the record is fabricated or overstated.
- **Architecture: real coherence debt.** A small coherent core
  (perception -> workspace -> action, with phi/sync_R partially wired into the
  learning signal) is buried under accretion: three parallel "agent"
  orchestrators, a fragmented self-model (6 of 13 `self_model` files orphaned),
  several untrained/measurement-only "consciousness" modules, and a base agent too
  weak to exercise the harder tasks. The more elegant path is consolidation, not
  more features.

## 1. Component wiring map

### Three parallel top-level "agents" (only one is the research loop)

| Entry point | Used by | Status |
|-------------|---------|--------|
| `scripts/training/train_rlhf.py` | the actual research training loop | **canonical**; wires components directly, uses neither orchestrator below |
| `models/core/consciousness_core.py` `ConsciousnessCore` | `simulations/api/simulation_manager.py:31`, `models/predictive/emotional_predictor.py:18` | orchestrator shell; CLAUDE.md documents it as not used in training |
| `models/agent/consciousness_agent.py` `ConsciousnessAgent` | `scripts/training/train_emotional_agent.py:10` | **deprecated** (CLAUDE.md 2026-04-06); parallel/dead track |

### Components inside `train_rlhf.py`, by role

- **Drives behaviour:** `SensoryTectum`, `GlobalWorkspace`, `ReentrantProcessor`,
  `AffectiveModulator`, `EmotionalRewardShaper`, `ActionSelectionCore` (the policy).
- **Wired into the learning signal:** `ConsciousnessGate` (phi -> intrinsic reward +
  exploration knob), workspace binding (`sync_R` -> `workspace_optimizer`),
  `RNDCuriosity` (-> intrinsic reward), `reward_predictor` (trains the tectum).
- **Measurement-only / logged:** EI (`compute_and_log_ei`), Levin metrics,
  `self_pred_skill`. (RIIU is an optional reward source behind `--enable-riiu`,
  else logged.)
- **Instantiated-but-inert / untrained:** `HolonicSystem` +
  `BioelectricSignalingNetwork` + `LevinConsciousnessEvaluator` (inference-mode,
  untrained), `SelfVectorModule` (trains by its own loss; behavioural path only via
  gate -> phi, which the WCST gating ablation showed gives no benefit),
  `MockSemanticModule` (deterministic stub), `SemanticPathway` (zero bid without
  Qwen2-VL).

### `models/self_model/` (13 files) classification

- **Wired in training:** `self_representation_core.py` (with its inline
  `DirectExperienceLearner` / `MetaLearningModule` / `SocialLearningNetwork`),
  `action_selection_core.py`, and `holonic_intelligence.py` +
  `bioelectric_signaling.py` (only as untrained diagnostics under
  `--enable-levin-metrics`).
- **Used only by non-training shells:** `self_representation_interface.py`
  (`ConsciousnessCore`), `embodiment_core.py` (deprecated `ConsciousnessAgent`).
- **Orphaned (no production importer found outside the cluster/tests):**
  `modular_self_representation.py`, `belief_system.py`, `meta_learner.py`,
  `meta_learning.py`, `intention_tracker.py`, `emotion_context_tracker.py`.
  (`modular_self_representation` -> `belief_system` is a self-contained orphan
  cluster; `intention_tracker` / `emotion_context_tracker` have zero importers.)

## 2. Causal pathways to action (what actually drives behaviour)

- Action: `action_core.select_action(broadcast, emotion_arousal=arousal * phi_exploration_scale)`
  ([train_rlhf.py:871](../scripts/training/train_rlhf.py#L871)), where
  `phi_exploration_scale = max(0.2, 1 - phi*10)`. So behaviour = f(broadcast,
  arousal, phi-scaled exploration).
- Learning signal: `reward_for_action_core = env_reward + 0.5*delta_phi + 0.1*curiosity`
  ([train_rlhf.py:1013-1014](../scripts/training/train_rlhf.py#L1013)). phi-delta and
  curiosity are genuinely in the policy reward.
- Binding: `workspace_optimizer` loss `= -reward * sync_R` (sync_R wired to learning).
- The tectum + gate are trained by a reward-prediction MSE + gate diversity loss
  (auxiliary), not by the policy reward; the policy (`action_core.update_policy`)
  is trained separately.

**Honest read:** phi and sync_R are causally wired into learning, but the pathways
are thin (a 0.5x phi-delta reward bonus and an exploration knob). EI, Levin, and
self_pred are measurement-only. The self-vector is behaviourally inert as wired.
So the system is partially integrated, mostly instrumented.

## 3. Self-model fragmentation + consolidation proposal

Of 13 `self_model` files, ~5 are wired (2 only as untrained diagnostics), 2 serve
only non-training shells, and 6 are orphaned. There are three self-representation
variants (`self_representation_core`, `modular_self_representation`,
`self_representation_interface`) and two standalone meta-learning files plus an
inline one. The new `SelfVectorModule` attaches to `self_representation_core`.

Proposal: designate `self_representation_core.py` as THE self-model; fold any
unique idea from `modular_self_representation` / `belief_system` into it or
deprecate them; remove the orphans; keep `self_representation_interface` only if
`ConsciousnessCore` (the API path) is retained.

## 4. Measurement-vs-integration gap (the Functionalist-Emergentism reality check)

The thesis requires the integrated/conscious state to be **causally efficacious**,
not epiphenomenal. Reality: phi and sync_R have thin causal pathways into learning;
the newest "consciousness" additions (Levin metrics, the self-vector) are
measurement-only or behaviourally inert. The Phase-5 meta-representational
self-model is **not causally central** today (it does not drive decisions in a way
that helps). This is the core coherence gap: a lot of the stack measures
consciousness signatures rather than constituting a causally efficacious conscious
process.

## 5. Agent-competence baseline (from disk)

- dark_room: DQN last-100 reward ~92.0 vs consciousness agent ~12.95 (recorded
  2026-03-31, `docs/results/experiment_comparison.md`). DQN dominates.
- WCST: the consciousness agent triggers 0-1 rule changes in 60 episodes
  (`runs/sv_recovery`, this session) - it never reaches the self-monitoring regime.
- navigation: residual self-prediction skill +0.35 (this session) - the one place
  the self-model is validated, because the self-state actually moves there.

**Implication:** signatures that require task competence (WCST self-monitoring,
insight moments) are unmeasurable because the agent cannot perform the tasks. Many
"FAILED / INCONCLUSIVE" results trace back to this, not to the metric.

## 6. Honest verdict and prioritized proposal

The coherent core is worth keeping; the debt around it is real and is what makes
the project feel like it is "bolting on." Proposal, ordered by value/effort
(execution is a separate, approved step - nothing here is done yet):

- **P1 (low effort / low risk): cut dead weight.** Deprecate or remove the 6
  orphaned `self_model` files and the deprecated `ConsciousnessAgent` /
  `train_emotional_agent.py` path, after confirming no test depends on them. Pure
  decluttering; immediately raises coherence.
- **P2 (low / low): pick one orchestrator story.** Document `train_rlhf` as the
  research loop and `ConsciousnessCore` as the API/integration shell (or deprecate
  it too); stop maintaining three.
- **P3 (medium / medium): unify the self-model and make it causally central.** One
  self-model (`self_representation_core`); feed the self-vector into action
  selection (not only the gate) or drop it pending a real use; validate on
  navigation where it is measurable.
- **P4 (medium / high value): take a stance on measurement vs integration.** Either
  make EI/Levin causally drive something, or demote them honestly to "diagnostics"
  in the docs so the architecture does not imply they are part of the conscious
  mechanism.
- **P5 (high / high value): agent competence.** Make the agent competent on one
  task it can master so signatures become measurable, or formally adopt navigation
  as the primary testbed.

**Recommendation:** P1 + P2 first (cheap, high-clarity decluttering), then P3
(unify the self-model and make it causally central on navigation), reassessing P4
and P5 with that clarity. This converts the project from "measure many signatures
bolted onto a weak agent" toward "a smaller, causally-coherent conscious process we
can actually test", which is what Functionalist Emergentism requires.

## Update 2026-05-31: P1a + P2 executed

**P1a (decluttering, done).** Removed 6 files verified orphaned (no production
importer, no test dependency, git-recoverable): `models/self_model/intention_tracker.py`,
`models/self_model/emotion_context_tracker.py`, `models/self_model/meta_learner.py`,
`models/self_model/meta_learning.py`, and the deprecated agent pair
`models/agent/consciousness_agent.py` + `scripts/training/train_emotional_agent.py`.
Honest note: `consciousness_agent.py`'s own docstring claimed it was "preserved for
backward compatibility with existing tests", but no test imported it. The claim was
stale, which is itself evidence of the accretion problem.

**P2 (one orchestrator story, done).** The canonical research loop is
`train_rlhf.py`; `ConsciousnessCore` is retained as the API / integration shell
(used by `simulations/api/simulation_manager.py`); the deprecated `ConsciousnessAgent`
is removed. Two clearly separated stories instead of three.

**Modular cluster (RESOLVED 2026-06-01).** `modular_self_representation.py` +
`belief_system.py` are removed. `tests/test_consciousness_system.py` was rewritten
onto the canonical `SelfRepresentationCore` (its assertions depend on the monitor's
`evaluate_development`, not the self-model's output, so the swap is clean),
preserving the integration test and `MemoryIntegrationCore` coverage. The self-model
is now a single canonical module (`self_representation_core.py`) plus its Phase 5
`SelfVectorModule`.

**P3 (make the self-model causally central, done).** The learned self-vector is
now concatenated onto the broadcast that drives the policy PFC behind
`--enable-self-vector-action` (default off; PFC input dim grows by
self_vector_dim only when on, baseline bit-identical). The policy consumes and
learns from the self-model. Behavioural value on navigation is measured
separately (single-seed first; >=3 seeds before any default change).

**P4 (measurement vs integration stance, resolved as DIAGNOSTICS).** Of the
consciousness signals, only phi-delta (0.5x term in the policy reward) and sync_R
(binding optimizer) have causal pathways into learning, and they are thin. EI, the
Levin metrics, RIIU phi, and the self-vector self-prediction skill are DIAGNOSTIC
measurements, not part of the causal conscious mechanism, and the docs state this
plainly (this audit; `docs/levin_metrics_grounding.md` disclaims the Levin
modules; `docs/preregistered_predictions.md` frames phi/EI as hypotheses under
test). We do not claim these diagnostics constitute consciousness; they are
signatures we measure. P3 is the one move that makes a signal (the self-vector)
causally central by construction.

**P5 (agent competence) remains open** and is the largest item: the base agent is
weak (dark_room 12.95 vs DQN 92.00; WCST 0-1 rule changes), which is what makes
many signatures unmeasurable. This needs its own focused effort (curriculum /
stronger base RL) and is not attempted as a side change.
