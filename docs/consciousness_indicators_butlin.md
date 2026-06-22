# Consciousness indicator properties: the project's evaluation rubric

## Why this document exists

Following the [2026-06-02 decision](decisions/2026_06_02_competence_reading_2.md) to
adopt reading #2, the agent is judged by **consciousness signatures**, not by control
reward against a task-specialized baseline. This document operationalizes that: it maps
the project's architecture and metrics onto the **indicator properties** of Butlin,
Long et al. (2023), "Consciousness in Artificial Intelligence: Insights from the
Science of Consciousness" ([arXiv:2308.08708](https://arxiv.org/abs/2308.08708);
peer-reviewed in [Trends in Cognitive Sciences 2025](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4)).

Butlin et al. derive these indicators in computational terms from the leading
neuroscientific theories: Recurrent Processing Theory (RPT), Global Workspace Theory
(GWT), Higher-Order Theories (HOT), Predictive Processing (PP), Attention Schema Theory
(AST), plus agency and embodiment (AE). These are the same theories this architecture
is built on, which is why the rubric fits: the project is, in effect, an attempt to
instantiate these indicators in a biologically grounded substrate.

Status is reported FAILED-first and honestly: IMPLEMENTED, PARTIAL, or ABSENT. "Partial"
means the mechanism exists but is incompletely wired, untrained, heuristic, or not yet
empirically demonstrated above baseline.

## The rubric

### Recurrent Processing Theory

| Indicator | Project component | Status |
|-----------|-------------------|--------|
| RPT-1: algorithmic recurrence in input modules | `ReentrantProcessor` (5-10 adaptive settle cycles), capsule intra-hierarchy reentrant feedback, RSSM recurrence in `SensoryTectum` | IMPLEMENTED |
| RPT-2: organized, integrated perceptual representations | Topographic map + inverse-effectiveness fusion, hierarchical capsule composition, AKOrN oscillatory binding | PARTIAL: mechanisms implemented; the in-training phi-binding coupling (Phi-1) FAILED across 9 runs (see roadmap status 2026-05-24), so "integration" is structural, not yet demonstrated as a measured phi signature during training |

### Global Workspace Theory

| Indicator | Project component | Status |
|-----------|-------------------|--------|
| GWT-1: multiple specialized systems operating in parallel | vision, audio, memory, body, semantic specialists competing in `GlobalWorkspace` | IMPLEMENTED |
| GWT-2: limited-capacity workspace with a bottleneck + selective attention | GNW ignition threshold + AKOrN binding selecting the winner; the 256-D broadcast is the low-dimensional bottleneck | IMPLEMENTED (and note: the 2026-06-02 localization shows this bottleneck is by design, the source of the control-vs-integration trade reading #2 accepts) |
| GWT-3: global broadcast available to all modules | `broadcast_payload` + `receive_broadcast` on each specialist | IMPLEMENTED |
| GWT-4: state-dependent attention to query modules in succession | reentrant cycles re-query modules with top-down feedback | PARTIAL: no explicit successive-query controller; querying is the settle loop, not a learned attention policy |

### Higher-Order Theories (Phase 5's primary target)

| Indicator | Project component | Status |
|-----------|-------------------|--------|
| HOT-1: generative, top-down / noisy perception | RSSM prior/posterior (generative), top-down reentrant prediction errors | PARTIAL |
| HOT-2: metacognitive monitoring (reliable representation vs noise) | confidence gate node, RSSM KL surprise, self-prediction skill (`self_pred_skill`) | PARTIAL: self-vector validated on navigation (skill +0.35) but inert on WCST; Phase 5 deliverable strengthens this |
| HOT-3: agency guided by belief-formation + belief update from metacognition | action selection core; self-vector feeding gate/policy (Phase 5 deliverables 1 and 3) | PARTIAL |
| HOT-4: sparse, smooth coding generating a quality space | `PhenomenologicalMapper` quality-space vector; capsule poses | PARTIAL: heuristic, not derived from IIT's formalism (documented in `qualia_mapper.py`) |

### Predictive Processing

| Indicator | Project component | Status |
|-----------|-------------------|--------|
| PP-1: input modules using predictive coding | RSSM predictive coding, reentrant prediction-error minimization, retinotopic encoder | PARTIAL: predictive coding present but the front-end is trained ad-hoc (reward-MSE + topographic), not by a principled free-energy objective. The active-inference reframing (roadmap Phase 6 / decision doc) is the path to make this IMPLEMENTED |

### Attention Schema Theory

| Indicator | Project component | Status |
|-----------|-------------------|--------|
| AST-1: predictive model representing and controlling the state of attention | `models/memory/attention_schema.py` (`AttentionSchema`) | PARTIAL: module exists; its control of workspace attention is not yet wired into the training loop |

### Agency and Embodiment

| Indicator | Project component | Status |
|-----------|-------------------|--------|
| AE-1: agency, flexible responsiveness to competing goals | action selection, reward shaping, competing interoceptive drives | PARTIAL: WCST cognitive flexibility weak (agent rarely enters the rule-shift regime) |
| AE-2: embodiment, modeling output-input contingencies and using it in control | RSSM models action-conditioned dynamics; body schema + interoceptive PAD loop | PARTIAL and a known weak point: the 2026-06-02 control-representation fix (predicting next observation from content + action) did NOT improve control, so the action-conditioning of perception is empirically weak |

## How the project's own metrics complement the indicators

The Butlin rubric is qualitative (which mechanisms are present). The project also has
quantitative signatures that test whether those mechanisms do measurable work:

- **IIT phi** (`models/evaluation/iit_phi.py`) - integration (RPT-2, GWT). In-training
  Phi-1 coupling FAILED; phi varies but does not track binding. Reported honestly.
- **Effective Information / causal emergence** (Hoel; `effective_information.py`) -
  whether the workspace is a stronger causal macro-variable than its parts.
- **Levin metrics** (`levin_consciousness_metrics.py`) - bioelectric complexity,
  collective intelligence, basal cognition; run as untrained diagnostics today.
- **Behavioral integration** on DMTS (working memory + binding) and WCST
  (meta-cognition + flexibility) - the consciousness-demanding tasks.
- **Insight detection** and **phenomenological mapping** - GWT ignition + quality space.

## How this rubric is used

1. It replaces control reward as the success criterion (decision 2026-06-02).
2. Each Phase 5 deliverable is tied to the indicator it advances (the self-vector loop
   and metacognition target HOT-2/HOT-3; the substrate-independence test targets
   measurable agency/integration on the consciousness tasks).
3. Progress is honest movement of an indicator from PARTIAL to IMPLEMENTED with a
   measured signature, never a claim of consciousness. Butlin et al. themselves
   conclude no current system is conscious; this rubric tracks indicator coverage, not
   a verdict.

## Status update 2026-06-21: the integrated content is identity-free (RPT-2 / causal efficacy)

This session characterized a property that bears directly on the integration indicators
and is reported here so the rubric reflects it honestly.

- **RPT-2 (integrated perceptual representations): characterized as LOSSY, not just
  unmeasured.** The collapse-locus probe (confirmed on a trained tectum,
  [collapse_locus_trained_2026_06_21.md](results/collapse_locus_trained_2026_06_21.md))
  shows stimulus identity is decodable at ~1.0 in obs_map but at chance in the RSSM
  latent, the capsule poses, the 256-D tectum_content, AND the post-GNW broadcast (the
  broadcast equals tectum_content,
  [perception_decodability_2026_06_09.md](results/perception_decodability_2026_06_09.md)).
  The integration pathway discards task-relevant identity at the RSSM step; identity is a
  low-variance direction in obs_map that MSE compression drops (PCA-confirmed,
  [collapse_locus_wmobs_2026_06_21.md](results/collapse_locus_wmobs_2026_06_21.md)). Two
  world-model reconstruction objectives at the locus FAILED to recover it. So "integrated
  representation" is present structurally but is identity-free in content.
- **Causal efficacy / substrate-independence (section 13): BLOCKED-AND-CHARACTERIZED.**
  The policy and workspace read identity-free content, so the agent cannot enter the
  DMTS/WCST diagnostic regimes, and SI-1 (a task-accuracy advantage over DQN) cannot be
  measured on a non-functional agent. The blocker is now characterized (perception
  collapse, reconstruction-robust) plus the separate RL second wall (2026-06-14/15). The
  pre-registered section-13 thresholds are NOT revised; it stays UNTESTED.
- **Honest scope.** This does not refute the structural presence of the GWT/RPT
  mechanisms (GWT-1..3 remain IMPLEMENTED structurally); it characterizes that the
  *content* flowing through them is task-uninformative on these tasks. A single
  unifying "low-variation" root was considered and RETRACTED: phi varies but is
  low-magnitude, sync_R is low-variation, and the ignition gate was threshold-saturated
  (fixed this session to selective ignition); these are distinct weaknesses, not one root.

## Caveat

Indicator coverage is necessary-evidence framing, not proof. A system can satisfy
indicators without being conscious, and the theories disagree. The rubric's value is
that it is rigorous, pre-committed, and built from the same science as the
architecture, so it disciplines claims and prevents substituting a convenient metric
(like task reward) for the hard question.
