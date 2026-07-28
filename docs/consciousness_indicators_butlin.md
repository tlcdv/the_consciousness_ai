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
- **Causal emergence** (Hoel; Causal Emergence 2.0 in `causal_emergence_svd.py`, superseding the
  deprecated EI in `effective_information.py`) - whether the workspace is a stronger causal
  macro-variable than its parts. Instrument under validation; it promotes no indicator on its own.
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

## Status update 2026-07-02: the signature instruments are largely degenerate on the current agent

A zero-compute assessment over five recent trained runs
([signature_assessment_2026_07.md](results/signature_assessment_2026_07.md)) checked
whether the measured-signature instruments discriminate anything before any further
coverage claims. Findings, applied to the rubric:

- **GWT-2 (bottleneck + selective attention): the ignition half is SATURATED, not
  selective in practice.** 99.79 to 100 percent of steps are flagged conscious in every
  run, including the two trained after the selective-ignition fix; the fix's quiet steps
  all fall in the first ~15 episodes and never recur. Mechanically, ignition fires when
  input energy exceeds its own running average, and the workspace input energy is
  near-constant on DMTS. Structural status stays IMPLEMENTED; the measured ignition
  signature is currently non-discriminating.
- **Causal-emergence evidence (EI, used to support the emergence framing of the rubric):
  the current instrument cannot provide it.** Gate-level EI is bit-identical
  (0.031178) in every 50-episode window of every run, reproduced exactly by a
  single-state trajectory: the gates never leave one joint tertile state. The macro
  level sits at or near its own constant floor (0.373712), and the "emergence ratio"
  ~12x at the frozen windows equals the ratio of the two floors, a state-space-size
  artifact that favors "emergence" under degeneracy. EI-based claims are suspended
  until the micro level actually transitions and the floor bias is handled.
- **RPT-2 (integration): the binding signature is objective-invariant.** Mean sync_R is
  0.2662 to 0.2666 across five different training objectives (same seed and env). The
  Phi-1 chapter stays closed; this adds that binding synchrony does not respond to the
  training objective at all in the tested regime.
- **Phi (IIT-adjacent magnitude): the one objective-sensitive signature.** Computed-step
  phi means: ~1.1e-3 in the three reconstruction-family runs vs ~3.1e-4 in the two
  action-conditioned world-model runs (~3.6x lower), CVs 0.78 to 1.2, absolute
  magnitudes near zero throughout. Single seed; hypothesis-grade sensitivity.
- **Inactive instruments.** RIIU phi (all variants), all five Levin metrics, and
  self-prediction are exactly zero in all five runs (default-off modules), and the PAD
  dominance channel is structurally zero. Coverage rows that depend on these modules
  have no measured signal in the assessed runs.

No indicator status changes: nothing here adds an IMPLEMENTED, and the structural
presence of the mechanisms is not refuted. What changes is the evidentiary baseline:
measured-signature claims must cite this assessment and its degeneracy findings first.
Section 13 remains BLOCKED-AND-CHARACTERIZED with thresholds untouched.

## Status update 2026-07-06: the integration pathway is repaired and integration markers respond (RPT-2 no longer lossy)

This session fixed the pathway the 2026-06-21 update flagged as the central defect. A
continuous RSSM latent ([b1_continuous_latent_2026_07.md](results/b1_continuous_latent_2026_07.md))
plus an all-levels capsule projection now carry stimulus identity all the way to
tectum_content / the broadcast (decodable 0.83 shape / 0.98 color, from chance before). An
ablation-and-markers study over the OFF -> HALF -> ON perception ladder
([signature_ablation_2026_07.md](results/signature_ablation_2026_07.md), the method of
arXiv:2512.19155) measured whether delivering identity to the broadcast moves the signatures.

- **RPT-2 (integrated perceptual representations): the "LOSSY / identity-free"
  characterization is SUPERSEDED for the fixed configuration.** The integrated content now
  carries identity, and three integration markers respond, robust across 3 seeds
  (non-overlapping HALF-vs-ON seed ranges): broadcast content variation (CV ~2x), phi max
  (~2.2x), and the floor-corrected macro EI (~3x). This is the first replicated,
  ablation-causal signature response to an architectural change in the project. RPT-2 stays
  PARTIAL, not IMPLEMENTED: it is single-configuration, DMTS-only, task reward is still flat,
  and two integration signals below do NOT respond.
- **GWT-2 (ignition): NOT moved.** The seed-42 ladder suggested the gate de-saturates, but
  the response did not survive 3 seeds (quiet-step counts overlap; seed 44 reverses).
  Ignition stays saturated and content-limited (consistent with the A2 probe showing the
  gate signal is phase-invariant). The perception fix alone does not make the gate selective.
- **RPT-2 binding (sync_R): still objective/perception-invariant** (0.251 to 0.257 across
  the ablation), consistent with the closed Phi-1 chapter.
- **EI causal emergence: macro level now carries real structure, micro level still frozen.**
  The corrected macro EI responds to the fix, but the gate/micro EI stays at the constant-
  trajectory floor (0.000 corrected) at every rung, so the causal-emergence RATIO remains
  ill-posed (a real macro numerator over a frozen micro denominator). The gate discretization
  is the next distinct locus, not addressed by this fix.

No indicator is promoted to IMPLEMENTED and no consciousness claim is made. The honest net:
the integration content defect that blocked every measured signature is repaired with a
replicated signature response, while binding, ignition selectivity, and gate-level emergence
remain open. Pre-registered thresholds (EI, Phi-1, section 13) are unchanged.

## Status update 2026-07-28: an external methodological standard this rubric has not been tested against (NO status changes)

A review of three papers on thalamic gating of human consciousness
([thalamic_gating_evidence.md](thalamic_gating_evidence.md)) surfaced a methodological gap
that affects how several rows below should be read. Nothing was measured on this agent, so
**no indicator moves, in either direction.** What follows is a note on evidentiary standing,
not a result.

- **No content specific contrast has ever been run here.** All three reviewed papers rest on
  holding the stimulus constant while the percept varies (near threshold detection with the
  motor response matched). Every verdict in `docs/results/` is state based: a contrast across
  configs, ablations, objectives or runs. The GWT-2 (limited capacity workspace and selective
  attention) and GWT-3 (global broadcast) rows are therefore supported by structural presence
  plus state based measurement only. Neither has faced the within state, matched stimulus
  test that the consciousness literature treats as the discriminating one.
- **Latency has never been measured.** The discriminating variable in Fang et al. (2024,
  preprint) is *when* aware and unaware trajectories diverge, per site. The A2 result that
  ignition is task phase invariant was measured in amplitude only. GWT-2's saturation may be
  a content problem, a threshold problem or a timing problem, and the third has not been
  tested.
- **The one measure that survived in Koch et al. (2016) is unimplemented here.** That review
  reports gamma synchrony and the P3b as failed markers and PCI as the measure that
  separates conscious from unconscious states at the single participant level. This
  repository's `PerturbationTester.calculate_pci_approximation` returns a random number. The
  three measures that are implemented (phi, EI, CE 2.0) have each been characterised as
  degenerate on this agent, and the review names the reason a perturbational measure escapes
  that failure mode.
- **Two existing negative results are externally corroborated.** Koch et al. report gamma
  synchrony as a failed marker of consciousness, and Fang et al. found no consciousness
  related activity in any gamma band in any thalamic nucleus. The closed Phi-1 chapter and
  the invariant `sync_R` (0.251 to 0.257 across the perception ablation) are consistent with
  the published record rather than anomalous. This does not promote or demote any row; it
  raises confidence that the RPT-2 binding non-response is a real property and not an
  instrument fault.

Net: no promotions, no demotions, no consciousness claim. The rubric gains a named external
standard (matched stimulus contrast, divergence latency, perturbational complexity) that the
GWT rows have not yet been held to. Pre-registered thresholds (EI, Phi-1, section 13) are
unchanged.

## Caveat

Indicator coverage is necessary-evidence framing, not proof. A system can satisfy
indicators without being conscious, and the theories disagree. The rubric's value is
that it is rigorous, pre-committed, and built from the same science as the
architecture, so it disciplines claims and prevents substituting a convenient metric
(like task reward) for the hard question.
