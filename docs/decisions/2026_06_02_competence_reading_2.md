# Decision (2026-06-02): adopt reading #2 - judge by consciousness signatures, keep the biological strategy

## Status: ADOPTED

## What prompted this

The P5 thread set out to explain why the consciousness agent's control reward on
dark_room (~15) sits far below a vanilla DQN on raw pixels (92). A diagnosis then a
two-step localization (all numbers loaded from disk, see
[`agent_competence_fix_2026_06_02.md`](../results/agent_competence_fix_2026_06_02.md))
isolated where the control-relevant signal is lost, holding the DQN learner constant:

| tap | DQN reward | stage |
|-----|------------|-------|
| pixels | 92.00 last-100 (105.31 / 1000 ep) | raw input |
| obs_map | 17.14 | post retinotopic encoder + fusion, PRE RSSM/capsule |
| tectum_content | 15.93 | post capsule collapse |
| broadcast | 14.65 | post GNW |

The signal is lost at the FIRST encoding stage (pixels -> obs_map). The capsule
collapse and the GNW each add only a small further loss. The policy is not the
bottleneck either (a plain A2C and the Go/No-Go core tie on the broadcast). The
bottleneck is the perceptual front-end.

This left an honest fork:
1. **Under-resourced front-end** - rebuild/retrain the encoder for control.
2. **Inherent to a biology-first design** - the perception is built for biological and
   topographic fidelity feeding a low-dimensional integrated workspace, not
   pixel-precise control; a ~15-20 dark_room ceiling is the cost of that design.

## The decision

**Adopt reading #2.** dark_room control reward is the cost of the biological design,
not the project's success metric. The project's purpose is to study whether
consciousness can emerge from a substrate built on the biological theories of how it
arose (Feinberg & Mallatt; Rouleau & Levin; GNW; IIT; predictive processing). The
success criteria are the **consciousness signatures**, not control performance against
a task-specialized baseline.

Rationale (the user's, recorded verbatim in intent): going away from the biological
approach would leave the project with no roadmap that has any evidential basis for
emergence. The biological grounding is the only map we trust. We keep it.

## What this changes

- **Retired as a target metric:** dark_room (and navigation) raw control reward. We do
  not optimize it, and we do not treat the gap to a pixel-DQN as a defect. It remains
  logged as a behavioral sanity signal only.
- **Adopted as the success criteria:** consciousness indicator properties (Butlin,
  Long et al. 2023, [arXiv:2308.08708](https://arxiv.org/abs/2308.08708); peer-reviewed
  [Trends in Cognitive Sciences 2025](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(25)00286-4)),
  mapped in [`consciousness_indicators_butlin.md`](../consciousness_indicators_butlin.md),
  together with the project's existing signatures: IIT phi, EI causal emergence
  (Hoel), the Levin metrics, phenomenological mapping, insight detection, and
  behavioral integration on the consciousness-demanding tasks (DMTS, WCST).

## The honest caveat (this is not "declare victory")

Reading #2 does not license ignoring competence. The agent must still PERCEIVE well
enough to ENTER the diagnostic regimes of the consciousness-demanding tasks, or the
signatures cannot be measured at all. DMTS and WCST are currently flat
(`docs/results/experiment_comparison.md`: DMTS -9.82 flat, WCST -1.94 flat; WCST
reaches 0-1 rule changes per episode, rarely entering the rule-shift regime), and the
localization implies the front-end may be too lossy even for that. So the competence
bar is redefined, not removed:

> "good enough perception to demonstrate the consciousness signatures on the
> consciousness-demanding tasks" replaces "match a pixel-DQN on control reward."

And that bar is TESTED, not assumed (a near-term Phase 5 prerequisite; see the
roadmap).

## Research-backed forward path (does not abandon the strategy)

- **Evaluation:** the Butlin indicator rubric makes "judge by signatures" operational
  and falsifiable, and it is derived from the same theories the architecture
  implements (RPT, GWT, HOT, predictive processing, attention-schema). See
  [`consciousness_indicators_butlin.md`](../consciousness_indicators_butlin.md).
- **Perception, biologically:** if the perception-supports-the-tasks check fails, the
  enabling fix is **active inference / active predictive coding** (Rao et al., Neural
  Computation 2024; Friston et al. 2023, *Active Inference*, MIT Press; Deep Active
  Inference, [arXiv:2505.19867](https://arxiv.org/abs/2505.19867)) - a
  neuroscience-grounded objective that couples perception to action via free-energy /
  prediction-error minimization. It replaces the current ad-hoc reward-MSE + TDANN
  front-end training with a principled biological one, deepening the strategy rather
  than leaving it. This is already on the roadmap (Phase 6 Section 6.1); the
  localization elevates it.
- **Roadmap:** continue Phase 5 (Dynamic Self-Representation & Meta-Cognition), the
  HOT-aligned, Rouleau-Levin-grounded next phase, judged by the indicator rubric.

## What is NOT changed

The biological architecture, its components, and the trusted roadmap stand. No
non-biological control encoder is introduced. The Phi-1 chapter stays closed. The
KomplexNet and RIIU code paths stay default-off diagnostics.
