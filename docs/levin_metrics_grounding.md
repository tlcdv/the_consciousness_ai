# Levin consciousness metrics: what they measure, grounding, and caveats

Companion to [`rouleau_levin_substrate_independence.md`](rouleau_levin_substrate_independence.md).
This documents the 5 metrics in
[`models/evaluation/levin_consciousness_metrics.py`](../models/evaluation/levin_consciousness_metrics.py),
activated in the training loop behind `--enable-levin-metrics` (Phase 5
deliverable 4, commit `a842267`). It exists so no future session treats these
numbers as more than they are.

## How they run

When `--enable-levin-metrics` is set, `train_rlhf.py` instantiates a
`HolonicSystem` and a `LevinConsciousnessEvaluator` in `init_components` and
computes the 5 metrics each step from the holonic output, the workspace
broadcast, the tectum content, and the gate state. The values are logged to
`metrics.csv` as `levin_*` columns.

**Caveat 1 (untrained modules).** The `HolonicSystem` and
`BioelectricSignalingNetwork` run in inference mode (`eval()`, `torch.no_grad()`)
and are NOT trained and NOT part of the policy gradient. They are fixed
measurement functions over the agent's activations. Whether to train them is a
Phase 5 deliverable-5 decision, not assumed here.

**Caveat 2 (proxies, not Levin's exact metrics).** These are computational
proxies inspired by Levin's framework, not implementations of his published
formulae. Levin's own collective-intelligence / cognitive-boundary measure is a
spatio-temporal goal-boundary (the scale a system measures and controls), Levin
2019 *The Computational Boundary of a "Self"*
([PMC6923654](https://pmc.ncbi.nlm.nih.gov/articles/PMC6923654/)). The faithful
goal-boundary version is scheduled as the Markov-blanket self-boundary detector
(Phase 5 deliverable 6).

## The 5 metrics

| Metric | What it computes | Status (2026-05-29 de-risk) |
|--------|------------------|------------------------------|
| `bioelectric_complexity` | Mean pairwise L2 distance between component bioelectric fields, normalised to [0,1]. Field differentiation as a complexity proxy. | USABLE (responds to input) |
| `morphological_adaptation` | 1 - cosine similarity between the current holonic `integrated_state` and the last few, averaged. How much the integration reorganises over time. | USABLE |
| `collective_intelligence` | Mean pairwise cosine similarity of the holon state vectors, mapped to [0,1]. Structural coherence of the holon collective. | USABLE (fixed 2026-05-29; was INERT) |
| `goal_directed_behavior` | Cosine alignment of goal vs outcome embeddings. | CONSTANT 0 placeholder (no goal/outcome embeddings until deliverable 5) |
| `basal_cognition` | Coefficient of variation of component activation means. Non-uniform sub-component activity as a basal-cognition proxy. | USABLE |

## The collective_intelligence fix (2026-05-29)

The original `evaluate_collective_intelligence` used `1 - normalized_entropy` of
the holonic attention matrix. With an untrained `HolonicSystem` the integration
attention is near-uniform regardless of input, so the metric was inert: constant
~2e-6 across 64 deliberately diverse inputs (std 4.47e-7) and effectively
constant under training (1 unique value in dark_room). Measured evidence:
[`docs/results/levin_derisk_2026_05_29.md`](results/levin_derisk_2026_05_29.md).

It now measures the mean pairwise cosine similarity of `holon_states`, which
varies with input because each holon applies its own learned map to the shared
input. Post-fix: std 6.41e-3 across the same 64 inputs, 64 unique values, range
0.485 to 0.523. This is a structural-coherence proxy, not Levin's goal-boundary
metric; the latter is deliverable 6.

`goal_directed_behavior` remains a constant-0 column on purpose: the baseline
wiring passes empty actions/goals/outcomes. Its definition (what counts as a
goal embedding and an outcome embedding) is fixed at the pre-registration of the
substrate-independence test (deliverable 5), not invented here.

## What these metrics are NOT

- Not a consciousness score. They are diagnostic correlates logged for the
  substrate-independence falsification test (deliverable 5).
- Not trained signals. See caveat 1.
- Not validated against any ground truth. The dynamic-range de-risk only
  established that 4 of 5 respond to input; whether they rise specifically with
  self-monitoring demand is the open empirical question deliverable 5 tests.

## References

- Levin, M. (2019). The Computational Boundary of a "Self". *Front. Psychol.* 10, 2688. [PMC6923654](https://pmc.ncbi.nlm.nih.gov/articles/PMC6923654/)
- Collective intelligence: a unifying concept for integrating biology across scales and substrates. *Commun. Biol.* 2024. [s42003-024-06037-4](https://www.nature.com/articles/s42003-024-06037-4)
- De-risk evidence: [`docs/results/levin_derisk_2026_05_29.md`](results/levin_derisk_2026_05_29.md)
