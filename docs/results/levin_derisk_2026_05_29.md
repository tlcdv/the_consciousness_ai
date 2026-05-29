# Levin metric de-risk: which of the 5 metrics actually respond to input?

**Date:** 2026-05-29
**Context:** Phase 5 deliverable 4 (Levin module activation, commit `a842267`) shipped
5 `levin_*` metrics to `metrics.csv`. The Step-1 smoke run showed
`collective_intelligence` pinned at `2e-6` (zero variance). That run was degenerate
(the dark_room agent got stuck, so the broadcast barely changed), so it could not
tell "the metric is inert" apart from "the input did not vary". This de-risk removes
that confound before any further Phase 5 work is built on these metrics.

All numbers below are loaded from disk in the same session that produced them.

## Method

Two probes:

1. **Controlled input-sensitivity probe** (`scripts/analysis/diagnose_levin_variance.py`):
   instantiate `HolonicSystem` + `LevinConsciousnessEvaluator` (seed 0, the same
   config the training loop uses), feed 64 deliberately DIVERSE inputs (normal,
   uniform, sparse one-hot, sinusoid, scaled-normal), and report per-metric
   min/max/std/unique. This isolates "does the metric respond to input?".
2. **In-situ probe**: `train_rlhf --enable-levin-metrics` on `navigation` (state
   genuinely varies) and `dark_room` (the smoke baseline), 5 episodes x 120 steps
   each, then per-metric std/unique from `metrics.csv` (600 rows each).

Verdict rule (deliberately lenient, to catch near-constant metrics, not grade
quality): a metric is **USABLE** if `std > 1e-4` AND it takes `>= 5` distinct
values (rounded to 9 dp); otherwise **INERT**.

## Controlled probe (64 diverse inputs, seed 0)

| metric | min | max | std | unique | verdict |
|--------|-----|-----|-----|--------|---------|
| bioelectric_complexity | 0.149719 | 0.178404 | 4.96e-03 | 64 | USABLE |
| morphological_adaptation | 0 | 1 | 1.23e-01 | 64 | USABLE |
| collective_intelligence | 1.26e-06 | 2.98e-06 | **4.47e-07** | 16 | **INERT** |
| goal_directed_behavior | 0 | 0 | 0 | 1 | INERT (by design) |
| basal_cognition | 0 | 1 | 2.43e-01 | 54 | USABLE |

## In-situ probe (600 steps each)

navigation:

| metric | min | max | std | unique | verdict |
|--------|-----|-----|-----|--------|---------|
| bioelectric_complexity | 0.153326 | 0.163888 | 4.29e-04 | 34 | USABLE |
| morphological_adaptation | 0 | 0.277283 | 1.37e-02 | 9 | USABLE |
| collective_intelligence | 1e-06 | 2e-06 | **3.31e-07** | 2 | **INERT** |
| goal_directed_behavior | 0 | 0 | 0 | 1 | INERT (by design) |
| basal_cognition | 0.717251 | 0.727594 | 4.24e-04 | 57 | USABLE |

dark_room:

| metric | min | max | std | unique | verdict |
|--------|-----|-----|-----|--------|---------|
| bioelectric_complexity | 0.18397 | 0.185371 | 1.79e-04 | 107 | USABLE |
| morphological_adaptation | 0 | 0.302711 | 1.49e-02 | 21 | USABLE |
| collective_intelligence | 2e-06 | 2e-06 | **4.24e-22** | 1 | **INERT** |
| goal_directed_behavior | 0 | 0 | 0 | 1 | INERT (by design) |
| basal_cognition | 0.700234 | 0.70418 | 9.89e-04 | 119 | USABLE |

## Verdict

All three conditions agree:

- **`collective_intelligence` is INERT.** It barely moves even across 64 wildly
  different synthetic inputs (std 4.47e-7), and is effectively constant under real
  training trajectories (1 unique value in dark_room). This is a metric defect, not
  a degenerate-agent artifact. **FAILED** the dynamic-range bar.
- **`goal_directed_behavior` is a constant-0 placeholder by design** (the baseline
  wiring passes empty actions/goals/outcomes). Not a defect; it is defined at the
  Deliverable-5 pre-registration.
- **`bioelectric_complexity`, `morphological_adaptation`, `basal_cognition` are
  USABLE** with real dynamic range in every condition.

## Root cause of the inert metric

`evaluate_collective_intelligence`
([models/evaluation/levin_consciousness_metrics.py:102](../../models/evaluation/levin_consciousness_metrics.py#L102))
applies `softmax` to the already-row-normalized holonic attention matrix and returns
`1 - normalized_entropy`. The `HolonicSystem` integration attention is untrained, so
it is near-uniform regardless of input, which makes the normalized entropy near-maximal
and constant. The metric therefore sits at ~0 and does not move. It also does not match
Michael Levin's actual notion of collective intelligence, which is a spatio-temporal
goal-boundary measure (the scale a system measures and controls), not the peakedness of
an attention matrix (Levin 2019, *The Computational Boundary of a "Self"*,
[PMC6923654](https://pmc.ncbi.nlm.nih.gov/articles/PMC6923654/)).

## Decision

Per the approved plan's decision gate:

- **Fix `collective_intelligence` (Phase 1a).** Replace the entropy-of-attention
  internals with an input-sensitive measure of holonic integration computed from
  `holon_states` (which the probe confirms varies with input), keeping the function
  name and signature stable and documenting honestly what it now measures.
- Leave `goal_directed_behavior` as a documented constant-0 placeholder until
  Deliverable 5 defines goal/outcome embeddings.
- Keep `bioelectric_complexity`, `morphological_adaptation`, `basal_cognition` as is.
- Add value-based tests asserting dynamic range (not just [0,1] bounds), and a
  regression test that `collective_intelligence` is no longer constant.

## Reproducibility

```bash
python -m scripts.analysis.diagnose_levin_variance --trials 64 --seed 0
python -m scripts.training.train_rlhf --env navigation --episodes 5 --max-steps 120 \
    --enable-levin-metrics --phi-sample-every 5 --log-dir runs/levin_derisk/navigation
python -m scripts.training.train_rlhf --env dark_room --episodes 5 --max-steps 120 \
    --enable-levin-metrics --phi-sample-every 5 --log-dir runs/levin_derisk/dark_room
```
