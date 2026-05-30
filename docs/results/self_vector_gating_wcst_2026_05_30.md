# Self-vector gating on WCST: does conditioning the gate on the self-model help?

**Date:** 2026-05-30
**Phase 5 deliverable 3 value test.** Step 3 added an optional path that feeds the
learned self_vector into `ConsciousnessGate` (commit `9fb1ce6`). This ablation asks
the deliverable's value question: does conditioning the gate on the self-model
improve performance on WCST, a task that demands self-monitoring (detecting one's
own performance drop after a hidden rule change)?

All numbers loaded from disk in the same session that produced them.

## Method

Two runs, identical except for the single intervention (whether the gate uses the
self_vector). Same seed, same everything else, so the comparison is clean.

| Run | Flag | Gate uses self_vector? | Self-vector trained? |
|-----|------|------------------------|----------------------|
| baseline | `--enable-self-vector` | NO | yes |
| gating | `--enable-self-vector-gating` | YES | yes |

Both: `--env wcst --episodes 80 --max-steps 150 --seed 42 --phi-sample-every 5`.

The GNW winner-take-all `KeyError: 'body'` that blocked longer WCST runs was fixed
first (commit `f62bed6`); the fix is in the shared path, so it applies to both arms
and does not bias the comparison.

## Results

| metric | baseline (gate ignores SV) | gating (gate uses SV) |
|--------|----------------------------|------------------------|
| episodes | 80 | 80 |
| reward mean | 0.723 | 0.691 |
| reward first-20 | 0.042 | 1.016 |
| reward last-20 | 0.756 | 0.431 |
| positive episodes | 52 / 80 | 50 / 80 |
| self_pred_skill mean | -0.4806 | -0.4810 |
| self_pred_skill last-500 steps | -0.4792 | -0.4777 |

## Verdict: FAILED

**Gating the gate on the self-vector did not improve WCST performance.** Reward
mean is essentially unchanged and slightly lower with gating (0.691 vs 0.723);
positive episodes are near-equal (50 vs 52); the last-20 reward is lower for the
gating run. There is no positive signal that conditioning the gate on the
self-model helps this self-monitoring task.

**The self-model does not beat persistence on the real WCST self-features.**
`self_pred_skill` is negative (~ -0.48) in both runs, meaning the learned self-vector
predicts the next step's first-order features WORSE than the trivial "next ==
current" persistence baseline. This answers the open question carried from Step 2
(2026-05-30 self-vector commit): the value test showed the mechanism CAN beat
persistence on synthetically learnable dynamics, but the actual WCST first-order
features (PAD emotion, interoception, learning velocity, etc.) change too smoothly
for this self-model to add predictive value in 80 episodes.

## Honest caveats and scope

- **Single seed (seed 42).** Per the project's >=3-seed rule, this is a
  hypothesis-level signal, not a definitive verdict. However, the result is null /
  slightly negative, not borderline, so there is no positive delta to confirm with
  more seeds. Spending 3-seed compute to firm up a null is low value; the honest
  reading is "no evidence self-vector gating helps WCST as currently designed."
- **The gating default stays OFF** (`--enable-self-vector-gating` defaults off). This
  ablation does not change any default. The mechanism remains available for future
  redesign work.
- **What this does NOT say.** It does not refute the meta-representation goal of
  Phase 5. It says this specific design (a 14-D first-order feature vector + an
  SPR-style one-step self-prediction, with the self-vector added to the gate input)
  does not yet help WCST. The self-prediction objective is sound (value test
  passed on learnable structure); the limitation is that WCST's self-features are
  near-persistence-predictable, so the self-vector carries little task-relevant
  signal into the gate.

## Implications for next steps (hypotheses, not conclusions)

- The first-order feature set may be too low-dimensional / too smooth. Richer or
  faster-changing self-features (e.g. action history, prediction-error signals,
  rule-hypothesis state) might give the self-prediction objective something to
  learn that persistence cannot trivially capture.
- The value of a self-model may show up on a different measure than raw WCST
  reward (e.g. recovery speed specifically in the trials right after a rule change,
  the self-monitoring-demanding window pre-registered for deliverable 5), which this
  episode-reward comparison does not isolate.
- Deliverable 5's pre-registered substrate-independence test
  ([preregistered_predictions.md](../preregistered_predictions.md) section 13) is
  the proper falsification frame; this ablation is a cheaper preliminary that
  already flags the risk that the self-model adds no behavioural value as wired.

## Reproducibility

```bash
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env wcst \
    --episodes 80 --max-steps 150 --enable-self-vector --seed 42 \
    --phi-sample-every 5 --log-dir runs/sv_ablation/wcst_baseline
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env wcst \
    --episodes 80 --max-steps 150 --enable-self-vector-gating --seed 42 \
    --phi-sample-every 5 --log-dir runs/sv_ablation/wcst_gating
```
