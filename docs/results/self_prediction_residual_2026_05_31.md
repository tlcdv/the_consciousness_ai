# Residual self-prediction: does the self-model now beat persistence?

**Date:** 2026-05-31
**Phase A of the self-vector foundation fix.** The Step-3 WCST gating ablation
([self_vector_gating_wcst_2026_05_30.md](self_vector_gating_wcst_2026_05_30.md))
failed because `self_pred_skill` was negative (~ -0.48): the self-model predicted
its own next first-order features WORSE than a persistence baseline. Root-cause
hypothesis: predicting the RAW next features forces the model to reproduce large,
near-constant feature values, and any small error there loses to persistence.

Fix: residual (delta) prediction. `SelfVectorModule.predict_next(features)` now
returns `features + predict(encode(features))`, so the predictor head outputs the
CHANGE and persistence corresponds to a zero delta. The model beats persistence
only if it captures systematic dynamics in the self-features.

All numbers loaded from disk in the same session that produced them.

## Method

Residual `predict_next` wired into the run_episode self-vector loop. Three short
runs, `--enable-self-vector --seed 42`, 25 episodes x 100 steps each, on WCST,
DMTS, and navigation. `self_pred_skill = 1 - mse / persistence` per step;
skill rises as the module trains, so the trained (later) steps matter most.

## Results

| env | rows | skill overall | 2nd-half | last-200 | max | frac steps > 0 |
|-----|------|---------------|----------|----------|-----|----------------|
| navigation | 2500 | -0.003 | **+0.125** | **+0.354** | +0.999 | 0.52 |
| dmts | 2500 | -0.300 | -0.290 | -0.348 | +0.995 | 0.42 |
| wcst | 2500 | -0.478 | -0.475 | -0.481 | +1.000 | 0.28 |

## Verdict: PARTIAL (PASS on navigation, FAIL on WCST)

- **Navigation: PASS.** Residual prediction takes skill from negative to clearly
  positive and RISING with training (2nd-half +0.125 -> last-200 +0.354). The
  self-model beats persistence on a task where the agent's self-state genuinely
  changes (movement between rooms, battery drain, broadcast dynamics). The
  foundation is no longer inert.
- **WCST: FAILED (unchanged at -0.48).** During card-sorting the agent's
  first-order self-features (PAD emotion, interoception) barely change, so
  persistence is unbeatable. This is a property of WCST's self-dynamics, not a
  mechanism flaw, and it explains the Step-3 WCST gating failure: the self-vector
  is inert ON WCST specifically.
- **DMTS: partial** (-0.30, better than WCST's -0.48 but still negative).

## What this means

The residual fix is correct and validated: where the agent's first-order
self-state actually moves, the self-model now learns to predict it better than
persistence (navigation last-200 +0.354, still trending up at run end, so longer
runs likely give higher skill). The Step-3 conclusion is refined: the self-vector
is not globally inert; it is inert on tasks (WCST, and largely DMTS) that do not
move the agent's first-order self-state.

This surfaces a real tension for the self-monitoring deliverable: WCST demands
self-monitoring behaviourally (detecting one's own performance drop after a rule
change) but does NOT move the current 14 first-order self-features, so the
self-prediction signal is inert there. Navigation moves the self-features but does
not demand self-monitoring.

## Options for the next step (for decision, not yet chosen)

1. **Phase B feature enrichment targeting WCST self-dynamics.** Add first-order
   features that DO move during WCST: reward-prediction-error, recent-accuracy /
   error signals, rule-change/feedback indicators. Then the self-state changes
   when the hidden rule changes, giving the self-prediction objective something
   to learn on the self-monitoring task. Re-measure skill on WCST.
2. **Test gating on navigation.** Re-run the gating ablation on navigation, where
   the self-vector now carries signal, to learn whether a working self-vector
   improves behaviour at all (decoupled from the WCST self-monitoring question).
3. **Reconsider the self-monitoring testbed.** Choose or design a task that both
   moves the self-state AND demands self-monitoring, so a single env exercises
   both halves of the deliverable.

## Honest caveats

- Single seed (seed 42), short runs (2500 steps; navigation skill still rising at
  the end). A multi-seed, longer navigation run would firm up the +0.35 figure.
- No defaults changed; `--enable-self-vector` stays off. This documents a
  mechanism improvement (residual prediction) and a measurement, not a new
  default.

## Reproducibility

```bash
for e in wcst dmts navigation; do
  PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env $e \
    --episodes 25 --max-steps 100 --enable-self-vector --seed 42 \
    --phi-sample-every 5 --log-dir runs/sv_residual/$e
done
```
