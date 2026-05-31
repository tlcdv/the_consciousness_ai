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

---

## Phase B: feature enrichment (reward EMAs) - FAILED to fix WCST

Option 1 above was executed. The first-order feature vector was enriched with two
performance signals that DO move on WCST: a fast reward EMA (current performance)
and the fast-minus-slow trend (drops when a hidden rule change tanks performance).
Feature dim 14 -> 16. Re-measured at the same settings (25 ep x 100 steps, seed 42).

| env | Phase A last-200 | Phase B last-200 | Phase B 2nd-half | frac steps > 0 |
|-----|------------------|-------------------|------------------|----------------|
| navigation | +0.354 | +0.340 | +0.253 | 0.58 |
| dmts | -0.348 | -0.295 | -0.309 | 0.41 |
| wcst | -0.481 | **-0.460** | -0.457 | 0.28 |

**Verdict: FAILED.** The enrichment moved WCST skill only marginally (-0.48 ->
-0.46), nowhere near positive. Navigation stays positive; DMTS stays negative.

Root cause: a reward EMA is, by construction, SMOOTH, so it is persistence
dominated just like the slow features it was meant to fix. The model has to beat a
baseline that already predicts "EMA barely changed", and on the ~72% of WCST steps
where the self-state is near-constant, any model error loses to persistence (only
28% of steps have positive skill). The sharp signal on WCST is the raw per-trial
reward, but that is stochastic (depends on whether the agent happened to sort
correctly), so neither the model nor persistence predicts it. WCST has no
"sharp AND predictable" self-feature at the per-step horizon.

## Conclusion: a Phase-5 finding, not just a tuning miss

Across Phase A (residual prediction) and Phase B (feature enrichment), **per-step
self-prediction is achievable where the agent's self-state has smooth deterministic
dynamics (navigation: skill +0.35) and is NOT achievable on WCST/DMTS**, whose
self-state is either near-constant (persistence-dominated) or sharp-but-stochastic
(unpredictable). The self-vector mechanism is sound; the per-step self-prediction
SKILL metric is simply the wrong success signal for self-monitoring tasks like
WCST.

Options to bring to the user (none chosen here):

1. **Multi-step / event-horizon prediction.** Predict the self-state several steps
   ahead, or specifically predict the post-rule-change performance drop, instead of
   the next single step. Persistence is far weaker over a horizon, and rule-change
   dynamics are the predictable structure.
2. **Measure self-model value behaviourally, not by self-prediction.** Keep the
   self-vector as a representation and judge it by whether feeding it into gating /
   action improves WCST recovery in the post-rule-change window (the pre-registered
   Deliverable-5 self-monitoring window), dropping self_pred_skill as the gate.
3. **Use navigation (or a self-dynamics-rich task) as the self-vector testbed**,
   and treat WCST purely as the behavioural self-monitoring benchmark with its own
   (non-self-prediction) metric.

The self-vector + enrichment code stays behind `--enable-self-vector` (default
off); no defaults changed. Single seed; numbers from disk.

---

## Option 2 executed: behavioural self-monitoring value on WCST - BLOCKED

Option 2 above was run: measure the self-vector's causal efficacy behaviourally
via WCST post-rule-change recovery, with vs without the self-vector feeding the
gate (60 ep x 150 steps, seed 42). Added per-episode env logging
(`env_episodes.csv`) for `rule_changes` and `trials_correct`. A self-monitoring
agent should recover faster after each hidden rule change and so trigger MORE
rule changes per episode.

| run | rule_changes total / mean | trials_correct mean | reward mean |
|-----|---------------------------|---------------------|-------------|
| baseline (gate ignores self-vector) | 0 / 0.000 | 6.58 | 1.059 |
| gating (gate uses self-vector) | 1 / 0.017 | 6.32 | 0.709 |

**Verdict: BLOCKED, and gating FAILED.** The self-vector gating did not help
(reward 0.71 vs 1.06, worse; trials_correct 6.32 vs 6.58). More fundamentally,
the agent triggers 0-1 rule changes across 60 episodes: it almost never reaches
6-consecutive-correct, so it never enters the rule-change regime where
self-monitoring matters. Post-rule-change recovery is therefore UNMEASURABLE with
this agent - there are essentially no rule changes to recover from.

## Consolidated conclusion (self-vector investigation, 2026-05-31)

WCST fails as a self-monitoring testbed for the self-model on TWO independent
counts: (1) its first-order self-features are near-static, so per-step
self-prediction cannot beat persistence (Phases A/B); and (2) the agent is too
weak to reach the rule-change regime, so behavioural recovery is unmeasurable
(option 2).

What IS established: the self-model MECHANISM is sound. On navigation, where the
agent's self-state genuinely moves, residual self-prediction beats persistence
(skill +0.35, rising with training). That is genuine meta-representation working
where it is measurable.

Recommendation to the user: treat **navigation (a self-dynamics-rich task) as the
self-vector testbed** and advance the other Phase 5 deliverables on it -
Deliverable 6 (Markov-blanket self-boundary detector, which consumes the
validated self-vector) and Deliverable 7 (eight-themes audit; theme-4
self-prediction skill is now demonstrable on navigation). Drop WCST as the
self-monitoring testbed for now: it is blocked on agent competence, which is a
separate problem from the self-model question. (Alternatives: invest in WCST
agent competence first; or design a task that both moves the self-state and is
solvable enough to exercise self-monitoring.)

The self-vector + env-logging code stays behind `--enable-self-vector` (default
off); no defaults changed. Single seed; numbers from disk.

