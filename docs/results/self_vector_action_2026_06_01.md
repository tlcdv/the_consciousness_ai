# P3 validation: does making the self-vector causally central help navigation?

**Date:** 2026-06-01
**P3 (architecture audit) validation.** P3 made the learned self-vector causally
central by concatenating it onto the broadcast that drives the policy PFC
(`--enable-self-vector-action`, default off). This measures whether that helps
behaviour on navigation, where the self-vector is a validated self-model
(self-prediction skill +0.35, `self_prediction_residual_2026_05_31.md`).

All numbers loaded from disk in the same session that produced them.

## Method

Two runs, identical except whether the self-vector feeds the policy PFC. Both
compute and train the self-vector; only the intervention feeds it into action
selection. Same seed, same everything else.

| run | flag | self-vector feeds policy? |
|-----|------|---------------------------|
| baseline | `--enable-self-vector` | NO |
| action | `--enable-self-vector-action` | YES |

Both: `--env navigation --episodes 50 --max-steps 120 --seed 42 --phi-sample-every 5`.

## Results

| run | mean reward | first-20 | last-20 | positive episodes |
|-----|-------------|----------|---------|-------------------|
| baseline | -1.071 | -1.039 | -1.139 | 1 / 50 |
| action | -1.015 | -0.988 | -1.094 | 2 / 50 |

## Verdict: FAILED to show benefit

Feeding the self-vector into the policy did NOT measurably improve navigation
reward: mean -1.015 vs -1.071 (a +0.056 difference, within noise), positive
episodes 2 vs 1. Both runs are firmly negative (mean ~ -1.0): the agent performs
poorly on navigation regardless of the self-vector.

## What this does and does not establish

- **Coherence (achieved).** P3's goal was to stop the self-model being a side
  module and make it causally central. That is now true by construction: the
  self-vector is concatenated into the PFC input and the policy learns from it
  (the PFC GRU input weights for the self-vector columns train via update_policy).
- **Behavioural value (not demonstrated).** The self-vector does not improve
  navigation reward at single seed. This is consistent with the architecture
  audit's P5 finding: the base agent is too weak (mean navigation reward ~ -1.0;
  dark_room 12.95 vs DQN 92.00; WCST 0-1 rule changes) for a self-model input to
  make a measurable behavioural difference. The bottleneck is agent competence,
  not the self-model wiring.

## Honest caveats and scope

- Single seed (seed 42). Per the project's >=3-seed rule, this is a
  hypothesis-level signal, not a verdict. But the result is null and the agent is
  weak, so chasing more seeds is not warranted before P5 (agent competence) is
  addressed.
- No defaults changed; `--enable-self-vector-action` stays off.

## Conclusion for the self-model arc

The self-model consolidation/coherence work is complete and honest: P1a (pruned
6 orphaned files), P2 (one orchestrator story), P3 (self-vector causally central),
P4 (EI/Levin/RIIU/self-prediction demoted to diagnostics in the docs). Capability
gains from the self-model now require P5 (a competent base agent) first; until
then the self-model is coherently wired but behaviourally inert because the agent
cannot exercise the tasks.

## Reproducibility

```bash
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env navigation \
    --episodes 50 --max-steps 120 --enable-self-vector --seed 42 \
    --phi-sample-every 5 --log-dir runs/sv_action/baseline
PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env navigation \
    --episodes 50 --max-steps 120 --enable-self-vector-action --seed 42 \
    --phi-sample-every 5 --log-dir runs/sv_action/action
```
