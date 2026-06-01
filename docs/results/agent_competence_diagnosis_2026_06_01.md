# P5 agent-competence diagnosis: policy or representation?

**Date:** 2026-06-01
**P5 (architecture audit).** The audit named agent competence as the bottleneck:
the consciousness agent is FLAT on DMTS/WCST and plateaus at 12.95 vs DQN's 92.00
on dark_room. This diagnosis isolates WHETHER the limiter is the policy algorithm
(the custom Go/No-Go actor-critic + stylized loss) or the broadcast representation
the policy acts on, before any fix. All numbers loaded from disk in-session.

## B1 - the gap (from `docs/results/experiment_comparison.md`)

| env | consciousness first-100 -> last-100 | DQN first-100 -> last-100 |
|-----|-------------------------------------|----------------------------|
| dark_room | 9.40 -> 12.95 | 52.72 -> 92.00 |
| DMTS | -9.82 -> -9.82 (flat) | -28.85 -> -4.07 |
| WCST | -1.94 -> -1.94 (flat) | 1.05 -> 2.06 |

The consciousness agent barely learns (flat on DMTS/WCST; marginal on dark_room).

## B2 - controlled probe: same broadcast, two policies

`--policy gonogo` (Go/No-Go `ActionSelectionCore`) vs `--policy standard`
(`StandardActorCritic`, a plain A2C on the SAME workspace broadcast). dark_room,
120 episodes x 100 steps, seed 42, everything else identical.

| policy | mean reward | first-30 | last-30 | max | positive episodes |
|--------|-------------|----------|---------|-----|-------------------|
| Go/No-Go | 14.80 | 16.88 | 16.00 | 100.0 | 31 / 120 |
| standard A2C | 15.45 | 17.55 | 14.18 | 100.0 | 58 / 120 |

## Verdict: the POLICY is NOT the bottleneck (FAILED hypothesis)

The "the Go/No-Go policy is the limiter" hypothesis is FAILED. A plain,
known-working A2C on the same broadcast is statistically tied with the Go/No-Go
core (mean 15.45 vs 14.80, within noise), and BOTH are ~10x below DQN-on-pixels
(92.00). Neither policy improves over the 120-episode run (Go/No-Go 16.9 -> 16.0;
A2C 17.6 -> 14.2). Swapping the policy does not move competence.

(The standard A2C reaches a positive-reward state more often, 58 vs 31 episodes,
but with the same mean reward, so it finds the light then fails to stay or
collects less per visit. Not a competence improvement.)

## Diagnosis: the bottleneck is the broadcast REPRESENTATION / credit assignment

DQN reaches 92 from raw PIXELS; a working RL algorithm on the BROADCAST (the
256-D abstraction produced by tectum -> workspace -> reentrant -> gate) plateaus
at ~15. So the broadcast is discarding control-relevant information that the
pixels contain.

Audit-grounded mechanism (`docs/architecture_audit_2026_05_31.md` section 2): the
perception / tectum is trained by an auxiliary reward-prediction MSE + a gate
diversity loss, NOT by the policy / control objective. The policy
(`action_core.update_policy`) trains separately on the broadcast but cannot send
gradient into the perception. So the broadcast is optimized for reward-prediction
and gate-diversity, never for being a controllable state representation. Any
policy on top inherits that ceiling, which is exactly what the probe shows.

## Fix proposal (next step, gated; >= 3 seeds before any default)

1. **Confirm the representation loss directly:** run a DQN (or the standard A2C)
   on the broadcast vs on raw pixels. If the strong learner also fails on the
   broadcast but succeeds on pixels, the representation loss is confirmed.
2. **Fix:** let the control objective shape perception, the single most likely
   lever. Either (a) allow the policy gradient to flow into the tectum/broadcast
   (end-to-end), or (b) add a control-relevant representation objective (e.g.
   predict action-conditioned next-state / value from the broadcast). Keep behind
   a flag; measure on dark_room (where DQN proves the task is learnable).

This is the opposite of the obvious first guess (rewrite the policy), which is
why the diagnosis was worth running before committing effort.

## Honest caveats

- Single seed (42); 120 episodes (short; neither policy showed a learning trend,
  so this run does not characterize asymptotic performance). The robust claim is
  the RELATIVE one: a standard policy on the broadcast does not beat the Go/No-Go
  core, and both are far below DQN-on-pixels. That is enough to rule the policy
  out as the primary bottleneck.
- No defaults changed; `--policy` stays `gonogo`. `StandardActorCritic` remains a
  diagnostic.

## Reproducibility

```bash
for p in gonogo standard; do
  PYPHI_WELCOME_OFF=yes python -m scripts.training.train_rlhf --env dark_room \
    --episodes 120 --max-steps 100 --policy $p --seed 42 --phi-sample-every 5 \
    --log-dir runs/p5_probe/$p
done
```
