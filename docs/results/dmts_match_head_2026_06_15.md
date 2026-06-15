# DMTS supervised match head: verdict (2026-06-15)

**FAILED.** Neither the acting head nor the aux head solves the DMTS match
in-loop. The acting head operates at chance (~0.47 for 2 choices), not the 0.845
offline decodability, with no learning trend over 80 episodes. The aux head does
not lift the RL policy above the baseline. All numbers below were loaded from
disk (`runs/mh/`, gitignored) in the session that wrote this doc.

## Setup

- Code: commit `bd3bbba` (`--enable-match-head`, `models/self_model/match_head.py`).
- 3 arms x 3 seeds (0, 1, 2), DMTS, 80 episodes x 200 max-steps,
  `--policy-input obsmem-conv --phi-sample-every 5`. GPU (RTX 4090), ~32 s/ep.
- DMTS defaults: `num_choices=2` (chance 0.5/trial), `num_trials=10`,
  fixation 10 + sample 20 + delay 15-40 + choice steps per trial, so ~3-4 trials
  complete within 200 steps.
- Arms: `baseline` (obsmem-conv Go/No-Go RL, no head), `acting` (head argmax
  drives the choice), `aux` (linear head on the shared PFC trunk; RL still acts).

## Results

Per-episode `trials_correct` (env's own metric) and `total_reward`, averaged
across the 3 seeds:

| arm | trials_correct per seed | mean +/- std | first-20 | last-20 | reward |
|-----|-------------------------|--------------|----------|---------|--------|
| baseline | 0.38, 0.75, 0.04 | 0.39 +/- 0.29 | 0.52 | 0.47 | -35.87 |
| acting   | 1.36, 1.44, 1.48 | **1.43 +/- 0.05** | 1.45 | 1.48 | -34.38 |
| aux      | 0.48, 1.06, 0.05 | 0.53 +/- 0.42 | 0.48 | 0.63 | -35.67 |

Match-head choice accuracy (step-weighted proxy from `match_head_acc`; carries
forward between choice steps, so it is approximate but robust at this magnitude):

| arm | head acc per seed (overall) | late-run | 
|-----|-----------------------------|----------|
| acting | 0.463, 0.488, 0.493 | ~0.47 |
| aux    | 0.461, 0.479, 0.508 | ~0.47 |

## Verdict

**acting: FAILED the capability / pipeline-ceiling test.** The head's in-loop
choice accuracy is ~0.47, i.e. chance for 2 alternatives, and is flat from
first-20 to last-20 episodes (1.45 -> 1.48 trials_correct). It does NOT reproduce
the 0.845 offline decodability. The higher trials_correct vs baseline (1.43 vs
0.39) is an artifact of the head reliably committing to a selection at every
choice (~chance x ~3 trials/episode = ~1.4), whereas the RL baseline often fails
to select (waits to timeout, scoring 0). Cross-checks agree the head is at
chance: the accuracy proxy (~0.47) and the ratio trials_correct / trials
(1.43 / ~3.4 ~= 0.42) both sit at ~0.5.

**aux: FAILED.** trials_correct 0.53 +/- 0.42 overlaps the baseline 0.39 +/- 0.29
within the (large) seed variance. The dense supervised gradient on the shared PFC
trunk did not lift autonomous RL. Seed variance is high for both (baseline and
aux each have one near-zero seed), so no aux advantage is supported.

## Why (hypotheses, not asserted causes)

The offline probe (PCA-80 + MLP, n~=280, held-out split;
`docs/results/rssm_working_memory_2026_06_12.md`, Final localization) already
PROVED the `[current obs_map ; held sample]` representation supports the match at
0.845. The signal is present. So the in-loop head failing at chance points at the
training procedure or the live input fidelity, not signal absence:

1. **Sparse online training (most likely).** The head gets one single-sample SGD
   step per choice, ~3 per episode, ~240 total over 80 episodes, with no batching
   and no replay. That is a far weaker supervised setup than the offline batched
   classifier. The flat accuracy trajectory is consistent with undertrained,
   high-variance single-sample updates.
2. **Live latch fidelity (secondary).** The offline 0.88 was measured on captured
   choice records. The in-loop `ObsMapSampleMemory` gate (blank-run-length
   heuristic) could capture a different or noisier slot during these runs. To
   test, capture the in-loop choice `[obs;mem]` and re-decode offline.

## Next (gated, FAILED-first, >= 3 seeds before any conclusion)

1. **Strengthen the head training before concluding the signal does not
   transfer.** Accumulate choice records into a replay buffer and train the head
   with mini-batches / several epochs per episode (mirroring the offline
   classifier). If trials_correct then approaches ~0.845 x trials, the failure was
   training sparsity. If it stays at chance, the live latch fidelity is the
   culprit (diagnose by re-decoding captured in-loop records, one per trial,
   leakage-free).
2. The mission-aligned autonomous fix (replace the stylized Go/No-Go loss in
   `models/self_model/action_selection_core.py` with a proper stochastic-policy
   gradient / PPO) remains the documented follow-on. The aux result (no lift from
   representation shaping) does not yet motivate it; it should follow a successful
   head-capability result, which has not been achieved.

## Honest scope

Single configuration (80 ep, online single-sample head). The acting head DID
confirm the action-override path works end to end (it reliably drives a selection
at the choice phase). What it did NOT do is match correctly: it is at chance. The
negative result is robust across 3 seeds (acting std 0.05). The "why" items are
hypotheses to test, not established causes.
