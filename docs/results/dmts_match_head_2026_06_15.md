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

---

## Update 2026-06-15: batched-head retry also FAILED

Tested the leading "training sparsity" hypothesis directly. Added a choice-record
replay buffer + mini-batch training (`--match-head-batched`, commit `c0e36a8`):
the acting head stores `(policy_state, target_position)` at every choice and trains
on mini-batches (64) every 10 steps instead of one single-sample SGD step per
choice. 3 seeds, DMTS, 80 ep x 200 steps (`runs/mhb/`, gitignored). num_choices=2.

| metric | single-sample acting | **batched acting** | baseline |
|--------|---------------------:|-------------------:|---------:|
| trials_correct (mean +/- std) | 1.43 +/- 0.05 | **1.50 +/- 0.05** | 0.39 +/- 0.29 |
| first-20 / last-20 | 1.45 / 1.48 | 1.52 / 1.60 | 0.52 / 0.47 |
| behavioral acc (trials_correct / ~3.4) | ~0.42 | **~0.44 (chance)** | - |
| head train/buffer acc (late) | ~0.47 | **0.697** | - |
| reward | -34.38 | -34.27 | -35.87 |

**Verdict: FAILED (3 seeds).** Batching did NOT make the head match correctly. The
behavioral metric `trials_correct` is 1.50 +/- 0.05, flat across training, ~chance
(0.44), and statistically the same as the single-sample head (1.43). The lift over
baseline is still the "always commits to a selection" artifact, not matching.

**The informative number: train accuracy plateaus at 0.70 while behavioral stays
at chance (0.44).** Two things follow. (1) The head does not generalize the
comparison to new trials (0.70 train vs 0.44 behavioral). (2) More telling, it
cannot even FIT the buffer cleanly (0.70, not ~1.0). A high-capacity conv that
plateaus at 0.70 on ~250 records suggests the buffer's `[obs;mem] -> target`
mapping is partly inconsistent, i.e. the live held sample is not reliably the
sample. This leans toward the **in-loop latch fidelity** explanation over pure
overfitting or training sparsity.

This RETRACTS the 2026-06-15 leading hypothesis ("sparse online training"): with
proper batched training the head still fails, so sparsity was not the cause.

### Honest correction to an earlier in-session claim

A 6-episode smoke reported 0.956 "accuracy", read at the time as encouraging. That
was the training/buffer accuracy on a tiny (~20-record) buffer that is trivially
memorized, not generalization. Over a real 80-episode run the train accuracy
settles to 0.70 and the behavioral accuracy is chance. The smoke number was
misleading and is corrected here.

### Decisive next diagnostic (gated, FAILED-first)

The offline probe got 0.845 (PCA-80 + MLP, held-out, one-per-trial). The in-loop
head gets ~0.44 behavioral and cannot fit the buffer past 0.70. To separate "the
head architecture/optimization is the problem" from "the in-loop signal is
degraded", **decode the captured in-loop choice records offline with the exact
PCA-80 + MLP protocol**:
- log the live `[obs;mem]` + `target_position` at each choice during a short run,
  then fit/test the offline classifier on those records (one per trial, leakage-free).
- If offline-on-in-loop-records reaches ~0.845, the signal is clean and the conv
  head is the bottleneck (use a PCA-bottleneck / MLP head, add regularization).
- If it is ~chance, the live `ObsMapSampleMemory` latch is degraded in-loop
  (fix the latch, not the head). The 0.70 train-accuracy plateau predicts this.

The mission-aligned PPO fix remains gated behind a successful head-capability
result, which has NOT been achieved in either the single-sample or batched
configuration.

---

## Update 2026-06-15: decisive offline decode (two compounding bottlenecks)

Ran the decisive diagnostic: captured the live training-loop obsmem-conv
policy_state `[current obs_map ; held sample]` + `target_position` at the first
choice frame of each trial (`train_rlhf.py --capture-choice-records`, 80 ep acting,
n=239, balanced), and decoded it offline two ways. Reproduced the prior "0.845"
under PROBE conditions (untrained components, scripted policy) with a new committed
script `scripts/analysis/probe_match_decode.py` (n=280, balanced) so probe-vs-in-loop
is apples-to-apples (same decoder, `scripts/analysis/decode_choice_records.py`).

| records | PCA-80 + MLP | PCA-80 + linear | conv MatchHead (offline, 200-300 ep) |
|---------|-------------:|----------------:|-------------------------------------:|
| **probe** (untrained, scripted), n=280 | **0.738** | 0.512 | train 0.582 / best-test 0.560 |
| **in-loop** (trained, acting), n=239 | **0.458** | 0.431 | train 0.737 / best-test 0.625 |

chance = 0.50-0.51. The conv "best-test" is the max over epochs (optimistic; with
~72-84 test samples it is within noise of chance).

### Two findings, both FAILED-first

1. **The in-loop signal is degraded.** Probe `[obs;mem]` decodes the match at
   0.738 (PCA+MLP), reproducing the prior ~0.845 qualitatively. The LIVE
   training-loop `[obs;mem]` decodes at **0.458 = chance**. So the representation
   the head actually sees at the decision does NOT carry the match. This is the
   primary reason every in-loop head (single-sample, batched) failed.

2. **The conv head is the wrong decoder.** Even on the CLEAN probe records, where
   PCA+MLP reaches 0.738, the conv `MatchHead` gets 0.56 = chance. The match is a
   non-local comparison (the held sample sits at the sample's spatial location; the
   choices sit at the choice locations), and a local conv with `AdaptiveAvgPool`
   cannot express "compare the sample to each choice region". The global PCA+MLP
   can. So the conv architecture would fail even if the in-loop signal were clean.

### What this corrects

The prior "the representation supports the match (0.845)"
(`rssm_working_memory_2026_06_12.md`, Final localization) holds ONLY under probe
conditions (untrained components, scripted policy). It does NOT transfer to the live
training loop, where the match is at chance. The 0.845 was an uncommitted
probe-condition number; `probe_match_decode.py` makes a reproducible version (0.738)
and `decode_choice_records.py` shows it does not hold in-loop.

### Not isolated

The cause of the in-loop degradation (0.738 probe -> 0.458 in-loop) is NOT pinned.
An early-vs-late split of the in-loop records (first half vs second half, capture =
episode order) was inconclusive: 0.528 vs 0.583, each within noise (n~36 test). So a
"trained-tectum-degrades-it-over-episodes" story is NOT supported. Candidates left
open: trained tectum (vs untrained probe), RSSM-state interaction, or a
latch/timing difference in-loop. To isolate: capture in-loop records with a
frozen/loaded tectum and with the latch instrumented, and decode each.

### Next (gated, FAILED-first)

The match-head approach is blocked by BOTH a degraded in-loop signal and an
inadequate conv decoder. A productive path must fix the signal first (the decoder is
moot if the in-loop `[obs;mem]` is at chance):
1. Isolate the in-loop degradation (frozen-tectum capture + latch instrumentation),
   then fix whichever stage destroys the match content.
2. Only then, replace the conv head with a global comparison head (flatten+MLP, or a
   sample-vs-choice correlation/attention head) that can express the non-local match.

The PPO fix remains gated behind a clean in-loop signal AND a head that can decode
it, neither of which holds. This connects to the project's standing finding that the
perception/tectum is trained by objectives (reward-MSE + TDANN) that are not aligned
with preserving task-relevant identity (2026-06-09/10 perception-collapse and
reconstruction results).

---

## Update 2026-06-16: in-loop degradation ISOLATED to tectum training

Ran the controlled A/B. `--freeze-tectum` (new flag) skips all tectum-encoder
training (reward-MSE + TDANN + control/recon), freezing the obs_map at init.
Captured a frozen-tectum run with the SAME seed (0), pipeline, and acting policy as
the trained-tectum capture, so the only difference is whether the tectum trains.
Decoded with the same protocol. All numbers from disk.

| records (seed 0 for in-loop) | n | PCA-80+MLP | conv MatchHead |
|------------------------------|--:|-----------:|---------------:|
| probe (untrained, scripted) | 280 | 0.738 | 0.56 |
| **frozen-tectum in-loop** (untrained, acting, full pipeline) | 234 | **0.746** | 0.535 |
| trained-tectum in-loop (trained, acting, full pipeline) | 239 | **0.458** | 0.625 |

chance ~0.50-0.52.

**Result: the in-loop degradation is caused by TECTUM TRAINING.** The controlled
pair frozen (0.746) vs trained (0.458), same seed / pipeline / policy, differs only
in whether the tectum encoder trains, and the match decodability drops by ~0.29. The
frozen-tectum in-loop (0.746) reproduces the probe (0.738), which also rules OUT the
pipeline and the `ObsMapSampleMemory` latch as causes (a frozen full pipeline matches
the tectum-only probe). The reward-MSE + TDANN objectives that train the retinotopic
encoder corrupt the obs_map's match-relevant content; this is the same misalignment
seen in the 2026-06-09/10 perception-collapse and reconstruction results, now shown
to actively DESTROY a task signal that the untrained encoder preserves.

Honest scope: single seed (0), but a clean within-seed controlled A/B with a large
effect (0.29, far outside the n~70 test-set noise). A 3-seed confirmation should
precede any default change. The conv head remains inadequate independently (0.535 on
the clean frozen records where PCA+MLP gets 0.746).

### The fix is now concrete (gated, FAILED-first, >= 3 seeds)

Both bottlenecks have actionable fixes:
1. **Signal**: read the match from a NON-trained obs_map. Either run the match
   pathway with `--freeze-tectum`, or stop-grad the obsmem tap, or replace the
   tectum training objective with one that preserves stimulus identity (active
   inference / identity-preserving reconstruction). The cheapest test is
   `--freeze-tectum`, which restores the 0.746 signal in-loop.
2. **Decoder**: replace the conv MatchHead (local conv + AdaptiveAvgPool, which
   cannot express the non-local sample-vs-choice comparison and gets chance even on
   clean records) with a GLOBAL head (flatten + MLP, like the PCA+MLP that decodes
   at 0.74).

Predicted capability test: `--freeze-tectum` + a flatten-MLP match head should drive
behavioral trials_correct toward ~0.74 x trials (vs the ~chance 1.5 of every prior
config), finally demonstrating in-loop DMTS matching. Only after that does the
mission-aligned PPO fix become worth pursuing.
