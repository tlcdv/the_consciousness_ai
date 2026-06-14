# RSSM working memory for DMTS (2026-06-11/12): the memory exists in h_state, but routing it to the policy FAILED to help matching

> ## CORRECTION (2026-06-14): the "h_state holds the sample at 99%" claim below is WRONG (train/test leakage)
>
> The central probe finding in this document, that the RSSM `h_state` decodes the
> DMTS sample at ~99% across the delay, is a TRAIN/TEST LEAKAGE ARTIFACT and is
> retracted. The probe decoded every delay step with a random train/test split.
> Consecutive delay frames within a trial are near-identical and share the same
> label, so the decoder memorized trials instead of generalizing across them.
>
> The leakage-free re-test (one record per trial, n=240, so no two records from the
> same trial; same `linear_decode`) shows `h_state` is at CHANCE, even when the
> stimulus is on screen:
>
> | representation | shape | color | chance |
> |----------------|------:|------:|-------:|
> | sample obs_map (on-screen, control) | 0.972 | 1.000 | 0.167 |
> | sample h_state | 0.208 | 0.167 | 0.167 |
> | delay h_state  | 0.111 | 0.139 | 0.167 |
>
> The `obs_map` control decodes the on-screen stimulus near-perfectly with the same
> n and method, so the test is valid; `h_state` simply does not encode the sample.
> The (untrained) RSSM does NOT build a usable working memory of the sample.
>
> Consequences: (1) the `--policy-input rssm` tap FAILED (verdict below stands) and
> the interference-protected latch (`models/self_model/working_memory_latch.py`) is
> moot, because there was never a retained sample to route or latch. Both mechanisms
> were built on this false premise. (2) The leakage-free finding is the one to carry
> forward: a working-memory capability for DMTS has to be BUILT (a mechanism that
> actually encodes and retains the sample), not merely wired, because the current
> RSSM does not retain it. (3) The probe's per-step decode of any recurrent/temporally
> correlated state is leakage-prone; decode one record per trial (or group-by-trial
> splits) for recurrent states. The on-screen `obs_map`/`tectum_content` findings in
> the perception docs are NOT affected (those are current-frame encodings, and the
> `obs_map` control here reconfirms `obs_map` is genuinely decodable).
>
> **Genuinely absent, not just non-linear (2026-06-14).** A non-linear decoder
> (PCA-50 + MLP, one-per-trial) also reads `h_state` at chance (sample shape 0.167,
> delay 0.139), while `obs_map` stays at 0.972 with the same pipeline. So the sample
> is genuinely absent from `h_state`; the `obs_map -> h_state` encoding discards it.
>
> **Build direction (this is the carry-forward).** A working-memory mechanism must
> capture from `obs_map` (which encodes the on-screen sample at ~1.0), NOT from
> `h_state` (empty). The `rssm` tap and the latch both used `h_state`, the wrong
> source. The remaining hard part is gating: distinguishing the sample (first
> stimulus) from the choices (second stimulus) without phase labels, which needs a
> learned gate (and therefore RL training) or a task-specific heuristic. Whatever is
> built must be validated leakage-free (one record per trial), never per-step.
>
> The original (leakage-inflated) write-up is kept below verbatim for the record.

## Resolution (2026-06-14): a gated obs_map memory DOES hold the sample at choice (leakage-free)

Acting on the corrected build direction (capture from `obs_map`, not `h_state`):
`models/self_model/working_memory_latch.py` gains `ObsMapSampleMemory`. It captures
the `obs_map` at the SAMPLE onset and holds it through the delay and choice. The gate
is causal and RL-free: it distinguishes the sample from the choices by the length of
the preceding blank (the sample follows the short ~10-step fixation blank, the
choices follow the long 15-40 step delay blank), capturing only on a
short-blank -> stimulus onset.

Leakage-free validation (one record per trial, n=240, `scripts/analysis/probe_wm_leakage_free.py`),
decode of the sample from the memory slot AT THE CHOICE PHASE:

| representation at choice | shape | color | chance |
|--------------------------|------:|------:|-------:|
| gated obs_map mem_slot | 0.875 | 0.889 | 0.167 |
| raw h_state (rssm tap / latch source) | ~0.11 | ~0.14 | 0.167 |

The gated `obs_map` memory holds the sample at the decision point at ~0.88, from the
source that actually encodes it. This is the first correctly-built, correctly-
validated working-memory result of the arc: it uses `obs_map` (not the empty
`h_state`) and is validated leakage-free (not per-step).

Honest scope. This validates the MECHANISM (the sample is available at the choice
phase). It does NOT validate behavior: whether a policy fed `mem_slot` learns to
match is an RL question, deferred (this laptop trains DMTS poorly: overnight sleep
stalls, ~6 min/episode late on the heavy taps). The gate is also DMTS-specific (it
keys on the fixation-vs-delay blank-length difference); a general working memory
needs a learned gate. Unit tests: `tests/test_working_memory_latch.py`.

After the 2026-06-10 investigation concluded that perception is not the bottleneck
for entering the consciousness-demanding regimes (the bottleneck is cognition:
working memory and rule inference, `obs_map_routing_2026_06_10.md`), this is the
first cognition target: working memory for DMTS. The method was localize-first (a
decodability probe of the recurrent state) before building, then a behavioral A/B.

## Probe finding (positive, and important): working memory already exists in the RSSM

Extended the perception-decodability probe to capture the RSSM deterministic
recurrent state `h_state` (the natural working-memory store) and to record the DMTS
delay and choice phases. Untrained components, seed 42. Decode of the held sample
during the delay (stimulus off-screen):

| label | obs_map | tectum_content | h_state | chance | n |
|-------|--------:|---------------:|--------:|-------:|---:|
| shape | 0.276 | 0.266 | **0.994** | 0.167 | 1074 |
| color | 0.245 | 0.248 | **0.997** | 0.167 | 1074 |
| size  | 0.536 | 0.536 | **0.994** | 0.500 | 1074 |

`h_state` holds the sample at ~99% across the blank delay, while `obs_map` and
`tectum_content` are at chance (the current frame is blank). So the RSSM recurrent
state does maintain working memory. The reason the policy never benefits is the same
capsule collapse the reconstruction experiment found
(`tectum_reconstruction_2026_06_10.md`): the policy reads `tectum_content`/`broadcast`
(post-collapse), which discard the held memory. The collapse destroys working memory
as well as current-stimulus identity.

## Choice-phase caveat (interference)

At the choice phase (decision point, choice stimuli on-screen), `h_state` decodes the
sample at chance (shape 0.250, color 0.167, size 0.500; n=40, noisy). The choice
stimuli appear to overwrite the sample in the recurrent state. n is small because the
choice phase is one step per trial, and an 8-episode re-probe OOM'd on this laptop
(16384-D arrays x thousands of frames), so this is suggestive, not conclusive. But
the mechanism (new stimulus overwrites the recurrent state) is physically expected.

## The fix tested: route h_state to the policy

`--policy-input rssm` (and `rssm-conv`) routes `h_state` to the policy's PFC. The bet:
the PFC's own gated GRU sees the sample for the 15-40 delay steps and latches it into
its own hidden state, holding it through the choice phase even as the RSSM's
`h_state` gets overwritten. Default `--policy-input broadcast` is unchanged (baseline
bit-identical).

## Verdict: FAILED

DMTS, seed 42, broadcast vs rssm, `trials_correct` (correct matches; +1.0 reward
each), all loaded from disk:

| arm | episodes | trials_correct (mean) | reward (mean) |
|-----|---------:|----------------------:|--------------:|
| broadcast | 100 (full) | 1.340 | -12.30 |
| broadcast | 59 (matched) | 1.407 | -12.32 |
| rssm | 59 | 0.627 | -26.70 |

The `rssm` tap makes DMTS matching worse, not better (0.627 vs broadcast's 1.407 at
matched episode count). Routing the held memory to the PFC does not let the policy
match; the PFC GRU does not latch the sample to improve behavior in this budget.

Robustness: the `rssm` arm was run three times (the first two stalled overnight when
the laptop slept; the third ground very slowly and was committed as a 59-episode
partial). All three agree: rssm `trials_correct` = 0.73 (45 ep), 0.79 (28 ep), 0.627
(59 ep), every value well below broadcast's stable ~1.34-1.57. broadcast does not
improve from 59 to 100 episodes (1.407 -> 1.340), so the partial does not understate
rssm by comparison. The verdict is robust to the missing 41 episodes.

## What this establishes and what is next (not built)

Established: working memory exists in the RSSM `h_state` (99% across the delay) and is
destroyed by the capsule collapse before the policy; and the simplest fix (route
`h_state` to the PFC and rely on its gate to latch) does not work, because the held
sample is also lost to choice-phase interference and the PFC does not latch it
unaided.

Next (not built this session): an explicit interference-protected latch, capture the
held representation while the world is blank (the delay), freeze it when stimuli
return (the choices), and feed the frozen sample to the policy at decision time. That
is what PFC working memory does biologically (gate the store against interference). A
learned update gate is the more general version.

## Reproduce

```
export PYPHI_WELCOME_OFF=yes
# working-memory probe (h_state decode at delay and choice phases)
python -m scripts.analysis.probe_perception_decodability \
    --episodes 2 --no-broadcast --envs dmts --seed 42 --out-dir runs/probe_rssm_memory
# behavioral A/B
python -m scripts.training.train_rlhf --env dmts --episodes 100 --max-steps 200 \
    --seed 42 --policy-input broadcast --phi-sample-every 5 --log-dir runs/p1_dmts_broadcast
python -m scripts.training.train_rlhf --env dmts --episodes 100 --max-steps 200 \
    --seed 42 --policy-input rssm --phi-sample-every 5 --log-dir runs/p1_dmts_rssm
```
