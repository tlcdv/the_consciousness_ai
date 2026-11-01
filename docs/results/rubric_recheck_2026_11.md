# FAILED: under the empirical rule, RPT-1 and GWT-3 pass no test, and GWT-1 passes only in its narrow text in the learning runs

Date: 2026-10-31. Read-only, on recorded weights. No run was trained and no weight changed. The rules are in
the docstring of `scripts/analysis/probe_rubric_recheck.py`, pushed before the first replay of each group
(`c1e6e5f` for the covert runs, `bc46b95` for the learning runs). **No score changes here.** The rubric
stays at 3 IMPLEMENTED, 11 PARTIAL, 14 total until the owner decides.

## Verdict, one row per indicator

Training mode, the mode of the live runs. The two groups are never pooled.

| Indicator, score today | Covert runs `ta_nolink_s801..803`, near starting weights | Learning runs `vg_learn_s601..603`, trained | Main measure |
|---|---|---|---|
| RPT-1, IMPLEMENTED | **FAILED**, the tectum claim holds at 0 of 3 seeds | **FAILED**, it holds at 1 of 3 seeds (602) | A carried state from another episode changes the vision content by 1.36e-06 to 1.85e-06 in the covert runs, against sampling noise of 0.0072 to 0.0082. In the learning runs it gives 4.96e-07, 0.0437 and 0.0671, against 0.000205, 0.0121 and 0.0878 |
| GWT-1, IMPLEMENTED | **FAILED**, 1 module of 5 counts (audio) | Row text **FAILED**, 2 modules of 5 count. Indicator text PASS at 3 of 3 seeds, with a limit stated below | Memory, body and semantic bids have 1 distinct value at 8000 of 8000 steps at all 6 seeds |
| GWT-3, IMPLEMENTED | **FAILED**, 0 of 3 seeds | **VOID** by control V2 at seeds 602 and 603 | In both groups and both modes, a perturbed broadcast changed the content and the raw bid of no module at any step, 0 of 38400 compared readings |

The other claims of the RPT-1 row, each with its own verdict.

| Claim of the row | Covert | Learning | Measure |
|---|---|---|---|
| Audio module recurrence (R2a) | FAILED | FAILED | Audio content after a swapped sound changes by exactly 0 at all 6 seeds |
| Settle cycles (R3) | FAILED | FAILED | With 1 cycle or with no feedback, module content changes at 0 steps at all 6 seeds |
| Capsule feedback (R4) | FAILED | FAILED | With no top down iterations, the change is below sampling noise at all 6 seeds |

**The covert verdicts need one reading.** In those runs the content barely follows anything. Frames from
another layout change it by 8.95e-06 to 1.46e-05 (`content_map_sensitivity_2026_11.md` finds the content
equal to the projection bias there). So their RPT-1 result cannot separate "no recurrence" from "a static
content path". It is still a true result about the output of the module. The rule is not edited for this,
and the learning runs and the upstream readings below are the data that can separate the two.

## Sources of the numbers

| Numbers | File under `runs/_audit_followup/rubric_recheck/` (local, gitignored) | Revision |
|---|---|---|
| Covert verdicts and tables | `result.json` | `3f94f0a` |
| Learning verdicts, tables and upstream readings | `learning/result.json` | `bc46b95` |
| Covert upstream readings | `covert_upstream/result.json` | `bc46b95` |
| Exact counts, with no tolerance | `exact_counts.json`, from `--count-arrays` | this PR |
| Per step readings | `readings_s<seed>_<mode>.npz` in each folder | as above |
| Record of each run | `facts` in each `result.json`, from `metrics.csv` and `steps.jsonl` | |

The training loop has no change between the two revisions. The diff of `scripts/training/train_rlhf.py`
touches only wiring after `run_episode` and behind flags that are off here.

## What the sources ask, and how far I read them

Paraphrased. No quotation.

- **RPT-1.** Input modules use algorithmic recurrence. The same operation is applied again, and what is
  retained from the past acts on present processing.
- **GWT-1.** Several specialised systems that can work in parallel. They need not be encapsulated.
- **GWT-3.** What the workspace holds is available to all modules. All modules take workspace states as
  input, the modules that process input included, so information flows back and influences their processing.

| Source | Reading depth |
|---|---|
| Butlin et al. 2023, arXiv 2308.08708v3 | Main text of sections 2.1, 2.2.1 to 2.2.3, 2.4.6 (the part on recurrence), 2.5 (the list), 3.1.1, 3.1.2 |
| Butlin et al. 2025, doi 10.1016/j.tics.2025.10.011 | The article text up to the acknowledgments, in the published version at LSE Research Online, eprint 130322. The publisher pages stopped at a bot check, which I did not pass |

**A local standard is not the source's standard.** The 2023 report calls RPT-1 a weak condition that many
systems meet. The 2025 article says that many indicators can be inferred from architecture and training.
Neither asks for an effect above sampling noise at a module's output. That bar is mine, set for the local
rule of 2026-08-11, which asks for empirical evidence of the property.

## Existing evidence, read first

| Document | What it already answered |
|---|---|
| `workspace_bids_live_2026_08.md` | In DMTS, four of five bids are constant. These runs repeat that for memory, body and semantic in the dark room |
| `top_winner_diagnosis_2026_10.md` | The feedback rule raises the bid of the module that is not in the broadcast, and vector size plays a part |
| `broadcast_carrier_2026_10.md` | Which vector the policy receives under each merge |
| `tectum_wave_detection_2026_09.md` | The zeroed `h_state` control, used here as R1 |
| `action_core_relay_defect_2026_10.md` | The default policy output does not depend on its input. Its relay weights changed in 0 of 172 runs |
| `content_map_sensitivity_2026_11.md` | Why the content barely follows the map, with three measured causes and two regimes |

Those documents could not answer the three tests. No record holds an arm where one input or state changes and the
noise stays the same, so the tests needed a replay.

## How the re-check ran

`run_episode` runs unchanged on the recorded config, weights of the end of episode 39, and the recorded
frames, sounds, rewards and step facts of episodes 38 (decoy level 3.0) and 39 (level 0.5). One change is
applied per arm, and each arm uses the same noise as the base replay. Steps 0 to 29 are left out. The full
list of arms and the differences from a live run are in the docstring. The main ones are these.

- Open loop. The room ignores the action, so a change cannot alter the next frame.
- Frozen weights. The sound predictor does not learn, so the audio bid in episode 38 differs from the record.
- The running statistics of the bids and of the ignition start at their live values, from the recorded series.

A change is labelled against the same change between two noise streams. It is **absent** at or below 1e-6,
**below noise** under that floor, and **above noise** at or over it. A claim is PASS at 3 of 3 seeds, FAILED at
2 or more failing seeds, UNDECIDED at 1.

**The second group is an addition, stated with its date.** I read the covert verdicts, then the document
named above showed that those runs stayed near their starting weights. So I added the learning runs and
the upstream readings in `bc46b95`, before any replay of the learning runs. No pushed rule changed.
The code at `bc46b95` printed GWT-3 FAILED beside V2 false, which the rule calls VOID. A later commit makes
the code apply the control. The numbers do not change.

## Controls

| Control | Covert | Learning |
|---|---|---|
| V0, two equal replays, largest change | 0 at 3 of 3 seeds, both modes | 0 at 3 of 3 seeds, both modes |
| V1, median cosine of replay and record, `audio_content` (rule 0.95) | 1.0 | 1.0 |
| V1, `tectum_content` (rule 0.90) | 0.99996 to 0.99997 | 0.99857 to 0.99992 at 601 and 602, 0.94083 and 0.94768 at 603 |
| V1, `obs_map`, episode 39 (rule 0.95) | 0.99954 to 0.99982 | 0.99137 to 0.99876 |
| V2, the gate changes at lag 0 in both broadcast arms | Holds at 3 of 3 seeds | **Fails at 602 and 603**, in `bc_other` |
| Second count, separate script, no import from the probe | 582 numbers agree | 870 numbers agree, and 288 of the covert upstream run |
| Covert arms run again at the later revision | 756 readings compared, 0 differ | |

Reported only. Agreement of the winner with the record is low in places, 0.524, 0.588, 0.524 in covert
episode 38 and 0.776, 0.800, 0.782 in episode 39. In the learning runs it is 0.624, 0.676, 0.176 and 0.935,
0.888, 0.753. Two replays that differ only in noise disagree on ignition at 0.200 to 0.226 of the steps in the
covert runs and 0.018 to 0.035 in the learning runs, so noise explains part of this. The rest is not tested. The
frozen sound predictor is one candidate, because the audio bid differs from the record by 0.249 to 0.264
(covert) and 0.164 to 0.223 (learning) in episode 38, against 0.006 to 0.017 in episode 39. At seed 603 the
replayed content differs from the record by 0.54 to 0.56 of its size. No verdict reads the winner, because
every arm is compared with the replay.

## RPT-1. Does recurrence change the output?

Effect against sampling noise, in the order of the seeds.

| Test | Covert 801, 802, 803 | Learning 601, 602, 603 |
|---|---|---|
| R1, `h_state` zeroed, vision content | 0.000234, 0.000214, 0.000215 against 0.00847, 0.00736, 0.00760. Below noise | 0.000228, 0.330, 0.849 against 0.000211, 0.0114, 0.0849. Above noise |
| R2, state of the other episode, lag 0, vision content | 1.85e-06, 1.74e-06, 1.36e-06 against 0.00821, 0.00731, 0.00719. Below noise | 4.96e-07 (absent), 0.0437 (above), 0.0671 (below) against 0.000205, 0.0121, 0.0878 |
| R2a, sound of the other episode one step before, audio content | 0, 0, 0. Absent | 0, 0, 0. Absent |
| R3, `settle_off` or `feedback_off`, vision and audio content | 0 at every step. Absent | 0 at every step. Absent |
| R4, `capsule_off`, vision content | 9.98e-05, 1.49e-04, 8.23e-05 against the same floors as R1. Below noise | 1.04e-04, 1.24e-03, 2.50e-03 against the same floors as R1. Below noise |

Reported beside the rules.

- **The vision bid reads the carried state.** With `h_state` zeroed it changes by 0.188, 0.218, 0.216 (covert) and
  0.156, 0.0433, 0.0716 (learning), above noise at 6 of 6 seeds. With the state of the other episode it changes
  by 0.000291 to 0.00041 (covert, below noise) and 0.000205, 0.0142, 0.0257 (learning, above noise at 602 and 603).
- **The audio bid has a history and the audio content has none.** A sound one step back changes the bid by 0.264
  to 0.357 (covert) and 0.173 to 0.402 (learning), through the previous band profile of the surprise predictor.
- **R1 at seed 601 is marginal**, 0.000228 against 0.000211.
- **What settle cycles do.** They change which vector the workspace holds, in one regime only. Exact counts of
  ignited steps with another broadcast, in training mode.

| Episode | Covert 801, 802, 803 | Learning 601, 602, 603 |
|---|---|---|
| 38, decoy level 3.0 | 88 of 119, 76 of 119, 88 of 118 | 0 of 117, 0 of 120, 0 of 114 |
| 39, decoy level 0.5 | 3 of 145, 0 of 149, 2 of 147 | 0 of 109, 0 of 117, 0 of 121 |

In the record of the whole runs, the feedback changed the broadcast at 2020, 1891 and 1820 of 8000 steps
(covert) and 90, 242 and 80 of 8000 (learning). The settle loop repeats the competition and a bid rule. It does not
run any module again, so by these counts it is not recurrence inside an input module.

**Upstream readings, reported only.** Relative change of `h_state` after the swap, against sampling noise in
training mode.

| Lag | Covert 801, 802, 803 (floor 1.39 to 1.40) | Learning 601, 602, 603 (floors 0.115, 0.513, 0.344) |
|---|---|---|
| 0 | 0.00347, 0.00342, 0.00236 | 0.00296, 0.862, 0.258 |
| 5 | 0.000171, 0.000169, 0.000130 | 0.00295, 0.863, 0.252 |
| 19 | 7.69e-05, 7.40e-05, 7.63e-05 | 0.00295, 0.864, 0.223 |

Evaluation mode, with a floor of 0, gives the same shape. Covert 0.0186, 0.0175, 0.0135 at lag 0 and 7.22e-05,
6.59e-05, 7.14e-05 at lag 19. Learning 0.00261, 0.832, 0.264 at lag 0 and 0.00260, 0.834, 0.238 at lag 19.

- The state carries its past in both groups, and `obs_map` under the same swap changes by exactly 0, as it must.
- In the covert runs the difference falls by about half per step and reaches a level near 7e-05 by lag 10.
- In the learning runs it does not fall in 19 steps. I did not test why. At 602 the effect on the content does not
  fall either (0.0428 against 0.00939 at lag 19).
- Only at seed 602 is this state difference above sampling noise in training mode.
- Earlier evidence limits what the state holds. In DMTS, `h_state` decoded the sample at chance
  (`rssm_working_memory_2026_06_12.md`), and in DMTS at 2 of 3 seeds a phase gradient did not need the carried state
  (`tectum_wave_detection_2026_09.md`).

## GWT-1. Does each module's output depend on its own input?

| Module | Its input in this room | Distinct bids in `metrics.csv`, covert and learning | Replay, its own input replaced | Counts |
|---|---|---|---|---|
| Vision | Frames | 7499 to 7887, 7727 to 7878 | Covert content 1.46e-05, 1.02e-05, 8.95e-06 and bid 0.0240, 0.0233, 0.0167, all below noise. Learning content 1.20e-06 (below), 0.0434 (above), 0.0647 (below), and bid 0.0353, 0.0181, 0.0363, above noise | Covert no. Learning yes |
| Audio | Sound | 7904 to 7929, 7924 to 7941 | Content 0.0457, 0.0423, 0.0511 and 0.0337, 0.0266, 0.0390, with no sampling in that path. Bid 0.257, 0.203, 0.320 and 0.279, 0.167, 0.242 | Yes at 6 of 6 |
| Memory | The previous broadcast | 1, the bid is 0.1 at 8000 of 8000 steps | Bid changes by exactly 0. The query raised an error at 199 of 199 steps per episode at all 6 seeds | No |
| Body | Energy | 1, the bid is 0.05 at 8000 of 8000 steps | Reported only, a battery of 0.2 moves the bid by 0.1. The recorded battery stays between 0.800 and 0.999 at all 6 seeds | No |
| Semantic | None | 1, the bid is 0 at 8000 of 8000 steps | `SemanticPathway.forward` runs 0 times, and its content is all zeros | No |

Vision and audio each compute once per step, 200 and 199 times per episode (a reset gives no sound at step 0).

**The limit on the PASS in the learning runs.** The vision module counts at seeds 601 and 603 through its bid only,
and that bid follows the sound more than the frames. Sound of the other episode moves the raw vision bid by 0.0911,
0.0372, 0.0682 in the learning runs and 0.189, 0.231, 0.241 in the covert runs. Frames of the other episode move it
by 0.0353, 0.0181, 0.0363 and 0.0240, 0.0233, 0.0167. The cause by code is `--bid-precision gain`, which weights
each sense bid by its share of the sense gain. The rule tests specialisation on content, where it holds. The
vision content follows the frames above sampling noise at 1 of 6 seeds, seed 602.

**The frames reach the map at all 6 seeds.** `obs_map` changes by 0.0458, 0.0452, 0.0336 (covert) and 0.0387, 0.297,
0.258 (learning) under `frame_other`, with no sampling in that path. The posterior mean changes by 0.0247, 0.0219, 0.0169
against noise of 0.480 to 0.505 (covert) and by 0.00190, 0.641, 0.108 against 0.0207, 0.215, 0.153 (learning). So the
break is after the map, as `content_map_sensitivity_2026_11.md` measures, and only seed 602 clears the noise at the
posterior. These upstream numbers are reported only.

**A defect found on the way.** On the GPU, `MemoryCore.get_similar_experiences` raises a device error at every step
(`torch.cat` of a CUDA query and a CPU emotion vector), and the `except Exception: pass` in `run_episode` hides it.
The bid would stay at 0.1 anyway, because the stub returns a score of 0. With retrieval on, an earlier result puts
the bid at its cap (`memory_retrieval_repair_2026_08.md`). I changed nothing here.

## GWT-3. Does a perturbed broadcast reach the modules?

The feedback path, in the loop and in the code. Calls per episode in the base replay, training mode.

| Module | `receive_broadcast` calls | What the rule reads and writes |
|---|---|---|
| Vision | 317 to 330 (covert 38), 200 (covert 39), 200 to 205 (learning) | The distance of the broadcast from its own vector. It returns a bid and writes no state (`sensory_tectum.py`) |
| Audio | 1 fewer than vision, no sound at step 0 | The same rule (`auditory_specialist.py`) |
| Body | 0. `ProprioceptiveProcessor` is not built in the loop | Keys and words of the broadcast, never the tensor (`embodiment_core.py`) |
| Semantic | 0. `specialists` in `run_episode` holds vision and audio only | For the dict that the workspace sends, it returns the bid unchanged (`semantic_pathway.py`) |
| Memory | No such rule | Its query reads the previous broadcast, and it raises on the GPU |

`tests/test_rubric_recheck_probe.py` pins each of these reads.

The perturbation is a random direction of the same size (`bc_random`) or the vector of the other sense (`bc_other`),
in every settle cycle, at ignited steps from 30, every 10 steps.

| Reading | Covert 801, 802, 803 | Learning 601, 602, 603 |
|---|---|---|
| Steps perturbed, `bc_random`, `bc_other` | 28, 26, 28 and 28, 26, 28 | 17, 27, 30 and 19, 27, 29 |
| Steps where the gate changed, `bc_random` | All | All |
| Steps where the gate changed, `bc_other` | 20, 18, 20 | 13, 4, 9 |
| **Next step content and raw bid of each module, median change** | 0 in both arms. Absent | 0 in both arms. Absent |
| **The same over all 400 steps, vision, audio and semantic content and the 5 raw bids** | 0 steps changed | 0 steps changed |
| Same step, final bid of vision, `bc_random` | 0.139, 0.138, 0.139, above noise | 0, 0.142, 0.137 |
| Same step, final bid of audio, `bc_random` | 0.000280, 0.000299, 0.000135, above noise | 0.135, 0.000990, 0.0678 |
| Same step, final bid of vision, `bc_other` | 0.333, 0.263, 0.300 | 0.460, 0.649, 0.575 |
| Policy mean, lag 0, `bc_random`, against its exploration noise | 0.000529, 0.000627, 0.000255 against 0.125, 0.133, 0.138. Below | 7.49e-05, 6.53e-06, 5.79e-05 against 0.207, 0.200, 0.199. Below |
| Policy mean, lag 0, `bc_other` | 0.00430, 0.00327, 0.00239. Below | 3.75e-05, 0, 4.71e-06. Below |

The full count is 38400 readings, which is 4 readings over 400 steps in 24 cases (2 groups, 3 seeds, 2 modes, 2 arms).
0 changed.

**Why V2 fails in the learning runs, and what the rule then says.** Those runs use the default merge. There `bc_other`
is often undone. The swap raises the bid of the module that is not in the broadcast, the order of the winners flips, and
the swap in the next cycle returns the first vector. Steps that end with the base broadcast and another winner are 7
of 19, 24 of 27 and 20 of 29. So the gate did not change at the median step of that arm at 602 and 603. By the rule the
verdict for this group is VOID. I do not replace it with FAILED. What stands next to it, as a report only, is
that `bc_random` reached the gate at 74 of 74 steps, and no module's next content or raw bid changed in either arm.

**Other readers of the broadcast.** The gate reads it and moves, at 0.00286 to 0.0747 of its size in `bc_random`. RND
curiosity and the memory store read it. The appraisal in `evaluate_emotion` has never run, as its own docstring states.
The policy reads it and stays below its noise, which agrees with `action_core_relay_defect_2026_10.md`. The policy
output was never applied in these runs. The action is zero at 24000 of 24000 covert steps. In the learning runs the
motor map and the value gate set it, nonzero at 7508, 7520 and 7518 of 8000 steps, and by code the policy output is
replaced or set to zero. So a path from the broadcast through the room is also cut in both groups.

## The same rules in evaluation mode

This is reported beside the verdicts and decides none. With no sampling the floor is 0, so any path reads above noise.

| Claim | Covert, training | Covert, evaluation | Learning, training | Learning, evaluation |
|---|---|---|---|---|
| RPT-1, tectum | FAILED | FAILED | FAILED | UNDECIDED, holds at 602 and 603 |
| RPT-1, capsule feedback | FAILED | PASS | FAILED | PASS |
| GWT-1, indicator text | FAILED | **PASS** | PASS | PASS |
| GWT-3, responses at the next step | None | None | None | None |

The covert GWT-1 line is the error of 2026-10-17 in another form. An offline check in evaluation mode would pass a
module whose content changes by 1.10e-07 to 3.60e-07 under another layout. No claim here rests on that mode.

## Reading for the owner

Every score is the owner's decision. These are my readings, and they are not rules.

| Indicator | Reading | Other reading the data allows | If the score moves |
|---|---|---|---|
| RPT-1 | PARTIAL. The test stated before the run FAILED in both groups. Row text would name the tectum only and drop the settle cycles and the capsule feedback as evidence | IMPLEMENTED on the weak source condition, with the tectum alone as the evidence. The vision bid reads the carried state at 6 of 6 seeds and the state holds its past. This accepts a bid in place of the content, after the data | Count down by 1 |
| GWT-1 | Stays IMPLEMENTED on the learning runs, with row text cut to two modules, vision and hearing, and the limit above in the row | PARTIAL, because the vision content follows its frames above noise at 1 of 6 seeds | None, or down by 1 |
| GWT-3 | PARTIAL. No module takes a workspace state as input to its processing. Two modules take one number in the same step, as a bid | None that I can support from these data | Count down by 1 |

If RPT-1 and GWT-3 both go to PARTIAL the count is 1 of 14, and the public coverage figure falls from 21.43 to 7.14
percent. A repair of GWT-3 is a design question and belongs to the workspace link work, because the content path
would have to read the broadcast.

## What this does not show

- It is not a claim about consciousness, and it does not test a design.
- Three seeds in each of two regimes, 2 episodes per seed. Seeds differ widely, so a range here is not a spread
  around one value.
- The weights of the end of episode 39 also drive episode 38.
- Open loop, so what a change would do through the room over many steps is not measured.
- The perturbed steps come every 10 steps, and the light pulses at 0.1 cycles per step, so they share a pulse phase.
  Seed 601 has 3 perturbed steps in episode 39. The zero count rests on the code path as well, and a test pins it.
- The default policy with no motor map, DMTS, and `--enable-memory-retrieval` or `--enable-mock-semantic` are not replayed.
  `workspace_bids_live_2026_08.md` covers the bids of DMTS with mock semantic.
- The upstream readings were added after the covert verdicts were read. They carry no verdict.
- The cause of the long memory of `h_state` in the learning runs, and of the difference at seed 601, is not tested.

## Runs

Python 3.12.10, torch 2.14.1+cu126, device CUDA, one job at a time. 150 replays per group by the log rows, and 54 for the
covert upstream arms. The learning run took about 2 h 30 min.

```bash
python -m scripts.analysis.probe_rubric_recheck --runs runs/ta_nolink_s801 runs/ta_nolink_s802 runs/ta_nolink_s803 \
    --output-json runs/_audit_followup/rubric_recheck/result.json --arrays-dir runs/_audit_followup/rubric_recheck
python -m scripts.analysis.probe_rubric_recheck --runs runs/vg_learn_s601 runs/vg_learn_s602 runs/vg_learn_s603 \
    --output-json runs/_audit_followup/rubric_recheck/learning/result.json \
    --arrays-dir runs/_audit_followup/rubric_recheck/learning
python -m scripts.analysis.probe_rubric_recheck --runs runs/ta_nolink_s801 runs/ta_nolink_s802 runs/ta_nolink_s803 \
    --arms-train base,replica,noise,history_swap,frame_other --arms-eval base,replica,history_swap,frame_other \
    --output-json runs/_audit_followup/rubric_recheck/covert_upstream/result.json \
    --arrays-dir runs/_audit_followup/rubric_recheck/covert_upstream
python -m scripts.analysis.probe_rubric_recheck --count-arrays runs/_audit_followup/rubric_recheck \
    runs/_audit_followup/rubric_recheck/learning --output-json runs/_audit_followup/rubric_recheck/exact_counts.json
```
