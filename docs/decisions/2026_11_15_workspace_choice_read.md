# Step P0 of design P: rules of the read of the broadcast at the choice step

Date: 2026-11-15. Status: rules written before the check script exists and before any run of the check.
No model is trained. No gate runs. No default changes. No code of the agent changes.

The design is P of `docs/decisions/2026_11_15_workspace_action_design.md`, chosen by the owner on
2026-11-15. In design P the policy reads the broadcast and chooses one of the candidates that the tectum
bound. This check is its first step. It can stop the design before any build.

No indicator moves with this check. Nothing here is a claim about consciousness.

## The question

At the step where the value gate chooses, does the recorded broadcast hold which tone is the louder one?

With the candidates in order of salience, that is the content a policy needs to find the light among
them. The check is a read of saved vectors with a held-out test. It is a correlation. It does not show
that anything uses the content, and it does not test a policy.

## The runs

Six saved runs in two counted groups, and three runs that are reported only.

| Group | Runs | Seeds | Episodes in each run | Part |
|---|---|---|---|---|
| Value learning | `vg_learn_s601`, `vg_learn_s602`, `vg_learn_s603` | 601, 602, 603 | 40 | Counted |
| Food switch | `rv_switch_s701`, `rv_switch_s702`, `rv_switch_s703` | 701, 702, 703 | 60 | Counted |
| Competing goals, benefit arm | `cg_gate_benefit_s1301`, `cg_gate_benefit_s1302`, `cg_gate_benefit_s1303` | 1301, 1302, 1303 | 60 | Reported only |

Every run used `--dark-room-decoy-level 0.5,3.0`, the default merge of the workspace (`legacy`) and
`--policy-input broadcast`. The competing goals runs are reported only, because their room has a wait, a
salience floor and forced episodes, and their code is of a later revision.

**What I read before these rules.** Field names of `value_gate.json`, `map_binding.json`, `steps.jsonl`
and `vectors.npz`. The number of episodes of each run. The number of episodes with each decoy level,
which is 20 and 20 in the value learning runs and 30 and 30 in the other runs. The presence of
`vectors.npz` in each episode. The decoy level and the merge in the recorded command of each run. I read
no value of a broadcast, no winner of the workspace and no ignition state at any step.

## Definitions, all fixed here

1. **Choice step.** The first step of an episode where the binding record has a band and at least one
   group with cells. This is the rule of `_wire_value_gate` in `scripts/training/train_rlhf.py`, and the
   definition of `docs/decisions/2026_11_11_workspace_priority_check.md`.
2. **Order inside a step.** The tectum writes the binding record, then the workspace competes, then the
   action is selected. So the broadcast of the choice step exists before the gate chooses.
3. **Label.** For each episode, 1 when the tone of the light is the louder one and 0 when the tone of the
   decoy is the louder one. It is read from the truth of the room, `_truth_decoy_level` in
   `internals.info_after_step` of the first step record. A level below 1 gives 1, and a level above 1
   gives 0.
4. **Row.** One row for each episode. It is the recorded vector of the choice step, 256 values.
5. **Streams.** `broadcast` is the stream under test. `audio_content` is the reference, read in the same
   way. `tectum_content` is reported only.
6. **The read.** Five folds over the episodes. The folds are made once for each run from a generator with
   seed 0, with each label divided as equally as possible between the folds. On the training rows of a
   fold, each dimension is brought to mean 0 and standard deviation 1, and a dimension with a standard
   deviation of 0 is set to 0. A ridge regression of the label on the rows follows. Its penalty is one of
   0.01, 0.1, 1, 10, 100 and 1000. It is chosen inside the training rows by four inner folds made in the
   same way, as the penalty with the highest mean AUC, and the smallest penalty on a tie. The AUC of the
   held-out rows is computed inside each fold. The score of the run is the mean of the five AUC values.
7. **Null.** The same read with the labels of the episodes in a random order, 200 times, from a
   generator with seed 1. The value is the 95th percentile of the 200 scores.
8. **Episode number control.** The same read with one input, the episode number.

## The rules

The label of one run.

| Label of the run | Condition |
|---|---|
| VOID | A control below fails, or the reference fails. The reference fails when the score of `audio_content` is below 0.90 or is not above its own null |
| HOLDS | Not VOID. The score of `broadcast` is 0.80 or more, is above its null, and is above the episode number control |
| AT NULL | Not VOID. The score of `broadcast` is not above its null |
| BETWEEN | Not VOID, and neither HOLDS nor AT NULL |

The verdict of the check, over the two counted groups.

| Verdict | Condition | What follows |
|---|---|---|
| VOID | A counted run is VOID | The cause is reported. Nothing else is read as a result |
| HOLDS | 3 of 3 runs HOLD in each counted group | Step P1 of design P can start, with its own rules |
| FAILED | 3 of 3 runs are AT NULL in each counted group | Design P stops with the default merge. The options for the owner are design E, a test of the other merge on new runs, or the repair of the path from the map to the content |
| UNDECIDED | Any other case | The table of the runs is reported. The owner decides |

The verdict is read VOID first, then FAILED, then HOLDS, then UNDECIDED.

## Controls with exact expected values

| Control | Expected value | On a mismatch |
|---|---|---|
| K1. Episodes of the run, and `vectors.npz` and `steps.jsonl` in each | 40 or 60 as in the table of the runs, each with 200 step records | The run is VOID |
| K2. Episodes with each label | 20 and 20, or 30 and 30 | The run is VOID |
| K3. The salience of the binding record at the choice step against the salience that the gate recorded for the episode | Equal for each band, to 1e-9 | The run is VOID |
| K4. The step index of the vector row against the choice step, from the `__steps` array of each stream | Equal | The run is VOID |
| K5. The mean of the 200 null scores | From 0.40 to 0.60 | The run is VOID |

The unit tests pin the read on built data before the first run. A planted signal in one dimension gives
a score of 0.95 or more. Pure noise with 60 rows gives a score that is not above its null. The fold maker
gives each fold both labels. The choice step function gives the same step as `find_choice_step` of
`scripts/analysis/check_workspace_priority.py` on a built record.

## Reported only, with no part in the verdict

1. The score of `tectum_content`, with its null.
2. Which vector the broadcast is. For each choice step, the largest absolute difference between
   `broadcast` and `audio_content`, and between `broadcast` and `tectum_content`. The count of episodes
   where a difference is 0.
3. The share of choice steps with an ignited workspace, and the count of each winner.
4. The same read with one row for each step from the first step of the episode to the choice step, with
   whole episodes held out.
5. The scores of the three competing goals runs.
6. The penalty that each outer fold chose.

## Limits, stated before the run

1. The check is a correlation on saved runs. It does not show that the workspace acts on behaviour.
2. The room gives the label a second cue. With a decoy level of 3.0 the total sound is louder than with a
   level of 0.5. So a read can pass from the total loudness alone, with no identity of a tone. The check
   cannot tell these apart. A gate of design P needs a room in which the total loudness does not give the
   answer, or a control for it. A pass here is necessary for design P and not sufficient.
3. The decoy levels come in a fixed order, so the label follows the parity of the episode number. The
   episode number control is a linear read and cannot see a parity. The broadcast has no input from the
   episode number by the code. I did not test this.
4. Each run has 40 or 60 rows of 256 values. A fold AUC from 8 or 12 held-out rows is coarse. The
   penalty is chosen inside the training rows for this reason.
5. The label is the louder tone. It is the place of the light among the candidates in order of salience
   only because the tone of each object is fixed in these rooms.
6. The two counted groups use the same room, binder and gate. They are not independent.
7. The saved runs used the default merge. The check says nothing about `top_winner`.
8. The weights of each run changed during the run. The rows of early and late episodes come from
   different weights.

## Correction of 2026-11-16, before the first run

The script and its tests were written on 2026-11-16. Two of the unit tests named above cannot be met as
written. The cause is a property of the read, and not an error of the code. I found it on built data.
I read no saved broadcast, no winner and no ignition state before this correction.

**What I measured on built data.** Rows of normal noise with a shift of 6 standard deviations in the
signal dimensions for the episodes with label 1. The mean score of 10 seeds, with the lowest in brackets.

| Episodes | Dimensions | Dimensions with the signal | Score |
|---|---|---|---|
| 40 | 256 | 1 | 0.726 (0.525) |
| 40 | 256 | 4 | 0.984 (0.925) |
| 40 | 256 | 16 | 1.000 (1.000) |
| 40 | 12 | 1 | 1.000 (1.000) |
| 60 | 256 | 1 | 0.734 (0.556) |
| 60 | 256 | 4 | 0.981 (0.917) |
| 60 | 256 | 16 | 1.000 (1.000) |

With a shift of 1 standard deviation, 40 episodes and 256 dimensions, 16 signal dimensions give 0.925
(0.850) and 64 give 1.000. On pure noise, the score was above the null of 100 label orders in 4 of 40
seeds at 40 episodes and in 1 of 40 seeds at 60 episodes. The mean of the null scores was from 0.468 to
0.555.

**The corrections.**

1. The test "a planted signal in one dimension gives a score of 0.95 or more" is replaced by three
   tests. A signal in 16 of 256 dimensions gives 0.95 or more. A signal in one of 12 dimensions gives
   0.95 or more. A signal in one of 256 dimensions gives a mean below 0.90 over 5 seeds, and this test
   pins the limit below.
2. The test "pure noise with 60 rows gives a score that is not above its null" is replaced. The null is
   a 95th percentile, so about 1 case of 20 is above it by chance. The new test asks for 3 of 20 seeds
   at most.
3. New limit 9. The read has little power for a label that only one or a few of the 256 dimensions
   hold. Each dimension is brought to the same spread, so the many dimensions without the label add
   noise to the read. A run that is AT NULL can come from such a code. The reference guards against
   this only when the audio vector and the broadcast hold the label in a similar number of dimensions.
4. Two points that the definitions left open are fixed in the script. The folds of a read are made
   from the labels that the read receives, so a null read makes its own folds from the labels in their
   random order. The inner folds come from a new generator with seed 0.
5. Control K4 is read as follows. Each stream has a row at each step from 0 to the choice step, and the
   row of the choice step is the row under test. Control K5 applies to the null of `broadcast` and of
   `audio_content`.

The read, the thresholds, the labels of a run, the verdict and the controls K1 to K5 do not change.

## What a result would not show

HOLDS would not show that a policy can learn the choice. Step P1 asks that. FAILED would not show that
the workspace cannot hold the content with another merge or another room. No result here moves GWT-2,
GWT-3 or AE-1.

## Files

| File | Content |
|---|---|
| `scripts/analysis/check_choice_read.py` | The check. Written after these rules are pushed |
| `tests/test_choice_read.py` | The unit tests named above |
| `docs/results/data/workspace_choice_read_2026_11/` | The output of the run |
| `docs/results/workspace_choice_read_2026_11.md` | The result, with the verdict first |
