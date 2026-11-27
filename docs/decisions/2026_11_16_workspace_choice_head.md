# Step P1 of design P: rules of the offline choice head

Date: 2026-11-16. Status: rules written before the script exists and before any run of the check.
Changed two times on 2026-11-16, before any code. The first change followed a review by the author.
The second followed an independent review. The section "Changes after the review" at the end lists each
change.
No run of the agent is made. No default changes. No code of the agent changes. Small heads are trained
offline on saved vectors, and they are not saved.

The design is P of `docs/decisions/2026_11_15_workspace_action_design.md`. Step P0
(`docs/results/workspace_choice_read_2026_11.md`) gave UNDECIDED, and the owner chose on 2026-11-16 to
go to step P1 with the default merge.

No indicator moves with this check. Nothing here is a claim about consciousness.

## The question

From the recorded broadcast at the choice step, and from the recorded outcome of the choice that the
value gate made, can a choice head learn which candidate gives the food?

Step P0 read the broadcast with the true label of each episode. A policy does not have that label. It
has one outcome for the one candidate that was followed. Step P1 trains with that feedback only. It
also uses a variant of the go and no-go head of the project and a plain value head, in place of a
ridge read.

## What I read before these rules

1. The result of step P0. The broadcast holds the louder tone in 4 of 6 runs, and it is an exact copy of
   the audio vector or the tectum vector in 279 of 300 episodes.
2. From `value_gate.json` of the six runs, the number of candidates of each episode (2 in each of the
   300 episodes), the names of the bands (18 and 23), and the type of the choice field.
3. The recorded command of `vg_learn_s601` and of `rv_switch_s701`. The food switch runs use
   `--dark-room-food-switch 30`.
4. `docs/results/value_gate_2026_10.md`, which names band 18 as the tone of the light and band 23 as
   the tone of the decoy.
5. The code of the two heads and of the loss of the policy.

I did not read the choices or the outcomes of the gate in this session for these rules. Earlier checks
of the project read them. I trained no head on a saved vector.

## The units

A unit is a set of episodes of one run in which the same object gives the food.

| Group | Units | Episodes of each unit | Band of the food | Part |
|---|---|---|---|---|
| Value learning | `vg_learn_s601`, `vg_learn_s602`, `vg_learn_s603` | 0 to 39 | 18 | Counted |
| Food switch, before the switch | `rv_switch_s701`, `rv_switch_s702`, `rv_switch_s703` | 0 to 29 | 18 | Counted |
| Food switch, after the switch | the same three runs | 30 to 59 | 23 | Reported only |

The food switch runs are divided because the object that gives the food changes at episode 30
(`simulations/environments/simple_visual_env.py`). A head that learns one relation between the broadcast
and the food cannot hold both parts. The part after the switch is reported only, because the gate
learns again there and its choices are of another kind.

## Definitions, all fixed here

1. **Choice step, input row.** As in step P0. The input of an episode is the recorded vector at the
   choice step, 256 values.
2. **Slots.** The two candidates of an episode in order of the salience that the gate recorded. Slot 0
   has the higher salience. Equal salience makes the unit VOID.
3. **Target.** 1 when the band of the food is in slot 0, and 0 when it is in slot 1. The head never
   receives the target. It is used for the score only.
4. **Feedback.** For each episode, the slot of the band that the gate chose, and the recorded outcome
   of that choice. Nothing else.
5. **Preference.** A head gives one number for each slot. The preference is the number of slot 0 minus
   the number of slot 1.
6. **Score of one seed.** Five folds over the episodes of the unit, made as in step P0 from a generator
   with seed 0, with each target divided as equally as possible. Each input dimension is brought to
   mean 0 and standard deviation 1 on the training episodes. The head is trained on the training
   episodes and gives a preference for each held-out episode. The AUC of the preference against the
   target is computed inside each fold. The score of one seed is the mean of the five values.
7. **Score.** The mean of the scores of one seed with the initial seeds 0, 1 and 2 of the head. The
   folds are the same for the three seeds.
8. **Null.** The score of one seed, with initial seed 0, with the input rows given to the episodes in a
   random order, 100 times, from a generator with seed 1. The feedback and the targets stay with their
   episodes. The value is the 95th percentile. The null of one seed is wider on average than the null of
   a mean of three, so the test is expected to be stricter with it, and this is not certain for one
   seed. A 95th percentile from 100 values also has its own error. I did not measure either effect.
9. **Episode number control.** The same score of the same head with one input, the episode number. In
   step P0 this control scored 0.300 and 0.417, below each null. So the condition "above its episode
   number control" is not expected to decide a label here.
10. **Blank control.** The score of one seed with an input of zeros. Each head then gives the same
    preference for each episode, and the AUC is 0.5 exactly. It is a check of the code and not evidence.
11. **Full feedback.** Each training episode is given two times, one time with each slot as the chosen
    slot. The outcome is 1 for the slot of the food and 0 for the other slot. Full feedback gives the
    head the target. So a score with full feedback is near to step P0 with another reader, and it is
    not a test of what a policy can learn. It is reported as a description next to the result with the
    recorded feedback. It does not show the cause of a failure, because it differs from the recorded
    feedback in three ways (both slots, two rows for each episode, and the target itself).
12. **Source control.** The same score of the same head, trained with the recorded feedback, with an
    input of 3 values. The first is 1 when
    the broadcast at the choice step equals the audio vector, the second is 1 when it equals the tectum
    vector, and the third is 1 when it equals neither. The comparison is exact, as in step P0.
13. **Score inside the audio source.** The AUC of the preference of the head with the broadcast, over
    the held-out episodes of the five folds together in which the broadcast equals the audio vector. It
    is the mean over the three seeds. It is not computed with less than 4 such episodes of each target.
    It has its own null, with no training. The targets of the episodes of the subset are put in a random
    order 10000 times, from a generator with seed 1, with the same preferences, and the value is the 95th
    percentile. With 4 and 4 episodes an AUC of 0.875 or more occurs by chance in 4 of 70 orders, so the
    number says little at the smallest count. The result gives the count of episodes of each target
    next to the score and its null.
14. **Choice-only control.** The same score of the same head with the recorded choices and with each
    outcome set to 1. It shows what a head scores by a copy of the choice of the gate, with no use of
    the outcome.

## The two heads

Both heads start from the default initial weights of the library with the initial seed. Both are
trained with Adam at a learning rate of 3e-4, which is the default of both policy classes, on all
training episodes at once, for 200 updates. Neither head is tuned.

**The go and no-go variant.** `PrefrontalCortex(256, 256)` and `BasalGanglia(256, 2)` of
`models/self_model/action_selection_core.py`, with one channel for each slot. The working memory starts
at zero for each episode.

It is named a variant in each table and in the verdict, because the loss of `update_policy` cannot be
used exactly. That loss takes the go target from the recorded action of the agent. The recorded action
is a movement with 4 values. A choice between 2 slots has no recorded action of that kind. So the action
of an episode is a vector with 1 for the chosen slot and 0 for the other slot. Each term of the loss
table below is the code, line for line. The use of the loss departs from the live agent in the five
ways of the list "Departures from the code" below.

| Term | Rule | Line of `update_policy` |
|---|---|---|
| Advantage | The outcome minus the value of the critic, with no gradient | 417 |
| Value | Mean squared error of the critic against the outcome, weight 0.5 | 418, 462 |
| Go | Mean squared error of the go signals of both channels against the action x (+1 when the advantage is above 0, -1 when it is below 0, 0 when it is 0). So the target of the other channel is 0 | 441, 442 |
| No-go | Binary cross entropy of the no-go signals of both channels against the same target, 1 when the advantage is below 0, else 0 | 445 to 448 |
| Brake | Mean squared error of the brake against the tanh of the size of the advantage, weight 0.3 | 457 to 460 |

Lines 445 to 448 give both channels the same no-go target. So the no-go signal learns no difference
between the slots, and the preference comes from the go signals.

**Departures from the code.** The line numbers are of `models/self_model/action_selection_core.py`.

| No. | The code | These rules |
|---|---|---|
| D0 | The go target comes from the recorded action, a movement with 4 values (line 441) | The action is 1 for the chosen slot and 0 for the other slot |
| D1 | The target of the critic is the discounted return of the shaped reward over the steps of a rollout (lines 331 to 335, 371, 396, 402 to 408) | The outcome of the gate, 0 or 1, for one row |
| D2 | Each row has the stored working memory of its step (lines 369, 395, 413) | The working memory is zero |
| D3 | One update for each rollout, and the buffer is then empty (lines 464 to 468) | 200 updates on the same rows |
| D4 | The policy receives the broadcast as recorded (lines 284 to 286) | Each dimension is brought to mean 0 and standard deviation 1 |
| D5 | The plain actor-critic has a policy loss with a log probability, a value loss, an entropy term, and a tanh on the mean (`models/self_model/standard_actor_critic.py`, lines 75, 105 to 117, 136) | The plain value head has the body of that class only |

A result of the variant is not a result of the go and no-go head of the live agent. A result of the
plain value head is not a result of the plain actor-critic.

The number of a slot is go x (1 - no-go) of its channel, with the dopamine term at 0 (line 177 of
`BasalGanglia.forward` without the brake). The brake and the relay are left out by choice. The brake is
one value for each episode. It does not change the sign of the preference, but it changes its size
between episodes, so it can change the AUC. The score with the brake in the number is reported only. The
relay is in no term of the loss in the code, so it stays at its initial weights.

**The plain value head.** The body of `StandardActorCritic` (`models/self_model/standard_actor_critic.py`),
which is two layers of 128 units with tanh, and one linear layer with 2 outputs. The loss is the mean
squared error of the output of the chosen slot against the outcome. The number of a slot is its output.

## The rules

The two heads of the rules below are the go and no-go variant and the plain value head.

The label of one unit, for one head, with the recorded feedback.

| Label | Condition |
|---|---|
| VOID | A control below fails, or the reference fails, or the count rule of the recorded feedback below is not met. The reference is the plain value head with the audio vector as input and the recorded feedback. It fails when its score is below 0.80 |
| LEARNS | Not VOID. The score of the head with the broadcast is above its null and above its episode number control |
| DOES NOT LEARN | Not VOID, and not LEARNS |

A head PASSES when it LEARNS with the recorded feedback in 2 of 3 units or more in each counted group.

| Verdict | Condition | What follows |
|---|---|---|
| VOID | A counted unit is VOID | The cause is reported. Nothing else is read as a result |
| FAILED | With the recorded feedback, each head LEARNS in 1 of the 6 counted units at most | The stop rule of step P1 in the design document is met. The owner decides between a stop of design P with the default merge and a continuation with a stated reason. The result with full feedback is reported next to it as a description. It does not show the cause |
| LEARNS | The go and no-go variant or the plain value head PASSES, or both | The rules of step P2, the build behind a flag. The owner chooses the head with the table of both |
| UNDECIDED | Any other case | The table of the units is reported. The owner decides |

The verdict is read VOID first, then FAILED, then LEARNS, then UNDECIDED. On FAILED the result
document writes FAILED as its first word.

By step P0, a failure with full feedback also is not expected, because full feedback gives the target
and P0 read the target in 4 of 6 runs. So no verdict of this check stops design P without the owner.

**What step P1 can add.** Step P0 already showed in which runs the content is present. Step P1 adds two
facts. The first is which head reads the content with the feedback of one outcome. The second is if the
recorded feedback of the gate is sufficient for that.

**Notes on a unit that LEARNS.** Each such unit gets two notes, each printed with its two scores.

| Note | Condition |
|---|---|
| "Above the source control" or "not above the source control" | The score of the head against the score of its source control |
| "Above the choice-only control" or "not above the choice-only control" | The score of the head against the score of its choice-only control |

A note compares two numbers from folds of 6 or 8 held-out episodes with no margin, so one episode can
change it. The notes do not change the verdict. The result document gives the count of each note, and
the score inside the audio source of each unit with its null and its counts.

**Prediction, written before the run.** By step P0, the units expected to learn are `vg_learn_s601`,
`vg_learn_s603`, `rv_switch_s701` and `rv_switch_s702`.

**Duties of the result document on the rule of 2 of 3.**

1. It states that the rule of 2 of 3 was set after step P0, on the same runs, and that the count of
   units is not a repeat of P0.
2. It names the units that learn, next to the label of each run in step P0.
3. It marks a LEARNS in `vg_learn_s602` or `rv_switch_s703` as a warning. Step P0 found no content
   there, so the cause can be a path other than the content. It is not counted as a stronger result.
4. It states that the food switch units have 30 episodes here and had 60 in step P0.
5. It states that a LEARNS shows one thing only. Given the content that P0 found, a head with this
   feedback and these training values reads it.

## The count rule of the recorded feedback

Written on 2026-11-16 before I read any choice or outcome. The owner asked for it. The count table is
read before any training, and the owner decides on the run after it.

**The table.** For each unit, the count of episodes for each chosen slot (0 or 1), outcome (0 or 1) and
target (0 or 1). It has 8 cells. It needs `value_gate.json` only. No saved vector is read.

**Why a rule is necessary.** The gate chooses by a softmax over salience + 2.0 x value at a temperature
of 0.25 (`models/core/value_gate.py`, lines 37 and 38). With a value near 1 for one band and equal
salience, the probability of that band is 0.9997 by this formula. If the gate chose the band of the
food in almost each episode, the outcome is 1 in almost each episode. The outcome then tells the head
nothing, and a head can score only by a copy of the choice.

**The rule.** The five training sets of a unit are the sets of the score, made from the targets with
seed 0. A slot is *informative* in a training set when the episodes of that set in which the gate chose
the slot have 2 or more with outcome 1 and 2 or more with outcome 0. A unit is VOID for the recorded
feedback when one or more of its five training sets has no informative slot.

| Number | Reason, given without a look at the data |
|---|---|
| Both outcomes for the same slot | A head learns the number of a slot only from the episodes in which that slot was chosen. With one kind of outcome for a slot, the best number for that slot is a constant, and the input has no use |
| One informative slot is sufficient, not two | The preference is the number of slot 0 minus the number of slot 1. When one of the two numbers follows the input and the other is a constant, the preference follows the input |
| 2 episodes of each outcome, not 1 | With 1 episode the head can fit that one row and nothing shows a relation to the input. 2 is the smallest count above that. It is a lower limit for "the question can be asked", and it does not show that a head can learn. The test on built data shows what a head learns from few such rows |
| Each of the five training sets, not the unit as a whole | A head is trained on a training set, not on the unit. A count over the unit can be met while one training set has none of the necessary episodes |
| No number for the share of outcome 1 | The rule above already fails when almost each outcome is 1. A second number would be a second rule for the same defect |

**What follows.** A counted unit that is VOID for the recorded feedback makes the verdict VOID, with the
cause "the recorded feedback cannot answer the question". The table is reported for each unit, with
the smallest count of the rule over the five training sets. This rule does not change after the table
is read.

**Why 2 of 3 and not 3 of 3.** Step P0 showed that the broadcast holds the label in 2 of 3 runs of each
group. A rule of 3 of 3 could not be met, and I know that before the run. I state that this rule was set
with the result of step P0 in view. The null of one unit is above chance in about 1 case of 20, so 1 of
6 units can be a chance case, and FAILED allows for that.

## Controls with exact expected values

| Control | Expected value | On a mismatch |
|---|---|---|
| K1 to K4 of step P0 | As in step P0, for the whole run | The units of the run are VOID |
| K6. Candidates of each episode | 2, with bands 18 and 23 | The unit is VOID |
| K7. Episodes with each target | 10 or more of each in a unit | The unit is VOID |
| K8. Blank control | 0.5 exactly, for each head. A check of the code | The unit is VOID |
| K9. The mean of the 100 null scores with the recorded feedback | From 0.40 to 0.60 | The unit is VOID. For the null with full feedback, the same range is checked, and on a mismatch the description with full feedback is not reported |
| K10. Outcome values | Each is 0 or 1 | The unit is VOID |

## Unit tests, pinned on built data before the first run

Step P0 named two tests with numbers that the read could not meet. So these rules name the tests and
leave each number to a dated note, which is written from built data only and pushed before the first
run.

1. The loss of the go and no-go variant, for a hand-made case, equals the value computed by hand from
   the table above. The case has an advantage above 0, one below 0 and one of exactly 0. With the same
   action vector, the loss also equals the terms of `update_policy` computed with the lines of the code.
2. A blank input gives a score of 0.5 exactly for each head.
3. The test on built data, which is the gate of the run. See the section below.
4. With pure noise as input, a head is above its null in a small part of the seeds. The note fixes the
   count.
5. A head never receives the target. The test changes the targets and checks that the trained weights
   are the same.
6. The slot order and the target of a built record are correct, with the food band in slot 0 and in
   slot 1.
7. Full feedback gives each training episode two times, with outcome 1 for the slot of the food and 0
   for the other slot.
8. The source control gives the 3 values for a built record of each kind.

A change of the learning rate or of the number of updates is a change of a rule.

## The test on built data, the gate of the run

It uses no saved run. It is made before the count table and before the run, and its numbers go into a
dated note in this file.

| Part | Value |
|---|---|
| Rows | 30 and 40, with the targets in equal parts |
| Input | 256 dimensions of normal noise. The signal is a shift of 1 standard deviation in 64 dimensions for the rows with target 1 |
| Signal cases | In each row, and in 60 percent of the rows only. The rows without the signal are drawn at random |
| Choosers | Random. Always slot 0. Always the slot of the food. The slot of the food in 80 percent of the rows |
| Outcome | 1 when the chosen slot is the slot of the food, else 0 |
| Data sets | 10 for each case, from generators with seeds 0 to 9 |
| Score | As in the run. Five folds, initial seeds 0, 1 and 2, both heads |
| Second pass | The same cases with each outcome set to 1, to show the size of the copy of the choice |

The note gives the mean score of each head in each case, with the lowest of the 10 data sets.

**The gate.** The run starts only if the plain value head reaches a mean score of 0.80 or more with the
signal in each row and the random chooser, at 30 rows and at 40 rows. If it does not, I stop and
report to the owner before any saved vector is used.

Each case in which a head does not learn is listed in the result document as a case in which a DOES NOT
LEARN says nothing about the content.

**Cost of this test.** 2 row counts x 2 signal cases x 4 choosers x 10 data sets x 3 seeds x 5 folds
give 2400 trainings for one head. The second pass gives 2400 more. So one head has 4800 trainings. At
the measured 1.05 seconds for the variant and 0.18 seconds for the plain value head, the estimate is
84 minutes and 14 minutes, about 1.6 hours together, on one CPU thread. It is an estimate.

## Reported only, with no part in the verdict

1. The count of episodes for each chosen slot, outcome and target. A head can learn about a slot only
   from the episodes in which the gate chose it.
2. The score of the go and no-go variant with the brake in the number of a slot.
3. The share of held-out episodes in which the sign of the preference picks the slot of the food, next
   to the share in which the recorded choice of the gate did.
4. Each head with the audio vector and with the tectum vector as input.
5. The units after the switch.
6. The score of one seed for each of the initial seeds 0, 1 and 2.
7. How often the target agrees with the label of step P0.

## Limits, stated before the run

1. The feedback is the record of another chooser. The gate chose by salience and by its learned value,
   so it chose some slots seldom. A live policy would make its own choices. A FAILED here can come from
   that. The owner decides what follows a FAILED for this reason.
2. The gate learned during the run, so its choices and outcomes change with the episode number. The
   weights of the agent also change. The episode number control is the guard, and it is a weak one.
3. The head receives one vector and a working memory at zero. The live policy receives each step and
   never resets its working memory. Step P1 does not test that.
4. A unit has 40 or 30 episodes. A fold has 8 or 6 held-out episodes. The go and no-go variant
   has 576396 weights and the plain value head has 49666, counted with a command on the classes. The heads are not tuned, so a head can fail from its training values and not from the
   content.
5. The total loudness gives the same answer as the identity of the louder tone (limit 2 of step P0).
   The runs have two decoy levels and the target follows the level, so no control on the saved runs
   separates the two. **Requirement for step P3, recorded here.** The gate of step P3 uses a room in
   which the total loudness does not give the target.
6. The broadcast is a copy of one module's vector, and the module depends on the label in some runs.
   A head can learn the target from which module supplied the vector.
7. The levels of the decoy come in a fixed order, so the target follows the parity of the episode
   number in these rooms. The episode number control does not guard against this. It gives the head a
   number that rises from the first episode to the last, and I do not expect a head with 200 updates
   to learn from it a target that changes at each episode. I did not test that. The parity as an input
   cannot be a control, because in these rooms the parity is the label, so that head would score near
   1.0 and no broadcast could be above it. Nothing in step P1 separates "the broadcast holds the sound
   of this episode" from "the broadcast holds a trace of the episode before". **Requirement for step
   P3, recorded here.** The gate of step P3 uses a room in which the levels of the decoy come in a
   random order.
8. The number of a slot of the go and no-go variant leaves out the brake and the relay. The build of step
   P2 must make its own choice here.
9. The two counted groups use the same room, binder and gate. They are not independent.
10. The score is a mean over 3 initial seeds of the head. The saved runs are still one seed of the
    agent for each unit.
11. The outcome is 0 or 1, and the critic of the go and no-go variant starts near 0. For an outcome of
    0 the advantage is minus the value of the critic, so its sign follows the sign of the critic at the
    start. A choice with no food can then get the go target +1 in the first updates. This is the
    behaviour of the loss of the code. The variant can fail from it and not from the content.
12. The source control and the score inside the audio source describe one path, the supplying module.
    They do not act on the total loudness (limit 5). The score inside the audio source pools the
    preferences of five heads from five folds into one AUC.
13. The sign of the advantage of limit 11 can also follow a small error at the end of the training,
    when the critic fits the outcome. I did not test this.
14. No control on the saved runs shows that a head uses the outcome in the way a policy must. The
    choice-only control shows only the size of a copy of the choice.

## What a result would not show

LEARNS would not show that the agent chooses with the broadcast. Only a gate with a cut of the
broadcast shows that, and that is step P3. FAILED would not show that a live policy with its own choices
cannot learn. No result here moves GWT-2, GWT-3 or AE-1.

## Cost

I measured the time of one training of 200 updates on built random data, 24 and 32 rows of 256
values, one CPU thread, 3 times each. The go and no-go variant took from 1.00 to 1.28 seconds. The
plain value head took 0.18 seconds. I trained on no saved vector for this.

The trainings of one head for one unit, with five folds.

| Part | Trainings |
|---|---|
| Score with the broadcast, 3 seeds, recorded feedback | 15 |
| Its null, 100 orders | 500 |
| Its episode number control, 3 seeds | 15 |
| Score with full feedback, 3 seeds | 15 |
| Its null, 100 orders | 500 |
| Its episode number control, 3 seeds | 15 |
| Source control, 3 seeds | 15 |
| Audio vector and tectum vector as input, 3 seeds each | 30 |
| Choice-only control, 3 seeds | 15 |
| Blank control | 5 |
| Sum | 1125 |

With 1.05 seconds for the variant and 0.18 seconds for the plain value head, one unit takes 1125 x 1.23
seconds, which is about 23 minutes. Nine units take about 3.5 hours. This is an estimate from the
measured time of one training. Full feedback gives two rows for each episode, and I did not measure
that case, so the true time can be higher. The estimate leaves out the reading of the files and the
AUC, which I expect to be small. It is below 6 hours, so the null stays at 100 orders for each score.
The job runs alone.

## Changes after the review

Made on 2026-11-16, before any script, test or run. The review is on the review issue of these rules.
I read no saved vector, no choice and no outcome of the gate for these changes.

1. The verdict FAILED now needs both feedbacks. The new verdict FAILED FOR THE RECORDED FEEDBACK goes to
   the owner. Full feedback has its own null and control, and the rules state that it gives the head
   the target.
2. The score is the mean over the initial seeds 0, 1 and 2. The null stays at seed 0.
3. The go and no-go head is now "the go and no-go variant" in each table and in the verdict. Its loss
   is the loss of `update_policy` line for line, with one departure, the action vector. Before, the
   loss acted on the chosen channel only, which made three departures that the rules did not state.
4. The source control and the score inside the audio source are added, with two notes for a unit that
   LEARNS.
5. Limit 7 says that nothing in step P1 guards against the parity, and records the requirement of a
   random order of the levels for step P3.
6. Limit 11 on the sign of the advantage at the start of the training is added.
7. The result document must compare the units that do not learn with the two runs of step P0.
8. The score of the variant with the brake is reported.
9. The cost is a new estimate from a measured training time.

Made on 2026-11-16 after an independent review of commit `1ec3499`, before any script, test or run. The
owner accepted its eight necessary changes and its optional change 2. I checked each code claim of the
review against the code first, and each agreed. I read no saved vector, no choice and no outcome of the
gate for these changes.

1. Independent review, change 1. The rows FAILED and FAILED FOR THE RECORDED FEEDBACK are one row
   FAILED again, with the recorded feedback only. The owner decides what follows. Full feedback is a
   description. The rules state that no verdict of this check stops design P without the owner.
2. Independent review, change 2. The choice-only control is added, with a note for a unit that LEARNS.
3. Independent review, change 3. Unit test 3 is replaced by the test on built data, with four choosers,
   two signal cases, a second pass with each outcome at 1, and a gate of 0.80 for the plain value head.
   It has 4800 trainings for one head, and the estimate of its time is about 1.6 hours.
4. Independent review, change 4. The departures D0 to D5 are listed with their lines. The plain head is
   named "the plain value head".
5. Independent review, change 5. The duties of the result document on the rule of 2 of 3, and a
   prediction of the units that learn, are added.
6. Independent review, change 6. Definition 8 says "is expected to be stricter, and this is not certain
   for one seed". Definition 9 gives the scores of the episode number control in step P0.
7. Independent review, change 7. The score inside the audio source has a null of 10000 orders with no
   training. The source control is trained with the recorded feedback. Each note is printed with its
   two scores.
8. Independent review, change 8. Limit 5 records a requirement for step P3. Limit 4 has the count of
   weights from a command. Limit 12 no longer says that the score inside the audio source does not act
   on the parity, because the review found that this is too strong and the code read that would decide
   it is not done.
9. Independent review, optional change 2. The count rule of the recorded feedback is written before any
   count is read.
10. Independent review, optional change 4. Limit 13 is added.
11. The order after the owner's yes is the test on built data, then the count table, then the owner's
    decision on the run.

## Files

| File | Content |
|---|---|
| `scripts/analysis/check_choice_head.py` | The check. Written after these rules are pushed |
| `tests/test_choice_head.py` | The unit tests named above |
| `docs/results/data/workspace_choice_head_2026_11/` | The output of the run |
| `docs/results/workspace_choice_head_2026_11.md` | The result, with the verdict first |

## Note of 2026-11-16 on the test on built data, written before it runs

The owner said yes to the test on built data, and asked for a table "with the null". The section of the
test above names no null. I fix it here, before the test runs.

1. **The null of the test.** The same case with no signal in the input, which is a signal share of 0.
   It has the same row counts, choosers, data sets, seeds and folds. Its scores show what a head scores
   when the input holds nothing. The report gives its mean and its highest value over the 10 data sets.
   The null of the run, with 100 orders of the input rows, is not used in this test. For one data set
   of each case it would add 500 trainings, which is 16000 trainings more for one head and one pass.
2. **The second pass** with each outcome at 1 has the same three signal shares.
3. **The expected count of result rows.** 2 passes x 2 row counts x 3 signal shares x 4 choosers x 10
   data sets x 2 heads = 960 rows, in `docs/results/data/workspace_choice_head_2026_11/built_test.csv`.
   The end of the job is verified by this count.
4. **Cost.** Each row has 15 trainings, so the test has 14400 trainings, 7200 for each head. At 1.05
   and 0.18 seconds, the estimate is about 2.1 hours for the variant and 0.4 hours for the plain value
   head, about 2.5 hours together, on one CPU thread. It replaces the estimate of 1.6 hours above.
5. **The gate does not change.** The plain value head must reach a mean score of 0.80 or more with the
   signal in each row and the random chooser, at 30 rows and at 40 rows, in the first pass.
6. **The variant is reported apart.** If the variant does not learn and the plain value head does, no
   training value changes. I report it and stop.
7. **The unit tests** are in `tests/test_choice_head.py`, 25 tests. One of them gives the same rows to
   `update_policy` of the policy class of the agent and to the loss of the variant, and the two totals
   are equal.

## Note of 2026-11-16 on the result of the test on built data

The gate of the run PASSED. The plain value head scored a mean of 0.904 at 30 rows and 0.897 at 40 rows, with the full signal and the random chooser, in the first pass. The limit is 0.80.

The go and no-go variant did not reach 0.80 in any case. It scored below 0.5 with the chooser that always takes the food. No training value was changed after these numbers.

The file `docs/results/data/workspace_choice_head_2026_11/built_test.csv` has 960 result rows, the count stated before the test, and no empty score. Each cell below is the mean of 10 data sets. The value in brackets is the lowest data set for a case with signal, and the highest data set for the null.

### The plain value head, first pass

| Rows | Signal share | random | slot_0 | food | food_80 |
|---|---|---|---|---|---|
| 30 | 1.0 | 0.904 (0.763) | 0.941 (0.904) | 0.999 (0.993) | 0.983 (0.948) |
| 30 | 0.6 | 0.639 (0.385) | 0.698 (0.585) | 0.753 (0.674) | 0.715 (0.644) |
| 30 | 0.0, the null | 0.524 (0.681) | 0.526 (0.607) | 0.484 (0.615) | 0.493 (0.570) |
| 40 | 1.0 | 0.897 (0.754) | 0.955 (0.929) | 0.997 (0.987) | 0.993 (0.975) |
| 40 | 0.6 | 0.700 (0.583) | 0.725 (0.662) | 0.806 (0.758) | 0.772 (0.704) |
| 40 | 0.0, the null | 0.548 (0.650) | 0.535 (0.658) | 0.560 (0.696) | 0.538 (0.625) |

### The go and no-go variant, first pass

| Rows | Signal share | random | slot_0 | food | food_80 |
|---|---|---|---|---|---|
| 30 | 1.0 | 0.742 (0.630) | 0.626 (0.430) | 0.302 (0.215) | 0.527 (0.407) |
| 30 | 0.6 | 0.602 (0.437) | 0.519 (0.385) | 0.444 (0.341) | 0.533 (0.474) |
| 30 | 0.0, the null | 0.501 (0.578) | 0.499 (0.600) | 0.521 (0.630) | 0.486 (0.622) |
| 40 | 1.0 | 0.760 (0.604) | 0.573 (0.438) | 0.361 (0.288) | 0.534 (0.462) |
| 40 | 0.6 | 0.608 (0.529) | 0.512 (0.392) | 0.392 (0.337) | 0.495 (0.371) |
| 40 | 0.0, the null | 0.512 (0.600) | 0.509 (0.596) | 0.505 (0.533) | 0.495 (0.554) |

### Second pass, each outcome set to 1, full signal, mean of 10 data sets

| Head | Rows | random | slot_0 | food | food_80 |
|---|---|---|---|---|---|
| The plain value head | 30 | 0.533 | 0.536 | 0.999 | 0.940 |
| The plain value head | 40 | 0.405 | 0.552 | 0.997 | 0.965 |
| The go and no-go variant | 30 | 0.466 | 0.444 | 0.302 | 0.377 |
| The go and no-go variant | 40 | 0.540 | 0.529 | 0.361 | 0.384 |

### What these numbers show

1. The plain value head learns the built signal from the chosen slot and its outcome, with each chooser, at the full signal. With a signal in 60 percent of the rows its mean is below 0.80 in each case but one (0.806, food, 40 rows), and one data set is below the highest null (0.385, random, 30 rows).
2. With the food and food_80 choosers, the plain value head has almost the same score when each outcome is 1 (0.999 and 0.940 against 0.999 and 0.983 at 30 rows). In those cases the score comes from the choice and not from the outcome. With the random and slot_0 choosers the score falls to the null when each outcome is 1, so there the outcome gives the score. This is the reason for the choice-only control of the run.
3. The go and no-go variant is above the null with the random chooser only (0.742 and 0.760, lowest data sets 0.630 and 0.604, highest null data sets 0.578 and 0.600). It is below 0.80 there.
4. The go and no-go variant scores below 0.5 with the food chooser (0.302 and 0.361, highest data sets 0.370 and 0.479). Its preference has the wrong sign in that case. The cause was not examined.
5. The cases in which a head does not learn at the full signal, by the mean against the highest null data set: the variant with food and with food_80 at 30 and 40 rows, and the variant with slot_0 at 40 rows (0.573 against 0.596).

### What this note does not decide

The test on built data does not say which chooser the gate of the saved runs is near to. The count table answers that, and it is not made yet. If the gate chose the food slot in most episodes, points 2 and 4 apply to the run. That is a hypothesis until the count table exists. The owner decides what follows for the variant.

## Note of 2026-11-16 on the count table, written before a count is read

The owner said yes to the count table on 2026-11-16. This note fixes what the rules left open. I wrote it before I read a choice or an outcome. The count rule itself does not change.

1. **What is read.** `value_gate.json` of each of the six runs, and from each episode the recorded salience of the two candidates, the choice and the outcome. The salience gives the slot order. No saved vector is read and no head is trained.
2. **The count of a training set.** For each slot, the smaller of its two outcome counts. The count of the rule of the set is that of its better slot. The set has an informative slot when this count is 2 or more. This is the rule in other words.
3. **Controls that the count table can make.** The count of episodes of `value_gate.json` (40 or 60), K6, K7, K10 and the rule of equal salience. A mismatch makes the unit VOID and is reported with its text. Controls K1 to K4 of step P0 need other files and stay for the run.
4. **Shares, a description only.** For each unit, the share of episodes in which the gate chose the slot of the food, the share in which it chose slot 0, and the share with outcome 1.
5. **The nearest chooser, a description only.** Each chooser of the test on built data has a pair of expected shares (slot of the food, slot 0) with equal targets. They are random (0.5, 0.5), slot_0 (0.5, 1.0), food (1.0, 0.5) and food_80 (0.8, 0.5). The nearest chooser of a unit is the one with the smallest distance to the pair of the unit. It has no part in VOID and no part in the verdict. The gate of the saved runs is not one of these four choosers.
6. **The units after the switch** are reported with the same table. They have no part in VOID of the check, as in the section "The units".
7. **Report.** If a counted unit is VOID, the report starts with VOID.

## Note of 2026-11-16 on the result of the count table

VOID. 4 of the 6 counted units are VOID for the recorded feedback: `vg_learn_s601`, `vg_learn_s602`, `vg_learn_s603` and `rv_switch_s701` before the switch. By the section "The count rule of the recorded feedback", a counted unit that is VOID makes the verdict of the check VOID, with the cause "the recorded feedback cannot answer the question". The owner decides what follows. The count rule was not changed.

The code and its tests were pushed in commit `184a007` before a count was read. The table is in `docs/results/data/workspace_choice_head_2026_11/count_table.json`. It was made from `value_gate.json` of the six runs only. No saved vector was read and no head was trained. No control mismatch occurred.

### The count table of each unit

Each cell is a count of episodes. "Food" means that the chosen slot is the slot of the food.

| Unit | Episodes | Slot 0, outcome 1, food | Slot 0, outcome 0, food | Slot 0, outcome 0, not food | Slot 1, outcome 1, food | Slot 1, outcome 0, food | Slot 1, outcome 0, not food |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 40 | 20 | 0 | 2 | 17 | 1 | 0 |
| `vg_learn_s602` | 40 | 19 | 0 | 3 | 17 | 0 | 1 |
| `vg_learn_s603` | 40 | 20 | 0 | 1 | 19 | 0 | 0 |
| `rv_switch_s701` before | 30 | 14 | 1 | 1 | 14 | 0 | 0 |
| `rv_switch_s702` before | 30 | 13 | 2 | 1 | 14 | 0 | 0 |
| `rv_switch_s703` before | 30 | 14 | 1 | 3 | 12 | 0 | 0 |
| `rv_switch_s701` after, reported only | 30 | 11 | 1 | 4 | 11 | 0 | 3 |
| `rv_switch_s702` after, reported only | 30 | 14 | 0 | 3 | 12 | 0 | 1 |
| `rv_switch_s703` after, reported only | 30 | 14 | 0 | 3 | 12 | 0 | 1 |

The two cells with outcome 1 and a slot that is not the slot of the food are 0 in each unit. Each unit has the two targets in equal parts.

### The five training sets, and VOID or not VOID

Each entry is the count of the rule of one training set: for each slot the smaller of its two outcome counts, and then the larger of the two slots. A set needs 2 or more. The value in brackets is "outcome 1, outcome 0" for slot 0 and then for slot 1.

| Unit | Set 0 | Set 1 | Set 2 | Set 3 | Set 4 | Smallest | State |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 1 (16,1 / 14,1) | 2 (16,2 / 13,1) | 2 (16,2 / 13,1) | 1 (16,1 / 15,0) | 2 (16,2 / 13,1) | 1 | VOID |
| `vg_learn_s602` | 2 (15,2 / 14,1) | 3 (15,3 / 13,1) | 3 (15,3 / 13,1) | 1 (15,1 / 15,1) | 3 (16,3 / 13,0) | 1 | VOID |
| `vg_learn_s603` | 0 (16,0 / 16,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 0 | VOID |
| `rv_switch_s701` before | 2 (11,2 / 11,0) | 2 (11,2 / 11,0) | 1 (12,1 / 11,0) | 2 (11,2 / 11,0) | 1 (11,1 / 12,0) | 1 | VOID |
| `rv_switch_s702` before | 3 (10,3 / 11,0) | 2 (11,2 / 11,0) | 2 (11,2 / 11,0) | 3 (10,3 / 11,0) | 2 (10,2 / 12,0) | 2 | Not VOID |
| `rv_switch_s703` before | 3 (11,3 / 10,0) | 4 (11,4 / 9,0) | 3 (12,3 / 9,0) | 4 (11,4 / 9,0) | 2 (11,2 / 11,0) | 2 | Not VOID |
| `rv_switch_s701` after | 2 (10,2 / 10,2) | 5 (9,5 / 8,2) | 5 (9,5 / 8,2) | 5 (8,5 / 8,3) | 3 (8,3 / 10,3) | 2 | Not VOID |
| `rv_switch_s702` after | 2 (11,2 / 10,1) | 3 (11,3 / 9,1) | 3 (12,3 / 9,0) | 3 (11,3 / 9,1) | 1 (11,1 / 11,1) | 1 | VOID, reported only |
| `rv_switch_s703` after | 2 (11,2 / 10,1) | 3 (11,3 / 9,1) | 3 (12,3 / 9,0) | 3 (11,3 / 9,1) | 1 (11,1 / 11,1) | 1 | VOID, reported only |

The training sets have 32 episodes for the value learning units and 24 episodes for the other units.

### Shares and the nearest chooser, a description only

| Unit | Share with the slot of the food chosen | Share with slot 0 chosen | Share with outcome 1 | Nearest chooser of the test on built data |
|---|---|---|---|---|
| `vg_learn_s601` | 0.950 | 0.550 | 0.925 | food |
| `vg_learn_s602` | 0.900 | 0.550 | 0.900 | food |
| `vg_learn_s603` | 0.975 | 0.525 | 0.975 | food |
| `rv_switch_s701` before | 0.967 | 0.533 | 0.933 | food |
| `rv_switch_s702` before | 0.967 | 0.533 | 0.900 | food |
| `rv_switch_s703` before | 0.900 | 0.600 | 0.867 | food |
| `rv_switch_s701` after | 0.767 | 0.533 | 0.733 | food_80 |
| `rv_switch_s702` after | 0.867 | 0.567 | 0.867 | food_80 |
| `rv_switch_s703` after | 0.867 | 0.567 | 0.867 | food_80 |

The nearest chooser is a description. The gate of the saved runs is not one of the four choosers, and this column has no part in VOID and no part in the verdict.

### What these counts show

1. The gate chose the slot of the food in 90.0 to 97.5 percent of the episodes of the counted units. The outcome is 1 in 86.7 to 97.5 percent of them. So the recorded outcome tells a head little, which is the case that the count rule was written for.
2. Slot 1 has an outcome of 0 in 0 or 1 episodes of each counted unit. The outcomes of 0 are almost all in slot 0, the slot with the higher salience.
3. Each counted unit is nearest to the food chooser. In the test on built data, the plain value head has the same score with that chooser when each outcome is set to 1 (0.999 at 30 rows), so its score there comes from the choice. The go and no-go variant scores below 0.5 with that chooser (0.302 and 0.361). This is a relation between two descriptions. It is not a result on the saved runs.
4. The two units that are not VOID are at the lower limit of the rule (smallest count 2).

### Open points, not examined

1. The units `rv_switch_s702` after and `rv_switch_s703` after have the same chosen slot, outcome and target in each of their 30 episodes. I checked this by a comparison of the slot records. The units of the same two runs before the switch differ. I did not examine the cause.
2. Six episodes have the slot of the food chosen and an outcome of 0 (1 in `vg_learn_s601`, 1 in `rv_switch_s701` before, 2 in `rv_switch_s702` before, 1 in `rv_switch_s703` before, 1 in `rv_switch_s701` after). The gate gives 0 when no food comes in its window of steps. I did not examine these episodes.
3. The wrong sign of the go and no-go variant with the food chooser in the test on built data stays open. The owner said not to examine it now.

## Correction of 2026-11-16: the food switch units divide at episode 29

The owner decided this correction on 2026-11-16. It changes the unit definition of the three food
switch runs and nothing else. The count rule and each other rule stay as they are.

**The error.** The section "The units" divides the food switch runs at episode 30, from the option
`--dark-room-food-switch 30` of their recorded command. In the saved runs the food moves at recorded
episode 29. The step records name the light as the food in episodes 0 to 28 and the decoy as the food
from episode 29 on.

**The cause.** The three runs `rv_switch_s701`, `rv_switch_s702` and `rv_switch_s703` were made on
2026-10-16 with the room of commit `ea3eabd`. That room counted one episode for each reset, and
`scripts/training/train_rlhf.py` resets the room once before its first episode (`seed_environment`).
So the flag at 30 moved the food at recorded episode 29. Commit `961dc1c` of 2026-10-17 repaired the
counter, after the three runs. `docs/results/reversal_2026_10.md` records the defect. I wrote the
rules of step P1 without a read of that document.

**The corrected units.**

| Group | Units | Episodes of each unit | Band of the food | Part |
|---|---|---|---|---|
| Value learning | `vg_learn_s601`, `vg_learn_s602`, `vg_learn_s603` | 0 to 39 | 18 | Counted. Not changed |
| Food switch, before the switch | `rv_switch_s701`, `rv_switch_s702`, `rv_switch_s703` | 0 to 28 | 18 | Counted |
| Food switch, after the switch | the same three runs | 29 to 59 | 23 | Reported only |

A unit before the switch now has 29 episodes, and a unit after the switch has 31. The two targets are
then not in equal parts in these units. Control K7 (10 episodes or more of each target) and the folds
of the rules apply as written.

**What this correction is not.** It is made after the count tables were read, so it is not blind. The
only change is the episode of the division, and its value comes from the step records and from the
commits above, not from a count. The count tables with the old division stay in the result document
for the record.

## Owner decision of 2026-11-27 on unit test 2, the blank control

Date: 2026-11-27. This is an owner decision. No reviewer judged it. No earlier line of this file was
edited. It changes the form of unit test 2 of the section "Unit tests, pinned on built data before the
first run" (line 272). That test checks the blank control of item 10 of the definitions (line 83).
Item 10 and K8 (line 259) stay as written. The change is in `tests/test_choice_head.py` only. No file
under `models`, `simulations` or `scripts` changes.

**The failure.** The workflow Core Tests failed on the public repository at commit `5df5e63`, in run
46018014774. The test
`tests/test_choice_head.py::TestBlankControl::test_an_input_of_zeros_scores_one_half_exactly[plain]`
gave 0.4333333333333334, and it expects 0.5. Another run of the same workflow passed the same test, with
the same test file and the same code, the same runner image (ubuntu-26.04, version 20261111.635), torch
2.15.0+cpu and numpy 2.5.4. The logs of the two runs do not name the processor.

**What the test needs.** With an input of zeros, each standardized row is exactly zero. Each fold holds
3 episodes of each target. So its AUC has 9 pairs, and it is a multiple of 1/18. A fold with six equal
preferences gives 0.5. When each fold has six equal preferences, the score is 45/90, which is 0.5
exactly. The failed value is 39/90 up to the last bit of floating point. So in that run the preferences
of at least one fold were not all equal.

**The read on another computer.** I ran the blank control fold by fold, read only, on Windows 11 with an
Intel Core i9-13900HX, torch 2.14.1+cu126 (on the processor, with MKL) and numpy 2.5.4. In each fold the
six preferences are equal, bit for bit, for both heads. For the plain value head they are
-0.2386723756790161, -0.19824907183647156, -0.15230244398117065, -0.15230244398117065 and
-0.18880918622016907 in folds 0 to 4. Each fold gives 0.5, and the score is 0.5. The result is the same
with 22, 4 and 1 threads, and with the instruction set of MKL limited to AVX2, AVX or SSE4_2. I did not
reproduce the failure.

**Equal rows can give unequal outputs.** I gave the trained heads of the five folds batches of 1 to 512
rows of zeros. A batch of 6 rows, the size of a fold, is tied for each head and each fold. Many other
sizes are not tied, for example 5, 7, 9 and 11 rows for the plain value head of fold 0. The differences
are 1 to 3 steps of float32, at most 8.94e-08 for the plain value head and 5.59e-09 for the go and no-go
variant. So on this computer the output for equal rows depends on the size of the batch, in the last
bits.

**A cause, not verified.** A processor or a build of the matrix library can compute some rows of a
product in another way than the other rows, for example 4 rows in one block and the 2 remaining rows
apart. Equal rows can then differ in the last bits, and a tie breaks. This can give 0.4333333333333334
exactly. If rows 4 and 5 of each batch of 6 get a slightly higher preference than rows 0 to 3, fold 3
gives 1/6, the four other folds give 0.5, and the mean is 0.4333333333333334. I tried each fixed set of
batch positions with a small shift up or down, 124 patterns. Four give this value exactly. They are two
cases, because a shift of some rows up orders the rows as a shift of the other rows down. In one case
rows 4 and 5 differ from rows 0 to 3, and fold 3 gives 1/6. In the other case rows 1 and 2 differ from
rows 0, 3, 4 and 5, and fold 2 gives 1/6. I could not run the test on the processor of the failed run,
so this cause is a hypothesis.

**The decision, in the owner's words.**

```text
- Yes to diagnose, read only, and then to fix the test. Do not change scripts/analysis/check_choice_head.py or any file under models, simulations or scripts/training, because the results of steps P1 and P2 used that code.
- The fix: in the blank control, round the preferences to 6 decimals before the AUC (for example with a monkeypatch of check.preferences), and assert that the preferences of each fold differ by 1e-6 or less. The expected score stays 0.5 exactly. Add no new test, so the test count stays 2918.
- Record the change in a dated section of the rules file of step P1 that defines this control, with 0 lines removed, before the test change.
```

**The new form of unit test 2.**

1. Inside the test, a monkeypatch of `check.preferences` keeps the preferences of each fold and gives
   the AUC the preferences rounded to 6 decimals (`np.round(x, 6)`). The function in
   `scripts/analysis/check_choice_head.py` does not change.
2. The test asserts that it kept the preferences of 5 folds, and that in each fold the largest
   preference minus the smallest is 1e-6 or less.
3. The expected score stays 0.5 exactly.
4. No test is added. The test keeps its name and its two cases, `variant` and `plain`.

**What the new form still checks.** A difference above 1e-6 between the preferences of a fold fails the
test. On this computer the differences between equal rows are 8.94e-08 or less.

**A limit of the new form.** Rounding gives equal numbers only when the preferences of a fold are on one
side of a boundary of the rounding, which is a multiple of 1e-6 plus 5e-7. On this computer the tied
preferences are 5.6e-08 to 4.8e-07 from the nearest boundary. The smallest distance, 5.6e-08, is in
folds 2 and 3 of the plain value head, the two folds of the hypothesis above. The larger output of that
head has float32 steps of 2.98e-08. So a difference of 2 steps toward that boundary gives two rounded
values, and then the score is not 0.5. I do not know where the preferences of the failed run were, so I
cannot say that the new form passes on that processor.

## Owner decisions of 2026-11-27 on the rounding of unit test 2

Date: 2026-11-27. These are owner decisions in the owner's words. No reviewer judged them. No earlier
line of this file was edited. They replace the rounding to 6 decimals of the section "Owner decision of
2026-11-27 on unit test 2, the blank control" above. That section stays as the record, and its limit
applies only to the rounded form. The change is in `tests/test_choice_head.py` only. No file under
`models`, `simulations` or `scripts` changes.

**The reason.** The section above names a limit of the rounding. A rounding to 6 decimals can split two
preferences that are near a boundary of the rounding. On a simulated processor, a difference of 2 float32
steps toward the nearest boundary of folds 2 and 3 of the plain value head failed the rounded form at the
score. The owner confirmed this check.

**The decisions, in the owner's words.** The first two sentences are the owner's reason.

```text
Your check of the rounding is correct. A rounding to 6 decimals can split two values that are near a rounding boundary. My decisions are these.
- The rounding to 6 decimals is replaced. In the blank control, the patched preferences first assert that the six preferences of the fold differ by 1e-6 or less. Then they return six equal values, each equal to the first preference of the fold. So the AUC gets an exact tie, and the expected score stays 0.5 exactly.
- The limit stays 1e-6. A fold whose preferences differ by more than 1e-6 fails the test, because then the head tells the blank rows apart.
- No change to scripts/analysis/check_choice_head.py or to any file under models, simulations or scripts/training. No new test, so the count stays 2918.
```

**The new form of unit test 2.**

1. Inside the test, a monkeypatch of `check.preferences` calls the function of
   `scripts/analysis/check_choice_head.py` for the test rows of a fold. It asserts that the largest
   preference minus the smallest is 1e-6 or less.
2. Then it returns six equal values, each equal to the first preference of the fold. So the AUC of each
   fold is an exact tie.
3. The test asserts that the patch ran for 5 folds, and that the score is 0.5 exactly.
4. No test is added. The test keeps its name and its two cases, `variant` and `plain`.

**What the new form checks.** A difference above 1e-6 between the preferences of a fold fails the test,
at the assertion of item 1. A difference of 1e-6 or less gives the score 0.5 exactly, wherever the
preferences lie, because the AUC does not see the differences. On this computer the differences between
equal rows are 8.94e-08 or less.
