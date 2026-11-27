# Rules of the check: why the go and no-go variant does not learn on built data

Date: 2026-11-16. Written before any code of this check. The owner chose this check on 2026-11-16
(option B of the decision issue after step P1). No code is written before a review of these rules and
the owner's yes.

Status: rules only. No test, no script and no result exist.

## The question

In the test on built data of step P1 (`docs/results/workspace_choice_head_2026_11.md`), the go and
no-go variant scored 0.302 at 30 rows and 0.361 at 40 rows with the food chooser, at the full signal.
It was below 0.80 in each case. The plain value head scored 0.999 and 0.997 in the same cases.

Why does the variant score below 0.5 with the food chooser, and below 0.80 in each case?

## What this check uses

Built data only. No saved run, no saved vector and no agent run. The data, the choosers, the folds,
the score and the two heads are those of the test on built data of step P1
(`docs/decisions/2026_11_16_workspace_choice_head.md`, `scripts/analysis/check_choice_head.py`). No
training value changes: Adam, learning rate 3e-4, 200 updates, initial seeds 0, 1 and 2.

A repair of the loss of the live agent is the owner's decision. It is not a part of this check. One
condition below changes how the advantage is computed. It is an ablation to test a cause. It is not a
proposal for a repair.

## What I read before these rules

1. The loss of the variant, `variant_loss_terms` in `scripts/analysis/check_choice_head.py`. A unit
   test shows that it equals `ActionSelectionCore.update_policy` on the same rows.
2. The terms. Advantage = outcome - value, with the value detached. Go target = action x (+1 when the
   advantage is above 0, -1 when it is below 0, 0 when it is 0), line 441 of
   `models/self_model/action_selection_core.py`. No-go target = 1 for both channels when the advantage
   is below 0, else 0 (lines 445 to 448). Preference = go x (1 - no-go) of slot 0 minus that of slot 1.
3. The numbers of the test on built data. I did not look at a loss curve, at a value of the critic or
   at a score at another update count. No such number exists yet.

## The reasoning behind the hypotheses

With the food chooser each outcome is 1. An untrained critic gives a value near 0, so the advantage is
above 0 and the go target of the chosen slot is +1. The chosen slot is the slot of the food, so this
target is correct. The critic then learns a value near 1. When its value for a row goes above 1, the
advantage of that row is below 0. Line 441 uses only the sign, so the go target of the chosen slot is
then -1, with the same size as before. The no-go target of both channels becomes 1.

With the random chooser the outcome is 1 in about half of the rows, and the critic cannot know the
choice from the input. A critic that fits the training rows closely has an advantage near 0 on them,
and the sign of a number near 0 is noise.

This reasoning is not a result. Each hypothesis below has a prediction that can fail.

## Conditions

Each condition trains the variant with the fixed training values on the cases of the test on built
data at the full signal (signal share 1.0): 30 and 40 rows, 10 data sets, 3 initial seeds, 5 folds.

| Condition | Content | Choosers |
|---|---|---|
| C1. As in step P1 | 200 updates on the training rows of the fold | random, slot_0, food, food_80 |
| C2. New rows at each update | At each update the head gets a new set of built rows of the same size, from the same generator with the same signal and the same chooser. The seed of a set comes from the data seed, the fold and the update. The rows are standardized with the mean and the deviation of the training rows of the fold. The test rows are those of C1 | random, slot_0, food, food_80 |
| C3. Ablation, the advantage from the untrained critic | The advantage of each update uses the value of a copy of the head at its initial weights. Each loss term stays, and the critic still learns. So the sign of the advantage cannot turn through learning | food only |
| C4. Reference, the plain value head | As C1, with the plain value head | random, slot_0, food, food_80 |

C3 is read for the food chooser only. With an outcome of 0, the sign of the advantage of an untrained
critic depends on the sign of its initial value, so C3 says nothing clear for the other choosers.

Why C2 and not "one update". Hypothesis b names departure D3: the live agent makes one update for each
rollout, and the offline head makes 200 updates on the same rows. The live agent makes many updates in
a run, each on new rows. So the offline form of "one update for each rollout" is many updates, each on
new rows. The score after exactly 1 update on the training rows is also reported, at the checkpoint of
update 1.

## Measurements

For each training, at each update, before the weights change:

1. The share of the rows of the update with an advantage below 0, and the share with an advantage of
   exactly 0.
2. The mean absolute advantage.
3. The mean go signal of the chosen slot, and the mean no-go signal.

For each training, at the checkpoints of 0, 1, 20, 40, 60, 80, 100, 120, 140, 160, 180 and 200 updates:

4. The score on the test rows of the fold, with the preference of the rules.
5. The same score with the go signals only (go of slot 0 minus go of slot 1), with no no-go factor.

A result row is one condition, row count, chooser, data set and checkpoint. It holds the means over the
3 seeds and the 5 folds. Measurements 1 to 3 are means over the updates from the last checkpoint to
this one. "The late updates" are updates 101 to 200.

## Hypotheses and predictions

Each number below is a mean over the 10 data sets, at 30 rows and at 40 rows. A prediction holds only
when it holds at both row counts.

### a. The sign of the advantage turns when the critic is near the outcome

The loss uses only the sign of the advantage (line 441). When the critic is near the outcome, the sign
turns and the go target of the chosen slot becomes -1.

| Prediction | Condition | Holds when |
|---|---|---|
| a1 | C1, food | The share of rows with an advantage below 0 is 0.05 or less at update 1, and its mean over the late updates is above 0.50 |
| a2 | C1, food | The score is 0.80 or more at one checkpoint or more from 1 to 100, and the score at 200 is below 0.50 |
| a3 | C3, food | The score at 200 is 0.80 or more |

Hypothesis a HOLDS when a1, a2 and a3 hold. It FAILS when a3 fails, because then the variant is below
0.80 with no turn of the sign. Each other case is NOT DECIDED.

### b. The cause is the 200 updates on the same rows (departure D3)

With one update for each rollout the sign does not turn.

| Prediction | Condition | Holds when |
|---|---|---|
| b1 | C2, food | The mean share of rows with an advantage below 0 over the late updates is 0.50 or less |
| b2 | C2, food | The score at 200 is 0.80 or more |
| b3 | C2, random | The score at 200 is 0.80 or more |

Hypothesis b HOLDS when b1, b2 and b3 hold. It FAILS when b1 fails or b2 fails. If b1 and b2 hold and b3
fails, the repeat of the rows explains the food chooser and not the random chooser, and the label is
PARTLY.

### c. The no-go factor is the cause (my addition)

The no-go target is 1 for both channels when the advantage is below 0. The factor (1 - no-go) could
remove or turn the preference.

| Prediction | Condition | Holds when |
|---|---|---|
| c1 | C1, food | The score at 200 with the go signals only is 0.80 or more |

Hypothesis c HOLDS when c1 holds. It FAILS when the score with the go signals only is within 0.05 of the
score with the preference of the rules. Each other case is NOT DECIDED.

### d. With the random chooser, the critic fits the training rows and the sign becomes noise (my addition)

| Prediction | Condition | Holds when |
|---|---|---|
| d1 | C1, random | The mean absolute advantage over updates 181 to 200 is below 0.25 |
| d2 | C1, random | The highest score at a checkpoint from 1 to 180 is 0.05 or more above the score at 200 |
| d3 | C2, random | The score at 200 is 0.80 or more. This is b3 |

Hypothesis d HOLDS when d1, d2 and d3 hold. It FAILS when d1 fails. Each other case is NOT DECIDED.

The limit of 0.25 is half of 0.5. A critic that does not know the choice has a best value near 0.5 with
the random chooser, and an absolute advantage near 0.5.

### What I expect, stated before the check

I expect that a holds, that b fails at b1 and b2 (the critic also reaches the outcome with new rows),
that c fails, and that d holds in d1 and d2. I do not know d3. This is an expectation and it has no
part in a label.

## Controls with exact expected values

| Control | Expected value | On a mismatch |
|---|---|---|
| V1. C1 at 200 updates against step P1 | The score of each case equals the score of the same case and head in `built_test.csv`, within 1e-9 | The check is VOID |
| V2. C4 at 200 updates against step P1 | The same, for the plain value head | The check is VOID |
| V3. The measurements do not change the training | The trained weights of one case are equal with and without the measurements | The check is VOID |
| V4. Count of result rows | 3120, see "Cost" | The job is not complete |
| V5. C3 | The value that gives the advantage is equal at update 1 and at update 200 | The check is VOID |

## Unit tests, written before the code of the check

1. The share of rows with an advantage below 0 and the mean absolute advantage, for a hand-made case.
2. The go target of the chosen slot is +1, -1 and 0 for an advantage above 0, below 0 and of 0.
3. The score with the go signals only, for hand-made signals.
4. The new rows of C2 differ between two updates and are the same for the same seed, fold and update.
5. The advantage of C3 does not change when the critic learns.
6. A checkpoint at 200 updates gives the same weights as `train_head` of step P1.
7. The count of result rows is 3120.

## What each result means for step P2

Step P2 is the build of the choice mode behind a flag that is off by default. The owner decides it.
This table says what I bring to the owner in each case.

| Result | Meaning | For step P2 |
|---|---|---|
| a HOLDS and b FAILS | The turn of the sign belongs to the loss. It also comes with new rows at each update, as in the live agent | Do not build with the variant as it is. The choice is to build with the plain value head, or I bring a repair of the loss to the owner. The repair is the owner's decision |
| b HOLDS | The defect comes from the offline training of step P1 (departure D3), not from the loss with new rows | Build with the variant is possible. The score of the variant in step P1 is then a property of the offline training |
| b PARTLY | New rows repair the food chooser and not the random chooser | The variant does not learn from the outcome with new rows. Build with the plain value head, or bring a repair to the owner |
| a FAILS | The cause is not the turn of the sign | Not known. I report the measurements and propose no build with the variant |
| c HOLDS | The cause is the no-go factor of the preference, which is a rule of step P1 and not a part of the loss | A change of the read of the preference is a new rule for the owner. No statement on the loss |
| d HOLDS with a HOLDS | Two effects of one cause: the loss keeps only the sign of an advantage that goes to 0 | As in the first row |
| Each other case | NOT DECIDED | I report the table and the owner decides |

One limit applies to each row. A score with the food chooser can come from the choice alone, as the
second pass of step P1 showed for the plain value head. So a high score with the food chooser does not
show that a head learns from the outcome. Only the random and slot_0 choosers test that. For step P2
the results b3 and d3 with the random chooser carry more weight than a result with the food chooser.

## Limits, stated before the check

1. Built data with one signal strength. Nothing here is a statement on the broadcast or on a saved run.
2. The check is offline. The live agent has a shaped reward, a return over a rollout, a working memory
   and a relay (departures D1, D2 and D4 of step P1). A cause found here is a cause in the offline
   variant.
3. C2 gives the head 200 x 24 or 200 x 32 different rows. This is far more data than a unit of step P1
   has. C2 tests the cause of the defect. It does not show what a head learns from 30 or 40 episodes.
4. C3 is not a repair. A critic that never informs the advantage is not a proposal for the agent.
5. The limits 0.80, 0.50, 0.05 and 0.25 are set here, before a number exists. 0.80 is the limit of step
   P1. 0.50 is chance for the score and one half for a share.
6. A label of HOLDS for one hypothesis is not a proof of the cause. C3 is the one ablation. For
   hypotheses b and d the evidence is one manipulation each.
7. The results are for initial seeds 0, 1 and 2 and for 10 data sets.

## Cost

| Condition | Cases | Trainings |
|---|---|---|
| C1 | 2 row counts x 4 choosers x 10 data sets = 80 | 80 x 3 seeds x 5 folds = 1200 |
| C2 | 80 | 1200 |
| C3 | 2 row counts x 1 chooser x 10 data sets = 20 | 300 |
| C4 | 80 | 1200 |
| Total | 260 | 3900 |

Result rows: 260 cases x 12 checkpoints = 3120.

Time. In step P1 one training of the variant took 1.00 to 1.28 seconds and one of the plain value head
took 0.18 seconds, measured on built data. 2700 trainings of the variant at 1.05 seconds are about 47
minutes, and 1200 of the plain value head are about 4 minutes. The measurements at each update and the
12 scores add time that I did not measure. My estimate is 1.0 to 1.5 hours, on one CPU thread, as one
job that runs alone. This is an estimate.

## Files

- Rules: this file.
- Script, after the review and the owner's yes: a mode of `scripts/analysis/check_choice_head.py`.
- Tests, before the script: `tests/test_choice_head.py`.
- Output: `docs/results/data/choice_head_variant_sign_2026_11/`.
- Result document: `docs/results/choice_head_variant_sign_2026_11.md`, with the label of each
  hypothesis first.


## Changes after the independent review

Date: 2026-11-16. The independent review is on the review issue of these rules, in two parts. The
owner accepted its 28 necessary changes and no optional change. The owner also confirmed the limit of
review point 2: a repair of the loss of the live agent is not a part of this check, and an ablation in
the analysis script is allowed if no file under `models/` changes and the result document does not
present it as a repair. Each code claim of the review was checked against the code at `b65d992` before
these changes, and each one agrees.

This section replaces each part of the rules above that it names. Where it and an earlier section
disagree, this section holds. No code exists.

### The conditions, after the changes

| Condition | Content | Choosers | Change |
|---|---|---|---|
| C1. As in step P1 | 200 updates on the training rows of the fold | random, slot_0, food, food_80 | |
| C2. No repeat of rows | A new set of built rows at each update, as described in "Conditions". It tests the repeat of rows only. It does not test the count of updates | random, slot_0, food, food_80 | 8 |
| C3a. Go sign from the untrained critic | The sign in the go target comes from a copy of the head at its initial weights. The no-go target and the brake target use the learning critic | food | 1 |
| C3b. No target turns | Each of the three targets (go, no-go, brake) uses the untrained critic. This is C3 of the rules above | food | 3 |
| C3c. No-go from the untrained critic | The no-go target uses the untrained critic. The go target and the brake target use the learning critic | food | 2 |
| C4. Reference, the plain value head | As C1, with the plain value head | random, slot_0, food, food_80 | |

In each C3 condition the critic still learns, and the value term is unchanged. Condition C3 of the
rules above is replaced by C3a, C3b and C3c (changes 1 to 3).

**Change 7.** The brake target and the no-go target stay together in C1, C2 and C3a. C3c changes the
no-go target alone and keeps the brake target with the learning critic, and C3b changes both. No
condition changes the brake target alone. The check cannot separate an effect of the brake target from
an effect of the no-go target, and the result document says so.

**Change 27.** C3a, C3b and C3c are a mode of `scripts/analysis/check_choice_head.py`. No file under
`models/` changes. The result document does not present C3a, C3b or C3c as a repair of the loss. A
repair of the loss of the live agent is the owner's decision and is not a part of this check.

**Change 12.** The check ends at 200 updates. The count of updates of the live agent in a run is not
known, and this check makes no statement on it.

### The measurements, after the changes

The measurements 1 to 5 of the rules above stay. These are added.

6. **For each training, the first update at which the share of rows with a negative advantage is above
   0.50** (change 11). If no update has it, the value is "none".
7. **In C3a, the share of rows with a negative advantage of the learning critic over updates 181 to
   200** (change 4). In each C3 condition the advantage of the untrained critic and the advantage of
   the learning critic are both recorded.
8. **The difference of the score at checkpoint 1 and the score at checkpoint 0**, for each case
   (change 10). It is reported. It is not a control and gives no VOID.

**Change 16. Rows for each training.** Next to the mean rows, the check writes one row for each
training and checkpoint: condition, row count, chooser, data set, initial seed, fold, checkpoint, the
score, the score with the go signals only, and the measurements 1 to 3 and 6 to 7 for that training.
The relation between the turn of the sign and the fall of the score in hypothesis a is read from these
rows (see a2 below). A mean share above 0.50 and a mean score below 0.50 can come from different
trainings, so the means alone do not show it.

**Change 17. The window.** "The late updates" of the rules above (updates 101 to 200) is replaced by
updates 181 to 200 wherever a share of negative advantage is read for a label. The reason is in the
review: the critic can swing around a constant outcome, and a window of 100 updates can hide a turn
that is present at the end.

### The limit of 0.05

**Change 19.** Each limit of 0.05 in the rules (a1 part 1, c, d2) becomes the larger of 0.05 and 2
standard errors of the mean over the 10 data sets. The standard error is computed from the 10 values
of the data sets of the case that the limit is applied to. It is stated here, before the check. The
result document gives the value of each limit as used.

### Which choosers the hypotheses cover

**Change 20.** Hypotheses a, b, c and d cover the food chooser and the random chooser only. The
slot_0 and food_80 choosers are measured and reported with no label. The result document says that the
question "why below 0.80 in each case" is not answered for slot_0 and food_80.

**Change 14. The copy of the choice.** With the food chooser the chosen slot is the slot of the food in
each row, so a head can score high from a copy of the choice with no learning from the outcome. A high
score with the food chooser in a2, a3, b2 and c1 can be a copy of the choice. Each of these predictions
says so in the result document. Only the random chooser has no copy path, because its choice does not
depend on the target.

### Hypothesis a, after the changes

The predictions of a stay, with these changes.

| Prediction | Condition | Holds when |
|---|---|---|
| a1, part 1 | C1, food | The share of rows with a negative advantage is at most the limit of change 19 at update 1. This is a precondition, not evidence |
| a1, part 2 | C1, food | The mean share of rows with a negative advantage over updates 181 to 200 is above 0.50 (change 17) |
| a2 | C1, food | The score is 0.80 or more at one checkpoint or more from 20 to 100, and the score at 200 is below 0.50. A high early score can be a copy of the choice (change 14). The evidence is the later fall. The rows for each training are reported with the count of trainings in which the first update of measurement 6 comes before the first checkpoint with a score below 0.50 |
| a3 | C3a, food | The score at 200 is 0.80 or more. Read with measurement 7: if the share of the learning critic over updates 181 to 200 is 0.50 or less, the no-go target did not turn in C3a, and C3a says nothing about hypothesis c (change 4) |
| C3b | C3b, food | The score at 200 is 0.80 or more. This is the case in which no target turns. A score of 0.80 or more is a copy of the choice and shows only that the go path can hold the choice when no target turns |
| C3c | C3c, food | The score at 200 is 0.80 or more |

**Change 5. The label of a.** This replaces the label rule of a in the rules above.

| Label | When |
|---|---|
| FAILS | In C1 with the food chooser the mean share of negative advantage over updates 181 to 200 is 0.50 or less, so no turn occurs. Or C3b at 200 is below 0.80, so the score stays low when no target can turn |
| HOLDS | a1, a2 and C3b at 0.80 or more hold, and C3a at 200 is 0.80 or more. A stop of the turn of the go target alone repairs the score |
| PARTLY | a1, a2 and C3b at 0.80 or more hold, and C3a at 200 is below 0.80. The turn of the advantage causes the low score, and the go target alone does not explain it. C3c then shows if a stop of the turn of the no-go target alone repairs the score |
| NOT DECIDED | Each other case, also when part 1 of a1 fails |

### Hypothesis b, after the changes

**Change 8.** Hypothesis b is tested in two readings. Reading 1 is the repeat of the same rows, tested
by C2. Reading 2 is the count of updates, tested by the checkpoints of C1. A condition with fewer
updates on the same rows is not necessary: a run of k updates is equal to the checkpoint at k of a run
of 200, because the loop has no value that depends on the total count.

Reading 1 keeps b1, b2 and b3, with the window of change 17 in b1. **Change 15.** b3 stays necessary
for "b HOLDS". A high score in b2 can be a copy of the choice (change 14).

**Change 9. Reading 2, the count of updates.**

| Prediction | Condition | Holds when |
|---|---|---|
| b4 | C1, food | At each checkpoint from 20 on, before the first checkpoint at which the mean share of negative advantage is above 0.50, the mean score is above 0.50. Checkpoints 0 and 1 are not used, because the score of an untrained head is near 0.5 |

Reading 2 HOLDS when b4 holds and a1 part 2 holds. It FAILS when the score is 0.50 or less at a
checkpoint from 20 on before the share goes above 0.50. If the share never goes above 0.50, reading 2
is NOT DECIDED. The label of b stays the label of reading 1. Reading 2 is reported next to it.

### Hypothesis c, after the changes

c1 stays. A high score with the go signals only can be a copy of the choice (change 14). The limit of
its FAILS rule is the limit of change 19. The review notes that the factor (1 - no-go) is 0 or more,
because the no-go signal comes from a sigmoid, and the no-go target is the same for both channels. So c
can explain a score below 0.80 and cannot easily explain a score far below 0.50. This is a reading of
the code and not a prediction.

### Hypothesis d, after the changes

**Change 21.** Hypothesis d is a mechanism of b for the random chooser. b3 and d3 are one number. The
label of d rests on d1 and d2.

| Prediction | Condition | Holds when |
|---|---|---|
| d1 | C1, random | The mean absolute advantage over updates 181 to 200 is below 0.25. Unchanged |
| d2 | C1, random | The mean score at checkpoints 40, 60, 80 and 100 is above the mean score at checkpoints 180 and 200 by at least the limit of change 19 (change 18). This replaces "the highest score at a checkpoint from 1 to 180", which can hold from noise alone |

d HOLDS when d1 and d2 hold. It FAILS when d1 fails. Each other case is NOT DECIDED. d3 is reported as
b3.

### Controls, after the changes

V1, V2, V3 and V5 stay. V5 applies to C3a, C3b and C3c.

| Control | Expected value | On a mismatch |
|---|---|---|
| V4. Count of result rows | 3600 mean rows and 54000 rows for each training and checkpoint (change 26) | The job is not complete |
| V6. The untrained critic in each C3 condition (change 6) | With the food chooser the share of rows with a negative advantage of the untrained critic is 0 at each update | The check is VOID |
| V7. The outcome with the food chooser (change 13) | The outcome of each row is 1. This replaces a pass with each outcome set to 1 | The check is VOID |
| V8. The seeds of C2 (change 25) | The seed of each new row set comes from a seed sequence that cannot give a data seed 0 to 9 | The check is VOID |
| V9. The test rows of C2 (change 25) | The test rows of each fold of C2 equal those of C1 | The check is VOID |

The difference of the scores at checkpoints 0 and 1 is not a control (change 10).

### Unit tests, after the changes

Unit tests 1 to 6 of the rules above stay. Unit test 7 counts 3600 mean rows and 54000 rows for each
training and checkpoint (change 26). **Change 28** adds these tests, written before the code of the
check:

8. In C3a only the sign of the go target uses the untrained critic. The no-go target and the brake
   target use the learning critic. Checked on a hand-made case in which the two critics give
   advantages of different sign.
9. In C3b each of the three targets uses the untrained critic, on the same case.
10. In C3c only the no-go target uses the untrained critic, on the same case.
11. The rows for each training have one row for each condition, row count, chooser, data set, initial
    seed, fold and checkpoint, and their mean over seeds and folds equals the mean row.
12. The seed sequence of C2 gives no data seed 0 to 9, and the same seed, fold and update give the same
    rows.

### What each result means for step P2, after the changes

This table replaces the table of the rules above.

| Result | Meaning | For step P2 |
|---|---|---|
| VOID (change 24) | A control failed | No statement. The author reports the control and the owner decides |
| a HOLDS and b FAILS | The turn of the sign of the go target explains the low score with the food chooser, and it also comes with new rows at each update | Do not build with the variant as it is. The choice is to build with the plain value head, or the author brings a repair of the loss to the owner. The advice rests mostly on the result of step P1 with the random chooser, because a explains the food chooser only |
| a PARTLY and b FAILS (change 24) | The turn of the advantage explains the low score, and the go target alone does not | As the row above, and the result of C3c is reported with it |
| b HOLDS (change 22) | The repeat of rows causes the defect in the offline check | The check does not show learning with the amount and the kind of data of a run: C2 gives 4800 or 6400 independent rows, and the rows of a rollout are not independent. A build with the variant then needs a check at the amount of data of a run |
| b PARTLY | New rows repair the food chooser and not the random chooser | The variant does not learn from the outcome with new rows. Build with the plain value head, or bring a repair to the owner |
| a FAILS | The turn of the sign is not the cause | Not known. The author reports the measurements and proposes no build with the variant |
| c HOLDS | The cause is the no-go factor of the preference, which is a rule of step P1 and not a part of the loss | A change of the read of the preference is a new rule for the owner. No statement on the loss |
| d HOLDS and a HOLDS (change 23) | Two results: the turn of the sign with the food chooser, and a critic that fits the training rows with the random chooser | No claim of one cause. Each result is read by its own row |
| Each other case | NOT DECIDED | The author reports the table and the owner decides |

**Change 24. More than one row.** When more than one row applies, each row is reported. The most
restrictive advice holds for step P2, in this order: VOID, then "do not build with the variant", then
"a build needs a check at the amount of data of a run", then "a new rule for the owner". The slot_0
and food_80 choosers have no row (change 20).

### Cost, after the changes

**Change 26.**

| Condition | Cases | Trainings |
|---|---|---|
| C1 | 2 row counts x 4 choosers x 10 data sets = 80 | 80 x 3 seeds x 5 folds = 1200 |
| C2 | 80 | 1200 |
| C3a | 2 row counts x 1 chooser x 10 data sets = 20 | 300 |
| C3b | 20 | 300 |
| C3c | 20 | 300 |
| C4 | 80 | 1200 |
| Total | 300 | 4500 |

Result rows: 300 cases x 12 checkpoints = 3600 mean rows, and 4500 trainings x 12 checkpoints = 54000
rows for each training and checkpoint.

Time. 3300 trainings of the variant at 1.05 seconds are about 58 minutes, and 1200 trainings of the
plain value head at 0.18 seconds are about 4 minutes, together about 61 minutes. The two times were
measured on built data in step P1 and are not measured for this check. The measurements at each
update, the second critic of the C3 conditions, the 12 scores of each training and the 54000 rows add
time that I did not measure. My estimate is 1.2 to 2 hours, on one CPU thread, as one job that runs
alone. This is an estimate.


## Changes after the independent review, part 3

Date: 2026-11-16. Part 3 of the independent review is on the review issue of these rules. It reviewed
the rules at commit `20d90c0`. The owner accepted its 7 necessary changes and its optional changes 1,
2, 3 and 4. The owner did not take optional change 5. Each code claim of part 3 was checked against the
code at `20d90c0` before these changes, and each one agrees.

This section replaces each part of the rules above that it names. Where it and an earlier section
disagree, this section holds. No earlier line of this file was edited. No code exists.

In this section, "necessary change" and "optional change" with a number are changes of part 3. "Change"
with a number and no other word is one of the 28 changes of parts 1 and 2.

### Necessary change 1. Reading 2 of hypothesis b

Prediction b4 stays. This label rule replaces the label rule of reading 2 in "Hypothesis b, after the
changes".

| Label | When |
|---|---|
| HOLDS | a1 part 2 holds, one checkpoint or more from 20 on comes before the first checkpoint with a mean share above 0.50, and the mean score is above 0.50 at each such checkpoint |
| FAILS | The mean score is 0.50 or less at one such checkpoint |
| NOT DECIDED | Each other case, also when no checkpoint from 20 on comes before the turn |

A score above 0.50 in b4 can be a copy of the choice. Reading 2 has no row in the table for step P2 and
changes no advice. The label of b stays the label of reading 1.

### Necessary change 2. The 10 values of the limit of change 19

For the FAILS rule of c the 10 values are, for each data set, the score with the go signals only minus
the score with the preference of the rules, at 200. For d2 they are, for each data set, the mean score
at checkpoints 40, 60, 80 and 100 minus the mean score at checkpoints 180 and 200. The standard error
is the sample standard deviation of the 10 values with the divisor 9, divided by the square root of 10.
It is computed apart at 30 rows and at 40 rows.

Part 3 of the review does not state the 10 values for a1 part 1. It shows that this limit cannot decide
a label. The share at update 1 in C1 with the food chooser is the number that V6 reads at update 1. If
V6 passes, the share is 0 and its standard error is 0. If V6 does not pass, the check is VOID.

### Necessary change 3. The row counts of the FAILS rules

Owner decision, 2026-11-16. The FAILS rule of a, of c and of reading 2 must be met at both row counts.
If it is met at one row count only, the label is NOT DECIDED.

### Necessary change 4 and optional change 4. V6, V7 and V9 first

The job computes V6 and V7 for each C3 case before the first training and stops on a mismatch. V9 also
runs before the first training. Part 3 of the review counts 8400 values of the untrained critic for V6:
3 initial seeds, 10 data sets, 5 folds, and 24 and 32 training rows.

### Necessary change 5. V8 and unit test 12 compare rows

Part 3 of the review found that a seed sequence of the data seed, fold 0 and update 0 gives the same
random numbers as the data seed alone. A control that compares seed numbers passes in this case, and
the new row set is then a part of the data set, with test rows among them.

New expected value of V8. No row of a new row set of C2 is equal to a row of the data set of its case,
before the rows are standardized. On a mismatch the check is VOID.

New unit test 12. No row of the new row set of C2 for fold 0 and the first update is equal to a row of
the data set of its case, before the rows are standardized. This is the case that fails with a seed
sequence of the data seed, the fold and the update when the update count starts at 0. The same seed,
fold and update give the same rows.

### Necessary change 6. The order of advice

This replaces the order in "Change 24. More than one row". When more than one row applies, each row is
reported, and the advice with the first place in this order holds for step P2.

| Place | Advice | Rows |
|---|---|---|
| 1 | VOID | "VOID" |
| 2 | Do not build with the variant | "a HOLDS and b FAILS", "a PARTLY and b FAILS", "b PARTLY" and "a FAILS" |
| 3 | A build needs a check at the amount of data of a run | "b HOLDS" |
| 4 | A new rule for the owner | "c HOLDS" |
| 5 | The author reports the table and the owner decides | "Each other case" |

The labels of d give no advice. The row "d HOLDS and a HOLDS" is a statement on the cause only. The
slot_0 and food_80 choosers have no row (change 20).

### Necessary change 7. The record of the section "Changes after the independent review"

Four changes of that section are changes of the author, not of the review.

| Change of the author | Place in that section | Judgment of part 3 |
|---|---|---|
| a2 uses checkpoints 20 to 100. The first rules said 1 to 100 | "Hypothesis a, after the changes" | ACCEPT |
| b1 uses updates 181 to 200. Change 17 names a1 part 2 only | "Change 17. The window" | ACCEPT |
| Reading 2 of hypothesis b has its own label | "Change 9. Reading 2, the count of updates" | CHANGE. Necessary change 1 gives the new rule |
| The limit of change 19 applies to a1 part 1 | "The limit of 0.05" | ACCEPT, with the definition of necessary change 2 |

In "Change 14. The copy of the choice", the sentence "Only the random chooser has no copy path, because
its choice does not depend on the target" is replaced by this. The random and slot_0 choosers have no
copy path. Only the random chooser has a label.

### Optional change 1. V6 with an advantage of 0

New expected value of V6. With the food chooser no advantage of the untrained critic is 0 or below, at
each update. A value of exactly 1 gives an advantage of 0 and a go target of 0. On a mismatch the check
is VOID.

### Optional change 2. Measurement 6 in the C3 conditions

Measurement 6 uses the advantage of the learning critic in each C3 condition.

### Optional change 3. Limits 4 and 6

In limit 4 of the rules, "C3" is replaced by "C3a, C3b and C3c". In limit 6, "C3 is the one ablation"
is replaced by "C3a, C3b and C3c are the ablations".

### Counts and cost, after part 3

| Item | Before part 3 | After part 3 |
|---|---|---|
| Controls | 9, V1 to V9 | 9, V1 to V9. V6 and V8 have a new expected value. V6, V7 and V9 run before the first training |
| Unit tests | 12 | 12. Unit test 12 has a new content |
| Cases, trainings, mean rows, rows for each training and checkpoint | 300, 4500, 3600, 54000 | No change |
| Estimated time | 1.2 to 2 hours | No change |

Part 3 adds no training and no result row. It adds the 8400 values of V6 before the first training and
the comparison of rows of V8, for at most 1200 trainings x 200 updates = 240000 new row sets of C2. The
time of this work is not measured. The estimate stays 1.2 to 2 hours, on one CPU thread, as one job
that runs alone. This is an estimate.


## Owner decisions after part 3

Date: 2026-11-16. After part 3 was applied, the author reported 5 points that are not in the review.
The owner decided each one. These are owner decisions. No reviewer judged them. Point 3 (the 10 values
for a1 part 1) has the decision "no change".

This section replaces each part of the rules above that it names. Where it and an earlier section
disagree, this section holds. No earlier line of this file was edited.

### Owner decision 1. The FAILS rule of a

The FAILS rule of a has two ways. They are a mean share of negative advantage of 0.50 or less over
updates 181 to 200 in C1 with the food chooser, and C3b at 200 below 0.80. The same way must be met at
both row counts. If one way is met at 30 rows and the other way at 40 rows, or a way is met at one row
count only, the label is NOT DECIDED.

### Owner decision 2. The HOLDS rule and the PARTLY rule of a

Each condition of the HOLDS rule and of the PARTLY rule of a must be met at both row counts. If C3a at
200 is 0.80 or more at one row count and below 0.80 at the other, the label is NOT DECIDED.

### Owner decision 4. Unit test 13

New unit test 13. On a mismatch of V6, V7 or V9 the job stops before the first training.

### Owner decision 5. V8 and unit test 12

V8 and unit test 12 compare the rows before the signal is added. This replaces "before the rows are
standardized" in necessary change 5. The other words of V8 and of unit test 12 stay.

### Count of unit tests

13 unit tests. Unit tests 1 to 12 are those of the sections above, with unit test 12 as changed here.
Unit test 13 is new. The count of controls stays 9, and the cost does not change.


## Owner decisions on the label code

Date: 2026-11-21. When the label code was written, the author found two cases in which the rules give
no single label, and one difference between the hypotheses. The owner decided each one. These are
owner decisions. No reviewer judged them.

This section replaces each part of the rules above that it names. Where it and an earlier section
disagree, this section holds. No earlier line of this file was edited.

### Owner decision 1. Label of a when part 1 of a1 fails

When a way of the FAILS rule of a is met at both row counts and part 1 of a1 fails, the label of a is
NOT DECIDED. Reason: part 1 of a1 is the precondition of the check of a. VOID keeps the first place in
the order of advice.

### Owner decision 2. Label of c when both of its rules are met

When c1 holds and the score with the go signals only is within the limit of the score with the
preference of the rules, the label of c is FAILS. Reason: the score does not change when the no-go
factor is removed, so the no-go factor is not the cause. A high score with the go signals only can be a
copy of the choice.

### Owner decision 3. The FAILS rules of b and d

No change. The result document states that b and d get FAILS when a prediction holds at one row count
only, and that a, c and reading 2 get NOT DECIDED in the same kind of case by the owner decisions
after part 3 and by necessary change 3 of part 3.
