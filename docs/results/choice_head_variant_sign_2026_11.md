# Check on the go and no-go variant: why it scores below 0.80 on built data

Date: 2026-11-22. Rules: `docs/decisions/2026_11_16_choice_head_variant_sign.md`, with each section up
to "Owner decisions on the label code". Code: `scripts/analysis/check_choice_head.py` at commit
`1a0067c`. Rows: `docs/results/data/choice_head_variant_sign_2026_11/mean_rows.csv` (called "the mean
rows" below) and `docs/results/data/choice_head_variant_sign_2026_11/training_rows.csv` (called "the
training rows" below). The output of the label code is
`docs/results/data/choice_head_variant_sign_2026_11/labels.json` (called "labels.json" below).

Hypothesis b (reading 1) FAILS and hypothesis c FAILS. No control failed, and the check is not VOID.
The label code ran with no error.

## Labels

| Hypothesis | Label |
|---|---|
| a. The sign of the advantage turns when the critic is near the outcome | HOLDS |
| b. The cause is the 200 updates on the same rows (reading 1, condition C2) | FAILS |
| Reading 2 of b. The count of updates | HOLDS |
| c. The no-go factor is the cause | FAILS |
| d. With the random chooser, the critic fits the training rows | HOLDS |

Each label comes from the command below on the two row files. Its output was read in full.

```
python -m scripts.analysis.check_choice_head --variant-sign-labels docs/results/data/choice_head_variant_sign_2026_11
```

A separate script read the mean rows with pandas and gave the same numbers for each value in the
tables below. Each number is a mean over the 10 data sets, from the mean rows, unless the text says
the training rows.

## The job

| Item | Value |
|---|---|
| Command | `python -u -m scripts.analysis.check_choice_head --variant-sign docs/results/data/choice_head_variant_sign_2026_11 --step-p1-csv docs/results/data/workspace_choice_head_2026_11/built_test.csv` |
| Interpreter | Python 3.12.10, torch 2.14.1, one CPU thread |
| Start and end | Started 2026-11-22 01:44:17. The last row was written at 02:56, and the process had ended at 02:57:08 |
| Other jobs | None. No other training, probe or test run was active at the start |
| Error output | Empty |

## Controls

| Control | Result | Source |
|---|---|---|
| V1. C1 at 200 equals step P1 within 1e-9 | Passed | The job compares each case of C1 before its rows are written, and stops on a mismatch. Each case was written. The complete counts of V4 alone show that no control mismatch occurred, because a mismatch stops the job before the rows of its case are written. The last case in the order (C4, 40 rows, food_80, data set 9) is checked by V2 before its rows are written. This record does not depend on the error output, which is not in the commit |
| V2. C4 at 200 equals step P1 within 1e-9 | Passed | As V1, for each case of C4 |
| V3. The measurements do not change the weights | Passed | The job checks the first case of C1 before the first row is written, and stops on a mismatch |
| V4. 3600 mean rows and 54000 training rows | Passed | `wc -l` gives 3601 and 54001 lines, each with one header line. The label code checks the counts again before it computes a label |
| V5. The value of the untrained critic is equal at the first and the last update | Passed | Checked in each training of C3a, C3b and C3c. A mismatch stops the job |
| V6. No advantage of the untrained critic is 0 or below | Passed | Before the first training, 8400 values for each of C3a, C3b and C3c, highest untrained value 0.07706 (job output). At each update in the training, a mismatch stops the job. The highest `untrained_share_at_or_below_0` in the training rows of C3 is 0.0 |
| V7. Each food outcome is 1 | Passed | 4200 rows before the first training (job output), and each new row set of C2 in the training |
| V8. No new row of C2 equals a row of the data set, before the signal is added | Passed | Checked for each new row set of C2. A mismatch stops the job |
| V9. The test rows of C2 equal those of C1 | Passed | 400 folds before the first training (job output) |

## Numbers behind the labels

### Hypothesis a, HOLDS

| Prediction | Condition | 30 rows | 40 rows | Holds |
|---|---|---|---|---|
| a1 part 1. Share of negative advantage at update 1 at most the limit | C1, food | 0.0 | 0.0 | Yes at both |
| a1 part 2. Mean share of negative advantage over updates 181 to 200 above 0.50 | C1, food | 0.5677 | 0.5526 | Yes at both |
| a2. Score 0.80 or more at a checkpoint from 20 to 100, and score at 200 below 0.50 | C1, food | highest 1.0, at 200 0.3022 | highest 1.0, at 200 0.3612 | Yes at both |
| a3. Score at 200 is 0.80 or more | C3a, food | 1.0 | 1.0 | Yes at both |
| C3b. Score at 200 is 0.80 or more | C3b, food | 1.0 | 1.0 | Yes at both |
| C3c. Score at 200 is 0.80 or more | C3c, food | 0.7452 | 0.8117 | At 40 rows only |
| Measurement 7. Share of negative advantage of the learning critic over updates 181 to 200 | C3a, food | 0.5889 | 0.6076 | Above 0.50, so C3a can say something about c |

a HOLDS because a1, a2 and C3b hold at both row counts and C3a at 200 is 0.80 or more at both row
counts. C3c is reported with the label and does not change it.

In C3c the score at 200 is 0.7452 and 0.8117, against 0.3022 and 0.3612 in C1. A stop of the turn of
the no-go target alone raises the score by 0.44 and 0.45. So the turn of the go target is sufficient
for the low score (C3a), and the no-go target also has an effect. c FAILS is about the no-go factor in
the read of the preference, not about the no-go target in the loss.

The check cannot separate an effect of the brake target from an effect of the no-go target. C3c
changes the no-go target alone, C3b changes both, and no condition changes the brake target alone
(change 7).

Share of negative advantage of the learning critic over updates 181 to 200, from the mean rows at
checkpoint 200:

| Condition | 30 rows | 40 rows |
|---|---|---|
| C3b, food | 0.5777 | 0.5669 |
| C3c, food | 0.4758 | 0.4729 |

Score of C1 with the food chooser at each checkpoint, from the mean rows:

| Checkpoint | 0 | 1 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Score, 30 rows | 0.5822 | 0.7822 | 1.0 | 1.0 | 1.0 | 0.9985 | 0.9230 | 0.3252 | 0.2000 | 0.3481 | 0.4030 | 0.3022 |
| Share of negative advantage, 30 rows | | 0.0 | 0.0 | 0.4425 | 0.2815 | 0.4095 | 0.5040 | 0.5594 | 0.5384 | 0.5295 | 0.5458 | 0.5677 |
| Score, 40 rows | 0.5717 | 0.8096 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9883 | 0.3912 | 0.1342 | 0.3600 | 0.4525 | 0.3612 |
| Share of negative advantage, 40 rows | | 0.0 | 0.0 | 0.4297 | 0.2734 | 0.3997 | 0.4886 | 0.5483 | 0.5502 | 0.5190 | 0.5394 | 0.5526 |

The share at a checkpoint is the mean over the updates from the checkpoint before to this one.

Trainings of C1 with the food chooser, from the training rows. Each row count has 150 trainings (3
initial seeds, 10 data sets, 5 folds).

| Row count | Trainings | First update with a share above 0.50 at or before the first checkpoint with a score below 0.50 | No such update | No score below 0.50 |
|---|---|---|---|---|
| 30 | 150 | 95 | 0 | 0 |
| 40 | 150 | 101 | 0 | 0 |

The count reads each checkpoint, also checkpoint 0. Each of the other 55 trainings at 30 rows and 49
at 40 rows has its first score below 0.50 at checkpoint 0, before any update. From checkpoint 20 on,
the first update with a share above 0.50 comes at or before the first score below 0.50 in 150 of 150
trainings at each row count. The first update with a share above 0.50 is 28 to 31 at 30 rows and 29
to 32 at 40 rows, in each training. The mean score stays at 0.9985 or more to checkpoint 80, and falls
below 0.50 at checkpoint 120. So the score falls about 90 updates after the first turn.

### Hypothesis b, reading 1, FAILS

| Prediction | Condition | 30 rows | 40 rows | Holds |
|---|---|---|---|---|
| b1. Mean share of negative advantage over the late updates 0.50 or less | C2, food | 0.5086 | 0.5088 | No at both |
| b2. Score at 200 is 0.80 or more | C2, food | 0.5948 | 0.3458 | No at both |
| b3. Score at 200 is 0.80 or more | C2, random | 1.0 | 1.0 | Yes at both |

b FAILS because b1 fails and b2 fails. Each fails at both row counts. b1 fails by 0.0086 and 0.0088
only. b FAILS does not depend on b1, because b2 fails at both row counts. b FAILS comes from the food
chooser only, and b3 holds.

### Reading 2 of b, HOLDS

| Item | 30 rows | 40 rows |
|---|---|---|
| First checkpoint with a mean share above 0.50 (C1, food) | 100 | 120 |
| Checkpoints from 20 on before it | 20, 40, 60, 80 | 20, 40, 60, 80, 100 |
| Mean score at those checkpoints | 1.0, 1.0, 1.0, 0.9985 | 1.0, 1.0, 1.0, 1.0, 0.9883 |
| b4 | Holds | Holds |

Reading 2 HOLDS because b4 holds and a1 part 2 holds at both row counts. Reading 2 has no row in the
table for step P2 and changes no advice.

Each score of b4 is with the food chooser and can be a copy of the choice (necessary change 1 of part
3). So reading 2 HOLDS does not show that the variant learns from the outcome with few updates. The
check ends at 200 updates, and the count of updates of the live agent in a run is not known (change
12).

### Hypothesis c, FAILS

| Item | Condition | 30 rows | 40 rows |
|---|---|---|---|
| c1. Score at 200 with the go signals only, 0.80 or more | C1, food | 0.3022, no | 0.3542, no |
| Score at 200 with the preference of the rules | C1, food | 0.3022 | 0.3612 |
| Mean of the 10 differences (go only minus preference) | C1, food | 0.0000 | -0.0071 |
| Within the limit of change 19 | | Yes | Yes |

c FAILS because the score with the go signals only is within the limit of the score with the
preference of the rules at both row counts. At 30 rows the mean difference is below 1e-17 in absolute
value (the label code gives -2.8e-18 and the separate script gives 5.6e-18, from a different order of
the sums).

### Hypothesis d, HOLDS

| Prediction | Condition | 30 rows | 40 rows | Holds |
|---|---|---|---|---|
| d1. Mean absolute advantage over updates 181 to 200 below 0.25 | C1, random | 0.02954 | 0.02903 | Yes at both |
| d2. Mean score at 40, 60, 80, 100 minus mean score at 180, 200, at least the limit | C1, random | 0.05426 | 0.11104 | Yes at both |

d HOLDS because d1 and d2 hold. At 30 rows the difference of d2 is 0.05426 and the limit is 0.05056.
d3 is b3 (1.0 at both row counts).

The 10 values of d2 at 30 rows are 0.19444, 0.02778, -0.03704, 0.05556, 0.10370, 0.10926, -0.01667,
0.13148, 0.02778 and -0.05370. 3 of the 10 values are below 0. At 40 rows each of the 10 values is
above 0.04 (lowest 0.04896).

Score of C1 with the random chooser at each checkpoint:

| Checkpoint | 0 | 1 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 rows | 0.5822 | 0.7563 | 0.9963 | 0.8407 | 0.9207 | 0.8748 | 0.6978 | 0.7763 | 0.7674 | 0.8089 | 0.8163 | 0.7422 |
| 40 rows | 0.5717 | 0.7875 | 0.9996 | 0.9550 | 0.9237 | 0.9192 | 0.8196 | 0.8029 | 0.70125 | 0.7450 | 0.8262 | 0.7604 |

## The limit of change 19 at each place

The limit is the larger of 0.05 and 2 standard errors. The standard error is the sample standard
deviation (divisor 9) of the 10 values of the data sets, divided by the square root of 10, computed
apart at each row count.

| Place | 30 rows: standard error, limit | 40 rows: standard error, limit |
|---|---|---|
| a1 part 1 (share at update 1, C1, food) | 0.0, 0.05 | 0.0, 0.05 |
| c (go only minus preference at 200, C1, food) | 0.002922, 0.05 | 0.001528, 0.05 |
| d2 (early mean minus late mean, C1, random) | 0.025279, 0.050557 | 0.017434, 0.05 |

## Rows of the table for step P2

| Row that applies | Place | Meaning in the rules | For step P2, in the words of the rules |
|---|---|---|---|
| a HOLDS and b FAILS | 2 | The turn of the sign of the go target explains the low score with the food chooser, and it also comes with new rows at each update | Do not build with the variant as it is. The choice is to build with the plain value head, or the author brings a repair of the loss to the owner. The advice rests mostly on the result of step P1 with the random chooser, because a explains the food chooser only |
| d HOLDS and a HOLDS | None | Two results: the turn of the sign with the food chooser, and a critic that fits the training rows with the random chooser | No claim of one cause. Each result is read by its own row |

The rows "c HOLDS", "b HOLDS", "b PARTLY", "a PARTLY and b FAILS" and "a FAILS" do not apply.

The advice of the first place is place 2: **Do not build with the variant.** The next step of design P
is the owner's decision. This document gives no advice beyond the table of the rules.

### The reason that the rules give for the advice

The advice follows from the rules as written, and no label or place changes. The sentence of the
rules "The advice rests mostly on the result of step P1 with the random chooser" does not hold in this
run, because C2 with the random chooser scores 1.0 at both row counts. b FAILS comes from the food
chooser only, and b3 holds.

| Chooser | Copy path | C1 at 200 (30 / 40) | C2 at 200 (30 / 40) | Label | In this document |
|---|---|---|---|---|---|
| food | Yes, each row | 0.3022 / 0.3612 | 0.5948 / 0.3458 | a, b2 | Yes |
| random | No | 0.7422 / 0.7604 | 1.0 / 1.0 | b3, d | Yes, as b3 and d3 |
| food_80 | Yes, 80 percent of rows | 0.5267 / 0.5338 | 1.0 / 1.0 | none | Yes, no-label table |
| slot_0 | No | 0.6259 / 0.5725 | 0.4578 / 0.4079 | none | Yes, no-label table |

Facts from the mean rows:

- **Fact 1.** With the random chooser, new rows raise the score from 0.74 and 0.76 to 1.0 at both row counts. The
   mean absolute advantage over updates 181 to 200 is 0.0295 and 0.0290 in C1, and 0.4988 and 0.5001
   in C2. So in C2 the critic does not fit the rows, and the sign keeps the information of the outcome.
   With d HOLDS, this check finds the cause of the low random score of step P1 in the repeat of rows.
- **Fact 2.** The rules row "a HOLDS and b FAILS" says that the advice "rests mostly on the result of step P1
   with the random chooser". By fact 1, the random chooser no longer supports the advice. It supports
   the opposite reading for that chooser.
- **Fact 3.** The label b FAILS comes from the food chooser only. b3 holds. The rule "b PARTLY" covers the
   opposite pattern (food repaired, random not). The pattern of this run (random repaired, food not)
   has no row of its own, and the rules give it FAILS.
- **Fact 4.** With the food chooser each outcome is 1. In C2 the share of negative advantage over updates 181 to
   200 is 0.5086 and 0.5088, and the score falls to 0.5948 and 0.3458. A high score there would be a
   copy of the choice. The fall shows that the variant loses a choice that it held, when each outcome
   is the same. This part of the advice has support.
- **Fact 6.** slot_0 has no copy path and no label. In C2 it scores 0.4578 and 0.4079, below chance, and the
   plain value head scores 0.9415 and 0.9554 on the same rows in C4. In C2 its mean absolute advantage
   over updates 181 to 200 is 0.1078 and 0.1061. This is a failure to learn with new rows that the
   check does not explain.

The facts are those of the independent review, with its numbers and its numbering. The owner asked for facts 1 to 4 and 6. Fact 5 (food_80 in C2) is not in this list.

The share "80 percent of rows" of food_80 is exact for the data sets of C1 (24 of 30 and 32 of 40
rows). In a new row set of C2 it is 19 of 24 rows and 26 of 32 rows, because the code rounds 0.8 times
the count of rows.

## Measurement 8

Measurement 8 is the score at checkpoint 1 minus the score at checkpoint 0, for each case (change 10).
It is not a control and gives no VOID. The value of each of the 300 cases is in labels.json, key
`measurement_8`. The table gives, for each condition and chooser, the mean, the lowest and the highest
of its 20 cases (10 data sets at 30 rows and 10 at 40 rows), computed from the mean rows. The values
in labels.json agree with the mean rows (largest difference 5.6e-17).

| Condition | Chooser | Mean | Lowest | Highest |
|---|---|---|---|---|
| C1 | random | 0.1950 | 0.1037 | 0.2500 |
| C1 | slot_0 | 0.1554 | 0.0741 | 0.2167 |
| C1 | food | 0.2190 | 0.1333 | 0.2815 |
| C1 | food_80 | 0.2101 | 0.1333 | 0.2593 |
| C2 | random | 0.1937 | 0.0815 | 0.2370 |
| C2 | slot_0 | 0.1575 | 0.0741 | 0.2148 |
| C2 | food | 0.1961 | 0.0889 | 0.2444 |
| C2 | food_80 | 0.1959 | 0.0741 | 0.2375 |
| C3a | food | 0.2190 | 0.1333 | 0.2815 |
| C3b | food | 0.2190 | 0.1333 | 0.2815 |
| C3c | food | 0.2190 | 0.1333 | 0.2815 |
| C4 | random | 0.1019 | 0.0708 | 0.1333 |
| C4 | slot_0 | 0.0975 | 0.0625 | 0.1407 |
| C4 | food | 0.1229 | 0.0963 | 0.1704 |
| C4 | food_80 | 0.1168 | 0.0917 | 0.1704 |

C3a, C3b and C3c have the same values as C1 with the food chooser. At the first update the untrained
critic is a copy of the head at its initial weights, so it gives the same advantage as the learning
critic.

## slot_0 and food_80, with no label

These choosers are measured and have no label and no row in the table for step P2 (change 20). The
question why the variant is below 0.80 is not answered for slot_0 and food_80.

| Condition | Chooser | Score at 200, 30 rows | Score at 200, 40 rows |
|---|---|---|---|
| C1 | slot_0 | 0.6259 | 0.5725 |
| C1 | food_80 | 0.5267 | 0.5338 |
| C2 | slot_0 | 0.4578 | 0.4079 |
| C2 | food_80 | 1.0 | 1.0 |
| C4 | slot_0 | 0.9415 | 0.9554 |
| C4 | food_80 | 0.9830 | 0.9929 |

## Scores at 200 of each condition with the food and the random chooser

| Condition | Chooser | 30 rows | 40 rows |
|---|---|---|---|
| C1 | food | 0.3022 | 0.3612 |
| C1 | random | 0.7422 | 0.7604 |
| C2 | food | 0.5948 | 0.3458 |
| C2 | random | 1.0 | 1.0 |
| C3a | food | 1.0 | 1.0 |
| C3b | food | 1.0 | 1.0 |
| C3c | food | 0.7452 | 0.8117 |
| C4 | food | 0.9985 | 0.9971 |
| C4 | random | 0.9037 | 0.8967 |

## What these results do not show

1. **A high score with the food chooser can be a copy of the choice.** With the food chooser the chosen
   slot is the slot of the food in each row. A head can score high with no learning from the outcome.
   This applies to the scores of a2 (the high early scores), a3 (C3a), C3b, C3c, b2, b4 and c1. The
   evidence of a2 is the later fall, not the early high score. The score of 1.0 of C3a and C3b shows
   only that the go path can hold the choice when no target turns.
2. **C3a, C3b and C3c are not a repair.** A critic that never informs the advantage is not a proposal
   for the agent.
3. **The row counts of the FAILS rules are not the same for each hypothesis.** By owner decision 3 of
   the label code, b and d get FAILS when a prediction fails at one row count only. a, c and reading 2
   get NOT DECIDED in the same kind of case, by the owner decisions after part 3 and by necessary
   change 3 of part 3. In this run b fails at both row counts (b1 and b2 each fail at 30 and at 40
   rows), and d1 holds at both row counts. So the different rule did not change a label here.
4. **Nothing here is a statement on the broadcast or on a saved run.** The check uses built data with
   one signal strength, offline. The live agent has a shaped reward, a return over a rollout, a working
   memory and a relay. A cause found here is a cause in the offline variant. No saved run was read.
5. **C2 is not the amount of data of a run.** C2 gives the head 200 new row sets. It tests a cause. It
   does not show what a head learns from 30 or 40 episodes.
6. **One manipulation each.** A HOLDS label is not a proof of the cause. C3 is the one ablation. The
   results are for initial seeds 0, 1 and 2 and for 10 data sets.
7. **d2 at 30 rows is near its limit.** The difference is 0.05426 and the limit is 0.05056.

## Changes after the independent review

Date: 2026-11-22. The independent review of commit `81ae71f` is on the review issue of this result. Its
verdict is ACCEPT WITH CHANGES. The owner accepted its 8 necessary changes and its 5 optional changes.
No label, no place of advice, no rule, no code and no row file changed. labels.json is new in the
commit. It is the output of the label code on the same rows and was not computed again.

| Change | Where in this document |
|---|---|
| Necessary 1. The advice | New section "The reason that the rules give for the advice", with the table of the choosers and facts 1 to 4 and 6 of the review |
| Necessary 2. The count of a2 | The paragraph below the table of the trainings of C1 with the food chooser. Checkpoint 0, 150 of 150 from checkpoint 20 on, first update 28 to 31 and 29 to 32, the fall about 90 updates after the first turn |
| Necessary 3. Reading 2 | The paragraph below the table of reading 2 (copy of the choice, change 12), and b4 in item 1 of "What these results do not show" |
| Necessary 4. Measurement 8 | New section "Measurement 8" with the mean, the lowest and the highest value for each condition and chooser. labels.json is committed and named at the start |
| Necessary 5. Change 7 | The paragraph on the brake target below the table of hypothesis a |
| Necessary 6. C3c | The paragraph on C3c below the table of hypothesis a |
| Necessary 7. Line 9 | The first paragraph now says first that b (reading 1) and c FAIL |
| Necessary 8. V4 and the error output | The row of V1 in the table of controls |
| Optional 1. Title | "why it scores below 0.80 on built data" |
| Optional 2. b1 | The paragraph below the table of hypothesis b |
| Optional 3. d2 | The 10 values of d2 at 30 rows below the table of hypothesis d, 3 of them below 0 |
| Optional 4. 0.70125 | The table of C1 with the random chooser, 40 rows, checkpoint 140 |
| Optional 5. Shares of C3b and C3c | The table of shares below the paragraph on the brake target |

Each number added by these changes was computed again from the two row files before the change. Two
numbers of the review differ from the rows at the last digit. The highest value of measurement 8 for
C1 with the food chooser is 0.2815 (the review gives 0.282), and the largest difference between a
mean row and the mean of its training rows is 3.3e-16 (the review gives 0.0). This document uses the
values of the rows.
