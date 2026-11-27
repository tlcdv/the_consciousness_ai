# Check 2 of step P2 of design P: the choice head on built data

Date: 2026-11-22. Rules: `docs/decisions/2026_11_22_workspace_choice_mode.md`, with its sections "Changes
after the independent review" and "Owner decision on the scaling". Code: `models/core/choice_head.py`
and the modes `--choice-mode-check` and `--choice-mode-labels` of `scripts/analysis/check_choice_head.py`,
at commit `43b5717`. Rows: `docs/results/data/workspace_choice_mode_2026_11/mean_rows.csv` (called "the
mean rows" below) and `training_rows.csv` in the same folder (called "the training rows"). The output of
the label code is `labels.json` in the same folder.

No control failed. The check is not VOID, and no label is FAILS. Part A HOLDS and part B HOLDS.

The check trains small heads on built data only. It read no saved run, ran no agent, and changed no file
of the agent. No saved run was read. The only statement on the broadcast is the read of the base control
run in limit 1, from the review issue of the rules. No indicator moves.

## Labels

| Part | Label | What it rests on |
|---|---|---|
| A. New rows at each update, as condition C2 | HOLDS | At 200 updates the score is 1.0000 at 30 rows and 1.0000 at 40 rows, and the mean D is 0.4837 and 0.4883, above its limits of 0.0582 and 0.05 |
| B. The live rule with the 60 rows of a run | HOLDS | The score is 0.8771 after 40 episodes and 0.9050 after 60, and the mean D is 0.3714 and 0.4006, above the limit of 0.05 at both |
| The arm of O2, a store of rows (reported only) | HOLDS by the rule of N11 | The score is 0.9528 after 40 episodes and 0.9668 after 60, and the mean D is 0.3808 and 0.4155 |

The row of "What each result means" for check 2 is "A HOLDS and B HOLDS". The rules give it this meaning.
On built data, the plain value head learns which slot gives the food from the outcomes of its own exploring
choices, with new rows, and with the rows of a run of 40 episodes. For the next step the rules say that
the build (work 4) can start after the owner's yes. The owner decides.

The score is the order of the preferences. A choice follows the larger output, which in part B took the
slot of the food in about 0.75 of the non random choices.

The labels come from the command below, on the two row files.

```
python -m scripts.analysis.check_choice_head --choice-mode-labels docs/results/data/workspace_choice_mode_2026_11
```

A separate script read the two row files with pandas and gave the same score, second pass, D, standard
error and limit for each label in the tables below.

## The job

| Item | Value |
|---|---|
| Command | `python -B -u -m scripts.analysis.check_choice_head --choice-mode-check docs/results/data/workspace_choice_mode_2026_11` |
| Interpreter | `.venv-tcai-py312`, Python 3.12.10, torch 2.14.1, one CPU thread |
| Before the run | No training, probe or test process. Free memory 5.55 GB, free disk 57.10 GB |
| Timing before the run | One training took 0.397 s in part A, 0.258 s in part B and 0.075 s in the arm of O2, as the mean of 5 trainings. Estimate 5.6 minutes |
| Start and end | Started 2026-11-22 07:43:48. The process had ended by 07:49:25, so about 5.6 minutes |
| Error output | Empty |
| Label code | 22 seconds, with the redraw of K2 |

The places where the code makes a choice that the rules leave open were listed before the run, on the
review issue of the rules. None of them can change the label of part A or of part B. They are the order
of the rows of a set (the order that `new_row_set` gives), the chooser argument of `built_case` (the rows
and the targets do not depend on it), the column `count_choices` for K2, blank shares at checkpoint 0,
the row count 40 written for part B and the arm of O2, the name `O2`, the data set of 40 rows for K3 of
part B, the number types of C2 and step P1, the rows 0 to e at update e of the arm of O2, and the name of
this document.

## Controls

| Control | Result | Source |
|---|---|---|
| K1. Rows | Passed. 760 mean rows and 11400 training rows | `wc -l` gives 761 and 11401 lines, each with one header line. The label code checks the counts again |
| K2. Exploration | Passed. 0 mismatches in the 1200 trainings | The label code draws again with `rng.random()` and then `rng.integers(0, 2)` for each choice and compares the count of each training row |
| K3. New rows | Passed | The job checks each new row set of part A and the rows of part B against the data set of their case before the signal, and stops on a mismatch. The job wrote each case |
| K5. Outcomes | Passed | The job checks each outcome before the head receives it, and stops on a mismatch |

The share of exploration draws is 0.250120 in each pass of part A, and 0.256556 in each pass of part B
and of the arm of O2. The two passes and the arm repeat the same draws, as the rules state.

## Part A, HOLDS

Each number is a mean over the 10 data sets of the mean over 3 initial seeds and 5 folds, from the mean
rows. D is the score of the first pass minus the score of the second pass, for each data set.

| Item, at 200 updates | 30 rows | 40 rows |
|---|---|---|
| Score, first pass | 1.0000 | 1.0000 |
| Score, second pass (each outcome 1) | 0.5163 | 0.5117 |
| Mean D | 0.4837 | 0.4883 |
| Standard error of D over the 10 data sets | 0.0291 | 0.0191 |
| Limit of change 19 | 0.0582 | 0.05 |
| 0.80 or more | Yes | Yes |
| D at the limit or above | Yes | Yes |

The 10 values of D are 0.5111, 0.6370, 0.4519, 0.4593, 0.5926, 0.5259, 0.3630, 0.4074, 0.5259 and 0.3630
at 30 rows, and 0.4458, 0.5500, 0.5042, 0.4167, 0.5750, 0.4917, 0.5292, 0.4125, 0.4208 and 0.5375 at 40
rows.

| Checkpoint (updates) | 0 | 1 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| First pass, 30 rows | 0.5385 | 0.6089 | 0.9985 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Second pass, 30 rows | 0.5385 | 0.5415 | 0.5807 | 0.5852 | 0.6059 | 0.5719 | 0.5526 | 0.5504 | 0.5430 | 0.5222 | 0.5126 | 0.5163 |
| First pass, 40 rows | 0.5196 | 0.6042 | 0.9992 | 0.9996 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Second pass, 40 rows | 0.5196 | 0.5279 | 0.5792 | 0.5975 | 0.5821 | 0.5837 | 0.5575 | 0.5479 | 0.5404 | 0.5354 | 0.5225 | 0.5117 |

The first checkpoint at which the score of the first pass reaches 0.80 is 20 updates at both row counts.

## Part B, HOLDS

Part B has one score, on the 40 rows of the data set of 40 rows (N3).

| Item | After 40 episodes | After 60 episodes |
|---|---|---|
| Score, first pass | 0.8771 | 0.9050 |
| Score, second pass | 0.5056 | 0.5044 |
| Mean D | 0.3714 | 0.4006 |
| Standard error of D | 0.0208 | 0.0119 |
| Limit of change 19 | 0.05 | 0.05 |
| 0.80 or more | Yes | Yes |
| D at the limit or above | Yes | Yes |

The 10 values of D are 0.4770, 0.3977, 0.2812, 0.3998, 0.3508, 0.3418, 0.3743, 0.4052, 0.4275 and 0.2590
after 40 episodes, and 0.4395, 0.4293, 0.3973, 0.3175, 0.3718, 0.3968, 0.3822, 0.4082, 0.4197 and 0.4438
after 60.

| Checkpoint (episodes) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| First pass | 0.5177 | 0.9063 | 0.9217 | 0.8942 | 0.8771 | 0.8977 | 0.9050 |
| Second pass | 0.5177 | 0.5742 | 0.5362 | 0.4759 | 0.5056 | 0.5073 | 0.5044 |

The first checkpoint at which the score of the first pass reaches 0.80 is 10 episodes.

## The arm of O2, reported only

The arm has the rows, the seeds, the draws and the two passes of part B, and makes one update on all rows
so far at each choice. By owner decision d it has no row in "What each result means", and a change of the
live learning rule stays a new rule for the owner.

| Item | After 40 episodes | After 60 episodes |
|---|---|---|
| Score, first pass | 0.9528 | 0.9668 |
| Score, second pass | 0.5719 | 0.5513 |
| Mean D | 0.3808 | 0.4155 |
| Standard error of D | 0.0217 | 0.0137 |
| Label by the rule of N11 | HOLDS | |

| Checkpoint (episodes) | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| First pass | 0.5177 | 0.7560 | 0.8897 | 0.9390 | 0.9528 | 0.9597 | 0.9668 |
| Second pass | 0.5177 | 0.5723 | 0.5902 | 0.5909 | 0.5719 | 0.5561 | 0.5513 |

## The choices, reported only

The score reads the order of the preferences on the score rows. A choice reads the sign of the
preference, and an exploration draw ignores it. With the share 0.25, a head with a perfect preference
takes the slot of the food in 0.875 of the choices in expectation (O6).

| Item, first pass | Part A, 30 rows | Part A, 40 rows | Part B | Arm of O2 |
|---|---|---|---|---|
| Share of choices of the slot of the food, over all choices | 0.8597 of 720000 | 0.8611 of 960000 | 0.6701 of 9000 | 0.6521 of 9000 |
| The same, in the last window | 0.8763 | 0.8748 | 0.6927 | 0.6927 |
| Share of choices of slot 0, in the last window | 0.4995 | 0.5011 | 0.6133 | 0.6773 |

The shares over all choices come from the training rows, each window weighted by its count of choices.
The other shares are from the mean rows. In the last window of part B and of the arm, the food slot was
taken in 1039 of 1500 choices in each. The two counts are equal by chance. The share is equal in 65 of the
150 trainings only. In part B the score after 60 episodes is 0.9050, and the share of choices of the slot
of the food over all choices is 0.6701, below the 0.875 of a perfect preference. These shares have no
part in a label. Step P3 scores the behaviour by its own rules (O6).

The share of the non random choices that took the slot of the food is estimated from the mean rows as
(share_food minus 0.5 x share_explored) / (1 minus share_explored), with the means over the 10 data sets of
the two shares in each window. The estimate is a mean of shares over data sets, so it is only near the true
share.

| Window to | Part A, 30 rows | Part A, 40 rows |
|---|---|---|
| 20 updates | 0.834 | 0.853 |
| 40 updates | 0.988 | 0.992 |
| 60 updates | 0.997 | 0.997 |
| 100 updates | 0.998 | 0.997 |
| 200 updates | 1.001 | 0.999 |

| Window to | Part B | Arm of O2 |
|---|---|---|
| 40 episodes | 0.781 | 0.751 |
| 50 episodes | 0.745 | 0.759 |
| 60 episodes | 0.759 | 0.759 |

## What these results do not show

1. **A pass on built data does not show that the head learns from the broadcast.** That is the question
   of step P3. The built rows have the signal in each row, one signal strength, independent rows, and no
   miss of the food. Step P0 is UNDECIDED. In the base control run the broadcast at the choice step was
   the audio vector in 7 of 10 episodes and the tectum vector in 3 of 10.
2. **The scaling was decided after a look at one run.** The owner decided the normalization of the input
   of the head after the read of the base control run, one run of 10 episodes with seed 73. It is an owner
   decision with no review.
3. **N1, N3 and N11 were set after the reviewer ran the second pass** of check 2 with its own code of the
   draws. No one ran the first pass before this run. Each rule was fixed before this run.
4. **The two row counts of part A are not two samples.** They share the initial weights, the draws and
   the noise of the first 24 rows of each new row set (O7).
5. **Part B is one training condition**, read on the 40 rows of one data set for each of the 10 data
   seeds (N3).
6. **The values are fixed.** The share 0.25 and the 5 updates for each choice were set before any number
   and not tuned. A result holds for these values only.
7. **The numbers are not comparable with step P1.** The seeds of the heads change with the data set (O1).
8. **No saved run was read.** The only statement on the broadcast is the read of the base control run in
   limit 1, from the review issue of the rules. `scripts/training/train_rlhf.py` did not run for this
   check.

## What follows

The rules give the row "A HOLDS and B HOLDS". The next work in the rules is work 4, the unit tests of group
B and the wiring, after the owner's yes. The owner decides. Before work 4 the owner decides on the live
rule with this fact. In part B the larger output took the slot of the food in about 0.75 of the non random
choices.

## Changes after the independent review

Date: 2026-11-22. The independent review of commit `07ea393` is on the review issue of this result. Its
verdict is ACCEPT WITH CHANGES, and the labels do not change. The owner accepted R1, R2 and R3 and the
optional changes S2 and S3, and did not take S1. R3 and S3 change the rules file. R1 and R2 change this
document.

| Change | Where |
|---|---|
| R1. The greedy choice | The text after the meaning sentence in "Labels", the estimate tables in "The choices, reported only", and the last sentence of "What follows". Each estimate was computed again from the mean rows and agrees with the review |
| R2. The sentence on the broadcast | The third paragraph and limit 8. The review names the issue by its number, and this document names it as the review issue of the rules |

The review also measured the greedy choice directly, with the code of the check run again on part B and
on the arm. After 60 episodes the larger output was the slot of the food on 0.777 of the score rows in
part B and 0.766 in the arm. These numbers are on the review issue. They are not in the committed rows.
