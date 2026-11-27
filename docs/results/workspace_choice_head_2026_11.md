# VOID. Step P1 of design P: the offline choice head

Date: 2026-11-16. Rules: `docs/decisions/2026_11_16_workspace_choice_head.md`, with its dated notes.
Each rule was pushed before the code that it governs and before each count was read. Script:
`scripts/analysis/check_choice_head.py`. Outputs: `docs/results/data/workspace_choice_head_2026_11/`.

## Verdict: VOID

The check is VOID. The cause is "the recorded feedback cannot answer the question". Each of the 6
counted units is VOID by the count rule of the recorded feedback, and one VOID counted unit makes the
verdict VOID. The owner stopped step P1 on 2026-11-16 after the count table.

This document was corrected on 2026-11-16. Its first tables divided the food switch runs at episode
30, and the food moves at episode 29 in those runs. With that error 4 of the 6 counted units were
VOID. The verdict is VOID in both cases. The tables with the error are kept below for the record.

VOID is not a result for design P and not a result against design P. The question of step P1 was
"can a head learn which candidate gives the food, from the outcome of the gate's choice". The saved
runs cannot answer it, because the gate chose the slot of the food in 89.7 to 97.5 percent of the
episodes of the counted units. The outcome is then 1 in almost each episode and tells a head little.

No head was trained on a saved vector. No saved vector was read for step P1. The count table reads
`value_gate.json` only. The heads were trained on built data only.

No indicator moves. Nothing here is a claim about consciousness.

## The count rule, written before a count was read

A slot is informative in a training set when the episodes of that set in which the gate chose the slot
have 2 or more with outcome 1 and 2 or more with outcome 0. A unit is VOID for the recorded feedback
when one or more of its five training sets has no informative slot. The rule and its reasons are in
the rules file, section "The count rule of the recorded feedback". The rule was not changed after the
table was read. The code and its tests were pushed in commit `184a007` before a count was read.

## The correction of the units

The owner decided on 2026-11-16 to make the count tables again with the correct division. The three
food switch runs were made on 2026-10-16 with the room of commit `ea3eabd`, which counted one episode
for each reset. The training script resets the room once before its first episode, so the option
`--dark-room-food-switch 30` moved the food at recorded episode 29. Commit `961dc1c` of 2026-10-17
repaired the counter, after the three runs. `docs/results/reversal_2026_10.md` records the defect. The
rules of step P1 took episode 30 and now have a dated correction. A unit before the switch is episodes
0 to 28, and a unit after the switch is episodes 29 to 59. The count rule did not change. A test for
the division was pushed in commit `490d00a` before the counts were read again.

This correction came after the first tables were read, so it is not blind. The only change is the
episode of the division.

### Units whose label changed

| Unit | With the division at 30 | With the division at 29 |
|---|---|---|
| `rv_switch_s702` before | Not VOID (smallest count 2) | VOID (smallest count 0) |
| `rv_switch_s703` before | Not VOID (smallest count 2) | VOID (smallest count 1) |
| `rv_switch_s702` after, reported only | VOID (smallest count 1) | Not VOID (smallest count 2) |
| `rv_switch_s703` after, reported only | VOID (smallest count 1) | Not VOID (smallest count 2) |

`rv_switch_s701` before stays VOID, and its smallest count went from 1 to 0. `rv_switch_s701` after
stays Not VOID, and its smallest count went from 2 to 4. The three value learning units have no switch
and did not change. The nearest chooser of `rv_switch_s703` before changed from food to food_80.

### VOID or not VOID, corrected

Each entry is the count of the rule of one training set. For each slot it takes the smaller of its two
outcome counts, and then the larger of the two slots. A set needs 2 or more. The value in brackets is
"outcome 1, outcome 0" for slot 0 and then for slot 1.

| Unit | Set 0 | Set 1 | Set 2 | Set 3 | Set 4 | Smallest | State |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 1 (16,1 / 14,1) | 2 (16,2 / 13,1) | 2 (16,2 / 13,1) | 1 (16,1 / 15,0) | 2 (16,2 / 13,1) | 1 | VOID |
| `vg_learn_s602` | 2 (15,2 / 14,1) | 3 (15,3 / 13,1) | 3 (15,3 / 13,1) | 1 (15,1 / 15,1) | 3 (16,3 / 13,0) | 1 | VOID |
| `vg_learn_s603` | 0 (16,0 / 16,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 0 | VOID |
| `rv_switch_s701` before | 1 (11,1 / 11,0) | 1 (11,1 / 11,0) | 1 (11,1 / 11,0) | 1 (11,1 / 11,0) | 0 (12,0 / 12,0) | 0 | VOID |
| `rv_switch_s702` before | 2 (10,2 / 11,0) | 2 (10,2 / 11,0) | 2 (10,2 / 11,0) | 2 (10,2 / 11,0) | 0 (12,0 / 12,0) | 0 | VOID |
| `rv_switch_s703` before | 2 (11,2 / 10,0) | 3 (11,3 / 9,0) | 3 (11,3 / 9,0) | 3 (11,3 / 9,0) | 1 (12,1 / 11,0) | 1 | VOID |
| `rv_switch_s701` after | 4 (10,4 / 8,2) | 6 (9,6 / 8,2) | 4 (9,4 / 10,2) | 6 (8,6 / 8,3) | 4 (8,4 / 10,3) | 4 | Not VOID, reported only |
| `rv_switch_s702` after | 3 (11,3 / 9,1) | 4 (11,4 / 9,1) | 3 (12,3 / 10,0) | 4 (11,4 / 9,1) | 2 (11,2 / 11,1) | 2 | Not VOID, reported only |
| `rv_switch_s703` after | 3 (11,3 / 9,1) | 4 (11,4 / 9,1) | 3 (12,3 / 10,0) | 4 (11,4 / 9,1) | 2 (11,2 / 11,1) | 2 | Not VOID, reported only |

A training set has 32 episodes for the value learning units, 23 or 24 for the units before the switch
and 24 or 25 for the units after it.

### The count table of each unit, corrected

Each cell is a count of episodes. "Food" means that the chosen slot is the slot of the food. The two
cells with outcome 1 and a slot that is not the slot of the food are 0 in each unit. A unit before the
switch has 14 episodes with target 1 of 29, and a unit after it has 15 of 31.

| Unit | Episodes | Slot 0, outcome 1, food | Slot 0, outcome 0, food | Slot 0, outcome 0, not food | Slot 1, outcome 1, food | Slot 1, outcome 0, food | Slot 1, outcome 0, not food |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 40 | 20 | 0 | 2 | 17 | 1 | 0 |
| `vg_learn_s602` | 40 | 19 | 0 | 3 | 17 | 0 | 1 |
| `vg_learn_s603` | 40 | 20 | 0 | 1 | 19 | 0 | 0 |
| `rv_switch_s701` before | 29 | 14 | 0 | 1 | 14 | 0 | 0 |
| `rv_switch_s702` before | 29 | 13 | 1 | 1 | 14 | 0 | 0 |
| `rv_switch_s703` before | 29 | 14 | 0 | 3 | 12 | 0 | 0 |
| `rv_switch_s701` after | 31 | 11 | 1 | 5 | 11 | 0 | 3 |
| `rv_switch_s702` after | 31 | 14 | 0 | 4 | 12 | 0 | 1 |
| `rv_switch_s703` after | 31 | 14 | 0 | 4 | 12 | 0 | 1 |

### Shares, corrected, a description only

| Unit | Slot of the food chosen | Slot 0 chosen | Outcome 1 | Nearest chooser of the test on built data |
|---|---|---|---|---|
| `vg_learn_s601` | 0.950 | 0.550 | 0.925 | food |
| `vg_learn_s602` | 0.900 | 0.550 | 0.900 | food |
| `vg_learn_s603` | 0.975 | 0.525 | 0.975 | food |
| `rv_switch_s701` before | 0.966 | 0.517 | 0.966 | food |
| `rv_switch_s702` before | 0.966 | 0.517 | 0.931 | food |
| `rv_switch_s703` before | 0.897 | 0.586 | 0.897 | food_80 |
| `rv_switch_s701` after | 0.742 | 0.548 | 0.710 | food_80 |
| `rv_switch_s702` after | 0.839 | 0.581 | 0.839 | food_80 |
| `rv_switch_s703` after | 0.839 | 0.581 | 0.839 | food_80 |

The nearest chooser is the chooser of the test on built data with the smallest distance to the pair of
shares of the unit. The distance rule was fixed in a dated note before a count was read. The gate of
the saved runs is not one of the four choosers. This column has no part in VOID.

## Tables with the error of one episode, kept for the record

The three sections that follow divide the food switch runs at episode 30. Their rows for the six food
switch units have the error. Their rows for the three value learning units are correct. The file of
these tables is `docs/results/data/workspace_choice_head_2026_11/count_table_division_at_30.json`.

### VOID or not VOID, with the error

Each entry is the count of the rule of one training set. For each slot it takes the smaller of its two
outcome counts, and then the larger of the two slots. A set needs 2 or more. The value in brackets is
"outcome 1, outcome 0" for slot 0 and then for slot 1.

| Unit | Set 0 | Set 1 | Set 2 | Set 3 | Set 4 | Smallest | State |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 1 (16,1 / 14,1) | 2 (16,2 / 13,1) | 2 (16,2 / 13,1) | 1 (16,1 / 15,0) | 2 (16,2 / 13,1) | 1 | VOID |
| `vg_learn_s602` | 2 (15,2 / 14,1) | 3 (15,3 / 13,1) | 3 (15,3 / 13,1) | 1 (15,1 / 15,1) | 3 (16,3 / 13,0) | 1 | VOID |
| `vg_learn_s603` | 0 (16,0 / 16,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 1 (16,1 / 15,0) | 0 | VOID |
| `rv_switch_s701` before | 2 (11,2 / 11,0) | 2 (11,2 / 11,0) | 1 (12,1 / 11,0) | 2 (11,2 / 11,0) | 1 (11,1 / 12,0) | 1 | VOID |
| `rv_switch_s702` before | 3 (10,3 / 11,0) | 2 (11,2 / 11,0) | 2 (11,2 / 11,0) | 3 (10,3 / 11,0) | 2 (10,2 / 12,0) | 2 | Not VOID |
| `rv_switch_s703` before | 3 (11,3 / 10,0) | 4 (11,4 / 9,0) | 3 (12,3 / 9,0) | 4 (11,4 / 9,0) | 2 (11,2 / 11,0) | 2 | Not VOID |
| `rv_switch_s701` after | 2 (10,2 / 10,2) | 5 (9,5 / 8,2) | 5 (9,5 / 8,2) | 5 (8,5 / 8,3) | 3 (8,3 / 10,3) | 2 | Not VOID, reported only |
| `rv_switch_s702` after | 2 (11,2 / 10,1) | 3 (11,3 / 9,1) | 3 (12,3 / 9,0) | 3 (11,3 / 9,1) | 1 (11,1 / 11,1) | 1 | VOID, reported only |
| `rv_switch_s703` after | 2 (11,2 / 10,1) | 3 (11,3 / 9,1) | 3 (12,3 / 9,0) | 3 (11,3 / 9,1) | 1 (11,1 / 11,1) | 1 | VOID, reported only |

A training set has 32 episodes for the value learning units and 24 episodes for the other units. The
two units that are not VOID are at the lower limit of the rule. "Before" is episodes 0 to 29 of a food
switch run, and "after" is episodes 30 to 59, which have no part in the verdict.

### The count table of each unit, with the error

Each cell is a count of episodes. "Food" means that the chosen slot is the slot of the food. The two
cells with outcome 1 and a slot that is not the slot of the food are 0 in each unit. Each unit has the
two targets in equal parts.

| Unit | Episodes | Slot 0, outcome 1, food | Slot 0, outcome 0, food | Slot 0, outcome 0, not food | Slot 1, outcome 1, food | Slot 1, outcome 0, food | Slot 1, outcome 0, not food |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 40 | 20 | 0 | 2 | 17 | 1 | 0 |
| `vg_learn_s602` | 40 | 19 | 0 | 3 | 17 | 0 | 1 |
| `vg_learn_s603` | 40 | 20 | 0 | 1 | 19 | 0 | 0 |
| `rv_switch_s701` before | 30 | 14 | 1 | 1 | 14 | 0 | 0 |
| `rv_switch_s702` before | 30 | 13 | 2 | 1 | 14 | 0 | 0 |
| `rv_switch_s703` before | 30 | 14 | 1 | 3 | 12 | 0 | 0 |
| `rv_switch_s701` after | 30 | 11 | 1 | 4 | 11 | 0 | 3 |
| `rv_switch_s702` after | 30 | 14 | 0 | 3 | 12 | 0 | 1 |
| `rv_switch_s703` after | 30 | 14 | 0 | 3 | 12 | 0 | 1 |

### Shares, with the error, a description only

| Unit | Slot of the food chosen | Slot 0 chosen | Outcome 1 | Nearest chooser of the test on built data |
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

The nearest chooser is the chooser of the test on built data with the smallest distance to the pair of
shares of the unit. The distance rule was fixed in a dated note before a count was read. The gate of
the saved runs is not one of the four choosers. This column has no part in VOID.

## The test on built data

This test was the gate of the run, and it uses no saved run. The input has 256 dimensions of normal
noise, with a shift of 1 standard deviation in 64 dimensions for the rows with target 1. A head
receives the chosen slot and its outcome, and never the target. The score is the mean AUC of five
folds, as a mean over initial seeds 0, 1 and 2. Each cell is the mean of 10 data sets. The value in
brackets is the lowest data set for a case with signal, and the highest data set for the null. The
null is the same case with no signal. The file `built_test.csv` has 960 result rows, the count stated
before the test.

The gate PASSED. The plain value head scored 0.904 at 30 rows and 0.897 at 40 rows with the full
signal and the random chooser. The limit was 0.80.

### The plain value head

| Rows | Signal share | random | slot_0 | food | food_80 |
|---|---|---|---|---|---|
| 30 | 1.0 | 0.904 (0.763) | 0.941 (0.904) | 0.999 (0.993) | 0.983 (0.948) |
| 30 | 0.6 | 0.639 (0.385) | 0.698 (0.585) | 0.753 (0.674) | 0.715 (0.644) |
| 30 | 0.0, the null | 0.524 (0.681) | 0.526 (0.607) | 0.484 (0.615) | 0.493 (0.570) |
| 40 | 1.0 | 0.897 (0.754) | 0.955 (0.929) | 0.997 (0.987) | 0.993 (0.975) |
| 40 | 0.6 | 0.700 (0.583) | 0.725 (0.662) | 0.806 (0.758) | 0.772 (0.704) |
| 40 | 0.0, the null | 0.548 (0.650) | 0.535 (0.658) | 0.560 (0.696) | 0.538 (0.625) |

### The go and no-go variant

| Rows | Signal share | random | slot_0 | food | food_80 |
|---|---|---|---|---|---|
| 30 | 1.0 | 0.742 (0.630) | 0.626 (0.430) | 0.302 (0.215) | 0.527 (0.407) |
| 30 | 0.6 | 0.602 (0.437) | 0.519 (0.385) | 0.444 (0.341) | 0.533 (0.474) |
| 30 | 0.0, the null | 0.501 (0.578) | 0.499 (0.600) | 0.521 (0.630) | 0.486 (0.622) |
| 40 | 1.0 | 0.760 (0.604) | 0.573 (0.438) | 0.361 (0.288) | 0.534 (0.462) |
| 40 | 0.6 | 0.608 (0.529) | 0.512 (0.392) | 0.392 (0.337) | 0.495 (0.371) |
| 40 | 0.0, the null | 0.512 (0.600) | 0.509 (0.596) | 0.505 (0.533) | 0.495 (0.554) |

The variant did not reach 0.80 in any case. It is above the null with the random chooser only. No
training value was changed after these numbers.

### Second pass, each outcome set to 1, full signal

| Head | Rows | random | slot_0 | food | food_80 |
|---|---|---|---|---|---|
| The plain value head | 30 | 0.533 | 0.536 | 0.999 | 0.940 |
| The plain value head | 40 | 0.405 | 0.552 | 0.997 | 0.965 |
| The go and no-go variant | 30 | 0.466 | 0.444 | 0.302 | 0.377 |
| The go and no-go variant | 40 | 0.540 | 0.529 | 0.361 | 0.384 |

With the food and food_80 choosers, the plain value head has almost the same score when each outcome
is 1. In those cases its score comes from the choice and not from the outcome. With the random and
slot_0 choosers its score falls to the null when each outcome is 1.

## How the two parts relate

With the corrected units, 5 of the 6 counted units are nearest to the food chooser, and
`rv_switch_s703` before the switch is nearest to food_80. On built data with the food chooser, a score of the
plain value head comes from the choice, and the go and no-go variant scores below 0.5 (0.302 and
0.361). This is a relation between two descriptions. It is not a result on the saved runs, and it is a
second reason, next to the count rule, that a run on these six runs could not answer the question.

## Data integrity check of two runs

With the division at episode 30, the units `rv_switch_s702` after and `rv_switch_s703` after have the
same chosen slot, outcome and target in each of their 30 episodes. With the corrected division their
count tables and their training sets are also equal. The owner asked for a check that the two runs are not copies. The
check compared the records of the runs and the md5 of their files. It read no content of a saved
vector for a score.

| Item | `rv_switch_s702` against `rv_switch_s703` |
|---|---|
| Recorded seed (`config.json`, `session.json`) | 702 and 703 |
| Recorded command (`session.json`, 54 arguments) | 2 arguments differ: the seed and the output folder |
| Start time (`ethics_manifest.json`) | 2026-10-16 22:21:39 UTC and 23:11:10 UTC |
| `vectors.npz`, episodes with equal md5 | 0 of 60 |
| `vectors.npz`, episodes with each array equal | 0 of 60 |
| `frames.npz`, `steps.jsonl`, `weights.pt`, episodes with equal md5 | 0 of 60 for each |
| `broadcast.npy`, `map_binding.json`, `metrics.csv`, `value_gate.json` | md5 differs for each |
| Recorded salience of the two candidates, episodes with equal values | 0 of 60 |
| Chosen band, episodes with the same band | 28 of 30 before the switch, 30 of 30 after |
| Outcome, episodes with the same outcome | 27 of 30 before the switch, 30 of 30 after |

The two runs are not copies. They have different seeds, different saved vectors and different recorded
salience in each episode. The equal slot records after the switch are an agreement of 30 choices and 30
outcomes between two different runs. I did not examine why the choices agree.

## Open points, not examined

1. The go and no-go variant has a preference with the wrong sign on built data with the food chooser.
   The owner said not to examine it now.
2. With the division at episode 30, six episodes had the slot of the food chosen and an outcome of 0.
   The note at the end of this document gives the cause of each one. With the corrected division three
   such episodes stay (1 in `vg_learn_s601`, 1 in `rv_switch_s702` before, 1 in `rv_switch_s701`
   after).
3. Why `rv_switch_s702` and `rv_switch_s703` agree in each choice and outcome after the switch.

## Limits

1. The test on built data shows what the two heads learn from a built signal of one strength. It does
   not show what a head learns from the broadcast.
2. The count rule is a lower limit for "the question can be asked". A unit that is not VOID is not
   shown to be answerable.
3. Controls K1 to K4 of step P0 were not made again for the count table. It checked the count of
   episodes of `value_gate.json`, K6, K7, K10 and the rule of equal salience. No mismatch occurred.
4. The nearest chooser compares two shares only.
5. The part of the script that reads the saved vectors was not written.

## What follows

Step P1 is stopped. The owner decides the next step of design P. The rules of step P1 say that no
verdict of this check stops design P without the owner.

## Commands

```
python -m scripts.analysis.check_choice_head --built-test docs/results/data/workspace_choice_head_2026_11/built_test.csv
python -m scripts.analysis.check_choice_head --count-table runs --out docs/results/data/workspace_choice_head_2026_11/count_table.json
python -m pytest tests/ -q --ignore=tests/test_emergence_governor.py
```

The second command made `count_table.json` with the corrected units. The file
`count_table_division_at_30.json` is the output of the same command before the correction.

The public suite, run alone on 2026-11-16 with the 46 tests of `tests/test_choice_head.py`: 2752
passed, 8 skipped.

## Note of 2026-11-16, after the verdict: the six episodes, and an error of one episode in three units

The owner asked for the cause of the six episodes with the slot of the food chosen and an outcome of
0. I read the step records of these six episodes (`steps.jsonl`) and the gate record. In each one the
gate record has 60 steps, which is the window of the gate (`OUTCOME_STEPS` in
`models/core/value_gate.py`). No step of these episodes has a reward of the room above 0.5. An episode
has 200 steps, so the step limit of the episode is not the cause in any of them.

| Episode | Cause |
|---|---|
| `vg_learn_s601`, episode 2 | The agent did not reach the food in the window. The distance to the light was 89.1 at step 12, 150.8 at step 40 and 159.7 at step 72. The reach is 60 |
| `rv_switch_s702`, episode 23 | The same. The distance to the light was 81.1, 146.1 and 155.8 at the same steps |
| `rv_switch_s701`, episode 29 | The agent reached the light (distance 6.1 at step 40). The room records the decoy as the food in this episode, so the light gave no food |
| `rv_switch_s702`, episode 29 | The same (distance 4.0 at step 40) |
| `rv_switch_s703`, episode 29 | The same (distance 3.8 at step 40) |
| `rv_switch_s701`, episode 38 | The gate chose the band of the decoy, which gives the food after the switch. The agent went to the light (distance 11.7 at step 40). Its smallest distance to the decoy in the window was 93.2 |

**An error in this document and in the rules of step P1.** The step records of the three food switch
runs name the light as the food in episodes 0 to 28 and the decoy as the food from episode 29 on
(`_truth_food`, read at episodes 0, 27, 28, 29, 30, 31, 38 and 59 of each run). The rules took the
switch at episode 30, from the option `--dark-room-food-switch 30`. So episode 29 of each food switch
run was in the wrong unit, with the wrong band of the food. The three value learning units do not
have this error, and they are VOID. So the verdict VOID does not depend on it.

Corrected on 2026-11-16. This note first said that I could not verify why the room counts the switch
one episode early, and that other documents could have the same error. Both statements were out of
date after a read-only check. The cause is the older counter of the room (commit `ea3eabd`), repaired
by commit `961dc1c` after the three runs. The other documents that divide a run at an episode of one of
these flags are correct: the reversal documents and the drive off documents name the defect, and the
competing goals runs agree with their flags. The count tables were made again with the correct
division by the owner's decision. They are in the section "The correction of the units".
