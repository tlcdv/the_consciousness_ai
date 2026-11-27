# Check 2b of step P2 of design P: three learning rules of the choice head on built data

Date: 2026-11-27. Branch `workspace-action-design`. Rules: `docs/decisions/2026_11_22_workspace_choice_mode.md`,
sections "Owner decision before work 4, and the rules of check 2b", "Changes after the independent review
of check 2b" and "Owner decision on the open choices of the code of check 2b". The rules, the tests and
the code were pushed before the run, in this order. The rules are at `1200e91`, the tests at `28a8a56`, the
code at `6658a33`, and the owner decision on the open choices at `d9e0b0c`.

## Verdict: FAILED for both new rules. No rule passes, so the learning rule of work 4 stays `chosen`

No control failed. Neither `both` nor `hopeful` passes the rule that chooses the learning rule. Each of
them fails condition 2 after 40 and after 60 episodes. Its `greedy_food` is not higher than that of
`chosen` by the limit of change 19. By the table of the rules, "No rule passes" gives `chosen`, the present
rule, for work 4. The rule goes to work 4 only after the owner's yes.

`both` had a higher `greedy_food` than `chosen` after 60 episodes in each of the 10 data sets, by 0.0167
to 0.0617, with a mean of 0.0353. The limit was 0.05, the floor of change 19, because 2 standard errors
were 0.0090. After 40 episodes the mean difference was 0.0163, below 2 standard errors (0.0176), and 3 of
the 10 differences were below 0. So `both` fails condition 2 after 40 episodes also with no floor. This is
reported only. It does not change the result.

The numbers below come from `docs/results/data/workspace_choice_rule_2026_11/labels.json` and
`mean_rows.csv` in the same folder, unless a row names another file. I computed the main numbers a second
time from `mean_rows.csv` with separate code, and they agree.

## Controls

| Control | Expected | Measured | State |
|---|---|---|---|
| K1. Rows | 420 mean rows and 6300 training rows | 420 and 6300 (row counts of `mean_rows.csv` and `training_rows.csv`) | Holds |
| K2. Exploration (N2), each rule apart | Each recorded count of exploration draws equals the count drawn again | 0 mismatches in 900 trainings. Share of exploration draws 0.25656 in each rule and pass | Holds |
| K3. New rows | No row of an episode of a training equals a row of the data set of 40 rows | The job stops on a mismatch. It did not stop | Holds |
| K5. Outcomes | Each outcome is the outcome of its choice, and 1 in the second pass | The job stops on a mismatch. It did not stop | Holds |
| K6 with R4 | Each training row of `chosen` equals its row of part B of check 2 in the score and the 7 measures, exactly | 2100 rows compared of 2100 expected, 0 mismatches, 0 rows with no partner. The committed rows are `docs/results/data/workspace_choice_mode_2026_11/training_rows.csv` | Holds |

## The job

| Item | Value |
|---|---|
| Before the run | No training, probe, test or render job ran. Available memory 10478 MB (`\Memory\Available MBytes`), free disk 58.6 GB |
| Timing (5 trainings of each rule, first pass, data set 0) | `chosen` 0.261 s, `both` 0.274 s, `hopeful` 0.254 s for each training. Estimate 3.95 minutes |
| Command | `python -m scripts.analysis.check_choice_head --choice-rule-check docs/results/data/workspace_choice_rule_2026_11`, then `--choice-rule-labels` on the same folder |
| Time | 239.4 s, alone, one CPU thread |
| End | Verified by K1 from the rows, not by the exit code |

## The result of each rule

All numbers are means over the 10 data sets of the first pass, unless the row says otherwise. D is the
paired difference, first pass minus second pass. The limit is the limit of change 19 of the 10 paired
differences.

| Measure | `chosen` | `both` | `hopeful` |
|---|---|---|---|
| Score after 40 episodes | 0.8771 | 0.8857 | 0.8788 |
| Score after 60 episodes | 0.9050 | 0.9195 | 0.8997 |
| D of the score after 40 (limit) | 0.3714 (0.05) | 0.3903 (0.05) | 0.4173 (0.0546) |
| D of the score after 60 (limit) | 0.4006 (0.05) | 0.4359 (0.05) | 0.4221 (0.05) |
| `greedy_food` after 40 episodes | 0.7562 | 0.7725 | 0.7553 |
| `greedy_food` after 60 episodes | 0.7767 | 0.8120 | 0.7753 |
| `greedy_food`, rule minus `chosen`, after 40 (limit) | | 0.0163 (0.05) | -0.0008 (0.05) |
| `greedy_food`, rule minus `chosen`, after 60 (limit) | | 0.0353 (0.05) | -0.0013 (0.05) |
| D of `greedy_food` after 60 (limit) | 0.2725 (0.05) | 0.3162 (0.05) | 0.2917 (0.05) |

| Condition | `both` | `hopeful` |
|---|---|---|
| 1. The score holds by N11 (0.80 or more, and D at the limit, after 40 and after 60) | Holds | Holds |
| 2. `greedy_food` above `chosen` by the limit, after 40 and after 60 | **Fails** at 40 and at 60 | **Fails** at 40 and at 60 |
| 3. D of `greedy_food` at the limit after 60 | Holds | Holds |
| The rule passes | **No** | **No** |

The tie rule of R1 is not used, because no rule passes.

The 10 paired differences of `greedy_food` after 60 episodes, `both` minus `chosen`, in the order of the
data seeds 0 to 9, are 0.0400, 0.0483, 0.0617, 0.0267, 0.0333, 0.0233, 0.0500, 0.0267, 0.0267 and 0.0167.
Their standard error is 0.0045. Reported only.

The scores and `greedy_food` of `chosen` equal those of part B of check 2 (score 0.8771 and 0.9050, and the
review of the result of check 2 gave `greedy_food` 0.756 and 0.777). K6 shows that the rows are the same.

## The second pass of each rule (R6 and O2)

Each outcome is 1 in the second pass. Reported only, with no part in the choice.

| Rule | `greedy_food` after 40 | 0.5 minus it (limit) | `greedy_food` after 60 | 0.5 minus it (limit) | Mark of O2 |
|---|---|---|---|---|---|
| `chosen` | 0.4970 | 0.0030 (0.05) | 0.5042 | -0.0042 (0.05) | |
| `both` | 0.5032 | -0.0032 (0.05) | 0.4958 | 0.0042 (0.05) | Not marked after 40 or after 60 |
| `hopeful` | 0.4688 | 0.0312 (0.05) | 0.4837 | 0.0163 (0.05) | |

The second pass score after 60 episodes was 0.5044 for `chosen`, 0.4836 for `both` and 0.4776 for
`hopeful`. R6 expected the D parts to hold whenever the score part and condition 2 hold. Condition 2 did
not hold for either rule, so this check of R6 did not occur. Each D part held.

## Curves, reported only

Means over the 10 data sets, first pass.

| Episodes | 0 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| `chosen`, score | 0.5177 | 0.9063 | 0.9217 | 0.8942 | 0.8771 | 0.8977 | 0.9050 |
| `both`, score | 0.5177 | 0.9235 | 0.9188 | 0.8979 | 0.8857 | 0.9141 | 0.9195 |
| `hopeful`, score | 0.5177 | 0.8053 | 0.8800 | 0.8731 | 0.8788 | 0.8908 | 0.8997 |
| `chosen`, `greedy_food` | 0.5108 | 0.7212 | 0.7437 | 0.7563 | 0.7562 | 0.7672 | 0.7767 |
| `both`, `greedy_food` | 0.5108 | 0.7405 | 0.7517 | 0.7777 | 0.7725 | 0.8027 | 0.8120 |
| `hopeful`, `greedy_food` | 0.5108 | 0.6752 | 0.7280 | 0.7470 | 0.7553 | 0.7663 | 0.7753 |
| `chosen`, `share_food` of the window | | 0.6087 | 0.6667 | 0.6627 | 0.7053 | 0.6847 | 0.6927 |
| `both`, `share_food` of the window | | 0.6120 | 0.6780 | 0.6813 | 0.7067 | 0.7040 | 0.7120 |
| `hopeful`, `share_food` of the window | | 0.5687 | 0.6460 | 0.6593 | 0.6840 | 0.6793 | 0.6893 |

At 0 episodes the three rules have the same score and `greedy_food`. The bias of `hopeful` adds the same
1 to both outputs, so it changes the preference by 2e-6 or less (C2 with R5). The score and `greedy_food` at
0 episodes are equal in each data set.

## The value gate of step P1, reported only (O5)

Read from `docs/results/data/workspace_choice_head_2026_11/count_table.json` (`share_food_slot_chosen`).
The lowest share of the episodes in which the value gate took the slot of the food, in the six counted
units, is 0.897 (26 of 29, `rv_switch_s703` before). The three units after the food switch had 0.742
(`rv_switch_s701` after), 0.839 (`rv_switch_s702` after) and 0.839 (`rv_switch_s703` after). These are a
share of the episodes of the gate, and `greedy_food` is a share of the score rows of a head. They have no
part in the choice.

## The learning rule for work 4

**`chosen`**, the present rule, by the row "No rule passes" of the choosing rule. The weakness that the
owner named before check 2b stays. After 60 episodes the larger output of the head takes the slot of the
food in 0.7767 of the score rows. The rules of step P3 must use this. Work 4 starts only after the owner's
yes. Change R2 (the refusal of the rooms with two foods) applies only when work 4 uses `both`, so it does
not apply.

## Limits

1. Built data, with the signal in each row, independent rows and no miss of the food.
2. The setting of part B only, with the share 0.25 and 5 updates for each choice. A rule is not tested in
   part A.
3. The limit of change 19 has a floor of 0.05. For `both` after 60 episodes the floor, not the standard
   error, decides condition 2. After 40 episodes the mean difference of `both` (0.0163) is below 2 standard
   errors (0.0176), so the result does not depend on the floor.
4. 10 data sets, 3 initial seeds and 5 repeats on one machine. The seed of a head is 100 x the data seed
   plus the initial seed.
5. A pass or a fail here does not show what the head learns from the broadcast. That is the question of
   step P3.
6. The owner decision to test a new learning rule was made after a look at the result of check 2 and at the
   review of it, and the two new rules were chosen then.
7. K6 compares the score and the window measures of 10 choices at 7 checkpoints. It does not compare each
   choice or each output.

## What I did not verify

- Why `both` gains about 0.035 of `greedy_food` after 60 episodes and not more. I did not run an ablation.
- Whether a sum in place of the mean in the loss of `both` changes any number (row 1 of the open choices).
  The owner kept the mean. The independent review ran the first pass of `both` with the sum. 0 of 1050 rows
  differed from the mean form.
- The result for a live run. No wiring exists.

## Files

| File | Content |
|---|---|
| `docs/results/data/workspace_choice_rule_2026_11/mean_rows.csv` | 420 mean rows |
| `docs/results/data/workspace_choice_rule_2026_11/training_rows.csv` | 6300 training rows |
| `docs/results/data/workspace_choice_rule_2026_11/labels.json` | The controls, the choosing rule, the second pass, the curves and the shares of the value gate |

## Changes after the independent review

Date: 2026-11-27. The independent review of commit `7e0b0ac` is on the review issue of this result. Its
verdict is ACCEPT WITH CHANGES, and the result does not change. The owner accepted the necessary change N1
and the optional changes O-a, O-b and O-c. Each change is in this document.

| Change | Where |
|---|---|
| N1. The floor at 40 episodes | The second paragraph of "Verdict" and limit 3. The mean difference after 40 episodes (0.0163), its 2 standard errors (0.0176) and the 3 differences below 0 were computed again from `mean_rows.csv` and agree with the review |
| O-a. The preference of `hopeful` at the start | The last paragraph of "Curves, reported only" |
| O-b. What K6 compares | Limit 7 |
| O-c. The sum in the loss of `both` | The item about row 1 in "What I did not verify" |

For O-c, the review ran the first pass of `both` again with `nn.MSELoss(reduction="sum")` in place of the
mean, and 0 of 1050 rows differed from the mean form in the score, `greedy_food` and `share_food`. These
numbers are on the review issue. They are not in the committed rows.
