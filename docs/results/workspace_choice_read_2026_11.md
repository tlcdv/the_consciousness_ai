# Step P0 of design P: the read of the broadcast at the choice step

Date of the run: 2026-11-16. Rules: `docs/decisions/2026_11_15_workspace_choice_read.md`, with its
correction of 2026-11-16. Both were pushed before the run. Script: `scripts/analysis/check_choice_read.py`.
Output: `docs/results/data/workspace_choice_read_2026_11/summary.json`.

## Verdict: UNDECIDED

The check did not give HOLDS and did not give FAILED. Of the six counted runs, 4 HOLD, 1 is AT NULL
and 1 is BETWEEN. HOLDS needed 3 of 3 runs in each group. Each group has 2 of 3. By the rules, the table
of the runs is reported and the owner decides.

No indicator moves. Nothing here is a claim about consciousness. The check is a correlation on saved
runs, and it does not show that anything uses the broadcast.

## The runs

Score is the mean AUC of five held-out folds. Null is the 95th percentile of 200 label orders.

| Run | Episodes | Broadcast | Its null | Audio (reference) | Its null | Episode number | Label of the run |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 40 | 0.975 | 0.688 | 1.000 | 0.738 | 0.300 | HOLDS |
| `vg_learn_s602` | 40 | 0.600 | 0.676 | 1.000 | 0.676 | 0.300 | AT NULL |
| `vg_learn_s603` | 40 | 0.800 | 0.682 | 1.000 | 0.726 | 0.300 | HOLDS |
| `rv_switch_s701` | 60 | 0.861 | 0.667 | 1.000 | 0.645 | 0.417 | HOLDS |
| `rv_switch_s702` | 60 | 0.836 | 0.656 | 1.000 | 0.678 | 0.417 | HOLDS |
| `rv_switch_s703` | 60 | 0.722 | 0.661 | 1.000 | 0.673 | 0.417 | BETWEEN |

The score of `vg_learn_s603` is 0.800 exactly, which is the threshold. Its five fold values are 0.750,
0.938, 0.812, 0.500 and 1.000. A fold has 8 held-out episodes in these runs, so one fold value moves in
steps of 0.0625.

## Controls

| Control | State |
|---|---|
| K1 to K4 | Passed in the six counted runs. The script stops a run with VOID on a mismatch, and no counted run is VOID |
| K5, the mean of the null scores | From 0.493 to 0.509 for `broadcast` and from 0.494 to 0.531 for `audio_content`. The range of the rule is 0.40 to 0.60 |
| Reference | The audio vector gives 1.000 in each of the six runs, with each of its five folds at 1.000 |
| Unit tests | `tests/test_choice_read.py`, 35 passed before the run |

## Reported only, with no part in the verdict

**1. The tectum vector.**

| Run | Tectum score | Its null |
|---|---|---|
| `vg_learn_s601` | 0.825 | 0.688 |
| `vg_learn_s602` | 0.425 | 0.626 |
| `vg_learn_s603` | 0.738 | 0.713 |
| `rv_switch_s701` | 0.411 | 0.650 |
| `rv_switch_s702` | 0.389 | 0.678 |
| `rv_switch_s703` | 0.406 | 0.656 |

**2 and 3. Which vector the broadcast is, and the state of the workspace, at the choice step.** The
choice step is step 12 in each episode of the six runs.

| Run | Broadcast equal to the audio vector | Equal to the tectum vector | Equal to neither | Ignited share | Winner vision | Winner audio | No winner |
|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 20 | 19 | 1 | 1.000 | 25 | 14 | 1 |
| `vg_learn_s602` | 25 | 14 | 1 | 1.000 | 30 | 9 | 1 |
| `vg_learn_s603` | 15 | 19 | 6 | 0.975 | 23 | 11 | 6 |
| `rv_switch_s701` | 41 | 18 | 1 | 1.000 | 48 | 11 | 1 |
| `rv_switch_s702` | 40 | 13 | 7 | 0.983 | 42 | 11 | 7 |
| `rv_switch_s703` | 31 | 24 | 5 | 1.000 | 38 | 17 | 5 |

So the broadcast is an exact copy of one of the two vectors in 279 of 300 episodes. It is the audio
vector in more episodes than the audio module wins. This agrees with the default merge, which does not
keep the vector of the winner.

**4. One row for each step from step 0 to the choice step, whole episodes held out.**

| Run | Rows | Broadcast score | Its null |
|---|---|---|---|
| `vg_learn_s601` | 520 | 0.756 | 0.604 |
| `vg_learn_s602` | 520 | 0.552 | 0.570 |
| `vg_learn_s603` | 520 | 0.636 | 0.589 |
| `rv_switch_s701` | 780 | 0.656 | 0.599 |
| `rv_switch_s702` | 780 | 0.670 | 0.570 |
| `rv_switch_s703` | 780 | 0.563 | 0.582 |

**5. The competing goals runs.** Not delivered. The three runs stop at control K3 in episode 0, with
the message "the bands of the gate and of the binding differ". These runs are reported only, so the
verdict does not change. I did not examine the cause.

**6. The penalty of each outer fold.** In `summary.json`. The audio read chose 0.01 in each fold of each
run. The broadcast read chose penalties from 0.01 to 100.

## One table made after the run

The rules did not ask for this table. I made it after I saw the scores, so it is a description and
not a test. File: `docs/results/data/workspace_choice_read_2026_11/source_table.json`, from the option
`--source-table` of the script.

It counts the episodes by the label and by the vector that the broadcast is at the choice step.

| Run | Label | Audio vector | Tectum vector | Neither |
|---|---|---|---|---|
| `vg_learn_s601` | light louder | 15 | 4 | 1 |
| `vg_learn_s601` | decoy louder | 5 | 15 | 0 |
| `vg_learn_s602` | light louder | 14 | 5 | 1 |
| `vg_learn_s602` | decoy louder | 11 | 9 | 0 |
| `vg_learn_s603` | light louder | 7 | 8 | 5 |
| `vg_learn_s603` | decoy louder | 8 | 11 | 1 |
| `rv_switch_s701` | light louder | 22 | 7 | 1 |
| `rv_switch_s701` | decoy louder | 19 | 11 | 0 |
| `rv_switch_s702` | light louder | 19 | 5 | 6 |
| `rv_switch_s702` | decoy louder | 21 | 8 | 1 |
| `rv_switch_s703` | light louder | 16 | 9 | 5 |
| `rv_switch_s703` | decoy louder | 15 | 15 | 0 |

Two facts from this table.

1. The broadcast is the tectum vector in 107 of 300 episodes. In those episodes it is a vector that
   does not hold the label in 4 of 6 runs (table 1 above).
2. The vector that the broadcast is depends on the label in some runs. In `vg_learn_s601` it is the
   audio vector in 15 of 20 episodes with the light louder and in 5 of 20 with the decoy louder. The
   case "neither" has 19 episodes with the light louder and 2 with the decoy louder over the six runs.

So a read of the broadcast can find the label in two ways. One is the content of the audio vector. The
other is which module supplied the vector. The check cannot tell these apart. I did not test which of
the two gives the scores.

## Limits

The nine limits of the rules stand. Three of them act on this result.

1. Limit 2. The total loudness gives the same label. The audio score of 1.000 can come from the total
   loudness alone.
2. Limit 4. A fold has 8 or 12 held-out episodes. One run is exactly at the threshold of 0.80.
3. Limit 6. The two groups use the same room, binder and gate. They are not independent.

One more limit comes from the result. The broadcast is a copy of one module's vector, and the module
changes from episode to episode. A policy that reads the broadcast then receives the content of the
louder tone only in a part of the episodes.

## What follows

The verdict is UNDECIDED, so the rules give no next step. The owner decides. I see three options, in
no order of preference fixed by the rules.

| Option | Content | Cost |
|---|---|---|
| A. Go to step P1 with the default merge | The offline choice head shows directly if a policy can learn the choice from these broadcasts. The read here says the content is present in a part of the episodes | Rules of P1, no new run of the agent |
| B. Test the other merge first | New runs with `top_winner`, then this same check on them. The saved runs cannot answer this | 6 new runs of the agent, with their own rules |
| C. Stop design P with the default merge | Treat 2 of 3 in each group as not enough, and choose among design E, the other merge, and the repair of the path from the map to the content | A new design decision |

A fresh run of this check with a changed threshold on the same six runs is not an option. The rules
forbid it.

## Commands

```
python -m pytest tests/test_choice_read.py -q
python -m scripts.analysis.check_choice_read --runs-root runs --summary docs/results/data/workspace_choice_read_2026_11/summary.json
python -m scripts.analysis.check_choice_read --runs-root runs --source-table docs/results/data/workspace_choice_read_2026_11/source_table.json
```
