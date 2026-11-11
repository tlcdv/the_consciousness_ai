# Workspace priority on the tectal map: offline check on saved runs

Date: 2026-11-11. Rules: `docs/decisions/2026_11_11_workspace_priority_check.md`, pushed before the
script ran. Script: `scripts/analysis/check_workspace_priority.py`. Tests:
`tests/test_workspace_priority_check.py`.

## The verdict

**EXCLUDED.** Rule 1 FAILED at 3 of 3 seeds of the food switch group. It held at 3 of 3 seeds of the
value learning group, at seed 601 exactly at the limit. Rule 2 held at 6 of 6 runs, and the rules file
states that a pass of rule 2 is weak evidence. All controls passed in 6 of 6 runs.

No model was trained. No gate ran. No default changed. The mechanism is not written. The check is a
correlation on saved runs and not an ablation.

## The command and its output

```
python -m scripts.analysis.check_workspace_priority --runs-root <runs folder> --summary <runs folder>/workspace_priority_check/summary.json
```

Run at commit `d11910a`. Output, one line for each run, put in a table.

| Run | Episodes | Control A | Ignited at the choice step | Audio winner share q | Changed at weight 1.0 | Share | Rule 1 (0.05 or more) | R squared | Rule 2 (below 0.90) |
|---|---|---|---|---|---|---|---|---|---|
| `vg_learn_s601` | 40 | 40/40 | 1.000 | 0.350 | 2 | 0.050 | holds | 0.2061 | holds |
| `vg_learn_s602` | 40 | 40/40 | 1.000 | 0.225 | 3 | 0.075 | holds | 0.1243 | holds |
| `vg_learn_s603` | 40 | 40/40 | 0.975 | 0.275 | 3 | 0.075 | holds | 0.1551 | holds |
| `rv_switch_s701` | 60 | 60/60 | 1.000 | 0.183 | 2 | 0.033 | FAILED | 0.0985 | holds |
| `rv_switch_s702` | 60 | 60/60 | 0.983 | 0.183 | 2 | 0.033 | FAILED | 0.0983 | holds |
| `rv_switch_s703` | 60 | 60/60 | 1.000 | 0.283 | 2 | 0.033 | FAILED | 0.1615 | holds |

The script printed "Controls A to D passed in 6 of 6 runs" and "VERDICT at weight 1.0: EXCLUDED".
Control E is a test (`test_a_constant_term_raises_and_gives_no_number`), and it passed.

## Report only, with no part in the verdict

Episodes with a changed predicted choice at the two other weights.

| Run | Weight 0.5 | Weight 1.0 | Weight 2.0 |
|---|---|---|---|
| `vg_learn_s601` | 0 | 2 | 13 |
| `vg_learn_s602` | 1 | 3 | 8 |
| `vg_learn_s603` | 2 | 3 | 9 |
| `rv_switch_s701` | 2 | 2 | 3 |
| `rv_switch_s702` | 1 | 2 | 7 |
| `rv_switch_s703` | 0 | 2 | 8 |

The count depends on the weight, as limit 3 of the rules file states. These numbers do not change the
verdict. A new verdict at another weight would need new rules and other runs.

## What the result means

1. At the weight fixed before the run, an audio winner at the choice step changes the predicted choice
   in 2 or 3 episodes of a run. That is 14 of 300 episodes over the six runs.
2. The workspace was ignited at the choice step in nearly every episode (0.975 to 1.000). Audio was the
   winner in 0.183 to 0.350 of the episodes. In the other episodes the term is 0 and cannot change
   the choice.
3. By the definition in the rules, the term goes to the loudest band only. So each changed choice is a
   change to the loudest band. The check does not show that the workspace holds information about
   the objects that the salience lacks.
4. The option with no link stays in place for this room.

## Limits

1. The check is a correlation on saved runs. It is not an ablation. It says nothing about behaviour.
2. Each episode is a one-step change. The values of later episodes stay as recorded.
3. The winner of the workspace is recorded as a sense and not as a band. The term cannot favour the
   quieter band.
4. Rule 2 is weak for a binary term. With the measured q of 0.183 to 0.350, the bound q / (2 - q) of
   the rules file is 0.101 to 0.212, and the measured R squared is 0.098 to 0.206.
5. Rule 1 held at seed 601 with 2 of 40 episodes, which is exactly 0.05. One episode fewer would fail
   it. The food switch runs have the same count of 2 with 60 episodes, so the share is 0.033.
6. The two groups use the same room, binder and gate. They are not independent.
7. One weight carried the verdict. The table above shows that the count rises with the weight.
8. The source of the selection rule (Gurney, Prescott and Redgrave 2001) was read at abstract level
   only, and this check does not use that rule.
9. Each changed choice moves toward the loudest band, so the check cannot show information beyond the
   salience. This was added after a review on 2026-11-11.
10. Control A tested the draw in 43 of 300 episodes. In the other 257 the choice was almost certain
    (largest probability 0.95 or more), so a draw out of order would also match there. All 43 matched.
    The counts for the six runs are 5, 6, 3, 11, 8 and 10, computed again from the saved runs after the
    review.

## What was not done

No live gate. No change to the value gate, the binder, the motor map or the workspace. No indicator of
the rubric moves with this result.
