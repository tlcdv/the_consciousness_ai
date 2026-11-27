# Step P2 of design P, checks 3 and 4 of the choice mode

Date: 2026-11-27. Rules: `docs/decisions/2026_11_22_workspace_choice_mode.md`, the sections "Check 3" and
"Check 4", N5, N6, N7 and O8 of "Changes after the independent review", and the three dated sections from
"Changes after the independent review of the wiring" to "Owner decisions on N1 and N8, and the yes for the
runs of checks 3 and 4". The wiring is at `b59d62d`. The code of the checks is
`scripts/analysis/check_choice_wiring.py` at `12155c1`. The build commit of the two new runs is `35188bf`.

**Check 3 is EQUAL. Check 4 is AS EXPECTED.** No check is VOID, no record differs, and no item of check 4
is a mismatch.

One seed of 10 episodes is not evidence about the question of design P. Nothing here is a claim about
consciousness, and no indicator moves.

The rules give these meanings ("What each result means"). For check 3 with "3a equal, and each md5 equal",
"The path with the flag off is the same, for this command", and the next step is check 4 (line 370). For
check 4 with "W1 to W6 as expected", "The wiring works as these rules state, for one seed and 10 episodes",
and "Step P2 is complete. The result document is written. The rules of step P3 are the owner's decision"
(line 377).

## Labels

| Check | Label | File | What it rests on |
|---|---|---|---|
| 3a, first check at HEAD before the runs | EQUAL | `runs/p2_checks/source_head.json` | The text of `run_episode` and of `init_components` at `35188bf` equals their text at `1366756` and at `32146d1` |
| 3a, at the commit of the run with the flag off (owner decision on row 3) | EQUAL | `runs/p2_checks/check3.json`, key `source` | The same comparison at `35188bf`, the commit in `ethics_manifest.json` of `runs/p2_ctrl_off_s73` |
| 3b | EQUAL | `runs/p2_checks/check3.json`, key `label` | Each of the 15 records of the table of check 3b has the same md5 in the two runs, and each run has 10 closed choices |
| 3 as a whole (owner decision on row 29) | EQUAL | `runs/p2_checks/check3.json`, key `check_3` | 3a EQUAL and 3b EQUAL |
| 4 | AS EXPECTED | `runs/p2_checks/check4.json` | W1 to W6 hold |

The run folders and the summary files are under `runs/` of the worktree, which git does not track. They
are not committed.

## The runs

| Run | Folder | Commit, from `ethics_manifest.json` | Counts | Size of the folder, in MB of 2^20 bytes |
|---|---|---|---|---|
| The base control run of work 0, 2026-11-22 | `runs/p2_ctrl_base_s73` | `1366756c086835ad0b9bdbe7eda73f4df4dce286` | 2001 lines in `metrics.csv`, `ep_0000` to `ep_0009` with 200 lines in each `steps.jsonl`, 10 closed choices (`check3.json`, key `counts`) | Not measured for this document |
| Check 3b, the flag off | `runs/p2_ctrl_off_s73` | `35188bffa8f74d1043ff6d1651e65516f9ed69bd` | The same four counts (`wc -l` of the files, and `check3.json`) | 975 MB |
| Check 4, `--choice-head plain` | `runs/p2_wiring_s73` | `35188bffa8f74d1043ff6d1651e65516f9ed69bd` | 2001 lines, `ep_0000` to `ep_0009` with 200 lines in each `steps.jsonl`, and no `value_gate.json` (`wc -l` of the files, and `check4.json`) | 972 MB |

The command of the two new runs is the command of check 3b (lines 299 to 312 of the rules), with
`--log-dir runs/p2_ctrl_off_s73`, and then with `--choice-head plain` and `--log-dir runs/p2_wiring_s73`.
Each ran alone in the background with `.venv-tcai-py312` and `-B`, from the worktree. The end of each run was
verified by the counts above, not by the exit code. The error output of each run has the warning of the
gammatone filter and the warning of the stage 2 appraisal, and no traceback.

The `argv` of `session.json`. In `runs/p2_ctrl_base_s73` and `runs/p2_ctrl_off_s73` it has 54 items. The
two differ only at the last item, 54 of 54, the value of `--log-dir`. In `runs/p2_wiring_s73` it has 56 items.
Items 53 and 54 are `--choice-head plain`, before `--log-dir`. It differs from `runs/p2_ctrl_off_s73` only by
these two items and by the value of `--log-dir`.

Before each run the author made the checks that the owner set for these runs. No training, probe, test or
render job of any project ran. The 14 Python processes were servers of other programs, and one of them used
about one CPU core, as at the base run. The Blender program was idle (0.06 to 0.16 CPU seconds in 5 seconds).
`nvidia-smi` showed no process of type C, and 9 desktop processes of type C+G with no memory value, at 0% GPU.
The available memory was 8483 MB before check 3b and 9550 MB before check 4, and the free disk on C: was
57.36 GB and 56.52 GB. HEAD was `35188bf`, equal to the remote branch, with no changed file. No merge commit
and no file under `models`, `simulations` or `scripts/training` came after `b59d62d`. The OneDrive item is
in the next section.

## OneDrive during the runs

FAILED. The check "OneDrive must not run" before the run of check 3b did not see `OneDrive.Sync.Service`,
because it looked only for a process with the exact name OneDrive. So the run of check 3b was made when that
check failed.

`OneDrive.Sync.Service` (PID 67740, started 2026-11-21 07:30:30 UTC) was present during the base run and
during the run of check 3b. By its PID and its start time it existed from 2026-11-21 07:30:30 UTC to
2026-11-27 08:21:00 UTC, after the end of the run of check 4 (the run took place from 08:14:28 to 08:17:03
UTC). Before the run of check 3b the author reported that OneDrive did not run. The check before check 4
found it, and the author stopped before check 4 and reported it. Before the run of check 4, no process named
OneDrive ran, and `OneDrive.Sync.Service` had PID 67740, the start time 2026-11-21 07:30:30 UTC and 33 MB.

The author read these values with the PowerShell command `Get-Process` on this computer. The memory is the
working set, in MB of 2^20 bytes. The values before the run of check 4 come from a read between 08:13:54 and
08:14:10 UTC. The time 08:21:00 UTC is from a read with `Get-Process -Id 67740` after the run of check 4 and
before the commit of this document. That read gave the same name and start time, and 33 MB. No file holds
these reads.

The owner decided on the stop report, in the owner's words.

- The run of check 3b counts, and check 3 is EQUAL. No cause of VOID in the rules applied. OneDrive.Sync.Service (PID 67740, started 2026-11-21 07:30:30 UTC) was present during the base run and during the run of check 3b, so the two runs had the same condition. The check of OneDrive before the run of check 3b looked only for the exact name OneDrive. The result document states these facts.
- For check 4, the OneDrive item of step 3 changes to this. No process named OneDrive runs. OneDrive.Sync.Service may run, as in the base run and in the run of check 3b. Report its PID, its start time and its memory before the run.
- Yes for the run of check 4 now. The other items of step 3 stay as they are.

Step 3 is the list of checks before each run.

An outside process can make a record differ or stop a run, and it cannot make two computations give the same
md5. The worktree is not in the folder that OneDrive syncs.

## Check 3a

From `runs/p2_checks/source_head.json` (the first check with `--source --build-commit HEAD`) and from the key
`source` of `runs/p2_checks/check3.json`. The md5 of the text of `run_episode` is
`1835d6131f6747c0e0d0c1ea638a5382`, and the md5 of the text of `init_components` is
`706695391929e8fae92b132c81b0fc23`, each at `35188bf`, `1366756` and `32146d1`. The build commit descends from
`b59d62d`, no commit after `b59d62d` changes a file under `models`, `simulations` or `scripts/training`
(`committed_code_files` is empty), and the net diff is empty (`changed_code_files`).

## Check 3b

From `runs/p2_checks/check3.json`, key `md5`.

| Record | md5 in `runs/p2_ctrl_base_s73` | md5 in `runs/p2_ctrl_off_s73` | Equal |
|---|---|---|---|
| `metrics.csv` | `29ae60f195d2da4dd9fec62d0169ea61` | `29ae60f195d2da4dd9fec62d0169ea61` | yes |
| `episodes.csv` | `eec99df11df07479798b26fd589001f8` | `eec99df11df07479798b26fd589001f8` | yes |
| `value_gate.json` | `25de1496d7b6a3487c74b1d0598ef0ad` | `25de1496d7b6a3487c74b1d0598ef0ad` | yes |
| `tectal_orienting.json` | `f9463f4bd6060379d78462a6e47cbbb3` | `f9463f4bd6060379d78462a6e47cbbb3` | yes |
| `map_binding.json` | `9638437f7e6c2db033a86cc686da45e1` | `9638437f7e6c2db033a86cc686da45e1` | yes |
| `movement ep_0000` | `c677cf8038d1297a3b34ecff791141e5` | `c677cf8038d1297a3b34ecff791141e5` | yes |
| `movement ep_0001` | `4c542000b39bb493729adec77e117b75` | `4c542000b39bb493729adec77e117b75` | yes |
| `movement ep_0002` | `e17073cebac01ccbde2e7f70a29aa730` | `e17073cebac01ccbde2e7f70a29aa730` | yes |
| `movement ep_0003` | `b6a6c62693cca174e7b1c6051ea63528` | `b6a6c62693cca174e7b1c6051ea63528` | yes |
| `movement ep_0004` | `0d3c7c501f771aa02f3f8511643b6cac` | `0d3c7c501f771aa02f3f8511643b6cac` | yes |
| `movement ep_0005` | `a6097488aa15d31e037edb830b3564b2` | `a6097488aa15d31e037edb830b3564b2` | yes |
| `movement ep_0006` | `dec1c8b49f78b806cfdce3a585d518d4` | `dec1c8b49f78b806cfdce3a585d518d4` | yes |
| `movement ep_0007` | `b2d7b69b69262d8c0c720bb323b83721` | `b2d7b69b69262d8c0c720bb323b83721` | yes |
| `movement ep_0008` | `f8e00d8c7841eb5b046d3f5647499316` | `f8e00d8c7841eb5b046d3f5647499316` | yes |
| `movement ep_0009` | `45378ae020a263fac1534a4c072a3ce4` | `45378ae020a263fac1534a4c072a3ce4` | yes |
| `environment.json` (O8, no part in the label) | `34de815e08f3f4a701d63e6d844f1c42` | `34de815e08f3f4a701d63e6d844f1c42` | yes |

- Each of the 16 md5 values of the base run equals the value that the report of the base control run gave
  on the review issue of the rules. So the code of the checks reads the records in the form of that report,
  and these 16 records of the base run did not change after that report.
- `git_tracked_changes` is false in both runs (`tracked_changes`).
- No key of `environment.json` differs (`environment_keys_that_differ` is empty).
- Reported only. `metrics.csv` has the md5 `29ae60f195d2da4dd9fec62d0169ea61` that the competing goals gate
  recorded, in both runs (`metrics_md5_is_competing_goals`).

## Check 4

From `runs/p2_checks/check4.json`. Each item holds, so no item has a cause.

| Item | What `check4.json` gives | Holds |
|---|---|---|
| W1 | 2001 lines in `metrics.csv`, and the folders `ep_0000` to `ep_0009` with 200 lines in each `steps.jsonl` (N7 and the owner decision on row 8) | yes |
| W2, in the form of N5 | The first choice step of episode 0 is step 12 in both runs. The lines of `metrics.csv` of steps 0 to 11 are equal. The rows of `broadcast`, `tectum_content` and `audio_content` of steps 0 to 12 in `vectors.npz` are equal, 13 rows of each | yes |
| W3, with C1 | In each of the 10 episodes, the first step with candidates in `map_binding.json` is step 12 with two candidates, and the episode has one choice at that step with a draw. 0 episodes have one candidate at their first step with candidates | yes |
| W4, with C2 | Each of the 10 rows of `choice_head_inputs.npy` equals, in each value, row `step` of `broadcast` in `vectors.npz` of the folder of its `episode`. 0 rows differ | yes |
| W5 | 10 closed choices with two candidates, each with `updates` 5. The Adam step of each of the 6 parameters is 50, which is 5 times 10. Each of the 6 tensors at the end differs from the initial weights of the head seed 2000076, and the largest change of a value is 0.00981389731168747 | yes |
| W6 | The run wrote no `value_gate.json` | yes |

`git_tracked_changes` is false in both runs (`tracked_changes`).

These are reported only, from the key `reported`. The count of exploration draws is 3. The outcomes of the
10 choices in order are 0, 0, 0, 1, 0, 1, 0, 1, 0 and 1. The outputs are those that the head gave before each
choice. The statistics are of each row of `choice_head_inputs.npy`, the broadcast row before the
normalization of the head.

| Choice (episode) | Output 0 | Output 1 | Mean of the input row | Standard deviation (divisor 256) | Largest absolute value |
|---|---|---|---|---|---|
| 0 | 0.1335 | 0.1163 | 0.000237 | 0.028431 | 0.051541 |
| 1 | -0.2048 | 0.2163 | -1.41e-08 | 0.999955 | 5.568561 |
| 2 | -0.2210 | -0.1515 | -1.12e-08 | 0.999955 | 5.533874 |
| 3 | 0.1348 | -0.1168 | 1.24e-09 | 0.999955 | 5.576191 |
| 4 | 0.0218 | -0.0492 | -0.0278 | 0.218417 | 0.522567 |
| 5 | 1.3879 | -0.0381 | 8.3e-09 | 0.999955 | 5.576482 |
| 6 | 1.3178 | -0.0308 | 2.44e-08 | 0.999955 | 5.529580 |
| 7 | 0.6752 | -0.0132 | 2.8e-09 | 0.999955 | 5.570104 |
| 8 | 0.1715 | 0.1564 | 0.0153 | 0.155734 | 0.438289 |
| 9 | 0.3394 | -0.0049 | -1.55e-08 | 0.999955 | 5.566613 |

The outputs are rounded to 4 decimals and the statistics to 6 decimals or 3 significant digits. The full
values are in `check4.json`.

## What the agent did in the run of check 4

From `runs/p2_wiring_s73/choice_head.json`. The settings are the head `plain`, the rule `chosen`, the share
0.25, 5 updates for each choice, the learning rate 0.0003, the head seed 2000076 and the exploration seed
3000078.

- 10 choices, one in each episode, each at step 12 with two candidates and no tie.
- The head chose slot 0 in 9 choices and slot 1 in 1 choice.
- 3 choices were exploration draws (episodes 2, 5 and 9).
- 4 choices had the outcome 1 (episodes 3, 5, 7 and 9), and 6 had the outcome 0. No choice was open at the end.

| Episode | Step | Bands of slot 0 and slot 1 | Tie | Larger output | Exploration draw | Chosen slot | Band followed | Outcome | Steps to the close | Updates |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 23, 18 | no | slot 0 | no | 0 | 23 | 0 | 60 | 5 |
| 1 | 12 | 18, 23 | no | slot 1 | no | 1 | 23 | 0 | 60 | 5 |
| 2 | 12 | 23, 18 | no | slot 1 | yes | 0 | 23 | 0 | 60 | 5 |
| 3 | 12 | 18, 23 | no | slot 0 | no | 0 | 18 | 1 | 1 | 5 |
| 4 | 12 | 23, 18 | no | slot 0 | no | 0 | 23 | 0 | 60 | 5 |
| 5 | 12 | 18, 23 | no | slot 0 | yes | 0 | 18 | 1 | 3 | 5 |
| 6 | 12 | 23, 18 | no | slot 0 | no | 0 | 23 | 0 | 60 | 5 |
| 7 | 12 | 18, 23 | no | slot 0 | no | 0 | 18 | 1 | 7 | 5 |
| 8 | 12 | 23, 18 | no | slot 0 | no | 0 | 23 | 0 | 60 | 5 |
| 9 | 12 | 18, 23 | no | slot 0 | yes | 0 | 18 | 1 | 3 | 5 |

This document makes no comparison with the value gate.

## Limits

1. One seed (73) and 10 episodes.
2. Check 3 covers one command, one seed and 10 episodes. Other flags are not covered (limit 7 of the rules).
3. Check 4 checks the wiring on one seed and 10 episodes. It does not test learning (limit 8 of the rules).
   The outputs, the outcomes and the statistics of the input rows are reported only.
4. Before the runs the code of the checks was tested only on made records (the unit tests D1 to D24 in
   `tests/test_choice_wiring.py`). Its md5 values of the base run equal those of the report of the base
   control run.
5. `OneDrive.Sync.Service` was present during the three runs. The owner decided that the run of check 3b
   counts (section "OneDrive during the runs").
6. The run folders and the summary files are not committed.
7. The checks of the computer before each run are reports of the author. No file holds them.
8. Check 3 holds on this computer, with the software that `environment.json` records.

## Commits

| Commit | Content |
|---|---|
| `b59d62d` | The wiring of the choice mode behind `--choice-head`, off by default |
| `1ade76a` | The rules after the review of the wiring (C1, C2, C3, O3 and the build commit) |
| `b08e13f` | The unit tests B14 (O1) and the CUDA states of B12 (O2) |
| `0e705fb` | The unit tests D1 to D19 of the code of checks 3 and 4, before the code |
| `448d5a7` | The code of checks 3 and 4 |
| `d0c183d` | The owner decisions on the 13 open rows of that code |
| `bd306f4` | The unit tests D20 to D24 of those decisions, before the code |
| `12155c1` | The code of those decisions |
| `35188bf` | The owner decisions on N1 and N8 and the yes for the runs. The build commit of the two new runs |
