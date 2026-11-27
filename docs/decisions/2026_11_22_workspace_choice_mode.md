# Step P2 of design P: rules of the choice mode with the plain value head

Date: 2026-11-22. Status: rules written before any code and before any run. No code of this step
exists. Each piece of work below needs the owner's yes before it starts.

The design is P of `docs/decisions/2026_11_15_workspace_action_design.md`. Step P0 gave UNDECIDED
(`docs/results/workspace_choice_read_2026_11.md`). Step P1 gave VOID
(`docs/results/workspace_choice_head_2026_11.md`). The check on the go and no-go variant gave its labels
(`docs/results/choice_head_variant_sign_2026_11.md`, commit `e602dc2`). On 2026-11-22 the owner accepted
that result and chose step P2 with the plain value head. The branch is not merged.

No indicator moves with this step. Nothing here is a claim about consciousness.

## The questions

1. On built data, does the plain value head learn which candidate gives the food from the outcomes of
   its own choices, when it explores by a fixed rule and receives new rows at each update?
2. Can the choice mode be built behind a flag that is off by default, so that a run with the flag off
   is the same, bit for bit, as a run of the code before the build?

The first question is a cheap check before any run. The second is step P2 of the design document.

## What I read before these rules

| Source | What I took from it |
|---|---|
| `docs/decisions/2026_11_15_workspace_action_design.md` | Step P2 is the choice mode behind a flag, with bit identity by the md5 of the records of the movement of a fixed seed, and with no refactor of `run_episode` and `init_components`. The policy makes one choice in each episode, as the value gate does. The table of the value gate is not used in this mode |
| `docs/results/workspace_choice_read_2026_11.md` | Step P0 is UNDECIDED. 4 of 6 runs hold. The choice step is step 12 in each episode of the six runs. At the choice step the broadcast is an exact copy of the audio vector or of the tectum vector in 279 of 300 episodes, and of the tectum vector in 107 of them |
| `docs/results/workspace_choice_head_2026_11.md` | Step P1 is VOID. The gate chose the slot of the food in 89.7 to 97.5 percent of the episodes of the counted units. On built data the plain value head scored 0.904 at 30 rows and 0.897 at 40 rows with the random chooser. With the food and food_80 choosers its score stays high when each outcome is set to 1 (0.999 and 0.997, 0.940 and 0.965), so in those cases the score comes from the choice |
| `docs/results/choice_head_variant_sign_2026_11.md` | The plain value head was trained on repeated rows only (condition C4). With new rows at each update (condition C2) the variant scored 1.0 with the random chooser at both row counts. Hypothesis d, that with 200 updates on the same rows the critic of the variant fits the training rows, HOLDS. A label is not a proof of the cause |
| `docs/results/reversal_2026_10.md` | `--dark-room-food-switch 30` moved the food at episode 29 in the saved food switch runs, and commit `961dc1c` repaired the counter of the room. The project checks bit identity with a control run of seed 73 and the md5 of `metrics.csv`. The value gate changed its choice within 5 to 10 episodes after the food moved |
| `docs/decisions/2026_11_08_competing_goals_gate.md` | The control of the value gate is seed 73, 10 episodes, the flags of the value gate gate, and `--existence-drive off --phi-control off`. It recorded the md5 `29ae60f195d2da4dd9fec62d0169ea61` for `metrics.csv`, with 2001 lines and 10 gate episodes |
| Code at `e602dc2` | `models/core/value_gate.py`. `_wire_value_gate`, `_wire_tectal_orienting`, `run_episode` and `main` in `scripts/training/train_rlhf.py`. `scripts/training/session_recorder.py`. `scripts/analysis/check_choice_head.py` |

These rules use no saved run, and I read no saved run for them. Each run that they name is a new run.

## The choice mode

| Item | Rule |
|---|---|
| Flag | `--choice-head`, with the values `off` (the default) and `plain` |
| Needs | `--tectal-orienting value`, and `--map-binding rhythm` or `rhythm-scrambled`, which then bind two bands |
| Refused with | `--policy-input` other than `broadcast`, `--value-outcome hyperbolic`, and `--value-salience-floor` |
| Place in the code | When the flag is `plain`, `main` calls a new function `_wire_choice_head` in place of `_wire_value_gate`. No line of `run_episode` or `init_components` changes. The new module is imported only inside the new function, as `_wire_value_gate` imports the value gate |
| Head | The plain value head of step P1, in a new file `models/core/choice_head.py`. It has a linear layer from 256 to 128 values, tanh, a linear layer from 128 to 128 values, tanh, and a linear layer from 128 to 2 values. This is the body of `StandardActorCritic`. The initial weights are the defaults of the library. Output 0 is the value of slot 0, and output 1 is the value of slot 1 |
| Input | The broadcast of the choice step, as the policy receives it. With `--policy-input broadcast`, `run_episode` passes the broadcast as the first argument of `action_core.select_action`. The head does not scale it |
| Candidates | The bands that have a group of one cell or more in the binding record of the step, with their salience, as the value gate reads them |
| Slots | Slot 0 is the candidate with the higher salience. When the two salience values are equal, the lower band number is slot 0, and the record marks the tie |
| Choice step | The first step with candidates after the start of an episode, as for the value gate. The choice is kept to the end of the episode |
| Exploration | See the section "How the head explores" |
| One candidate | No draw and no update. The head follows that candidate, and the record marks the case |
| Outcome | The rule of the value gate with the binary outcome. The outcome is 1 when a step reward above 0.5 comes within 60 steps after the choice. It is 0 at the 60th step without such a reward. It is 0 when a step without sound starts the next episode first |
| Learning | When a choice with two candidates closes, the head makes 5 updates with Adam at a learning rate of 3e-4. The loss is the mean squared error of the output of the chosen slot against the outcome, on the row of that choice only. No row is kept for a later update |
| Movement | The motor map orients to the group of the chosen band, as it orients to the band that the value gate chooses. `--tectal-orienting-idle` acts as it does with the value gate |
| Order in a step | The policy is called first, then the head, then the motor map, as in `_wire_value_gate`. So each random stream of the policy stays as it is |
| Random streams | The initial weights come from the seed (run seed + 2000003) on the CPU generator of torch. The state of that generator is saved before the head is built and put back after it, and no CUDA generator is seeded. The exploration uses a numpy generator of its own with the seed (run seed + 3000005). So the head draws no number from a random stream of the agent. An update draws no number |
| Table of the value gate | Not built and not used. The run writes no `value_gate.json` |
| Record | `choice_head.json`, `choice_head_inputs.npy`, `choice_head.pt`, and `tectal_orienting.json` with the mode `head`. See the section "The record of the head" |
| A loss that is not finite | The head raises `FloatingPointError` |

**Why 5 updates for each choice.** A run of 40 episodes then gives 200 updates, which is the count of
updates of step P1 and of the check on the variant. Each update has one row, and not 24 or 32 rows as
in step P1. The value is set here, before any number of this step exists, and is not tuned.

**Why no row is kept.** In the check on the variant, hypothesis d HOLDS (with 200 updates on the same
rows the critic fits the training rows), and new rows at each update gave a score of 1.0 with the random
chooser. A live run gives a new row at each choice. The rule uses only that new row and keeps no store of
rows.

**What the head replaces.** The value gate gives a learned value to each heard tone
(`models/core/value_gate.py`, which cites Kim and Hikosaka, Neuron 2013). The head gives a learned value
to each candidate, and reads it from the broadcast. It takes the place of the table of the gate. It adds
no encoder of the frame. No biology source was read for these rules.

## How the head explores

Step P1 was VOID because the value gate chose the slot of the food in 89.7 to 97.5 percent of the
episodes. The outcome was then 1 in almost each episode, and it told a head little. In the counted
units of step P1 the gate chose the other slot in 2.5 to 10.3 percent of the episodes. So the head
explores by a fixed rule.

**The rule.** At each choice with two candidates, the head draws two numbers from its own generator,
always, whatever its outputs are. The first number is uniform between 0 and 1. The second number is 0
or 1 with equal chances. When the first number is below 0.25, the choice is an exploration draw, and the
head takes the slot of the second number. Else the head takes the slot with the larger output, and slot
0 when the two outputs are equal.

**The share 0.25.** With this share, the slot that the head does not prefer is chosen in 1 of 8 choices
or more, so in about 5 of 40 episodes. The share is set here before any number of this step exists, and
it is not tuned. A result of the check holds for this share only.

**What exploration costs.** In an exploration draw the head can take the slot without the food. So a run
with the flag on gives food in fewer episodes than the preference of the head alone would give. How
step P3 scores the behaviour with exploration is a requirement of step P3.

## The record of the head

`choice_head.json` holds the settings (the head, the share 0.25, the 5 updates, the learning rate and
the two seeds) and one entry for each choice, in the order of the choices.

| Key | Content |
|---|---|
| `step` | The step of the episode at which the head chose |
| `salience` | The candidates and their salience, as in `value_gate.json` |
| `slots` | The band of slot 0 and the band of slot 1 |
| `tie` | True when the two salience values are equal |
| `outputs` | The two outputs of the head before the choice |
| `greedy_slot` | The slot with the larger output |
| `explored` | True for an exploration draw |
| `chosen_slot`, `choice` | The chosen slot, and its band |
| `outcome`, `steps` | The outcome, and the count of steps from the choice to its close, as in `value_gate.json` |
| `loss_before`, `loss_after` | The loss of the row before the first update and after the fifth update |

`choice_head_inputs.npy` holds one row of 256 float32 values for each choice, in the same order. Each row
is the input that the head read. `choice_head.pt` holds the weights at the end of the run. A choice that
is open at the end of the run is recorded with no outcome.

## The order of the work

| No. | Work | Needs |
|---|---|---|
| 1 | The unit tests of group A, pushed first. Then `models/core/choice_head.py` and the check as a mode of `scripts/analysis/check_choice_head.py` | The owner's yes for code |
| 2 | Check 2, the check on built data, run alone. The end is verified by the row counts | The same yes |
| 3 | The result document of check 2, with VOID or FAILED first, and a review issue. Then a stop | |
| 4 | The unit tests of group B, pushed first. Then the wiring in `scripts/training/train_rlhf.py` | The owner's yes for the build, after the result of check 2 |
| 5 | Check 3, the source of two functions and the bit identity with the flag off. Two control runs, one at a time | The owner's yes for the runs |
| 6 | Check 4, the wiring run with the flag on | The same yes |
| 7 | The result document of step P2, and a review issue. Then a stop | |

The rules of step P3 are the owner's decision after step 7.

## Check 1. Unit tests

Each test is written and pushed before the code that it tests.

**Group A, before check 2.**

| No. | The test |
|---|---|
| A1 | The head has the layers of the plain value head of step P1. For the same seed, its initial weights equal those of `build_head("plain", 256, seed)` of `scripts/analysis/check_choice_head.py` |
| A2 | To build the head leaves the state of the CPU generator of torch, of each CUDA generator that exists, and of the global numpy generator unchanged |
| A3 | With an exploration share of 0, the choice is the slot with the larger output, and slot 0 when the two outputs are equal |
| A4 | Over 10000 choices with the share 0.25, the share of exploration draws is 0.235 to 0.265, and slot 0 takes 0.47 to 0.53 of the exploration draws |
| A5 | Each choice takes exactly two numbers from the generator of the head, whatever the outputs are |
| A6 | The choices of a row set do not depend on its targets. A change of the targets changes no choice before the first update |
| A7 | The loss reads the output of the chosen slot only. After one update on a row made for the test, the weights of the last layer for the other slot are unchanged |
| A8 | After a closed choice the head makes 5 updates on the row of that choice, and the next update uses the next row only |
| A9 | The rows and the targets of a new row set of part A equal those of `new_row_set` of condition C2 for the same data set, fold and update |
| A10 | No seed sequence of check 2 equals the seed of a data set or of another stream of check 2 or of condition C2. No row of a new row set of part A, and no row of part B, equals a row of the data set of its case before the signal is added |
| A11 | In the first pass, the outcome of each choice is 1 exactly when the chosen slot is the slot of the food. In the second pass each outcome is 1 |
| A12 | The training rows have one row for each part, pass, row count, data set, initial seed, fold and checkpoint, and the mean of a group over the initial seeds and the folds equals its mean row |
| A13 to A18 | The label code gives each label of check 2 on rows made for the test. The cases are part A HOLDS, FAILS and NOT DECIDED, part B HOLDS, PARTLY and FAILS, and VOID by control K2 and by control K4. One test for each case, with more than one case in a test where the rows are the same |

**Group B, before check 3.**

| No. | The test |
|---|---|
| B1 | The flag has the default `off`. `plain` needs `--tectal-orienting value` and a rhythm binding of two bands, and it is refused with `--policy-input` other than `broadcast`, `--value-outcome hyperbolic` and `--value-salience-floor` |
| B2 | With the flag off, `main` calls `_wire_value_gate` with the same arguments as before and does not call `_wire_choice_head`. No file of a run gains a key |
| B3 | With a tectum and an action core made for the test, as in `tests/test_value_gate.py`, the wrapper calls the policy first, then the head, then the motor map |
| B4 | The head waits for the first step with candidates, chooses once, and the motor map follows the group of the chosen band to the end of the episode |
| B5 | Slot 0 is the candidate with the higher salience. Equal salience puts the lower band in slot 0, and the record marks the tie |
| B6 | With one candidate there is no draw and no update, the head follows the candidate, and the record marks the case |
| B7 | On the same sequences of rewards and binding records, each choice closes at the same step and with the same outcome as `ValueGate` with the binary outcome. The cases are food at once, no food in 60 steps, and a step without sound before the end of the window |
| B8 | After a closed choice with two candidates the head makes 5 updates on that row only |
| B9 | The head reads the first argument of `select_action`, and the record holds that vector in each value |
| B10 | The record files exist with the keys of the section "The record of the head". No `value_gate.json` is written |
| B11 | A loss that is not finite raises `FloatingPointError` |
| B12 | After the wiring with the flag on, the states of the random generators of the agent equal those of the same wiring with the flag off. A choice takes numbers only from the generator of the head |

## Check 2. The check on built data

This check trains small heads on built data only. It reads no saved run and runs no agent.

### What it uses

| Item | Value |
|---|---|
| Data sets | `built_case` of step P1 with the full signal. Each row has 256 dimensions of normal noise, and the rows with target 1 have a shift of 1 standard deviation in 64 dimensions. Data sets 0 to 9, with 30 and 40 rows |
| Folds | The 5 folds of step P1 and of the check on the variant (`fold_split`). The score uses the test rows of the fold |
| Head | The head of `models/core/choice_head.py`, the same code as the live mode. Initial seeds 0, 1 and 2 |
| Chooser | The exploration rule of the head, with its generator seeded with the sequence (initial seed, data seed, fold, 2) |
| Loss and optimizer | Those of the head. Adam at 3e-4 |
| Score | The AUC of the preference (output 0 minus output 1) against the target on the test rows of the fold. The target is 1 when the food is in slot 0. The head never receives the target |
| Machine | One CPU thread |

The seed sequence of the rows of part B ends in 1, and the seed sequence of the exploration ends in 2.
The rules of the check on the variant found that a sequence of a data seed and two zeros gives the
numbers of the data seed alone. A sequence of four values that ends in 1 or 2 differs in its last value
from each data seed and from each sequence of C2.

### Part A. New rows at each update, as condition C2

At update u, from 1 to 200, the head receives the new row set of condition C2 of the check on the variant
for the data set, the fold and u (`new_row_set`, seed sequence (data seed, fold, u)). The set has the size
of the training rows of the fold, 24 or 32 rows, and it is standardized with the training rows of the
fold, as in C2. The head computes its outputs for the rows and chooses a slot for each row by the
exploration rule. The outcome of a row is 1 when the chosen slot is the slot of the food
(`built_outcomes`). Then the head makes one update on the set. The score is read after 0, 1, 20, 40, 60,
80, 100, 120, 140, 160, 180 and 200 updates, on the test rows of the fold, standardized as in C2.

### Part B. The rows of a run (my addition)

Part B uses the rule of the live mode with the amount of data of a run. The rows of the episodes are the
60 rows of `built_case(60, 1.0, "random", seed sequence (data seed, fold, 0, 1))`. The choices of that
call are not used. At episode e the head receives one row, chooses a slot by the exploration rule,
receives the outcome, and makes 5 updates on that row. The rows are not standardized, as in the live
mode, and the test rows of the fold are used with no standardization too. The score is read after 0, 10,
20, 30, 40, 50 and 60 episodes. The score after 40 episodes is the score of a run of 40 episodes,
because the loop has no value that depends on the total count of episodes.

Part B is my addition. The owner asked for part A. Part A gives the head 4800 or 6400 rows, and a run
gives 40 or 60. The table for step P2 in the rules of the check on the variant says that C2 does not show
learning with the amount of data of a run.

### Second pass

Each part runs again with each outcome set to 1. A score in the second pass shows what the head gets
from its own choices when the outcome tells it nothing.

### Measures, for each training and checkpoint

| Measure | Content |
|---|---|
| `score` | The score of the checkpoint |
| `share_explored` | The share of exploration draws among the choices since the checkpoint before |
| `share_food` | The share of choices of the slot of the food, over the same choices |
| `share_slot_0` | The share of choices of slot 0, over the same choices |
| `mean_outcome` | The mean outcome, over the same choices |
| `informative` | The count rule of step P1 over each choice up to the checkpoint. For each slot, the smaller of its counts of outcome 1 and outcome 0, and then the larger of the two slots |

### Controls with exact expected values

| Control | Expected value | On a mismatch |
|---|---|---|
| K1. Rows | 760 mean rows and 11400 training rows | The check is not complete |
| K2. Exploration | The share of exploration draws over all choices of a part and a pass is 0.24 to 0.26 | VOID |
| K3. New rows | No row of a new row set of part A, and no row of part B, equals a row of the data set of its case before the signal is added | VOID |
| K4. Second pass | The score of the second pass is below 0.60 at each row count, at 200 updates in part A and at 40 and 60 episodes in part B | VOID |
| K5. Outcomes | In the first pass each outcome equals `built_outcomes` of its choice and target, and in the second pass each outcome is 1 | VOID |

K3 and K5 stop the job on a mismatch. K1, K2 and K4 are read from the rows by the label code. Part B
has 36000 choices in each pass, so a share of 0.24 is 4 standard errors below 0.25.

### Labels

The score of a row count is the mean over the 10 data sets of the mean over the 3 initial seeds and the
5 folds, in the first pass. A label holds only when its rule holds at both row counts. The limit 0.80
is the limit of step P1 and of the check on the variant.

| Part | Label | Rule |
|---|---|---|
| A | HOLDS | The score at 200 updates is 0.80 or more at both row counts |
| A | FAILS | The score at 200 updates is below 0.80 at both row counts |
| A | NOT DECIDED | Each other case |
| B | HOLDS | The score after 40 episodes is 0.80 or more at both row counts |
| B | PARTLY | B does not hold, and the score after 60 episodes is 0.80 or more at both row counts |
| B | FAILS | The score after 60 episodes is below 0.80 at both row counts |
| B | NOT DECIDED | Each other case |

These are reported only, with no part in a label. They are the score at each checkpoint, the first
checkpoint at which the score reaches 0.80, the measures of the table above, and the second pass at each
checkpoint.

### Expected counts and cost

| Part | Cases | Trainings | Checkpoints | Training rows | Mean rows |
|---|---|---|---|---|---|
| A | 2 row counts x 10 data sets x 2 passes = 40 | 40 x 3 seeds x 5 folds = 600 | 12 | 7200 | 480 |
| B | 40 | 600 | 7 | 4200 | 280 |
| Total | 80 | 1200 | | 11400 | 760 |

The files are `mean_rows.csv` and `training_rows.csv` in
`docs/results/data/workspace_choice_mode_2026_11/`. The label mode writes `labels.json` to the same
folder. The time is not measured. Before the run the author times 5 trainings of each part and states
the estimate, as for the check on the variant.

### Limits of check 2

1. The data are built. The signal has one strength, it is in each row, and the outcome has no miss.
2. A pass shows that the head learns a built signal from the outcomes of its own choices. It does not
   show that the head learns from the broadcast. That is the question of step P3.
3. The share 0.25 and the 5 updates for each choice are set here, and a result holds for these values
   only.
4. The rows of part B are independent. The rows of a live run are not.

## Check 3. The source of two functions, and bit identity with the flag off

**3a. The source.** A script compares the text of `run_episode` and of `init_components` at the build
commit with their text at the base commit. The base commit is the last commit before the first commit of
the wiring code. The texts must be equal.

**3b. Two control runs**, one at a time, with the same command. One runs at the base commit, and one runs
at the build commit with no `--choice-head` flag. The command is the Gate B7 flags
(`docs/decisions/2026_09_16_precision_weighted_bids.md`), the flags of the learn arm of the value gate
gate (`docs/decisions/2026_10_16_value_gate.md`), and `--existence-drive off --phi-control off`, as in the
control of the competing goals gate.

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
    --seed 73 --enable-audio --rssm-latent-mode continuous \
    --capsule-workspace-source all_levels --existence-drive off --phi-control off \
    --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
    --dark-room-view agent_centered --vision-bid-reduction zscore \
    --audio-salience surprise \
    --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain \
    --dark-room-agent-mark ring --dark-room-agent-colour 217,119,87 \
    --dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible \
    --dark-room-start-between --dark-room-pulse --dark-room-sound-direction off \
    --tectal-orienting value --tectal-orienting-idle still --map-binding rhythm \
    --log-dir runs/p2_ctrl_base_s73
```

The run at the build commit uses `--log-dir runs/p2_ctrl_off_s73`. Each record below must have the same
md5 in the two runs.

| Record | Content |
|---|---|
| `metrics.csv` | The control file of the project |
| `episodes.csv` | One line for each episode |
| `value_gate.json` | The choices that set the target of the motor map |
| `tectal_orienting.json` | The target of the motor map at each step |
| `map_binding.json` | The candidates at each step |
| The movement record of each of the 10 episodes | From `episodes/ep_NNNN/steps.jsonl`, for each line, the fields `step`, `action`, `reward`, `env_phase` and `internals.info_after_step`, written as JSON with sorted keys, one line for each step. `action` is the command that the room received. `info_after_step` holds the distance to the light and the other fields of the room after the step. The field `seconds_inputs_to_end` is left out, because it is wall time and changes between identical runs (`scripts/training/session_recorder.py`) |

Each run must also have 2001 lines in `metrics.csv` and 10 closed choices in `value_gate.json`.

It is reported only whether `metrics.csv` has the md5 `29ae60f195d2da4dd9fec62d0169ea61` that the
competing goals gate recorded. The code changed after that record, so a difference is not a failure of
check 3.

## Check 4. The wiring run with the flag on

One run at the build commit, with the command of check 3b, `--choice-head plain` and
`--log-dir runs/p2_wiring_s73`.

| No. | Expected value |
|---|---|
| W1 | The run ends with 2001 lines in `metrics.csv` and 10 recorded episodes |
| W2 | In episode 0, the choice step is the step of the first choice in `runs/p2_ctrl_off_s73`, and the movement record of each step before it equals the record of that run |
| W3 | Each episode with two candidates has exactly one choice, at the first step with candidates in `map_binding.json`. An episode with fewer candidates has no choice with a draw |
| W4 | Each row of `choice_head_inputs.npy` equals, in each value, the broadcast of its step in `vectors.npz` of its episode |
| W5 | Each closed choice with two candidates is followed by 5 updates, and the weights at the end differ from the initial weights |
| W6 | The run writes no `value_gate.json` |

W2 shows that the head draws no number from a random stream of the agent before its first choice. W4
shows that the head reads the vector that the session record holds as the broadcast.

These are reported only. They are the count of exploration draws, the outputs, the outcomes, and the
mean, the standard deviation and the largest absolute value of the input rows. The scale of the live
broadcast is not known to these rules.

## What each result means

**Check 2.**

| Result | Meaning | For the next step |
|---|---|---|
| VOID | A control failed | No statement. The author reports the control, and the owner decides |
| A HOLDS and B HOLDS | On built data, the plain value head learns which slot gives the food from the outcomes of its own exploring choices, with new rows, and with the rows of a run of 40 episodes | The build (work 4) can start after the owner's yes |
| A HOLDS and B PARTLY | The same, with the rows of a run of 60 episodes and not of 40 | As above. A gate of step P3 then needs runs of 60 episodes or more. The owner decides |
| A HOLDS and B FAILS | The head learns with 4800 or 6400 new rows, and not with the 60 rows of a run | The author proposes no build. A change of the learning rule (more updates for each choice, a store of rows, or longer runs) is a new rule with its own check. The owner decides |
| A FAILS and B FAILS | The head does not learn from its own exploring choices, even with many new rows. The check does not separate the exploration rule from the head as the cause | The author proposes no build. The owner decides |
| Each other case | NOT DECIDED | The author reports the tables, and the owner decides |

**Check 3.**

| Result | Meaning | For the next step |
|---|---|---|
| 3a equal, and each md5 equal | The path with the flag off is the same, for this command | Check 4 |
| A difference in 3a or 3b | The path with the flag off changed | No run with the flag on. The author reports the first record and the first step that differ, and the owner decides |

**Check 4.**

| Result | Meaning | For the next step |
|---|---|---|
| W1 to W6 as expected | The wiring works as these rules state, for one seed and 10 episodes | Step P2 is complete. The result document is written. The rules of step P3 are the owner's decision |
| A mismatch | The wiring does not work as stated | No other run with the flag on. The author reports the item, and the owner decides |

No result of step P2 stops design P without the owner.

## Limits

1. The result of the check on the variant is on built data only. Its labels are for the go and no-go
   variant offline. It did not train the plain value head with new rows.
2. Step P0 is UNDECIDED. The broadcast held the louder tone in 4 of 6 runs. At the choice step it is a
   copy of the audio vector or of the tectum vector in 279 of 300 episodes, and the tectum vector did not
   hold the label in 4 of 6 runs. So a head that reads the broadcast receives the content of the louder
   tone in a part of the episodes only. Check 2 puts the signal in each row.
3. Step P0 also found that the module that supplies the broadcast depends on the label in some runs. A
   head can learn the target from which module supplied the vector. Step P2 cannot show which of the two
   a head uses.
4. Check 2 is on built data, with independent rows and no miss of the food. In a live run the rows come
   one at a time, the decoy levels of the saved rooms come in a fixed order, and the agent can miss the
   food after a correct choice.
5. The share 0.25 and the 5 updates are set here and not tuned.
6. The head reads the broadcast with no scaling. The rows of check 2 have a mean of 0 and a standard
   deviation of 1 in each dimension. The scale of the live broadcast is not known to these rules.
7. Check 3 covers one command, one seed and 10 episodes. Other flags are not covered.
8. Check 4 checks the wiring on one seed and 10 episodes. It does not test learning.
9. No biology source was read for these rules.

## Requirements of step P3, named only

These are not rules. Step P3 needs its own rules and its own review.

1. Fresh seeds, 3 or more, with each run alone (design document).
2. A room with the decoy levels in a random order. In the saved rooms they come in a fixed order, so the
   parity of the episode number gives the target.
3. A room in which the total loudness does not give the target (limit 2 of step P0).
4. If a food switch is used, the present counter of the room. The saved food switch runs moved the food
   at episode 29 with the flag at 30, before the repair in commit `961dc1c`.
5. An arm with the table of the value gate as it is today, as the reference (design document).
6. A cut of the broadcast at the input of the head, with a blank and with the broadcast of another
   episode (design document). In each cut step a check reads the input of the head and confirms that the
   head received the cut vector. A cut that did not reach the head is VOID (internal plan).
7. Each criterion named by its level, with one line on what a pass supports and one line on what a fail
   counts against. The levels are that the room moves the bids, that the bids set the broadcast, that the
   broadcast moves the choice, and that the choice needs the broadcast (internal plan).
8. Each measured behaviour classed as one that survives the cut or one that needs the broadcast. Step P3
   passes only if one behaviour or more needs the broadcast (internal plan).
9. The ratio of check B at 0.5 or more, and the merge that the gate uses with its reason (design
   document).
10. A rule for how the behaviour is scored with exploration (these rules, section "How the head
    explores").
11. The broadcast at the choice step of each episode is recorded, so that the read of step P0 can run on
    the runs of step P3. The session record keeps `vectors.npz` and `steps.jsonl` of each episode, and
    the record of the head gives the step and the input row of each choice. The script of step P0 reads
    the choice step from `value_gate.json` and `map_binding.json`, so it needs a change to read the record
    of the head.

## Ethics framework

Each run declares the existence drive (rule E1). The control runs and the wiring run use
`--existence-drive off`, as the control of the competing goals gate. The build adds no input channel,
and no signal of the head lowers `energy`, raises `fatigue` or raises `damage` (rule E6). The head has
no term of self preservation (rule E2).

## Files

| File | Content |
|---|---|
| `docs/decisions/2026_11_22_workspace_choice_mode.md` | These rules |
| `models/core/choice_head.py` | New. The head, the exploration, the window of the outcome, the update and the record |
| `scripts/analysis/check_choice_head.py` | New modes for check 2 and its labels |
| `scripts/training/train_rlhf.py` | The flag, its checks, `_wire_choice_head`, and its call in `main` |
| `tests/test_choice_mode.py` | New. The unit tests of groups A and B |
| `docs/results/data/workspace_choice_mode_2026_11/` | The rows and the labels of check 2 |
| `docs/results/workspace_choice_mode_2026_11.md` | The result of step P2 |

## Decisions for the owner

1. The yes for the tests and the code of work 1 and for check 2.
2. After the result of check 2, the yes for the build.
3. The yes for the control runs of check 3 and the wiring run of check 4.
4. After step P2, the rules of step P3.

## Changes after the independent review

Date: 2026-11-22. The independent review of commit `8d63e5e` is on the review issue of these rules. Its
verdict is APPROVE WITH CHANGES. The owner accepted its 11 necessary changes N1 to N11 and its optional
changes O1, O2, O4, O5, O6, O7 and O8. The owner did not take O3.

This section replaces each part of the rules above that it names. Where it and an earlier section
disagree, this section holds. No earlier line of this file was edited. No code of step P2 exists.

Each claim about the code in the review was checked against the code at `8d63e5e` before this section
was written. Each one agrees, with one difference of line numbers. The review gives lines 2009 to 2010
of `scripts/training/train_rlhf.py` for the intrinsic bonus in the reward. At `8d63e5e` those lines store
the experience in the memory. The bonus is at lines 2005 and 2006, and lines 1998 and 1999 set the
curiosity to 0 at a step with a positive reward of the room. The claim itself agrees. This section uses
the correct lines.

### N1. K4 and the labels of check 2

K4 is removed as a control. A6 tests that a choice does not read the target. The second pass is reported
at each checkpoint.

The paired difference D is added to the labels. For each data set, D is the mean over the 3 initial seeds
and the 5 folds of the score of the first pass minus the score of the second pass, at the checkpoint of
the label. Its limit is the limit of change 19 of the check on the variant, the larger of 0.05 and 2
standard errors over the 10 data sets (`limit_of_change_19`, `scripts/analysis/check_choice_head.py`
lines 846 to 852). The two passes have the same rows, the same initial weights and the same draws, and
only the outcome differs. So D measures what the outcome adds, and it removes what the head gets from its
own choices.

Part A HOLDS when, at both row counts, the score at 200 updates is 0.80 or more and the mean D is at the
limit or above. A score of 0.80 or more with D below the limit gives NOT DECIDED. FAILS stays as it is.
Part B uses the same condition on D (N11).

In the unit tests, the case "VOID by control K4" is replaced with the case "a score of 0.80 or more with
D below the limit".

### N2. K2

A choice takes its two numbers by `rng.random()` and then `rng.integers(0, 2)`, in this order.

For each training row, the label code draws again from a new generator with the same seed sequence and
counts the exploration draws of the window. The expected value is the recorded count of that training
row, in each training row. So the training rows record the count of exploration draws of each window. A
mismatch makes the check VOID. This replaces the range 0.24 to 0.26 of K2. The share of each part and
pass is reported.

The sentence on 36000 choices in the section of the controls was wrong. The review gives this text for
part B before N3. "Part B has 18000 choices in each pass and 9000 independent draws, because the seed has
no pass and no row count." With N3 part B has one row count. So it has 9000 choices in each pass, and the
two passes and the arm of O2 repeat the same 9000 draws.

### N3. Part B has one training condition

The rows, the draws and the initial weights of part B have no row count. So the trainings at 30 and at 40
rows are the same, and the first 30 rows of the data set of 40 rows have the noise of the data set of 30
rows.

Each training of part B is scored on all 40 rows of the data set of 40 rows, with no standardization. Part
B trains on none of these rows. The labels of part B use this one score.

Then part B has 10 data sets x 2 passes = 20 cases, 300 trainings, 2100 training rows and 140 mean rows.
K1 becomes 620 mean rows and 9300 training rows before the arm of O2 (see "Counts and cost, after the
changes"). A12 follows. The five "folds" of part B are five repeats with their own rows and draws.

### N4. The record of the head

Each entry of `choice_head.json` gets the key `episode`, the index of the episode of the run, and the key
`updates`, which is 5 after a closed choice with two candidates and else 0.

The head gets the episode with no change of `run_episode` or `init_components`, for example with a wrap of
`env.reset` as for the tectal attention (`scripts/training/train_rlhf.py` lines 2528 to 2534). B10 tests
it.

W3 and W4 use `episode`. W5 reads `updates` and the step count of the Adam state, so `choice_head.pt` also
holds the state of the optimizer.

**Note of the author, with no review.** `seed_environment` resets the room once before the first episode
(`scripts/training/train_rlhf.py` line 941). `main` calls it at line 3524, after the wiring of the value
gate at line 3486. So a count of the calls of `env.reset` from the wiring on is one higher than the index
of the episode. This is the error of one episode that the saved food switch runs have. B10 therefore checks
that `episode` equals the index of the episode folder of the session record.

### N5. W2, and the seed of the head

With `--tectal-orienting-idle still`, the command before the first choice is zero whatever the policy draws
(`scripts/training/train_rlhf.py` lines 2489 to 2490). With an empty group the motor map has no target
(`models/core/tectal_motor_map.py` lines 45 to 53 and 91 to 97). So the agent does not move, and `action`
and `info_after_step` of those steps cannot show a change of a stream. Only `reward` can, through 0.1 times
the RND curiosity of the broadcast (`scripts/training/train_rlhf.py` lines 1988 and 2005 to 2006), at steps
with a room reward of 0 or below.

W2 becomes this rule. For each step of episode 0 before the first choice step, the line of `metrics.csv`
must be equal in `runs/p2_ctrl_off_s73` and `runs/p2_wiring_s73`. For each step up to and including the
first choice step, the rows of `broadcast`, `tectum_content` and `audio_content` in `vectors.npz` must be
equal. These depend on the draws of the agent, for example the draw of the continuous latent
(`models/core/sensory_tectum.py` lines 336 to 339).

One rule is added. The live head does not call `build_head` or `torch.manual_seed`, because
`torch.manual_seed` seeds each CUDA generator (`torch/random.py` lines 49 and 67), and the runs use CUDA
when it is present (`scripts/training/train_rlhf.py` line 125). The head seeds the CPU generator alone,
inside the save and restore of its state. A1 still compares the weights with `build_head`.

### N6. The scale before the build

The review asks for the base control of check 3b before work 4, with the base commit at the head of the
branch after work 3. Owner decision a below moves this run to work 0, before work 1, with the base commit
at the commit of this section.

The read of the records of that run is named here, before the run, and it is reported only.

1. The history of `map_binding.json` is cut into episodes of 200 records, as `read_run` does
   (`scripts/analysis/check_choice_read.py` lines 237 to 238).
2. The choice step of each of the 10 episodes is the first step with candidates, as `find_choice_step`
   finds it (`scripts/analysis/check_choice_read.py` lines 55 to 60).
3. At that step, the rows of `broadcast`, `audio_content` and `tectum_content` are read from `vectors.npz`
   of the episode, by the step numbers that it holds.
4. The source of the broadcast is "zero" when each of its 256 values is 0, else "audio vector" when it
   equals the row of `audio_content` in each value, else "tectum vector" when it equals the row of
   `tectum_content` in each value, else "other". Equality is exact, as in
   `scripts/analysis/check_choice_read.py` lines 257 to 264.
5. For the broadcast row, the read reports the mean, the standard deviation of the 256 values (divisor
   256) and the largest absolute value.

The owner decides on a scaling of the input of the head after this read (owner decisions a and b). This
run needs the owner's yes.

### N7. Checks 3 and 4, a run that does not complete, and the cost

The tables of checks 3 and 4 get a VOID row. A run that does not reach 2001 lines of `metrics.csv` and 10
episodes (a crash, a full disk, a stop) is VOID. It is not "a difference" and not "a mismatch". The author
reports the cause, and the owner decides.

Before each run the author reads the running jobs, the free memory and the free disk, and states the
time and disk estimate.

### N8. Text

**N8a. Why no row is kept.** This replaces the paragraph "Why no row is kept". The rule matches the live
mode, which has one new row at each choice. Hypothesis d is about the critic of the go and no-go variant.
The loss of the plain value head has no advantage (`scripts/analysis/check_choice_head.py` lines 135 to
138), and with repeated rows it scored 0.904 and 0.897 in step P1. So hypothesis d gives no reason
against a store of rows.

**N8b.** In the section "How the head explores", "in 1 of 8 choices or more" becomes "in 1 of 8 choices in
expectation, when the head has a strict preference".

**N8c. What the head replaces.** This replaces the paragraph "What the head replaces". The head reads the
broadcast and gives a value to slot 0 (the louder candidate) and to slot 1. The value gate gives a value
to each band. The head reads the output of the workspace and adds no encoder of the frame or of another
sense input. It is a generic network, the body of `StandardActorCritic`, with no biological source. It
takes the place of the table of the value gate in this mode only. The design of the gate cites Kim and
Hikosaka (Neuron 2013), and the head does not rest on that source.

**N8d.** The row "A HOLDS and B FAILS" of "What each result means" gets this meaning. The head learns with
many new rows and one update on each set. It does not learn with the live rule (5 updates on one row) and
the 60 rows of a run. The check does not separate the count of rows from the update rule.

**N8e. The movement record.** `reward` is the reward that goes to the action core, the reward of the room
plus the intrinsic bonus, which holds 0.1 times the RND curiosity of the broadcast
(`scripts/training/train_rlhf.py` lines 1988 and 2005 to 2006).

**N8f. Requirement 11.** The script of step P0 finds the choice step in `map_binding.json`. It takes one
entry of `value_gate.json` for each episode, by its position (`scripts/analysis/check_choice_read.py`
lines 228 to 237).

### N9. A10

A10 covers the stream of the rows of part B and the stream of the exploration. The fold stream (seed 0)
is the stream of data set 0, as in step P1 and C2, and it stays for the comparison with them.

### N10. Requirement 6 of step P3

Requirement 6 adds a cut that keeps the module and changes the content. The input of the head is replaced
with the vector of the same module from another episode with the other target. Step P3 also scores the
choice within each source of the broadcast (audio vector, tectum vector, zero).

### N11. Labels of part B that do not overlap

With the one score of N3 and the D of N1. This replaces the labels of part B.

| Label | Rule |
|---|---|
| HOLDS | The score is 0.80 or more after 40 and after 60 episodes, and D is at the limit or above at both |
| PARTLY | The score is below 0.80 after 40 episodes and 0.80 or more after 60, and D is at the limit or above after 60 |
| FAILS | The score is below 0.80 after 40 and after 60 episodes |
| NOT DECIDED | Each other case, for example 0.80 or more after 40 and below 0.80 after 60, or a score of 0.80 or more with D below the limit |

### Optional changes

**O1.** The initial seed changes with the data set. The head of initial seed s in data set d has the seed
100 x d + s, with s = 0, 1 and 2, so the 30 heads differ. A1 compares with `build_head` of that seed. The
numbers of check 2 are then not comparable with step P1, which used the seeds 0, 1 and 2. The seed
sequence of the exploration stays (s, data seed, fold, 2).

**O2.** Part B gets an arm with a store of rows. At each choice it makes one update on all rows so far, as
the rule of step P1. It has the rows, the seeds, the draws and the two passes of part B. Owner decision d
makes it reported only.

**O4.** The head runs on the CPU and reads a float32 copy of the broadcast, as `tensor_or_none` does
(`scripts/training/session_capture.py` lines 35 to 41).

**O5.** With `--choice-head plain`, the flag `--value-learning-rate` with a value other than the default
0.3 is refused. The review gave the choice between a refusal and a record that the flag is not used. The
author takes the refusal, because B1 can test it.

**O6.** Requirement 10 of step P3. With the share 0.25, a head with a perfect preference takes the slot of
the food in 0.875 of the choices in expectation. The value gate took it in 0.897 to 0.975 of the episodes
of the counted units of step P1. So step P3 also scores `greedy_slot`.

**O7.** A limit of check 2 is added. The two row counts of part A share the initial weights, the draws and
the noise of the first 24 rows of each new row set. So they are not two samples.

**O8.** Check 3b also compares `environment.json` of the two runs (`scripts/training/session_recorder.py`
lines 168 and 314 to 326), so that a change of the software is seen apart from a change of the code.

### Owner decisions of 2026-11-22 on this section

These are owner decisions. No reviewer judged them.

a. **Order.** The base control run of check 3b and the read of N6 come first, before work 1. The base
   commit is the commit of this section. This replaces "before work 4" in N6. Check 3b then covers each
   commit from the rules to the build.
b. **Scaling.** If the owner decides on a scaling of the input of the head after the read of N6, the head
   of work 1 has that scaling, and check 2 tests the head with it.
c. **What the review saw.** N1, N3 and N11 were set after the reviewer ran the second pass of check 2 with
   its own code of the draws. The numbers are on the review issue of these rules. No one ran the first
   pass.
d. **O2.** The arm of O2 is reported only. Its labels use the rule of N11, and it has no row in "What each
   result means". A change of the live learning rule stays a new rule for the owner.

### The order of the work, after the changes

| No. | Work | Needs |
|---|---|---|
| 0 | The checks of N7. The base control run of check 3b at the commit of this section, alone, with the command of check 3b and `--log-dir runs/p2_ctrl_base_s73`. The end is verified by 2001 lines in `metrics.csv` and 10 episodes. Then the read of N6, reported on the review issue. Then a stop | The owner's yes for this one run |
| 1 | The owner's decision on a scaling of the input of the head (owner decision b) | The owner |
| 2 | Work 1. The unit tests of group A, pushed first. Then `models/core/choice_head.py`, with the scaling if the owner decided one, and the check as a mode of `scripts/analysis/check_choice_head.py` | The owner's yes for code |
| 3 | Work 2. Check 2, run alone. The end is verified by K1 | The same yes |
| 4 | Work 3. The result document of check 2, with VOID or FAILED first, and a review issue. Then a stop | |
| 5 | Work 4. The unit tests of group B, pushed first. Then the wiring | The owner's yes for the build |
| 6 | Work 5. Check 3a, and the run of check 3b at the build commit with the flag off, compared with the run of work 0. The checks of N7 before the run | The owner's yes for the runs |
| 7 | Work 6. Check 4, the wiring run. The checks of N7 before the run | The same yes |
| 8 | Work 7. The result document of step P2, and a review issue. Then a stop | |

### Controls of check 2, after the changes

| Control | Expected value | On a mismatch |
|---|---|---|
| K1. Rows | 760 mean rows and 11400 training rows | The check is not complete |
| K2. Exploration | In each training row, the count of exploration draws equals the count that the label code draws again (N2) | VOID |
| K3. New rows | Unchanged | VOID |
| K4 | Removed (N1) | |
| K5. Outcomes | Unchanged | VOID |

### Unit tests, after the changes

| No. | Change |
|---|---|
| A1 | It compares with `build_head` of the seed 100 x data seed + s (O1) |
| A5 | The two numbers are taken by `rng.random()` and then `rng.integers(0, 2)` (N2) |
| A10 | It covers the streams of N9 |
| A12 | It uses the counts of N3 and of the arm of O2 |
| A13 to A21 | These replace A13 to A18, with one test for each case. Part A HOLDS. Part A FAILS. Part A NOT DECIDED from a split of the row counts. Part A NOT DECIDED from a score of 0.80 or more with D below the limit (N1). Part B HOLDS. Part B PARTLY. Part B FAILS. Part B NOT DECIDED from a score of 0.80 or more after 40 episodes and below 0.80 after 60 (N11). VOID by K2, a recorded count of exploration draws that differs from the count drawn again (N2) |
| A22 | New. Part B scores each training on the 40 rows of the data set of 40 rows, with no standardization, and trains on none of them (N3) |
| A23 | New. The arm of O2 makes one update on all rows so far at each choice (O2) |
| B1 | It also checks the refusal of `--value-learning-rate` with a value other than 0.3 (O5) |
| B10 | It also checks the keys `episode` and `updates`, that `episode` equals the index of the episode folder of the session record, and that `choice_head.pt` holds the state of the optimizer (N4) |

The count of unit tests is now 35, 23 in group A (A1 to A23) and 12 in group B (B1 to B12). It was 30.

### Counts and cost, after the changes

| Part | Cases | Trainings | Checkpoints | Training rows | Mean rows |
|---|---|---|---|---|---|
| A | 2 row counts x 10 data sets x 2 passes = 40 | 40 x 3 seeds x 5 folds = 600 | 12 | 7200 | 480 |
| B (N3) | 10 data sets x 2 passes = 20 | 20 x 3 seeds x 5 repeats = 300 | 7 | 2100 | 140 |
| Arm of O2 | 20 | 300 | 7 | 2100 | 140 |
| Total | 80 | 1200 | | 11400 | 760 |

The totals are equal to the totals before the changes by chance. N3 takes 140 mean rows and 2100 training
rows from part B, and the arm of O2 adds the same counts.

| Work | Cost |
|---|---|
| 0 | One run of 10 episodes of 200 steps. By the records of the competing goals gate (60 episodes in about 15 minutes, and 8 episodes in 49 MB with the light tier only), about 2.5 minutes and about 60 MB with the light tier only. The heavy tiers are written only when the free disk is 20 GB or more at the start of an episode. The read of N6 takes seconds. These are estimates, not a measurement of this command |
| 2 | 1200 trainings on built data. The temporary code of the reviewer took 91.8 s for 300 trainings of part A and 43.3 s for 150 trainings of part B, about 0.3 s for each. So 1200 trainings take about 6 minutes. That is the code of the reviewer, not of the author. The author times 5 trainings of each part before the run |
| 5, 6 | One run of the size of work 0 each |

## Owner decision on the scaling

Date: 2026-11-22. This is an owner decision. No reviewer judged it. No earlier line of this file was
edited.

The owner made it after a look at one run of 10 episodes with seed 73, the base control run of work 0
(`runs/p2_ctrl_base_s73`). In that run the broadcast at the choice step was the audio vector in 7 of 10
episodes, with a standard deviation of about 1.0, and the tectum vector in 3 of 10 episodes, with a
standard deviation of 0.028 to 0.048.

**The decision.** The head normalizes each input row before its first layer with
`torch.nn.functional.layer_norm` over the 256 values, with no weight, no bias and eps 1e-5. A row of
zeros stays zero.

**The reasons.** The tectum rows are 21 to 35 times smaller than the audio rows, so without a scaling the
head can tell the module by the scale and can hardly read the tectum content. The scale of the tectum
vector can change with its training. The audio rows are already the output of a LayerNorm, so they
change very little. The scaling removes the scale as a cue of the module, but not each cue, so
requirement 6 of step P3 (N10) stays.

**Changes.**

1. `choice_head_inputs.npy` holds the broadcast row before the normalization, so W4 stays as it is.
2. The normalization is a part of the head. So each row that the head reads passes through it. These are the rows
   of a choice, the rows of an update and the test rows of a score. In part A of check 2 the rows are
   first standardized as in C2, and then the head normalizes them. In part B, in the arm of O2 and in
   the live mode the head normalizes the rows as they come.
3. Three new unit tests in group A.

| No. | The test |
|---|---|
| A24 | The output of the head equals the body applied to the normalized row |
| A25 | A row of zeros gives a finite output |
| A26 | Check 2 passes its rows through the same normalization |

4. Limit 6 ("The head reads the broadcast with no scaling") is replaced by this text. The head
   normalizes each input row with `layer_norm` over its 256 values, by the owner decision on the
   scaling. In the base control run the broadcast at the choice step had a standard deviation of
   0.999955 in the 7 episodes with the audio vector and of 0.028 to 0.048 in the 3 episodes with the
   tectum vector, and a largest absolute value of 5.53 to 5.58 and of 0.052 to 0.129. That read is one
   run of 10 episodes with seed 73.
5. The count of unit tests is now 38, 26 in group A (A1 to A26) and 12 in group B (B1 to B12).

## Changes after the independent review of check 2

Date: 2026-11-22. The independent review of the result of check 2 (commit `07ea393`) is on the review
issue of that result. The owner accepted R1, R2 and R3 and the optional changes S2 and S3, and did not
take S1. R1 and R2 change the result document. R3 and S3 change these rules. No earlier line of this file
was edited.

**R3. The wrap of `env.reset` and the index of the episode.** This replaces the note of the author under
N4.

- `seed_environment` resets the room only when the run has a seed (`scripts/training/train_rlhf.py` lines
  938 to 941). `main` builds the room at line 3523 and calls `seed_environment` at line 3524.
- The wiring of the value gate is at line 3486, before the room exists. So the wrap of `env.reset` is not
  made in `_wire_choice_head`. `main` makes it after line 3524, at the place where the tectal attention
  wraps `env.reset` (line 3532).
- `run_episode` resets the room once at the start of each episode (line 1066). With the wrap made after
  line 3524, the first call of `env.reset` that the wrap counts is the reset of episode 0. So during
  episode k the count of calls is k + 1, and `episode` is the count minus 1. This holds in a run with a
  seed and in a run with no seed.
- B10 tests a run with a seed and a run with no seed, and checks in both that `episode` equals the index
  of the episode folder of the session record.

**S3.** With `--choice-head plain`, an explicit `--value-learning-rate 0.3` passes with no message, because
it is the default. The head does not use the value.

**S2.** `greedy_food` is a reported measure of each checkpoint in check 2b and in step P3. It is the share
of the score rows on which the larger output (slot 0 when the two are equal) is the slot of the food. It is
read directly from the outputs of the head on the score rows.

## Owner decision before work 4, and the rules of check 2b

Date: 2026-11-22. No earlier line of this file was edited. No code of check 2b exists.

**Owner decision, with no review.** The learning rule of the head changes before the wiring, if a check on
built data shows a better rule. The reason is this. The head acts on its larger output. With the live rule
its larger output took the slot of the food in only about 0.77 of the score rows after 60 episodes (the
review of the result of check 2), against about 1.0 in part A. The loss trains only the output of the
chosen slot. A weak head can make step P3 fail for a reason that is not the question of design P.

### The question of check 2b

On built data, with the setting of part B, does another learning rule make the larger output of the head
take the slot of the food more often than the present rule does, with a score that still holds?

### The cause, from the code

`chosen_slot_loss` trains only the output of the chosen slot (`models/core/choice_head.py` lines 75 to 77).
The review measured the mean outputs after 60 episodes of part B on the score rows. Output 1 stayed at
0.274 on the rows with the food in slot 1, while output 0 was 0.910 on the rows with the food in slot 0.
So on many rows with the food in slot 1, output 0 is still the larger one. Each rule below adds a target
or a start for the output that the present rule leaves alone. No ablation separates this cause from
others.

### The three rules

| Name | Rule | Reason from the code | What a miss of the food does |
|---|---|---|---|
| `chosen` | The present rule. The mean squared error of the output of the chosen slot against the outcome | The rule of check 2 | The agent chose the slot of the food and did not reach the food in 60 steps, so the outcome is 0. The output of the food slot on that row is trained toward 0. One wrong target, on one row that is not used again |
| `both` | Rule (a). The output of the chosen slot trains toward the outcome, and the output of the other slot toward 1 minus the outcome. The loss is the mean squared error over the two outputs | In this room one of the two candidates holds the food, so an outcome of the chosen slot also says what the other slot holds. Then each output gets a target at each choice | The output of the food slot is trained toward 0, and the output of the other slot toward 1. Two wrong targets on one row, and the preference on that row turns |
| `hopeful` | Rule (b). The present loss, and both outputs start near 1. After the default initial weights, 1 is added to each of the two biases of the last layer | An output that is never chosen keeps a high value, so the larger output tests that slot until its value falls | As for `chosen`. The output of the food slot on that row is trained toward 0, and the head then tends to test the other slot on rows like it. One wrong target |

In the counted units of step P1, three episodes had the slot of the food chosen and an outcome of 0
(`docs/results/workspace_choice_head_2026_11.md`). Check 2b has no miss, because the built outcome is
exact.

With each outcome set to 1 (the second pass), `both` trains the other output toward 0. So in the second pass
`both` learns to prefer the slot that it chose. D measures what the outcome adds.

### The setting

The setting of part B of check 2, with each value unchanged. The 60 rows of the episodes from
`part_b_rows`, the score on the 40 rows of the data set of 40 rows with no standardization, one choice for
each episode, 5 Adam updates on the row of each choice at a learning rate of 3e-4, the share 0.25, the
layer norm of the input, the seeds of O1 (100 x data seed + initial seed), the exploration seed sequence
(s, data seed, repeat, 2), 10 data sets, 3 initial seeds, 5 repeats, the checkpoints after 0, 10, 20, 30,
40, 50 and 60 episodes, and two passes. The three rules have the same rows, the same initial weights apart
from the bias of `hopeful`, and the same draws.

### Measures

The measures of check 2 at each checkpoint, and `greedy_food` (S2). The paired difference D of N1 is
computed for the score and for `greedy_food`, the first pass minus the second pass, for each data set.

### Controls with exact expected values

| Control | Expected value | On a mismatch |
|---|---|---|
| K1. Rows | 420 mean rows and 6300 training rows | The check is not complete |
| K2. Exploration | As in check 2 (N2), in each training row | VOID |
| K3. New rows | As in check 2 | VOID |
| K5. Outcomes | As in check 2 | VOID |
| K6. The present rule | The score of `chosen` at each checkpoint of each training equals the score of part B in `docs/results/data/workspace_choice_mode_2026_11/training_rows.csv`, in both passes | VOID |

### The rule that chooses the learning rule for work 4, stated before the check

All numbers are means over the 10 data sets, from the mean rows. The limit is the limit of change 19 over
the 10 data sets.

A rule `both` or `hopeful` passes when each of these holds.

1. Its score holds by the rule of N11 (0.80 or more after 40 and after 60 episodes, and D of the score at
   the limit or above at both).
2. Its `greedy_food` is higher than that of `chosen`, by the limit or more, after 40 and after 60 episodes.
   The difference is paired for each data set, because the rules have the same rows and draws.
3. D of its `greedy_food` is at the limit or above after 60 episodes.

| Result | The learning rule of work 4 |
|---|---|
| VOID | No statement. The author reports the control, and the owner decides |
| No rule passes | `chosen`, the present rule |
| One rule passes | That rule |
| Both rules pass | The rule with the higher mean `greedy_food` after 60 episodes. With equal means, `both` |

The chosen rule goes to work 4 only after the owner's yes. The value 0.897, the lowest share of the
episodes in which the value gate took the slot of the food in the counted units of step P1, is reported
next to `greedy_food` with no part in the choice.

### Unit tests, written before the code

| No. | The test |
|---|---|
| C1 | `both` on a made row. The target of the chosen output is the outcome, the target of the other output is 1 minus the outcome, and the loss equals the closed form |
| C2 | `hopeful`. On any row the outputs at the start equal those of `chosen` plus 1, and to build the head draws no number from a stream of the agent |
| C3 | `chosen` gives the score rows of part B of check 2 for one training, in each pass (K6) |
| C4 | `greedy_food` on made outputs, with slot 0 on a tie |
| C5 | The three rules take the same two numbers for each choice (K2 applies to each) |
| C6 | The second pass of `both` gives the other output the target 0 |
| C7 | The choosing rule on made mean rows. The cases are no rule passes, one passes, both pass with different means, both pass with equal means, and VOID |
| C8 | The rows of each training and their means, and the counts 420 and 6300 |

### Counts and cost

| Item | Count |
|---|---|
| Cases | 3 rules x 2 passes x 10 data sets = 60 |
| Trainings | 60 x 3 initial seeds x 5 repeats = 900 |
| Training rows | 900 x 7 checkpoints = 6300 |
| Mean rows | 60 x 7 = 420 |

One training of part B took 0.258 s in the timing of check 2, so 900 trainings take about 4 minutes. The
author times 5 trainings of each rule before the run. The code is a rule option of `ChoiceHead` in
`models/core/choice_head.py`, with `chosen` as the default, and a mode of
`scripts/analysis/check_choice_head.py`. The rows go to `docs/results/data/workspace_choice_rule_2026_11/`.

### Limits

1. Built data, with the signal in each row, independent rows and no miss of the food.
2. The setting of part B only. A rule chosen here is not tested in part A.
3. `both` rests on one food in two candidates. A room with no food or with two foods breaks that.
4. A pass does not show that the head learns from the broadcast. That is the question of step P3.
5. The owner decision on the learning rule was made after a look at the result of check 2 and at the
   review of it.

## Changes after the independent review of check 2b

Date: 2026-11-25. The independent review of commit `eac6632` is on the review issue of check 2b. Its
verdict is APPROVE WITH CHANGES. The owner accepted its necessary changes R1 to R6 and its optional
changes O2, O3, O4 and O5. The owner did not take O1, so check 2b has no arm with misses.

This section replaces each part of the section "Owner decision before work 4, and the rules of check 2b"
that it names. Where the two disagree, this section holds. No earlier line of this file was edited. No
code of check 2b exists.

**Owner decision on point 4 of the review, with no review.** `hopeful` is the rule that the owner meant.
It is example (b) of the two examples of a better rule that the owner gave before check 2b ("both
outputs start at 1, so the head tests a slot until its value falls"). The owner accepts the form of the
author. 1 is added to each of the two biases of the last layer, so the outputs start near 1. R5 sets the
tolerance of test C2 for this form.

**The claims of the review about the code.** Each one was checked against the code at `5d235f8`, the head
of the branch when this section was written. The files that the review names did not change after
`eac6632`. Each claim agrees. Three notes, none a disagreement.

- The review gives lines 938 to 943 of `scripts/training/train_rlhf.py` for `seed_environment`, and R3 of
  the section "Changes after the independent review of check 2" gives 938 to 941. The function body is
  at lines 938 to 943, and `env.reset(seed=seed)` is at line 941. Both ranges hold the reset.
- The time of 0.55 s for each training of part B (O4) is computed from the review of the result of check
  2. Part B took 165.6 s there, and part B has 300 trainings, so 0.552 s for each.
- The numbers of the temporary computation of the review were computed again from the untrained heads of
  the 10 score sets, with no update. Each score set has 20 targets of 1. The outputs of `hopeful` at the
  start were 0.4818 to 1.5216, mean 1.0103, standard deviation 0.1501. 0 of 30 heads gave outputs bit
  equal to those of `chosen` plus 1, the largest difference of an output was 1.19e-07, and the largest
  difference of a preference was 2.09e-07.

### R1. Tie rule

This replaces the row at line 900, "| Both rules pass | The rule with the higher mean `greedy_food` after
60 episodes. With equal means, `both` |". The text "With equal means, `both`" is removed.

| Result | The learning rule of work 4 |
|---|---|
| Both rules pass | For each data set, the difference is the `greedy_food` of `both` minus that of `hopeful` after 60 episodes. When the absolute value of the mean of the 10 differences is below the limit of change 19 of those differences, the rule is `hopeful`, because it makes no assumption on the room. Else the rule with the higher mean |

C7 gets the cases "both pass, difference within the limit" and "both pass, difference above the limit,
each way". This replaces the case "both pass with equal means" at line 916.

### R2. `both` and a room with two foods

When work 4 uses `both`, the flag refuses `--dark-room-food-sizes` and `--dark-room-food-sizes-after`,
because with them both candidates give food and `both` trains the output of a slot with food toward 0.

A unit test of group B checks the refusal. It is B13, and it is written in work 4, with the other tests of
group B, before the wiring.

Limit 3 at line 937 ("`both` rests on one food in two candidates. A room with no food or with two foods
breaks that.") stays, and it gets this text. The competing goals room
(`docs/decisions/2026_11_08_competing_goals_gate.md`) is a room with two foods. The flags
`--dark-room-food-sizes` and `--dark-room-food-sizes-after` make it.

### R3. The misses

This replaces the sentence at lines 850 and 851, "In the counted units of step P1, three episodes had the
slot of the food chosen and an outcome of 0 (`docs/results/workspace_choice_head_2026_11.md`)." The
sentence "Check 2b has no miss, because the built outcome is exact." at lines 851 and 852 stays.

In step P1, with the corrected division, three episodes had the slot of the food chosen and an outcome of
0. Two are in the counted units (`vg_learn_s601`, `rv_switch_s702` before), in 195 food choices (0.010),
and one is in `rv_switch_s701` after, a unit reported only. Over the nine units it is 3 in 270 food choices
(0.011).

### R4. K6

K6 also compares, in each training row of `chosen` in both passes, `count_choices`, `count_explored`,
`share_explored`, `share_food`, `share_slot_0`, `mean_outcome` and `informative` with the committed rows of
part B, exactly. These follow each choice, so they show the code path more closely than an AUC on 40 rows.
C3 follows. This adds to the row of K6 at line 880 and to the row of C3 at line 912.

### R5. C2

This replaces the text "the outputs at the start equal those of `chosen` plus 1" in the row of C2 at line
911. The outputs at the start equal those of `chosen` plus 1 within 1e-6 in each value. The preference at
the start equals that of `chosen` within 2e-6.

### R6. The D parts

Limit 6 is added. The D part of condition 1 and condition 3 are controls. The second pass has no
information on the food, so they are expected to hold whenever the score part of condition 1 and
condition 2 hold. They can fail only if a rule gets a high score or `greedy_food` with each outcome 1.

The mean second pass `greedy_food` of each rule is reported after 40 and 60 episodes.

### O2

Report the second pass `greedy_food` of each rule, and mark `both` when its mean is below 0.5 by the limit
or more.

### O3

Condition 2 states its limit. The limit of condition 2 is the limit of change 19 of the 10 paired
differences, rule minus `chosen`, at each of 40 and 60 episodes.

### O4. The cost

This replaces the sentence at lines 928 and 929, "One training of part B took 0.258 s in the timing of
check 2, so 900 trainings take about 4 minutes." The cost is about 4 to 8 minutes. One training of part B
took 0.258 s in the timing of check 2 (900 x 0.258 s = 232 s) and 0.55 s in the review of the result of
check 2 (900 x 0.552 s = 497 s). The author times 5 trainings of each rule before the run.

### O5

In the paragraph of 0.897 at lines 902 to 904, this is added. The three units after the food switch of
step P1 had 0.742 to 0.839 (`rv_switch_s701` after 0.742, `rv_switch_s702` after and `rv_switch_s703`
after 0.839), reported only.

### Unit tests and counts, after the changes

| No. | Change |
|---|---|
| C2 | R5 |
| C3 | R4 |
| C7 | R1. The cases are no rule passes, one passes, both pass with a difference within the limit, both pass with a difference above the limit with `both` higher, both pass with a difference above the limit with `hopeful` higher, and VOID |
| B13 | New (R2). With `--choice-head plain` and the rule `both`, the flag refuses `--dark-room-food-sizes` and `--dark-room-food-sizes-after`. Work 4 |

The count of numbered unit tests is now 47, 26 in group A (A1 to A26), 13 in group B (B1 to B13) and 8 in
group C (C1 to C8). Before this section it was 46. Groups A and C exist before work 4. Group B is written
in work 4.

The counts of check 2b do not change. They are 900 trainings, 6300 training rows and 420 mean rows.

## Owner decision on the open choices of the code of check 2b

Date: 2026-11-25. The code of check 2b is at `6658a33`. In a report on the issue of the review of the rules of check 2b, the author listed 14 places where the code makes a choice that the rules leave open. The owner decided each of them before the timing and the run of check 2b, when no number of check 2b existed. Each one stays as coded. No earlier line of this file was edited.

- Row 1, the loss of `both`. The loss is `nn.MSELoss()` with its mean over the two outputs of the row, as the words "the mean squared error over the two outputs" at line 847 say. Each update of check 2b reads one row, so the sum of the two squared errors is this mean times 2. A constant factor of the loss changes the update of Adam only through its eps of 1e-8.
- Row 4, K2. K2 (N2) applies to the training rows of each rule apart, as C5 at line 914 says ("K2 applies to each"). The grouping key of the label code of check 2 has no rule.
- Row 10, the limit of O2. It is the limit of change 19 of the 10 second pass `greedy_food` numbers of the rule, as the sentence at lines 884 and 885 says ("The limit is the limit of change 19 over the 10 data sets"). The mark of `both` is reported after 40 and after 60 episodes. It has no part in the choice.

## Owner decision before work 4

Date: 2026-11-27. Check 2b gave "No rule passes", so the learning rule of work 4 is `chosen`. It is the rule of the row "Learning" in the section "The choice mode" (line 53), the mean squared error of the output of the chosen slot against the outcome. The independent review of the result accepted it, with changes to the text of the result only. The owner gave the yes for work 4. No earlier line of this file was edited.

- The flag `--choice-head` keeps only the values `off` and `plain`. It gets no value or option that selects `both` or `hopeful`.
- B13 is not written. The flag has no value that selects `both`, so the refusal of R2 has nothing to check. A later change that adds `both` needs the refusal of R2 and its test.
- The count of numbered unit tests is now 46, 26 in group A (A1 to A26), 12 in group B (B1 to B12) and 8 in group C (C1 to C8).

## Changes after the independent review of the wiring

Date: 2026-11-27. The independent review of work 4, the wiring at `b59d62d`, is on the review issue of the
wiring. Its verdict is ACCEPT WITH CHANGES. The owner accepted the review, its necessary changes C1, C2
and C3, and its optional changes O1, O2, O3 and O4, each in the words of the review. For C3 and for the
build commit, the owner gave a rule in the owner's own words. No reviewer judged these two rules.

This section replaces each part of the rules above that it names. Where it and an earlier section
disagree, this section holds. No earlier line of this file was edited. No code of checks 3 and 4 exists,
and no run of check 3b or check 4 was made.

### The build commit

Owner rule. The build commit is the commit on which check 3b and check 4 run. Each commit after `b59d62d`
must change no file under `models`, `simulations` or `scripts/training`. The code of checks 3 and 4 shows
this with a git diff, and the check is VOID if a file changed.

This gives the meaning of "the build commit" in checks 3a, 3b and 4.

### C1. W3 and an episode with one candidate at its first step with candidates

In the words of the review. "In W3, an episode with two candidates is an episode whose first step with
candidates has two candidates. At that step the code makes its one choice. An episode whose first step
with candidates has one candidate has one choice with `one_candidate` true and no draw, also when a later
step has two candidates. W3 reports the count of such episodes." The code of check 4 uses this
definition.

### C2. The part of B10 that is not tested

In the words of the review. "B10 does not test the session record. The code of check 4 does it. For each
entry of `choice_head.json`, the folder `episodes/ep_NNNN` with NNNN equal to `episode` exists, and W4
compares the row of the entry with row `step` of `vectors.npz` of that folder. `step` must also equal the
first step with candidates of that episode in `map_binding.json`, cut into episodes of 200 records (N6)."

So W4 and W3 find an error of one episode or of one step. The check that `episode` equals the index of
the episode folder of the session record moves from B10 (line 724, and lines 808 and 809 of R3) to check
4.

### C3. The base commit of check 3a

The rules named one base commit, the last commit before the first commit of the wiring code (lines 289 to
291). That is `32146d1`. Owner decision a (lines 677 to 679) made `1366756` the base commit of the base
control run. The report of work 4 compared with `84b678b`.

Owner rule. Check 3a compares the text of `run_episode` and `init_components` at the build commit with
their text at `1366756` and at `32146d1`, and reports both. Each must be equal.

This replaces 3a at lines 289 to 291.

### O3. Requirement 12 of step P3

In the words of the review. "The choice is kept to the end of the episode" holds in the code only because
the room has a step without sound at step 0 only. New room code for step P3 must keep this, or the rules
of P3 say what a step without sound inside an episode does.

This is requirement 12 of the section "Requirements of step P3, named only". The owner decided that row
10 of the report of work 4 needs no decision, because with the command of check 3b the only step without
sound is step 0 of each episode.

### O1, O2 and O4

O1 is a new test of group B, B14, in the words of the review. It reads the text of `main` with `ast` and
checks one call of `_wire_orienting(args, action_core, tectum)`, the call of `_check_choice_head_flags`
after `_check_competing_goals_flags`, and the call of `_wire_choice_head_episodes` only under
`live_choice_head is not None`. The number B13 stays with the refusal of R2, which is not written.

O2 changes B12. B12 also compares the state of each CUDA generator when CUDA exists.

B14 and the change of B12 test code that exists at `b59d62d`. So they must pass with no change of the
code.

O4 changes row 1 of the table in the report of work 4, by a new comment on the review issue of the
wiring. It changes no rule.

The count of numbered unit tests is now 47, 26 in group A (A1 to A26), 13 in group B (B1 to B12 and B14)
and 8 in group C (C1 to C8). The code of checks 3 and 4 gets its own unit tests on made records, written
and pushed before that code.

## Owner decisions on the open places of the code of checks 3 and 4

Date: 2026-11-27. The code of checks 3 and 4 is `scripts/analysis/check_choice_wiring.py` at `448d5a7`, with
its unit tests D1 to D19 in `tests/test_choice_wiring.py`. In a report on the review issue of the wiring,
the author listed 30 places where the rules leave that code open, and marked 13 rows that can change a
result. The owner decided the 13 rows before any run of checks 3 and 4. These are owner decisions in the
owner's words. No reviewer judged them. No earlier line of this file was edited.

The rows, as the report of the author names them, in short and in the words of the author.

| Row | The place |
|---|---|
| 1 | The rule of the build commit, read with the net git diff of two commits |
| 2 | A build commit that does not descend from `b59d62d` |
| 3 | The commit at which check 3a is read |
| 6 | The label of check 3a when the rule of the build commit fails and a text differs |
| 8 | What "10 episodes" of N7 is |
| 12 | A difference of `environment.json` (O8) |
| 14 | A run made with a changed tracked file |
| 16 | The two runs of check 4 with different commits |
| 18 | The run that gives the first choice step of W2 |
| 20 | A vector of W2 with no row at the choice step, or with a NaN |
| 24 | "Row `step` of `vectors.npz`" in W4 (C2) |
| 27 | A complete run with no record of the head |
| 29 | The label of check 3 as a whole |

The decisions, in the owner's words.

- Row 1. The code follows the words of my rule. The check is VOID when a commit after b59d62d changes a file under models, simulations or scripts/training, also when a later commit puts the file back. The code reads this with git log --name-only and keeps the net git diff.
- Row 2. As coded.
- Row 3. The label of check 3a is read at the commit that ethics_manifest.json of the run with the flag off records. The mode --identity makes the comparison of 3a at that commit and reports it. The mode --source with HEAD before the runs is a first check only. If it is not EQUAL, no run is made.
- Row 6. As coded. VOID is read first.
- Row 8. A run is VOID by N7 also when a steps.jsonl of the 10 episodes does not have exactly 200 lines.
- Row 12. As coded. environment.json is compared and reported, with no part in the label, as O8 says.
- Row 14. A check is VOID when git_tracked_changes is true in ethics_manifest.json of one of its runs.
- Rows 16, 18, 20, 24 and 27. As coded.
- Row 29. The label of check 3 as a whole is VOID when 3a or 3b is VOID. Else it is A DIFFERENCE when 3a or 3b gives A DIFFERENCE. Else it is EQUAL. The mode --identity reports it.
- Yes for the tests and the code of these changes. No run before my next yes.

The unit tests of rows 1, 3, 8, 14 and 29 go in `tests/test_choice_wiring.py`, and they are pushed before
the change of the code.

## Owner decisions on N1 and N8, and the yes for the runs of checks 3 and 4

Date: 2026-11-27. The code of checks 3 and 4 is at `12155c1`. In a report on the review issue of the
wiring, the author listed 11 new places where the rules leave that code open, and marked N1 and N8 as
places that can change a result. N1 is a merge commit in the range of the rule of the build commit, which
`git log --name-only` does not read. N8 is a manifest whose `git_tracked_changes` is not true or false. The
owner decided both and gave the yes for the runs, before any run of checks 3 and 4. These are owner
decisions in the owner's words. No reviewer judged them. No earlier line of this file was edited.

- N1. As coded. Before each run the author checks that git rev-list --merges b59d62d..HEAD gives no commit. If it gives one, no run is made.
- N8. As coded. The author reports git_tracked_changes of each run from the reports of --identity and --wiring. If a value is not false, the author stops and reports before any other run or result text.
- Yes for the run of check 3b, and then for the run of check 4 only if check 3 is EQUAL. One run at a time.
- Yes for the read of runs/p2_ctrl_base_s73 by the mode --identity, and of its session.json for the command. No other read of a saved run.
- A run that is VOID is not made again before my yes.
