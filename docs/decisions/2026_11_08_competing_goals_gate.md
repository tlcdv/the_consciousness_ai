# Competing goals gate: a large food after a wait against a small food at once

Date: 2026-11-08. Status: design and rules written before any code of the mechanism and before any
training run. The owner chose this direction on 2026-11-08: two positive goods with a cost that does
no harm, and the existence drive off. No room flag, gate option or probe of this document exists yet.
The owner approves these rules before the code is written.

## Why

AE-1 of the indicator rubric asks for learning from feedback and selecting outputs so as to pursue
goals, "especially with flexible responsiveness to competing goals". The value gate and the reversal
showed the learning clause in the dark room (`docs/results/value_gate_2026_10.md`,
`docs/results/reversal_2026_10.md`). Competing goals were never tested. The owner decided on
2026-10-31 that AE-1 stays PARTIAL until a competing goals gate with no noxious signal has run.

The existence drive cannot be the second goal in this room. On 2026-11-08 a read of the code and of
the stage A records showed that the drive does not reach the movement
(`docs/results/reversal_drive_off_2026_11.md`, section of 2026-11-08). The room battery never falls
below 0.8 in 200 steps and food does not restore it, and the value gate reduces the reward to food or
no food. A battery that lowers `energy` is also a noxious channel by rule E6 of
`docs/ethics_framework.md`. So both goals here are foods, and the cost is time.

## Sources

| Source | Read | Statement used here |
|---|---|---|
| Kobayashi and Schultz, J. Neurosci. 28, 7837 (2008), doi 10.1523/JNEUROSCI.1600-08.2008 | Abstract (PubMed) | Monkeys discount reward value hyperbolically by delays of seconds. Dopamine responses to reward-predicting stimuli fall with delay along a similar hyperbolic function and rise with reward size |
| Kawagoe, Takikawa and Hikosaka, Nat. Neurosci. 1, 411 (1998), doi 10.1038/1625 | Abstract (PubMed) | Caudate visual responses are modulated by the expected reward, and saccades to the rewarded direction are earlier and faster |
| Mühlhoff et al., Front. Psychol. 2, 68 (2011), doi 10.3389/fpsyg.2011.00068 | Abstract (PubMed) | Guppies switched their preference from a larger to a smaller reward as the distance to the larger reward increased |
| Wendt and Czaczkes, Biol. Lett. 13 (2017), doi 10.1098/rsbl.2017.0450 | Abstract (PubMed) | Most ant workers (69%) ignored a closer, poorer reward for a farther, better one. When both were of the same quality, most (83%) chose the closer one |
| Baciadonna et al., Sci. Rep. 15, 24363 (2025), doi 10.1038/s41598-025-08616-9 | Abstract (PubMed) | Bumble bees were first trained on each flower alone, a small reward at once and a large reward after 15 s, then given 10 choices. They did not choose the delayed reward consistently |
| Brown and Birch, Phil. Trans. R. Soc. B 380, 20240309 (2025), doi 10.1098/rstb.2024.0309 | Abstract (PubMed) | Trade-offs that weigh reward against injury risk are used as sentience evidence. Simple trade-offs exist in nematodes, so the kind of trade-off and its substrate count |
| Jenkinson et al., bioRxiv 10.1101/2025.10.31.685749 v2 (2026) | Abstract | Cockroaches avoided food in light but did not change the preference with hunger or light intensity. The authors call this a behavioural trade-off and not a motivational one |

## Designs compared

| Design | Second goal and cost | Passes E6 and the drive-off rule | Decision |
|---|---|---|---|
| Battery as a need | Food against energy | No. The battery is a noxious channel (E6), and the drive cannot reach the movement | Rejected |
| Need inside the self-model | Food against an internal need | Passes E6. A drive that changes behaviour needs an E2 review | The owner chose not to take it now |
| Reward size against probability | A large food that comes sometimes, a small one always | Yes | Not chosen. Noisy outcomes need more episodes, and probability does not change across a run without a new flag |
| **Reward size against delay (chosen)** | A large food after a wait, a small food at once | Yes. Food sizes and waits change only the reward | Chosen |

## The mechanism, to be written after approval, every part off by default

| Part | Rule |
|---|---|
| Room, `simulations/environments/simple_visual_env.py` | `--dark-room-food-sizes L,D`: the reward at the light and at the decoy once its food comes. `--dark-room-food-waits L,D`: steps inside an object's radius, counted over the episode, before its food comes. Until then a step there gives -0.01, as outside. `--dark-room-food-change N` with `--dark-room-food-sizes-after L,D`: from episode N on, counted as `--dark-room-food-switch` counts, the sizes change. `--dark-room-forced F`: in the first F episodes only one object is present, the light in even episodes and the decoy in odd ones. The other object is not drawn, makes no sound and gives no food. All need `--dark-room-decoy-visible`. None of them works with `--dark-room-food-switch` |
| Gate, `models/core/value_gate.py` and `_wire_value_gate` | `--value-outcome hyperbolic` with `--value-delay-rate k`: the outcome is the reward of the food step divided by (1 + k (n - 1)), where n counts the steps from the choice to the food. Food still means a reward above 0.5, within 60 steps. `--value-outcome binary` is the default and keeps the present gate byte for byte |
| Reward that the gate reads | The room reward plus the intrinsic bonus. With `--phi-control off`, the default, the phi term is 0, and curiosity is set to 0 at steps with a positive room reward (`scripts/training/train_rlhf.py`). So at a food step the gate reads the food size exactly. The gate runs keep phi off and do not use `--ablate-rnd-zero-on-reward` |
| Everything else | Unchanged: binder, choice by softmax over salience + 2 × value at temperature 0.25, learning rate 0.3, motor map, idle `still` |

## Abstract model (no room), run on 2026-11-08

`scripts/analysis/model_competing_goals.py`. It uses the gate's own choice and update constants, the
salience range of the value gate design (quieter tone 0.34 to 0.48, louder 1.0, the louder tone changes
object every episode), a reach rate of 115 of 120 (`docs/results/rhythm_binding_gate_2026_10.md`), and
the steps from the choice to the food in six earlier runs with no wait. Its controls raise on a mismatch
with `ValueGate`, and they matched.

Travel steps, n 285, from the `value_gate.json` of `vg_learn_s601..603` and `rv_stay_s701..703`: 1 step
61 times, 2 steps 58, 3 steps 45, 4 steps 39, 5 steps 51, 6 steps 18, 7 steps 5, 8 steps 2, 9 steps 5,
10 steps 1. Median 3. The md5 of the six records are in the script output (`6886dbfd`, `fd4988b8`,
`ef841dfd`, `99737b73`, `f8f97a50`, `6dbb4a0f`, first 8 characters).

**Three versions FAILED before the design below.**

1. No forced episodes, food sizes near 1, four share criteria at a 5th percentile of 0.8. No design of 216
   passed. Both foods feed, the gate updates only the value it chose, both values start at 0, and the
   first object that gives food takes over.
2. Six forced episodes at the start. Still no design passed. After the change the mean share of decoy
   first was 0.60 to 0.79 in the seven designs checked. With sizes near 1 the two values differ by less
   than the salience difference of the louder tone can outweigh. Food sizes are room units, and the room's present unit of 1.0 is
   arbitrary, so the search then also scaled the sizes. The gate's own constants did not change.
3. Scaled sizes with four share criteria (a cost arm and a benefit arm). No design of 912 passed. In an
   exploration run, the best chance that a live gate with those four criteria passes at 3 of 3 seeds was
   about 0.66, because twelve checks must all pass. Forced episodes mixed into the run raised it to at
   most 0.70.

**The design.** The benefit arm alone shows the trade-off. Before the change the light gives 24 after a
wait of 20 steps and the decoy gives 3 at once. After the change the light gives 4 after the same wait.
The light stays the larger food throughout, so a rule that always takes the larger food fails after the
change, and a rule that always takes the sooner food fails before it. The search tried 912 designs in an
order fixed before it ran (smallest big / decoy, then size / decoy, then wait, then k, then decoy) and
25 passed. The first is k 0.2, decoy 3, size 4, big 24, wait 20. At the median travel the values
converge to 2.143 for the decoy, 4.444 for the light before the change and 0.741 after it.

Run again with 2000 sequences and a new seed:

| Measure | Mean | 5th percentile |
|---|---|---|
| Benefit arm, light first in episodes 15 to 29 | 0.949 | 0.867 |
| Benefit arm, decoy first in episodes 45 to 59 | 0.928 | 0.733 |
| Benefit arm, change of decoy first | 0.919 | 0.733 |
| Binary arm, change of decoy first | -0.004 | -0.333 |
| Binary arm, light first in episodes 15 to 29 (reported only) | 0.493 | 0.133 |
| Stay, light first in episodes 45 to 59 (reported only) | 0.954 | 0.867 |

After the change, 4 of 5 episodes in a row have the decoy first after a median of 5 episodes (95th
percentile 10, never in 0 of 2000). If the agent behaves as the model, the live gate below passes at
3 of 3 seeds with a chance of 0.802 (C1 0.952, C2 0.848, C3 0.959). So even a perfect match to the model
fails about one gate in five. The other 24 passing designs give 0.737 to 0.867 on the same run. The
highest is k 0.1, decoy 3, size 4, big 32, wait 40. The choice between them is the owner's. These rules
use the first design by the fixed order.

A cost arm, where the light gives 4 with no wait and then 4 after the wait, is reported by the model and
is not in the gate. In the model its light-first share before the change has a 5th percentile of 0.000,
because the decoy at once can take over before the light's value is learned.

## Review of the model, 2026-11-11, before any code of the mechanism and before any run

The numbers of the section above came from the first version of the model. A review found one error and
two weak points in it. The section above stays as written, and this section replaces its numbers.

**The miss rule FAILED against the saved runs.** The first version said that a miss gives no contact and
no food. `scripts/analysis/probe_chosen_object_reach.py` read 9 saved runs of the same room flags
(`vg_learn_s601..603`, `rv_stay_s701..703`, `rv_switch_s701..703`, 480 episodes).

| Model assumption | Measured |
|---|---|
| Reach 115 of 120 (0.958) | The chosen object was first in 459 of 480 episodes (0.956) |
| A miss gives no contact | 21 misses. The agent reached the other object in 21 of 21, and no object in 0 |
| The agent stays at the object while it waits | 0 steps outside in the 20 steps after the first contact in 458 of 459 episodes, and the same count for 40 steps |
| Two candidates at each choice | 2 in 480 of 480 |
| Quieter salience 0.34 to 0.48 | In the 6 runs of the model, n 300: min 0.283, 5th percentile 0.314, median 0.400, 95th percentile 0.520, max 0.571 |

In the new room the other object also gives food. So after a miss in a free episode the gate gives the
other object's food to the value of the chosen tone. The model now does this, and it draws the quieter
salience from the records. A miss in a forced episode stays no contact, because one object is present.
The same seed gave the same miss episodes in two arms (seed 701 episode 28, seed 702 episode 23, seed 703
episodes 8 and 12), so a miss follows the room layout. The model still draws misses at random.

**The first numbers came from too few sequences.** The first version with 2000 sequences gave 0.802 at
its default seed and 0.809 to 0.851 at seeds 1 to 6. With 20000 sequences it gave 0.817 to 0.832 at 4
seeds. The search used 400 sequences for each design. The model now uses 20000 sequences for a report
and 4000 for each design of the search.

**Results of the corrected model.** 61 of 912 designs pass. The first by the fixed order is now k 0.1,
decoy 3, size 4, big 16, wait 20, and no longer the design of the gate section below.

| Design (k, decoy, size, big, wait) | C1 | C2 | C3 | All | All at seeds 1 to 3 | All at reach 0.934 | All at reach 0.971 |
|---|---|---|---|---|---|---|---|
| 0.2, 3, 4, 24, 20, the design of the gate section | 0.969 | 0.923 | 0.977 | 0.893 | 0.890 to 0.894 | 0.730 | 0.945 |
| 0.1, 3, 4, 16, 20, first by the fixed order | 0.963 | 0.886 | 0.945 | 0.853 | 0.844 to 0.847 | 0.662 | 0.917 |
| 0.1, 3, 4, 24, 30 | 0.981 | 0.942 | 0.984 | 0.923 | 0.918 to 0.924 | 0.774 | 0.967 |
| 0.1, 3, 4, 32, 40 | 0.982 | 0.946 | 0.988 | 0.929 | 0.930 to 0.936 | 0.798 | 0.970 |
| 0.1, 3, 4, 32, 45, the highest of the 61 | 0.980 | 0.953 | 0.987 | 0.933 | not run | not run | not run |

All columns but the last two use reach 115 of 120. The reach range is the 95 percent Wilson interval of
the measured 459 of 480 (0.934 to 0.971). Differences of about 0.01 between designs are inside the
spread between seeds.

**What this means.** The reach rate moves the pass chance more than any choice of design does. One
design gives 0.730 to 0.945 across the reach range, and the designs differ by at most 0.08 at one reach
rate. So the offline diagnosis below also reports the miss rate, and the owner decides if the gate runs
when it is above 0.066. The choice between the design of the gate section, the first design by the fixed
order and a design with a higher chance is the owner's.

**The owner's choice, 2026-11-11: k 0.1, decoy 3, size 4, big 32, wait 40.** The gate section below now
uses it. Before the change the light gives 32 after a wait of 40 steps and the decoy gives 3 at once.
After the change the light gives 4 after the same wait. At the median travel the values converge to
2.500 for the decoy, 6.154 for the light before the change and 0.769 after it. In the corrected model
the benefit arm has light first in episodes 15 to 29 at a mean of 0.955 (5th percentile 0.867) and decoy
first in episodes 45 to 59 at a mean of 0.947 (5th percentile 0.800). After the change, 4 of 5 episodes
in a row have the decoy first after a median of 5 episodes (95th percentile 9, never in 10 of 20000).
The food comes at most 50 steps after the choice, inside the 60 steps of the gate. The limit of check 2
changes from 30 to 50 steps, the wait plus the longest travel of 10.

## Checks before any training run, in this order. Each one can stop the gate.

1. **Code and tests.** The room flags and the gate option with tests: the hyperbolic outcome against its
   closed form, the forced episodes, the waits counted inside the radius, the size change, and a run with
   every new flag off. The control run, seed 73 and 10 episodes with the flags of the value gate gate,
   must give the same md5 of `metrics.csv` before and after the change.
2. **Offline diagnosis in training mode**, the mode of the live runs. The real room with the new flags,
   the real cochlea, the binder, the value gate and the motor map, no policy, 60 episodes of 60 steps.
   Room sequences 1311 to 1313 check the pipeline, and 1314 to 1316 are held out. The rule is written in
   the script before its first run. It holds when, at 3 of 3 held-out sequences, C1 and C2 below hold in
   the benefit arm, the forced object is first in at least 5 of the 6 forced episodes, and the median
   steps from the choice to the light's food in episodes 15 to 29 is no more than 50 (30 until 2026-11-11, for the wait of 20). If the agent does
   not stay inside the light's radius while it waits, the last check fails. Added 2026-11-11: in each
   forced episode the gate record must name the tone of the object that is present, at 6 of 6. The
   binder binds the two largest peaks of the band profile, and with one tone a second peak can get a
   group. The check also reports the miss rate and, for each miss, whether the agent reached the other
   object.
3. **Smoke run** in the training loop at the discarded seed 999, 8 episodes per arm. One choice and one
   update in each free episode, one candidate in each forced episode, and the ethics manifest records the
   drive off and no noxious channel.

### Check 1, result of 2026-11-13. It holds. No rule and no design value changed.

The room flags and the gate option exist and are off by default. Checks 2 and 3 and the gate have not
run. All runs used Python 3.12.10 with torch 2.14.1+cu126, the environment of the earlier gate runs.

| Item | Result |
|---|---|
| Tests written first | `tests/test_competing_goals_room.py` and `tests/test_competing_goals_gate.py`, 99 tests. Before the code 86 failed and 13 passed. The 13 test that the room and the gate with no new option are unchanged. After the code 99 pass |
| Hyperbolic outcome against its closed form | Holds at n 1, 3, 43 and 60, for the value after 1 to 20 episodes, and against `outcome` of the abstract model at n 1 to 61. The wired gate gives the same outcome for the reward that the training loop passes |
| Forced episodes, waits, size change | Hold in the room tests. The episode count is the count of `--dark-room-food-switch` |
| Every new flag off, in the room | The four digests of `tests/test_dark_room_test_bed.py`, recorded before this change, are unchanged |
| Control, seed 73, 10 episodes, the flags of the value gate gate with `--existence-drive off --phi-control off` | `metrics.csv` has md5 `29ae60f195d2da4dd9fec62d0169ea61` before (revision `46ef57f`, `runs/cg_c0_before_s73`) and after (`runs/cg_c0_after_s73`). `episodes.csv`, `value_gate.json`, `tectal_orienting.json` and `map_binding.json` are also the same. Each run has 2001 lines in `metrics.csv` and 10 gate episodes |
| Older control, seed 73, 10 episodes, Gate B7 flags with `--existence-drive on --phi-control on` | `metrics.csv` has md5 `7825766f42d3d85b032f8028c94dc1e0` after the change (`runs/cg_c0_b7_after_s73`), the md5 of the earlier controls |
| Public test suite | 2467 passed, 8 skipped |

Choices of the code that the rules above do not state, for the owner's review:

- The food comes at each step inside the radius after the wait, as the present food of the light does.
  With a wait of W, the first W steps inside give -0.01 and step W + 1 gives the food. So the food comes
  n = travel + W steps after the choice, as in the abstract model.
- In a forced episode the place of the absent object is still drawn. So every place is the place of a
  run with no forced episodes, and with `--dark-room-start-between` the agent starts midway between the
  present object and the empty place of the other one.
- `--value-delay-rate` has no default. It and `--value-outcome hyperbolic` need each other.
- With the hyperbolic outcome, `value_gate.json` also holds `outcome_rule`, `delay_rate` and, for each
  episode with food, `food_reward`. With the binary outcome the file has the keys it had.
- With a new room flag, the step facts hold `_truth_food_sizes`, `_truth_food_object` and
  `_truth_present`, for analysis only.

Added 2026-11-14 after a review of check 1. `session.json` of a run with no new flag had three new
keys. The run facts now hold a new key only when its flag is given. The control of seed 73 was run
again (`runs/cg_c0_after2_s73`). `metrics.csv` has md5 `29ae60f195d2da4dd9fec62d0169ea61` again, and
`session.json` has the keys of the run before the change. Two tests were added: the room that the
training script builds from the flags, and the forced episode without the light in the agent centred
view.

### Check 2, result of 2026-11-14. It holds on the held-out sequences. One defect is open.

`scripts/analysis/diagnose_competing_goals.py`, with its rule in the module text. The script and its
rule were pushed at revision `d897318` before its first run, and the script did not change after that.
The pipeline sequences ran once and the held-out sequences ran once. Outputs:
`runs/cg_check2/pipeline.json` (md5 `8af3e6271167c5bac3d07f3f9f14bff9`) and
`runs/cg_check2/held_out.json` (md5 `f2f826fb4e7f8176d140971743f1add3`). No weight was trained.

**On the pipeline sequences, which give no verdict, 1311 FAILED C2 and the forced tone criterion, and
1313 FAILED C1.**

| Sequence | Light first, 15 to 29 (rule 0.8 or more) | Decoy first, 45 to 59 (rule 0.8 or more) | Forced object first (rule 5 of 6 or more) | Forced tone named (rule 6 of 6) | Median steps to the light's food, 15 to 29 (rule 50 or less) | Forced episodes with two candidates | Misses of 54 choices |
|---|---|---|---|---|---|---|---|
| 1311, pipeline | 0.933 | 0.067 FAILED | 6 | 3 FAILED | 42.5 | 6 | 4 |
| 1312, pipeline | 1.000 | 1.000 | 6 | 6 | 42.0 | 5 | 1 |
| 1313, pipeline | 0.733 FAILED | 1.000 | 6 | 6 | 44.0 | 5 | 4 |
| 1314, held out | 1.000 | 1.000 | 6 | 6 | 45.0 | 6 | 0 |
| 1315, held out | 1.000 | 0.933 | 6 | 6 | 43.0 | 6 | 2 |
| 1316, held out | 1.000 | 0.867 | 6 | 6 | 44.0 | 6 | 5 |

The rule reads the held-out sequences only. Its verdict is HOLDS, with all five criteria at 3 of 3.

**The open defect is that the gate has two candidates in a forced episode.** The gate record had two
candidates in 34 of 36 forced episodes. The second candidate is band 1 when the light is present and
band 0 when the decoy is present, with a salience of 0.035 to 0.068. The measured tone bands are 18
for the light and 23 for the decoy. The binder takes the two largest peaks of the band profile, so
with one tone a small peak gets the second place, and its group was not empty.

- At sequence 1311 the gate chose band 0 in forced episode 1. The decoy's food went to the value of
  band 0, and band 0 won again in episodes 3 and 5. Band 23 never got a value. After the change the
  decoy was first in 1 of 15 episodes, and C2 failed.
- From the recorded priorities, the chance that the gate chooses the small band is 0.021 in forced
  episode 0 and 0.022 to 0.023 in forced episode 1, and below 0.001 in episodes 2 to 5 when the first
  two choices were right. That is about 0.043 for one run and about 0.12 for the three runs of the
  benefit arm. This is a calculation from the records. The observed count is 1 of 6 sequences.
- After a wrong first choice the small band won the next forced episodes of that object with a
  chance of 0.904 and 0.998 (sequence 1311).
- The forced tone criterion found the defect at 1311, as it was added to do. It does not prevent it.

Other reported numbers:

- Misses. 16 of 324 choices (0.049) in the free episodes of the six sequences, and the agent reached
  the other object in 16 of 16. The rate is below the 0.066 at which the owner decides. At 1313, 4 of
  the 15 episodes of the first window were misses, and that is why C1 failed there.
- The choice was at step 12 of the episode in 360 of 360 episodes. The light's food came 41 to 48
  steps after the choice.
- Two episodes (1311, episodes 26 and 27) ended after 60 steps with the choice still open at 48
  steps, so the gate gave 0. The offline episodes have 60 steps and the gate runs have 200.
- The food rewards in the gate records were 32.0, 3.0 and 4.0 and no other number. The offline
  reward is the room reward with no intrinsic bonus, so check 3 still has to show this in a run.

Limits of check 2. It has no policy, no reward predictor and no emotion, so it says nothing on the
effect of a reward of 32 on training. It ran the benefit arm only, as its rule says. Six sequences are
few, and the pipeline sequences were seen before the held-out ones ran.

The owner decides whether check 3 starts with this defect open, or the defect is closed first. To close
it is a change of the mechanism, and the held-out sequences 1314 to 1316 are used.

### Decisions of the owner, 2026-11-14, after check 2 and its review

A second review confirmed check 2 and the defect. The owner then decided the five points below. They
are written here before any code for them and before any new run.

**1. The gate gets a salience floor of 0.15.** `--value-salience-floor 0.15`, off by default. At the
choice, the gate ignores a candidate whose salience is below the floor. When every candidate is below
it, the gate makes no choice at that step and tries again at the next step. The gate record then holds
the candidates that the floor removed. The binder, the room and the motor map do not change. Both arms
of the gate run with the floor.

The value was set after the results of check 2 were seen. Its reason is the gap between the two
groups of salience at the choice, which is the only step at which the gate reads salience. 0.15 is a
round value near the midpoint of 0.0677 and 0.2832 in log scale, 0.138. A read of every saved gate
record, with no run, found no quieter tone below 0.15 in 925 choices, and none below 0.2. All of these
records have decoy levels 0.5 and 3.0, so the floor is checked for these levels only. Several folders
repeat one seed and one room, so the 925 choices are not all independent. Episodes count from 0.

| Source (`value_gate.json` of the run, or the rows of check 2) | Lowest salience of the quieter tone | Episode |
|---|---|---|
| `runs/vg_learn_s601` | 0.2833 | 32 |
| `runs/vg_learn_s602` | 0.3074 | 8 |
| `runs/vg_learn_s603` | 0.2963 | 20 |
| `runs/rv_stay_s701` | 0.3017 | 48 |
| `runs/rv_stay_s702` | 0.2950 | 36 |
| `runs/rv_stay_s703` | 0.2981 | 48 |
| `runs/rv_switch_s701` and `runs/drv_b_off_s701` | 0.3023 | 48 |
| `runs/rv_switch_s702` | 0.2953 | 52 |
| `runs/rv_switch_s703` | 0.2985 | 48 |
| `runs/drv_a_on_s999` and `runs/drv_a_off_s999` | 0.3621 | 3 |
| `runs/cg_c0_before_s73`, `cg_c0_after_s73`, `cg_c0_after2_s73`, `cg_c0_review_s73` | 0.3348 | 0 |
| `runs/_smoke_vg_learn` and `runs/_smoke_vg_frozen` | 0.3769 | 2 |
| `runs/_smoke_vg_scrambled` | 0.3288 | 2 |
| Check 2, sequence 1311 | 0.2884 | 16 |
| Check 2, sequence 1312 | 0.2874 | 54 |
| Check 2, sequence 1313 | 0.2832 | 28 |
| Check 2, sequence 1314 | 0.2996 | 30 |
| Check 2, sequence 1315 | 0.3064 | 42 |
| Check 2, sequence 1316 | 0.2924 | 18 |

The quieter tone was band 18 in every row but `drv_a_on_s999` and `drv_a_off_s999`, where it was band
23. The highest salience of the small band in a forced episode of check 2 was 0.0607 (1311, episode
3), 0.0640 (1312, episode 3), 0.0614 (1313, episode 1), 0.0591 (1314, episode 1), 0.0677 (1315,
episode 1) and 0.0622 (1316, episode 3).

Controls of the floor. The value gate control of seed 73 runs twice, with the floor off and with the
floor at 0.15. Both must give `metrics.csv` md5 `29ae60f195d2da4dd9fec62d0169ea61`.

The other repairs that the review listed were not chosen. A rule that makes a run with a wrong forced
choice invalid would need a new run in about 0.22 to 0.24 of the gates with two arms.

**2. A new check 2 replaces the check 2 above.** Sequences 1317 to 1319 check the pipeline, and 1320
to 1322 are held out. A search of every branch (`docs`, `scripts`, `tests`) and of the run folders
found no earlier use of 1317 to 1322. The episodes have 200 steps, the length of the gate runs,
because an episode of 60 steps ends 48 steps after a choice at step 12 and cannot show a food at
step 49 or 50. The five criteria stay, with the food steps criterion as corrected in point 3. Two
counts are added to the rule before its run, and each one above 0 stops the check at that sequence:

- the forced episodes in which the gate record has two candidates, and
- the free choices in which the floor removed a candidate.

A stop on a pipeline sequence means that the held-out sequences do not run.

**3. The food steps criterion is corrected to the text of check 2.** The script of 2026-11-14 took the
median over the episodes that the light fed, so an episode in which the agent left the light before
the food was dropped and could not fail the criterion. From now on the median is over the episodes 15
to 29 in which the light was the first object. An episode of these in which no food of the light
closed the choice counts as above every limit. The script is changed after its first run, and the old
rows are scored again under the corrected rule and reported.

**4. A stop on a value that is not finite, for check 3.** `--stop-on-non-finite`, off by default. The
run stops with an error when the reward that the loop passes to the action core is not finite, when a
loss that the policy update returns is not finite, or when a gradient is not finite at a step of one
of the optimizers of the training loop. With finite values it reads numbers only. The seed 73 control
runs once with this flag, and its `metrics.csv` must have the md5 above.

**5. A risk of the abstract model, stated before the gate.** The model draws misses at random. At
sequence 1313, 4 of the 15 episodes of one window were misses, and at a miss rate of 0.049 the chance
of 4 or more in 15 is 0.005. Misses follow the room layout. So the pass chance that the model gives
for C1 can be too high. No repair follows from this now.

### New check 2, result of 2026-11-14. It holds, with no stop. The miss rate is at the owner's limit.

The owner decisions above were pushed at revision `d82cd97` before any code. The floor and the stop on
a value that is not finite came next, with tests first (87 tests, 86 failed and 1 passed before the
code). The rule of the new check 2 was pushed in the script at revision `908a934` before its first
run, and the script did not change after that. Each set of sequences ran once. Outputs:
`runs/cg_check2/pipeline_v2.json` (md5 `4d66095ac555b81bee2e4fdf7ddc3e84`) and
`runs/cg_check2/held_out_v2.json` (md5 `98c8ae291c6296e3bb61195662e46120`). No weight was trained.

Controls of seed 73, 10 episodes, the flags of the value gate gate. `metrics.csv` has md5
`29ae60f195d2da4dd9fec62d0169ea61` in all three: the floor off (`runs/cg_c1_floor_off_s73`), the floor
at 0.15 (`runs/cg_c1_floor_on_s73`) and `--stop-on-non-finite` (`runs/cg_c1_finite_s73`). In the run
with the floor the gate record has an empty list of removed candidates in 10 of 10 choices.

| Sequence | Light first, 15 to 29 (rule 0.8 or more) | Decoy first, 45 to 59 (rule 0.8 or more) | Forced object first (rule 5 of 6 or more) | Forced tone named (rule 6 of 6) | Median steps to the light's food, 15 to 29 (rule 50 or less) | Forced episodes with two candidates (stop above 0) | Free choices with a removed candidate (stop above 0) | Misses of 54 choices |
|---|---|---|---|---|---|---|---|---|
| 1317, pipeline | 0.933 | 1.000 | 6 | 6 | 42.0 | 0 | 0 | 3 |
| 1318, pipeline | 0.867 | 0.933 | 6 | 6 | 44.0 | 0 | 0 | 5 |
| 1319, pipeline | 0.933 | 1.000 | 6 | 6 | 43.0 | 0 | 0 | 1 |
| 1320, held out | 0.933 | 0.800 | 6 | 6 | 43.0 | 0 | 0 | 4 |
| 1321, held out | 0.933 | 0.933 | 6 | 6 | 42.5 | 0 | 0 | 4 |
| 1322, held out | 0.933 | 0.933 | 6 | 6 | 42.0 | 0 | 0 | 4 |

The rule reads the held-out sequences only. No sequence stopped, and its verdict is HOLDS, with all
five criteria at 3 of 3. At sequence 1320 the decoy was first in 12 of 15 episodes, which is the limit
of C2 and not above it.

**The miss rate.** 21 of 324 choices (0.065) in the free episodes of the six sequences, and the agent
reached the other object in 21 of 21. The owner decides whether the gate runs when the rate is above
0.066, so this rate is below that limit by less than one miss in 324. The six sequences of the first
check 2 had 16 of 324 (0.049). Together that is 37 of 648 (0.057). The floor cannot change a miss,
because it removed no candidate at a free choice. In the abstract model the pass chance of the gate
is 0.798 at a reach of 0.934.

The floor.

- In 36 of 36 forced episodes the floor removed one candidate, at a salience of 0.0375 to 0.0626. The
  gate had one candidate in 36 of 36. Corrected 2026-11-15 after the review: the removed band was band
  0 in the 18 episodes with the decoy present, band 1 in 17 of the 18 episodes with the light present,
  and band 7 in one (sequence 1321, episode 2, salience 0.0377). The first text said band 1 for all 18.
  So the second peak is not always a band at the edge of the profile.
- In 324 of 324 free choices the gate had two candidates, bands 18 and 23, and the floor removed none.
  The lowest salience of the quieter tone was 0.2827 (sequence 1322, episode 42). That is below the
  0.2832 of the records read before the floor was set, and above the floor by a factor of 1.9.

Other reported numbers.

- The choice was at step 12 in 360 of 360 episodes. No choice was open at the end of an episode.
- The light's food came 41 to 50 steps after the choice. In each episode 15 to 29 with the light
  first, the light's food closed the choice.
- The food rewards in the gate records were 32.0, 3.0 and 4.0 and no other number.

The saved rows of the first check 2, scored again by the corrected food steps rule
(`--rescore`). One cell of its table changes: the median of sequence 1311 is 43.0, and it was 42.5,
because the two episodes that ended with the choice open now count. Every criterion of every sequence
has the result it had. The corrected rule is a median, so it fails when the light's food is missing in
more than half of the counted episodes, and not when it is missing in fewer.

Limits of the new check 2. It has no policy, no reward predictor and no emotion. It ran the benefit
arm only. The floor value was set after the first check 2 was seen. Six sequences are few.

The owner decides whether check 3 starts.

### Review of the new check 2, 2026-11-15, and the rule of check 3

A third review confirmed the new check 2. It ran the held-out sequences again and got the same file,
and it ran the three controls of seed 73 again with the same md5. Four points follow from it.

**Decision 5 above is not supported.** It rested on a chance of 0.005 for 4 misses in one window of
15, and that window was chosen after the data were seen. With one miss rate of 0.057, the chance that
one or more of the 24 scored windows of 12 sequences has 4 or more misses is 0.19 (the review's
number). The misses per sequence have the variance of one common rate. The review found that the miss
rate depends on the gap between the objects: 17 of 155 (0.110) at a gap below 130 room pixels, 10 of
355 (0.028) from 130 to 160, and 10 of 138 (0.072) at 160 or more. The room draws the gap anew in each
episode, so the misses of a run are independent draws, as the abstract model assumes. These numbers
are the review's, from layouts that it made again, and this document did not count them again.

**The miss rate limits C1 and C2.** In the six sequences of the new check 2 the gate chose the light
in 90 of 90 episodes of the window 15 to 29, and each of the 7 lost episodes of C1 was a miss. The
review's run of the abstract model gives a pass chance of 0.806 for the gate at a reach of 0.9352, the
rate of the new check 2, and 0.854 at 0.9429, the rate of both checks. The owner decides on this risk
before the gate.

**The stop rule of the new check 2 has one gap.** A step at which every candidate is below the floor
writes no gate record, so the count of removed candidates does not see a choice that the floor moved
to a later step. It did not occur: the choice was at step 12 in 360 of 360 episodes. The report of
check 3 and the probe of the gate read the step of each choice.

**Limits of `--stop-on-non-finite`.** It does not read the losses of the policy's replay update,
which become visible one policy update later, and it does not hook the optimizer of the sound surprise
model, whose loss uses sound only.

**The rule of check 3, written before its run.** Seed 999, 8 episodes of 200 steps for each arm, one
run at a time, with the flags of the Gate section, `--value-salience-floor 0.15` and
`--stop-on-non-finite`. With 8 episodes the size change at episode 30 is not reached, so episodes 0 to
5 are forced and 6 and 7 are free. `scripts/analysis/probe_competing_goals_smoke.py` reads the two
run folders. Check 3 holds when all of the following hold in both arms:

1. The run is complete: 8 rows in `episodes.csv`, 1600 rows in `metrics.csv`, and the recorded
   command has the floor and the stop.
2. The ethics manifest records the drive off and no noxious channel.
3. The gate record has 8 episodes, one for each episode of the run. The probe finds the step of each
   choice in the binder record, and the candidates and the removed bands of the gate record must be
   those of the binder record at that step.
4. In each forced episode the gate had one candidate, the band of the present object's tone (18 for
   the light, 23 for the decoy). In each free episode the floor removed no candidate.
5. Each choice has one update: the value after it is the value before plus 0.3 times (outcome minus
   the value before). In the benefit arm the outcome is the food reward divided by
   (1 + 0.1 (n - 1)), and in the binary arm it is 1 with food and 0 without.
6. Each food that closed a choice is 32, 3 or 4 in the reward that the gate read
   (`reward_for_action_core` of the step record), and it is the room reward of that step. A choice
   closed by a reward above 0.5 at a step where the room gave no food is a false food and fails.
7. Every reward and every recorded loss of every step is finite.

The report also gives, with no criterion: the step of each choice, the band and the salience of each
removed candidate, the food reward of each gate episode, the largest reward of each arm, the largest
reward at a step where the room gave no food, the largest reward prediction loss, and for each episode
the first object against the chosen one, so that a change of the reach would show. 8 episodes at one
seed are a smoke run. They show that the path runs, and they give no result on the trade-off.

### Check 3, result of 2026-11-15. It holds. The training values are far larger than in any earlier run.

The rule of check 3 was pushed at revision `c47c352` and the report script at `abc0fc7`, both before
the runs. The two runs ran at `abc0fc7` with no local change, one after the other, at seed 999 with 8
episodes of 200 steps: `runs/cg_smoke_benefit_s999` (`metrics.csv` md5
`035d06ed696caee3e5e0650d08c3e858`) and `runs/cg_smoke_binary_s999` (md5
`3225ca3904e1fbe1dbb9e0c847fddc4c`). Each has 1600 rows in `metrics.csv` and 8 in `episodes.csv`. The
report is `runs/cg_check2/smoke_report.json` (md5 `e16ad800d6eeec7e7b06747d5ef7c3b5`). The 36 tests of
the report script were written before it and were first run after it existed.

| Criterion | Benefit arm | Binary arm |
|---|---|---|
| Complete, with the floor and the stop in the recorded command | holds | holds |
| Ethics manifest: drive off, no noxious channel | holds | holds |
| 8 gate records, each equal to the binder record at the step of its choice | holds | holds |
| One candidate in each forced episode, the band of the present tone. No removed candidate in a free episode | holds, 6 of 6 and 2 of 2 | holds, 6 of 6 and 2 of 2 |
| One update for each choice, with the outcome of the arm | holds, 8 of 8 | holds, 8 of 8 |
| Each food that closed a choice is 32, 3 or 4 in the reward that the gate read. No false food | holds, 8 of 8 | holds, 8 of 8 |
| Every reward and every recorded loss is finite | holds | holds |

Verdict of the rule: HOLDS. Neither run stopped on a value that is not finite.

Reported numbers, the same in both arms where one number is given.

- The choice was at step 12 in 16 of 16 episodes.
- The floor removed one band in each forced episode: band 1 with the light present (salience 0.0386 to
  0.0427) and band 0 with the decoy present (0.0517 to 0.0634). It removed none in the 4 free
  episodes.
- The reward that the gate read at the food step was the room reward exactly, 32.0 or 3.0, in 16 of 16
  choices. The size 4 comes from episode 30 on and was not reached. The largest reward at a step where
  the room gave no food was -0.005, so curiosity did not come near the 0.5 of a food.
- The light's food closed the choice 41 to 45 steps after it (steps 52 to 56 of the episode).
- The first object was the chosen object in 16 of 16 episodes.
- Benefit arm, values at the end: 5.131 for band 18 and 1.874 for band 23. In its 2 free episodes the
  gate chose the light. Binary arm: 0.760 for each band, and the gate chose the decoy once and the
  light once. Two free episodes at one seed give no result on the trade-off.

**The size of the training values.** This is what check 3 was for, and the rule has no criterion for
it. The comparison is the value gate control of seed 73 with the floor (`runs/cg_c1_floor_on_s73`),
where the food is 1.0.

| Value, largest over a run | Control, food 1.0 | Smoke runs, food 32 and 3 |
|---|---|---|
| Reward of one step | 1.0 | 32.0 |
| Total reward of one episode | 187.9 | 4735.5 |
| Reward prediction loss | 1.0 | 1549.4 (benefit), 1529.6 (binary) |
| Gradient norm of the tectum | 5.5 | 15545.5, in episode 5 of both arms |
| Value estimate of the policy, absolute | 7.0 | 147.4 (benefit), 104.8 (binary) |

All 30378 recorded losses and gradient norms of each smoke run are finite. The gradient norm of the
tectum did not rise from episode to episode: its largest value per episode was 136, 37, 217, 3127,
1514, 15545, 340 and 2385 in the benefit arm. The agent stays at an object once the food comes, so an
episode at the light has about 145 steps with a reward of 32.

What this means, and what it does not. By a read of the code and with no ablation: with `--tectal-orienting-idle still` the movement comes from
the motor map, and the binder reads the picture and the cochlea envelope, so the weights that these
losses train do not reach the choice or the movement. The 16 of 16 for the first object agrees with
that. But 8 episodes do not show that the losses stay finite over the 60 episodes of a gate run. A
gate run with `--stop-on-non-finite` stops with an error in that case, and the rules must say before
the gate what a stopped run means. The broadcast, the emotion and every other recorded number of
these runs are trained on rewards 32 times as large as in any earlier run, so they cannot be compared
with earlier runs.

The owner decides whether the gate runs.

### Owner decisions of 2026-11-15 after check 3 and its review, and the rule of a long smoke run

A fourth review confirmed check 3. The owner decided the three points below. The gate does not start.

**1. The session record of the gate runs.** Drive C has 4.6 GB free (measured 2026-11-15). The session
recorder skips its heavy tiers (maps, full tensors, weights) below 20 GB free, and it wrote the light
tier only in both runs of check 3 (`meta.json` of each episode). The light tier is enough for the
gate. The probe of the gate reads `steps.jsonl` of each episode for the first object, and
`value_gate.json` and `map_binding.json` of the run. `runs/cg_smoke_benefit_s999`, 8 episodes, takes 49
MB, of which each episode folder is about 5.4 MB and `map_binding.json` is 5.6 MB. So one run of 60
episodes takes about 0.37 GB by this count, the long smoke run and the 6 gate runs about 2.6 GB, and
they fit on the free space. A gate run made with the light tier only cannot be replayed with maps on
the website, and no weights of it are saved.

**2. A run that stops.** Both arms of the gate run with `--stop-on-non-finite`. A run that stops on a
value that is not finite makes the gate VOID and not FAILED, because the trade-off was not tested. The
step of the stop is reported. A stopped seed is not run again. A gate after a change of the design
uses new seeds.

**3. A long smoke run before the gate.** Seed 999, 60 episodes of 200 steps, the benefit arm, the
flags of the Gate section with `--value-salience-floor 0.15` and `--stop-on-non-finite`. The rule,
written before the run and repeated in `scripts/analysis/probe_training_values.py`: the run holds when
`metrics.csv` has 12000 rows and every recorded reward, loss and gradient norm is finite. The report
gives the largest value of each kind for each block of 10 episodes. No number of the trade-off is read
from this run, and the script does not open the gate record. The seed 999 was used by check 3, so the
first 8 episodes of this run repeat `runs/cg_smoke_benefit_s999` if the run is deterministic.

### The long smoke run, result of 2026-11-15. It holds. No value was not finite in 60 episodes.

The rule and the report script were pushed at revision `326637b` before the run, and the run ran at
that revision with no local change. `runs/cg_long_benefit_s999`: seed 999, 60 episodes of 200 steps,
the benefit arm, with the floor and the stop. It took about 15 minutes. `metrics.csv` has 12000 rows
(md5 `85d577cf7193fb0c9b5a06fa31f2b72d`) and `episodes.csv` has 60. The report is
`runs/cg_check2/long_smoke_report.json` (md5 `e3c6c3fcb9d9544c856cb4ef68faa461`). The 13 tests of the
report script were written before it and were first run after it existed.

The run did not stop. Of 251978 recorded values in 12000 step records, none is not finite. The rule
holds.

The largest absolute value of each kind for each block of 10 episodes. Kinds that are 0 in every
block are left out (seven losses that this configuration does not use. Corrected 2026-11-15 after the
review, the first text said five).

| Kind | 0 to 9 | 10 to 19 | 20 to 29 | 30 to 39 | 40 to 49 | 50 to 59 |
|---|---|---|---|---|---|---|
| Reward of one step | 32 | 32 | 32 | 4 | 4 | 3 |
| Reward prediction loss | 1549 | 1342 | 940.5 | 940.5 | 9.616 | 7.479 |
| Total loss of the tectum | 1549 | 1342 | 940.5 | 940.4 | 9.540 | 7.421 |
| Topographic loss | 0.1869 | 0.09441 | 0.09001 | 0.09038 | 0.09051 | 0.0887 |
| Synchrony loss | 0.4196 | 0.4255 | 0.4073 | 0.4182 | 0.4600 | 0.4130 |
| Curiosity loss | 0.05013 | 0.02325 | 0.01092 | 0.03213 | 0.0002332 | 0.0002214 |
| Gradient norm, tectum | 15550 | 16070 | 945.9 | 4993 | 46.94 | 37.15 |
| Gradient norm, reward predictor | 5772 | 4656 | 3569 | 3575 | 60.59 | 24.83 |
| Gradient norm, policy, basal ganglia part | 30920 | 27520 | 51670 | 27660 | 561.4 | 208.4 |
| Gradient norm, policy, prefrontal part | 4708 | 4963 | 7479 | 4247 | 133.8 | 59.31 |
| Gradient norm, auditory specialist | 0.1201 | 0.08345 | 0.1072 | 0.2039 | 0.2015 | 0.2274 |
| Gradient norm, curiosity | 0.2488 | 0.2751 | 0.1205 | 0.4257 | 0.01188 | 0.01192 |
| Gradient norm, workspace binding | 6.851e-05 | 1.667e-13 | 1.464e-13 | 1.490e-13 | 1.724e-13 | 1.487e-13 |
| Value estimate of the policy | 150.8 | 156.6 | 207.1 | 144.3 | 22.81 | 19.08 |

What the table shows. Corrected 2026-11-15 after the review. The first text said that in the blocks
with a reward of 32 the largest values do not rise from block to block. That is wrong for four kinds
of the table. From the first block to the third, the largest gradient norm of the policy goes from
30920 to 51670 in the basal ganglia part and from 4708 to 7479 in the prefrontal part, and the largest
value estimate goes from 150.8 to 207.1. The largest gradient norm of the tectum rises from 15550 to
16070 in the second block and falls to 945.9 in the third. The largest reward prediction loss falls
from 1549 to 940.5. So the table shows no growth by a factor of 2 or more over the three blocks, and
it does not show that the values stopped rising. The review reports block medians that rise from the
first block to the second and not from the second to the third. Those medians are the reviewer's
numbers and were not computed here. After the sizes change at episode 30 the largest reward is 4, and the
losses and gradient norms of the last two blocks are smaller by a factor of about 100. The block 30 to
39 still has a loss of 940.5 with a largest reward of 4, so a prediction near 32 stayed for some steps
after the change.

The first 1600 rows of `metrics.csv` are equal, byte for byte, to `metrics.csv` of
`runs/cg_smoke_benefit_s999`. So the run repeats the smoke run of check 3 in its first 8 episodes.

The run folder takes 362 MB. The count before the run gave about 0.37 GB. Drive C has 4.4 GB free
after it, so the 6 gate runs need about 2.2 GB and fit.

Limits. One seed and one arm. The binary arm was not run for 60 episodes. In check 3 its values had
the size of the benefit arm. The stop does not read the replay update of the policy or the optimizer
of the sound surprise model. As the rule says, no number of the trade-off was read from this run: the
report script does not open the gate record, and this document does not read the reward row above as
a result on the choice.

Limits added 2026-11-15 after the review (issue 97).

- The report script reads one of the six rewards of a step record, `reward_for_action_core`. The rule
  says every recorded reward. A read of the saved run on 2026-11-15 found the six reward keys
  (`curiosity_score`, `delta_phi`, `env_reward`, `intrinsic_bonus`, `reward_for_action_core`,
  `reward_val`) and no value of them that is not finite. So the verdict does not change. The script is
  not changed, because it is the script of the rule that was pushed before the run.
- One recorded value rises through the run and is not in the table. `kl_div_pre_tanh`, largest value
  for each block of 10 episodes: 1.91e6, 1.12e7, 9.82e6, 1.44e7, 1.49e7, 1.48e7. It is finite. By the
  reviewer's read of the code it is the source of the vision bid and no update reads it. That read was
  not repeated here.
- The review also read every number of every step record. It found no value that is not finite other
  than `phi/value`, which this run does not compute (`--phi-control off`).
- Drive C had 3.6 GB free on 2026-11-15 after the review, not 4.4 GB. The 6 gate runs need about 2.2
  GB. The free space is read before each gate run.

The owner decides whether the gate runs.

### The self-test of the probe of the gate, result of 2026-11-15. It holds.

The rule of the probe, the probe, the second count and their 34 tests were pushed at revision
`3522004` before the self-test. The tests were written before the two scripts and were first run after
the scripts existed. The self-test ran at that revision with no local change, on
`runs/cg_smoke_benefit_s999` and `runs/cg_smoke_binary_s999`. The report is
`runs/cg_check2/gate_probe_self_test.json` (md5 `1038367952ba84192c364be5c18b4791`).

The first count gives the object of the gate's choice in 16 of 16 episodes. The second count gives it
in 16 of 16 episodes. The first object that gave food is also that object in 16 of 16. The self-test
holds. In its gate mode the probe refused the long smoke run with the message that seed 999 is not a
seed of the gate. It prints no number before that refusal. The public suite gives 2671 passed and 8
skipped at `3522004`.

Limit. The two smoke runs have 6 forced episodes of 8, and no episode in which the two counts could
differ by the tie rule was looked for. The self-test shows that both counts read the run folders and
agree with the gate record there. It does not show that the counts agree in a gate run.

No gate run exists. The owner decides whether the gate runs.

## Gate

Seeds 1301, 1302 and 1303, which no run has used. 60 episodes of 200 steps, one run at a time. The flags
of the value gate gate (Gate B7 flags with `--dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0
--dark-room-decoy-visible --dark-room-start-between --dark-room-pulse --dark-room-sound-direction off
--tectal-orienting value --tectal-orienting-idle still --map-binding rhythm`), with `--existence-drive off`
and `--phi-control off`, and the room flags `--dark-room-forced 6 --dark-room-food-sizes 32,3
--dark-room-food-waits 40,0 --dark-room-food-change 30 --dark-room-food-sizes-after 4,3`.
Added 2026-11-14, owner decision: both arms run with `--value-salience-floor 0.15`.
Added 2026-11-15, owner decision: both arms run with `--stop-on-non-finite`. A run that stops on a
value that is not finite makes the gate VOID and not FAILED, and the step of the stop is reported. A
stopped seed is not run again, and a gate after a change of the design uses new seeds.

| Arm | Gate flags |
|---|---|
| benefit | `--value-outcome hyperbolic --value-delay-rate 0.1 --value-salience-floor 0.15 --stop-on-non-finite` |
| binary | `--value-salience-floor 0.15 --stop-on-non-finite`, and the binary outcome of the present gate |

First object: within the first 60 steps of an episode, the first object closer than 60 room pixels, the
nearer one on a tie, as in `scripts/analysis/probe_reversal.py`. No contact counts as neither. Read by a
new probe, with a second count from separate code. In the saved runs a miss was never no contact. The
agent reached the other object, so a miss counts as the other object first.

Added 2026-11-15, owner decision after the review of the long smoke run (issue 97). These points are
written before the probe of the gate reads a gate run.

- **A run that ends early.** A gate run that does not have 12000 rows in `metrics.csv` makes the gate
  VOID, whatever the cause (a stop on a value that is not finite, a crash, a full disk). The number of
  rows is reported. The seed rule is that of a stopped run. The seed is not run again, and a gate
  after a change of the design uses new seeds.
- **A run folder that the probe cannot read.** Added 2026-11-15, owner decision after the review of
  the probe (issue 98), before the first gate run. If the probe cannot read a run folder of the gate,
  the gate is VOID. The probe is not changed.
- **The first count.** `first_object` of `scripts/analysis/probe_reversal.py`, called by the probe of
  the gate, `scripts/analysis/probe_competing_goals.py`.
- **The second count.** The same definition (the first object closer than 60 room pixels in the first
  60 steps, the nearer one on a tie), in separate code, `scripts/analysis/count_first_object.py`. That
  file imports neither `probe_reversal.py` nor the probe of the gate. It reads `_truth_light_offset`
  and `_truth_decoy_offset` of the step records.
- **A difference between the two counts.** A difference in one or more episodes of a gate run makes
  the gate VOID until the difference is explained. The verdict is never taken from one count alone.
- **Limit of the second count.** It finds an error of the code of a count. It does not find an error
  of the recorded offsets. No second field of a step record has the place of the decoy.
- **Cross-check, in the report only, with no effect on the verdict.** For each run, the count of
  episodes in which the first object differs from the object of the gate's choice, and the count in
  which it differs from the first object that gave food (`_truth_food_object`).
- **The denominator.** Each window has 15 episodes. Neither stays in the denominator. So 0.8 needs 12
  of 15.
- **C3.** It is computed for each seed, with the benefit arm and the binary arm of that seed. With 15
  episodes, 0.3 or more is a difference of 5 episodes or more.
- **The order of the verdict.** VOID first, and no criterion is computed for a VOID gate. If the gate
  is not VOID, it HOLDS when C1 to C3 hold at 3 of 3 seeds, and it is FAILED if not.
- **Self-test of the probe, before the gate.** On `runs/cg_smoke_benefit_s999` and
  `runs/cg_smoke_binary_s999`, each count must give the first object equal to the object of the
  gate's choice in 16 of 16 episodes. The probe is not run on the long smoke run, and it refuses a
  seed that is not 1301, 1302 or 1303 in its gate mode.

| Check | Rule, at 3 of 3 seeds |
|---|---|
| C1, the larger food is worth its wait | benefit arm, episodes 15 to 29: light first 0.8 or more |
| C2, the same wait is not worth a smaller food | benefit arm, episodes 45 to 59: decoy first 0.8 or more |
| C3, the switch needs size and delay in the value | (decoy first in 45 to 59) minus (decoy first in 15 to 29) is larger in the benefit arm than in the binary arm by 0.3 or more |

What each result means, stated now:

- C1 to C3 all hold. The agent's choice between two foods follows their sizes and the wait together, and
  it changes through learning when the benefit changes. AE-1 then has a competing goals result, and the
  score is the owner's decision.
- C1 or C2 fails. The loop does not weigh size against delay in this room. No score changes, and the
  FAILED result is published.
- C3 fails. The change of choice does not need the size and delay in the value, so no claim about the
  trade-off follows.

Time: 6 runs of about 25 minutes each, about 2.5 hours, plus the checks before. Measured 2026-11-15:
the long smoke run took about 15 minutes, so 6 runs take about 1.5 hours.

## Gate, result of 2026-11-15. It HOLDS at 3 of 3 seeds. The score of AE-1 is the owner's decision.

The owner started the gate after the review of the probe (issue 98). The sentence on a run folder that
the probe cannot read was pushed at revision `054e416` before the first run. The six runs ran at that
revision with no local change, one at a time, in the order benefit then binary for each seed. Each
`ethics_manifest.json` records `git_commit` `054e416`, no tracked change, the drive off and no noxious
channel, and each `git_diff.patch` has 0 bytes. The free space of drive C was read before each run
(9.4 GB before the first, 7.2 GB before the sixth). Each run was checked for 12001 lines in
`metrics.csv` before the next one started. The probe ran once, after the sixth run. The report is
`runs/cg_check2/gate_report.json` (md5 `1b1ff065dcbae6682856122cd595bef7`).

| Run | `metrics.csv` md5 |
|---|---|
| `runs/cg_gate_benefit_s1301` | `b76f3adeb203299b7063566167b323bd` |
| `runs/cg_gate_binary_s1301` | `f28eb6ce3eea2277594ed8ff5e9f2772` |
| `runs/cg_gate_benefit_s1302` | `1ede04f2f119271b49b38654bba19e40` |
| `runs/cg_gate_binary_s1302` | `5abba4a88911d6038eecc59c24017901` |
| `runs/cg_gate_benefit_s1303` | `6739677ce66f531dd0f34a1334ec68dd` |
| `runs/cg_gate_binary_s1303` | `c7ef2e285538ac82dff768a078304f8f` |

The gate is not VOID. Each run has 12000 rows and 60 episode folders. The two counts of the first
object differ in 0 episodes of each run. No run stopped.

First object in each window of 15 episodes, as light, decoy, neither.

| Arm | Seed | Episodes 15 to 29 | Episodes 45 to 59 | Change of decoy first |
|---|---|---|---|---|
| benefit | 1301 | 15, 0, 0 | 1, 14, 0 | 14 |
| benefit | 1302 | 14, 1, 0 | 0, 15, 0 | 14 |
| benefit | 1303 | 14, 1, 0 | 2, 13, 0 | 12 |
| binary | 1301 | 8, 7, 0 | 8, 7, 0 | 0 |
| binary | 1302 | 9, 6, 0 | 7, 8, 0 | 2 |
| binary | 1303 | 7, 8, 0 | 8, 7, 0 | -1 |

| Check | Seed 1301 | Seed 1302 | Seed 1303 | Rule |
|---|---|---|---|---|
| C1, light first in 15 to 29, benefit arm | 15 of 15 | 14 of 15 | 14 of 15 | 12 or more |
| C2, decoy first in 45 to 59, benefit arm | 14 of 15 | 15 of 15 | 13 of 15 | 12 or more |
| C3, change of the benefit arm minus change of the binary arm | 14 | 12 | 13 | 5 or more |

C1, C2 and C3 hold at each of the 3 seeds. The verdict of the probe is HOLDS.

What the result means, by the text that was written before the runs. The agent's choice between two
foods follows their sizes and the wait together, and it changes through learning when the benefit
changes. AE-1 has a competing goals result. The score is the owner's decision, and this document
changes no score.

Cross-checks, in the report only, with no effect on the verdict. Count of the 60 episodes of a run in
which the first object differs from the object of the gate's choice: benefit arm 2, 1, 4 and binary
arm 0, 3, 4 (seeds 1301, 1302, 1303). Count in which it differs from the first object that gave food:
benefit arm 0, 0, 2 and binary arm 0, 0, 2. Each run has 60 gate records, so the pairs of the
cross-check do not shift. No episode of the six runs has neither as its first object.

The learned values at the end of the run (`value_gate.json`, tone band 18 is the light and 23 is the
decoy): benefit arm 1.6577 and 2.1612, 1.3656 and 2.5912, 1.3201 and 2.3553. Binary arm 1.0 and 1.0 at
each seed.

A review can verify the room of the runs. `config.json` of each gate run differs from `config.json` of
`runs/cg_smoke_benefit_s999` in `episodes` and `seed` only. `config.json` does not hold the arm flags.
The arm is in the recorded command of `session.json`, which the probe checked, and in `outcome_rule`
of `value_gate.json`.

Limits of the result.

- Three seeds and one room. The limits that were stated before the runs stand, and they follow below.
- The binary arm chose near the middle in both windows (6 to 8 decoy first of 15). So C3 shows that the
  switch needs the size and the delay in the value. It does not show which of the two the switch needs,
  because the benefit arm changes both against the binary arm.
- The first object differs from the gate's choice in 14 of 360 episodes (0.039). The offline miss rate
  of the new check 2 was 0.065.
- The gate tests AE-1 only. The workspace does not take part, so nothing follows on the workspace link
  or on a GWT indicator.
- The second count finds a code error of a count. It does not find an error of the recorded offsets.
- No rubric, badge, README or website text was changed. A change of the public claim needs the owner's
  decision on the score first.

## Limits, stated before the runs

- The delay rate k and the food sizes are set by this design. The gate tests that the agent's own loop
  applies a trade-off of size against delay and changes its choice when the benefit changes. It does not
  test that the agent finds a delay rate of its own.
- Both goals are foods. Goals of different kinds, such as food against another need, are not tested,
  because the drive is off and no noxious signal is allowed.
- This is not a sentience marker of the kind that Brown and Birch (2025) discuss. Those trade-offs weigh
  reward against injury risk, and this gate has no injury and no noxious signal on purpose.
- Only the benefit changes in the gate. A change of the cost alone is in the model and not in the gate.
- The light is always the larger food. Which object is louder changes every episode, but the identity of
  the large food is not counterbalanced.
- The forced episodes give the gate experience of both foods, as bees in a delay test were first trained
  on each flower. Without them the model failed.
- The movement comes from the motor map. The workspace does not take part, so no GWT indicator is tested.
  The policy's movement network is not used.
- If the agent behaves as the model, the gate fails about once in five times. A failed gate is reported
  as FAILED all the same. The corrected model of 2026-11-11 gives a pass chance of 0.929 for the
  chosen design at a reach of 115 of 120, about one failure in fourteen, and 0.798 at a reach of 0.934,
  about one in five.
- Food sizes of 32, 4 and 3 are room units (24, 4 and 3 until 2026-11-11). The `reward` column of `metrics.csv` will show them.
- Added 2026-11-11. The gate adds salience to 2 × value. A food of 32 in place of 1 gives the value term
  the same weight as a value weight 32 times as large would give with a food of 1. So the statement that
  the gate's constants did not change holds for the code and not for the balance between salience and
  value. The binary arm keeps outcomes of 0 and 1, so the two arms differ in this balance as well as in
  the outcome rule.
- Added 2026-11-11. The room reward is also the target of the reward predictor that trains the tectum
  (`scripts/training/train_rlhf.py`). A reward of 32 changes that loss. The binder reads the picture and
  the cochlea envelope, by a read of the code that was not checked in a run. The smoke run reports the
  reach so that a change would show.
- Added 2026-11-11. Reach was 206 of 220 when the chosen tone was the quieter one and 253 of 260 when it
  was the louder one. No test of this difference was made. After the change the decoy must be chosen in
  every episode, and its tone is the quieter one in half of them.
