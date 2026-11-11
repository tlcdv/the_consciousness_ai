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

## Gate

Seeds 1301, 1302 and 1303, which no run has used. 60 episodes of 200 steps, one run at a time. The flags
of the value gate gate (Gate B7 flags with `--dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0
--dark-room-decoy-visible --dark-room-start-between --dark-room-pulse --dark-room-sound-direction off
--tectal-orienting value --tectal-orienting-idle still --map-binding rhythm`), with `--existence-drive off`
and `--phi-control off`, and the room flags `--dark-room-forced 6 --dark-room-food-sizes 32,3
--dark-room-food-waits 40,0 --dark-room-food-change 30 --dark-room-food-sizes-after 4,3`.

| Arm | Gate flags |
|---|---|
| benefit | `--value-outcome hyperbolic --value-delay-rate 0.1` |
| binary | none, the present gate |

First object: within the first 60 steps of an episode, the first object closer than 60 room pixels, the
nearer one on a tie, as in `scripts/analysis/probe_reversal.py`. No contact counts as neither. Read by a
new probe, with a second count from separate code. In the saved runs a miss was never no contact. The
agent reached the other object, so a miss counts as the other object first.

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

Time: 6 runs of about 25 minutes each, about 2.5 hours, plus the checks before.

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
