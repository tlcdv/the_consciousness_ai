# FAILED: the broadcast does not reach the action, and the task link of the dark room gates is a room to bid link

Date: 2026-11-11. Read-only checks on saved runs. No run was trained, no weight changed, and no default
changed. The probe is `scripts/analysis/probe_workspace_action.py`. The rules of checks A to C are in its
docstring and were written before its first run. The rules of check D were written after checks A to C
were read and before the first run of check D. The outputs are in `docs/results/data/workspace_action_2026_11/`.

No indicator moves. The rubric and the clock stay as they are. Nothing here is a claim about consciousness.

## Verdict, failure first

1. **FAILED. The broadcast does not reach the action in the saved gate runs.** In 11 of 12 replayed runs
   of Gates B2, B3, B4 and B7 the action mean of the policy does not depend on the recorded broadcast.
   One run is undecided. In the runs with `--tectal-orienting value` the code replaces the action of the
   policy, as earlier documents state.
2. **The task criterion of Gates B2 to B8 was never a test of behaviour.** It compares the share of steps
   with the light out of view with the share of ignited steps where hearing has the highest bid. That is
   a link from the room to the bids. With a policy that ignores its input, the link cannot run from the
   workspace to the movement.
3. **The episode index does not explain the task link in Gates B3, B4, B6 and B7.** The link holds with
   the index removed and it holds inside single episodes. Gate B5 is undecided. Gate B2 has no link.
4. **The gain weighting produces the link inside the step, at the gates where the check is valid.** With
   the weighting of `--bid-precision gain` divided out of the recorded bids, the difference inside the
   episode falls from 0.280 to -0.205 in Gate B7 (10 seeds). The reading is SUPPORTED at 3 of 6 gates and
   VOID at 3, where control K4 fails.

## What was tried before

| Attempt | What it measured | What it excluded | Source |
|---|---|---|---|
| Gates B2 to B8, criterion 4 | Per episode, the audio share of ignited steps against the share of steps with the light out of view | Nothing about behaviour. The index is "reported only" | `dark_room_senses_2026_09.md`, sections Gate B2 to Gate B8; `scripts/analysis/probe_gate_b2.py`, rule (4) |
| Broadcast carrier | Which vector the policy receives | With the default merge the policy receives the vector of the recorded winner at 0.025 to 0.095 of ignited steps | `broadcast_carrier_2026_10.md` |
| Relay defect | Policy replay at seed 194, and a scan of saved weights | The relay layer changed in 0 of 172 runs. The output is close to tanh(relay bias) | `action_core_relay_defect_2026_10.md` |
| Value gate and reversal | Learning through the selection of the tectum | The workspace leader changed at 0.0000 of the steps | `value_gate_2026_10.md` |
| Rubric check of GWT-3 | A perturbed broadcast, read at the next step | 0 of 38400 readings changed. The policy mean stays below its noise | `rubric_recheck_2026_11.md` |
| Drive off | Identity of the movement records | The existence drive does not reach the movement | `reversal_drive_off_2026_11.md` |

## The path from the broadcast to the room

| Point where the signal can stop | Code |
|---|---|
| 1. The default merge keeps the vector of the weakest admitted module | `models/core/global_workspace.py`, the `broadcast_merge` block |
| 2. The policy reads the broadcast only with `--policy-input broadcast` | `scripts/training/train_rlhf.py`, `policy_state` in `run_episode` |
| 3. The relay layer of the policy is in no loss term | `models/self_model/action_selection_core.py`, `BasalGanglia.thalamic_relay` |
| 4. Exploration noise is added to the action mean | `ActionSelectionCore.select_action` |
| 5. The motor map replaces the action of the policy, or sets it to zero | `_wire_tectal_orienting` and `_wire_value_gate` in `train_rlhf.py` |
| 6. The room receives the action | `env.step(env_action)` in `run_episode` |

## Check A. Which runs used which path

Read from the `session.json` of each run. 200 runs have one.

| Path | Runs |
|---|---|
| `--tectal-orienting off`, the action of the policy reaches the room | 159 dark room runs (all the Gate B runs) and 4 DMTS runs |
| `value`, idle `still` | 13 |
| `group`, covert, the action is zero | 8 |
| `group`, `active` or `audio`, overt | 16 |

Every run used `--policy gonogo` and `--policy-input broadcast`. 28 of the 159 runs used
`--broadcast-merge top_winner`.

## Check B. Policy replay on the recorded broadcast

The policy is rebuilt from the weights of the last recorded episode and fed the recorded policy input of
that episode. The ratio is the largest spread of the replayed action mean over the steps, divided by the
smallest spread of the recorded action minus that mean. The rule gives SUPPORTED at 0.5 or more and
EXCLUDED at 0.1 or less.

| Run | Action mean | Mean for an input of zeros | tanh(relay bias) | Ratio | Label |
|---|---|---|---|---|---|
| `gate_b2_s48` | (-0.2093, 0.5538) | (-0.2093, 0.5538) | (-0.1984, 0.5532) | 0.0201 | EXCLUDED |
| `gate_b2_s49` | (-0.5393, -0.2798) | (-0.5367, -0.2791) | (-0.5482, -0.2825) | 0.0154 | EXCLUDED |
| `gate_b2_s50` | (-0.1377, -0.4317) | (-0.1345, -0.4326) | (-0.1718, -0.4112) | 0.0177 | EXCLUDED |
| `gate_b3_s54` | (0.4819, 0.4655) | (0.4831, 0.4676) | (0.5107, 0.4946) | 0.0103 | EXCLUDED |
| `gate_b3_s55` | (0.2768, 0.2306) | (0.2809, 0.2072) | (0.2914, 0.1816) | 0.0579 | EXCLUDED |
| `gate_b3_s56` | (-0.4907, 0.6171) | (-0.4916, 0.6095) | (-0.5103, 0.5877) | 0.0156 | EXCLUDED |
| `gate_b4_s57` | (0.1645, -0.5832) | (0.1481, -0.5784) | (0.1080, -0.5663) | 0.0499 | EXCLUDED |
| `gate_b4_s58` | (0.4509, -0.2365) | (0.4304, -0.2524) | (0.4396, -0.2468) | 0.0728 | EXCLUDED |
| `gate_b4_s59` | (0.5203, 0.3081) | (0.5174, 0.3001) | (0.5153, 0.3073) | 0.0368 | EXCLUDED |
| `gate_b7_s73` | (-0.1808, 0.5356) | (-0.2226, 0.5470) | (-0.2249, 0.5448) | 0.1861 | UNDECIDED |
| `gate_b7_s74` | (0.1125, -0.0513) | (0.1102, -0.0551) | (0.1041, -0.0652) | 0.0485 | EXCLUDED |
| `gate_b7_s75` | (-0.2510, -0.1651) | (-0.2496, -0.1638) | (-0.2705, -0.1837) | 0.0072 | EXCLUDED |

Control K1 holds at 12 of 12 runs. The recorded policy input is equal to the recorded broadcast, with a
largest difference of 0. The replay with the inputs in a random order gives the same mean to the third
decimal place at every run. The spread of the recorded action around the mean is 0.103 to 0.156, which
is the exploration noise.

The direction of movement in these runs is the relay bias of the seed. So the share of steps with the
light out of view, which is the f of the task criterion, comes from the start position, the layout and
that fixed drift.

## Check C. The task link against the episode index

Control K3 holds. The probe gives the pooled rho of the result document at all 7 gates on disk, and the
index rho of Gate B4 (-0.194 / 0.512 / 0.815).

C1 removes the index. Inside each seed, the ranks of both shares are replaced by their residuals on the
rank of the episode number. C2 holds the index. It compares, inside one episode, the audio share of
ignited steps out of view and in view, against in-view labels rotated inside the episode.

| Gate | Raw pooled rho | C1, r with the index removed (p) | Seeds above 0 | C2, mean difference inside the episode (p) | Episodes, above 0 | Reading by the rule |
|---|---|---|---|---|---|---|
| B2 | -0.128 | 0.024 (0.462) | 2 of 3 | -0.265 (0.9995) | 13, 3 | No link is present |
| B3 | 0.446 | 0.538 (0.001) | 3 of 3 | 0.136 (0.019) | 11, 6 | Index EXCLUDED |
| B4 | 0.458 | 0.424 (0.013) | 3 of 3 | 0.337 (0.0005) | 11, 6 | Index EXCLUDED |
| B4, layouts moved | 0.289 | 0.397 (0.019) | 3 of 3 | 0.209 (0.0005) | 9, 5 | Index EXCLUDED |
| B5 | 0.200 | 0.203 (0.149) | 2 of 3 | 0.305 (0.0005) | 12, 6 | UNDECIDED |
| B6 | 0.362 | 0.321 (0.002) | 9 of 10 | 0.209 (0.0005) | 37, 21 | Index EXCLUDED |
| B7 | 0.388 | 0.362 (0.0002) | 8 of 10 | 0.190 (0.0005) | 38, 24 | Index EXCLUDED |

The rule gives "index SUPPORTED" for Gate B2, because both p values are 0.05 or more. That label does not
fit. Gate B2 has no positive link for the index to explain, so the table says so.

C3, reported only. Gate B4 seeds 57 to 59 against the same seeds with the layouts moved by one episode,
30 pairs of the same seed and episode number. The change of the audio share follows the change of the
out-of-view share, rho 0.487, one-sided p 0.003.

The 10 Gate B8 run folders named in the result document are not on disk, so Gate B8 is not checked.

## Check D. The link with the gain weighting removed from the step

With `--bid-precision gain` each sense bid is multiplied by twice its share of the total gain
(`models/core/precision_weighting.py`). The vision gain is the strongest pooled cell of the frame outside
the agent's own mark, so a light in view raises it. The gains are not recorded. The probe computes them
again from the recorded frame and waveform with the functions of that module, divides them out of the
recorded bids, and repeats C2 on the order of the two raw bids.

| Gate | Recoverable steps (K5, rule 0.8) | Raw order names the winner (K4, rule 0.9) | Vision share of the gain, in view / out of view | Difference with the weighting (p) | Difference without it (p) | Episodes | Reading |
|---|---|---|---|---|---|---|---|
| B3 | 0.910 | **0.882** | 0.651 / 0.461 | 0.138 (0.043) | -0.361 (1.0) | 9 | VOID |
| B4 | 0.840 | **0.865** | 0.651 / 0.463 | 0.602 (0.0005) | 0.137 (0.067) | 7 | VOID |
| B4, layouts moved | 0.826 | 0.911 | 0.652 / 0.462 | 0.300 (0.005) | -0.195 (0.949) | 6 | SUPPORTED |
| B5 | 0.849 | 0.927 | 0.636 / 0.448 | 0.526 (0.0005) | 0.138 (0.083) | 5 | SUPPORTED |
| B6 | 0.844 | **0.873** | 0.655 / 0.455 | 0.261 (0.0005) | -0.200 (1.0) | 28 | VOID |
| B7 | 0.853 | 0.968 | 0.645 / 0.457 | 0.280 (0.0005) | -0.205 (1.0) | 27 | SUPPORTED |

Bold marks a failed control. The three VOID gates are B3, B4 and B6. The probe does not replay the learned
valence, the bound bids or the settle cycles, so the order of the raw bids names the recorded winner at
fewer than 0.9 of their steps. Their numbers are in the table as a report and carry no reading.

Gate B7 is the largest valid set, with 10 seeds and no learned valence. There the difference is 0.280
with the weighting and -0.205 without it. In Gate B4 with moved layouts and in Gate B5 the difference
without the weighting is not above its null. Gate B2 ran with no weighting, and its difference inside
the episode is -0.265 (check C). That agrees in sign, on other seeds, and is not an ablation.

## What this shows

- In the gate runs the policy output is a constant of the seed plus noise. No content of the workspace
  changes the movement.
- The task link of the gates with `--bid-precision gain` is present with the episode index removed, and
  inside single episodes.
- At the gates where the check is valid, that link is the gain weighting acting on the bids in the same
  step. A light in view raises the vision share of the gain from about 0.45 to about 0.65.

## What this does not show

- It does not show what the agent would do with a repaired policy, another merge, or no motor map.
- Check B replays one episode per run with the weights of its end, and starts the working memory of the
  policy at zero. The live loop never resets that memory.
- Check C is a correlation. C2 is above 0 in 21 of 37 episodes in Gate B6 and 24 of 38 in Gate B7, so a
  part of the episodes carries the mean.
- Check D removes the weighting from one step. The weights of each run learned with the weighting in
  place, and the ignition is the recorded one. A run with the weighting off at the same seeds was not
  made.
- A separate trend with the episode number is not excluded. The audio share rises with the episode
  number at 10 of 10 seeds in Gate B7. Its cause was not tested.
- The rules of check D were written after checks A to C were read.
- No biology source was read for this document. It proposes no mechanism.

## Open decisions for the owner

1. The text of the task criterion in `dark_room_senses_2026_09.md` and in later gates. It reads as a test
   of the task. It measures the room acting on the bids.
2. A path from the workspace to behaviour has three cuts to close. They are the merge, the relay layer
   and the replacement by the motor map. The ratio of check B can be the first check of any repair, because it
   can fail.
3. A live test with the weighting off at the Gate B7 seeds would be the ablation for check D. By the run
   times in the gate document, the cost is about 3.5 minutes per seed.

## Reproduce

```
python -m scripts.analysis.probe_workspace_action --runs-root runs --checks a b c d
python -m pytest tests/test_workspace_action_probe.py -q
```
