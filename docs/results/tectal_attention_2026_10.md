# The tectal attention gain: FAILED, A1 to A3 at 3 of 3 seeds

Date: runs of 2026-10-17, written 2026-10-31. Rules: `docs/decisions/2026_10_17_tectal_attention.md`,
pushed before the runs (commit ee5d696).

## Verdict

FAILED. A gain of 1.5 on the tectum's own map, at the cells that the binding selected, did not bring
the direction of the selected object into the workspace content or the broadcast. The arm with the
gain and the arm without it gave the same scores. The gain was present in the map. GWT-2 stays
PARTIAL.

The offline check before the runs was wrong too. It passed in evaluation mode, and the live runs train.
That was my error. In training mode the same check gives about 0 (table below).

## Checks

Score: the mean cosine between the decoded and the true direction of the selected object, chance 0,
with whole episodes held out.

| Check | Rule | 801 | 802 | 803 |
|---|---|---|---|---|
| A1 the content carries the selection | link arm, content ≥ 0.5 | 0.019 | 0.022 | -0.086 |
| A2 the broadcast carries it | link arm, broadcast at ignited steps ≥ 0.5 | 0.039 | 0.066 | -0.100 |
| A3 the link causes it | link above no link for both vectors, both no-link scores ≤ 0.2, permutation p < 0.05 for both link scores | false | false | false |

| Readings for A3 | 801 | 802 | 803 |
|---|---|---|---|
| No link, content | 0.017 | 0.021 | -0.087 |
| No link, broadcast | 0.040 | 0.058 | -0.120 |
| Larger permutation p of the two link scores | 1.0000 | 1.0000 | 0.7811 |
| Counted steps, link arm, content and broadcast | 7293, 5780 | 7480, 6039 | 7480, 6115 |

The gate probe, run again on 2026-10-31, gave the same content in its output file. A second count with
separate code (a closed form ridge in numpy, its own folds and labels) gave the same scores to 6
decimals in both arms at all three seeds.

## Why the offline check passed and the gate failed

After the data, reported only.

The rules report an offline check that passed, with scores of 0.850 to 0.977 at gain 1.5. That check
set the tectum to evaluation mode. In evaluation mode the continuous latent is its mean, and in
training mode it is a sample (`_reparam`, `models/core/sensory_tectum.py`). In the tectum file and the
capsule file, `_reparam` is the only step that depends on the mode. `scripts/training/train_rlhf.py`
does not set the tectum to evaluation mode, so the live runs use the sample.

The same offline check with the same weights, in both modes (room seed 521):

| Weights | Mode | No gain | Gain 1.5 |
|---|---|---|---|
| `vg_learn_s601`, episode 39 | evaluation, as the check ran | -0.132 | 0.850 |
| `vg_learn_s601`, episode 39 | training | -0.060 | -0.015 |
| `ta_nolink_s801`, episode 39 | evaluation | -0.208 | 0.959 |
| `ta_nolink_s801`, episode 39 | training | -0.020 | -0.018 |

The first row repeats the values in the rules, so the recovered script is the same check.

The size of the effect, with the fixed weights of the no-link arm at episode 39 and identical noise
draws in the compared passes (room seed 521, 2 episodes, steps after 15). Each value is a mean relative
change of `workspace_content`.

| Mode | Weights | From the gain | From one step to the next |
|---|---|---|---|
| Evaluation | 801 | 1.26e-06 | 5.79e-08 |
| Evaluation | 802 | 5.59e-07 | 2.68e-08 |
| Evaluation | 803 | 6.53e-07 | 2.35e-08 |
| Training | 801 | 4.25e-05 | 8.55e-03 |
| Training | 802 | 3.84e-05 | 7.29e-03 |
| Training | 803 | 4.08e-05 | 7.74e-03 |

Control: with a gain of 1 the content equals the no-gain content exactly, and the script raises if it
does not.

- In evaluation mode the content hardly changes from step to step, and the effect of the gain is 21 to
  28 times that change. A decoder with standardized inputs reads it.
- In training mode the change from step to step is 190 to 201 times the effect of the gain. The
  decoder reads nothing.
- In both modes the effect is small. The group holds a median of 30 to 37 of the 256 map cells, and a
  gain of 1.5 on them changes the content by at most 4.25e-05 of its size.
- The effect of the gain is 34 to 69 times larger in training mode than in evaluation mode. I did not
  test why.

The two arms of the live runs at the same steps:

| Reading | 801 (episode 10) | 802 (episode 39) | 803 (episode 39) |
|---|---|---|---|
| Steps with the same binding group, whole run | 8000 of 8000 | 8000 of 8000 | 8000 of 8000 |
| Gain on the recorded map, median (lowest to highest) | 1.501 (1.494 to 1.508) | 1.476 (1.438 to 1.517) | 1.466 (1.454 to 1.481) |
| Content, lowest cosine between the arms | 0.9999998 | 0.9999999 | 0.9999999 |
| Content, mean relative difference between the arms | 5.11e-05 | 9.06e-05 | 5.94e-05 |
| Content, mean relative change between steps, no-link arm | 7.67e-03 | 7.36e-03 | 7.69e-03 |
| `h_state`, largest absolute difference between the arms | 0.032 | 0.066 | 0.044 |
| Vision bid, largest difference between the arms at one step | 0.167 | 0.192 | 0.309 |

The gain on the map is the contrast of the group cells against the other cells, link arm over no-link
arm, in one recorded episode. The gain reached the map, the recurrent state and the vision bid. It did
not reach the content at a size that the decoder can read.

## Reported only

Stated in the rules:

| Reading | 801 | 802 | 803 |
|---|---|---|---|
| Steps with the gain, of 8000 | 7293 | 7480 | 7480 |
| Ignited steps with vision as the highest bid and as the broadcast, link arm | 6263 of 6263 | 6376 of 6376 | 6458 of 6458 |
| The same, no-link arm | 6233 of 6233 | 6384 of 6384 | 6395 of 6395 |
| Ignited steps with two or more modules admitted, link arm | 0.882 | 0.881 | 0.875 |
| Vision bid, mean, link arm | 0.598 | 0.631 | 0.640 |
| Vision bid, mean, no-link arm | 0.603 | 0.631 | 0.640 |

Added after the data, the score in later episodes:

| Window | Arm, vector | 801 | 802 | 803 |
|---|---|---|---|---|
| Episodes 21 to 40 | link, content | 0.062 | -0.009 | 0.094 |
| Episodes 21 to 40 | no link, content | 0.061 | -0.008 | 0.095 |
| Episodes 31 to 40 | link, content | 0.268 | -0.096 | 0.362 |
| Episodes 31 to 40 | no link, content | 0.268 | -0.105 | 0.364 |
| Episodes 31 to 40 | link, broadcast | 0.253 | -0.165 | 0.336 |
| Episodes 31 to 40 | no link, broadcast | 0.246 | -0.150 | 0.350 |

The two arms stay equal in each window. The scores at seeds 801 and 803 in the last window are not a
result of the gain.

Added after the data, the offline check in evaluation mode with earlier fixed weights of seed 801 (room
seed 521, gain 1.5): 0.217 at episode 0, 0.189 at episode 10, and 0.959 at episode 39.

## What happened during the runs

- The control and `ta_link_s801` finished at revision ee5d696. Free disk space then read 18 GB, against
  44 GB at the start, and the run script stopped before the next run at its guard of 21 GB. One run
  writes about 1.2 GB. I did not find the cause of the rest of the drop.
- The session recorder has its own disk guard of 20 GB. So `ta_link_s801` has `weights.pt` and
  `maps.npz` at episodes 0 and 10 only. Its `vectors.npz` and `steps.jsonl` are complete, so the gate
  scores are not affected. For that seed, the arm comparison above reads episode 10.
- With the owner's approval, I deleted `rv_repro29_s701` and the ten run folders of Gate B8 with their
  logs. The Gate B8 verdict stands in its document. Those raw folders no longer exist.
- The other five runs ran at the same revision, with no changed file, from 07:47. No run was repeated.

## What this shows, and what it does not

- In this room, the selection of the tectum does not reach the workspace vector through a gain on the
  tectum's map. With or without the gain, the broadcast is the vision vector at every ignited step, and
  that vector hardly depends on the map.
- Why the capsule output hardly depends on the map is not known. Sampling noise explains why the offline
  check and the gate disagree. It does not explain why the effect is so small in both modes.
- This does not show that no design can carry the selection. It tests one design, at one gain, in one
  room. I did not check other tasks.
- The binding, the motor map, the value gate and the reversal stand. They act through the tectum and the
  motor map, and not through the workspace.
- The policy cannot use the broadcast (`docs/results/action_core_relay_defect_2026_10.md`), so this was
  not a behavioural test.
- No indicator moves. The GWT-2 score stays the owner's decision. The flags stay off by default.
- An offline check must run in the same module mode as its gate.
- Added 2026-10-31. The covert runs kept the tectum near its starting state, and the offline check passes at
  only 2 of 4 weight sets in evaluation mode (`docs/results/content_map_sensitivity_2026_11.md`).

## Runs

Revision ee5d696, Python 3.12.10, torch 2.14.1. The control with all new flags off (seed 73) gave the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`. Each of the six runs has 41 lines in `episodes.csv`,
8001 lines in `metrics.csv`, 8000 entries in `map_binding.json` and 40 recorded episodes. The commands
and the output are in `runs/_audit_followup/oct13/` (local, gitignored):

| File | Content |
|---|---|
| `run_attention.sh`, `run_attention_resume.sh`, and their `.out` files | The exact commands, the disk stop, the resume |
| `attention_result.json` | The gate |
| `attention_recheck.out`, `attention_result_recheck.json` | The gate probe, run again |
| `attention_second_count.out` | The second count |
| `attention_diagnosis.out` | The arms, the size of the effect, the offline check in both modes |
| `attention_late_episodes.out`, `attention_live_weights.out` | The later episodes, the earlier weights |

Reproduce the readings:

```bash
python -m scripts.analysis.probe_tectal_attention --link runs/ta_link_s801 runs/ta_link_s802 runs/ta_link_s803 --nolink runs/ta_nolink_s801 runs/ta_nolink_s802 runs/ta_nolink_s803
```

```bash
python -m scripts.analysis.diagnose_attention_modes --runs runs
```
