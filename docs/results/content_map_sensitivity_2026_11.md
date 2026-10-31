# Why the workspace content ignores the tectum map: my hypothesis FAILED as stated, and three causes are measured

Date: 2026-10-31. Read-only, with no change of weights. Rules: the docstring of
`scripts/analysis/probe_content_map_sensitivity.py`, pushed before the run (commit 858cac3).

## Verdict

FAILED. All four predictions fail as written. The hypothesis
holds exactly in the covert runs and does not hold in the learning runs.

| Prediction | Rule | Covert runs `ta_nolink_s801..803` | Learning runs `vg_learn_s601..603` | Verdict |
|---|---|---|---|---|
| P1 | Pose rms falls at each of the 3 routing levels | Rises from level 0 to level 2 in evaluation mode (1.12e-06, 4.94e-06, 8.73e-06 at seed 801) | Falls | FAILED |
| P2 | Content minus bias is at most 0.1 of the content | Holds, 0.00008 to 0.00607 | 0.930 to 1.142 | FAILED |
| P3 | The largest fall is at the projection | Holds, 7728 to 12470 times | Routing 38.9, RSSM 7.7, RSSM 5.1 | FAILED |
| P4 | Direction from `obs_map` at least 0.5, from content at most 0.2 | Not part of this rule | `obs_map` 0.218, -0.131, -0.027 | FAILED |

P4 was also a weak test, and I chose its label badly. "The object with the larger x" nearly always lies
to the right of the agent, so a constant guess already scores above 0. In the covert runs, the null with
swapped episodes reaches 0.56 to 0.66. `obs_map` itself is at or below its null, so P4 says nothing about
the path from the map to the content.

The outcome rule points to the pose scale against the bias in the covert runs. For the learning runs it
says "only sampling noise", because no single stage falls 100 times. That reading is wrong, as the
evaluation mode row of the last table shows. The rule did not allow for a fall spread over several stages.

## What was measured

The gain is 1.5 on the selected cells. All values are from weights of episode 39, room seed 521, steps
after 15. "Eval" is evaluation mode, and "train" is training mode, the mode of the live runs.

| Run | Posterior rms (eval) | Primary capsule rms (eval) | Level 0 rms (eval) | Content minus bias share (eval) | Change of the content from the gain, eval and train | Change between steps, train |
|---|---|---|---|---|---|---|
| Covert 801 | 0.086 | 0.0133 | 1.12e-06 | 0.00010 | 1.26e-06, 4.25e-05 | 8.55e-03 |
| Covert 802 | 0.082 | 0.0071 | 3.56e-07 | 0.00015 | 5.59e-07, 3.84e-05 | 7.29e-03 |
| Covert 803 | 0.082 | 0.0070 | 3.21e-07 | 0.00008 | 6.53e-07, 4.08e-05 | 7.74e-03 |
| Learning 601 | 1.75 | 0.354 | 0.194 | 0.930 | 4.70e-06, 4.84e-06 | 2.09e-04 |
| Learning 602 | 0.887 | 0.353 | 0.204 | 1.022 | 4.74e-03, 4.07e-03 | 4.41e-03 |
| Learning 603 | 1.77 | 0.354 | 0.128 | 1.142 | 1.17e-02, 9.36e-03 | 2.18e-02 |
| DMTS 42, contrast | 1.17 | 0.354 | 0.0211 | 0.571 | 5.30e-05, 5.84e-05 | 3.03e-03 |
| DMTS 43, contrast | 2.78 | 0.354 | 0.00326 | 0.145 | 3.12e-05, 3.77e-05 | 5.20e-04 |
| DMTS 44, contrast | 0.963 | 0.353 | 0.0130 | 0.387 | 1.23e-04, 1.23e-04 | 8.39e-03 |

DMTS has no binder, so its arm uses a fixed central block of 36 cells. The seeds differ widely, so a single
checkpoint must not stand for a group. My first reading of this table took seed 601 as the whole group,
and it was wrong.

## Three causes

**1. Sampling noise enters at the latent, at a fixed size.** `cont_logvar` stays within -0.036 to 0.046
in all 30 saved weight sets of these six runs, so the sample has a standard deviation of about 1. The rms
of the sample matches this. It is 1.00 with a mean of rms 0.09, and 2.00 with a mean of rms 1.73. In
training mode, the sample changes by 0.70 to 1.41 of its size between steps. Pure noise would give 1.41.

**2. The capsule stage has two failure states.** Squash shrinks a short vector by about the square of its
length, and it sets a long vector to length 1.

- In the covert runs the primary capsules are short in evaluation mode, level 0 is near 1e-06, and the
  content is the bias of `workspace_proj`. The mean content lies on that bias, with cosine 1.000 and a
  difference of 0.06 to 0.10 percent. Its motion is 0.0024 to 0.0027, against a norm of 0.43 to 0.47.
- In the learning runs and the DMTS checkpoints the primary capsules saturate. An rms of 0.353 to 0.354 is
  1/√8, which is length 1 for an 8-element capsule. A change of size cannot pass that point. Only a change
  of direction can. The content is then a fixed vector of norm 1.7 to 2.4 that is not the bias (cosine
  -0.03 to 0.46), and 95 percent of its motion lies in 1 or 2 dimensions.

**3. The upper capsule levels carry almost nothing.** In the dark room runs, levels 1 and 2 have an rms of
at most 1.3e-04 and 2.6e-05. Level 0 reaches 0.204 in the learning runs. So `--capsule-workspace-source
all_levels` in effect reads level 0 in those runs. From the code, the top-down step feeds levels 1 and 2
through the feedback projections. I did not test if that is the reason.

## A larger effect is not a readable one

The offline check of the rules, in both modes (room seed 521, gain 1.5). Score: direction of the selected
object, chance 0.

| Weights | Evaluation, no gain | Evaluation, gain | Training, no gain | Training, gain |
|---|---|---|---|---|
| Learning 601 | -0.132 | 0.850 | -0.060 | -0.015 |
| Covert 801 | -0.208 | 0.959 | -0.020 | -0.018 |
| Learning 602 | -0.184 | -0.142 | -0.167 | -0.066 |
| Learning 603 | -0.103 | -0.112 | -0.113 | -0.114 |

- The check of 2026-10-17 used only the first row's weights. It passes at 2 of these 4 weight sets in
  evaluation mode, and at 0 of 4 in training mode.
- At seeds 602 and 603 the gain moves the content 1000 to 2500 times more than at seed 601. The direction
  still cannot be read, even without sampling noise. So this gain does not write the place into the
  content in a linear form.
- The two passes come from weights where the content hardly moves. A small change on a static vector is
  readable after standardization. This does not show that any part of the agent could use it.
- So taking the posterior mean in the content path would not be enough. Evaluation mode is that condition,
  and it fails at 2 of 4 weight sets.

## A new limit on the tectal attention gate

The covert runs kept their tectum near its starting state. This was read after the data.

| Saved weights | Posterior rms, last 50 steps, episode 0 | The same, episode 39 | Routing `W` rms at episode 39, 3 levels (start about 0.0100) |
|---|---|---|---|
| Covert 801 to 803 | 0.082 to 0.090 | 0.088 to 0.092 | 0.0101, 0.0100, 0.0100 |
| Learning 601 to 603 | 0.079 to 0.088 | 0.863 to 1.750 | 0.0122 to 0.0143, 0.0123 to 0.0129, 0.0100 |

So the gate of `docs/results/tectal_attention_2026_10.md` tested a nearly untrained capsule path. The gate
still FAILED by its rules, and this is a limit on what that failure covers. I did not test why the covert
runs stay there. In them the agent does not move and does not reach food.

## Reported only: live decode

Recorded vectors of each whole run (7960 steps, whole episodes held out, 1 split). The direction label is
the weak one named above, so these scores are not read. The tone is the band of the selected group, as a
two-class read, against the share of the larger class.

| Stream | Learning 601, 602, 603 | Larger class | Covert 801, 802, 803 | Larger class |
|---|---|---|---|---|
| `tectum_content` | 0.831, 0.825, 0.854 | 0.870, 0.850, 0.917 | 0.437, 0.459, 0.461 | 0.513, 0.500, 0.500 |
| `broadcast` | 0.906, 0.882, 0.929 | the same | 0.506, 0.553, 0.532 | the same |
| `capsule_poses` | 0.805, 0.821, 0.902 | the same | 0.368, 0.491, 0.430 | the same |

Only the broadcast of the learning runs lies above the larger class, at 3 of 3 seeds. Those runs use the
default merge, and the broadcast is often the audio vector there (`docs/results/broadcast_carrier_2026_10.md`).
That is a possible cause. I did not test it.

## Controls

- A gain of 1 gave the no-gain values exactly at every tap and step, in every run and mode. The probe raises
  if not.
- In the dark room runs, the random arm matched the gain arm's size at `obs_map` within 1 percent. The probe
  raises if not. The random arm's falls are in the output file.
- The covert 801 content values repeat `diagnose_attention_modes.py` from separate code (1.26e-06 and
  4.25e-05, with 5.79e-08 and 8.55e-03 between steps).
- The tests pin the closed form of squash on a short vector, the fall, the offset share and the labels. They
  found one defect in the tone read before the run.

## What this shows, and what it does not

- The path from the map to the content has at least three separate problems. A fix of one does not repair
  the others. Six checkpoints show this in the dark room, and three show part of it in DMTS.
- This is not a test of a design. No mechanism was changed, and no indicator moves. GWT-2 stays PARTIAL.
- The saved weights behind the table are episode 39, at one room sequence and 2 episodes. Decodes are one
  split each.
- I did not test why the posterior grows in the learning runs, why `cont_logvar` does not move, or if a
  nonlinear read would find the place at seeds 602 and 603.
- The earlier claim that the vectors tell known stimuli apart is not contradicted. Those stimuli are
  pixel-identical, and in evaluation mode a tiny difference on a static vector is readable. This raises a
  question for those results that this document does not answer.

## Runs

Python 3.12.10, revision 858cac3, no changed file. Output in `runs/_audit_followup/oct13/` (local,
gitignored): `content_map.out`, `content_map.json`, `content_map_tables.out`, `content_map_offline.out`.

```bash
python -m scripts.analysis.probe_content_map_sensitivity --runs runs --output-json content_map.json
```
