# The learned value of vision locks the workspace at some seeds. Prediction PASSED at 3 seeds

**Without `--learned-valence`, the 3 Gate B6 seeds that fired the kill rule stay under the
limit.** Vision takes 0.786, 0.847 and 0.665 of ignited steps at seeds 64, 65 and 68,
against 0.968, 0.976 and 0.996 with the flag. The agent spends about the same number of
steps in the light in both cases. The prediction was written before the runs.

**The reliability weighting is not needed for the lock.** With `--learned-valence` and
without `--bid-precision gain`, seeds 64 and 65 still fire the kill rule (0.967 and 0.985).

Three seeds, one configuration. No default is changed and no indicator moves.

**Follow-up, Gate B7. PASSED.** The gate without `--learned-valence` at 10 fresh seeds,
option 2 of the decisions below, was written before its runs and passed both questions.
No seed fired the kill rule (top share 0.522 to 0.789). The table is in
`docs/results/dark_room_senses_2026_09.md`.

**Follow-up, Gate B8. PASSED.** The gate with the learned valence and
`--valence-boost saturating` at 10 fresh seeds, option 3 of the decisions below, passed
both questions (top share 0.516 to 0.890). One seed in 10 reached a learned value in the
range that locked before. A paired check at seeds 64, 65 and 68, stated before its runs
in `docs/decisions/2026_09_16_precision_weighted_bids.md`, PASSED. With the limit the
vision share is 0.910, 0.839 and 0.736 at a mean learned value of vision of 3.061, 2.943
and 5.212.

## The question

Gate B6 FAILED on competition at 3 of 10 seeds, where vision took 0.968 to 0.996 of
ignited steps (`docs/results/dark_room_senses_2026_09.md`). The same happened at seed 55
in Gate B3 and at seed 57 when the layouts were moved by one episode. The cause was open.

## The candidate cause, from the code

With `--learned-valence`, the affective modulator adds `valence_gain * |value|` to the bid
of each module (`models/emotion/affective_modulator.py:175`, `valence_gain` 0.15). The
value is learned by a linear temporal difference critic over the bids
(`models/emotion/learned_valence.py`), and it has no upper limit. A raw sense bid is at
most 1.0. While the agent is in the light it receives reward 1.0 at every step. If the
vision bid is high during those steps, the value of vision grows, and with it the boost
that vision receives at every later step.

## Observation on recorded runs

Read-only probe, `python -m scripts.analysis.probe_valence_lockin --runs <run folders>`,
pinned by `tests/test_valence_lockin_probe.py`. 22 recorded runs with the Gate B4 flags.

| Run | Vision share | Steps in the light | Mean value, vision | Mean value, hearing |
|---|---|---|---|---|
| Gate B6 seed 68 | **0.996** | 435 | 4.940 | 0.451 |
| Seed 57, moved layouts | **0.983** | 390 | 4.117 | 1.207 |
| Gate B6 seed 65 | **0.976** | 372 | 2.799 | 0.200 |
| Gate B6 seed 64 | **0.968** | 310 | 3.145 | 0.751 |
| Gate B3 seed 55 | **0.961** | 191 | 2.396 | 0.392 |
| Gate B4 seed 57 | 0.928 | 265 | 2.902 | 0.810 |
| Seed 58, moved layouts | 0.873 | 124 | 1.013 | 0.295 |
| Gate B5 seed 61 | 0.835 | 34 | -0.007 | -0.008 |
| Gate B4 seed 58 | 0.798 | 111 | 0.957 | 0.254 |
| Gate B6 seed 69 | 0.783 | 57 | 0.354 | 0.249 |
| Gate B6 seed 71 | 0.776 | 25 | -0.012 | 0.063 |
| Gate B3 seed 54 | 0.775 | 44 | 0.246 | 0.104 |
| Gate B6 seed 66 | 0.748 | 248 | 1.624 | 0.464 |
| Gate B6 seed 72 | 0.726 | 68 | 0.599 | 0.229 |
| Gate B6 seed 70 | 0.712 | 225 | 1.553 | 0.901 |
| Gate B5 seed 60 | 0.708 | 190 | 0.615 | 0.145 |
| Gate B6 seed 63 | 0.695 | 51 | 0.293 | 0.075 |
| Gate B3 seed 56 | 0.685 | 64 | 0.369 | 0.222 |
| Seed 59, moved layouts | 0.644 | 24 | -0.091 | -0.010 |
| Gate B5 seed 62 | 0.641 | 43 | 0.485 | 0.051 |
| Gate B6 seed 67 | 0.579 | 92 | 0.212 | 0.192 |
| Gate B4 seed 59 | 0.488 | 11 | 0.027 | -0.081 |

Bold marks a vision share of 0.95 or more, the kill limit. The vision share here counts all
recorded steps, so it can differ in the third decimal from the gate probe, which leaves out
step 0 of each episode.

What the table and the per episode output show.

1. The 5 runs at or above the kill limit are among the 6 runs with the highest mean value
   of vision. The sixth is Gate B4 seed 57, with a mean of 2.902 and a share of 0.928, just
   under the limit.
2. Spearman rho across runs between the vision share and the mean value gap (vision minus
   hearing) is 0.636 over the 10 Gate B6 runs and 0.748 over the other 12 runs.
3. In each of the 5 runs the value of vision rises in one long episode in the light. Gate
   B6 seed 64, episode 3, 188 steps in the light, value 0.02 to 7.02. Seed 65, episode 5,
   148 steps, 1.73 to 6.29. Seed 68, episode 2, 200 steps, 1.51 to 8.03. Gate B3 seed 55,
   episode 3, 161 steps, 0.14 to 5.87. Seed 57 with moved layouts, episodes 3 and 4, 169
   and 200 steps, 0.08 to 10.85. After that episode vision wins 0.85 or more of ignited
   steps in every later episode of those runs.
4. With a value of 6 to 8 the boost to the vision bid is 0.9 to 1.2. That is as large as
   the largest raw bid.
5. Two Gate B6 seeds had the same rise late and stayed under the limit over the whole run.
   Seed 66, episode 6, 155 steps in the light, value -0.01 to 5.31, and vision wins 0.95 to
   1.00 in the 3 episodes after it. Seed 70, episode 5, 178 steps, value -0.03 to 5.31,
   where the value of hearing also rose to 2.88 and vision wins 0.44 to 0.96 after it.

This is a description of recorded runs. A rho across seeds is not a cause.

## The ablation, stated before it is run

Seeds 64, 65 and 68, the 3 Gate B6 seeds that fired the kill rule. Code at the revision of
the commit that adds this document. Gate B6 flags except as stated.

- **Arm A, without `--learned-valence`.** Prediction P1. The vision share of ignited steps,
  as the gate probe computes it, is below 0.95 at all 3 seeds. If it is 0.95 or more at a
  seed, P1 FAILED and the learned value is not needed for the kill at that seed.
- **Arm B, with `--learned-valence` and with `--bid-precision off`.** No prediction. It is
  reported only, to show if the reliability weighting takes part.

Rules. One run per seed and arm. A run with fewer than 10 episodes or 2000 steps on disk
is run again with the same seed, and that is reported. No other rerun.

Limit, stated now. A run without the flag takes another path through the room, so it
spends another number of steps in the light. P1 tests whether the kill needs the flag at
these seeds. It does not show that the boost is the only route to a kill.

## Result of the ablation, 2026-10-03. P1 PASSED

Run at revision `3c70cff`, after the section above was committed. Each run has 10 episodes
and 2000 steps on disk. No run was repeated. The vision share is the top share of the gate
probe, 1990 steps judged per run.

| | Seed 64 | Seed 65 | Seed 68 |
|---|---|---|---|
| Gate B6, both flags. Vision share | **0.968** | **0.976** | **0.996** |
| Gate B6. Steps in the light | 310 | 372 | 435 |
| Arm A, without `--learned-valence`. Vision share | 0.786 | 0.847 | 0.665 |
| Arm A. Steps in the light | 307 | 389 | 439 |
| Arm B, without `--bid-precision gain`. Vision share | **0.967** | **0.985** | 0.812 |
| Arm B. Steps in the light | 300 | 389 | 441 |
| Arm B. Mean value, vision / hearing | 2.461 / 1.440 | 2.773 / 0.824 | 2.957 / 1.668 |

Bold marks a share at or above the kill limit of 0.95.

**P1 PASSED.** Without the learned valence the vision share is below 0.95 at all 3 seeds,
and criteria (1) to (3) of the gate pass at all 3.

The limit stated before the runs did not apply in practice. The agent spent nearly the
same number of steps in the light without the flag (307, 389, 439) as with it (310, 372,
435). The time in the light is therefore not what differs between the arms. The boost is.

Arm B, reported only. Seeds 64 and 65 fire the kill rule without the reliability
weighting. Seed 68 does not (0.812). There the value of hearing is also high (1.668), so
the gap between the two values is smaller.

Also seen, and not a result. In Arm A the task criterion of the gate probe gives a pooled
rho of 0.404 with a one-sided p of 0.015 over 30 episodes, and in Arm B a pooled rho of
-0.346. Neither was predicted, and 3 seeds chosen because they failed are not a sample.

## What this establishes

- At seeds 64, 65 and 68 the kill needs `--learned-valence`. The same seeds pass the
  competition criteria without it.
- The mechanism matches the code. The boost is `valence_gain * |value|`, the value has no
  upper limit, and it rises to between 6 and 8 in one long episode in the light.

## What this does NOT establish

- Not that the learned valence is wrong as an idea. The defect is that the boost has no
  limit and that one module can take all of it.
- Not what happens at other seeds without the flag. Three seeds were run, and they were
  chosen because they failed.
- Not that removing the flag keeps the task link. That needs a gate written before its
  runs, on fresh seeds.
- Nothing about affect. A learned value is a number in a linear critic.

## Decisions for the project owner

1. Keep `--learned-valence` in the gate configuration as it is, and accept that about 3
   seeds in 10 lock.
2. Run the gate without `--learned-valence` at 10 fresh seeds, with the rule written
   first. This costs 10 runs and no new code. Done 2026-10-03 as Gate B7. PASSED.
3. Give the boost a limit, behind a new default-off flag, and then run the same gate. One
   form is to take the value of each module as a share of the sum over modules, so that
   the values rank the modules and their sum cannot grow. This needs new code with tests
   and its own pre-stated gate. The form chosen 2026-10-03 is `--valence-boost saturating`,
   a boost of `valence_gain * tanh(|value|)`. It needs no new constant and it leaves the
   small-value behaviour as it was. It is judged by Gate B8 in
   `docs/decisions/2026_09_16_precision_weighted_bids.md`.

## Reproduce

```
python -m scripts.analysis.probe_valence_lockin --runs runs/gate_b6_s63 ... runs/gate_b6_s72
python -m scripts.analysis.probe_gate_b2 --runs runs/lockin_noval_s64 runs/lockin_noval_s65 runs/lockin_noval_s68
python -m scripts.analysis.probe_gate_b2 --runs runs/lockin_nogain_s64 runs/lockin_nogain_s65 runs/lockin_nogain_s68
```

The run command is the Gate B6 command in
`docs/decisions/2026_09_16_precision_weighted_bids.md`, without `--learned-valence` for
Arm A and without `--bid-precision gain` for Arm B.
