# Owner override and pre-stated Gate B3: reliability-weighted sense bids

Date of decision: 2026-09-16. Decided by the owner. Written BEFORE the runs it judges.

## What failed first

The offline gain check (`scripts/analysis/probe_sense_gain.py`, seeds 51 to 53) **FAILED**.

| Criterion | Seed 51 | Seed 52 | Seed 53 |
|---|---|---|---|
| (R1) AUC of vision gain, light in view, at least 0.70 | 0.997 | 0.981 | 0.904 |
| (R2) AUC of the audio share, light out of view, at least 0.70 | 0.996 | 0.961 | 0.857 |
| (R3) usable in-view runs, at least 3 | **1** | 7 | 5 |
| (R3) median ratio, at least 0.80 | 1.000 | 1.000 | 1.000 |

R3 failed at seed 51 because the light stayed in view long enough only once, so the
criterion could not be measured there. The gate's own rule was "a FAIL means no build".

## The override

The owner decided on 2026-09-16 to build anyway, and to judge the result with a live gate.
Reasons recorded at the time of the decision:

- the R3 failure is "not measurable", not a measured habituation;
- R1 and R3 can pass by the construction of the room (the light is the brightest object
  and fills whole grid cells), so more offline checks add little;
- only a live run answers whether the weighting changes which module wins.

Nothing in the offline check is evidence that the weighting works.

## What was built

`models/core/precision_weighting.py`, behind `--bid-precision off|gain`, default `off`.

- vision gain: the strongest pooled cell of the frame outside the agent's own disc;
- audio gain: the waveform RMS over the tone RMS at the source;
- each sense bid is multiplied by twice its share of the total gain, so equal gains leave
  both bids unchanged, and the bids stay inside [0, 1].

Sources: Ma, Beck, Latham and Pouget 2006 (gain carries reliability); Fetsch et al. 2009
and 2011 (weights change within a trial, read from the current input); Feldman and Friston
2010 and Stein and Stanford 2008 (salience habituates, reliability does not); Ohshiro,
Angelaki and DeAngelis 2011 (divisive normalization of the shares).

Bit identity with the flag off: `metrics.csv` md5 `7c101c58eecc4eb91409c491a5ad0e26` for the
standard check (`--env dark_room --episodes 2 --max-steps 30 --seed 42 --existence-drive on`),
measured twice with the new code.

## Gate B3, pre-stated

The Gate B2 criteria, unchanged, including the task criterion that failed. The full text is
in the docstring of `scripts/analysis/probe_gate_b2.py`:

1. competition: the runner-up takes at least 0.05 of ignited steps;
2. silence: below 0.50 of steps;
3. selectivity: against the within-episode shuffled null;
4. task: pooled Spearman rho above 0 with a one-sided permutation p below 0.05, and rho
   above 0 at 2 or more of the 3 seeds;
5. KILL: any module at 0.95 or more of ignited steps.

Data: new seeds 54, 55 and 56, never run before, with the Gate B2 flags plus
`--bid-precision gain`:

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
    --seed <54|55|56> --enable-audio --rssm-latent-mode continuous \
    --capsule-workspace-source all_levels --existence-drive on \
    --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
    --dark-room-view agent_centered --vision-bid-reduction zscore \
    --audio-salience surprise --learned-valence \
    --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain \
    --log-dir runs/gate_b3_s<seed>
```

A PASS on criterion 4 is the first evidence that hearing wins more when the light is out of
view. A FAIL is recorded as a FAIL, FAILED first, in `docs/results/dark_room_senses_2026_09.md`.
A comparison against the Gate B2 runs is not part of this gate: those runs used other seeds.

A PASS is not evidence of affect. Finding a light is automatic approach, which Feinberg and
Mallatt exclude as evidence of affect.

## Gate B4, pre-stated 2026-09-16, before the runs it judges

Same criteria again, on new seeds 57, 58 and 59, with the Gate B3 flags plus the agent drawn
as the project mark in the site colour.

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
    --seed <57|58|59> --enable-audio --rssm-latent-mode continuous \
    --capsule-workspace-source all_levels --existence-drive on \
    --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
    --dark-room-view agent_centered --vision-bid-reduction zscore \
    --audio-salience surprise --learned-valence \
    --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain \
    --dark-room-agent-mark ring --dark-room-agent-colour 217,119,87 \
    --log-dir runs/gate_b4_s<seed>
```

Why the run is made. The agent's own body is drawn into the frames it receives, so the mark
and the colour change the input. The change cannot ride along with an older result, and the
published replays must show what the agent actually saw. This gate measures whether the
findings of Gate B3 survive the change of appearance.

What a result means here. A PASS does not confirm Gate B3, because the seeds differ and the
pixels differ. A FAIL does not refute it either. The two gates are separate measurements, and
no number from one is carried into the other.

## Result of Gate B4, 2026-09-16. PASSED

Every criterion passed at all 3 seeds, the first full pass in this line of work.

Note added 2026-10-03. These runs are reproduced bit for bit by the code with the seeding
call repaired. With the dark room layouts moved by one episode, the defect present from
2026-09-25 to 2026-10-03, the same seeds FAILED. See Gate B5 and Gate B6 below.

| Criterion | Seed 57 | Seed 58 | Seed 59 |
|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.072 | 0.202 | 0.488 |
| (2) silence, below 0.50 | 0.133 | 0.247 | 0.322 |
| (3) selectivity, difference / null p95 | 0.083 / 0.011 | 0.249 / 0.159 | 0.095 / 0.017 |
| KILL, a module at 0.95 or more | 0.928 | 0.798 | 0.512 |
| (4) task, rho per seed | 0.562 | 0.275 | 0.736 |

Criterion 4 pooled over 30 episodes gives rho 0.458 with a one-sided permutation p of 0.006.

What this does NOT show, stated with the result.

- It is not a repeat of Gate B3. The seeds differ and the pixels differ, so the two gates are
  separate measurements and no number carries from one to the other.
- The confound is still measured and still not excluded. The audio share against the episode
  index gives rho -0.194, 0.512 and 0.815, so 2 of 3 seeds rise with the episode number.
- Seed 57 sits at 0.928, close to the 0.95 kill limit.
- Nothing here is evidence of affect. Finding a light is automatic approach.

## Gate B5, pre-stated 2026-10-03, before the runs it judges

Why the run is made. Gate B4 passed on runs of 2026-09-16. Those runs were made before the
environment's trial generator was seeded from `--seed` (2026-09-25), so they cannot be
reproduced. On the current code the same seeds 57, 58 and 59 FAILED the same criteria
(`docs/results/audit_gpu_checks_2026_10.md`). Seeds 57 to 59 have now been seen on the
current code, so they cannot serve as a test. This gate asks whether the Gate B4 criteria
hold on the current code at seeds that nobody has looked at.

Same criteria, same thresholds, same probe (`scripts/analysis/probe_gate_b2.py`). New seeds
60, 61 and 62. Same flags as Gate B4, default `--broadcast-merge`.

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200     --seed <60|61|62> --enable-audio --rssm-latent-mode continuous     --capsule-workspace-source all_levels --existence-drive on     --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision     --dark-room-view agent_centered --vision-bid-reduction zscore     --audio-salience surprise --learned-valence     --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain     --dark-room-agent-mark ring --dark-room-agent-colour 217,119,87     --log-dir runs/gate_b5_s<seed>
python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b5_s60 runs/gate_b5_s61 runs/gate_b5_s62
```

Rules fixed before the runs.

- One run per seed. No rerun and no change of seed after a result is seen. The current code
  is deterministic at a fixed seed, so a rerun would give the same file.
- Completion is read from the csv row counts, 10 episodes and 2000 steps per run.
- The verdict is the last line of the probe output. No criterion or threshold is changed.

What a result means here. A PASS says the criteria hold on the current code at 3 fresh seeds.
It does not restore the runs of 2026-09-16 and it does not undo the failure at seeds 57 to 59.
A FAIL says the Gate B4 pass does not hold on the current code. In both cases the public
text must describe Gate B4 as a result of runs that cannot be reproduced.

**Correction, 2026-10-03, written after the runs.** The reason given above is wrong in one
point. The runs of 2026-09-16 can be reproduced. The dark room takes its layout from the
numpy seed, and a defect in the seeding call of 2026-09-25 moved every layout by one
episode. The gate stands as written, the same criteria at fresh seeds. Its runs received
moved layouts, like every seeded dark room run of that period.

## Result of Gate B5, 2026-10-03. FAILED on the task criterion

Run at revision `b1c1afb`, after the gate above was committed. One run per seed, 10
episodes and 2000 steps each, 1990 steps judged.

| Criterion | Seed 60 | Seed 61 | Seed 62 |
|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.292 | 0.166 | 0.359 |
| (2) silence, below 0.50 | 0.298 | 0.332 | 0.320 |
| (3) selectivity, difference / null p95 | 0.160 / 0.064 | 0.119 / 0.067 | 0.121 / 0.034 |
| KILL, a module at 0.95 or more | 0.708 | 0.834 | 0.641 |
| (4) task, rho per seed | 0.294 | 0.007 | 0.063 |

Criterion 4 pooled over 30 episodes gives rho 0.200 with a one-sided permutation p of 0.095.
The limit is 0.05, so the criterion FAILED and the gate FAILED. Criteria 1 to 3 and the kill
rule passed at all 3 seeds.

Consequence. The gate result depends on the layouts. Gate B4 passed. The same seeds with
layouts moved by one episode FAILED, and the fresh seeds of this gate FAILED on the task
criterion only. Competition between hearing and vision holds at 5 of the 6 runs with moved
layouts. Three seeds do not settle the task link. The full table is in
`docs/results/dark_room_senses_2026_09.md`.

## Gate B6, pre-stated 2026-10-03, before the runs it judges

Why the run is made. The task criterion passed in Gate B3 and Gate B4. It failed in Gate B5
and at seeds 57 to 59 with moved layouts. Per seed rho is above 0 in all 12 of those runs.
A gate of 3 seeds has 30 episodes and does not separate a weak link from no link. Gate B6
uses 10 seeds and 100 episodes.

Code. The revision of the commit that adds this section. It holds the repair of the seeding
call, so the layouts are the ones `--seed` gives. Flags of Gate B4, default
`--broadcast-merge`. Seeds 63 to 72, never run before.

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
    --seed <63 to 72> --enable-audio --rssm-latent-mode continuous \
    --capsule-workspace-source all_levels --existence-drive on \
    --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
    --dark-room-view agent_centered --vision-bid-reduction zscore \
    --audio-salience surprise --learned-valence \
    --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain \
    --dark-room-agent-mark ring --dark-room-agent-colour 217,119,87 \
    --log-dir runs/gate_b6_s<seed>
python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b6_s63 ... runs/gate_b6_s72
```

Two questions, each with its own verdict. Both are fixed here.

1. **Task link.** This is the reason for the gate. Criterion (4) of the probe, unchanged.
   Pooled Spearman rho of the out-of-view share against the audio share above 0, a
   one-sided permutation p below 0.05, and at least 15 qualifying episodes. The rule
   "rho above 0 at 2 or more of the 3 seeds" keeps its proportion of two thirds, rounded
   up, which is 7 of 10. The probe was extended to more than 3 seeds before the runs, and
   the rule is pinned by tests.
2. **Competition at every seed.** Criteria (1) to (3) and the kill rule, unchanged, at each
   of the 10 seeds. One failing seed fails this question. Of the 12 runs so far, 2 fired
   the kill rule (seed 55 in Gate B3 and seed 57 with moved layouts), so this question can
   fail again for that known and open reason.

Gate B6 as a whole PASSES only when both questions pass. The two answers are reported
separately in any case.

Rules fixed before the runs.

- One run per seed. A run with fewer than 10 episodes or 2000 steps on disk is run again
  with the same seed, and that is reported. No other rerun.
- No change of seeds, criteria or thresholds after a result is seen.
- Reported and never gated. The rho of the audio share against the episode index, per seed.

What a result means here. A task PASS says the link holds at 10 fresh seeds. The confound
with the episode index is not excluded by this design. A task FAIL says reliability-weighted
bids give no measurable task link at this sample size, and the claim is dropped until a new
mechanism is proposed. Neither result is evidence of affect.

## Result of Gate B6, 2026-10-03. FAILED on competition at 3 of 10 seeds. The task link PASSED

Run at revision `5ad584e`, after the gate above was committed. Seeds 63 to 72, 10 episodes
and 2000 steps per run on disk, 1990 steps judged per run, 100 qualifying episodes.

The first run of seed 71 stopped 6 seconds after its start with exit code 127 and wrote no
step. Its log is empty. It was run again with the same seed, as the rule above states, and
that run is complete. The incomplete folder is kept as `runs/gate_b6_s71_incomplete`.

| Criterion | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
|---|---|---|---|---|---|---|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.306 | **0.032** | **0.024** | 0.252 | 0.422 | **0.004** | 0.217 | 0.289 | 0.224 | 0.274 |
| (2) silence, below 0.50 | 0.336 | 0.130 | 0.154 | 0.203 | 0.324 | 0.088 | 0.299 | 0.256 | 0.345 | 0.286 |
| (3) selectivity, difference | 0.106 | 0.149 | 0.151 | 0.119 | 0.147 | 0.082 | 0.180 | 0.150 | 0.071 | 0.175 |
| (3) selectivity, null p95 | 0.014 | 0.149 | 0.050 | 0.092 | 0.054 | -0.006 | 0.116 | 0.083 | 0.017 | 0.126 |
| KILL, a module at 0.95 or more | 0.694 | **0.968** | **0.976** | 0.748 | 0.578 | **0.996** | 0.783 | 0.711 | 0.776 | 0.726 |
| (4) task, rho per seed | 0.073 | 0.349 | -0.038 | 0.613 | 0.457 | -0.202 | 0.492 | 0.535 | 0.493 | 0.383 |
| Audio share against episode index, rho | 0.745 | -0.049 | 0.219 | -0.340 | 0.224 | 0.493 | 0.839 | 0.292 | 0.316 | 0.559 |

Bold marks a failed criterion. Vision is the top module at all 10 seeds.

**Question 1, the task link. PASSED.** Pooled over 100 episodes rho is 0.362 with a
one-sided permutation p of 0.002. Rho is above 0 at 8 of 10 seeds and the rule asks for 7.
The 2 seeds with a rho below 0 are seeds 65 and 68, where the kill rule fired and hearing
won 0.024 and 0.004 of ignited steps.

**Question 2, competition at every seed. FAILED.** Seeds 64, 65 and 68 fired the kill rule
(vision 0.968, 0.976 and 0.996 of ignited steps) and failed criterion (1). The other 7 seeds
passed criteria (1) to (3).

**Gate B6 as a whole FAILED**, because both questions must pass.

What this does NOT show, stated with the result.

- The confound with the episode index is not excluded. The audio share rises with the
  episode number at 8 of 10 seeds, and with rho 0.45 or more at 4 of them.
- Vision takes 0.95 or more of ignited steps at 3 of 10 fresh seeds. Why it does so at
  some seeds is still open. It also happened at seed 55 in Gate B3.
- Nothing here is evidence of affect. Finding a light is automatic approach.

## Gate B7, pre-stated 2026-10-03, before the runs it judges

Why the run is made. Gate B6 FAILED on competition at seeds 64, 65 and 68. An ablation at
those 3 seeds showed that the kill needs `--learned-valence`
(`docs/results/vision_lockin_2026_10.md`). Those seeds were chosen because they failed, so
they are not a sample. Gate B7 asks the same two questions as Gate B6 at 10 fresh seeds,
with the Gate B6 flags and without `--learned-valence`.

Code. The revision of the commit that adds this section. Seeds 73 to 82, never run before.

```
python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
    --seed <73 to 82> --enable-audio --rssm-latent-mode continuous \
    --capsule-workspace-source all_levels --existence-drive on \
    --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
    --dark-room-view agent_centered --vision-bid-reduction zscore \
    --audio-salience surprise \
    --ignition-rule tolerance --ignition-tolerance-sd 1.0 --bid-precision gain \
    --dark-room-agent-mark ring --dark-room-agent-colour 217,119,87 \
    --log-dir runs/gate_b7_s<seed>
python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b7_s73 ... runs/gate_b7_s82
```

Two questions, each with its own verdict. Both are the Gate B6 questions, unchanged.

1. **Competition at every seed.** This is the reason for the gate. Criteria (1) to (3) and
   the kill rule at each of the 10 seeds. One failing seed fails this question.
2. **Task link.** Criterion (4) of the probe. Pooled Spearman rho above 0, a one-sided
   permutation p below 0.05, at least 15 qualifying episodes, and rho above 0 at 7 or more
   of the 10 seeds.

Gate B7 as a whole PASSES only when both questions pass. The two answers are reported
separately in any case.

Rules fixed before the runs.

- One run per seed. A run with fewer than 10 episodes or 2000 steps on disk is run again
  with the same seed, and that is reported. No other rerun.
- No change of seeds, criteria or thresholds after a result is seen.
- Reported and never gated. The rho of the audio share against the episode index, per seed.

What a result means here.

- Competition PASSED. Without the learned valence no module takes the workspace at 10
  fresh seeds. It does not show that the lock cannot occur by another route.
- Competition FAILED. The lock has a second route that does not need the learned value.
- Task link PASSED. The link does not need the learned valence. The confound with the
  episode index is not excluded by this design.
- Task link FAILED. In this configuration the link is not measurable at this sample size.
  It does not show that the learned valence causes the link, because Gate B6 and Gate B7
  use different seeds.

No result here changes a default, and no result moves an indicator. Neither result is
evidence of affect.

## Result of Gate B7, 2026-10-03. PASSED both questions

Run at revision `4a01ba2`, after the gate above was committed. Seeds 73 to 82, 10 episodes
and 2000 steps per run on disk, 1990 steps judged per run, 100 qualifying episodes. No run
was repeated.

| Criterion | 73 | 74 | 75 | 76 | 77 | 78 | 79 | 80 | 81 | 82 |
|---|---|---|---|---|---|---|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.436 | 0.213 | 0.264 | 0.317 | 0.285 | 0.478 | 0.344 | 0.347 | 0.211 | 0.336 |
| (2) silence, below 0.50 | 0.351 | 0.268 | 0.300 | 0.314 | 0.311 | 0.333 | 0.303 | 0.310 | 0.355 | 0.295 |
| (3) selectivity, difference | 0.103 | 0.264 | 0.150 | 0.118 | 0.140 | 0.092 | 0.070 | 0.131 | 0.074 | 0.137 |
| (3) selectivity, null p95 | 0.014 | 0.072 | 0.069 | 0.044 | 0.029 | 0.019 | 0.008 | 0.047 | 0.017 | 0.070 |
| KILL, a module at 0.95 or more | 0.564 | 0.787 | 0.736 | 0.683 | 0.715 | 0.522 | 0.656 | 0.653 | 0.789 | 0.664 |
| (4) task, rho per seed | 0.469 | 0.671 | 0.244 | 0.508 | 0.322 | -0.313 | 0.620 | -0.289 | 0.529 | 0.726 |
| Audio share against episode index, rho | 0.285 | 0.280 | 0.200 | 0.705 | 0.394 | 0.523 | 0.236 | 0.875 | 0.697 | 0.608 |

No criterion failed. Vision is the top module at all 10 seeds.

**Question 1, competition at every seed. PASSED.** Criteria (1) to (3) pass at all 10
seeds and the kill rule fires at none. The top share is 0.522 to 0.789.

**Question 2, the task link. PASSED.** Pooled over 100 episodes rho is 0.388 with a
one-sided permutation p of 0.0005. Rho is above 0 at 8 of 10 seeds and the rule asks for 7.

**Gate B7 as a whole PASSED.** This is the first pass of the full gate at 10 seeds. The
configuration is the Gate B6 configuration without `--learned-valence`.

What this does NOT show, stated with the result.

1. The confound with the episode index is not excluded. The audio share rises with the
   episode number at all 10 seeds, with rho 0.45 or more at 5 of them.
2. The gate judges which module has the highest bid. It does not judge which vector the
   policy receives. With the default `--broadcast-merge legacy`, when two modules pass the
   threshold the broadcast holds the vector of the weaker one
   (`models/core/global_workspace.py`, lines 116 to 121). How often that occurs was not
   measured by this gate.
3. Gate B6 and Gate B7 use different seeds. The pair does not show what the learned
   valence adds or removes at a given seed.
4. A passed criterion is not evidence of affect. Finding a light is automatic approach.

No default is changed and no indicator moves.

## Result, 2026-09-16: Gate B3 FAILED

Seed 55 fired the KILL rule (vision 0.961 of ignited steps) and failed criterion (1) with a
runner-up share of 0.039. Seeds 54 and 56 passed criteria 1 to 3. Criterion (4) passed for
the first time: pooled rho 0.446, one-sided p 0.005, rho above 0 at all 3 seeds. The full
table is in `docs/results/dark_room_senses_2026_09.md`. The measured confound with the
episode index is stated there and is not excluded.
