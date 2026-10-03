# FAILED: vision and hearing now compete, and the workspace is silent on two thirds of steps

> **Update, same day (Gate B2 below).** A tolerance ignition rule, chosen in an offline
> replay and tested live on fresh seeds 48 to 50, removed the silence (0.307 to 0.316) with
> real competition (audio wins 0.120 to 0.501 of ignited steps) and selective ignition.
> Gate B2 still FAILED on the task criterion: hearing does not win more when the light is
> out of view (pooled rho -0.128, p 0.904).
>
> **Update 2026-09-16 (Gate B3 below). FAILED on the KILL rule at seed 55**, where vision
> takes 0.961 of ignited steps and the runner-up 0.039. With reliability-weighted bids
> (`--bid-precision gain`) the TASK criterion passed for the first time. Pooled rho 0.446,
> one-sided p 0.005, rho above 0 at all 3 seeds. A confound is measured and not excluded.
> The audio share also rises with the episode index (rho 0.85 / 0.61 / 0.15).
>
> **Update 2026-09-16 (Gate B4 below). PASSED every criterion at 3 more seeds**, with the
> agent drawn as the project mark in the site colour (`--dark-room-agent-mark ring`). This is
> the first full pass. It is NOT a repeat of Gate B3, because the seeds differ and the pixels
> the agent receives differ. The episode-index confound is still there at 2 of 3 seeds.
>
> **Update 2026-10-03 (Gate B5 below). Gate B4 is reproduced bit for bit, and Gate B5
> FAILED on the task criterion.** From 2026-09-25 to 2026-10-03 a defect in the seeding
> call moved the dark room layouts by one episode. With the moved layouts the Gate B4
> seeds 57 to 59 FAILED the gate (kill rule and competition at seed 57, task pooled rho
> 0.289, p 0.084). Gate B5, written before its runs, used fresh seeds 60 to 62, also with
> moved layouts. Competition, silence, selectivity and the kill rule passed at all 3
> seeds. The task criterion FAILED (pooled rho 0.200, one-sided p 0.095). The agent is
> the same in all of these runs. The gate result depends on the layouts, so 3 seeds do
> not settle the task link.
>
> **Update 2026-10-03 (Gate B6 below). FAILED on competition at 3 of 10 fresh seeds. The
> task link PASSED.** Gate B6 was written before its runs and used seeds 63 to 72 on the
> code with the seeding call repaired. Pooled over 100 episodes, hearing wins a larger
> share in episodes where the light is out of view (rho 0.362, one-sided p 0.002, rho
> above 0 at 8 of 10 seeds). At seeds 64, 65 and 68 vision took 0.968 to 0.996 of ignited
> steps and fired the kill rule, so the gate as a whole FAILED. The confound with the
> episode index is not excluded.

**Gate B FAILED at all 3 seeds** on silence and on the task variable. **Its competition
criterion passed at all 3 seeds:** hearing wins 17 to 25 percent of ignited steps, the
first time a second module has competed in this project (vision won 99 to 100 percent
before). The silence is the workspace ignition rule, not the senses.

Before the learning changes, the dark room itself was repaired and checked. That check
failed twice because of the gate design and passed on the third, pre-stated gate on
fresh seeds. All three results are recorded below.

No indicator moves. The Butlin rubric is unchanged. Nothing here is evidence of affect:
Feinberg and Mallatt exclude automatic approach and avoidance as evidence, and finding a
light is that kind of task.

## Why this work was done

On three seeds (1800 steps) the original dark room gave the agent almost nothing to
hear and everything to see (`modality_starvation_2026_09.md`, correction header): the
proximity tone needed a distance below 10 in a room 224 wide, the collision sound never
played, the sound was mono, the audio noise had no seed, the audio salience network was
in no optimizer, and the frame showed the whole room from above.

## What was built (all default off; the default run is byte-identical)

| Flag | What it does |
|---|---|
| `--dark-room-view agent_centered` | The agent sees a 96-pixel window of the 224-pixel room around itself; outside the room is gray |
| `--dark-room-audio binaural --dark-room-audio-channels 4` | The light emits a tone heard across the room; left/right and upper/lower ear pairs with time and level differences |
| `--dark-room-collision` | A wall contact is reported and plays the collision sound; touch, not damage (ethics rule E6) |
| `--vision-bid-reduction zscore` | Vision bid = sigmoid of a running z-score of the tectum KL, instead of tanh of the KL (candidate S5 in `bid_counterfactual_2026_09.md`) |
| `--audio-salience surprise` | Audio bid = running z-score of the error of predicting the next cochlear band profile; the predictor trains itself and habituates |
| `--learned-valence` | Per-module values learned from the temporal difference error of the external task reward replace the fixed approach/threat module sets |

Also fixed on the way: the environment audio noise is now seeded from `--seed`; the
stereo direction estimator had opposite signs for its time and level terms; a bid
normalizer overflow on large KL jumps; the running variance start of 1.0 hid changes in
small signals, so the audio normalizer starts at 0.0.

Check: `metrics.csv` md5 `7c101c58eecc4eb91409c491a5ad0e26` for the standard 2-episode
dark_room run with default flags, before and after every change.

## Gate A: do the repaired senses carry information?

Probe: `scripts/analysis/probe_dark_room_senses.py`. Every gate was written into its
docstring before its run.

**Gate A, FAILED.** Seeds 42 to 44, sessions driven by the untrained agent. Criterion (ii)
failed at seed 42 (sound level against distance, rho -0.768 against a limit of -0.8).
Criterion (i) was not measurable: the agent stayed at one wall (collisions on 90 to 98
percent of steps), the light's direction changed sign at most 4 times per seed, and a
shuffled control scored 0.79 to 1.00.

**Gate A2, FAILED.** Seeds 42 to 44, environment driven by a random-heading controller.
Two gate-design errors: a shuffled control is not a valid chance level when one sign
dominates (0.8 squared plus 0.2 squared is 0.68), and the view criterion used one episode,
which is one random start.

**Gate A3, PASS at all 3 seeds.** Corrected statistics, seeds 45 to 47 never run before.

| Criterion | Limit | seed 45 | seed 46 | seed 47 |
|---|---|---|---|---|
| balanced sign accuracy, left/right | at least 0.90 | 0.9994 | 0.9990 | 0.9970 |
| balanced sign accuracy, up/down | at least 0.90 | 0.9984 | 0.9985 | 0.9993 |
| shuffled control, left/right and up/down | at most 0.60 | 0.506, 0.523 | 0.496, 0.521 | 0.514, 0.477 |
| sound level against distance, no-collision steps | -0.8 or stronger | -0.998 | -0.994 | -0.997 |
| light out of view, all steps | at least 0.50 | 0.560 | 0.729 | 0.644 |
| collisions | at least 1 | 213 | 149 | 134 |
| reach the light: sound-only rule against random heading | margin at least 0.10 | 1.00 against 0.58 | 1.00 against 0.56 | 1.00 against 0.42 |

The sound-only rule has no learning: keep the heading while the sound gets louder, pick
a new random heading when it gets quieter.

## Gate B: do the modules compete, and does hearing win when the light is out of view?

Probe: `scripts/analysis/probe_gate_b.py`. Seeds 42 to 44, 10 episodes of 200 steps, all
flags above on, existence drive on. Row counts verified: 2000 metric rows and 10 complete
episode records per seed.

| Criterion | Limit | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| (1) runner-up share of ignited steps | at least 0.05 | **0.172 audio** | **0.253 audio** | **0.231 audio** |
| (2) steps without a winner | below 0.50 | **0.655 FAILED** | **0.694 FAILED** | **0.615 FAILED** |
| (3) audio share out of view minus in view | above null p95 | **-0.160 FAILED** (p95 +0.027) | **+0.428 FAILED** (p95 +0.428) | **+0.209 FAILED** (p95 +0.209) |
| KILL: one module at least 0.95 | | not triggered (0.828) | not triggered (0.747) | not triggered (0.769) |

At seed 43, episodes 8 and 9 never ignite.

**A weakness in criterion (3), found after the run.** The view state is constant for a
whole episode in 2 of 10, 4 of 8 and 5 of 10 ignited episodes, so shuffling 20-step
blocks within episodes leaves those labels unchanged, and 1, 9 and 12 percent of null
draws equal the observed value. The criterion cannot separate the view state from
changes between episodes. Audio wins in runs of whole episodes (seed 44, episodes 4 to 7:
0.81 to 0.95), not step by step. That pattern is recorded here and is not a result.

## Why the workspace is silent

All silence is non-ignition. Share of steps not ignited: 0.656 / 0.695 / 0.616; ignited
but without a winner: 0.000 at every seed. The median ignition salience is -0.059 /
0.000 / -0.046.

The rule is `input_energy >= EMA(input_energy)` with an averaging factor of 0.95
(`models/core/global_workspace.py`), so the input is compared with its own average over
about 20 steps. When vision was fixed at 1.0, the input sat on its own average and the
rule was satisfied. With bids that measure change, the input falls below its recent
average on most steps. The arousal-adjusted `ignition_threshold` does not take part in
the ignition decision; it only sets the soft threshold for choosing winners. This was
predicted by `bid_counterfactual_2026_09.md`, which saw lowering the vision bid silence
the workspace instead of passing the win.

**Silence feeds itself inside each step.** The reentrant loop runs the competition up to
5 times per step, and the running average moves on every cycle. Measured: 1990 / 1989 /
1976 of 2000 steps settle in 2 cycles, and every step that took 3 cycles ignited. With an
empty broadcast, a specialist's `receive_broadcast` lowers its bid by 5 percent, so the
second cycle compares a lower input with an average that already holds the first. The
largest bid's change during settle has a median of -0.030 / 0.000 / -0.027 on silent
steps and +0.218 / +0.100 / +0.107 on ignited steps. The first cycle decides the step,
and the feedback repeats that decision.

## Offline replay of ignition options

Probe: `scripts/analysis/probe_ignition_replay.py`. The recorded Gate B inputs were
replayed through a real workspace and settle loop; the specialists' feedback used the real
`receive_broadcast` rules. Gates were written before the replay.

**Fidelity PASS** with the current rule: ignition agreement 0.973 / 0.970 / 0.997, silence
0.655 against 0.656 recorded / 0.726 against 0.695 / 0.615 against 0.616.

| Option | Silence (need below 0.50) | Runner-up | Verdict |
|---|---|---|---|
| V1: average moves once per step, read before update | 0.681 / 0.305 / 0.621 | 0.167 / 0.081 / 0.232 | not promising |
| V2: an empty broadcast leaves bids unchanged | 0.692 / 0.714 / 0.655 | 0.188 / 0.234 / 0.241 | not promising |
| V3: V1 with ignition at input >= average - 1.0 running sd | **0.289 / 0.130 / 0.286** | **0.181 / 0.205 / 0.174** | **promising at all 3 seeds** |
| V12: V1 and V2 | 0.720 / 0.309 / 0.654 | 0.185 / 0.098 / 0.232 | not promising |

Removing the per-cycle movement of the average or the bid cut alone does not remove the
silence; the tolerance band does. This is an open-loop replay: actions and learning did
not respond to the new rule. It does not show that V3 ignition still carries information
(it ignites on 0.71 to 0.87 of steps), and it does not test the task variable.

## Gate B2: the tolerance ignition rule, live, on fresh seeds

`--ignition-rule tolerance --ignition-tolerance-sd 1.0` implements V3; on the seed 42
replay its ignition and winners agree with V3 on 1.0000 of steps. Because V3 was chosen
on seeds 42 to 44, Gate B2 used seeds 48 to 50, never run before. Probe:
`scripts/analysis/probe_gate_b2.py`, gate written before the runs.

**Gate B2 FAILED on the task criterion. Competition, silence and selectivity passed at
all 3 seeds.**

| Criterion | Limit | seed 48 | seed 49 | seed 50 |
|---|---|---|---|---|
| (1) runner-up share of ignited steps | at least 0.05 | 0.392 audio | 0.499 (audio wins 0.501) | 0.120 audio |
| (2) steps without a winner | below 0.50 | 0.316 | 0.314 | 0.307 |
| (3) top-bid difference, ignited minus silent | above null p95 | 0.050 (p95 0.022) | 0.115 (p95 0.010) | 0.099 (p95 0.055) |
| (4) rho, audio share against out-of-view share, per episode | pooled above 0, p below 0.05 | -0.622 | -0.400 | +0.375 |

Pooled over 30 episodes: rho -0.128, one-sided p 0.904. Reported only: the audio share
follows the episode index (rho 0.770 / -0.248 / 0.517) more than the view state.

A possible reason, from the design and not measured: the audio bid measures change in
the sound, and no bid carries how reliable vision is at that moment. The rule that one
sense counts more when the other is weak exists inside the tectum fusion, not in the
workspace competition.

## Why the task criterion failed, and a reliability check that also FAILED

The bids move against the hypothesis (Gate B2 records, 1990 steps per seed): the audio bid
is higher with the light in view at all 3 seeds (0.517 against 0.419 out of view /
0.509 against 0.437 / 0.480 against 0.433), and the vision bid does not fall in darkness
(0.554 in view against 0.706 out / 0.425 against 0.474 / 0.734 against 0.622). Surprise
measures change, not how useful a sense is.

The multisensory literature weights each cue by its reliability: sound dominates when
vision is blurred and poorly localized (Alais and Burr 2004), cues combine in proportion to
inverse variance (Ernst and Banks 2002), and MSTd activity predicts reliability weights
(Fetsch et al. 2012). Before building reliability-weighted bids, a read-only check
(`scripts/analysis/probe_sense_reliability.py`, gate written before computing) tested two
task-agnostic measures on the Gate B2 records.

**FAILED at seeds 48 and 50; nothing was built.** Vision reliability (concentration of
change in the pooled frame) separated the light in view with AUC 0.618 / 0.990 / 0.554,
and the audio weight separated it out of view with AUC 0.512 / 0.987 / 0.528, against a
limit of 0.70. Audio coherence alone was higher with the light in view (AUC for out of
view 0.101 / 0.673 / 0.300). Hypothesis, not a result: the running mean that removes the
agent's own disc also removes a light that stays in view.

## The gain check (FAILED) and Gate B3 (FAILED on the KILL rule), 2026-09-16

A second read-only check (`scripts/analysis/probe_sense_gain.py`, gate written before the
runs) replaced the change-based measure with the GAIN of the current response, which the
sources say carries reliability and does not habituate (Ma et al. 2006; Fetsch et al. 2009
and 2011; Feldman and Friston 2010; Stein and Stanford 2008). Vision gain is the strongest
pooled cell of the frame outside the agent's own disc; audio gain is the waveform RMS over
the tone RMS at the source.

**FAILED at seed 51** on new seeds 51 to 53: criteria R1 (0.997 / 0.981 / 0.904) and R2
(0.996 / 0.961 / 0.857) passed, and R3 (no habituation) could not be measured at seed 51,
where the light stayed in view long enough only once (1 usable run, 3 required). R1 and R3
can also pass by the construction of the room, as the probe states before its numbers. The
owner recorded an override on 2026-09-16 and asked for a live test
(`docs/decisions/2026_09_16_precision_weighted_bids.md`).

Built: `models/core/precision_weighting.py` behind `--bid-precision gain` (default off,
md5 `7c101c58eecc4eb91409c491a5ad0e26` unchanged). Each sense bid is multiplied by twice its
share of the total gain, so equal gains change nothing.

**Gate B3 FAILED.** The Gate B2 criteria, unchanged, on new seeds 54 to 56, 1990 steps each:

| Criterion | Seed 54 | Seed 55 | Seed 56 |
|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.225 | **0.039** | 0.315 |
| (2) silence, below 0.50 | 0.303 | 0.208 | 0.335 |
| (3) selectivity, top-bid difference against null p95 | 0.134 / 0.054 | 0.084 / 0.045 | 0.095 / 0.015 |
| KILL, a module at 0.95 or more | 0.775 | **0.961 vision** | 0.685 |

**(4) TASK PASSED for the first time:** pooled Spearman rho 0.446 over 30 episodes,
one-sided permutation p 0.005, and rho above 0 at all 3 seeds (0.554 / 0.386 / 0.550). The
gate still FAILS, because the KILL rule fired at seed 55 and criterion (1) failed there.

Confound, measured and not excluded. The audio share also rises with the episode index
(rho 0.853 / 0.607 / 0.152), which the gate lists as reported only. The task link and
learning over time are not separated by this design.

## Gate B4, the same criteria with the agent drawn as the project mark. PASSED

**Note added 2026-10-03.** The three runs of this gate are reproduced bit for bit by the
code with the seeding call repaired, at seeds 57, 58 and 59. The same seeds with the
layouts moved by one episode fail the gate. Read this section together with Gate B5 and
Gate B6 below.

The agent's own body is drawn into the frames it receives, so its shape and colour are run
configuration. `--dark-room-agent-mark ring --dark-room-agent-colour 217,119,87` draws two
rings and a centre dot in the site colour, and the run facts record it
(`docs/decisions/2026_09_16_precision_weighted_bids.md`). New seeds 57 to 59, same flags as
Gate B3 otherwise, same criteria, 1990 steps each.

| Criterion | Seed 57 | Seed 58 | Seed 59 |
|---|---|---|---|
| (1) runner-up share, at least 0.05 | 0.072 | 0.202 | 0.488 |
| (2) silence, below 0.50 | 0.133 | 0.247 | 0.322 |
| (3) selectivity, difference / null p95 | 0.083 / 0.011 | 0.249 / 0.159 | 0.095 / 0.017 |
| KILL, a module at 0.95 or more | 0.928 | 0.798 | 0.512 |
| (4) task, rho per seed | 0.562 | 0.275 | 0.736 |

Criterion 4 pooled over 30 episodes gives rho 0.458 and a one-sided permutation p of 0.006.
**Every criterion passed at every seed. This is the first full pass of this gate.**

Four limits stated with the pass.

1. It is not a repeat of Gate B3. The seeds differ and so do the pixels, so the two gates are
   separate measurements.
2. The episode-index confound stands. The audio share rises with the episode number at 2 of
   the 3 seeds (rho -0.194 / 0.512 / 0.815).
3. Seed 57 reaches 0.928, close to the 0.95 kill limit.
4. A passed gate is not evidence of affect. Finding a light is automatic approach, which
   Feinberg and Mallatt exclude as evidence.

## Gate B5, the Gate B4 criteria on the current code. FAILED on the task criterion

Written before its runs (`docs/decisions/2026_09_16_precision_weighted_bids.md`, Gate B5).
Two sets of runs with the Gate B4 flags and the default `--broadcast-merge`. Both were
made while the seeding defect was present, so each episode received the layout of the next
episode of its seed. The agent code is the same as in Gate B4. The runs are deterministic
and repeat at revision `b1c1afb`. Two runs of seed 57 gave identical files.

The first set uses the Gate B4 seeds 57 to 59 with the moved layouts. It is not a gate,
because it was run before a gate was written for it
(`docs/results/audit_gpu_checks_2026_10.md`). The second set is Gate B5 at fresh seeds 60 to
62, one run per seed, 1990 steps judged per run.

| Criterion | Seeds 57 / 58 / 59, moved layouts | Gate B5, seeds 60 / 61 / 62 |
|---|---|---|
| (1) runner-up share, at least 0.05 | **0.017** / 0.127 / 0.356 | 0.292 / 0.166 / 0.359 |
| (2) silence, below 0.50 | 0.104 / 0.239 / 0.307 | 0.298 / 0.332 / 0.320 |
| (3) selectivity, difference | 0.140 / 0.240 / 0.156 | 0.160 / 0.119 / 0.121 |
| (3) selectivity, null p95 | 0.072 / 0.160 / 0.058 | 0.064 / 0.067 / 0.034 |
| KILL, a module at 0.95 or more | **0.983** / 0.873 / 0.644 | 0.708 / 0.834 / 0.641 |
| (4) task, rho per seed | 0.181 / 0.239 / 0.577 | 0.294 / 0.007 / 0.063 |
| (4) task, pooled rho and one-sided p | **0.289, 0.084** | **0.200, 0.095** |
| Audio share against episode index, rho | -0.149 / 0.460 / 0.450 | -0.067 / 0.779 / 0.511 |
| Probe verdict | FAILED | FAILED: (4) task |

Bold marks a failed criterion. Vision is the top module in all 6 runs.

**Gate B5 FAILED.** Criteria 1 to 3 and the kill rule passed at all 3 fresh seeds. The
task criterion failed. The pooled rho is above 0 and its one-sided p of 0.095 is above the
limit of 0.05.

Read over these 6 runs.

1. Competition, silence and selectivity hold at 5 of 6 seeds. Seed 57 fails on the kill
   rule, where vision takes 0.983 of ignited steps.
2. The task link is not shown. It fails in both sets. Per seed rho is above 0 at all 6
   seeds and is close to 0 at seeds 61 and 62.
3. The task criterion passed in Gate B3 (which FAILED on the kill rule) and in Gate B4,
   and failed in both sets here. Per seed rho is above 0 in all 12 runs of these four
   sets. Three seeds do not settle the link.
4. The episode-index confound stands. The audio share rises with the episode number at 4
   of 6 seeds (rho 0.45 or more).

## Gate B6, 10 fresh seeds on the repaired code. FAILED on competition at 3 seeds, the task link PASSED

Written before its runs (`docs/decisions/2026_09_16_precision_weighted_bids.md`, Gate B6),
with two questions and a separate verdict for each. Seeds 63 to 72, Gate B4 flags, default
`--broadcast-merge`, revision `5ad584e`. 1990 steps judged per run and 100 qualifying
episodes. The first run of seed 71 stopped after 6 seconds with exit code 127 and no step
on disk. It was run again with the same seed, as the gate rule states.

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

**The task link PASSED.** Pooled rho 0.362 over 100 episodes, one-sided permutation p 0.002.
Rho is above 0 at 8 of 10 seeds, and the rule asks for 7. The 2 seeds below 0 are seeds 65
and 68, where hearing won 0.024 and 0.004 of ignited steps.

**Competition at every seed FAILED.** Seeds 64, 65 and 68 fired the kill rule and failed
criterion (1). Seeds 63, 66, 67, 69, 70, 71 and 72 passed criteria (1) to (3).

**Gate B6 as a whole FAILED**, because both questions must pass.

Three limits stand with this result.

1. The confound with the episode index is not excluded. The audio share rises with the
   episode number at 8 of 10 seeds, with rho 0.45 or more at 4 of them.
2. At 3 of 10 fresh seeds vision takes 0.95 or more of ignited steps. The cause is open.
3. A passed criterion is not evidence of affect. Finding a light is automatic approach.

## What this establishes

- The repaired dark room gives the agent direction and distance to the light by sound,
  and weak vision (Gate A3, 3 fresh seeds).
- With change-based bids, hearing competes with vision for the workspace at 3 seeds.
- With reliability-weighted bids and the ring mark, hearing competes with vision at 3 of 3
  seeds in Gate B4, at 5 of the 6 runs with moved layouts (not seed 57) and at 7 of 10
  seeds in Gate B6.
- With the same flags, hearing wins a larger share of ignited steps in episodes where the
  light is out of view. Gate B6, 10 fresh seeds and 100 episodes, pooled rho 0.362,
  one-sided p 0.002. This is a correlation across episodes, and the episode index is a
  measured confound.
- The vision bid no longer sits at a fixed ceiling.
- The ignition rule, not the senses, now blocks the workspace on most steps.

## What this does NOT establish

- **No cause for the task link.** The criterion passed in Gate B3, Gate B4 and Gate B6,
  and FAILED at 3 seeds with the layouts moved by one episode (seeds 57 to 59, pooled rho
  0.289, p 0.084) and in Gate B5 (seeds 60 to 62, pooled rho 0.200, p 0.095). The design
  does not separate the task from learning over time, because the audio share also rises
  with the episode number.
- **No stable competition at every seed.** Vision takes 0.95 or more of ignited steps at
  seed 55 (Gate B3), at seed 57 with moved layouts, and at seeds 64, 65 and 68 (Gate B6).
- **No task variable with change-based bids.** In Gate B2 hearing does not measurably win
  more when the light is out of view. With reliability-weighted bids the task criterion
  passed at seeds 54 to 56, but that gate FAILED on the KILL rule, and the confound with
  the episode index is not excluded. A passed criterion inside a failed gate is not a
  result.
- **Nothing about content or learning.** 10 episodes, untrained start, one agent
  configuration. The untrained agent pushes against a wall for most steps.
- **No affect.** Approaching a light is excluded as evidence of affect.
- **No biological fidelity claim.** The ear model, the surprise predictor and the linear
  valence critic are engineering stand-ins. The research basis (Feinberg and Mallatt on
  egocentric aligned tectal maps and separate valence; Bennett on adaptation and temporal
  difference learning; Stein and Meredith on multisensory rules; Knudsen on map
  calibration) is paraphrased from secondary summaries and source notes; details are not
  verified against the books.
- **Runs with the existence drive on.** Growth stages would need it off (ethics rule E3);
  that configuration is not tested here.

## Next

1. Done the same day: the tolerance ignition rule (Gate B2 criteria 1 to 3).
2. Done 2026-09-16: reliability-weighted bids (`--bid-precision gain`), judged by Gate B3,
   which FAILED on the KILL rule at seed 55.
3. Open: why vision takes 0.961 of ignited steps at seed 55 while the same flags give
   0.775 and 0.685 at the other two seeds.
4. Open: separate the task link from learning over time. The audio share rises with the
   episode index in these runs.
5. An agent that moves: the untrained policy stays at walls in the agent-driven sessions.
6. Done 2026-10-03: Gate B5 at fresh seeds 60 to 62. FAILED on the task criterion. Open:
   why vision takes 0.983 of ignited steps at seed 57 when the layouts move by one
   episode.
7. Done 2026-10-03: Gate B6, 10 fresh seeds on the repaired code. The task link PASSED
   and competition FAILED at 3 of 10 seeds.
8. Open: why vision takes 0.95 or more of ignited steps at some seeds (55, 57 with moved
   layouts, 64, 65, 68).
9. Open: a design that separates the task link from learning over time.

## Reproduce

```
python -m scripts.analysis.probe_sense_gain --runs runs/gain_check_s51 runs/gain_check_s52 runs/gain_check_s53
python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b3_s54 runs/gate_b3_s55 runs/gate_b3_s56
python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b5_s60 runs/gate_b5_s61 runs/gate_b5_s62
python -m scripts.analysis.probe_gate_b2 --runs runs/gate_b6_s63 runs/gate_b6_s64 runs/gate_b6_s65 \
    runs/gate_b6_s66 runs/gate_b6_s67 runs/gate_b6_s68 runs/gate_b6_s69 runs/gate_b6_s70 \
    runs/gate_b6_s71 runs/gate_b6_s72

python -m scripts.analysis.probe_dark_room_senses --gate-a3 --seeds 45 46 47

python -m scripts.training.train_rlhf --env dark_room --episodes 10 --max-steps 200 \
  --seed 42 --enable-audio --rssm-latent-mode continuous \
  --capsule-workspace-source all_levels --existence-drive on \
  --dark-room-audio binaural --dark-room-audio-channels 4 --dark-room-collision \
  --dark-room-view agent_centered --vision-bid-reduction zscore \
  --audio-salience surprise --learned-valence --log-dir runs/gate_b_s42
python -m scripts.analysis.probe_gate_b --runs runs/gate_b_s42 runs/gate_b_s43 runs/gate_b_s44
```

Each Gate B seed took 3 min 23 s to 3 min 41 s (measured from file times) and wrote
956 to 966 MB of session record.
