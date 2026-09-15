# FAILED: vision and hearing now compete, and the workspace is silent on two thirds of steps

> **Update, same day (Gate B2 below).** A tolerance ignition rule, chosen in an offline
> replay and tested live on fresh seeds 48 to 50, removed the silence (0.307 to 0.316) with
> real competition (audio wins 0.120 to 0.501 of ignited steps) and selective ignition.
> Gate B2 still FAILED on the task criterion: hearing does not win more when the light is
> out of view (pooled rho -0.128, p 0.904).

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

## What this establishes

- The repaired dark room gives the agent direction and distance to the light by sound,
  and weak vision (Gate A3, 3 fresh seeds).
- With change-based bids, hearing competes with vision for the workspace at 3 seeds.
- The vision bid no longer sits at a fixed ceiling.
- The ignition rule, not the senses, now blocks the workspace on most steps.

## What this does NOT establish

- **No task variable.** Hearing does not measurably win more when the light is out of
  view; criterion (3) failed and its null is too weak to show an effect.
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
2. The task link is missing: a design decision on whether a sense's bid should carry how
   reliable that sense is at the moment (for example, vision when the light is out of
   view), as the multisensory rules describe for the tectum.
3. An agent that moves: the untrained policy stays at walls in the agent-driven sessions.

## Reproduce

```
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
