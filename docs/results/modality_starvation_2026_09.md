# One of five modules receives sensory data, and that is why the workspace is degenerate

> **CORRECTED 2026-09-15. The audio half of this document is wrong in its meaning.**
>
> **The varying audio bid came from noise, not from sound that carries information.**
> Repeated at 3 seeds with seeded audio and full session records
> (`runs/modality_dr_seeded_s4{2,3,4}`, 600 steps each, same flags):
>
> | | seed 42 | seed 43 | seed 44 |
> |---|---|---|---|
> | audio bid, distinct values | 590 | 588 | 582 |
> | vision bid, distinct values | 1 | 1 | 1 |
> | winner | vision 590, silent 10 | vision 596, silent 4 | vision 591, silent 9 |
> | closest distance to the light | 86.7 | 84.9 | 43.5 |
> | steps in the light | 0 | 0 | 23 |
>
> In those 1800 steps the dark_room sound carried almost nothing:
>
> - the proximity tone plays only below a distance of 10 in a room 224 wide; the distance
>   was never below 10;
> - the in-light chord peaks at 0.078 against 0.064 for noise alone, and on the 23
>   in-light steps the recorded peaks (0.060 to 0.074) were inside the noise range;
> - the collision sound never played, because the environment never reported a collision;
> - the sound was mono, so no direction reached the agent;
> - the auditory `salience_net` is in no optimizer, so the audio bid is an untrained network
>   responding to random noise.
>
> **The numbers below cannot be reproduced exactly.** The environment audio noise had no
> seed until 2026-09-15: the same command run twice gave different `bid_audio` from step 1
> (0.476660013 against 0.476288438). The audio is now seeded from `--seed`.
>
> What still stands: only vision receives informative sensory data on dark_room as well as
> on DMTS, the vision bid is fixed at 1.0, and vision wins every ignited step, now at 3
> seeds. What does not stand: "a module's bid becomes live as soon as its modality is
> present" as evidence that the audio module processes information, and the audio row of
> "the oscillators follow it" as evidence of anything but noise driving the amplitudes.

**On DMTS, the task every workspace, bid, binding and `sync_R` measurement in this
project has used, exactly ONE of the five competing modules receives real sensory
data.** The other four are fed zeros, literals, or a stub.

**This corrects `oscillator_frozen_2026_09.md`, published earlier the same day.** The
binding layer is not architecturally frozen. It is frozen ON DMTS, because every bid
feeding it is a constant. Given an environment that supplies a second modality, it
moves: `sync_R` goes from **1 distinct value to 553**.

No indicator moves. The rubric stays 3 IMPLEMENTED, 11 PARTIAL of 14. The clock does
not move.

## The structural finding, from code, no run required

| Module | What it receives on DMTS | Real data? |
|---|---|---|
| vision | 224x224 RGB frames from the environment | **YES** |
| audio | `torch.zeros(1, tectum_feature_dim, 2)` at `train_rlhf.py:1015` and `:1019`, because DMTS emits no `audio_waveform` | no |
| semantic | a mock embedding, and off by default | no |
| body | a literal 0.15 or 0.05 from an energy counter at `train_rlhf.py:1072-1073` | no |
| memory | `PineconeIndexStub`, score hardcoded 0.0 by default | no |

**No environment in the repository supplies proprioception at all.** A repo-wide search
for `raw_proprioception` and `proprioception` across `simulations/` and the training
loop returns nothing, so `ProprioceptiveProcessor` has no possible input.

**Three of the four task environments emit zero audio.** `dmts_env.py`,
`navigation_env.py` and `wcst_env.py` contain no audio reference. Only
`simple_visual_env.py`, the `dark_room` task, mixes in `DarkRoomAudioMixin`, which
generates genuine FM tones with ADSR envelopes at 16 kHz.

## The measurement: give a second module real data

Real training loop, `--env dark_room --enable-audio`, 3 episodes, 600 steps, seed 42,
with the perception-fixed flags. Logged bid columns read directly from
`runs/_modality_dr/metrics.csv`.

| Bid | distinct values | min | max | sd |
|---|---|---|---|---|
| `bid_vision` | 1 | 1.000000 | 1.000000 | 0.000e+00 |
| **`bid_audio`** | **587** | 0.000000 | **0.546942** | 3.848e-02 |
| `bid_memory` | 1 | 0.100000 | 0.100000 | 0.000e+00 |
| `bid_body` | 1 | 0.050000 | 0.050000 | 0.000e+00 |
| `bid_semantic` | 1 | 0.000000 | 0.000000 | 0.000e+00 |

**The audio bid is alive the moment there is audio.** On DMTS it has one distinct
value. Here it has 587.

And the oscillators follow it:

| Quantity | DMTS | dark_room with audio |
|---|---|---|
| `sync_R` distinct values | **1** | **553** |
| `sync_R` range | 0.450000 only | 0.217940 to 0.342056 |
| `sync_R` sd | 0.000e+00 | 9.026e-03 |

The mechanism is direct. The Kuramoto amplitudes ARE the bids
(`oscillatory_binding.py:165-167`). Constant bids give constant amplitudes, the
oscillators converge, and the layer stops moving. A varying bid keeps it moving.

## What it does NOT fix, and this is the useful half

**The winner is still vision, 590 of 600 steps.** Because `bid_vision` is pinned at
exactly 1.000000 by the `tanh` saturation at `sensory_tectum.py:456`, while the one
live competitor tops out at 0.546942. A live opponent cannot win against a ceiling.

So the two defects are complementary, and each alone is harmless to fix:

- **`bid_counterfactual_2026_09.md`**: de-saturating vision on DMTS produces no
  competition, because no opponent varies. KILL.
- **This document**: giving an opponent real data produces no competition, because
  vision is at the ceiling.

**Neither repair works alone. Together they are the first combination that could
produce a real competition.** That is a hypothesis, and it is the first one this line
of work has produced that is not already known to fail.

## What this establishes

- One of five modules is fed on DMTS, verified from code.
- No environment supplies proprioception, so the body module cannot be fed at all
  without new environment work.
- A module's bid becomes live as soon as its modality is present: 1 distinct value to
  587.
- The oscillatory binding layer is NOT architecturally frozen. It is starved.
- The vision ceiling, not the opponent's floor, is what keeps the winner fixed once a
  live opponent exists.

## What this does NOT establish

- **3 episodes, 600 steps, one seed. Hypothesis grade.** Under the standing rule this
  is not a result.
- **Effectively untrained.** After 3 episodes the auditory `salience_net` is near its
  initialization, so the audio bid's variation is a projection of varying input, not
  demonstrated learned salience. The structural claim, that the bid varies when the
  modality is present, does not depend on training. Any claim about WHAT it encodes
  does, and none is made.
- **It does not show the combination works.** Neither repair has been run together.
- **It does not test content.** Whether a live audio bid tracks the stimulus is
  untested, and the project's record on that question is eight failures.
- **It does not re-score GWT-1 or GWT-2.**

## Correction to `oscillator_frozen_2026_09.md`

That document says the binding layer "emits one value forever" and calls the cause a
missing `reset_state`. On DMTS both statements hold and the measurements stand. As a
statement about the ARCHITECTURE the framing was wrong: the layer emits one value
because its input is constant, and the missing reset is why it cannot recover once
converged. On `dark_room` with audio it never freezes in the first place.

The content result in that document is unaffected: the reset transient on DMTS
decodes the stimulus at 0.1721 against a null p95 of 0.2221, and that remains a
failure.

## Next

1. **The combined test.** De-saturated vision bid plus a genuinely fed second module,
   on `dark_room`, 3 seeds. This is the first configuration where a changing winner is
   even possible.
2. **Decide whether DMTS is the right task for workspace work at all.** It exercises
   one modality. Every degeneracy finding in this project was measured on it.
3. **The body module cannot be fixed from the model side.** It needs an environment
   that emits proprioception, which does not exist yet.

## Reproduce

```
python -m scripts.training.train_rlhf --env dark_room --episodes 3 --max-steps 200 \
  --seed 42 --enable-audio --rssm-latent-mode continuous \
  --capsule-workspace-source all_levels --log-dir runs/_modality_dr
```

About 6 minutes. Then read the `bid_*` and `sync_r` columns of
`runs/_modality_dr/metrics.csv`.
