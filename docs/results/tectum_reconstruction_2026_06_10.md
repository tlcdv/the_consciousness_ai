# Reconstruction objective on tectum_content (2026-06-10): FAILED, the collapse is architectural

The 2026-06-09 perception-decodability probe localized the agent's competence
bottleneck to the `obs_map -> tectum_content` collapse: a linear decoder reads
stimulus identity (shape/color/size) off the spatial `obs_map` perfectly (1.000) but
only at chance off the 256-D `tectum_content`. The probe's own hypothesis (marked as
such) was that a reconstruction / variational-free-energy objective would close the
gap, because reconstructing the frame forces the latent to preserve shape/color,
which the current tectum objectives (reward-prediction MSE + TDANN) do not ask for.

This is the test of that hypothesis. It FAILED.

## What was built

`models/core/tectum_reconstruction.py`: `TectumReconstructionHead` reconstructs the
downsampled current frame from `tectum_content`. Gradient flows into the tectum
(encoder, RSSM, capsules) through `tectum_content`, the target frame is stop-grad.
Wired into the tectum optimizer alongside the reward predictor, mirroring the
existing `control_representation.py` pattern. Behind `--enable-recon` (default off,
baseline bit-identical). `recon_loss` logged to `metrics.csv`. Two variants:

- naive: plain MSE over all target pixels.
- foreground (`recon.foreground=True`, the training default): per-element weighted
  MSE, weights proportional to each pixel's deviation from the frame mean, so the
  stimulus dominates and the trivial black background does not.

## Method

For each variant: train DMTS 100 episodes, 200 steps, seed 42, `--enable-recon
--save-tectum`, then re-run the perception-decodability probe with `--load-tectum`
on the saved tectum. The reconstruction objective trains the tectum every 5 steps
regardless of whether the agent learns the RL task, so this isolates the perception
fix from RL competence. The reward-only baseline is the existing
`runs/train_dmts_100/tectum.pt` checkpoint, re-probed in the same session for fresh
numbers under the identical probe protocol (seed 42).

## Results (DMTS sample phase, test accuracy; all loaded from disk this session)

| feature | reward-only baseline | recon (naive) | recon (foreground) | obs_map ceiling | uniform chance | majority |
|---------|---------------------:|--------------:|-------------------:|----------------:|---------------:|---------:|
| shape (6) | 0.179 | 0.275 | 0.233 | 1.000 | 0.167 | 0.275 |
| color (6) | 0.163 | 0.225 | 0.179 | 1.000 | 0.167 | 0.225 |
| size (2)  | 0.542 | 0.525 | 0.454 | 1.000 | 0.500 | 0.525 |

Reconstruction-loss trajectories (both objectives trained, the loss fell):

| variant | recon_loss first-50 | recon_loss last-50 | recon_loss min |
|---------|--------------------:|-------------------:|---------------:|
| naive | 2.17e-03 | 4.88e-04 | 4.86e-06 |
| foreground | 1.56e-02 | 5.10e-03 | 4.99e-05 |

Episode reward stayed flat and negative in both (naive mean -11.94, foreground mean
-11.70); the agent does not learn DMTS, as expected and as in the reward-only
baseline.

## Verdict: FAILED

Neither variant moves `tectum_content` off the chance/majority baseline. Every value
sits within noise of its `majority` or `uniform_chance` column; none approaches the
1.000 that `obs_map` carries. The stimulus identity is present upstream (obs_map =
1.000) and absent in the 256-D content after the collapse, and reconstruction
pressure does not put it back.

The foreground variant is the load-bearing result. The naive run was ambiguous: its
loss fell to 5e-4 by reconstructing the mostly-black background, so the stimulus
contributed almost nothing and "the objective was too weak / diluted" was a live
explanation. The foreground variant removes that explanation. Its loss is an order
of magnitude higher (the stimulus now dominates the target) and still falls 3x, so
the objective is genuinely training the tectum to reconstruct the stimulus region,
and `tectum_content` still cannot decode even color, the most trivial global feature.

So the collapse is architectural, not a missing-objective problem. The RSSM + capsule
composition + workspace projection that compress 16384-D `obs_map` into 256-D
`tectum_content` discard stimulus identity, and a reconstruction gradient flowing
back through that bottleneck does not reshape the 256-D code to preserve it within
this capacity and training budget. The probe is a separate linear classifier on
`tectum_content`; if identity were linearly present it would be found, and it is not.

## Caveats (do not over-read)

- Single seed, single configuration, conv-fallback (untrained DINOv2 projection),
  DMTS only, 100 episodes. A measurement, not a law.
- "Architectural" here means: not fixed by this reconstruction objective at this
  capacity/budget. A much higher-capacity decoder, a wider `tectum_content`, or a
  far longer run could in principle differ. What is ruled out is the specific,
  cheap hypothesis the probe named (a plain reconstruction term closes the gap).
- The agent never learns DMTS, so there is no task-reward signal interacting with
  the reconstruction objective. That is the point (it isolates perception), but it
  also means this does not test reconstruction-plus-competence jointly.

## What this overturns and establishes

- Overturns the probe's optimistic hypothesis that "a reconstruction / free-energy
  objective would close the gap" as stated. Adding the likelihood term alone does
  not.
- Establishes that the lossy stage resists identity-preserving pressure applied at
  `tectum_content`. The information is destroyed inside the obs_map -> tectum_content
  compression and not recoverable by gradient at the output of that compression.

## Next directions (gated, FAILED-first, >= 3 seeds before any default change)

1. Route the policy / workspace at `obs_map` (which is decodable, 1.000) rather than
   the collapsed `tectum_content`. This is the assessment's cheaper workaround. It
   is currently BLOCKED by a latent bug found this session: `--policy-input spatial`
   sets `policy_input_dim` but `ActionSelectionCore` sizes its PFC from
   `workspace_dim` and ignores it, so the flag crashes the Go/No-Go policy
   (`RuntimeError: input has inconsistent input_size: got 16384 expected 256`); it
   only ever worked with the `--policy dqn` diagnostic. Fixing it also needs the
   memory/replay path to keep the raw 256-D broadcast (as the P3 self-vector path
   already does) so memory coherence is unaffected.
2. Change the bottleneck itself: a wider `tectum_content`, an `obs_map` skip
   connection into the workspace, or a composition that preserves identity rather
   than pooling it away. Larger, biology-respecting change.
3. Keep the reconstruction head as a documented default-off negative result (as
   `control_representation.py` is kept), available as the likelihood term if the
   bottleneck is later widened or if a higher-capacity decoder is tried.

The reconstruction objective is the likelihood term of the active-inference
unification (`docs/active_inference_unification.md`). This result says that term,
applied at `tectum_content` at this capacity, is not sufficient on its own; it does
not retire active inference, it constrains where the objective must act (upstream of
or at the collapse, not at its output).

## Reproduce

```
export PYPHI_WELCOME_OFF=yes
# baseline re-probe (existing reward-only checkpoint)
python -m scripts.analysis.probe_perception_decodability \
    --episodes 2 --no-broadcast --envs dmts --seed 42 \
    --load-tectum runs/train_dmts_100/tectum.pt --out-dir runs/probe_baseline_reprobe

# naive reconstruction
python -m scripts.training.train_rlhf --env dmts --episodes 100 --max-steps 200 \
    --seed 42 --enable-recon --phi-sample-every 5 \
    --save-tectum runs/train_dmts_100_recon/tectum.pt --log-dir runs/train_dmts_100_recon
# (set recon.foreground=False in build_config to reproduce the naive variant exactly)
python -m scripts.analysis.probe_perception_decodability \
    --episodes 2 --no-broadcast --envs dmts --seed 42 \
    --load-tectum runs/train_dmts_100_recon/tectum.pt --out-dir runs/probe_recon

# foreground-weighted reconstruction (training default)
python -m scripts.training.train_rlhf --env dmts --episodes 100 --max-steps 200 \
    --seed 42 --enable-recon --phi-sample-every 5 \
    --save-tectum runs/train_dmts_100_reconfg/tectum.pt --log-dir runs/train_dmts_100_reconfg
python -m scripts.analysis.probe_perception_decodability \
    --episodes 2 --no-broadcast --envs dmts --seed 42 \
    --load-tectum runs/train_dmts_100_reconfg/tectum.pt --out-dir runs/probe_reconfg
```

Unit tests: `pytest tests/test_tectum_reconstruction.py -q` (8 tests).
