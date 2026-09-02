# The broadcast is ALIVE. Its content rides in 2 to 4 dimensions of 256, on a fixed offset

**The conscious content of this architecture carries stimulus identity, and every
instrument that reads its length or its raw distance is blind to that.** The broadcast
decodes which of 6 shapes was shown at 0.757, 0.691 and 0.771 against majority baselines
of 0.283, 0.291 and 0.228, with shuffled floors of 0.111, 0.128 and 0.239. Margins over
the shuffled floor are 0.65, 0.56 and 0.53, at 3 seeds, on a trial-grouped split.

Geometrically it is a large fixed vector plus a small motion: **95 percent of the variance
lives in 2 to 4 of its 256 dimensions**, and the moving part is 5 to 15 percent the size
of the fixed part.

Under the standing fork this is answer (a): the signal is there and the instruments are
mis-reading it. It is not answer (b).

3 seeds, 40 episodes, 8000 steps each, dmts, `--enable-audio --enable-mock-semantic`.

## Why this measurement exists

`ws_state = (broadcast_mag, phi, sync_r)` keeps only `broadcast.norm()`. The pipeline has
always recorded the broadcast's LENGTH and discarded its DIRECTION, so the direction had
never been measured during a run. A `broadcast.npy` sidecar now records the full
[steps, 256] matrix.

## Result: the content is there

Sample-phase steps only, 6-class `sample_shape`, split holding out whole trials chosen at
random.

| seed | steps | trials | held out | majority | broadcast | shuffled | margin |
|---|---|---|---|---|---|---|---|
| 42 | 2975 | 158 | 905 | 0.2829 | **0.7569** | 0.1105 | +0.6464 |
| 43 | 2837 | 154 | 881 | 0.2906 | **0.6913** | 0.1283 | +0.5630 |
| 44 | 2953 | 153 | 859 | 0.2282 | **0.7707** | 0.2386 | +0.5320 |

The shuffled floor sits at or below the majority baseline at every seed, so this is not
overfitting. Every one of the 256 dimensions takes between 171 and 221 distinct values
over 8000 steps, so the features are not degenerate.

## Geometry: why an instrument would miss it

| seed | norm of mean | RMS of centered part | varying / fixed | consecutive cosine | PCA dims for 95% |
|---|---|---|---|---|---|
| 42 | 0.389838 | 0.054745 | 0.140429 | 0.996665882 | **2** of 256 |
| 43 | 0.392768 | 0.019178 | 0.048827 | 0.997639187 | **4** of 256 |
| 44 | 0.389557 | 0.057158 | 0.146726 | 0.994263238 | **2** of 256 |

The broadcast is a fixed vector of norm about 0.39 with a wobble 5 to 15 percent its size,
and that wobble is confined to two to four directions.

This is the mechanism behind a long list of degenerate readings. `broadcast_mag` is the
norm of a vector dominated by its constant part, so it moves very little. Any pairwise
cosine or distance is dominated by the same constant. An instrument built on either sees a
system that barely changes, while a linear read of the full vector recovers the stimulus
at 0.69 to 0.77.

## A correction to the estimate that prompted this

`memory_retrieval_repair_2026_08.md` inferred a cosine of 0.99999976 to 1.0 by inverting
the memory bid formula, and read it as the broadcast pointing one way for a whole run.

**The direct measurement is 0.9943 to 0.9976, not 0.9999998.** The earlier figure was
inflated by its method: it compared each broadcast against the BEST match among roughly a
hundred recent stored broadcasts, and a maximum over a hundred candidates is
systematically higher than the cosine between consecutive steps. The direction is highly
stable. It is not as static as that number implied, and that number should not be quoted.

The qualitative reading in that document survives: the broadcast changes length more
readily than direction. The magnitude of the effect was overstated.

## What this settles

The broadcast is not frozen and it is not contentless. Any instrument reporting a
degenerate reading at this level is a statement about the instrument until it is shown to
survive on the varying subspace.

Together with `workspace_state_variance_2026_08.md`, which found the 3-tuple alive at 2552
to 3848 distinct values while the binning resolved 4 or 5 states of 8, the workspace level
now has two independent findings on the (a) side of the fork.

## What this does NOT settle

- **No instrument is repaired.** That an instrument would see more by projecting onto the
  varying subspace, or by centering first, is a proposal. Nothing was changed and no
  instrument was re-run.
- **No indicator moves.** The count stays 3 IMPLEMENTED, 11 PARTIAL, 14 total. GWT-2 was
  re-scored on the competition failure, which is a separate matter from whether the
  broadcast carries content.
- **A linear probe is a lower bound.** It shows a linear read suffices. It does not
  establish that anything downstream in the architecture performs that read, and the
  policy demonstrably does not learn the task.
- **2 to 4 dimensions is not obviously enough.** A 256-dimensional workspace whose
  variance is 95 percent captured by 2 directions is a very narrow channel. Whether that
  is the intended bottleneck or a collapse is not decided here.
- **Direction stability is not explained.** Why the broadcast holds one direction to a
  cosine of 0.995 across a run is not addressed.

## Reproduce

```
python -m scripts.training.train_rlhf --env dmts --episodes 40 --seed 42 \
  --rssm-latent-mode continuous --capsule-workspace-source all_levels \
  --enable-audio --enable-mock-semantic --log-dir runs/bcast_s42
python -m scripts.analysis.probe_broadcast_geometry
```
