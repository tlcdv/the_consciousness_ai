# The workspace level is ALIVE and badly discretized. The binning wastes two of eight bins

**The signal varies and the binning cannot see it.** Every component of the workspace
state has between 167 and 1245 distinct values across 8000 steps, and their sum takes 2552
to 3848 distinct values. The pipeline resolves that into 4 or 5 states out of 8. Quantile
binning on the same data resolves 8 of 8, at all six runs.

Under the standing fork this is answer (a): the discretization is blind, and it is
repairable. It is not answer (b).

Read from `metrics.csv` in six runs already on disk, 3 seeds in each of two conditions,
8000 steps each. No training, no model loaded, no new compute.

## What the "workspace level" actually is

`train_rlhf.py:1895` builds `ws_state = (broadcast_mag, phi, sync_r)`, a 3-tuple of
scalars. It is NOT the 256-dimensional broadcast, which is a separate object and a
separate question. `metrics_logger.py:546-547` then does this:

```
ws_flat     = sum(ws_state)                      # three scalars added together
ws_discrete = discretize_continuous(ws_flat, 8)  # clips to [0,1], then floor(x * 7)
```

Three quantities on unrelated scales are summed, and the sum is binned on fixed [0,1]
boundaries.

## The components are alive

| run | broadcast_mag distinct | phi distinct | sync_r distinct |
|---|---|---|---|
| bidlog_s42 | 215 | 1187 | 988 |
| memfix_s42 | 215 | 1187 | 1245 |
| memfix_s43 | 167 | 1104 | 1019 |
| memfix_s44 | 211 | 1161 | 992 |

Standard deviations: `broadcast_mag` 8.9e-03 to 2.1e-02 around a mean of 0.393, `phi`
about 1.1e-03 around a mean of 1.2e-03, `sync_r` 8.5e-03 to 1.6e-02 around a mean of 0.45
to 0.56.

Every component clears the pre-stated bar of more than 100 distinct values, and none
falls to the dead bar of 10 or fewer values or a standard deviation below 1e-6.

## The binning throws it away

| run | ws_flat distinct | ws_flat range | states resolved | bins used |
|---|---|---|---|---|
| bidlog_s42 | 3848 | [0.3991, 0.9843] | 4 of 8 | 2, 3, 5, 6 |
| bidlog_s43 | 2682 | [0.3442, 0.9481] | 4 of 8 | 2, 3, 5, 6 |
| bidlog_s44 | 2552 | [0.3470, 1.0353] | 5 of 8 | 2, 3, 5, 6, 7 |
| memfix_s42 | 3837 | [0.3991, 1.0314] | 4 of 8 | 2, 3, 6, 7 |
| memfix_s43 | 2624 | [0.3442, 1.0283] | 4 of 8 | 2, 5, 6, 7 |
| memfix_s44 | 2574 | [0.3470, 1.0764] | 5 of 8 | 2, 3, 5, 6, 7 |

**Bins 0 and 1 are structurally unreachable.** `floor(x * 7)` needs `x < 0.286` to land in
either, and `ws_flat` never falls below 0.344 in any run. A quarter of the state space is
unusable before the agent does anything.

## A prediction that was wrong, recorded

Reading `discretize_continuous` suggested the clip to [0,1] was the cause, since
`broadcast_mag` is an L2 norm over 256 dimensions with no reason to lie in that range.
**That was wrong.** `broadcast_mag` has a mean of 0.393, and only 0.0 to 1.4 percent of
`ws_flat` exceeds 1.0 in any run.

The cause is not clipping. It is that fixed [0,1] boundaries are being applied to a signal
that lives in roughly [0.34, 1.08].

## The fix already exists one level down

This is the same failure the gate level had, and the gate level was repaired. The 2026-07
diagnosis found four of five gate nodes carrying real variation in a band about 0.01 wide
around 0.49, which fixed tertile boundaries at 1/3 and 2/3 could not resolve. The answer
was `--gate-binning quantile`, which takes boundaries from the window's own distribution
(`metrics_logger.py:469-500`).

The workspace level has no equivalent. Applying the same idea to this data:

| run | fixed [0,1] | quantile |
|---|---|---|
| bidlog_s42 | 4 of 8 | **8 of 8** |
| bidlog_s43 | 4 of 8 | **8 of 8** |
| bidlog_s44 | 5 of 8 | **8 of 8** |
| memfix_s42 | 4 of 8 | **8 of 8** |
| memfix_s43 | 4 of 8 | **8 of 8** |
| memfix_s44 | 5 of 8 | **8 of 8** |

Full coverage at every run. This is reported as a demonstration on existing data, not as
an implemented change. Nothing was modified.

## What this settles

The workspace level is not frozen. Any statement that it "reached only 3 of 8 states" is a
statement about the binning, not about the agent.

## What this does NOT settle

- **The 256-D broadcast is a different object and is not measured here.** The cosine lead
  from the memory repair concerns that vector, not this 3-tuple. It remains open.
- **Nothing is implemented.** A quantile option for the workspace level is a proposal.
  Whether it changes any downstream number is untested, and the instruments that consume
  this discretization (workspace EI, CE 2.0) are already deprecated or retired for
  unrelated reasons, so a coarser-than-necessary binning may not be why they failed.
- **Summing three incommensurable scalars is not addressed.** Better binning of a sum does
  not make the sum meaningful, and `broadcast_mag + phi + sync_r` adds an L2 norm, an
  integrated-information estimate and a Kuramoto order parameter. That is a separate
  defect from the one measured here.
- **No indicator moves.** The count stays 3 IMPLEMENTED, 11 PARTIAL, 14 total.

## Reproduce

```
python -m scripts.analysis.probe_workspace_state_variance
```
