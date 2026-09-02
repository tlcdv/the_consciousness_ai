# FAILED: phi carries no stimulus identity. Neither does broadcast_mag

**`phi` does not track which shape was shown, at any of 3 seeds.** Its eta-squared for
6-class `sample_shape` on computed steps is 0.027118, 0.008728 and 0.025970, against
permutation null 95th percentiles of 0.055994, 0.062348 and 0.065874. At two of three
seeds it sits below the null MEAN.

`broadcast_mag` fails the same test at all 3 seeds: 0.009817, 0.003803 and 0.013990
against null p95 of 0.013473, 0.005973 and 0.023570.

This matters more than one more null result. **`phi` was the last instrument standing.**
EI is deprecated. CE 2.0 carries a degeneracy confound. `is_conscious` is pinned at 1. PCI
reads exactly 0.0 at the gate site (`pci_trained_2026_08.md`). sync_R reports the bids
(`sync_r_content_2026_09.md`). The vision bid is saturated. `phi` was the one signal the
project record names as having ever moved under an intervention, and it is the only
quantity that passes the strict non-degeneracy bar on its own.

It moves. It does not move with the stimulus.

3 seeds, 40 episodes, 8000 steps each, dmts, `--enable-audio --enable-mock-semantic`, from
`runs/bcast_s4{2,3,4}` already on disk. No training was run.

## A coverage trap this test had to avoid

`phi_method` is "skipped" at 6400 of 8000 steps and "pyphi" at 1599. The logged `phi` at a
skipped step is the previous computed value carried forward. A content test over all steps
would measure a step-function hold, which tracks the clock rather than the stimulus.

The gated test uses computed steps only: 584, 556 and 584 sample-phase steps, over 156,
147 and 152 trials. The effective sample is trials, and that count matches the broadcast
test which found a large effect (153 to 158 trials), so this is not an underpowered null.

The all-steps figures are reported beside it and are not gated. They agree: 0.024746,
0.006542 and 0.022859, all below their own nulls.

## Non-degeneracy: phi is alive, broadcast_mag is half pinned

| seed | phi distinct | phi modal share | broadcast_mag distinct | broadcast_mag modal share |
|---|---|---|---|---|
| 42 | 1198 | 0.44% | 219 | **39.67%** |
| 43 | 1267 | 0.25% | 170 | **43.35%** |
| 44 | 1128 | 0.38% | 217 | **41.73%** |

`phi` passes the strict bar at all three seeds. It is the only quantity measured on this
axis that does. Its failure is not degeneracy. It varies freely and carries nothing.

`broadcast_mag` fails the strict bar on the pinned clause, and fails the content test too.

## A prediction from the previous session, now confirmed

`broadcast_geometry_2026_08.md` found the 256-D broadcast decodes shape at 0.757, 0.691
and 0.771, and predicted that any instrument reading only its LENGTH would be blind to
that, because the length is dominated by a fixed offset 7 to 20 times larger than the
varying part.

`broadcast_mag` is `broadcast.norm()`. It is now measured contentless at 3 seeds. **The
prediction was stated before this measurement and it held.**

## The pattern across everything measured

| Quantity | Type | Carries shape? |
|---|---|---|
| `obs_map` | vector | yes, ~1.0 |
| `kl_map` | vector | yes, 0.84 / 0.71 / 0.76 |
| 256-D broadcast | vector | yes, 0.76 / 0.69 / 0.77 |
| vision bid, 4 unsupervised reductions | scalar | no |
| sync_R | scalar | no |
| `phi` | scalar | **no** |
| `broadcast_mag` | scalar | **no** |

Three vector representations carry the stimulus. Four scalar families do not.

**This is an empirical pattern at seven cases, not a theorem.** It does not prove that no
scalar could carry the content. It does show that no scalar this architecture currently
computes does, and that includes every quantity the instrument set reads.

## What this settles

The standing fork asks whether low internal variance is (a) blind discretization or (b)
dead dynamics. Neither describes this. `phi` varies across 1128 to 1267 distinct values
and carries no stimulus identity. No binning repair reaches a signal that varies that
freely, and no instrument can report content that is not in its input.

The content is in the architecture. It is in `obs_map`, in `kl_map` and in the 256-D
broadcast. Every point where this architecture reduces one of those to a single number
discards it, and every instrument reads the single number.

## What this does NOT settle

- **This is a linear-effect test on a scalar.** eta-squared measures variance explained by
  group membership. A scalar could in principle relate to shape non-monotonically in a way
  eta-squared misses. Nothing here rules that out.
- **It says nothing about `phi` as a theoretical quantity.** It reports that this
  implementation, on this architecture, on this task, does not vary with the stimulus.
- **No indicator is re-scored.** The count stays 3 IMPLEMENTED, 11 PARTIAL, 14 total, and
  the clock does not move.
- **No instrument is repaired or retired here.** That decision belongs to the inventory
  the instrument map asks for, and to the owner.
- **The C1 competence wall is untouched.** The policy still does not learn the task, and
  that is a separate and unfixed problem.

## Reproduce

```
python -m scripts.analysis.probe_scalar_content
```
