# The interventional TPM: the root is the DISCRETIZATION, not the estimator

**Result: a fourth outcome the pre-stated gate did not enumerate.** Forcing the gate into
all 243 distinct joint states produces 243 distinct continuous effects, so the gate has
real causal structure under intervention. Every one of them falls in a single tertile
bin, so the discretization destroys that structure before any metric can see it.

The estimator is not the defect. The architecture is not the defect. The 1/3 boundaries
are. This confirms the 2026-07-26 gate-binning result by an independent route.

One trained checkpoint (`runs/gate_ckpt_s42`, dmts, seed 42, 100 episodes), offline, no
training run. Single seed and single checkpoint. No indicator moves.

## Why this was run

Every EI and CE 2.0 number this project produced came from a TPM built by counting
observed transitions, while the theory specifies an interventional matrix under a
maximum-entropy `do()`. Five separate degeneracy findings rested on that estimator, so a
single common cause was plausible.

## Two defects found before the probe could run

**The trained gate was never saved.** It trains inside `tectum_optimizer` but the only
save wrote `tectum.state_dict()`, so every offline probe rebuilt it from random init.
Fixed 2026-08-11; this run produced the first persisted trained gate.

**The local EI formula omits Hoel's degeneracy term.** It computes
`log2(n) - mean_i H(row_i)` where Hoel's EI is `H(<TPM>) - mean_i H(row_i)`. Verified
against Hoel's KL form, which agrees with the entropy form to 2.7e-15
(`tests/test_ei_degeneracy_term.py`). The legacy form returns the MAXIMUM for a totally
degenerate system. It is masked on a smoothed observational matrix and appears
immediately on a sharp interventional one, so running this probe without the fix would
have reported a spectacular success for a dead gate.

## Results

| matrix | EI legacy | EI corrected | CE 2.0 | complexity |
|---|---|---|---|---|
| observational | 0.021826 | 0.020412 | 0.758312 | 1 |
| interventional FULL, tertile | 0.002272 | **0.000000** | 0.000000 | 0 |
| interventional VISITED, tertile | 0.000009 | 0.000009 | 0.004065 | 1 |
| interventional FULL, **quantile** | 0.002272 | 0.002201 | 0.016380 | 55 |

### Clause 2, the failing null, fires exactly

The frozen-trajectory references for 243 states at 800 steps:

- `constant_trajectory_floor` = **0.021826**
- `frozen_trajectory_ce2_value` = **0.758312**

The observational row reproduces **both to six decimal places**. The observational matrix
is not a weak measurement of the system. It is the Laplace floor and nothing else. 1 of
243 joint states was visited across 800 steps.

### The EI formula defect, confirmed on real data

On the interventional FULL tertile matrix, where all 243 causes drive one effect, the
legacy formula reports 0.002272 and the corrected formula reports exactly 0.000000. Zero
is right: a cause that tells you nothing about the effect carries no information.

## The control that overturned the first verdict

The probe's first run reported **ARCHITECTURAL**, because the binned effects collapsed to
one state. That was wrong, and only a control caught it.

Forcing 243 distinct causes and reading the **continuous** output:

| node | range across causes | std |
|---|---|---|
| attention | 4.626e-02 | 1.108e-02 |
| stability | 2.463e-02 | 5.229e-03 |
| coherence | 2.839e-02 | 6.409e-03 |
| confidence | 1.488e-02 | 3.318e-03 |
| adaptation | 5.155e-05 | 1.208e-05 |

**243 distinct causes produce 243 distinct effects.** Four nodes move by 1e-02 or more.
Adaptation is dead at 5e-05, the same node the gate-binning work found inert.

All 243 effects land in tertile bin 112, because they live in a band about 0.046 wide
around 0.47 while the boundaries sit at 0.333 and 0.667. Re-binning the same effects by
their own quantiles gives **56 distinct states** instead of 1.

A probe reading only binned states cannot distinguish "no causal structure" from "causal
structure the binning cannot represent". Those support opposite verdicts. The control is
now a mandatory part of the probe and gates the ARCHITECTURAL branch.

## What this does NOT show

- **The structure is real but very weak.** Even under quantile binning, corrected EI is
  0.002201 against a ceiling of log2(243) = 7.924813. Finding that the gate is not dead
  is not the same as finding it informative.
- **The intervention holds the gate input FIXED** at the mean observed input, which
  isolates the recurrence. A different input distribution could give a different answer,
  and this does not measure the gate's behaviour under varying input.
- **One checkpoint, one seed, one environment.** Deterministic given the checkpoint, but
  not replicated across seeds.
- **It does not rescue any instrument.** No value here clears the acceptance bar. It
  relocates the defect.
- **It says nothing about the seven docs resting on the observational estimator.** The
  direction is knowable, since the missing EI term can only inflate, so historical EI
  figures are upper bounds. Magnitudes are not assessed here.

## Consequence

The `--gate-binning quantile` flag stops being a hypothesis-grade curiosity and becomes
the only discretization under which the gate level carries any information at all, now
supported by an interventional result as well as the original observational one. A
default flip still needs 3 seeds.

## Reproduce

```
python -m scripts.analysis.probe_interventional_tpm --load-tectum runs/gate_ckpt_s42/tectum.pt
```

Deterministic given the checkpoint. Runtime about two minutes.
