# Gate discretization: the micro level is alive, the fixed tertile binning was blind

**Result: the gate collapse is a binning artifact, not dead gate dynamics, and
per-window quantile binning resolves it.** Single seed (42), dark_room, hypothesis
grade. Ships behind `--gate-binning quantile` (default `tertile`, baseline
bit-identical). No indicator moves, no consciousness claim.

## Why this was checked
The CE 2.0 pilot ([ce2_pilot_calibration_2026_07.md](ce2_pilot_calibration_2026_07.md))
found the gate and workspace CE 2.0 values were frozen-input artifacts, matching the
2026-07-02 EI assessment. Both traced to the same thing: the 243-state gate TPM never
leaves one joint state. That could be (a) the gate signal is dead, in which case no
binning helps, or (b) the gates vary but the fixed tertile boundaries do not resolve
it. This had never been measured: "saturated" was inferred from the EI floor, and the
raw gate node values were not logged.

## Measurement: the gates are alive
`metrics.csv` now logs the 5 raw ConsciousnessGate node values. Over a 4000-step
dark_room run (seed 42):

| gate node | distinct values | mean | std | range |
|-----------|-----------------|------|-----|-------|
| attention | 726 | 0.4963 | 1.49e-03 | 0.4877 - 0.4987 |
| stability | 863 | 0.4857 | 5.22e-03 | 0.4665 - 0.4945 |
| adaptation | 37 | 0.0101 | 6.08e-06 | 0.0101 - 0.0101 |
| coherence | 809 | 0.4871 | 3.56e-03 | 0.4827 - 0.4945 |
| confidence | 755 | 0.4959 | 3.12e-03 | 0.4924 - 0.5102 |

Four of five nodes carry real variation (std ~1e-3, hundreds of distinct values), ~13
orders of magnitude above float noise. They just live in a ~0.01-wide band around
0.49, while the fixed tertile boundaries are at 0.333 and 0.667, so every node lands
in the middle tertile every step. One joint state out of 243, permanently. Adaptation
is the exception: std 6e-6, effectively inactive, pinned at 0.0101 (a separate open
question, not addressed here).

## Pre-stated gate and verdict
Stated in the analysis script before the data was read: PASS requires >= 2 nodes with
std > 1e-3 AND quantile binning yielding >= 10 distinct joint states AND its CE 2.0
materially below the frozen reference. On the 4000-step run:

| binning | distinct joint states | CE 2.0 (243-state gate) |
|---------|----------------------|--------------------------|
| fixed tertile (current) | 1 | 0.923095 (= frozen-input reference) |
| quantile (proposed) | 28 | 0.649174 |

All three PASS criteria met. The quantile risk (manufacturing states from float noise)
is guarded two ways: a std > 1e-3 substantiveness check, and a var_floor (1e-4) that
pins any near-constant node (adaptation) to one bin so it adds no spurious states.

## Live end-to-end confirmation
A 10-episode run with `--gate-binning quantile` (seed 42), read from
`runs/gb_quantile/episodes.csv`:
- `ce2_gates_states` = 12 (was 1 under tertile), no degeneracy warning fired.
- `ce2_gates` = 0.664521, below the frozen ceiling 0.877884.
- `ei_gates_corr` = 0.063247: the first non-zero floor-corrected gate EI (pinned at
  0.0 under tertile). The shared discretization revives EI and CE 2.0 together.

## What this does NOT show
- `ce2_ratio` was 1.076 (workspace > gates) in that run, but `ce2_workspace_states`
  was only 3: the workspace level is STILL nearly degenerate. Its discretization
  (`discretize_continuous` on the broadcast sum, 8 states) is a separate issue not
  addressed here, so this ratio is not a validated CE2-1 onset signal and may still
  carry a state-count artifact. No emergence claim.
- Single seed, dark_room, short windows. Quantile stays default-off; >= 3 seeds are
  required before any default change or before CE2-1/CE2-2 can be read on it.

## Changes
- `--gate-binning {tertile,quantile}` (default tertile, baseline bit-identical),
  wired via `ConsciousnessMetricsLogger.set_gate_binning`; shared by EI and CE 2.0 so
  both score the same TPM. Near-constant nodes pinned (var_floor).
- `metrics.csv` logs the 5 raw gate node values; `episodes.csv` logs
  `ce2_gates_states` / `ce2_workspace_states`.
- Tests: tertile is bit-identical to the old inline logic; quantile resolves a
  narrow-band signal; a dead dimension adds no states.

## Next
Fix the workspace discretization the same way (it is the remaining frozen level), then
re-run the pilot over >= 3 seeds. Only then can CE2-1/CE2-2 be tested rather than
blocked by degeneracy.
