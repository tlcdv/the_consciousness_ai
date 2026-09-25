# FAILED: with the natural frequency fixed, sync_R moves at every step and still carries no stimulus identity

**The content test fails at all 3 seeds with `--akorn-natural-frequency` on, and at all 3 with
it off.** With the flag on, the eta-squared of sync_R for 6-class `sample_shape` is 0.002128,
0.002300 and 0.011996, against permutation null p95 values of 0.016775, 0.004435 and 0.039142.
Every observed value sits below its own null p95. Verdict (c) of the standing fork, varying but
contentless, holds with the fix.

**What the fix changes is the degeneracy, and a matched arm attributes that change to it.** The
flag-off arm uses the same code, seeds and flags with only the flag removed. There, 48.39,
83.74 and 85.43 percent of steps sit at one sync_R value (0.450108000). With the flag on,
0.0375 percent do, and sync_R clears the strict non-degeneracy bar at all 3 seeds. The
oscillators no longer settle into one fixed state. That movement comes from the restored
intrinsic rotation, and the content test shows it carries nothing about the stimulus.

## The defect

`KuramotoLayer.forward` computed the intrinsic rotation as

```
rotation = torch.einsum('ndd,bnd->bnd', omega, current_phases)
```

The repeated index reads only the diagonal of each `omega[n]`, and `omega` is forced
skew-symmetric, so its diagonal is zero. The rotation was exactly zero for every input. Check,
run 2026-09-24, repo einsum max 0.0 against 5.29 for the matrix-vector product
`'nde,bne->bnd'` on the same random skew-symmetric `omega`. With no intrinsic frequency the
oscillators can only relax to a fixed point, and at that point sync_R equals mean(bids), which
is the mechanism `sync_r_content_2026_09.md` described.

The fix is behind `--akorn-natural-frequency` (default off). With the flag off the update is
unchanged, pinned by `tests/test_akorn_natural_frequency.py::test_binding_system_default_matches_legacy_layer`
and `test_legacy_default_has_no_rotation`. With the flag on, one step with zero coupling
rotates a 2-D oscillator by exactly atan(dt * w), which the test checks in closed form.

## The Brian2 validation had never run

`models/validation/brian2_binding_validation.py` imported `radian` from the top level of
`brian2`. Brian2 2.5.1 does not export it there, so the `ImportError` set
`BRIAN2_AVAILABLE = False` and every Brian2 test skipped on this machine. Behind that import
were three more defects, each of which would have stopped the network from building. The
variable name `amp` is reserved by Brian2 as a unit, the synapse model never declared the
weight `w`, and the coupling term had units Hz squared against Hz.

After those fixes, `validate_binding` still compares AKOrN with a different system. It reads
omega as `P[1,0]` while `forward()` uses `P - P^T`, it applies a rotation the legacy layer
omits, it divides the coupling by N, it uses sender-only amplitude, and it maps AKOrN steps
and Brian2 seconds onto unrelated time axes. Its docstring now says so. The existing
`test_full_validation_passes` asserts correlation above 0.5 and passes, which shows that
check cannot detect a mismatched system.

`validate_binding_matched` integrates the continuous limit of the layer's own 2-D update,
`dTheta_i/dt = w_i + K a_i sum_j W_ij a_j sin(Theta_j - Theta_i)` with step k at time k*dt.
Its maximum phase error shrinks with the step size, which is the expected behaviour of an
Euler step against its flow.

| rotation | dt 0.05 | dt 0.025 | dt 0.0125 |
|---|---|---|---|
| legacy (w = 0) | 0.0052 | 0.0027 | 0.0013 |
| fixed | 0.0574 | 0.0152 | 0.0043 |

Max phase error in radians over a 2-unit span, 4 oscillators, one parameter draw
(`torch.manual_seed(0)`). This is a deterministic check of the integrator, so it has no seeds.
`tests/test_brian2_matched_validation.py` pins the convergence and a negative control, where a
frequency mismatch must raise the error more than tenfold.

## Content test, flag on

Runs `runs/omega_s42`, `omega_s43`, `omega_s44`. Same command as the runs behind
`sync_r_content_2026_09.md` plus the flag and the two flags later made mandatory:

```
python -m scripts.training.train_rlhf --env dmts --episodes 40 --seed <s> \
  --rssm-latent-mode continuous --capsule-workspace-source all_levels \
  --enable-audio --enable-mock-semantic --existence-drive on --record-episodes none \
  --akorn-natural-frequency --log-dir runs/omega_s<s>
python -m scripts.analysis.probe_sync_r_content --run-prefix omega_s
```

Completion verified from disk, 8000 metrics rows and 40 episode rows per seed.

| seed | distinct / 8000 | span | at modal value | eta2 | null mean | null p95 | verdict |
|---|---|---|---|---|---|---|---|
| 42 | 7897 | 0.4878 | 0.0375% | 0.002128 | 0.007061 | 0.016775 | FAIL |
| 43 | 7919 | 0.4525 | 0.0375% | 0.002300 | 0.002106 | 0.004435 | FAIL |
| 44 | 7844 | 0.4980 | 0.0375% | 0.011996 | 0.017389 | 0.039142 | FAIL |

The probe's section 2 control (converged sync_R equals the bid mean within 1e-6) runs the
layer at its default, flag off, so it describes the legacy layer and says nothing about the
flag-on runs.

## Reset probe, flag on

**FAILED at all 3 checkpoints, in both arms.** The 5-module alignment vector does not decode
`sample_shape` above its permutation null p95.

```
python -m scripts.analysis.probe_oscillator_reset --checkpoint runs/gate3_s<s> --seed <s>   --episodes 70 --content --permutations 500 --akorn-natural-frequency
```

| checkpoint | arm | acc | null p95 | null mean | verdict |
|---|---|---|---|---|---|
| gate3_s42 | A no reset | 0.1926 | 0.2264 | 0.1813 | (c) |
| gate3_s42 | B reset per episode | 0.1972 | 0.2271 | 0.1798 | (c) |
| gate3_s43 | A no reset | 0.1966 | 0.2220 | 0.1796 | (c) |
| gate3_s43 | B reset per episode | 0.1326 | 0.2270 | 0.1789 | (c) |
| gate3_s44 | A no reset | 0.1678 | 0.2168 | 0.1723 | (c) |
| gate3_s44 | B reset per episode | 0.2066 | 0.2216 | 0.1738 | (c) |

203 trials per run, 5-fold CV, uniform chance 0.1667. Raw output `runs/omega_reset_s4{2,3,4}.txt`.

**The three checkpoints are not three independent oscillator trajectories.** In arm A the
sync_R summary is identical at all three, 13591 distinct values, min 0.001816, max 0.449814,
sd 1.046e-01, and the winner shares are identical too (semantic 0.527, vision 0.473). The
probe seeds the oscillator with `--replay-seed 20260913` at every run, and on DMTS the bids
are constant (`modality_starvation_2026_09.md`), so the oscillator sees the same input at
every checkpoint. The many distinct values come from the intrinsic rotation the fix restores,
not from the stimulus. The content tests differ only because the environment seeds, and so
the trial labels, differ. The arm A dynamics are one trajectory, and no seed replication is
claimed for them.

## Matched flag-off arm

Runs `runs/omegaoff_s42`, `omegaoff_s43`, `omegaoff_s44`, the command above without
`--akorn-natural-frequency`, same branch. Completion verified from disk, 8000 metrics rows and
40 episode rows per seed. `python -m scripts.analysis.probe_sync_r_content --run-prefix omegaoff_s`.

| seed | arm | distinct / 8000 | span | at modal value | strict bar | eta2 | null p95 | content |
|---|---|---|---|---|---|---|---|---|
| 42 | flag off | 1256 | 0.1235 | 48.3875% | FAIL | 0.001656 | 0.006277 | FAIL |
| 42 | flag on | 7897 | 0.4878 | 0.0375% | PASS | 0.002128 | 0.016775 | FAIL |
| 43 | flag off | 1006 | 0.1739 | 83.7375% | FAIL | 0.009122 | 0.040486 | FAIL |
| 43 | flag on | 7919 | 0.4525 | 0.0375% | PASS | 0.002300 | 0.004435 | FAIL |
| 44 | flag off | 995 | 0.1757 | 85.4250% | FAIL | 0.033750 | 0.060986 | FAIL |
| 44 | flag on | 7844 | 0.4980 | 0.0375% | PASS | 0.011996 | 0.039142 | FAIL |

The flag-off arm reproduces the modal value 0.450108000 at all 3 seeds, the value
`sync_r_content_2026_09.md` recorded.

## The baseline check, and why md5 cannot decide it here

Smoke runs, DMTS, 1 episode, 30 steps, seed 42, flag off, the same flags as above, compared
column by column on `metrics.csv`. Two runs of this branch and two runs of `main`.

| pair | columns that differ |
|---|---|
| main run 1 vs main run 2 | env_sample_shape (30 rows) |
| branch run 1 vs branch run 2 | env_sample_shape (30), ignition_salience (20), bid_semantic (20), gate_attention (1), broadcast_mag (1) |
| branch run 1 vs main run 1 | ignition_salience (20), bid_semantic (20), broadcast_mag (4), gate_attention (1) |
| branch run 2 vs main run 2 | env_sample_shape (30), broadcast_mag (5) |

Two runs of identical code differ, so no md5 comparison can pass on this machine. Two sources
vary between runs. The DMTS stimulus draw is not seeded by `--seed` (next section), and some
float columns differ in the last digit between runs of the same code, next to the CuBLAS
nondeterminism warning the runs print. **sync_r is identical in every pair**, and it is the
only column the binding layer writes. With the unit pin tests, that is the evidence that the
flag-off path is unchanged.

## A separate defect found here, not fixed here

`--seed` does not seed the DMTS environment. Its help text says it seeds "the env.reset call
on episode 0", but `run_episode` calls `env.reset()` with no seed and `DMTSEnv` draws trials
from an unseeded `np.random.default_rng()`. Two runs with the same `--seed` show different
`env_sample_shape` sequences. Fixing it would change the stimulus sequence of every future run,
so it is left for the owner.

## What this does and does not change

It does not reopen Phi-1, and it does not move RPT-2 or GWT-2 in the rubric. sync_R remains
contentless with or without the rotation. The oscillators are still 5 abstract units, one per
module, with no spatial layout, so no traveling wave can exist in them.
