# Does the reversal need the existence drive? Rules before any run

Date: 2026-10-31. Status: rules written before any run. The owner decided that the value gate and reversal evidence is
to be repeated with `--existence-drive off` before any AE-1 decision. All runs of those gates used the drive on.

## What the code says, before any run

In the reversal gate the agent moves by the motor map. The target comes from the rhythm binder and the value gate.

| Part | Reads | Can the drive reach it? |
|---|---|---|
| Rhythm binder (`models/core/rhythm_binding.py`) | The frames and the frozen cochlear bands | Not by my reading |
| Motor map (`models/core/tectal_motor_map.py`) | The binder and its own noise generator | Not by my reading |
| Value gate (`_wire_value_gate`, `scripts/training/train_rlhf.py`) | Salience, and `raw_reward` | Possibly |

`raw_reward` is the reward of the room plus `0.5 * delta_phi_reward + 0.1 * curiosity_score`, and a value above 0.5 counts
as food (`models/core/value_gate.py`). The drive changes the emotion and the self-model paths, so it could reach phi and
curiosity. So the drive reaches the movement only if those bonuses cross 0.5. This is a reading of the code and not a
measurement.

If the movement does not depend on the drive, a gate with the drive off cannot test the drive, and it would cost about 2.6
hours. So the identity comes first.

## Stage A: identity on a short run

A discarded seed, 999, 6 episodes of 200 steps, the flags of the reversal gate in `docs/decisions/2026_10_16_reversal.md`
with `--dark-room-food-switch 3 --record-episodes none`, once with the drive on and once off.

A holds if `value_gate.json`, `tectal_orienting.json` and `map_binding.json` are byte for byte the same in both runs. The
`metrics.csv` is expected to differ, and how it differs is reported only.

- A holds: go to stage B.
- A fails: the drive reaches the movement, and the full gate of stage C is justified.

## Stage B: identity on one full run

Seed 701, 60 episodes, with `--dark-room-food-switch 29`, which reproduces the gate run with the fixed flag
(`docs/results/reversal_2026_10.md`), and the drive off. About 26 minutes.

B holds if `value_gate.json` has the md5 `ba464e91176fc8036aead3c4cdd6fa5e` and `tectal_orienting.json` and `map_binding.json`
equal those of `runs/rv_switch_s701`. That run used the drive on, at an older revision. If B fails, one more run with the drive on
at the current revision separates a change of the code from an effect of the drive.

**If A and B hold, the movement at these runs does not depend on the drive, and I stop without the full gate.** The result is
deterministic, so I claim no seed replication. The R1 to R3 numbers of seed 701 then hold with the drive off by identity. The other
two seeds are a reading of the code. A repeat on fresh seeds would be a replication of the reversal and not a test of the drive, so it
is the owner's decision.

## Stage C: the full gate, only if A or B shows that the drive reaches the movement

Seeds 901, 902 and 903, which no run has used, 60 episodes, one run at a time. The flags of the reversal gate with
`--existence-drive off`. Arms reversal (`--dark-room-food-switch 30`) and stay. Read by `scripts/analysis/probe_reversal.py`, with
a second count from separate code.

| Check | Rule, at 3 of 3 seeds |
|---|---|
| R1 learns before the move | reversal arm, episodes 21 to 30: light first ≥ 0.8 |
| R2 reverses | reversal arm, episodes 46 to 60: decoy first ≥ 0.8 |
| R3 the change follows the food | decoy first in episodes 46 to 60, reversal arm above the stay arm, pooled one-sided Fisher p < 0.05, difference > 0 at each seed |

A control with all new flags off must give the md5 `7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv` of seed 73. About 2.6 hours.

## Limits, stated before the runs

- The frozen and scrambled controls of the value gate are not repeated in any stage.
- A passed identity shows only that this behaviour does not read the drive. It does not show that the drive is absent from the
  agent. Its emotion and reward paths still run.
- If the drive does not reach the behaviour, then the behaviour of this gate does not come from a homeostatic struggle. That bears on
  how public text describes the agent, and it is the owner's decision.
- The AE-1 score stays the owner's decision.

## Added 2026-11-15: identity on seeds 702 and 703, rule written before the runs

Owner decision of 2026-11-15. The identity result of the reversal with the drive off counts as the
second result that the AE-1 row of the rubric names. Until now the identity was measured on seed 701
and was a reading of the code for seeds 702 and 703. So these two seeds run with the drive off first.

- Seeds 702 and 703. The command is the recorded command of `session.json` of `runs/rv_switch_s702`
  and `runs/rv_switch_s703`, with `--existence-drive off` and a new log folder
  (`runs/drv_b_off_s702`, `runs/drv_b_off_s703`).
- One more flag changes, by the owner's word of 2026-11-15, and no other. The recorded commands have
  `--dark-room-food-switch 30`. They ran before the fix of that flag. After the fix the same room is
  `--dark-room-food-switch 29` (`docs/results/reversal_2026_10.md`), and stage B of seed 701 used 29.
  So the two runs use 29.
- 60 episodes of 200 steps. One run at a time. The free space of drive C is read before each run.
- Each run must have 12001 lines in `metrics.csv`. If not, that seed is VOID.
- A seed holds if `value_gate.json`, `tectal_orienting.json` and `map_binding.json` have these md5
  values, in this order. They are the md5 values of the gate runs with the drive on.

| Seed | `value_gate.json` | `tectal_orienting.json` | `map_binding.json` |
|---|---|---|---|
| 702 | `37cad8a52e9a53cf05dd010a2a1a006e` | `7f0d2ced927da484d546544c90cb486e` | `5e8a1aff0d5b5a1baa5a626cd4adf919` |
| 703 | `df0519e99e9f28775b6370bf348184f5` | `22a44959e0e6fdb546f312b1d1655c94` | `10b694c08b62b2021a6b1f92275de5ea` |

- If a record differs, that seed runs once more with `--existence-drive on` at the same revision. If
  that run is equal to the drive off run, the code changed the record and the drive did not. If it
  differs, the drive reaches the movement and the result is FAILED. FAILED is reported first.
- This is a test of identity. R1 to R3 are not computed again.

The result goes into `docs/results/reversal_drive_off_2026_11.md` with the md5 values and the
manifests of the runs (commit, no tracked change, drive off).
